import ms_mole.loss as losses
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from massspecgym.models.base import Stage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric, CosineSimilarity
from torch_geometric.utils import unbatch
import massspecgym.utils as utils
from torchmetrics.functional.retrieval import retrieval_hit_rate


class MLP(nn.Module):
    def __init__(
        self,
        n_inputs=990,
        n_outputs=256,
        layer_dims=[1024, 512],
        layer_or_batchnorm="layer",
        dropout=0.2,
    ):
        super().__init__()

        c = n_inputs
        layers = []
        for i in layer_dims:
            layers.append(nn.Linear(c, i))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(
                nn.LayerNorm(i) if layer_or_batchnorm == "layer" else nn.BatchNorm1d(i)
            )
            c = i

        layers.append(nn.Linear(c, n_outputs))

        self.net = nn.Sequential(*layers)

        self.hsize = n_outputs

    def forward(self, x):
        return self.net(x)


loss_mapper = {
    "bce": losses.FingerprintBCELoss,
    "fl": losses.FingerprintFocalLoss,
    "cosine": losses.FingerprintCosineSimLoss,
    "iou": losses.FingerprintIoULoss,
    "list_fp_cos": losses.FingerprintContrastiveFPCosineLoss,
    "list_embed_cos": losses.FingerprintContrastiveEmbedCosineLoss,
    "list_cross": losses.FingerprintContrastiveCrossEncoderLoss,
    "rnn_01": losses.FingerprintRNNSubset01MaximizerLoss,
}


class FingerprintPredicter(RetrievalMassSpecGymModel):
    def __init__(
        self,
        n_in=1000,  # number of bins
        layer_dims=[1024, 1024, 1024],  # hidden layer sizes
        layer_or_batchnorm="layer",
        dropout=0.25,
        loss="bce",
        loss_kwargs={},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mlp = MLP(
            n_inputs=n_in,
            n_outputs=layer_dims[-1],
            layer_dims=layer_dims[:-1],
            layer_or_batchnorm=layer_or_batchnorm,
            dropout=dropout,
        )
        self.loss = loss_mapper[loss](layer_dims[-1], **loss_kwargs)

    def forward(self, x):
        return self.mlp(x)

    def step(self, batch, stage):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x = batch["spec"]
        fp_true = batch["mol"]
        cands = batch["candidates"].int()
        batch_ptr = batch["batch_ptr"]

        # Predict fingerprint
        embedding = self(x)

        # Calculate loss
        loss = self.loss(embedding, fp_true, cands, batch_ptr, batch["labels"])

        return dict(loss=loss)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log(
            "train_loss",
            outputs["loss"],
            batch_size=batch["spec"].size(0),
            sync_dist=True,
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        x = batch["spec"]
        fp_true = batch["mol"]
        cands = batch["candidates"].int()
        batch_ptr = batch["batch_ptr"]

        # Predict fingerprint
        embedding = self(x)

        # Calculate loss
        loss = self.loss(embedding, fp_true, cands, batch_ptr, batch["labels"])

        if self.loss.pred_fp or isinstance(
            self.loss, losses.FingerprintRNNSubset01MaximizerLoss
        ):
            fp_pred = self.loss.predict_fingerprint(embedding).to(self.dtype)
            self.log_fingerprint(fp_pred, fp_true, "val")
            input_to_retrieve = fp_pred
        else:
            input_to_retrieve = embedding

        self.log_retrieval(input_to_retrieve, cands, batch)

        return dict(loss=loss)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.log(
            "val_loss",
            outputs["loss"],
            batch_size=batch["spec"].size(0),
            sync_dist=True,
            prog_bar=True,
        )

    def log_fingerprint(self, fp_pred, fp_true, stage="val"):
        tanimotos = batch_samplewise_tanimoto(fp_pred, fp_true, reduce=False)
        cont_ious = cont_iou(fp_pred, fp_true)

        self._update_metric(
            "%s_fingerprint_av_contiou" % stage,
            MeanMetric,
            (cont_ious,),
            batch_size=fp_true.size(0),
        )

        self._update_metric(
            "%s_fingerprint_av_tanim" % stage,
            MeanMetric,
            (tanimotos,),
            batch_size=fp_true.size(0),
        )
        self._update_metric(
            "%s_fingerprint_perc_close_match" % stage,
            MeanMetric,
            ((tanimotos > 0.675).to(float),),
            batch_size=fp_true.size(0),
        )
        self._update_metric(
            "%s_fingerprint_perc_meaningful_match" % stage,
            MeanMetric,
            ((tanimotos > 0.40).to(float),),
            batch_size=fp_true.size(0),
        )

        self._update_metric(
            "%s_fingerprint_av_cossim" % stage,
            CosineSimilarity,
            (fp_pred, fp_true),
            batch_size=fp_true.size(0),
            metric_kwargs=dict(reduction="mean"),
        )

    def log_retrieval(self, embed_or_fp, cand_fp, batch):
        if isinstance(
            self.loss,
            (
                losses.FingerprintContrastiveFPCosineLoss,
                losses.FingerprintContrastiveEmbedCosineLoss,
                losses.FingerprintContrastiveCrossEncoderLoss,
            ),
        ):
            scores = self.loss.reranker(embed_or_fp, cand_fp, batch["batch_ptr"])
            self.evaluate_retrieval_step(
                scores,
                batch["labels"],
                batch["batch_ptr"],
                stage=Stage("val"),
                name="ranker",
            )
        else:
            fp_pred_repeated = embed_or_fp.repeat_interleave(batch["batch_ptr"], dim=0)
            scores = nn.functional.cosine_similarity(fp_pred_repeated, cand_fp)
            self.evaluate_retrieval_step(
                scores,
                batch["labels"],
                batch["batch_ptr"],
                stage=Stage("val"),
                name="cossim",
            )

            scores = batch_samplewise_tanimoto(fp_pred_repeated, cand_fp)
            self.evaluate_retrieval_step(
                scores,
                batch["labels"],
                batch["batch_ptr"],
                stage=Stage("val"),
                name="tanim",
            )

            scores = cont_iou(fp_pred_repeated, cand_fp)
            self.evaluate_retrieval_step(
                scores,
                batch["labels"],
                batch["batch_ptr"],
                stage=Stage("val"),
                name="contiou",
            )

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("No support yet")

    def on_test_batch_end(self, outputs, batch, batch_idx):
        raise NotImplementedError("No support yet")

    def evaluate_retrieval_step(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        batch_ptr: torch.Tensor,
        stage: Stage,
        name: str,
    ) -> dict[str, torch.Tensor]:
        # Initialize return dictionary to store metric values per sample
        metric_vals = {}

        # This makes it so that - in the event all scores are equal - not always the first element is sorted as the top prediction
        some_noise = torch.randn_like(scores) * torch.finfo(scores.dtype).eps
        scores_w_noise = scores + some_noise

        # Evaluate hitrate at different top-k values
        indexes = utils.batch_ptr_to_batch_idx(batch_ptr)
        scores = unbatch(scores_w_noise, indexes)
        labels = unbatch(labels, indexes)

        for at_k in self.at_ks:
            hit_rates = []
            for scores_sample, labels_sample in zip(scores, labels):
                hit_rates.append(
                    retrieval_hit_rate(scores_sample, labels_sample, top_k=at_k)
                )
            hit_rates = torch.tensor(hit_rates, device=batch_ptr.device)

            metric_name = f"{stage.to_pref()}{name}_hit_rate@{at_k}"
            self._update_metric(
                metric_name,
                MeanMetric,
                (hit_rates,),
                batch_size=batch_ptr.size(0),
                bootstrap=stage == Stage.TEST,
            )
            metric_vals[metric_name] = hit_rates

        return metric_vals

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # warm up lr
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)


def batch_samplewise_tanimoto(pred_fp, true_fp, threshold=0.5, reduce=False):
    _and = (true_fp.int() & (pred_fp > threshold)).sum(-1)
    _or = (true_fp.int() | (pred_fp > threshold)).sum(-1)
    if reduce:
        return (_and / _or).mean()
    else:
        return _and / _or


def cont_iou(fp_pred, fp_true):
    total = (fp_pred + fp_true.to(fp_pred.dtype)).sum(-1)
    difference = (fp_pred - fp_true.to(fp_pred.dtype)).abs().sum(-1)
    return (total - difference) / (total + difference)
