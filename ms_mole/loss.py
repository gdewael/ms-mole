import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import unbatch
import massspecgym.utils as utils
from importlib.resources import files
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def cont_iou(fp_pred, fp_true):
    total = (fp_pred + fp_true.to(fp_pred.dtype)).sum(-1)
    difference = (fp_pred - fp_true.to(fp_pred.dtype)).abs().sum(-1)
    return (total - difference) / (total + difference)


class FingerprintLossBase(nn.Module):
    def __init__(self, embedding_dim, pred_fp=True):
        super().__init__()
        self.pred_fp = pred_fp
        if pred_fp:
            self.pred_head = nn.Linear(embedding_dim, 4096)
        else:
            self.pred_head = nn.Identity()

    def forward(self, embed, true_fp, cand_fp, batch_ptr, labels):
        raise NotImplementedError

    def predict_fingerprint(self, embed):
        if self.pred_fp:
            return F.sigmoid(self.pred_head(embed))
        else:
            raise NotImplementedError


class FingerprintBCELoss(FingerprintLossBase):
    def __init__(self, embedding_dim):
        super().__init__(embedding_dim, pred_fp=True)

    def forward(self, embeds, true_fp, *args):
        logits = self.pred_head(embeds)
        return F.binary_cross_entropy_with_logits(
            logits,
            true_fp.to(logits.dtype),
        )


class FingerprintFocalLoss(FingerprintLossBase):
    def __init__(self, embedding_dim, gamma=2):
        super().__init__(embedding_dim, pred_fp=True)
        self.gamma = gamma

    def forward(self, embeds, true_fp, *args):
        logits = self.pred_head(embeds)
        CE = F.binary_cross_entropy_with_logits(
            logits, true_fp.to(logits.dtype), reduction="none"
        )
        pt = torch.exp(-CE)
        return ((1 - pt) ** self.gamma * CE).mean()


class FingerprintCosineSimLoss(FingerprintLossBase):
    def __init__(self, embedding_dim):
        super().__init__(embedding_dim, pred_fp=True)

    def forward(self, embeds, true_fp, *args):
        logits = self.pred_head(embeds)
        return F.cosine_embedding_loss(
            torch.sigmoid(logits),
            true_fp.to(logits.dtype),
            torch.tensor([1]).to(logits.device),
        )


class FingerprintIoULoss(FingerprintLossBase):
    def __init__(self, embedding_dim):
        super().__init__(embedding_dim, pred_fp=True)

    def forward(self, embeds, true_fp, *args):
        logits = self.pred_head(embeds)
        preds = torch.sigmoid(logits)
        intersection = (preds * true_fp.to(preds.dtype)).sum(-1)
        total = (preds + true_fp.to(preds.dtype)).sum(-1)

        iou = intersection / (total - intersection)
        return 1 - iou.mean()


def listwise_contrastive_loss(scores, labels, temp):
    contrastive_loss = []
    for sc, l in zip(scores, labels):
        true_label = torch.where(l)[0][[0]]
        contrastive_loss.append(
            F.cross_entropy(
                torch.cat([sc[true_label], sc[~l]]) / temp,
                torch.tensor(0).to(sc.device),
            )
        )

    return torch.stack(contrastive_loss).mean()


def pairwise_contrastive_loss(scores, labels, temp):
    contrastive_loss = []
    for sc, l in zip(scores, labels):
        true_label = torch.where(l)[0][[0]]
        contrastive_loss.append(
            F.cross_entropy(
                torch.stack([sc[true_label].expand(sc[~l].size()), sc[~l]]).T / temp,
                torch.zeros(sc[~l].size()).to(sc.device).long(),
            )
        )

    return torch.stack(contrastive_loss).mean()


class FingerprintContrastiveFPCosineLoss(FingerprintLossBase):
    def __init__(self, embedding_dim, listwise=True, temp=1):
        super().__init__(embedding_dim, pred_fp=True)

        self.temp = temp
        if listwise:
            self.loss_compute_fn = listwise_contrastive_loss
        else:
            self.loss_compute_fn = pairwise_contrastive_loss

    def forward(self, embeds, true_fp, cand_fp, batch_ptr, labels):
        pred_fp = F.sigmoid(self.pred_head(embeds))
        scores = self.reranker(pred_fp, cand_fp, batch_ptr)

        indexes = utils.batch_ptr_to_batch_idx(batch_ptr)
        scores = unbatch(scores, indexes)
        labels = unbatch(labels, indexes)

        return self.loss_compute_fn(scores, labels, self.temp)

    def reranker(self, embeds, cand_fp, batch_ptr):
        preds = embeds.repeat_interleave(batch_ptr, dim=0)
        return F.cosine_similarity(preds, cand_fp.to(preds.dtype))


class FingerprintContrastiveEmbedCosineLoss(FingerprintLossBase):
    def __init__(self, embedding_dim, contrastive_dim=128, listwise=True, temp=1):
        super().__init__(embedding_dim, pred_fp=False)

        self.temp = temp
        if listwise:
            self.loss_compute_fn = listwise_contrastive_loss
        else:
            self.loss_compute_fn = pairwise_contrastive_loss

        self.pred_head = nn.Linear(embedding_dim, contrastive_dim)
        self.fp_head = nn.Linear(4096, contrastive_dim)

    def forward(self, embeds, true_fp, cand_fp, batch_ptr, labels):
        scores = self.reranker(embeds, cand_fp, batch_ptr)

        indexes = utils.batch_ptr_to_batch_idx(batch_ptr)
        scores = unbatch(scores, indexes)
        labels = unbatch(labels, indexes)

        return self.loss_compute_fn(scores, labels, self.temp)

    def reranker(self, embeds, cand_fp, batch_ptr):
        contrastive_fp_embeds = self.pred_head(embeds)
        contrastive_cand_embeds = self.fp_head(cand_fp.to(embeds.dtype))

        preds = contrastive_fp_embeds.repeat_interleave(batch_ptr, dim=0)
        return F.cosine_similarity(preds, contrastive_cand_embeds)


class FingerprintContrastiveCrossEncoderLoss(FingerprintLossBase):
    def __init__(
        self,
        embedding_dim,
        contrastive_dim=None,
        dropout=0.25,
        on_fp=True,
        listwise=True,
        temp=1,
    ):
        super().__init__(embedding_dim, pred_fp=False)

        self.temp = temp
        if listwise:
            self.loss_compute_fn = listwise_contrastive_loss
        else:
            self.loss_compute_fn = pairwise_contrastive_loss

        if on_fp:
            self.pred_head = nn.Sequential(nn.Linear(embedding_dim, 4096), nn.Sigmoid())
            self.fp_head = nn.Identity()
            dim_in_cross_encoder = 4096 * 3
        else:
            self.pred_head = nn.Linear(embedding_dim, contrastive_dim)
            self.fp_head = nn.Linear(4096, contrastive_dim)
            dim_in_cross_encoder = contrastive_dim * 3

        self.cross_encoder = nn.Sequential(
            nn.Linear(dim_in_cross_encoder, dim_in_cross_encoder // 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim_in_cross_encoder // 8),
            nn.Linear(dim_in_cross_encoder // 8, dim_in_cross_encoder // 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim_in_cross_encoder // 16),
            nn.Linear(dim_in_cross_encoder // 16, 1),
        )

        if listwise:
            self.loss_compute_fn = listwise_contrastive_loss
        else:
            self.loss_compute_fn = pairwise_contrastive_loss

    def forward(self, embeds, true_fp, cand_fp, batch_ptr, labels):
        scores = self.reranker(embeds, cand_fp, batch_ptr)

        indexes = utils.batch_ptr_to_batch_idx(batch_ptr)
        scores = unbatch(scores, indexes)
        labels = unbatch(labels, indexes)

        return self.loss_compute_fn(scores, labels, self.temp)

    def reranker(self, embeds, cand_fp, batch_ptr):
        contrastive_fp_embeds = self.pred_head(embeds)
        contrastive_cand_embeds = self.fp_head(cand_fp.to(embeds.dtype))

        preds = contrastive_fp_embeds.repeat_interleave(batch_ptr, dim=0)
        combined = torch.cat(
            [preds, contrastive_cand_embeds, preds * contrastive_cand_embeds], 1
        )
        return self.cross_encoder(combined).squeeze(-1)


class FingerprintRNNSubset01MaximizerLoss(FingerprintLossBase):
    def __init__(
        self,
        embedding_dim,
    ):
        super().__init__(embedding_dim, pred_fp=False)

        self.gru = nn.GRU(4096 + 2, embedding_dim)
        self.output_head = nn.Linear(embedding_dim, 4096 + 2)

        path = files("ms_mole.utils").joinpath("label_freq_sort.npy")
        self.labelfreq_sort = torch.tensor(np.load(path))

    def forward(self, embeds, true_fp, cand_fp, batch_ptr, labels):
        true_fps_sorted_by_freq = true_fp[:, self.labelfreq_sort.to(true_fp.device)]
        true_fps_sorted_by_freq = torch.cat(
            [
                torch.ones(len(true_fp), 1).to(true_fp.device),
                true_fps_sorted_by_freq,
                torch.ones(len(true_fp), 1).to(true_fp.device),
            ],
            1,
        )

        ix, labs = torch.where(true_fps_sorted_by_freq)
        labelset_size = true_fps_sorted_by_freq.sum(1).int().cpu()

        sequences = torch.split(F.one_hot(labs), labelset_size.tolist())
        input_sequences = [seq[:-1] for seq in sequences]

        sequences = torch.split(labs, labelset_size.tolist())
        true_sequences = [seq[1:] for seq in sequences]

        padded_input = pad_sequence(input_sequences, batch_first=True)
        packed_input = pack_padded_sequence(
            padded_input, labelset_size - 1, batch_first=True, enforce_sorted=False
        )

        out, _ = self.gru(packed_input.to(embeds.dtype), embeds.unsqueeze(0))
        out_seqs, out_lens = pad_packed_sequence(out, batch_first=True)
        out_seqs = self.output_head(out_seqs)

        padded_trues = pad_sequence(true_sequences, batch_first=True)

        loss = F.cross_entropy(
            out_seqs.transpose(1, 2), padded_trues.long(), reduction="none"
        )[padded_trues != 0].mean()
        return loss

    def predict_fingerprint(self, embeds):
        embed_ = embeds.unsqueeze(0)
        seq = (
            F.one_hot(torch.zeros(len(embeds), dtype=torch.long), num_classes=4098)
            .unsqueeze(0)
            .to(embeds.dtype)
            .to(embeds.device)
        )

        for _ in range(150):
            _, hn = self.gru(seq, embed_)
            prob_vector = self.output_head(hn)
            prob_vector[:, torch.arange(len(prob_vector)), seq.argmax(-1)] = (
                torch.finfo(prob_vector.dtype).min
            )
            seq = torch.cat(
                [
                    seq,
                    F.one_hot(prob_vector.argmax(-1), num_classes=4098).to(
                        embeds.dtype
                    ),
                ],
                0,
            )

        seq_ = (seq[1:].argmax(-1) - 1).T
        pred_labelvector = torch.zeros((len(embeds), 4096)).to(embeds.device)
        for i in range(len(pred_labelvector)):
            where_ = torch.where(seq_[i] == 4096)[0]
            if len(where_) > 0:
                pred_labelvector[i, self.labelfreq_sort[seq_[i][: where_[0]].cpu()]] = 1
            else:
                pred_labelvector[i, self.labelfreq_sort[seq_[i].cpu()]] = 1
        return pred_labelvector
