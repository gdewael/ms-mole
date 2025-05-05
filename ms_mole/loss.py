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
    difference =  (fp_pred - fp_true.to(fp_pred.dtype)).abs().sum(-1)
    return ((total - difference) / (total + difference))

class FPCrossEntropyLoss(nn.Module):
    def __init__(self, weighted=False):
        super().__init__()
        if weighted:
            path = files("ms_mole.utils").joinpath("weights.npy")
            self.weights = torch.tensor(np.load(path))
        self.weighted = weighted
    def forward(self, logits, true_fp, *args):
        if not self.weighted:
            return F.binary_cross_entropy_with_logits(
                logits,
                true_fp.to(logits.dtype),
            )
        else:
            weights = self.weights.to(logits.device).to(logits.dtype)[
                torch.arange(len(true_fp))[:, None], true_fp.int()
            ]

            l = F.binary_cross_entropy_with_logits(
                logits,
                true_fp.to(logits.dtype),
                reduction="none"
            )
            return (l*weights).mean()

class FPFocalLoss(nn.Module):
    def __init__(self, gamma=2, weighted=False):
        super().__init__()
        self.gamma=gamma
        if weighted:
            path = files("ms_mole.utils").joinpath("weights.npy")
            self.weights = torch.tensor(np.load(path))
        self.weighted = weighted
        
    def forward(self, logits, true_fp, *args):
        CE = F.binary_cross_entropy_with_logits(
            logits,
            true_fp.to(logits.dtype),
            reduction="none"
        )
        pt = torch.exp(-CE)
        if not self.weighted:
            return ((1-pt)**self.gamma * CE).mean()
        else:
            weights = self.weights.to(logits.device).to(logits.dtype)[
                torch.arange(len(true_fp))[:, None], true_fp.int()
            ]
            return ((1-pt)**self.gamma * CE * weights).mean()

class FPCosineSimLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, true_fp, *args):
        return F.cosine_embedding_loss(
            torch.sigmoid(logits),
            true_fp.to(logits.dtype),
            torch.tensor([1]).to(logits.device)
        )

class FPIoULoss(nn.Module):
    def __init__(self, jml_version=True):
        super().__init__()

        if jml_version:
            self.forward = self.forward_default
        else:
            self.forward = self.forward_jml

    @staticmethod
    def forward_default(logits, true_fp, *args):
        preds = torch.sigmoid(logits)
        intersection = (preds * true_fp.to(preds.dtype)).sum(-1)
        total = (preds + true_fp.to(preds.dtype)).sum(-1)

        iou = intersection / (total - intersection)
        return 1 - iou.mean()
    
    @staticmethod
    def forward_jml(logits, true_fp, *args):
        preds = torch.sigmoid(logits)
        total = (preds + true_fp.to(preds.dtype)).sum(-1)
        difference =  (preds - true_fp.to(preds.dtype)).abs().sum(-1)

        return 1 - ((total - difference) / (total + difference)).mean()


def listwise_contrastive_loss(scores, labels, temp):
    contrastive_loss = []
    for sc, l in zip(scores, labels):
        true_label = torch.where(l)[0][[0]]
        contrastive_loss.append(F.cross_entropy(
            torch.cat([sc[true_label], sc[~l]])/temp,
            torch.tensor(0).to(sc.device)
        ))
    
    return torch.stack(contrastive_loss).mean()

def pairwise_contrastive_loss(scores, labels, temp):
    contrastive_loss = []
    for sc, l in zip(scores, labels):
        true_label = torch.where(l)[0][[0]]
        contrastive_loss.append(F.cross_entropy(
            torch.stack([sc[true_label].expand(sc[~l].size()), sc[~l]]).T/temp,
            torch.zeros(sc[~l].size()).to(sc.device).long(),
        ))
    
    return torch.stack(contrastive_loss).mean()
    

class FPBiencoderRankLearner(nn.Module):
    def __init__(
        self,
        temp=0.1,
        n_bits=4096,
        dropout=0.2,
        sim_func = "cossim",
        projector=False,
        listwise=True,
    ):
        super().__init__()

        self.temp = temp
        if sim_func == "cossim":
            self.sim_func = lambda x, y : F.cosine_similarity(x, y)
        elif sim_func == "iou":
            self.sim_func = lambda x, y : cont_iou(x, y)
    
        self.proj = projector
        if self.proj:
            assert sim_func == "cossim"
            self.projector = nn.Sequential(
                nn.Linear(n_bits, n_bits//8),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(n_bits//8),
            )

        if listwise:
            self.loss_compute_fn = listwise_contrastive_loss
        else:
            self.loss_compute_fn = pairwise_contrastive_loss
              
    def forward(self, logits, true_fp, cand_fp, batch_ptr, labels):
        preds = F.sigmoid(logits).repeat_interleave(batch_ptr, dim=0)
        scores = self.reranker(preds, cand_fp)

        indexes = utils.batch_ptr_to_batch_idx(batch_ptr)
        scores = unbatch(scores, indexes)
        labels = unbatch(labels, indexes)

        return self.loss_compute_fn(scores, labels, self.temp)
    
    def reranker(self, preds, cand_fp):
        cand_fp = cand_fp.to(preds.dtype)

        if self.proj:
            preds = self.projector(preds)
            cand_fp = self.projector(cand_fp)
        return self.sim_func(
            preds,
            cand_fp
        )

class FPCrossEncoderRankLearner(nn.Module):
    def __init__(
        self,
        temp=0.1,
        n_bits=4096,
        dropout=0.2,
        projector=False,
        listwise=True,
    ):
        super().__init__()
        
        self.temp = temp
        
        self.proj = projector
        if self.proj:
            self.projector = nn.Sequential(
                nn.Linear(n_bits, n_bits//8),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(n_bits//8),
            )
        
        dim_in_cross_encoder = (n_bits//8*3 if self.proj else n_bits*3)
        self.cross_encoder = nn.Sequential(
            nn.Linear(dim_in_cross_encoder, n_bits//8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(n_bits//8),
            nn.Linear(n_bits//8, n_bits//16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(n_bits//16),
            nn.Linear(n_bits//16, 1),
        )

        if listwise:
            self.loss_compute_fn = listwise_contrastive_loss
        else:
            self.loss_compute_fn = pairwise_contrastive_loss
    
    def forward(self, logits, true_fp, cand_fp, batch_ptr, labels):
        preds = F.sigmoid(logits).repeat_interleave(batch_ptr, dim=0)
        scores = self.reranker(preds, cand_fp)

        indexes = utils.batch_ptr_to_batch_idx(batch_ptr)
        scores = unbatch(scores, indexes)
        labels = unbatch(labels, indexes)
        
        return self.loss_compute_fn(scores, labels, self.temp)
    
    def reranker(self, preds, cand_fp):
        cand_fp = cand_fp.to(preds.dtype)

        if self.proj:
            preds = self.projector(preds)
            cand_fp = self.projector(cand_fp)
        

        combined = torch.cat([
            preds,
            cand_fp,
            preds*cand_fp
        ], 1)
        return self.cross_encoder(combined).squeeze(-1)


class RNNClassifierChain(nn.Module):
    def __init__(
        self,
        hidden_size=64,
        n_bits=4096,
    ):
        super().__init__()
        
        self.gru = nn.GRU(n_bits+2, hidden_size)
        self.output_head = nn.Linear(hidden_size, n_bits+2)

        path = files("ms_mole.utils").joinpath("label_freq_sort.npy")
        self.labelfreq_sort = torch.tensor(np.load(path))
    
    def forward(self, embeds, true_fp, cand_fp, batch_ptr, labels):
        true_fps_sorted_by_freq = true_fp[:, self.labelfreq_sort.to(true_fp.device)]
        true_fps_sorted_by_freq = torch.cat(
            [torch.ones(len(true_fp), 1).to(true_fp.device), true_fps_sorted_by_freq, torch.ones(len(true_fp),1).to(true_fp.device)]
        , 1)

        ix, labs = torch.where(true_fps_sorted_by_freq)
        labelset_size = true_fps_sorted_by_freq.sum(1).int().cpu()

        sequences = torch.split(F.one_hot(labs), labelset_size.tolist())
        input_sequences = [seq[:-1] for seq in sequences]

        sequences = torch.split(labs, labelset_size.tolist())
        true_sequences = [seq[1:] for seq in sequences]

        padded_input = pad_sequence(input_sequences, batch_first=True)
        packed_input = pack_padded_sequence(padded_input, labelset_size-1, batch_first=True, enforce_sorted=False)

        out, _ = self.gru(packed_input.to(embeds.dtype), embeds.unsqueeze(0))
        out_seqs, out_lens = pad_packed_sequence(out, batch_first=True)
        out_seqs = self.output_head(out_seqs)

        padded_trues = pad_sequence(true_sequences, batch_first=True)

        loss = F.cross_entropy(
            out_seqs.transpose(1,2),
            padded_trues.long(), reduction="none"
        )[padded_trues!=0].mean()
        return loss

    def infer(self, embeds):
        embed_ = embeds.unsqueeze(0)
        seq = F.one_hot(torch.zeros(len(embeds), dtype=torch.long), num_classes=4098).unsqueeze(0).to(embeds.dtype).to(embeds.device)

        for _ in range(150):
            _, hn = self.gru(seq, embed_)
            prob_vector = self.output_head(hn)
            prob_vector[:, torch.arange(len(prob_vector)), seq.argmax(-1)] = torch.finfo(prob_vector.dtype).min
            seq = torch.cat([seq, F.one_hot(prob_vector.argmax(-1), num_classes=4098).to(embeds.dtype)], 0)
        
        seq_ = (seq[1:].argmax(-1)-1).T
        pred_labelvector = torch.zeros((len(embeds), 4096)).to(embeds.device)
        for i in range(len(pred_labelvector)):
            where_ = torch.where(seq_[i]==4096)[0]
            if len(where_) > 0:
                pred_labelvector[i, self.labelfreq_sort[seq_[i][:where_[0]].cpu()]] = 1
            else:
                pred_labelvector[i, self.labelfreq_sort[seq_[i].cpu()]] = 1
        return pred_labelvector
    

loss_str_to_fun_mapper = {
    "bce": FPCrossEntropyLoss,
    "fl": FPFocalLoss,
    "cossim": FPCosineSimLoss,
    "iou": FPIoULoss,
    "bienc": FPBiencoderRankLearner,
    "cross": FPCrossEncoderRankLearner,
}

class FPLoss(nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_bits,
        bitwise_loss = None,
        fpwise_loss = None,
        rankwise_loss = None,
        rnn_clfchain = False,
        bitwise_lambd = 1.0,
        fpwise_lambd = 1.0,
        rankwise_lambd = 1.0,
        bitwise_kwargs = {},
        fpwise_kwargs = {},
        rankwise_kwargs = {},
    ):
        super().__init__()
        assert bitwise_loss in [None, "bce", "fl"]
        assert fpwise_loss in [None, "cossim", "iou"]
        assert rankwise_loss in [None, "bienc", "cross"]
        
        if rnn_clfchain:
            self.rnncc = RNNClassifierChain(embedding_dim, n_bits)
            self.forward = self.rnncc.forward
            self.rankwise_loss = False
        else:
            self.fp_pred_head = nn.Linear(embedding_dim, n_bits)

            loss_names = [bitwise_loss, fpwise_loss, rankwise_loss]
            assert any(loss_names), "At least one loss function should be specified"
            loss_kwargs = [bitwise_kwargs, fpwise_kwargs, rankwise_kwargs]
            loss_weights = [bitwise_lambd, fpwise_lambd, rankwise_lambd]

            self.losses = []
            self.losses_w = []
            for l_name, l_kw, l_w in zip(loss_names, loss_kwargs, loss_weights):
                if l_name is not None:
                    self.losses.append(loss_str_to_fun_mapper[l_name](**l_kw))
                    self.losses_w.append(l_w)

            if rankwise_loss is not None:
                self.rankwise_loss = True
                self.ranker = self.losses[-1].reranker
            else:
                self.rankwise_loss = False
            self.losses = nn.ModuleList(self.losses)
        

    def forward(self, embed, true_fp, cand_fp, batch_ptr, labels):
        fp_pred_logits = self.fp_pred_head(embed)

        loss = sum([
            w * l(fp_pred_logits, true_fp, cand_fp, batch_ptr, labels)
            for l, w in zip(self.losses, self.losses_w)
        ])
        return loss
