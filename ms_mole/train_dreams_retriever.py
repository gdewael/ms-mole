"""
Training script for molecular retrieval using DreaMS spectrum embeddings.

Identical to train_retriever.py except:
  - The spectrum input is a precomputed (N, 1024) DreaMS embedding file
    (produced by ms_mole/utils/precompute_dreams_embeddings.py) instead of
    a binned spectrum.
  - No --bin_width / SpecBinner; model n_in is fixed at 1024.
  - Adds --dreams_embs_pth argument.

No existing files are modified by this script.
"""

import argparse
import os
import ast
import numpy as np
import torch

import ms_mole.loss as loss
from ms_mole.data import RetrievalDataset_PrecompFPandInchi, MsMoleMassSpecDataModule
from ms_mole.models import FingerprintPredicter
from massspecgym.data.transforms import MolFingerprinter
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data.dataset import Subset
import json


# ---------------------------------------------------------------------------
# DreaMS-specific dataset: replaces item["spec"] with precomputed embedding
# ---------------------------------------------------------------------------

def _dummy_spec_transform(spec):
    """Placeholder so MassSpecDataset doesn't crash during parent __init__."""
    return np.zeros(1, dtype=np.float32)


class RetrievalDataset_DreaMS(RetrievalDataset_PrecompFPandInchi):
    def __init__(self, dreams_embs_pth, **kwargs):
        kwargs.setdefault("spec_transform", _dummy_spec_transform)
        super().__init__(**kwargs)
        # Memory-map so multiple workers share physical pages
        self.dreams_embs = np.load(dreams_embs_pth, mmap_mode="r")

    def __getitem__(self, i):
        item = super().__getitem__(i)
        item["spec"] = torch.tensor(
            np.array(self.dreams_embs[i]), dtype=torch.float32
        )
        return item


# ---------------------------------------------------------------------------
# DreaMS-specific data module: swaps in RetrievalDataset_DreaMS
# ---------------------------------------------------------------------------

class DreaMSDataModule(MsMoleMassSpecDataModule):
    def __init__(self, dreams_embs_pth, **kwargs):
        # mol_transform is required by MassSpecDataset but unused for spec
        kwargs.setdefault("mol_transform", MolFingerprinter(fp_size=4096))
        super().__init__(**kwargs)
        self.dreams_embs_pth = dreams_embs_pth

    def setup(self, stage=None):
        if self.train_dataset is None:
            self.dataset = RetrievalDataset_DreaMS(
                dreams_embs_pth=self.dreams_embs_pth,
                pth=self.pth,
                fp_pth=self.fp_pth,
                inchi_pth=self.inchi_pth,
                candidates_pth=self.train_cands_pth,
                candidates_fp_pth=self.train_cands_fp_pth,
                candidates_inchi_pth=self.train_cands_inchi_pth,
                n_max_cands=self.train_n_max_cands,
            )
            self.split = self.dataset.metadata[["identifier", "fold"]]
            self.split = self.split.set_index("identifier")["fold"]
            split_mask = self.split.loc[self.dataset.metadata["identifier"]].values
            self._split_mask = split_mask  # store for val/test block
            self.train_dataset = Subset(
                self.dataset, np.where(split_mask == "train")[0]
            )

        if self.val_dataset is None:
            split_mask = self._split_mask
            valtest_dataset = RetrievalDataset_DreaMS(
                dreams_embs_pth=self.dreams_embs_pth,
                pth=self.pth,
                fp_pth=self.fp_pth,
                inchi_pth=self.inchi_pth,
                candidates_pth=self.valtest_cands_pth,
                candidates_fp_pth=self.valtest_cands_fp_pth,
                candidates_inchi_pth=self.valtest_cands_inchi_pth,
                n_max_cands=None,
            )
            self.val_dataset = Subset(valtest_dataset, np.where(split_mask == "val")[0])
            self.test_dataset = Subset(
                valtest_dataset, np.where(split_mask == "test")[0]
            )


# ---------------------------------------------------------------------------
# Helpers (copied from train_retriever.py)
# ---------------------------------------------------------------------------

def append_dict_to_json_file(new_dict, file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON content must be a list of dictionaries")
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(new_dict)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def none_or_int(value):
    if value is None or value == "None":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value: '{value}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        description="Molecular retrieval with DreaMS spectrum embeddings.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("dataset_path", type=str, metavar="dataset_path")
    parser.add_argument("helper_files_dir", type=str, metavar="helper_files_dir")
    parser.add_argument("logs_path", type=str, metavar="logs_path")
    parser.add_argument("--logging_file", type=str, default="res.json")

    # DreaMS embeddings (required)
    parser.add_argument(
        "--dreams_embs_pth", type=str, required=True,
        help="Path to precomputed (N, 1024) DreaMS embeddings .npy (row-aligned with TSV).",
    )

    parser.add_argument("--candidate_setting_train", type=str, choices=["mass", "formula"])
    parser.add_argument("--candidate_setting_eval", type=str, choices=["mass", "formula"])
    # Optional explicit candidate JSON paths (for non-MassSpecGym datasets, e.g. enveda)
    parser.add_argument("--train_cands_pth", type=str, default=None)
    parser.add_argument("--valtest_cands_pth", type=str, default=None)

    parser.add_argument(
        "--fp_type", type=str,
        choices=["morgan_2_4096", "morgan_4_4096", "morgan_6_4096",
                 "morgan_8_4096", "rdkit_4096", "map4_4096"],
        default="morgan_2_4096",
    )
    parser.add_argument("--n_max_cands", type=none_or_int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--devices", type=ast.literal_eval, default=[0])
    parser.add_argument("--precision", type=str, default="32-true")

    parser.add_argument("--layer_dim", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument(
        "--loss", type=str, default="list_fp_cos",
        choices=["bce", "fl", "cosine", "iou", "list_fp_cos", "list_embed_cos",
                 "list_fp_cross", "list_embed_cross", "rnn_01", "combined"],
    )
    parser.add_argument("--fl_gamma", type=float, default=5.0)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--contrastive_dim", type=int, default=256)
    parser.add_argument("--rankwise_listwise", type=boolean, default=True)
    parser.add_argument("--comb_weight", type=float, default=0.5)

    args = parser.parse_args()

    _train_cands_pth = args.train_cands_pth or os.path.join(
        args.helper_files_dir,
        "MassSpecGym_retrieval_candidates_%s.json" % args.candidate_setting_train,
    )
    _valtest_cands_pth = args.valtest_cands_pth or os.path.join(
        args.helper_files_dir,
        "MassSpecGym_retrieval_candidates_%s.json" % args.candidate_setting_eval,
    )

    data_module = DreaMSDataModule(
        dreams_embs_pth=args.dreams_embs_pth,
        pth=args.dataset_path,
        fp_pth=os.path.join(args.helper_files_dir, "%s_targets.npy" % args.fp_type),
        inchi_pth=os.path.join(args.helper_files_dir, "Inchis_targets.npy"),
        train_cands_pth=_train_cands_pth,
        train_cands_fp_pth=os.path.join(
            args.helper_files_dir,
            "%s_%scands.npz" % (args.fp_type, args.candidate_setting_train),
        ),
        train_cands_inchi_pth=os.path.join(
            args.helper_files_dir,
            "Inchis_%scands.npz" % args.candidate_setting_train,
        ),
        valtest_cands_pth=_valtest_cands_pth,
        valtest_cands_fp_pth=os.path.join(
            args.helper_files_dir,
            "%s_%scands.npz" % (args.fp_type, args.candidate_setting_eval),
        ),
        valtest_cands_inchi_pth=os.path.join(
            args.helper_files_dir,
            "Inchis_%scands.npz" % args.candidate_setting_eval,
        ),
        train_n_max_cands=args.n_max_cands,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )

    data_module.setup()

    loss_kwargs_dict = {
        "bce": {},
        "fl": {"gamma": args.fl_gamma},
        "cosine": {},
        "iou": {},
        "list_fp_cos": {"listwise": args.rankwise_listwise, "temp": args.temp},
        "list_embed_cos": {
            "contrastive_dim": args.contrastive_dim,
            "listwise": args.rankwise_listwise,
            "temp": args.temp,
        },
        "list_fp_cross": {
            "contrastive_dim": args.contrastive_dim,
            "listwise": args.rankwise_listwise,
            "temp": args.temp,
            "on_fp": True,
            "dropout": args.dropout,
        },
        "list_embed_cross": {
            "contrastive_dim": args.contrastive_dim,
            "listwise": args.rankwise_listwise,
            "temp": args.temp,
            "on_fp": False,
            "dropout": args.dropout,
        },
        "rnn_01": {},
        "combined": {
            "listwise": args.rankwise_listwise,
            "temp": args.temp,
            "comb_weight": args.comb_weight,
        },
    }

    model = FingerprintPredicter(
        n_in=1024,  # DreaMS embedding dimension
        layer_dims=[args.layer_dim] * args.n_layers,
        layer_or_batchnorm="layer",
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=0,
        df_test_path=None,
        loss=args.loss,
        loss_kwargs=loss_kwargs_dict[args.loss],
    )

    logger = TensorBoardLogger(
        "/".join(args.logs_path.split("/")[:-1]),
        name=args.logs_path.split("/")[-1],
    )

    val_ckpts = [
        ModelCheckpoint(monitor=None, filename="last-{epoch}-{step}"),
        ModelCheckpoint(monitor="val_loss", mode="max", filename="loss-{epoch}-{step}"),
    ]
    test_on = []

    if model.loss.pred_fp or isinstance(
        model.loss, loss.FingerprintRNNSubset01MaximizerLoss
    ):
        fp_acc_ckpt = ModelCheckpoint(
            monitor="val_fingerprint_av_tanim",
            mode="max",
            filename="fpacctanim-{epoch}-{step}",
        )
        val_ckpts += [fp_acc_ckpt]
        test_on += [(fp_acc_ckpt, "val_fingerprint_av_tanim", "tanimoto")]

    if isinstance(
        model.loss,
        (
            loss.FingerprintContrastiveFPCosineLoss,
            loss.FingerprintContrastiveEmbedCosineLoss,
            loss.FingerprintContrastiveCrossEncoderLoss,
            loss.CombinedLoss,
        ),
    ):
        ranker1_ckpt = ModelCheckpoint(
            monitor="val_ranker_hit_rate@1", mode="max",
            filename="ranker1-{epoch}-{step}",
        )
        ranker5_ckpt = ModelCheckpoint(
            monitor="val_ranker_hit_rate@5", mode="max",
            filename="ranker5-{epoch}-{step}",
        )
        ranker20_ckpt = ModelCheckpoint(
            monitor="val_ranker_hit_rate@20", mode="max",
            filename="ranker20-{epoch}-{step}",
        )
        val_ckpts += [ranker1_ckpt, ranker5_ckpt, ranker20_ckpt]
        test_on += [
            (ranker1_ckpt, "val_ranker_hit_rate@1", "contrastive_hr@1"),
            (ranker5_ckpt, "val_ranker_hit_rate@5", "contrastive_hr@5"),
            (ranker20_ckpt, "val_ranker_hit_rate@20", "contrastive_hr@20"),
        ]
    else:
        cossim1_ckpt  = ModelCheckpoint(monitor="val_cossim_hit_rate@1",  mode="max", filename="cossim1-{epoch}-{step}")
        cossim5_ckpt  = ModelCheckpoint(monitor="val_cossim_hit_rate@5",  mode="max", filename="cossim5-{epoch}-{step}")
        cossim20_ckpt = ModelCheckpoint(monitor="val_cossim_hit_rate@20", mode="max", filename="cossim20-{epoch}-{step}")
        tanim1_ckpt   = ModelCheckpoint(monitor="val_tanim_hit_rate@1",   mode="max", filename="tanim1-{epoch}-{step}")
        tanim5_ckpt   = ModelCheckpoint(monitor="val_tanim_hit_rate@5",   mode="max", filename="tanim5-{epoch}-{step}")
        tanim20_ckpt  = ModelCheckpoint(monitor="val_tanim_hit_rate@20",  mode="max", filename="tanim20-{epoch}-{step}")
        contiou1_ckpt  = ModelCheckpoint(monitor="val_contiou_hit_rate@1",  mode="max", filename="contiou1-{epoch}-{step}")
        contiou5_ckpt  = ModelCheckpoint(monitor="val_contiou_hit_rate@5",  mode="max", filename="contiou5-{epoch}-{step}")
        contiou20_ckpt = ModelCheckpoint(monitor="val_contiou_hit_rate@20", mode="max", filename="contiou20-{epoch}-{step}")
        val_ckpts += [
            cossim1_ckpt, cossim5_ckpt, cossim20_ckpt,
            tanim1_ckpt, tanim5_ckpt, tanim20_ckpt,
            contiou1_ckpt, contiou5_ckpt, contiou20_ckpt,
        ]
        test_on += [
            (cossim1_ckpt,  "val_cossim_hit_rate@1",   "cossim_hr@1"),
            (cossim5_ckpt,  "val_cossim_hit_rate@5",   "cossim_hr@5"),
            (cossim20_ckpt, "val_cossim_hit_rate@20",  "cossim_hr@20"),
            (tanim1_ckpt,   "val_tanim_hit_rate@1",    "tanim_hr@1"),
            (tanim5_ckpt,   "val_tanim_hit_rate@5",    "tanim_hr@5"),
            (tanim20_ckpt,  "val_tanim_hit_rate@20",   "tanim_hr@20"),
            (contiou1_ckpt,  "val_contiou_hit_rate@1",  "contiou_hr@1"),
            (contiou5_ckpt,  "val_contiou_hit_rate@5",  "contiou_hr@5"),
            (contiou20_ckpt, "val_contiou_hit_rate@20", "contiou_hr@20"),
        ]

    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="auto",
        gradient_clip_val=1,
        max_epochs=50,
        callbacks=val_ckpts,
        plugins=[LightningEnvironment()],
        logger=logger,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        precision=args.precision,
    )

    trainer.validate(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)

    val_res_dict = {"model": val_ckpts[0].best_model_path, "stage": "val"}
    for ckpt, monitor, name in test_on:
        res = trainer.validate(model, data_module.val_dataloader(), ckpt_path=ckpt.best_model_path)[0]
        val_res_dict[name] = res[monitor]

    test_res_dict = {"model": val_ckpts[0].best_model_path, "stage": "test"}
    for ckpt, monitor, name in test_on:
        res = trainer.validate(model, data_module.test_dataloader(), ckpt_path=ckpt.best_model_path)[0]
        test_res_dict[name] = res[monitor]

    append_dict_to_json_file(val_res_dict, os.path.join(args.logs_path, args.logging_file))
    append_dict_to_json_file(test_res_dict, os.path.join(args.logs_path, args.logging_file))


if __name__ == "__main__":
    main()
