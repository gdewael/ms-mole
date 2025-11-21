from ms_mole.data import RetrievalDataset_PrecompFPandInchi, MsMoleMassSpecDataModule
from massspecgym.data.transforms import MolFingerprinter, SpecBinner
from ms_mole.models import FingerprintPredicter
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import LightningEnvironment
import argparse
import os
import ast
import ms_mole.loss as loss
import json


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
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def none_or_int(value):
    if value is None:
        return None
    if value == "None":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value: '{value}'")


def main():
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        description="Molecular Retrieval script.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument(
        "dataset_path", type=str, metavar="dataset_path", help="dataset_path"
    )
    parser.add_argument(
        "helper_files_dir",
        type=str,
        metavar="helper_files_dir",
        help="helper_files_dir",
    )
    parser.add_argument("logs_path", type=str, metavar="logs_path", help="logs_path")
    parser.add_argument(
        "--logging_file", type=str, default="res.json"
    )

    parser.add_argument(
        "--candidate_setting_train", type=str, choices=["mass", "formula"],
    )
    parser.add_argument(
        "--candidate_setting_eval", type=str, choices=["mass", "formula"],
    )
    parser.add_argument(
        "--fp_type", type=str,
        choices=[
            "morgan_2_4096",
            "morgan_4_4096",
            "morgan_6_4096",
            "morgan_8_4096",
            "rdkit_4096",
            "map4_4096"
        ],
        default="morgan_2_4096",
    )
    parser.add_argument("--n_max_cands", type=none_or_int, default=None)

    parser.add_argument("--bin_width", type=float, default=0.1, help="bin_width")

    parser.add_argument("--batch_size", type=int, default=64, help="Bsz")
    parser.add_argument("--n_workers", type=int, default=8, help="n_workers")
    parser.add_argument("--devices", type=ast.literal_eval, default=[0])
    parser.add_argument("--precision", type=str, default="32-true")

    parser.add_argument("--layer_dim", type=int, default=1024, help="layer dim")
    parser.add_argument("--n_layers", type=int, default=3, help="n layers in mlp")
    parser.add_argument("--dropout", type=float, default=0.25, help="dropout")
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument(
        "--loss",
        type=str,
        default="cosine",
        choices=[
            "bce",
            "fl",
            "cosine",
            "iou",
            "list_fp_cos",
            "list_embed_cos",
            "list_fp_cross",
            "list_embed_cross",
            "rnn_01",
        ],
    )
    parser.add_argument("--fl_gamma", type=float, default=5.0, help="")
    parser.add_argument("--temp", type=float, default=1.0, help="")
    parser.add_argument("--contrastive_dim", type=int, default=256, help="")
    parser.add_argument("--rankwise_listwise", type=boolean, default=True, help="")

    parser.add_argument("--checkpoint_path", type=str, default=None, help="")
    parser.add_argument("--freeze_checkpoint", type=boolean, default=False, help="")

    args = parser.parse_args()

    data_module = MsMoleMassSpecDataModule(
        pth=args.dataset_path,
        fp_pth=os.path.join(args.helper_files_dir, "%s_targets.npy" % args.fp_type),
        inchi_pth=os.path.join(args.helper_files_dir, "Inchis_targets.npy"),
        train_cands_pth=os.path.join(
            args.helper_files_dir,
            "MassSpecGym_retrieval_candidates_%s.json" % args.candidate_setting_train,
        ),
        train_cands_fp_pth=os.path.join(
            args.helper_files_dir,
            "%s_%scands.npz"
            % (args.fp_type, args.candidate_setting_train),
        ),
        train_cands_inchi_pth=os.path.join(
            args.helper_files_dir,
            "Inchis_%scands.npz" % args.candidate_setting_train,
        ),
        valtest_cands_pth=os.path.join(
            args.helper_files_dir,
            "MassSpecGym_retrieval_candidates_%s.json" % args.candidate_setting_eval,
        ),
        valtest_cands_fp_pth=os.path.join(
            args.helper_files_dir,
            "%s_%scands.npz"
            % (args.fp_type, args.candidate_setting_eval),
        ),
        valtest_cands_inchi_pth=os.path.join(
            args.helper_files_dir,
            "Inchis_%scands.npz" % args.candidate_setting_eval,
        ),
        train_n_max_cands=args.n_max_cands,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        spec_transform=SpecBinner(
            max_mz=1005, bin_width=args.bin_width, to_rel_intensities=True
        ),
        mol_transform=MolFingerprinter(fp_size=4096),
    )

    data_module.setup()

    # if args.checkpoint_path == "None":
    #    args.checkpoint_path = None

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
    }

    model = FingerprintPredicter(
        n_in=int(1005 / args.bin_width),  # number of bins
        layer_dims=[args.layer_dim] * args.n_layers,  # hidden layer sizes
        layer_or_batchnorm="layer",
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=0,
        df_test_path=None,
        loss=args.loss,
        loss_kwargs=loss_kwargs_dict[args.loss],
    )

    # if args.checkpoint_path is not None:
    #    pretrained_model = FingerprintPredicter.load_from_checkpoint(args.checkpoint_path)

    #    pretrained_mlp_state_dict = pretrained_model.mlp.state_dict()
    #    model_mlp_statedict = model.mlp.state_dict()
    #    model_mlp_statedict.update(pretrained_mlp_state_dict)
    #    model.mlp.load_state_dict(model_mlp_statedict)

    #    pretrained_fppredhead_state_dict = pretrained_model.loss.fp_pred_head.state_dict()
    #    model_fppredhead_statedict = model.loss.fp_pred_head.state_dict()
    #    model_fppredhead_statedict.update(pretrained_fppredhead_state_dict)
    #    model.loss.fp_pred_head.load_state_dict(model_fppredhead_statedict)

    #    if args.freeze_checkpoint:
    #        model.mlp.requires_grad_(False)
    #        model.loss.fp_pred_head.requires_grad_(False)

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
        ),
    ):
        ranker1_ckpt = ModelCheckpoint(
            monitor="val_ranker_hit_rate@1",
            mode="max",
            filename="ranker1-{epoch}-{step}",
        )
        ranker5_ckpt = ModelCheckpoint(
            monitor="val_ranker_hit_rate@5",
            mode="max",
            filename="ranker5-{epoch}-{step}",
        )
        ranker20_ckpt = ModelCheckpoint(
            monitor="val_ranker_hit_rate@20",
            mode="max",
            filename="ranker20-{epoch}-{step}",
        )
        val_ckpts += [ranker1_ckpt, ranker5_ckpt, ranker20_ckpt]
        test_on += [
            (ranker1_ckpt, "val_ranker_hit_rate@1", "contrastive_hr@1"),
            (ranker5_ckpt, "val_ranker_hit_rate@5", "contrastive_hr@5"),
            (ranker20_ckpt, "val_ranker_hit_rate@20", "contrastive_hr@20"),
        ]

    else:
        cossim1_ckpt = ModelCheckpoint(
            monitor="val_cossim_hit_rate@1",
            mode="max",
            filename="cossim1-{epoch}-{step}",
        )
        cossim5_ckpt = ModelCheckpoint(
            monitor="val_cossim_hit_rate@5",
            mode="max",
            filename="cossim5-{epoch}-{step}",
        )
        cossim20_ckpt = ModelCheckpoint(
            monitor="val_cossim_hit_rate@20",
            mode="max",
            filename="cossim20-{epoch}-{step}",
        )
        tanim1_ckpt = ModelCheckpoint(
            monitor="val_tanim_hit_rate@1", mode="max", filename="tanim1-{epoch}-{step}"
        )
        tanim5_ckpt = ModelCheckpoint(
            monitor="val_tanim_hit_rate@5", mode="max", filename="tanim5-{epoch}-{step}"
        )
        tanim20_ckpt = ModelCheckpoint(
            monitor="val_tanim_hit_rate@20",
            mode="max",
            filename="tanim20-{epoch}-{step}",
        )
        contiou1_ckpt = ModelCheckpoint(
            monitor="val_contiou_hit_rate@1",
            mode="max",
            filename="contiou1-{epoch}-{step}",
        )
        contiou5_ckpt = ModelCheckpoint(
            monitor="val_contiou_hit_rate@5",
            mode="max",
            filename="contiou5-{epoch}-{step}",
        )
        contiou20_ckpt = ModelCheckpoint(
            monitor="val_contiou_hit_rate@20",
            mode="max",
            filename="contiou20-{epoch}-{step}",
        )
        val_ckpts += [
            cossim1_ckpt,
            cossim5_ckpt,
            cossim20_ckpt,
            tanim1_ckpt,
            tanim5_ckpt,
            tanim20_ckpt,
            contiou1_ckpt,
            contiou5_ckpt,
            contiou20_ckpt,
        ]
        test_on += [
            (cossim1_ckpt, "val_cossim_hit_rate@1", "cossim_hr@1"),
            (cossim5_ckpt, "val_cossim_hit_rate@5", "cossim_hr@5"),
            (cossim20_ckpt, "val_cossim_hit_rate@20", "cossim_hr@20"),
            (tanim1_ckpt, "val_tanim_hit_rate@1", "tanim_hr@1"),
            (tanim5_ckpt, "val_tanim_hit_rate@5", "tanim_hr@5"),
            (tanim20_ckpt, "val_tanim_hit_rate@20", "tanim_hr@20"),
            (contiou1_ckpt, "val_contiou_hit_rate@1", "contiou_hr@1"),
            (contiou5_ckpt, "val_contiou_hit_rate@5", "contiou_hr@5"),
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

    val_res_dict = {
        "model": val_ckpts[0].best_model_path,
        "stage": "val",
    }
    for case in test_on:
        ckpt, monitor, name = case
        res = trainer.validate(
            model, data_module.val_dataloader(), ckpt_path=ckpt.best_model_path
        )[0]
        val_res_dict[name] = res[monitor]

    test_res_dict = {
        "model": val_ckpts[0].best_model_path,
        "stage": "test",
    }
    for case in test_on:
        ckpt, monitor, name = case
        res = trainer.validate(
            model, data_module.test_dataloader(), ckpt_path=ckpt.best_model_path
        )[0]
        test_res_dict[name] = res[monitor]

    append_dict_to_json_file(val_res_dict, os.path.join(args.logs_path, args.logging_file))
    append_dict_to_json_file(test_res_dict, os.path.join(args.logs_path, args.logging_file))


if __name__ == "__main__":
    main()
