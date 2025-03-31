from ms_mole.data import (
    RetrievalDataset_PrecompFPandInchi,
)
from massspecgym.data.transforms import MolFingerprinter, SpecBinner
from massspecgym.data.data_module import MassSpecDataModule
from ms_mole.models import FingerprintPredicter
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import LightningEnvironment
import argparse
import os
import ast

def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def main():
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        description="Pre-training script.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("dataset_path", type=str, metavar="dataset_path", help="dataset_path")
    parser.add_argument("helper_files_dir", type=str, metavar="helper_files_dir", help="helper_files_dir")
    parser.add_argument("logs_path", type=str, metavar="logs_path", help="logs_path")
    parser.add_argument("--skip_test", type=boolean, default=True)
    parser.add_argument("--df_test_path", type=str, default=None)
    parser.add_argument("--bonus_challenge", type=boolean, default=True,)

    parser.add_argument("--bin_width", type=float, default=0.1, help="bin_width")
    parser.add_argument("--batch_size", type=int, default=128, help="Bsz")
    parser.add_argument("--devices",type=ast.literal_eval,default=[0])
    parser.add_argument("--precision", type=str, default="bf16-mixed")

    parser.add_argument("--layer_dim", type=int, default=1024, help="layer dim")
    parser.add_argument("--n_layers", type=int, default=3, help="n layers in mlp")
    parser.add_argument("--dropout", type=float, default=0.25, help="dropout")
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--bitwise_loss", type=str, default=None, help="")
    parser.add_argument("--fpwise_loss", type=str, default=None, help="")
    parser.add_argument("--rankwise_loss", type=str, default=None, help="")

    parser.add_argument("--bitwise_lambd", type=float, default=1.0, help="")
    parser.add_argument("--fpwise_lambd", type=float, default=1.0, help="")
    parser.add_argument("--rankwise_lambd", type=float, default=1.0, help="")

    parser.add_argument("--bitwise_weighted", type=boolean, default=False, help="")
    parser.add_argument("--bitwise_fl_gamma", type=float, default=2.0, help="")

    parser.add_argument("--fpwise_iou_jml_v", type=boolean, default=True, help="")

    parser.add_argument("--rankwise_temp", type=float, default=1.0, help="")
    parser.add_argument("--rankwise_dropout", type=float, default=0.25, help="")
    parser.add_argument("--rankwise_sim_func", type=str, default="cossim", help="")
    parser.add_argument("--rankwise_projector", type=boolean, default=False, help="")
    


    args = parser.parse_args()

    dataset = RetrievalDataset_PrecompFPandInchi(
        spec_transform=SpecBinner(max_mz = 1005, bin_width=args.bin_width, to_rel_intensities=True),
        mol_transform=MolFingerprinter(fp_size=4096),
        pth=args.dataset_path,
        fp_pth=os.path.join(args.helper_files_dir, "fp_4096.npy"),
        inchi_pth=os.path.join(args.helper_files_dir, "inchis.npy"),
        candidates_pth=os.path.join(args.helper_files_dir, "MassSpecGym_retrieval_candidates_%s.json" % ("formula" if args.bonus_challenge else "mass")),
        candidates_fp_pth=os.path.join(args.helper_files_dir, "MassSpecGym_retrieval_candidates_%s_fps.npz" % ("formula" if args.bonus_challenge else "mass")),
        candidates_inchi_pth=os.path.join(args.helper_files_dir, "MassSpecGym_retrieval_candidates_%s_inchi.npz" % ("formula" if args.bonus_challenge else "mass")),
    )

    data_module = MassSpecDataModule(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=8,
    )

    data_module.prepare_data()
    data_module.setup()

    if args.bitwise_loss == "None":
        args.bitwise_loss = None
    if args.fpwise_loss == "None":
        args.fpwise_loss = None
    if args.rankwise_loss == "None":
        args.rankwise_loss = None

    loss_kwargs_dict = {
        "bce" : {"weighted" : args.bitwise_weighted},
        "fl" : {"gamma" : args.bitwise_fl_gamma, "weighted" : args.bitwise_weighted},
        "cossim" : {},
        "iou" : {"jml_version" : args.fpwise_iou_jml_v},
        "bienc" : {
            "temp": args.rankwise_temp,
            "n_bits" : 4096,
            "dropout":args.rankwise_dropout,
            "sim_func": args.rankwise_sim_func,
            "projector":args.rankwise_projector
            },
        "cross" : {
            "temp": args.rankwise_temp,
            "n_bits" : 4096,
            "dropout":args.rankwise_dropout,
            "projector":args.rankwise_projector
            },
        None : {},
    }

    model = FingerprintPredicter(
        n_in = int(1005/args.bin_width),  # number of bins
        layer_dims = [args.layer_dim] * args.n_layers,  # hidden layer sizes
        n_bits = 4096,  # fingerprint size
        layer_or_batchnorm = "layer",
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=0,
        df_test_path=args.df_test_path,
        bitwise_loss = args.bitwise_loss, # "bce", "fl"
        fpwise_loss = args.fpwise_loss, # "cossim", "iou"
        rankwise_loss = args.rankwise_loss, # "bienc", "cross"
        bitwise_lambd = args.bitwise_lambd,
        fpwise_lambd = args.fpwise_lambd,
        rankwise_lambd = args.rankwise_lambd,
        bitwise_kwargs = loss_kwargs_dict[args.bitwise_loss], # {"weighted" : False} / {"weighted" : False, "gamma" : 2}
        fpwise_kwargs = loss_kwargs_dict[args.fpwise_loss], # {} / {"jml_version" : True}
        rankwise_kwargs = loss_kwargs_dict[args.rankwise_loss], # {"temp": 0.1, "n_bits" : 4096, "dropout":0.2, sim_func:"cossim", "projector":False} / all same without sim_func

    )

    logger = TensorBoardLogger(
        "/".join(args.logs_path.split("/")[:-1]),
        name=args.logs_path.split("/")[-1],
    )

    val_ckpts = [
        ModelCheckpoint(monitor=None, filename="last-{epoch}-{step}"),
        ModelCheckpoint(monitor="val_loss", mode="max", filename="loss-{epoch}-{step}"),
        ModelCheckpoint(monitor="val_fingerprint_av_tanim", mode="max", filename="fpacctanim-{epoch}-{step}"),
        ModelCheckpoint(monitor="val_cossim_hit_rate@1", mode="max", filename="cossim1-{epoch}-{step}"),
        ModelCheckpoint(monitor="val_tanim_hit_rate@1", mode="max", filename="tanim1-{epoch}-{step}"),
        ModelCheckpoint(monitor="val_contiou_hit_rate@1", mode="max", filename="contiou1-{epoch}-{step}"),
        ModelCheckpoint(monitor="val_cossim_hit_rate@5", mode="max", filename="cossim5-{epoch}-{step}"),
        ModelCheckpoint(monitor="val_tanim_hit_rate@5", mode="max", filename="tanim5-{epoch}-{step}"),
        ModelCheckpoint(monitor="val_contiou_hit_rate@5", mode="max", filename="contiou5-{epoch}-{step}"),
        ModelCheckpoint(monitor="val_cossim_hit_rate@20", mode="max", filename="cossim20-{epoch}-{step}"),
        ModelCheckpoint(monitor="val_tanim_hit_rate@20", mode="max", filename="tanim20-{epoch}-{step}"),
        ModelCheckpoint(monitor="val_contiou_hit_rate@20", mode="max", filename="contiou20-{epoch}-{step}"),
    ]
    if args.rankwise_loss is not None:
        val_ckpts += [
            ModelCheckpoint(monitor="val_ranker_hit_rate@1", mode="max", filename="ranker1-{epoch}-{step}"),
            ModelCheckpoint(monitor="val_ranker_hit_rate@5", mode="max", filename="ranker5-{epoch}-{step}"),
            ModelCheckpoint(monitor="val_ranker_hit_rate@20", mode="max", filename="ranker20-{epoch}-{step}"),
        ]
    
    callbacks = val_ckpts

    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="auto",
        gradient_clip_val=1,
        max_epochs=50,
        callbacks=callbacks,
        plugins=[LightningEnvironment()],
        logger=logger,
        precision=args.precision,
    )

    trainer.validate(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)

    trainer.validate(model, datamodule=data_module, ckpt_path="best")
    if not args.skip_test:
        trainer.test(model, datamodule=data_module, ckpt_path="best")

if __name__ == '__main__':
    main()