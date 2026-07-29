"""
Precompute DreaMS spectrum embeddings for a MassSpecGym-format TSV.

Reads mzs / intensities / precursor_mz from each TSV row, runs them through
the pretrained DreaMS embedding model, and saves a (N_rows, 1024) float32
numpy array whose row order matches the TSV row order.  Works for both
MassSpecGym and enveda TSVs (same format).

Requirements:
    pip install "dreams @ git+https://github.com/pluskal-lab/DreaMS.git"

Usage:
    python precompute_dreams_embeddings.py \
        data/enveda/enveda.tsv \
        data/enveda/dreams_embeddings.npy \
        [--batch_size 256] [--device cuda]
"""

import argparse
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dreams.api import PreTrainedModel
from dreams.utils.dformats import DataFormatA
from dreams.utils.data import SpectrumPreprocessor


EMBED_DIM = 1024       # DreaMS output size


class _SpectraDataset(Dataset):
    """Reads a MassSpecGym-format TSV and returns preprocessed DreaMS tensors."""

    def __init__(self, tsv_path):
        self.preprocessor = SpectrumPreprocessor(DataFormatA(), n_highest_peaks=100)
        self.rows = []  # list of (mzs_array, ints_array, precursor_mz)

        print(f"Reading {tsv_path} ...")
        with open(tsv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                mzs = np.array([float(x) for x in row["mzs"].split(",")])
                ints = np.array([float(x) for x in row["intensities"].split(",")])
                prec_mz = float(row["precursor_mz"])
                self.rows.append((mzs, ints, prec_mz))

        print(f"  {len(self.rows)} spectra loaded.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        mzs, ints, prec_mz = self.rows[i]
        spectrum = np.stack([mzs, ints], axis=1)  # (n_peaks, 2)
        ms = self.preprocessor(spectrum, prec_mz, high_form=True)
        return torch.tensor(ms, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute DreaMS embeddings from a MassSpecGym-format TSV."
    )
    parser.add_argument("tsv_path", type=str, help="Path to MassSpecGym-format .tsv")
    parser.add_argument("out_npy", type=str, help="Output path for (N, 1024) embeddings .npy")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = _SpectraDataset(args.tsv_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    print(f"Loading DreaMS model on {args.device} ...")
    model = PreTrainedModel.from_name("DreaMS_embedding").model
    model = model.to(args.device).eval()

    all_embs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding"):
            embs = model(batch.to(args.device))   # (B, 1024)
            all_embs.append(embs.cpu().float().numpy())

    embeddings = np.concatenate(all_embs, axis=0)   # (N, 1024)
    assert embeddings.shape == (len(dataset), EMBED_DIM), embeddings.shape

    np.save(args.out_npy, embeddings)
    print(f"Saved {embeddings.shape} to {args.out_npy}")


if __name__ == "__main__":
    main()
