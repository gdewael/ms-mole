from massspecgym.data.datasets import RetrievalDataset
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from massspecgym.data.datasets import MassSpecDataset
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from torch.utils.data.dataset import Subset


def bits_to_fparray(arr):
    return np.unpackbits(arr).reshape(-1, 4096).astype(bool)


class RetrievalDataset_PrecompFPandInchi(RetrievalDataset):
    def __init__(
        self,
        fp_pth=None,
        inchi_pth=None,
        candidates_fp_pth=None,
        candidates_inchi_pth=None,
        n_max_cands=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metadata["fp_4096"] = list(bits_to_fparray(np.load(fp_pth)))
        self.metadata["inchikey"] = list(np.load(inchi_pth))

        self.candidate_fps = dict(np.load(candidates_fp_pth))
        self.candidate_inchi = dict(np.load(candidates_inchi_pth))
        self.n_max_cands = n_max_cands

    def __getitem__(self, i):

        item = super(RetrievalDataset, self).__getitem__(i, transform_mol=False)

        # Save the original SMILES representation of the query molecule (for evaluation)
        item["smiles"] = item["mol"]

        # Get candidates
        if item["mol"] not in self.candidates:
            raise ValueError(f'No candidates for the query molecule {item["mol"]}.')
        item["candidates"] = get_n_max(self.candidates[item["mol"]], self.n_max_cands)

        # Save the original SMILES representations of the canidates (for evaluation)
        item["candidates_smiles"] = item["candidates"]

        # Transform the query and candidate molecules
        item["mol"] = self.metadata["fp_4096"].iloc[i].astype(np.int32)
        item["candidates"] = get_n_max(
            bits_to_fparray(self.candidate_fps[item["smiles"]]), self.n_max_cands
        )
        if isinstance(item["mol"], np.ndarray):
            item["mol"] = torch.as_tensor(item["mol"], dtype=self.dtype)
        if isinstance(item["candidates"], np.ndarray):
            item["candidates"] = torch.as_tensor(item["candidates"], dtype=self.dtype)

        item["labels"] = [(c == item["mol"]).all().item() for c in item["candidates"]]

        if not any(item["labels"]):
            raise ValueError(
                f'Query molecule {item["mol"]} not found in the candidates list.'
            )

        return item


# adapted from https://github.com/pluskal-lab/MassSpecGym/blob/main/massspecgym/data/data_module.py
class MsMoleMassSpecDataModule(pl.LightningDataModule):
    """
    Data module containing a mass spectrometry dataset. This class is responsible for loading, splitting, and wrapping
    the dataset into data loaders according to pre-defined train, validation, test folds.
    """

    def __init__(
        self,
        pth: str = None,
        fp_pth: str = None,
        inchi_pth=None,
        train_cands_pth=None,
        train_cands_fp_pth=None,
        train_cands_inchi_pth=None,
        valtest_cands_pth=None,
        valtest_cands_fp_pth=None,
        valtest_cands_inchi_pth=None,
        train_n_max_cands=None,
        batch_size: int = 64,
        num_workers: int = 8,
        persistent_workers: bool = True,
        spec_transform=None,
        mol_transform=None,
    ):
        """
        Args:
            split_pth (Optional[Path], optional): Path to a .tsv file with columns "identifier" and "fold",
                corresponding to dataset item IDs, and "fold", containg "train", "val", "test"
                values. Default is None, in which case the split from the `dataset` is used.
        """
        super().__init__()
        self.pth = pth
        self.fp_pth = fp_pth
        self.inchi_pth = inchi_pth
        self.train_cands_pth = train_cands_pth
        self.train_cands_fp_pth = train_cands_fp_pth
        self.train_cands_inchi_pth = train_cands_inchi_pth
        self.valtest_cands_pth = valtest_cands_pth
        self.valtest_cands_fp_pth = valtest_cands_fp_pth
        self.valtest_cands_inchi_pth = valtest_cands_inchi_pth
        self.train_n_max_cands = train_n_max_cands
        self.batch_size = batch_size
        self.spec_transform = spec_transform
        self.mol_transform = mol_transform
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers if num_workers > 0 else False

        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        """Pre-processing to be executed only on a single main device when using distributed training."""
        pass

    def setup(self, stage=None):
        """Pre-processing to be executed on every device when using distributed training."""

        if self.train_dataset is None:
            self.dataset = RetrievalDataset_PrecompFPandInchi(
                spec_transform=self.spec_transform,
                mol_transform=self.mol_transform,
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
            # Split dataset
            split_mask = self.split.loc[self.dataset.metadata["identifier"]].values
            self.train_dataset = Subset(
                self.dataset, np.where(split_mask == "train")[0]
            )

        if self.val_dataset is None:
            valtest_dataset = RetrievalDataset_PrecompFPandInchi(
                spec_transform=self.spec_transform,
                mol_transform=self.mol_transform,
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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )


def get_n_max(object, n):
    if n is not None:
        return object[:n]
    else:
        return object
