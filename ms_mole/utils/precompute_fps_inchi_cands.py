import pandas as pd
from massspecgym.data.transforms import MolFingerprinter
from massspecgym.data.transforms import SpecTransform, MolTransform, MolToInChIKey
import json
from tqdm import tqdm
import numpy as np


fingerprinter = MolFingerprinter(fp_size=4096)
mol_label_transform = MolToInChIKey()

## 1. precomp fingerprints of mols themselves:
data = pd.read_csv("/data/home/gaetandw/msms/data/MassSpecGym.tsv", sep="\t")
fp_4096 = [fingerprinter(data["smiles"][i]) for i in range(len(data))]
np.save("./fp_4096.npy", np.stack(fp_4096).astype(bool))
inchis = [mol_label_transform(data["smiles"][i]) for i in range(len(data))]
np.save("./inchis.npy", np.array(inchis))

## 2. precomp fingerprints of cands (1):
with open(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_candidates_mass.json", "r"
) as file:
    candidates = json.load(file)
candidates_fp = {}
for k, v in tqdm(candidates.items(), total=len(candidates)):
    candidates_fp[k] = np.packbits(
        np.stack([fingerprinter(smile) for smile in v]).astype(bool), axis=None
    )
np.savez(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_candidates_mass_fps.npz",
    **candidates_fp
)

## 2. precomp fingerprints of cands (2):
with open(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_candidates_formula.json", "r"
) as file:
    candidates = json.load(file)
candidates_fp = {}
for k, v in tqdm(candidates.items(), total=len(candidates)):
    candidates_fp[k] = np.packbits(
        np.stack([fingerprinter(smile) for smile in v]).astype(bool), axis=None
    )
np.savez(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_candidates_formula_fps.npz",
    **candidates_fp
)

## 3. precomp inchi of cands (1):
with open(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_candidates_mass.json", "r"
) as file:
    candidates = json.load(file)
candidates_inchi = {}
for k, v in tqdm(candidates.items(), total=len(candidates)):
    candidates_inchi[k] = np.array([mol_label_transform(smile) for smile in v])
np.savez(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_candidates_mass_inchi.npz",
    **candidates_inchi
)

## 3. precomp inchi of cands (2):
with open(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_candidates_formula.json", "r"
) as file:
    candidates = json.load(file)
candidates_inchi = {}
for k, v in tqdm(candidates.items(), total=len(candidates)):
    candidates_inchi[k] = np.array([mol_label_transform(smile) for smile in v])
np.savez(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_candidates_formula_inchi.npz",
    **candidates_inchi
)
