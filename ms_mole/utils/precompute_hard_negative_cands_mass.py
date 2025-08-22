from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula, CalcExactMolWt
import pandas as pd
from massspecgym.data.transforms import MolFingerprinter
import torch
import numpy as np
from massspecgym.data.transforms import MolToInChIKey
from tqdm import tqdm
import json

df_pubchem = pd.read_csv(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_molecules_pubchem_118M.tsv",
    sep="\t",
)
df_pubchem.shape, df_pubchem.columns


df_pubchem = df_pubchem.sort_values("mass")

df_massspecgym = pd.read_csv("/data/home/gaetandw/msms/data/MassSpecGym.tsv", sep="\t")
smiles = df_massspecgym["smiles"]
print(len(df_massspecgym), len(smiles))
smiles = smiles.drop_duplicates()
print(len(df_massspecgym), len(smiles))

smiles_to_mass = {}
for sm in smiles:
    mol = Chem.MolFromSmiles(sm)
    mass = CalcExactMolWt(mol)
    smiles_to_mass[sm] = mass

smiles_by_mass = [k for k, v in sorted(smiles_to_mass.items(), key=lambda item: item[1])]

start_to_look_in = 0
smiles_to_matches = []
for sm in tqdm(smiles_by_mass, total=len(smiles_by_mass)):

    mass = smiles_to_mass[sm]
    mass_eps = mass * 1e-6 * 10  # 10 ppm

    matches = []

    c = 0
    mass_cand = df_pubchem["mass"].iloc[start_to_look_in+c]
    while not (abs(mass_cand - mass) < mass_eps):
        c += 1
        mass_cand = df_pubchem["mass"].iloc[start_to_look_in+c]

    while (abs(mass_cand - mass) < mass_eps):
        matches.append(start_to_look_in+c)
        c+=1
        mass_cand = df_pubchem["mass"].iloc[start_to_look_in+c]

    start_to_look_in = matches[0]
    smiles_to_matches.append(matches)

t = []
for sm, matches in zip(smiles_by_mass, smiles_to_matches):
    mol = Chem.MolFromSmiles(sm)
    inchi_key_2D = Chem.MolToInchiKey(mol).split("-")[0]


    subs_form = df_pubchem.iloc[matches]
    subs_form = subs_form[subs_form["inchi_key_2D"] != inchi_key_2D]
    t.append(subs_form["smiles"].values)

def batch_samplewise_tanimoto(pred_fp, true_fp, threshold=0.5, reduce=False):
    _and = (true_fp.int() & (pred_fp > threshold)).sum(-1)
    _or = (true_fp.int() | (pred_fp > threshold)).sum(-1)
    if reduce:
        return (_and / _or).mean()
    else:
        return _and / _or


fingerprinter = MolFingerprinter(fp_size=4096)

similarities = []
from tqdm import tqdm
cands_keep_all = []
for i in tqdm(range(len(t)), total=len(t)):
    cands = t[i]
    query_mol = smiles_by_mass[i]

    if len(cands) > 0:
        cands_keep = []
        fingerprints_keep = []
        for smile in cands:
            try:
                fingerprints_keep.append(fingerprinter(smile))
                cands_keep.append(smile)
            except:
                continue
        pp = np.stack(fingerprints_keep).astype(bool)

        similarities.append(
            batch_samplewise_tanimoto(
                torch.tensor(fingerprinter(query_mol)), torch.tensor(pp)
            ).numpy()
        )
        cands_keep_all.append(cands_keep)
    else:
        similarities.append([])

cands = {k: np.array(v) for k, v in zip(smiles_by_mass, cands_keep_all)}
sims = {k: v for k, v in zip(smiles_by_mass, similarities)}




candidates_fp_random = {}
candidates_fp_hard = {}
candidates_inchi_random = {}
candidates_inchi_hard = {}
candidates_json_random = {}
candidates_json_hard = {}

fingerprinter = MolFingerprinter(fp_size=4096)
mol_label_transform = MolToInChIKey()

for sm in tqdm(list(cands), total=len(cands)):
    sims_sample = sims[sm]
    cands_sample = cands[sm]

    asort = np.argsort(sims_sample)[::-1]

    sims_sample_asort = sims_sample[asort]
    cands_sample_asort = cands_sample[asort]

    if len(cands_sample_asort) > 1023:
        sampled_random = np.random.choice(
            len(cands_sample_asort), (1023,), replace=False
        )
        cands_random = [sm] + list(cands_sample_asort[sampled_random])
        cands_hardneg = [sm] + list(cands_sample_asort[:1023])
    else:
        cands_random = [sm] + list(cands_sample_asort)
        cands_hardneg = [sm] + list(cands_sample_asort)

    candidates_fp_random[sm] = np.packbits(
        np.stack([fingerprinter(smile) for smile in cands_random]).astype(bool),
        axis=None,
    )
    candidates_fp_hard[sm] = np.packbits(
        np.stack([fingerprinter(smile) for smile in cands_hardneg]).astype(bool),
        axis=None,
    )

    candidates_inchi_random[sm] = np.array(
        [mol_label_transform(smile) for smile in cands_random]
    )
    candidates_inchi_hard[sm] = np.array(
        [mol_label_transform(smile) for smile in cands_hardneg]
    )

    candidates_json_hard[sm] = cands_hardneg
    candidates_json_random[sm] = cands_random

with open(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_candidates_massrandom.json", "w"
) as fp:
    json.dump(candidates_json_random, fp)
with open(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_candidates_masshard.json", "w"
) as fp:
    json.dump(candidates_json_hard, fp)

np.savez(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_candidates_massrandom_inchi.npz",
    **candidates_inchi_random
)
np.savez(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_candidates_masshard_inchi.npz",
    **candidates_inchi_hard
)
np.savez(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_candidates_massrandom_fps.npz",
    **candidates_fp_random
)
np.savez(
    "/data/home/gaetandw/msms/data/MassSpecGym_retrieval_candidates_masshard_fps.npz",
    **candidates_fp_hard
)
