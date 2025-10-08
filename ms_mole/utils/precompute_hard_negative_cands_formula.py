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
import sys


pubchem_path = str(sys.argv[1]) #/path/to/MassSpecGym_retrieval_molecules_pubchem_118M.tsv
MassSpecGymTSV = str(sys.argv[2]) #/path/to/MassSpecGym.tsv
save_cands_json = str(sys.argv[3]) #/path/to/MassSpecGym_retrieval_candidates_hard.json
save_cands_fps = str(sys.argv[4]) #/path/to/MassSpecGym_retrieval_candidates_hard_fps.npz
save_cands_inchi=  str(sys.argv[5]) #/path/to/MassSpecGym_retrieval_candidates_hard_inchi.npz

df_pubchem = pd.read_csv(
    pubchem_path,
    sep="\t",
)
df_pubchem.shape, df_pubchem.columns


df_massspecgym = pd.read_csv(MassSpecGymTSV, sep="\t")
smiles = df_massspecgym["smiles"]
print(len(df_massspecgym), len(smiles))
smiles = smiles.drop_duplicates()
print(len(df_massspecgym), len(smiles))


formula_dict = dict()
for i in tqdm(range(len(df_pubchem["formula"])), total=len(df_pubchem["formula"])):
    form = df_pubchem["formula"].iloc[i]
    if form not in formula_dict:
        formula_dict[form] = [i]
    else:
        formula_dict[form].append(i)


t = []
for sm in smiles:
    mol = Chem.MolFromSmiles(sm)
    inchi_key_2D = Chem.MolToInchiKey(mol).split("-")[0]
    formula = CalcMolFormula(mol)

    try:
        formula_matches = formula_dict[formula]
    except:
        formula_matches = []

    subs_form = df_pubchem.iloc[formula_matches]
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

for i in tqdm(range(len(t)), total=len(t)):
    cands = t[i]
    query_mol = smiles.iloc[i]

    if len(cands) > 0:
        pp = np.stack([fingerprinter(smile) for smile in cands]).astype(bool)

        similarities.append(
            batch_samplewise_tanimoto(
                torch.tensor(fingerprinter(query_mol)), torch.tensor(pp)
            ).numpy()
        )
    else:
        similarities.append([])

cands = {k: v for k, v in zip(smiles, t)}
sims = {k: v for k, v in zip(smiles, similarities)}


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
    save_cands_json, "w"
) as fp:
    json.dump(candidates_json_hard, fp)
np.savez(
    save_cands_inchi,
    **candidates_inchi_hard
)
np.savez(
    save_cands_fps,
    **candidates_fp_hard
)
