import pandas as pd
#from massspecgym.data.transforms import MolFingerprinter
#from massspecgym.data.transforms import MolToInChIKey
import json
from tqdm import tqdm
import numpy as np
import argparse
from functools import partial
from rdkit.Chem import AllChem, DataStructs
from rdkit import Chem

class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
):
    pass

def main():
    parser = argparse.ArgumentParser(
        description="Data preprocessing launching pad. Choose a routine/datafile to preprocess.",
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "type",
        type=str,
        metavar="type",
        choices=[
            "inchi",
            "morgan_2_4096",
            "morgan_4_4096",
            "morgan_6_4096",
            "morgan_8_4096",
            "rdkit_4096",
            "biosynfoni",
            "maccs",
            "map4_4096",
        ],
        help="Type of data to preprocess, choices: {%(choices)s}",
    )
    parser.add_argument(
        "MassSpecGymTSV",
        type=str,
        metavar="MassSpecGymTSV",
        help="/path/to/MassSpecGym.tsv",
    )
    parser.add_argument(
        "candidates_mass_json",
        type=str,
        metavar="candidates_mass_json",
        help="/path/to/MassSpecGym_retrieval_candidates_mass.json",
    )
    parser.add_argument(
        "candidates_formula_json",
        type=str,
        metavar="candidates_formula_json",
        help="/path/to/MassSpecGym_retrieval_candidates_formula.json",
    )
    parser.add_argument(
        "save_target_transforms",
        type=str,
        metavar="save_target_transforms",
        help="/path/to/save_target_transforms.npy",
    )
    parser.add_argument(
        "save_candidate_mass_transforms",
        type=str,
        metavar="save_candidate_mass_transforms",
        help="/path/to/save_candidate_mass_transforms.npz",
    )
    parser.add_argument(
        "save_candidate_formula_transforms",
        type=str,
        metavar="save_candidate_formula_transforms",
        help="/path/to/save_candidate_formula_transforms.npz",
    )

    args = parser.parse_args()

    mapper = {
        "inchi": preprocess_inchi,
        "morgan_2_4096": partial(preprocess_morgan_4096, radius=2),
        "morgan_4_4096": partial(preprocess_morgan_4096, radius=4),
        "morgan_6_4096": partial(preprocess_morgan_4096, radius=6),
        "morgan_8_4096": partial(preprocess_morgan_4096, radius=8),
        "rdkit_4096": preprocess_rdkit_4096,
        "biosynfoni" : preprocess_biosynfoni,
        "map4_4096": partial(preprocess_map4, radius=2, fp_size = 4096),
        "maccs" : preprocess_maccs,
    }

    mapper[args.type](args)
    return None

def preprocess_inchi(args):
    mol_label_transform = MolToInChIKey()
    data = pd.read_csv(args.MassSpecGymTSV, sep="\t")

    print("Preprocessing targets")
    inchis = [mol_label_transform(data["smiles"][i]) for i in range(len(data))]
    np.save(args.save_target_transforms, np.array(inchis))

    print("Preprocessing mass-based candidates")
    with open(args.candidates_mass_json, "r") as file:
        candidates = json.load(file)
    candidates_inchi = {}
    for k, v in tqdm(candidates.items(), total=len(candidates)):
        candidates_inchi[k] = np.array([mol_label_transform(smile) for smile in v])
    np.savez(
    args.save_candidate_mass_transforms,
        **candidates_inchi
    )

    print("Preprocessing formula-based candidates")
    with open(args.candidates_formula_json, "r") as file:
        candidates = json.load(file)
    candidates_inchi = {}
    for k, v in tqdm(candidates.items(), total=len(candidates)):
        candidates_inchi[k] = np.array([mol_label_transform(smile) for smile in v])
    np.savez(
        args.save_candidate_formula_transforms,
        **candidates_inchi
    )
    return None

def preprocess_morgan_4096(args, radius=2):
    fingerprinter = MolFingerprinter(fp_size=4096, radius=radius)
    data = pd.read_csv(args.MassSpecGymTSV, sep="\t")

    print("Preprocessing targets")
    fps_target = [fingerprinter(data["smiles"][i]) for i in range(len(data))]
    np.save(args.save_target_transforms, np.packbits(np.stack(fps_target).astype(bool), axis=None))

    print("Preprocessing mass-based candidates")
    with open(args.candidates_mass_json, "r") as file:
        candidates = json.load(file)
    candidates_fp = {}
    for k, v in tqdm(candidates.items(), total=len(candidates)):
        candidates_fp[k] = np.packbits(
            np.stack([fingerprinter(smile) for smile in v]).astype(bool), axis=None
        )
    np.savez(args.save_candidate_mass_transforms,**candidates_fp)

    print("Preprocessing formula-based candidates")
    with open(args.candidates_formula_json, "r") as file:
        candidates = json.load(file)
    candidates_fp = {}
    for k, v in tqdm(candidates.items(), total=len(candidates)):
        candidates_fp[k] = np.packbits(
            np.stack([fingerprinter(smile) for smile in v]).astype(bool), axis=None
        )
    np.savez(
        args.save_candidate_formula_transforms,
        **candidates_fp
    )
    return None

class RdkitFingerprinter():
    def __init__(self, fp_size: int = 4096):
        self.fp_size = fp_size
        
    def __call__(self, mol : str):
        mol = Chem.MolFromSmiles(mol)
        fp = Chem.RDKFingerprint(mol, fpSize=self.fp_size)

        fp_np = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, fp_np)
        fp = fp_np
        return fp

def preprocess_rdkit_4096(args):
    fingerprinter = RdkitFingerprinter(fp_size=4096)
    data = pd.read_csv(args.MassSpecGymTSV, sep="\t")

    print("Preprocessing targets")
    fps_target = [fingerprinter(data["smiles"][i]) for i in range(len(data))]
    np.save(args.save_target_transforms, np.packbits(np.stack(fps_target).astype(bool), axis=None))

    print("Preprocessing mass-based candidates")
    with open(args.candidates_mass_json, "r") as file:
        candidates = json.load(file)
    candidates_fp = {}
    for k, v in tqdm(candidates.items(), total=len(candidates)):
        candidates_fp[k] = np.packbits(
            np.stack([fingerprinter(smile) for smile in v]).astype(bool), axis=None
        )
    np.savez(args.save_candidate_mass_transforms,**candidates_fp)

    print("Preprocessing formula-based candidates")
    with open(args.candidates_formula_json, "r") as file:
        candidates = json.load(file)
    candidates_fp = {}
    for k, v in tqdm(candidates.items(), total=len(candidates)):
        candidates_fp[k] = np.packbits(
            np.stack([fingerprinter(smile) for smile in v]).astype(bool), axis=None
        )
    np.savez(
        args.save_candidate_formula_transforms,
        **candidates_fp
    )
    return None

class MACCSFingerprinter():
    def __init__(self):
        None
        
    def __call__(self, mol : str):
        mol = Chem.MolFromSmiles(mol)
        fp = AllChem.GetMACCSKeysFingerprint(mol)

        fp_np = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, fp_np)
        fp = fp_np
        return fp

def preprocess_maccs(args):
    fingerprinter = MACCSFingerprinter()
    data = pd.read_csv(args.MassSpecGymTSV, sep="\t")

    print("Preprocessing targets")
    fps_target = [fingerprinter(data["smiles"][i]) for i in range(len(data))]
    np.save(args.save_target_transforms, np.packbits(np.stack(fps_target).astype(bool), axis=None))

    print("Preprocessing mass-based candidates")
    with open(args.candidates_mass_json, "r") as file:
        candidates = json.load(file)
    candidates_fp = {}
    for k, v in tqdm(candidates.items(), total=len(candidates)):
        candidates_fp[k] = np.packbits(
            np.stack([fingerprinter(smile) for smile in v]).astype(bool), axis=None
        )
    np.savez(args.save_candidate_mass_transforms,**candidates_fp)

    print("Preprocessing formula-based candidates")
    with open(args.candidates_formula_json, "r") as file:
        candidates = json.load(file)
    candidates_fp = {}
    for k, v in tqdm(candidates.items(), total=len(candidates)):
        candidates_fp[k] = np.packbits(
            np.stack([fingerprinter(smile) for smile in v]).astype(bool), axis=None
        )
    np.savez(
        args.save_candidate_formula_transforms,
        **candidates_fp
    )
    return None


def preprocess_maccs(args):
    fingerprinter = MACCSFingerprinter()
    data = pd.read_csv(args.MassSpecGymTSV, sep="\t")

    print("Preprocessing targets")
    fps_target = [fingerprinter(data["smiles"][i]) for i in range(len(data))]
    np.save(args.save_target_transforms, np.packbits(np.stack(fps_target).astype(bool), axis=None))

    print("Preprocessing mass-based candidates")
    with open(args.candidates_mass_json, "r") as file:
        candidates = json.load(file)
    candidates_fp = {}
    for k, v in tqdm(candidates.items(), total=len(candidates)):
        candidates_fp[k] = np.packbits(
            np.stack([fingerprinter(smile) for smile in v]).astype(bool), axis=None
        )
    np.savez(args.save_candidate_mass_transforms,**candidates_fp)

    print("Preprocessing formula-based candidates")
    with open(args.candidates_formula_json, "r") as file:
        candidates = json.load(file)
    candidates_fp = {}
    for k, v in tqdm(candidates.items(), total=len(candidates)):
        candidates_fp[k] = np.packbits(
            np.stack([fingerprinter(smile) for smile in v]).astype(bool), axis=None
        )
    np.savez(
        args.save_candidate_formula_transforms,
        **candidates_fp
    )
    return None


#from biosynfoni import Biosynfoni
class BiosynfoniFingerprinter():
    def __init__(self):
        None
        
    def __call__(self, mol : str):
        mol = Chem.MolFromSmiles(mol)
        fp = Biosynfoni(mol).fingerprint
        return (np.array(fp) > 0).astype(np.int32)
    
def preprocess_biosynfoni(args):
    fingerprinter = BiosynfoniFingerprinter()
    data = pd.read_csv(args.MassSpecGymTSV, sep="\t")

    print("Preprocessing targets")
    fps_target = [fingerprinter(data["smiles"][i]) for i in range(len(data))]
    np.save(args.save_target_transforms, np.packbits(np.stack(fps_target).astype(bool), axis=None))

    print("Preprocessing mass-based candidates")
    with open(args.candidates_mass_json, "r") as file:
        candidates = json.load(file)
    candidates_fp = {}
    for k, v in tqdm(candidates.items(), total=len(candidates)):
        candidates_fp[k] = np.packbits(
            np.stack([fingerprinter(smile) for smile in v]).astype(bool), axis=None
        )
    np.savez(args.save_candidate_mass_transforms,**candidates_fp)

    print("Preprocessing formula-based candidates")
    with open(args.candidates_formula_json, "r") as file:
        candidates = json.load(file)
    candidates_fp = {}
    for k, v in tqdm(candidates.items(), total=len(candidates)):
        candidates_fp[k] = np.packbits(
            np.stack([fingerprinter(smile) for smile in v]).astype(bool), axis=None
        )
    np.savez(
        args.save_candidate_formula_transforms,
        **candidates_fp
    )
    return None

#from map4 import MAP4Calculator
class MAPFingerprinter():
    def __init__(self, radius: int = 2, fp_size: int = 4096):
        self.map_calc = MAP4Calculator(dimensions=fp_size, radius=radius, is_folded=True)
        
    def __call__(self, mol : str):
        mol = Chem.MolFromSmiles(mol)
        fp = self.map_calc.calculate(mol)

        return fp.astype(np.int32)

def preprocess_map4(args, radius=2, fp_size=4096):
    fingerprinter = MAPFingerprinter(radius=radius, fp_size=fp_size)
    data = pd.read_csv(args.MassSpecGymTSV, sep="\t")

    print("Preprocessing targets")
    fps_target = [fingerprinter(data["smiles"][i]) for i in range(len(data))]
    np.save(args.save_target_transforms, np.packbits(np.stack(fps_target).astype(bool), axis=None))

    print("Preprocessing mass-based candidates")
    with open(args.candidates_mass_json, "r") as file:
        candidates = json.load(file)
    candidates_fp = {}
    for k, v in tqdm(candidates.items(), total=len(candidates)):
        candidates_fp[k] = np.packbits(
            np.stack([fingerprinter(smile) for smile in v]).astype(bool), axis=None
        )
    np.savez(args.save_candidate_mass_transforms,**candidates_fp)

    print("Preprocessing formula-based candidates")
    with open(args.candidates_formula_json, "r") as file:
        candidates = json.load(file)
    candidates_fp = {}
    for k, v in tqdm(candidates.items(), total=len(candidates)):
        candidates_fp[k] = np.packbits(
            np.stack([fingerprinter(smile) for smile in v]).astype(bool), axis=None
        )
    np.savez(
        args.save_candidate_formula_transforms,
        **candidates_fp
    )
    return None


if __name__ == "__main__":
    main()