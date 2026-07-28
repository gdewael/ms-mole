import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import argparse
import os
from functools import partial
from multiprocessing import Pool
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
    parser.add_argument(
        "--n_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel worker processes (default: all CPU cores).",
    )

    args = parser.parse_args()

    mapper = {
        "inchi": preprocess_inchi,
        "morgan_2_4096": partial(preprocess_morgan_4096, radius=2),
        "morgan_4_4096": partial(preprocess_morgan_4096, radius=4),
        "morgan_6_4096": partial(preprocess_morgan_4096, radius=6),
        "morgan_8_4096": partial(preprocess_morgan_4096, radius=8),
        "rdkit_4096": preprocess_rdkit_4096,
        "biosynfoni": preprocess_biosynfoni,
        "map4_4096": partial(preprocess_map4, radius=2, fp_size=4096),
        "maccs": preprocess_maccs,
    }

    mapper[args.type](args)


# ---------------------------------------------------------------------------
# Parallel helpers
# ---------------------------------------------------------------------------

def _chunksize(n_tasks, n_workers):
    return max(1, n_tasks // (n_workers * 8))


def _parallel_targets(smiles_series, worker_fn, worker_args, n_workers):
    """Deduplicate SMILES, compute FPs in parallel, expand back to full order."""
    unique_smiles = list(dict.fromkeys(smiles_series))  # deduplicated, insertion order
    tasks = [(s,) + worker_args for s in unique_smiles]
    cs = _chunksize(len(tasks), n_workers)
    with Pool(n_workers) as pool:
        unique_results = list(tqdm(
            pool.imap(worker_fn, tasks, chunksize=cs),
            total=len(tasks), desc="  targets (unique)",
        ))
    unique_fps = dict(zip(unique_smiles, unique_results))
    return [unique_fps[s] for s in smiles_series]


def _parallel_cands(cands_dict, worker_fn, worker_args, n_workers, desc):
    """Process each key's candidate list in parallel; return {key: result}."""
    tasks = [(k, v) + worker_args for k, v in cands_dict.items()]
    cs = _chunksize(len(tasks), n_workers)
    out = {}
    with Pool(n_workers) as pool:
        for key, val in tqdm(
            pool.imap_unordered(worker_fn, tasks, chunksize=cs),
            total=len(tasks), desc=desc,
        ):
            out[key] = val
    return out


# ---------------------------------------------------------------------------
# Module-level worker functions (must be picklable — no closures / lambdas)
# ---------------------------------------------------------------------------

def _w_inchi_single(args):
    (smile,) = args
    try:
        mol = Chem.MolFromSmiles(smile)
        return Chem.MolToInchiKey(mol) if mol else ""
    except Exception:
        return ""


def _w_inchi_key(args):
    key, smiles_list = args[0], args[1]
    inchis = []
    for smile in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smile)
            inchis.append(Chem.MolToInchiKey(mol) if mol else "")
        except Exception:
            inchis.append("")
    return key, np.array(inchis)


def _w_morgan_single(args):
    smile, radius, fp_size = args
    try:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)
        arr = np.zeros((fp_size,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(fp_size, dtype=np.uint8)


def _w_morgan_key(args):
    key, smiles_list, radius, fp_size = args[0], args[1], args[2], args[3]
    fps = []
    for smile in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smile)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)
            arr = np.zeros((fp_size,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        except Exception:
            fps.append(np.zeros(fp_size, dtype=np.uint8))
    return key, np.packbits(np.stack(fps).astype(bool), axis=None)


def _w_rdkit_single(args):
    smile, fp_size = args
    try:
        mol = Chem.MolFromSmiles(smile)
        fp = Chem.RDKFingerprint(mol, fpSize=fp_size)
        arr = np.zeros((fp_size,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(fp_size, dtype=np.uint8)


def _w_rdkit_key(args):
    key, smiles_list, fp_size = args[0], args[1], args[2]
    fps = []
    for smile in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smile)
            fp = Chem.RDKFingerprint(mol, fpSize=fp_size)
            arr = np.zeros((fp_size,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        except Exception:
            fps.append(np.zeros(fp_size, dtype=np.uint8))
    return key, np.packbits(np.stack(fps).astype(bool), axis=None)


def _w_maccs_single(args):
    (smile,) = args
    try:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        arr = np.zeros((167,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(167, dtype=np.uint8)


def _w_maccs_key(args):
    key, smiles_list = args[0], args[1]
    fps = []
    for smile in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smile)
            fp = AllChem.GetMACCSKeysFingerprint(mol)
            arr = np.zeros((167,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        except Exception:
            fps.append(np.zeros(167, dtype=np.uint8))
    return key, np.packbits(np.stack(fps).astype(bool), axis=None)


# ---------------------------------------------------------------------------
# Preprocess functions
# ---------------------------------------------------------------------------

def preprocess_inchi(args):
    data = pd.read_csv(args.MassSpecGymTSV, sep="\t")

    print("Preprocessing targets (InChI keys)")
    fps_target = _parallel_targets(data["smiles"], _w_inchi_single, (), args.n_workers)
    np.save(args.save_target_transforms, np.array(fps_target))

    for json_path, out_path, label in [
        (args.candidates_mass_json,    args.save_candidate_mass_transforms,    "mass cands"),
        (args.candidates_formula_json, args.save_candidate_formula_transforms, "formula cands"),
    ]:
        print(f"Preprocessing {label}")
        with open(json_path) as f:
            candidates = json.load(f)
        result = _parallel_cands(candidates, _w_inchi_key, (), args.n_workers, label)
        np.savez(out_path, **result)


def preprocess_morgan_4096(args, radius=2):
    fp_size = 4096
    data = pd.read_csv(args.MassSpecGymTSV, sep="\t")

    print(f"Preprocessing targets (Morgan r={radius})")
    fps_target = _parallel_targets(
        data["smiles"], _w_morgan_single, (radius, fp_size), args.n_workers
    )
    np.save(
        args.save_target_transforms,
        np.packbits(np.stack(fps_target).astype(bool), axis=None),
    )

    for json_path, out_path, label in [
        (args.candidates_mass_json,    args.save_candidate_mass_transforms,    "mass cands"),
        (args.candidates_formula_json, args.save_candidate_formula_transforms, "formula cands"),
    ]:
        print(f"Preprocessing {label}")
        with open(json_path) as f:
            candidates = json.load(f)
        result = _parallel_cands(candidates, _w_morgan_key, (radius, fp_size), args.n_workers, label)
        np.savez(out_path, **result)


def preprocess_rdkit_4096(args):
    fp_size = 4096
    data = pd.read_csv(args.MassSpecGymTSV, sep="\t")

    print("Preprocessing targets (RDKit FP)")
    fps_target = _parallel_targets(
        data["smiles"], _w_rdkit_single, (fp_size,), args.n_workers
    )
    np.save(
        args.save_target_transforms,
        np.packbits(np.stack(fps_target).astype(bool), axis=None),
    )

    for json_path, out_path, label in [
        (args.candidates_mass_json,    args.save_candidate_mass_transforms,    "mass cands"),
        (args.candidates_formula_json, args.save_candidate_formula_transforms, "formula cands"),
    ]:
        print(f"Preprocessing {label}")
        with open(json_path) as f:
            candidates = json.load(f)
        result = _parallel_cands(candidates, _w_rdkit_key, (fp_size,), args.n_workers, label)
        np.savez(out_path, **result)


def preprocess_maccs(args):
    data = pd.read_csv(args.MassSpecGymTSV, sep="\t")

    print("Preprocessing targets (MACCS)")
    fps_target = _parallel_targets(data["smiles"], _w_maccs_single, (), args.n_workers)
    np.save(
        args.save_target_transforms,
        np.packbits(np.stack(fps_target).astype(bool), axis=None),
    )

    for json_path, out_path, label in [
        (args.candidates_mass_json,    args.save_candidate_mass_transforms,    "mass cands"),
        (args.candidates_formula_json, args.save_candidate_formula_transforms, "formula cands"),
    ]:
        print(f"Preprocessing {label}")
        with open(json_path) as f:
            candidates = json.load(f)
        result = _parallel_cands(candidates, _w_maccs_key, (), args.n_workers, label)
        np.savez(out_path, **result)


# ---------------------------------------------------------------------------
# Optional / less-used types (special dependencies, kept sequential)
# ---------------------------------------------------------------------------

# from biosynfoni import Biosynfoni
def preprocess_biosynfoni(args):
    data = pd.read_csv(args.MassSpecGymTSV, sep="\t")

    def fingerprinter(smile):
        mol = Chem.MolFromSmiles(smile)
        fp = Biosynfoni(mol).fingerprint  # noqa: F821
        return (np.array(fp) > 0).astype(np.int32)

    print("Preprocessing targets (Biosynfoni)")
    fps_target = [fingerprinter(s) for s in tqdm(data["smiles"])]
    np.save(args.save_target_transforms, np.packbits(np.stack(fps_target).astype(bool), axis=None))

    for json_path, out_path, label in [
        (args.candidates_mass_json,    args.save_candidate_mass_transforms,    "mass cands"),
        (args.candidates_formula_json, args.save_candidate_formula_transforms, "formula cands"),
    ]:
        print(f"Preprocessing {label}")
        with open(json_path) as f:
            candidates = json.load(f)
        result = {}
        for k, v in tqdm(candidates.items(), total=len(candidates)):
            result[k] = np.packbits(
                np.stack([fingerprinter(s) for s in v]).astype(bool), axis=None
            )
        np.savez(out_path, **result)


# from map4 import MAP4Calculator
def preprocess_map4(args, radius=2, fp_size=4096):
    data = pd.read_csv(args.MassSpecGymTSV, sep="\t")
    map_calc = MAP4Calculator(dimensions=fp_size, radius=radius, is_folded=True)  # noqa: F821

    def fingerprinter(smile):
        mol = Chem.MolFromSmiles(smile)
        return map_calc.calculate(mol).astype(np.int32)

    print("Preprocessing targets (MAP4)")
    fps_target = [fingerprinter(s) for s in tqdm(data["smiles"])]
    np.save(args.save_target_transforms, np.packbits(np.stack(fps_target).astype(bool), axis=None))

    for json_path, out_path, label in [
        (args.candidates_mass_json,    args.save_candidate_mass_transforms,    "mass cands"),
        (args.candidates_formula_json, args.save_candidate_formula_transforms, "formula cands"),
    ]:
        print(f"Preprocessing {label}")
        with open(json_path) as f:
            candidates = json.load(f)
        result = {}
        for k, v in tqdm(candidates.items(), total=len(candidates)):
            result[k] = np.packbits(
                np.stack([fingerprinter(s) for s in v]).astype(bool), axis=None
            )
        np.savez(out_path, **result)


if __name__ == "__main__":
    main()
