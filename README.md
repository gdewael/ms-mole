# Source code for "Small molecule retrieval from tandem mass spectrometry: What are we optimizing for?"

Internally, all code is structured as a pip-installable package named `ms-mole` (Mass Spectral MOlecule Loss function Evaluation).

## Install

```bash
conda create --name "msmole" python==3.11
conda activate msmole
pip install -e ./ms-mole/
```

## Code structure

All code is packaged under `ms_mole`.

- `data.py`: contains `RetrievalDataset_PrecompFPandInchi`, a subclass of [the original `RetrievalDataset`](https://github.com/pluskal-lab/MassSpecGym/blob/f525a5e55a39ec4caa4f1a51e64acd046713179e/massspecgym/data/datasets.py#L147) in MassSpecGym. It retains the same functionality as the original, but is somewhat faster because in implementation because it precomputes some stuff so it doesn't need to be computed in the Dataset object. In addition it contains `MsMoleMassSpecDataModule`, a subclass of `pl.LightningDataModule` used to perform data loading in the main script.
- `loss.py`: definitions for all tested fingerprint prediction loss functions.
- `models.py`: contains `FingerprintPredicter`, a subclass of [the original `RetrievalMassSpecGymModel`](https://github.com/pluskal-lab/MassSpecGym/blob/f525a5e55a39ec4caa4f1a51e64acd046713179e/massspecgym/models/retrieval/base.py#L14) in MassSpecGym. This base class contains many predefined hooks to compute relevant retrieval metrics. `FingerprintPredicter` retains the same base functionality, but streamlined so that during validation checks, three types of metrics are logged: (1) loss, (2) retrieval (using different sim funcs), and (3) fingerprint accuracy (in terms of average Tanimoto sim)
- `train_retriever.py` main training script.


## Reproduction steps

- First, set up environment (see above).
- Download all MassSpecGym data [here](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/tree/main)
- Run any of the following to generate candidate and target fingerprints:
```bash
python ms-mole/ms_mole/utils/precompute_fps_inchi_cands.py inchi data/MassSpecGym.tsv data/MassSpecGym_retrieval_candidates_mass.json data/MassSpecGym_retrieval_candidates_formula.json data/Inchis_targets.npy data/Inchis_masscands.npz data/Inchis_formulacands.npz --n_workers 16

python ms-mole/ms_mole/utils/precompute_fps_inchi_cands.py morgan_2_4096 data/MassSpecGym.tsv data/MassSpecGym_retrieval_candidates_mass.json data/MassSpecGym_retrieval_candidates_formula.json data/morgan_2_4096_targets.npy data/morgan_2_4096_masscands.npz data/morgan_2_4096_formulacands.npz --n_workers 16
python ms-mole/ms_mole/utils/precompute_fps_inchi_cands.py morgan_4_4096 data/MassSpecGym.tsv data/MassSpecGym_retrieval_candidates_mass.json data/MassSpecGym_retrieval_candidates_formula.json data/morgan_4_4096_targets.npy data/morgan_4_4096_masscands.npz data/morgan_4_4096_formulacands.npz --n_workers 16
python ms-mole/ms_mole/utils/precompute_fps_inchi_cands.py morgan_6_4096 data/MassSpecGym.tsv data/MassSpecGym_retrieval_candidates_mass.json data/MassSpecGym_retrieval_candidates_formula.json data/morgan_6_4096_targets.npy data/morgan_6_4096_masscands.npz data/morgan_6_4096_formulacands.npz --n_workers 16
python ms-mole/ms_mole/utils/precompute_fps_inchi_cands.py morgan_8_4096 data/MassSpecGym.tsv data/MassSpecGym_retrieval_candidates_mass.json data/MassSpecGym_retrieval_candidates_formula.json data/morgan_8_4096_targets.npy data/morgan_8_4096_masscands.npz data/morgan_8_4096_formulacands.npz --n_workers 16

python ms-mole/ms_mole/utils/precompute_fps_inchi_cands.py rdkit_4096 data/MassSpecGym.tsv data/MassSpecGym_retrieval_candidates_mass.json data/MassSpecGym_retrieval_candidates_formula.json data/rdkit_4096_targets.npy data/rdkit_4096_masscands.npz data/rdkit_4096_formulacands.npz --n_workers 16

python ms-mole/ms_mole/utils/precompute_fps_inchi_cands.py biosynfoni data/MassSpecGym.tsv data/MassSpecGym_retrieval_candidates_mass.json data/MassSpecGym_retrieval_candidates_formula.json data/biosynfoni_targets.npy data/biosynfoni_masscands.npz data/biosynfoni_formulacands.npz
python ms-mole/ms_mole/utils/precompute_fps_inchi_cands.py maccs data/MassSpecGym.tsv data/MassSpecGym_retrieval_candidates_mass.json data/MassSpecGym_retrieval_candidates_formula.json data/maccs_targets.npy data/maccs_masscands.npz data/maccs_formulacands.npz --n_workers 16
python ms-mole/ms_mole/utils/precompute_fps_inchi_cands.py map4_4096 data/MassSpecGym.tsv data/MassSpecGym_retrieval_candidates_mass.json data/MassSpecGym_retrieval_candidates_formula.json data/map4_4096_targets.npy data/map4_4096_masscands.npz data/map4_4096_formulacands.npz
```

- (Optionally, run any of the precompute_hard_negative_cands scripts in the same folder to perform the appendix experiments with hard negative candidate sets).
- The generated outputs make up auxiliary files used in the main training script: `ms_mole/train_retriever.py`. Run `python ms_mole/train_retriever.py --help` to see how to use the training script.


## Enveda-180

The candidate files (`enveda_cands_mass.json`, `enveda_cands_formula.json`) and the MGF (`enveda-180.mgf.gz`) are assumed to live in `data/enveda/`. The train/val/test split file is versioned in the repo at `ms_mole/utils/split_enveda.tsv`.

**Step 1 — Create enveda TSV**

Converts the MGF to the MassSpecGym TSV format. Peaks are filtered to the top 128 by intensity (enveda spectra can have hundreds of peaks). Folds are read from the pre-defined split file.

```bash
python ms-mole/ms_mole/utils/create_enveda_tsv.py \
    data/enveda/enveda-180.mgf.gz \
    ms-mole/ms_mole/utils/split_enveda.tsv \
    data/enveda/enveda.tsv \
    --max_peaks 128
```

**Step 2 — Precompute InChI keys**

```bash
python ms-mole/ms_mole/utils/precompute_fps_inchi_cands.py inchi \
    data/enveda/enveda.tsv \
    data/enveda/enveda_cands_mass.json \
    data/enveda/enveda_cands_formula.json \
    data/enveda/Inchis_targets.npy \
    data/enveda/Inchis_masscands.npz \
    data/enveda/Inchis_formulacands.npz \
    --n_workers 16
```

**Step 3 — Precompute fingerprints**

```bash
python ms-mole/ms_mole/utils/precompute_fps_inchi_cands.py morgan_2_4096 \
    data/enveda/enveda.tsv \
    data/enveda/enveda_cands_mass.json \
    data/enveda/enveda_cands_formula.json \
    data/enveda/morgan_2_4096_targets.npy \
    data/enveda/morgan_2_4096_masscands.npz \
    data/enveda/morgan_2_4096_formulacands.npz \
    --n_workers 16
```

Repeat step 3 for other fingerprint types if needed (substitute `morgan_2_4096` with e.g. `morgan_4_4096`, `rdkit_4096`).

**Step 4 — Generate PBS job scripts**

```bash
python ms-mole/ms_mole/utils/generate_scripts_enveda.py \
    scripts/enveda_mass/ \
    /kyukon/home/gent/431/vsc43136/data_vo/msms/ms-mole/ms_mole/train_retriever.py \
    12:00:00 \
    /data/gent/vo/000/gvo00048/vsc43136/msms/data/enveda/ \
    /data/gent/vo/000/gvo00048/vsc43136/msms/logs/enveda/ \
    list_fp_cos mass mass 256
```

**Notes**
- Both `enveda_cands_mass.json` and `enveda_cands_formula.json` already cap at 256 candidates per query — no truncation needed.
- `train_retriever.py` now accepts `--train_cands_pth` and `--valtest_cands_pth` to pass the enveda candidate JSONs directly, bypassing the hardcoded `MassSpecGym_retrieval_candidates_*.json` filename pattern. Existing MassSpecGym scripts are unaffected.

## DreaMS

[DreaMS](https://github.com/pluskal-lab/DreaMS) is a pretrained transformer spectrum encoder that produces 1024-dimensional embeddings. The integration here is **probe-only**: embeddings are precomputed once (frozen), then a small MLP head is trained on top — identical architecture and loss functions to `train_retriever.py`.

No existing files are modified; `train_dreams_retriever.py` and `precompute_dreams_embeddings.py` are fully standalone.

**Install DreaMS**

```bash
pip install "dreams @ git+https://github.com/pluskal-lab/DreaMS.git"
```

**Step 1 — Precompute embeddings** (run once per dataset, GPU recommended)

```bash
# MassSpecGym
python ms-mole/ms_mole/utils/precompute_dreams_embeddings.py \
    data/MassSpecGym.tsv \
    data/dreams_embeddings.npy \
    --batch_size 256

# Enveda (requires enveda.tsv from the Enveda section above)
python ms-mole/ms_mole/utils/precompute_dreams_embeddings.py \
    data/enveda/enveda.tsv \
    data/enveda/dreams_embeddings.npy \
    --batch_size 256
```

Output: `(N_rows, 1024)` float32 array, row-aligned with the TSV.

**Step 2 — Train**

```bash
# MassSpecGym example
python ms-mole/ms_mole/train_dreams_retriever.py \
    data/MassSpecGym.tsv \
    data/ \
    logs/dreams_massspecgym/list_fp_cos_0.0001 \
    --dreams_embs_pth data/dreams_embeddings.npy \
    --candidate_setting_train mass \
    --candidate_setting_eval mass \
    --loss list_fp_cos --temp 0.5 --lr 0.0001 --n_max_cands 256

# Enveda example
python ms-mole/ms_mole/train_dreams_retriever.py \
    data/enveda/enveda.tsv \
    data/enveda/ \
    logs/dreams_enveda/list_fp_cos_0.0001 \
    --dreams_embs_pth data/enveda/dreams_embeddings.npy \
    --train_cands_pth data/enveda/enveda_cands_mass.json \
    --valtest_cands_pth data/enveda/enveda_cands_mass.json \
    --candidate_setting_train mass \
    --candidate_setting_eval mass \
    --loss list_fp_cos --temp 0.5 --lr 0.0001 --n_max_cands 256
```
