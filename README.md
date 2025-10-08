# Source code for "Small molecule retrieval from tandem mass spectrometry: What are we optimizing for?"

Internally, all code is structured as a pip-installable package named ms-mole (Mass Spectral MOlecule Loss function Evaluation).

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
- Run `ms_mole/utils/precompute_fps_inchi_cands.py` (check script for command line arguments)
- (Optionally, run any of the precompute_hard_negative_cands scripts in the same folder to perform the appendix experiments with hard negative candidate sets).
- The generated outputs make up auxiliary files used in the main training script: `ms_mole/train_retriever.py`. Run `python ms_mole/train_retriever.py --help` to see how to use the training script.
