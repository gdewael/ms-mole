# ms-mole
MOlecular retrieval Loss function Evaluation

## Install

```bash
conda create --name "gaetan_msmole" python==3.11
conda activate gaetan_msmole
pip install -e ./ms-mole/
```


## Code structure

All code is packaged under `ms_mole`.

- `data.py`: contains `RetrievalDataset_PrecompFPandInchi`, a subclass of [the original `RetrievalDataset`](https://github.com/pluskal-lab/MassSpecGym/blob/f525a5e55a39ec4caa4f1a51e64acd046713179e/massspecgym/data/datasets.py#L147) in MassSpecGym. It retains the same functionality as the original, but is somewhat faster because in implementation because it precomputes some stuff so it doesn't need to be computed in the Dataset object.
- `loss.py`: all tested fingerprint prediction loss functions: (1) Bitwise BCE, (2) Bitwise FL, (3) Vectorwise CosSim Loss, (4) Vectorwise IoU Loss, (5) Rankwise Bi-encoder, (6) Rankwise Cross-encoder. A central interface to all these loss functions, is `ms_mole.loss.FPLoss`, where losses can be arbitrarily combined.
- `models.py`: contains `FingerprintPredicter`, a subclass of [the original `RetrievalMassSpecGymModel`](https://github.com/pluskal-lab/MassSpecGym/blob/f525a5e55a39ec4caa4f1a51e64acd046713179e/massspecgym/models/retrieval/base.py#L14) in MassSpecGym. This base class contains many predefined hooks to compute relevant retrieval metrics. `FingerprintPredicter` retains the same base functionality, but streamlined so that during validation checks, three types of metrics are logged: (1) loss, (2) retrieval (using different sim funcs), and (3) fingerprint accuracy (in terms of average Tanimoto sim)
- `train_retriever.py` example training script

Under `./notebooks/example.ipynb` is an example of how to play around with the modules.

## Loss functions for MS/MS-based molecule retrieval

Visual representation of all loss functions:

<img src="./assets/drawing.svg" align="center" width="650">
