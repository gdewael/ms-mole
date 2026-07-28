"""
Generate PBS job scripts for enveda experiments.
Mirrors generate_scripts_hpc.py but points to the enveda dataset and
uses the enveda candidate JSONs via --train_cands_pth / --valtest_cands_pth.

Preprocessing (run once on HPC before generating scripts):
    # 1. Create enveda.tsv
    python ms_mole/utils/create_enveda_tsv.py \
        /hpc/data/enveda/enveda-180.mgf.gz \
        /hpc/data/enveda/split_enveda.tsv \
        /hpc/data/enveda/enveda.tsv \
        --max_peaks 128

    # 2. Precompute InChI keys
    python ms_mole/utils/precompute_fps_inchi_cands.py inchi \
        /hpc/data/enveda/enveda.tsv \
        /hpc/data/enveda/enveda_cands_mass.json \
        /hpc/data/enveda/enveda_cands_formula.json \
        /hpc/data/enveda/Inchis_targets.npy \
        /hpc/data/enveda/Inchis_masscands.npz \
        /hpc/data/enveda/Inchis_formulacands.npz

    # 3. Precompute fingerprints (repeat for other fp types if needed)
    python ms_mole/utils/precompute_fps_inchi_cands.py morgan_2_4096 \
        /hpc/data/enveda/enveda.tsv \
        /hpc/data/enveda/enveda_cands_mass.json \
        /hpc/data/enveda/enveda_cands_formula.json \
        /hpc/data/enveda/morgan_2_4096_targets.npy \
        /hpc/data/enveda/morgan_2_4096_masscands.npz \
        /hpc/data/enveda/morgan_2_4096_formulacands.npz

Usage:
    python generate_scripts_enveda.py \
        /path/to/output/scripts/ \
        /hpc/path/to/ms-mole/ms_mole/train_retriever.py \
        12:00:00 \
        /hpc/data/enveda/ \
        /hpc/logs/enveda/ \
        list_fp_cos \
        mass mass \
        256
"""

import sys
import os


def generate_prefix(walltime="12:00:00"):
    return f"""#!/bin/bash
#PBS -l nodes=1:ppn=8,gpus=1
#PBS -l walltime={walltime}

cd $PBS_O_WORKDIR

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

"""


def write_one_run(file, script_loc, data_folder, logs_folder, walltime, p):
    with open(file, "a") as f:
        f.write(generate_prefix(walltime))
        run_name = (
            f"{p['loss']}_{p['lr']}_"
            f"train{p['candidate_setting_train']}_eval{p['candidate_setting_eval']}"
        )
        f.write(
            f"""python {script_loc} \\
{data_folder}enveda.tsv \\
{data_folder} \\
{logs_folder}{run_name} \\
--train_cands_pth {data_folder}enveda_cands_{p['candidate_setting_train']}.json \\
--valtest_cands_pth {data_folder}enveda_cands_{p['candidate_setting_eval']}.json \\
--candidate_setting_train {p['candidate_setting_train']} \\
--candidate_setting_eval {p['candidate_setting_eval']} \\
--n_max_cands {p['n_max_cands']} \\
--lr {p['lr']} \\
--loss {p['loss']} \\
--temp {p['temp']} \\
--rankwise_listwise {p['rankwise_listwise']} \\
--batch_size {p['batch_size']} \\
"""
        )


def main():
    if len(sys.argv) < 9:
        print(__doc__)
        sys.exit(1)

    filefolder = sys.argv[1]
    script_loc = sys.argv[2]
    walltime = sys.argv[3]
    data_folder = sys.argv[4].rstrip("/") + "/"
    logs_folder = sys.argv[5].rstrip("/") + "/"
    loss = sys.argv[6]
    candidate_setting_train = sys.argv[7]
    candidate_setting_eval = sys.argv[8]
    n_max_cands = sys.argv[9] if len(sys.argv) > 9 else "256"
    temp = sys.argv[10] if len(sys.argv) > 10 else "0.5"
    rankwise_listwise = sys.argv[11] if len(sys.argv) > 11 else "True"
    batch_size = sys.argv[12] if len(sys.argv) > 12 else "32"

    os.makedirs(filefolder, exist_ok=True)

    c = 0
    for lr in [5e-5, 7e-5, 1e-4, 3e-4, 5e-4]:
        for _ in range(5):
            filename = os.path.join(filefolder, f"{loss}_{c}.pbs")
            write_one_run(
                filename,
                script_loc,
                data_folder,
                logs_folder,
                walltime,
                {
                    "loss": loss,
                    "lr": lr,
                    "candidate_setting_train": candidate_setting_train,
                    "candidate_setting_eval": candidate_setting_eval,
                    "n_max_cands": n_max_cands,
                    "temp": temp,
                    "rankwise_listwise": rankwise_listwise,
                    "batch_size": batch_size,
                },
            )
            c += 1


if __name__ == "__main__":
    main()
