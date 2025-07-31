from sklearn.model_selection import ParameterSampler
import sys
import os
import numpy as np


def generate_prefix(walltime="12:00:00"):

    prefix = f"""#!/bin/bash
#PBS -l nodes=1:ppn=8,gpus=1
#PBS -l walltime={walltime}

cd $PBS_O_WORKDIR

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

"""
    return prefix


def write_one_model_run(file, script_loc, data_folder, logs_folder, walltime, p):
    with open(file, "a") as f:
        f.write(generate_prefix(walltime))

        f.write(
            f"""python {script_loc} \
{data_folder}MassSpecGym.tsv \
{data_folder} \
{logs_folder}{p["loss"]}_{p["lr"]} \
--candidate_setting_train {p["candidate_setting_train"]} \
--candidate_setting_eval {p["candidate_setting_eval"]} \
--n_max_cands {p["n_max_cands"]} \
--lr {p["lr"]} \
--loss {p["loss"]} \
--fl_gamma {p["fl_gamma"]} \
--temp {p["temp"]} \
--rankwise_listwise {p["rankwise_listwise"]} \
"""
        )


def main():
    filefolder = str(sys.argv[1])
    script_loc = str(sys.argv[2])
    walltime = str(sys.argv[3])
    data_folder = str(sys.argv[4])
    logs_folder = str(sys.argv[5])
    loss = str(sys.argv[6])
    candidate_setting_train = str(sys.argv[7])
    candidate_setting_eval = str(sys.argv[8])
    n_max_cands = str(sys.argv[9])
    fl_gamma = str(sys.argv[10])
    temp = str(sys.argv[11])
    rankwise_listwise = str(sys.argv[12])

    parameter_dict = {
        "loss": loss,
        "candidate_setting_train": candidate_setting_train,
        "candidate_setting_eval": candidate_setting_eval,
        "n_max_cands": n_max_cands,
        "fl_gamma": fl_gamma,
        "temp": temp,
        "rankwise_listwise": rankwise_listwise,
    }
    c = 0
    for lr in [5e-5, 7e-5, 1e-4, 3e-4, 5e-4]:
        for _ in range(5):
            filename = os.path.join(filefolder, loss + "_" + str(c) + ".pbs")
            parameter_dict["lr"] = lr
            write_one_model_run(
                filename,
                script_loc,
                data_folder.rstrip("/") + "/",
                logs_folder.rstrip("/") + "/",
                walltime,
                parameter_dict,
            )
            c += 1


if __name__ == "__main__":
    main()
