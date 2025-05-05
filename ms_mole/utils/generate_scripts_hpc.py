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

    loss_f = {
        None : "None",
        "bce" : f'{p["bitwise_loss"]}-w-{p["bitwise_weighted"]}',
        "fl" : f'{p["bitwise_loss"]}-w-{p["bitwise_weighted"]}',
        "cossim" : f'{p["fpwise_loss"]}',
        "iou" : f'{p["fpwise_loss"]}',
        "bienc" : f'{p["rankwise_loss"]}-t-{p["rankwise_temp"]}',
        "cross" : f'{p["rankwise_loss"]}-t-{p["rankwise_temp"]}',
    }

    with open(file, "a") as f:
        f.write(generate_prefix(walltime))

        f.write(
f"""python {script_loc} \
{data_folder}MassSpecGym.tsv \
{data_folder} \
{logs_folder}{loss_f[p["bitwise_loss"]]}_{loss_f[p["fpwise_loss"]]}_{loss_f[p["rankwise_loss"]]}_{p["lambdas"][0]:.2f}_{p["lambdas"][1]:.2f}_{p["lambdas"][2]:.2f}_{p["lr"]}_{p["batch_size"]} \
--bonus_challenge {p["bonus_challenge"]} \
--batch_size {p["batch_size"]} \
--devices {p["devices"]} \
--precision 32-true \
--lr {p["lr"]} \
--bitwise_loss {p["bitwise_loss"]} \
--fpwise_loss {p["fpwise_loss"]} \
--rankwise_loss {p["rankwise_loss"]} \
--bitwise_lambd {p["lambdas"][0]} \
--fpwise_lambd {p["lambdas"][1]} \
--rankwise_lambd {p["lambdas"][2]} \
--bitwise_weighted {p["bitwise_weighted"]} \
--rankwise_temp {p["rankwise_temp"]} \
--checkpoint_path {p["ckpt_path"]} \
--freeze_checkpoint {p["freeze_ckpt"]} \
--rankwise_listwise {p["rankwise_listwise"]} \
--rnn_clfchain {p["rnncc"]} \
"""
        )


def determine_lambda_grid(bitwise_loss, fpwise_loss, rankwise_loss):
    losses = (bitwise_loss, fpwise_loss, rankwise_loss)
    losses_present = [ix for ix, l in enumerate(losses) if l != None]
    if len(losses_present) == 0:
        return [[0.0, 0.0, 0.0]]
    
    if len(losses_present) == 1:
        lambdas = [0.0, 0.0, 0.0]
        lambdas[losses_present[0]] = 1.0
        return [lambdas]
    
    elif len(losses_present) == 2:
        deltas = np.arange(1/6, .9, 1/6)
        t = np.zeros((len(deltas), 3))

        t[:, losses_present[0]] = deltas
        t[:, losses_present[1]] = 1-deltas
        return [list(tt) for tt in t]

    elif len(losses_present) == 3:
        deltas = np.arange(1/6, .9, 1/6)
        t = []
        for p in deltas:
            for q in deltas-p:
                if q > 0:
                    r = 1 - p - q
                    if r>0:
                        t.append([p, q, r])
        return t


def main():
    filefolder = str(sys.argv[1])
    script_loc = str(sys.argv[2])
    walltime = str(sys.argv[3])
    data_folder = str(sys.argv[4])
    logs_folder = str(sys.argv[5])
    bitwise_loss = str(sys.argv[6])
    fpwise_loss = str(sys.argv[7])
    rankwise_loss = str(sys.argv[8])
    weighted = str(sys.argv[9])
    device = int(sys.argv[10])
    n_models_to_tune = int(sys.argv[11])
    bonus_challenge = str(sys.argv[12])
    ckpt_path = str(sys.argv[13])
    freeze_ckpt = str(sys.argv[14])
    rankwise_listwise = str(sys.argv[15])
    rnncc = str(sys.argv[16])


    if bitwise_loss == "None":
        bitwise_loss = None
    if fpwise_loss == "None":
        fpwise_loss = None
    if rankwise_loss == "None":
        rankwise_loss = None

    if ckpt_path == "None":
        ckpt_path = None

    grid = {
        "devices" : [[device]],
        "lr" : [0.00005, 0.00007, 0.0001, 0.0003, 0.0005, 0.0007, 0.001],
        "batch_size" : [32, 64, 128],
        "bitwise_loss" : [bitwise_loss],
        "fpwise_loss" : [fpwise_loss],
        "rankwise_loss" : [rankwise_loss],
        "lambdas" : determine_lambda_grid(bitwise_loss, fpwise_loss, rankwise_loss),
        "bitwise_weighted" : [weighted == "True"],
        "rankwise_temp" : [0.1, 0.5, 1.0, 5, 10],
        "bonus_challenge" : [bonus_challenge],
        "ckpt_path" : [ckpt_path],
        "freeze_ckpt" : [freeze_ckpt == "True"],
        "rankwise_listwise" : [rankwise_listwise == "True"],
        "rnncc" : [rnncc == "True"],
    }

    param_list = list(ParameterSampler(grid, n_iter=n_models_to_tune))

    for ix, p_list in enumerate(param_list):
        filename = os.path.join(filefolder, str(ix)+".pbs")
        write_one_model_run(filename, script_loc, data_folder.rstrip("/")+"/", logs_folder.rstrip("/")+"/", walltime, p_list)


if __name__ == '__main__':
    main()