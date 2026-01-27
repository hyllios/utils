#!/usr/bin/env bash

#SBATCH -p compute_GPU
#SBATCH --gres=gpu:rtx_4060_ti:1


model="esen-oam"
pressure="75"
python -u rerun_json.py 3d $model input_data.json $model.json $pressure #> $model.log 2>&1
