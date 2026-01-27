#!/usr/bin/env bash

model="esen-oam"
python -u postprocess.py --json . --csv $model.csv --dim 3d
