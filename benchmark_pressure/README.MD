# Pressure benchmarl tools

This repository contains tools for doing the benchmark under pressure

## Contents

- Results from our uMLIP calculation in data, on a test set
- Example of calculation using our scripts
- Post-processing utilities to plot the data

## Scripts

### `rerun_json.py`
Used to compute the relaxation with uMLIP and pressure. Be careful to configure paths in `umlip_defs.json` according to your setup.
An example to use can be find with run_umlip.sh. One should be carefull to activate the environment needed with the package to use the specific uMLIP installed.

### `postprocess.py`
Creates CSV files to gather results.

### `plot_paper.py`
Script to plot the same figure than in the paper using data folder.

## Usage Example

Example for each scripts can be found in the associated .sh job.

Ensure that your `umlip_defs.json` paths are correctly configured for your environment before running the calculations and that you have already installed the uMLIP you want to use.

## Data

We provide an example of input with input_data.json, containing 10 data points.

In data folder, one can find the csv results with all data we used in our paper.