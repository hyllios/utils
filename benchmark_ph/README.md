# uMLIP Phonon Calculation Tools

This repository contains tools for computing phonon properties using the uMLIP framework.

## Contents

- Example YAML file downloadable from the MDR database
- Calculation scripts for phonon property analysis
- Post-processing utilities for gathering data

## Scripts

### `calc_phonopy.py`
Used to compute phonons from uMLIP. Be careful to configure paths in `umlip_defs.json` according to your setup.

### `calc_ref.py`
Processes phonon calculations from PBE and PBEsol reference data.

### `postprocess.py`
Creates CSV files to gather results.

## Usage Example

To run Mattersim-v1 with the provided example:

```bash
python calc_phonopy.py --model=mattersim-v1 --ref=example/ --relax
```

Then run postprocess to obtain a csv:

```bash
python postprocess.py --yaml mattersim-v1/
```

## Setup

Ensure that your `umlip_defs.json` paths are correctly configured for your environment before running the calculations and that you have already installed the uMLIP you want to use.

## Data

The example YAML file is provided as a reference point and can be used to test the workflow.