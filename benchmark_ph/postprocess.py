#%%
import os, sys, glob
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_params():
    from argparse import ArgumentParser

    parent_parser = ArgumentParser(usage="Extract properties to benchmark")
    parent_parser.add_argument('--yaml', type = str, default = "",
        help='Yaml files')
    parent_parser.add_argument('--csv', type = str, default = "benchmark.csv",
        help='Destination csv file')
    parent_parser.add_argument('--restart', action='store_true')

    return parent_parser.parse_args()


if __name__ == '__main__':
    params = get_params()

    if params.restart:
        properties = pd.read_csv(params.csv).to_dict('records')
    else:
        properties = []

    ymls = glob.glob(f"{params.yaml}*.yaml")
    for yml_fname in tqdm(ymls):
        mp_id = os.path.splitext(os.path.basename(yml_fname))[0]
        if any(p["mp_id"] == mp_id for p in properties):
            continue

        with open(yml_fname) as f:
            yml = yaml.unsafe_load(f)

        prop = {}
        prop["mp_id"] = mp_id
        prop["nsites"] = yml["nsites"]

        prop["energy_pa"] = None if yml['energy'] is None else yml['energy']/yml['nsites']
        prop["volume_pa"] = yml['volume']/yml['nsites']

        ## entropy heat_capacity free_e at 300K
        index_300k = yml['temperatures'].index(300.0)
        prop["entropy"] = yml['entropy'][index_300k]
        prop["heat_capacity"] = yml['heat_capacity'][index_300k]
        prop["free_energy"] = yml['free_e'][index_300k]

        ## from THz to K
        phonon_freq = np.array(yml['phonon_freq'])*47.992

        ## Calculate max and avg frequencies
        prop["max_freq"] = np.max(phonon_freq)
        prop["avg_freq"] = np.average(phonon_freq)

        unstable = np.any(phonon_freq[0][:3] < -35.0)
        unstable = unstable or np.any(phonon_freq[0][3:] < 0.0)
        unstable = unstable or np.any(phonon_freq[1:][:] < 0.0)

        prop["stable"] = not unstable

        properties.append(prop)

    df = pd.DataFrame(properties)
    df.set_index('mp_id', inplace=True)
    df.to_csv(params.csv)
