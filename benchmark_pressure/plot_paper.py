#%%
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from scipy.stats import gaussian_kde

#%%
path_csv = "./data"

reference = "pbe"
all_datasets = [
    "m3gnet",
    "mace-mpa-0", 
    "sevenn-mf-ompa",
    "dpa3-v1-openlam",
    "grace-2l-oam",
    "orb-v3-conservative-inf",
    "mattersim-v1",
    "esen-oam",
    "mattersim-ap-ft-0",
    "esen-ap-ft-0",
]
renamed_datasets = {
    "m3gnet": "M3GNet",
    "mace-mpa-0": "MACE-MPA-0",
    "sevenn-mf-ompa": "SevenNet-MF-OMPA",
    "dpa3-v1-openlam": "DPA3-v1-OpenLam",
    "grace-2l-oam": "GRACE-2L-OAM",
    "orb-v3-conservative-inf": "ORB-v3-Conservative-Inf",
    "mattersim-v1": "MatterSim-v1",
    "esen-oam": "eSEN-30M-OAM",
    "mattersim-ap-ft-0": "MatterSim-ap-ft-0",
    "esen-ap-ft-0": "eSEN-ap-ft-0",
}
list_pressure = ["P000", "P025", "P050", "P075", "P100", "P125", "P150"]
pressure_values = [0, 25, 50, 75, 100, 125, 150] 
df_dict = {}

"""Load and process data for all pressures"""

for pressure in tqdm(list_pressure, desc="Loading pressure data"):
    pressure_path = os.path.join(path_csv, pressure)
    if not os.path.exists(pressure_path):
        continue
    print(f"Processing pressure: {pressure}")
    # load reference pbe
    ref_file = f"{reference}.csv"
    if not os.path.exists(os.path.join(pressure_path, ref_file)):
        continue

    df = pd.read_csv(os.path.join(pressure_path, ref_file))
    df = df.rename(columns={
        "volume": f"volume_per_atom_{reference}", 
        "energy_per_atom": f"energy_per_atom_{reference}"
    })
    
    # get stats on all datasets of umlip
    for dataset in all_datasets:
        dataset_file = os.path.join(pressure_path, f"{dataset}.csv")
        if not os.path.exists(dataset_file):
            continue
        renamed_dataset = renamed_datasets[dataset]
        mlip = pd.read_csv(dataset_file)
        mlip = mlip.rename(columns={
            "volume": f"dif_volume_per_atom_{renamed_dataset}", 
            "energy_per_atom": f"dif_energy_per_atom_{renamed_dataset}"
        })
        
        df = df.merge(mlip, on="mat_id", how="left")
        
        # get the difference, the volume in the csv is in per atom
        df[f"dif_volume_per_atom_{renamed_dataset}"] -= df[f"volume_per_atom_{reference}"]
        df[f"dif_energy_per_atom_{renamed_dataset}"] -= df[f"energy_per_atom_{reference}"]
    # Store the processed DataFrame
    df_dict[pressure] = df

#%%
"""filter nan and infinite and output how many, from which model"""
for pressure, df in df_dict.items():
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_count = df.isna().sum().sum()
    print(f"Pressure: {pressure}, NaN count: {nan_count}")
    for col in df.columns:
        if df[col].isna().any():
            print(f"  Model: {col}, NaN count: {df[col].isna().sum()}")

#%%
"""Print error statistics for each dataset"""
statistics = {}
for pressure, df in df_dict.items():
    statistics[pressure] = {}
    for model_name in renamed_datasets.values():
        for v in ("dif_volume_per_atom", "dif_energy_per_atom"):
            c = f"{v}_{model_name}"
            if c in df.columns:
                statistics[pressure][c] = {
                    "count": df[c].count(),
                    "ME": df[c].mean(),
                    "MAE": df[c].abs().mean(),
                    "RMSE": np.sqrt((df[c]**2).mean()),
                    "MAPE": (df[c]/df[f"{v.split('dif_')[1]}_{reference}"]).abs().mean()*100
                }
            else:
                print(f'error: {c} not found in DataFrame')
## Format the output per model then pressure

#%%
## Plot    
prefix = "dif_"
error_limits ={
    'volume_per_atom': (-5, 5),
    'energy_per_atom': (-2, 2)
}
df_list_filtered = []
for press, df in df_dict.items():
    #remove ref
    error_cols = [c for c in df.columns if c.startswith(prefix)]    
    subdf = df[["mat_id"] + error_cols].copy()
    
    # Rename columns to model names
    for col in error_cols:
        model_name = col.replace(prefix, "")
        subdf[model_name] = subdf[col]
    
    # Drop original columns
    subdf.drop(columns=error_cols, inplace=True)
    
    # Melt to long format
    subdf_melted = subdf.melt(
        id_vars="mat_id", 
        var_name="model", 
        value_name="error"
    )
    
    # Add pressure info
    subdf_melted["pressure"] = press
    subdf_melted["pressure_value"] = int(press[1:])  # Extract numeric value
    
    df_list_filtered.append(subdf_melted)

# Combine all data
df_all = pd.concat(df_list_filtered, ignore_index=True)

# Filter outliers
# df_all = df_all[(df_all["error"] >= error_limits[0]) & 
                # (df_all["error"] <= error_limits[1])]


#%%
def setup_publication_style(black_style=False):
    """Setup consistent plotting style for publication"""
    rc = {
        'figure.figsize': (15, 12),
        'font.size': 24,
        'xtick.labelsize': 24, 
        'ytick.labelsize': 24, 
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'grid.color': '#cccccc',
        'grid.linewidth': 0.5,
    }
    sns.set_style("whitegrid")
    
    plt.rcParams.update(rc)
    sns.set_theme(rc=rc)

# Setup style
setup_publication_style(black_style=False)

# Prepare data
error_limits ={
    'volume_per_atom': (-8, 8), #20 times our range of plot
    'energy_per_atom': (-4, 4)
} 

# Set up the plot
models = [a.split('per_atom_')[1] for a in df_dict['P000'].columns if 'dif' in a and 'volume' in a]
n_models = len(models)

n_cols = 2
n_rows = (n_models + n_cols - 1) // n_cols





pressure_values = [int(press.split('P')[1]) for press in df_dict.keys()]
pressure_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(pressure_values)))
pressure_color_map = dict(zip(pressure_values, pressure_colors))

for error_type in ["dif_volume_per_atom", "dif_energy_per_atom"]:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), 
                        sharex=True, sharey=False)
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    axes = axes.flatten() if n_models > 1 else [axes]
    for i, model in enumerate(models):
        ax = axes[i]
        # space between vertical pressure curves
        y_spacing = 0.3
        max_height = 0.25

        for j, pressure in enumerate(df_dict.keys()):
            pressure_data = df_dict[pressure][error_type + f"_{model}"]

            # remove nan and inf
            original_count = len(pressure_data)
            pressure_data = pressure_data.replace([np.inf, -np.inf], np.nan)
            pressure_data = pressure_data.dropna()
            nan_inf_count = original_count - len(pressure_data)
            print(f"{error_type}_{model} at {pressure}: Removed {nan_inf_count} NaN/inf values")

            # remove big outliers

            error_key = error_type.replace('dif_', '')
            lower_limit, upper_limit = error_limits[error_key]
            
            before_outlier_removal = len(pressure_data)
            pressure_data = pressure_data[(pressure_data >= lower_limit) & (pressure_data <= upper_limit)]
            outlier_count = before_outlier_removal - len(pressure_data)
            print(f"{error_type}_{model} at {pressure}: Removed {outlier_count} outliers outside [{lower_limit}, {upper_limit}]")

            # do kde
            errors = pressure_data
            kde = gaussian_kde(errors, bw_method='scott')
            
            # create smooth x values with better range
            if error_type == "dif_volume_per_atom":
                x_smooth = np.linspace(-0.5, 0.5, 300)
            else:
                x_smooth = np.linspace(-0.25, 0.25, 300)
            y_smooth = kde(x_smooth)
            
            # normalize and offset
            y_smooth = y_smooth / y_smooth.max() * max_height
            y_offset = j * y_spacing
            y_smooth += y_offset
            
            # color for this pressure
            color = pressure_color_map[int(pressure.split('P')[1])]

            ax.fill_between(x_smooth, y_offset, y_smooth, 
                            alpha=0.8, color=color, 
                            label=f'{pressure} GPa')
            
            #  outline for better definition
            ax.plot(x_smooth, y_smooth, color=color, 
                    linewidth=1.2, alpha=0.9)
            
            # baseline
            ax.axhline(y=y_offset, color='gray', linestyle='-', 
                        alpha=0.3, linewidth=0.8)
        
        if error_type == "dif_volume_per_atom":
            ax.set_xlim(-0.5, 0.5)
        else:
            ax.set_xlim(-0.25, 0.25)
            p=1
        ax.set_ylim(-0.1, len(pressure_values) * y_spacing + 0.1)
        ax.set_title(model, fontsize=24, pad=10)
        
        # Remove y-axis ticks and labels
        ax.set_yticks([])
        ax.set_ylabel("")
        
        # vertical line at x=0
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.5, axis='x')

    # Set common x-label
    unit = r"A$^3/atom$" if error_type == "dif_volume_per_atom" else r"eV/atom"
    label_title = "Volume" if error_type == "dif_volume_per_atom" else "Energy"
    fig.text(0.5, 0.035, r"$\Delta$ {error_type} [{unit}]".format(error_type=label_title,unit=unit), 
            ha='center', va='bottom', fontsize=24)
    for ax in axes[:n_models]:
        sns.despine(ax=ax, left=True, bottom=False)

    plt.tight_layout()
    plt.legend(
        handles=[plt.Line2D([0], [0], color=pressure_color_map[p], lw=4) 
                    for p in pressure_values],
        labels=[f"{p} GPa" for p in pressure_values],
        title="Pressure", bbox_to_anchor=(1.018, -0.3), ncol=7,
        handletextpad=0.1,
        fontsize=19, title_fontsize=20
    )

    plt.subplots_adjust(bottom=0.08)

    # save png or pdf
    # plt.savefig(f".//publication_{error_type}_error_pressure.png", 
            # bbox_inches='tight', facecolor='white')
    plt.savefig(f".//publication_{error_type}_error_pressure.pdf", 
            bbox_inches='tight', facecolor='white')
    plt.show()
