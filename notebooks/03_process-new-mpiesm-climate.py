# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Process MPI-ESM E-driven data - except CO2
#
# We want to create annual global mean data from each variable. For CO2 we also want to calculate the annual global mean mass mixing ratio.

# %%
import os
from dotenv import load_dotenv
import iris
from iris.util import equalise_attributes
import iris.coord_categorisation as cat
import glob
import matplotlib.pyplot as pl
import warnings
import pandas as pd
from tqdm.auto import tqdm

# %%
load_dotenv(override=True)

# %%
datadir = os.getenv("DATADIR")
datadir

# %%
variables = ['tas', 'rsdt', 'rsut', 'rlut']
experiments = ['esm-ssp119', 'esm-ssp126', 'esm-ssp245', 'esm-ssp370', 'esm-ssp534-over']
ensemble_members = {
    'esm-ssp119': 10,
    'esm-ssp126': 10,
    'esm-ssp245': 30,
    'esm-ssp370': 10,
    'esm-ssp534-over': 10,
}

# %%
# hege does model / scenario then puts all variables and runs in the same folder

# %%
output = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for experiment in tqdm(experiments):
        output[experiment] = {}
        for iens in tqdm(range(ensemble_members[experiment]), leave=False):
            runid = f'r{iens+1}i1p1f1'
            tempoutput = {}
            for variable in variables:
                source_files = (glob.glob(os.path.join(datadir, "MPI-ESM1-2-LR", experiment, variable, f"*{runid}*")))
                cubes = iris.load(source_files)
                equalise_attributes(cubes);
                cube = cubes.concatenate_cube()
                area_weights = iris.analysis.cartography.area_weights(cube)
                cat.add_year(cube, 'time', name='year')
                cube_gm = cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN, weights=area_weights)
                cube_agm = cube_gm.aggregated_by('year', iris.analysis.MEAN)
                tempoutput[variable] = cube_agm.data
            output[experiment][runid] = pd.DataFrame(tempoutput, index=cube_agm.coord('year').points)
            os.makedirs(os.path.join('..', 'output', 'processed', 'MPI-ESM1-2-LR', experiment, 'climate'), exist_ok=True)
            output[experiment][runid].to_csv(os.path.join('..', 'output', 'processed', 'MPI-ESM1-2-LR', experiment, 'climate', f'{runid}.csv'))

# %%
