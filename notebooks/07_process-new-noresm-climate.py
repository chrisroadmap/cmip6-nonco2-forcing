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
experiments = ['esm-ssp119', 'esm-ssp245', 'esm-ssp534-over']

# I only want ensemble members that run to 2300
ensemble_members = {
    'esm-ssp119': 3,
    'esm-ssp245': 3,
    'esm-ssp534-over': 3,
}

# %%
# hege does model / scenario then puts all variables and runs in the same folder
# we have a small issue here in that we switch from f1 to f2 in 2100 and r1i1f1f1
# is extended (we want to use f2)
# however, no subtraction / drift correction is taking place here... so be naive

debug = False

# %%
output = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for experiment in tqdm(experiments, disable=debug):
        output[experiment] = {}
        for iens in tqdm(range(ensemble_members[experiment]), leave=False, disable=debug):
            for ifnum in tqdm(range(1, 3), leave=False, disable=debug):
                runid = f'r{iens+1}i1p1f{ifnum}'
                tempoutput = {}
                for variable in variables:
                    if debug:
                        print(experiment, runid, variable)
                    source_files = (glob.glob(os.path.join(datadir, "NorESM2-LM", experiment, variable, f"*{runid}*")))
                    cubes = iris.load(source_files)
                    equalise_attributes(cubes);
                    cube = cubes.concatenate_cube()
                    area_weights = iris.analysis.cartography.area_weights(cube)
                    cat.add_year(cube, 'time', name='year')
                    cube_gm = cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN, weights=area_weights)
                    cube_agm = cube_gm.aggregated_by('year', iris.analysis.MEAN)
                    tempoutput[variable] = cube_agm.data
                output[experiment][runid] = pd.DataFrame(tempoutput, index=cube_agm.coord('year').points)
                os.makedirs(os.path.join('..', 'output', 'processed', 'NorESM2-LM', experiment, 'climate'), exist_ok=True)
                output[experiment][runid].to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', experiment, 'climate', f'{runid}.csv'))

# %%
