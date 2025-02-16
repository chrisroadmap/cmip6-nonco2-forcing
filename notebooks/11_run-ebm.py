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

# %%
import pandas as pd
from fair.energy_balance_model import EnergyBalanceModel
import matplotlib.pyplot as pl
import numpy as np

# %%
ensemble_members = {
    'MPI-ESM1-2-LR': {
        'ssp119': 10,
        'ssp245': 30,
    },
}

# %%
forcing = {}
for model in ['MPI-ESM1-2-LR']:
    forcing[model] = {}
    for expt in ['ssp119', 'ssp245']:
        forcing[model][expt] = {}
        for ens in range(1, ensemble_members[model][expt]+1):
            forcing[model][expt][ens] = pd.read_csv(f'../data/transient_forcing_estimates/MPI-ESM1-2-LR/{expt}/MPI-ESM1-2-LR_historicaland{expt}_r{ens}i1p1f1_ERF.csv', index_col=0)

# %%
df_ebm = pd.read_csv('../data/4xCO2_cummins_ebm3_cmip6.csv')

# %%
mpi_ebm = df_ebm[df_ebm['model']=='MPI-ESM1-2-LR']

# %%
mpi_ebm

# %%
temp = {}
for model in ['MPI-ESM1-2-LR']:
    temp[model] = {}
    for expt in ['ssp119', 'ssp245']:
        temp[model][expt] = {}
        for ens in range(1, ensemble_members[model][expt]+1):
            ebm3 = EnergyBalanceModel(
                ocean_heat_capacity=[5.152185, 10.663705, 94.835548],
                ocean_heat_transfer=[1.401413, 1.947001, 1.074361],
                deep_ocean_efficacy=1.261039,
                gamma_autocorrelation=2.402813,
                sigma_xi=0.714644,
                sigma_eta=0.46604,
                forcing_4co2=8.931733,
                stochastic_run=True,
                seed=ens+1
            )
            ebm3.add_forcing(forcing = forcing[model][expt][ens]['ERF'].values, timestep=1)
            ebm3.run()
            temp[model][expt][ens] = ebm3.temperature[:, 0]

# %%
for ens in range(1, ensemble_members['MPI-ESM1-2-LR']['ssp245']+1):
    pl.plot(np.arange(1850, 2101), temp['MPI-ESM1-2-LR']['ssp245'][ens])

# %%
for ens in range(1, ensemble_members['MPI-ESM1-2-LR']['ssp119']+1):
    pl.plot(np.arange(1850, 2101), temp['MPI-ESM1-2-LR']['ssp119'][ens])

# %%
