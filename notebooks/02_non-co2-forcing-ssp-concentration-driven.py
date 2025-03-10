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
# # Calculate transient non-CO2 forcing in each model/scenario
#
# We take the CO2 ERF from the 1pct runs and subtract this from Hege's ERF calculation to get the non-CO2 ERF

# %%
import os
import glob
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# %%
x0=284.316999854786

def myhre(x, d):
    return d * np.log(x/x0)
    

# def etminan(x, a, b, d):
#     return (a * (x - x0)**2 + b * (x - x0) + d) * np.log(x / x0)

# %% [markdown]
# ## CO2 ERF as a function of concentration in each model/scenario

# %%
# load up Myhre parameters and CO2 concentration
df_myhre = pd.read_csv('../output/myhre_forcing_params.csv', index_col=0)
# df_etminan = pd.read_csv('../output/etminan_forcing_params.csv', index_col=0)
df_co2_conc_ssp = pd.read_csv('../data/ssp_co2_concentration.csv', index_col=0)

# %%
models = list(df_myhre.index)

# %%
erf_co2 = {}
scenarios = df_co2_conc_ssp.columns
for model, params in df_myhre.iterrows():
    erf_co2[model] = {}
    # C_alpha_max in Meinshausen, and the log term to use
    # print(df_co2_conc_ssp.loc[1850,scenario] - params['b1'] / (2 * params['a1']), params['d1'] - params['b1']**2/(4*params['a1']))
    for scenario in scenarios:
        erf_co2[model][scenario] = myhre(df_co2_conc_ssp[scenario], params['d1'])
        # erf_co2[model][scenario] = etminan(df_co2_conc_ssp[scenario], df_co2_conc_ssp.loc[1850,scenario], params['a1'], params['b1'], params['d1'])

# %%
erf_all = {}
erf_nonco2 = {}
for model in models:
    erf_all[model] = {}
    erf_nonco2[model] = {}
    scenario_paths = glob.glob(os.path.join(*os.path.normpath(f'../data/transient_forcing_estimates/{model}/*').split(os.sep)))
    for scenario_path in scenario_paths:
        scenario = scenario_path.split(os.sep)[-1]
        if scenario in scenarios:
            model_run_paths = glob.glob(os.path.join(*os.path.normpath(f'../data/transient_forcing_estimates/{model}/{scenario}/*').split(os.sep)))
            n_runs = len(model_run_paths)
            for i_run, model_run_path in enumerate(model_run_paths):
                this_erf = pd.read_csv(model_run_path)['ERF'].values
                if i_run==0:
                    erf = np.zeros_like(this_erf)
                truncate_at = min(len(this_erf), len(erf))
                erf = erf[:truncate_at] + this_erf[:truncate_at]
            erf_all[model][scenario] = erf / n_runs
            erf_nonco2[model][scenario] = erf_all[model][scenario] - erf_co2[model][scenario][:truncate_at]

# %%
colors = {
    'ssp119': '#00a9cf',
    'ssp126': '#003466',
    'ssp245': '#f69320',
    'ssp370': '#df0000',
    'ssp585': '#980002'
}

# %%
fig, ax = pl.subplots(5, 9, figsize=(16, 9))
i_model = 0
for model in erf_nonco2:
    plot_this = False
    for scenario in scenarios:
        if scenario in erf_nonco2[model]:
            plot_this = True
    if plot_this:
        ax_i = i_model//9
        ax_j = i_model%9
        for scenario in scenarios:
            if scenario in erf_nonco2[model]:
                ax[ax_i,ax_j].plot(erf_nonco2[model][scenario], color=colors[scenario])
        ax[ax_i,ax_j].set_title(model, size=9)
        i_model = i_model + 1
fig.tight_layout()
os.makedirs('../plots/', exist_ok=True)
pl.savefig('../plots/non_co2_erf_by_model.png')

# %%
fig, ax = pl.subplots(2, 3, figsize=(16, 9))
i_scenario = 0
for scenario in colors:
    ax_i = i_scenario//3
    ax_j = i_scenario%3
    for model in erf_co2:
        ax[ax_i,ax_j].plot(erf_co2[model][scenario], color=colors[scenario])
    ax[ax_i,ax_j].set_title(scenario, size=9)
    i_scenario = i_scenario + 1
fig.tight_layout()

# %%
fig, ax = pl.subplots(2, 3, figsize=(16, 9))
i_scenario = 0
for scenario in colors:
    ax_i = i_scenario//3
    ax_j = i_scenario%3
    for model in erf_co2:
        ax[ax_i,ax_j].plot(erf_co2[model][scenario], color=colors[scenario])
    ax[ax_i,ax_j].set_title(scenario, size=9)
    i_scenario = i_scenario + 1
    ax[ax_i,ax_j].set_xlim(1850, 2100)
    ax[ax_i,ax_j].set_ylim(0, 14)
fig.tight_layout()
pl.savefig('../plots/co2_erf.png')

# %%
fig, ax = pl.subplots(2, 3, figsize=(16, 9))
i_scenario = 0
for scenario in colors:
    ax_i = i_scenario//3
    ax_j = i_scenario%3
    for model in erf_nonco2:
        if scenario in erf_nonco2[model]:
            ax[ax_i,ax_j].plot(erf_nonco2[model][scenario], color=colors[scenario])
    ax[ax_i,ax_j].set_title(scenario, size=9)
    i_scenario = i_scenario + 1
fig.tight_layout()

# %%
fig, ax = pl.subplots(2, 3, figsize=(16, 9))
i_scenario = 0
for scenario in colors:
    i_model = 0
    ax_i = i_scenario//3
    ax_j = i_scenario%3
    for model in erf_nonco2:
        if scenario in erf_nonco2[model]:
            ax[ax_i,ax_j].plot(erf_nonco2[model][scenario], color=colors[scenario])
            i_model = i_model + 1
    ax[ax_i,ax_j].set_title(scenario, size=9)
    ax[ax_i,ax_j].text(1855, 3.5, i_model)
    i_scenario = i_scenario + 1
    ax[ax_i,ax_j].set_xlim(1850, 2100)
    ax[ax_i,ax_j].set_ylim(-4, 4)
fig.tight_layout()
pl.savefig('../plots/nonco2_erf.png')

# %%
fig, ax = pl.subplots(2, 3, figsize=(16, 9))
i_scenario = 0
for scenario in colors:
    i_model = 0
    ax_i = i_scenario//3
    ax_j = i_scenario%3
    for model in erf_nonco2:
        if scenario in erf_nonco2[model]:
            years = erf_nonco2[model][scenario].index
            smoothed = savgol_filter(
                np.concatenate(
                    (np.zeros(10), 
                     erf_nonco2[model][scenario], 
                     erf_nonco2[model][scenario].values[-1]*np.ones(10))
                ), 11, 1)[10:-10]
            ax[ax_i,ax_j].plot(years, smoothed, color=colors[scenario])
            i_model = i_model + 1
    ax[ax_i,ax_j].set_title(scenario, size=9)
    ax[ax_i,ax_j].text(1855, 3.5, i_model)
    i_scenario = i_scenario + 1
    ax[ax_i,ax_j].set_xlim(1850, 2100)
    ax[ax_i,ax_j].set_ylim(-2, 4)
fig.tight_layout()
pl.savefig('../plots/nonco2_erf_smoothed.png')

# %%
fig, ax = pl.subplots(2, 3, figsize=(16, 9))
i_scenario = 0
for scenario in colors:
    i_model = 0
    ax_i = i_scenario//3
    ax_j = i_scenario%3
    for model in erf_nonco2:
        if scenario in erf_nonco2[model]:
            years = erf_nonco2[model][scenario].index
            nonco2_smoothed = savgol_filter(
                np.concatenate(
                    (np.zeros(10), 
                     erf_nonco2[model][scenario].values, 
                     erf_nonco2[model][scenario].values[-1]*np.ones(10))
                ), 11, 1)[10:-10]
            ax[ax_i,ax_j].plot(years, nonco2_smoothed / (nonco2_smoothed + erf_co2[model][scenario].values[:len(years)]), color=colors[scenario])
            i_model = i_model + 1
    ax[ax_i,ax_j].set_title(scenario, size=9)
    ax[ax_i,ax_j].text(2092, -0.9, i_model)
    i_scenario = i_scenario + 1
    ax[ax_i,ax_j].set_xlim(2000, 2100)
    ax[ax_i,ax_j].set_ylim(-1, 0.5)
fig.tight_layout()
pl.savefig('../plots/nonco2_to_total_ratio_smoothed.png')

# %%
os.makedirs('../output/non-co2-erf/', exist_ok=True)

# %%
for model in models:
    if len(erf_nonco2[model]) > 0:
        df_out = pd.DataFrame(erf_nonco2[model])
        df_out.to_csv(f'../output/non-co2-erf/{model}.csv')

# %%
len(erf_nonco2[model])

# %%
savgol_filter(np.concatenate(
                    (np.zeros(10), 
                     erf_nonco2[model][scenario], 
                     erf_nonco2[model][scenario].values[-1]*np.ones(10))
                ), 11, 1)[:-10]

# %%
