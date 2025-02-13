#!/usr/bin/env python
# coding: utf-8

# # Process MPI-ESM E-driven data - CO2
# 
# We want to create annual global mean data from each variable. For CO2 we also want to calculate the annual global mean mass mixing ratio.

# In[1]:


import os
import pandas as pd
import numpy as np


df = pd.read_csv('../data/atm_CO2mass_NorESM2-LM_GtC/atm_CO2mass_NorESM2-LM_ssp119.csv', index_col=0)
r1f1_hist = df.loc[1850.5:2014.5, 'r1i1p1f1/2'].values / 2.124
r2f1_hist = df.loc[1850.5:2014.5, 'r2i1p1f1/2'].values / 2.124
r3f1_hist = df.loc[1850.5:2014.5, 'r3i1p1f1/2'].values / 2.124

r1f1_119 = df.loc[2015.5:2100.5, 'r1i1p1f1/2'].values / 2.124
r2f1_119 = df.loc[2015.5:2100.5, 'r2i1p1f1/2'].values / 2.124
r3f1_119 = df.loc[2015.5:2100.5, 'r3i1p1f1/2'].values / 2.124

r1f2_119 = df.loc[2101.5:, 'r1i1p1f1/2'].values / 2.124
r2f2_119 = df.loc[2101.5:, 'r2i1p1f1/2'].values / 2.124
r3f2_119 = df.loc[2101.5:, 'r3i1p1f1/2'].values / 2.124

df = pd.read_csv('../data/atm_CO2mass_NorESM2-LM_GtC/atm_CO2mass_NorESM2-LM_ssp245.csv', index_col=0)
r1f1_245 = df.loc[2015.5:2100.5, 'r1i1p1f1/2'].values / 2.124
r2f1_245 = df.loc[2015.5:2100.5, 'r2i1p1f1/2'].values / 2.124
r3f1_245 = df.loc[2015.5:2100.5, 'r3i1p1f1/2'].values / 2.124

r1f2_245 = df.loc[2101.5:, 'r1i1p1f1/2'].values / 2.124
r2f2_245 = df.loc[2101.5:, 'r2i1p1f1/2'].values / 2.124
r3f2_245 = df.loc[2101.5:, 'r3i1p1f1/2'].values / 2.124

df = pd.read_csv('../data/atm_CO2mass_NorESM2-LM_GtC/atm_CO2mass_NorESM2-LM_ssp534.csv', index_col=0)
r1f1_534 = df.loc[2015.5:2100.5, 'r1i1p1f1/2'].values / 2.124
r2f1_534 = df.loc[2015.5:2100.5, 'r2i1p1f1/2'].values / 2.124
r3f1_534 = df.loc[2015.5:2100.5, 'r3i1p1f1/2'].values / 2.124

r1f2_534 = df.loc[2101.5:, 'r1i1p1f1/2'].values / 2.124
r2f2_534 = df.loc[2101.5:, 'r2i1p1f1/2'].values / 2.124
r3f2_534 = df.loc[2101.5:, 'r3i1p1f1/2'].values / 2.124

os.makedirs(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-hist', 'co2_vmr'), exist_ok=True)
os.makedirs(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp119', 'co2_vmr'), exist_ok=True)
os.makedirs(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp245', 'co2_vmr'), exist_ok=True)
os.makedirs(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp534-over', 'co2_vmr'), exist_ok=True)

pd.DataFrame({'co2': r1f1_hist}, index=np.arange(1850, 2015)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-hist', 'co2_vmr', 'r1i1p1f1.csv'))
pd.DataFrame({'co2': r2f1_hist}, index=np.arange(1850, 2015)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-hist', 'co2_vmr', 'r2i1p1f1.csv'))
pd.DataFrame({'co2': r3f1_hist}, index=np.arange(1850, 2015)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-hist', 'co2_vmr', 'r3i1p1f1.csv'))

pd.DataFrame({'co2': r1f1_119}, index=np.arange(2015, 2101)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp119', 'co2_vmr', 'r1i1p1f1.csv'))
pd.DataFrame({'co2': r2f1_119}, index=np.arange(2015, 2101)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp119', 'co2_vmr', 'r2i1p1f1.csv'))
pd.DataFrame({'co2': r3f1_119}, index=np.arange(2015, 2101)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp119', 'co2_vmr', 'r3i1p1f1.csv'))

pd.DataFrame({'co2': r1f1_245}, index=np.arange(2015, 2101)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp245', 'co2_vmr', 'r1i1p1f1.csv'))
pd.DataFrame({'co2': r2f1_245}, index=np.arange(2015, 2101)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp245', 'co2_vmr', 'r2i1p1f1.csv'))
pd.DataFrame({'co2': r3f1_245}, index=np.arange(2015, 2101)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp245', 'co2_vmr', 'r3i1p1f1.csv'))

pd.DataFrame({'co2': r1f1_534}, index=np.arange(2015, 2101)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp534-over', 'co2_vmr', 'r1i1p1f1.csv'))
pd.DataFrame({'co2': r2f1_534}, index=np.arange(2015, 2101)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp534-over', 'co2_vmr', 'r2i1p1f1.csv'))
pd.DataFrame({'co2': r3f1_534}, index=np.arange(2015, 2101)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp534-over', 'co2_vmr', 'r3i1p1f1.csv'))

pd.DataFrame({'co2': r1f2_119}, index=np.arange(2101, 2300)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp119', 'co2_vmr', 'r1i1p1f2.csv'))
pd.DataFrame({'co2': r2f2_119}, index=np.arange(2101, 2300)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp119', 'co2_vmr', 'r2i1p1f2.csv'))
pd.DataFrame({'co2': r3f2_119}, index=np.arange(2101, 2300)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp119', 'co2_vmr', 'r3i1p1f2.csv'))

pd.DataFrame({'co2': r1f2_245}, index=np.arange(2101, 2300)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp245', 'co2_vmr', 'r1i1p1f2.csv'))
pd.DataFrame({'co2': r2f2_245}, index=np.arange(2101, 2300)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp245', 'co2_vmr', 'r2i1p1f2.csv'))
pd.DataFrame({'co2': r3f2_245}, index=np.arange(2101, 2300)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp245', 'co2_vmr', 'r3i1p1f2.csv'))

pd.DataFrame({'co2': r1f2_534}, index=np.arange(2101, 2300)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp534-over', 'co2_vmr', 'r1i1p1f2.csv'))
pd.DataFrame({'co2': r2f2_534}, index=np.arange(2101, 2300)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp534-over', 'co2_vmr', 'r2i1p1f2.csv'))
pd.DataFrame({'co2': r3f2_534}, index=np.arange(2101, 2300)).to_csv(os.path.join('..', 'output', 'processed', 'NorESM2-LM', 'esm-ssp534-over', 'co2_vmr', 'r3i1p1f2.csv'))




