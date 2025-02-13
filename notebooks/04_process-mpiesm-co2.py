#!/usr/bin/env python
# coding: utf-8

# # Process MPI-ESM E-driven data - CO2
# 
# We want to create annual global mean data from each variable. For CO2 we also want to calculate the annual global mean mass mixing ratio.

# In[1]:


import os
from dotenv import load_dotenv
import iris
from iris.util import equalise_attributes
import iris.coord_categorisation as cat
import glob
import matplotlib.pyplot as pl
import warnings
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# In[2]:


load_dotenv(override=True)


# In[3]:


datadir = os.getenv("DATADIR")
datadir


# In[4]:


variables = ['co23D']
experiments = ['esm-hist', 'esm-ssp119', 'esm-ssp126', 'esm-ssp245', 'esm-ssp370', 'esm-ssp534-over', 'esm-ssp585']
ensemble_members = {
    'esm-hist': 10,  # actually 30 but 11-30 failed to download
    'esm-ssp119': 10,
    'esm-ssp126': 10,
    'esm-ssp245': 30,
    'esm-ssp370': 10,
    'esm-ssp534-over': 10,
    'esm-ssp585': 10
}


# In[5]:


# hege does model / scenario then puts all variables and runs in the same folder


# In[6]:


# then we want to calculate what the pressure bounds are from ps
# formula = "p = ap + b*ps"


# In[ ]:


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
                cubes_co23d = iris.load(source_files, "mass_fraction_of_carbon_dioxide_tracer_in_air")
                cubes_ps = iris.load(source_files, "Surface Air Pressure")
                cube_p = iris.load(source_files, "vertical coordinate formula term: ap(k+1/2)")[0]
                hybrid_bounds = cube_p.coord('atmosphere_hybrid_sigma_pressure_coordinate').bounds
                equalise_attributes(cubes_co23d);
                equalise_attributes(cubes_ps);
                cube_co23d = cubes_co23d.concatenate_cube()
                cube_ps = cubes_ps.concatenate_cube()
                area_weights = iris.analysis.cartography.area_weights(cube_ps)
                mass_atmos = (cube_ps * area_weights).data.sum() / 9.80665 / 3420
                hybrid_thickness = -np.diff(hybrid_bounds, axis=1).squeeze()  # proportion of atmospheric mass in each layer; a rescaling of ps
                time_weights = cube_ps.coord("time").bounds[:, 1] - cube_ps.coord("time").bounds[:, 0]
                co2_mmr = np.ones((cube_ps.shape[0]//12)) * np.nan
                for iyear in range(cube_ps.shape[0]//12):
                    mass_atm4d_year = cube_ps.data[12*iyear:12*iyear+12, None, :, :] * hybrid_thickness[None, :, None, None] * area_weights[12*iyear:12*iyear+12, None, :, :]
                    mass_co24d_year = cube_co23d.data[12*iyear:12*iyear+12, ...] * mass_atm4d_year
                    co2_mmr[iyear] = np.average(np.sum(mass_co24d_year, axis=(1,2,3))/np.sum(mass_atm4d_year, axis=(1,2,3)), weights = time_weights[12*iyear:12*iyear+12])
                co2_vmr = 28.97 / 44.009 * co2_mmr * 1e6
                cat.add_year(cube_ps, 'time', name='year')
                cube_gm = cube_ps.collapsed(['latitude', 'longitude'], iris.analysis.MEAN, weights=area_weights)
                cube_agm = cube_gm.aggregated_by('year', iris.analysis.MEAN)
                tempoutput[variable] = co2_vmr
            output[experiment][runid] = pd.DataFrame(co2_vmr, index=cube_agm.coord('year').points, columns=['co2'])
            os.makedirs(os.path.join('..', 'output', 'processed', 'MPI-ESM1-2-LR', experiment, 'co2_vmr'), exist_ok=True)
            output[experiment][runid].to_csv(os.path.join('..', 'output', 'processed', 'MPI-ESM1-2-LR', experiment, 'co2_vmr', f'{runid}.csv'))


# In[ ]:




