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
# # Calculate anomalies after dedrifing

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl


# %%
def forcing_F13(tasdata, Ndata, model, years = None):
    parameter_table = pd.read_csv('../data/best_estimated_parameters_allmembers4xCO2.csv', index_col=0)
    if years == None:
        GregoryT2x = parameter_table.loc[model,'GregoryT2x']
        GregoryF2x = parameter_table.loc[model,'GregoryF2x']
    if years == '1-20':
        GregoryT2x = parameter_table.loc[model,'GregoryT2x_1-20']
        GregoryF2x = parameter_table.loc[model,'GregoryF2x_1-20']
    fbpar = GregoryF2x/GregoryT2x
    F = Ndata + fbpar*tasdata
    return F


# %%
def tas_predictors(t, fixed_par, exptype = 'stepforcing', timevaryingforcing = []):
    # compute components/predictors for T_n(t) = exp(-t/tau_n)*F(t) (* is a convolution)
    # input for stepforcing: years, fixed parameters (timescales for stepforcing)
    # stepforcing_ computes response to unit forcing,
    # to be multiplied by the actual forcing afterwards
    
    # timevaryingforcing: need a forcing time series input
    if exptype == 'stepforcing':
        timescales = fixed_par; dim = len(timescales)
        predictors = np.zeros((len(t),dim))
        for i in range(0,dim): 
            predictors[:,i] = (1 - np.exp((-t)/timescales[i]))
    elif exptype == 'timevaryingforcing': # need forcing input
        # compute components T_n(t) = exp(-t/tau_n)*F(t) (Here * is a convolution)
        timescales = fixed_par
        lf = len(timevaryingforcing); dim = len(timescales)
        predictors = np.full((lf,dim),np.nan)   

        # compute exact predictors by integrating greens function
        for k in range(0,dim):
            # dot after 0 to create floating point numbers:
            intgreensti = np.full((lf,lf),0.)   
            for t in range(0,lf):
                # compute one new contribution to the matrix:
                intgreensti[t,0] = timescales[k]*(np.exp(-t/timescales[k]) - np.exp(-(t+1)/timescales[k]))
                # take the rest from row above:
                if t > 0:
                    intgreensti[t,1:(t+1)] = intgreensti[t-1,0:t]
            # compute discretized convolution integral by this matrix product:
            predictors[:,k] = intgreensti@np.array(timevaryingforcing)
    else:
        print('unknown exptype')
    return predictors


# %%
def dpy(start_year, end_year, ds_calendar): # days per year
    leap_boolean = [leap_year(year, calendar = ds_calendar)\
                    for year in range(start_year, end_year)]
    leap_int = np.multiply(leap_boolean,1) # converts True/False to 1/0
    
    noleap_dpy = np.array(dpm[ds_calendar]).sum()
    leap_dpy = noleap_dpy + leap_int  
    return leap_dpy


# %%
dpm = {'noleap': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '365_day': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], # I assume this is the same as noleap
       'standard': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'julian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], ##### I think this should be correct
       'proleptic_gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '360_day': [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
      }


# %%
# function copied from: http://xarray.pydata.org/en/stable/examples/monthly-means.html
def leap_year(year, calendar='standard'):
    """Determine if year is a leap year"""
    leap = False
    if ((calendar in ['standard', 'gregorian',
        'proleptic_gregorian', 'julian']) and
        (year % 4 == 0)):
        leap = True
        if ((calendar == 'proleptic_gregorian') and
            (year % 100 == 0) and
            (year % 400 != 0)):
            leap = False
        elif ((calendar in ['standard', 'gregorian']) and
                 (year % 100 == 0) and (year % 400 != 0) and
                 (year < 1583)):
            leap = False
    return leap


# %%
branch_time_in_parent = 60265
parent_experiment_id = 'esm-hist'

# %%
columnames_branchinfo_overview = ['exp', 'member', 'piControl branch time (days)', 'nearest time in table (days)', 'days difference', 'piControl branch time (year)']
branchinfo_overview_df = pd.DataFrame(columns = columnames_branchinfo_overview)


#1,NorESM2-LM,esm-hist,r1i1p1f1,esm-piControl,r1i1p1f1,[0.],[521950.],days since 0421-01-01,Hybrid-restart from year 1851-01-01 of esm-piControl
#2,NorESM2-LM,esm-hist,r2i1p1f1,esm-piControl,r1i1p1f1,[0.],[547550.],days since 1751-01-01,Hybrid-restart from year 1901-01-01 of esm-piControl
#3,NorESM2-LM,esm-hist,r3i1p1f1,esm-piControl,r1i1p1f1,[0.],[73000.],days since 1751-01-01,Hybrid-restart from year 1951-01-01 of esm-piControl


# %%
model = 'NorESM2-LM'
experiments = [
    'esm-ssp119',
    'esm-ssp245',
    'esm-ssp534-over',
]
var_list = ['tas', 'rlut', 'rsut', 'rsdt']
historical_path = f'../data/processed_data/{model}/esm-hist/climate/'

# %%
# 3 members; where are the branch points?
available_members_1850_2100 = [f'r{run}i1p1f1' for run in range(1, 4)]
available_members_2101_2299 = [f'r{run}i1p1f2' for run in range(1, 4)] 

# %%
piControl_path = f'../data/processed_data/{model}/esm-piControl/climate/'
branch_time_file = f'../data/branch_times/{model}_branch_times.csv'
table = pd.read_table(branch_time_file,index_col=0, sep = ',')
for exp in experiments:
    exptable = table.loc[table['exp'] == exp]
    exp_path=f'../output/processed/{model}/{exp}/climate/'
    for member in available_members_1850_2100 + available_members_2101_2299:
        os.makedirs(os.path.join('..', 'output', 'processed', model, exp, 'anomalies'), exist_ok=True)
        member_df = exptable.loc[exptable['member'] == member]
        member_calendar = 'proleptic_gregorian'

        # load exp data
        exp_filename = member + '.csv'
        exp_data = pd.read_table(exp_path + exp_filename, index_col=0, sep = ',')
        if np.isnan(exp_data).values.any():
            print('Warning: data contain NaN')
        exp_years = exp_data.index.values
        if len(str(exp_years[0]))>4:
            # then it contains info about start month too,
            # because experiment does not start in january
            exp_years = [str(yr)[:4] for yr in exp_years] # this code is not tested yet
        exp_start_year = exp_years[0]
        exp_len = len(exp_years)

        # find historical parent member 
        parent_member = member_df['parent_variant_id'].values[0]
        parent_table = table.loc[table['exp'] == 'esm-hist']
        parent_df = parent_table.loc[parent_table['member'] == parent_member]
        piControl_timeunit_start_year = int(parent_df['parent_time_units'].values[0][11:15])

        # find first year of historical parent (usually 1850)
        historical_parent_filename = model + '_esm-hist_' + parent_member + '_means.csv'
        historical_parent_data = pd.read_table(historical_path + historical_parent_filename, index_col=0, sep = ',')
        first_year_historical_parent = historical_parent_data.index.values[0]

        # check branch for historical parent only
        branch_time_days = int(float(parent_df['branch_time_in_parent'].values[0]))
        #branch_time_days = int(''.join(filter(str.isdigit, parent_df['branch_time_in_parent'].values[0])))

        #branch_time_days = parent_df['branch_time_in_parent'].values[0]
        piControl_member = parent_df['parent_variant_id'].values[0]

        # load piControl values. 
        piControl_filename = model + '_esm-piControl_' + piControl_member + '_means.csv'
        piControl_data = pd.read_table(piControl_path + piControl_filename, index_col=0, sep = ',')
        if np.isnan(piControl_data).values.any():
            print('Warning: piControl data contain NaN')
        piControl_years = piControl_data.index.values
        piControl_start_year = piControl_years[0]

        piControl_start_diff = piControl_start_year - piControl_timeunit_start_year
        if piControl_start_year != piControl_timeunit_start_year:
            print('Note: piControl starts', piControl_start_diff, 'years after its time unit starts')
            #piControl_timeunit_start_year = piControl_timeunit_correction(model, exp, member, piControl_timeunit_start_year, piControl_start_year)

        if model in ['CanESM5', 'CanESM5-CanOE']:
            len_days_table = 6000 # since piControl starts a long time after its time unit starts
        else:
            len_days_table = 1500
        days_table = np.append([0],np.cumsum(dpy(piControl_timeunit_start_year,piControl_timeunit_start_year+len_days_table, member_calendar)))    
        # find index of element closest to branch_time_days:
        years_since_piControl_timeunit_start = (np.abs(days_table - branch_time_days)).argmin()
        years_since_piControl_start = years_since_piControl_timeunit_start - piControl_start_diff

        # years_since_piControl_start = branch_time_correction(
        #     model, exp, member, branch_time_days, piControl_timeunit_start_year, piControl_start_year, years_since_piControl_start
        # )
        # write function to correct this for some models
        # applies for NorESM

        piControl_branch_year = piControl_start_year + years_since_piControl_start

        # collect info in overview table:
        exp_branchinfo_df = pd.DataFrame([[exp, member, branch_time_days, days_table[years_since_piControl_timeunit_start], days_table[years_since_piControl_timeunit_start] - branch_time_days, piControl_branch_year]], columns = columnames_branchinfo_overview)
        branchinfo_overview_df = pd.concat([branchinfo_overview_df, exp_branchinfo_df], ignore_index = True)

        years_since_piControl_branch = exp_years - first_year_historical_parent
        corr_piControl_years = piControl_branch_year + years_since_piControl_branch #np.arange(165,251)

        # Anomalies and piControl_linfit should have the same size and time index as exp_data
        # therefore we just copy, and then overwrite the values
        anomalies = exp_data.copy(deep=True)
        piControl_linfit = exp_data.copy(deep=True)
        for var in var_list:
            p1 = np.polyfit(piControl_years, piControl_data[var], 1)

            # make linear fit
            if set(corr_piControl_years).issubset(set(piControl_years)):
                # then all corr_piControl_years are available
                piControl_linfit[var] = np.polyval(p1,corr_piControl_years)
            else:
                # extend the linear fit outside the range of the original piControl years
                # used just for plotting before?
                #corr_piControl_years = list(set(corr_piControl_years).union(set(piControl_years)))
                #corr_piControl_years.sort()
                piControl_linfit[var] = np.polyval(p1,corr_piControl_years)

            anomalies[var] = exp_data[var] - piControl_linfit[var]
        anomalies.to_csv(os.path.join('..', 'output', 'processed', model, exp, 'anomalies', f'{member}.csv'))

# %%
