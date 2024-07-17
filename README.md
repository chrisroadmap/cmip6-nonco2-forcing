# Estimating non-CO2 forcing from CMIP6 models

This repository estimates the non-CO2 component of the effective radiative forcing from CMIP6 models, and it also processes data from the MPI-ESM and NorESM emissions-driven simulations that were produced under the WorldTrans Horizon EU project.

TODO: explain how the repo works and what everything does in the notebooks

## Setting up environment

After cloning, first create your `conda` environment using

```
$ conda env create -f environment.yml
$ conda activate cmip6-nonco2-forcing
$ nbstripout --install
```

Then, create a new file in the root directory called `.env`. Within here create an environment variable called `DATADIR` that will point to a space on your personal disk where large datafiles from the MPI-ESM and NorESM models are downloaded to.

Example `.env` file:

```
DATADIR=/data/users/cjsmith/WorldTrans/ESM-runs
```
