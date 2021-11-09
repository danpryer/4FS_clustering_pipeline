# First step of the clustering pipeline:
# Sets up the required folder structures and variables,
# and saves the relevant information to a pickle file.

# imports
import os
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from data_io import save_object
from structures import catalogue_variables

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

### Change the below for each new set of catalogues that you run the pipeline for

# The master name of the current catalogues to run the pipeline on.
# The pickle file storing these varibles will have this name
# with '_catalogue_vars' appended to the end.
vars_name = '2021_05'

# Optional: keep a log of the names of the input and output raw data files
# from 4FS that the reduced catalogues are sourced from
raw_input_file = 'cat-20210525'
raw_output_file = 'iwg2_20210610_run03fix4newtiling'

# define variables for folder structures. Variables should end with a '/' character.
# This is where the data and results of the pipeline will be stored, not the pipeline
# code itself.
root_path = '/cosma6/data/dp004/dc-prye1/4fs_clustering_project/' # root directory which will store all of the data files and results

# Folder names for the input and output folders:
input_folder = '2021_05_Input/'
output_folder = '2021_05_Output/'

# *** TODO: add in base file names, eg. 'input_reduced_' and 'output_reduced_'?

# define cosmological parameters
cosmo = FlatLambdaCDM(
H0=100.0 * u.km / u.s / u.Mpc,
Om0=0.3089 , Ob0=0.0486)
hubble_h = 0.6774 # little h

# tracers in the catalogues
tracers = ['BG', 'LRG', 'QSO', 'LyA']
catalogues = ['input', 'output']

# Sky fraction covered by survey. Currently the CRS has a footprint of 7500 deg^2
# which is about 18% of the full 41000 deg^2.
fsky = 0.18

# multiplier amount for how many more random points you want than data points,
# for use in clustering analysis
rand_multi = 50

# *** TODO: add options for doing mag cut? e.g. have a mag lower and mag upper?

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# The code below does the rest in terms of folder structure setup

# first assign variables to a class
vars = catalogue_variables(vars_name, root_path, input_folder, output_folder, cosmo, hubble_h,
                            catalogues, tracers, rand_multi, raw_input_file, raw_output_file, fsky)

# make paths
vars.make_paths()


# save pickle object
save_object(vars, 'catalogue_vars/' + vars.vars_name + '_catalogue_vars')
