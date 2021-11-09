# Main program to calculate the power spectrum, using the nbodykit package,
# info found on their main website: https://nbodykit.readthedocs.io/en/latest/index.html
# and source code on github: https://github.com/bccp/nbodykit
# Run as an array job equalling the number of catalogues that need to be processed

import sys
import time
import resource # track memory usage
from nbodykit.lab import *

from data_io import load_object, save_object
from FFTConvPower import calc_convolved_FFT_power
from structures import results_powerspec, powerspec_measurement


# ******************************************************************************

print('Sys platform = %s.'%(sys.platform), flush=True)
start = time.time()

# On linux, getrusage (for memory monitoring) returns in kiB
# On Mac systems, getrusage returns in B
scale = 1.0
if 'linux' in sys.platform:
    scale = 2**10

# get the job rank from the command line arg
job_rank = int(sys.argv[1])

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************


# *** Run variables, change these below as necessary

# full path to the master variables for the catalogues to generate randoms for
cat_vars_path = 'catalogue_vars/2021_05_catalogue_vars'

# specify if working in 'real' or 'redshift' space
real_or_redshift = 'real'

# mass assignment scheme to be used ['cic', 'tsc']
window = 'tsc'

# resolution of mesh that data is painted to
Nmesh = 1024

# aliasing corrections
compensated = True # Jing aliasing correction for the gridding window convolution
interlaced = True # sefusatti interlacing correction for aliasing

# Variables and bin parameters for the measurements.
# These are given as arrays and should match your catalogues variable
# i.e. if your catalogues = ['BG', 'LRG', 'QSO', 'LyA'] then the 0th element in
# the arrays below is for the BGs, the 1st is for LRG etc..
kmin = [0.01, 0.01, 0.01, 0.01] # min k. max is default set to k_max = k_nyq
dk = [0.05, 0.04, 0.025, 0.025] # linear spacing of kbins
Pk_fid = [1e4, 1e4, 1e4, 1e4] # fiducial P_0(k) for FKP weights
multipoles=[0,2,4] # which power spectrum multipoles to calculate, must always include 0*

# If one wants to add a custom string to the end of the results filenames, this can
# be specified below, and should start with an underscore as it will be tagged
# on to the end of an existing string. For example if you want to do a run with
# finer bins this could be '_finebins' etc.
custom_name = ''


# * the results_powerspec class in structures.py is only coded to take the 0,2,4
# pole so if you want to calculate and record more than this, modify the class
# and the code below when storing the results after the calculation

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# pull variables from the cat_vars object and determine bin parameters from above arrays
cat_vars = load_object(cat_vars_path)

in_folder = cat_vars.data_input # path to input and output data catalogues
out_folder = cat_vars.data_output
in_randoms = cat_vars.randoms_input # path to input and output random catalogues
out_randoms = cat_vars.randoms_output

catalogues = cat_vars.catalogues # get the array of catalogues e.g. ['input', 'output']
tracers = cat_vars.tracers # get the array of tracers e.g. ['BG', 'LRG', ..]
n_tracers = len(tracers)

cosmo_nbody = cosmology.Cosmology(h=cat_vars.hubble_h).match(Omega0_m=cat_vars.cosmo.Om0)

# If we have 2 catalogue types and 4 tracers, we run 2*4 = 8 jobs
# so to get the catalogue and tracer that each job is responsible for:
# if this_rank < n_tracers:
#     catalogue = catalogues[0] # e.g. 'input'
#     tracer = tracers[this_rank]
#     tracer_rank = this_rank
# else:
#     catalogue = catalogues[1] # e.g. 'output'
#     tracer_rank = this_rank - n_tracers
#     tracer = tracers[tracer_rank]

if job_rank < n_tracers:
    catalogue = catalogues[0] # e.g. 'input'
    tracer = tracers[job_rank]
    tracer_rank = job_rank
else:
    catalogue = catalogues[1] # e.g. 'output'
    tracer_rank = job_rank - n_tracers
    tracer = tracers[tracer_rank]

kmin = kmin[tracer_rank]
# kmax = kmax[tracer_rank]
dk = dk[tracer_rank]
Pk_fid = Pk_fid[tracer_rank]

# determine save folder
if catalogue=='input':
    save_folder = cat_vars.results_input
else:
    save_folder = cat_vars.results_output

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

print('Job %s loading %s %s data...'%(job_rank, tracer, catalogue), flush=True)

# start by loading the data and random catalogues...
if catalogue == 'input':
    target = in_folder + 'input_reduced_' + tracer + '.fits'
    target_rand = in_randoms + 'table_' + tracer + '.fits'
else:
    target = out_folder + 'output_reduced_' + tracer + '.fits'
    target_rand = out_randoms + 'table_' + tracer + '.fits'


Z_col = 'REDSHIFT_ESTIMATE'
space_name = ''
if (real_or_redshift == 'redshift') and (tracer in ['BG', 'LRG']):
    Z_col = 'redshift_S'
    space_name = '_zspace'

# read in this source data to nbodykits catalogue class
cat_data = FITSCatalog(target)
cat_rand = FITSCatalog(target_rand)
Ndata = cat_data.csize
Nrand = cat_data.csize
Z_min = (cat_data[Z_col].min()).compute()
Z_max = (cat_data[Z_col].max()).compute()
Z_mid = 0.5*(Z_min + Z_max)

print('Data loaded, calculating P(k)...', flush=True)


cosmo = cosmology.Cosmology(h=cat_vars.hubble_h).match(Omega0_m=cat_vars.cosmo.Om0)


# setup results class to store overall measurement variables and quantities
results = results_powerspec(tracer, catalogue, window, Nmesh, interlaced, compensated,
                Ndata, Nrand, Nbins_Z=1, Z_edges=[Z_min, Z_max], Z_bins_mid=Z_mid)

# call the main function stored in the FFTConvPower.py file
results = calc_convolved_FFT_power(results, cat_data, cat_rand, cosmo, cat_vars, real_or_redshift, tracer, catalogue,
                                Z_min, Z_max, Z_mid, bin_no=0,
                                Nmesh=Nmesh, window=window, compensated=compensated, interlaced=interlaced,
                                multipoles=multipoles, kmin=kmin, dk=dk, Pk_fid=Pk_fid)


# save the results
save_object(results, save_folder + 'results_' + tracer + '_pk_' + str(Nmesh) + '_' + window + space_name + custom_name)

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# End messages

# peak memory print
mem = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2**30, 2) # in GiB
# mem_tot = comm.allreduce(mem, op=MPI.SUM)
print('Peak memory = %s GiB'%(mem), flush=True)

# print('Peak total memory = %s'%(mem_tot), flush=True)
end = time.time()
elapsed = end - start
print('Powerspec measurements complete. Total elapsed time = %s minutes.'%(round(elapsed/60,1)), flush=True)
