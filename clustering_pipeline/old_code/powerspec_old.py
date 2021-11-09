# Main module to calculate the power spectrum, using the nbodykit package,
# info found on their main website: https://nbodykit.readthedocs.io/en/latest/index.html
# and source code on github: https://github.com/bccp/nbodykit
# Run as an array job equalling the number of catalogues that need to be processed

# imports
import numpy as np
import pandas as pd
# from mpi4py import MPI
import resource # track memory usage
import time
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.interpolate import InterpolatedUnivariateSpline
import fitsio
import sys
from nbodykit.lab import *
from nbodykit import setup_logging
from data_io import load_object, save_object
from structures import results_powerspec

# ******************************************************************************

# # start MPI
# comm = MPI.COMM_WORLD
# root = 0
# ntasks = MPI.COMM_WORLD.Get_size()
# this_rank = comm.rank
#
# # if on the root note then start clock
# if comm.rank == root:
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
assignment = 'tsc'

# resolution of mesh that data is painted to
Nmesh = 1024

# aliasing corrections
compensation = True # Jing aliasing correction for the gridding window convolution
interlacing = True # sefusatti interlacing correction for aliasing

# Variables and bin parameters for the measurements.
# These are given as arrays and should match your catalogues variable
# i.e. if your catalogues = ['BG', 'LRG', 'QSO', 'LyA'] then the 0th element in
# the arrays below is for the BGs, the 1st is for LRG etc..
# nbins_arr = [25, 25, 25, 25] # number of bins
kmin = [0.01, 0.01, 0.01, 0.01] # min and
kmax = [None, None, None, None] # max k of bin edges. set to None to use k_max = k_nyq
dk = [0.005, 0.005, 0.005, 0.005] # linear spacing of kbins
pk_fid = [1e4, 1e4, 1e4, 1e4] # fiducial P_0(k) for FKP weights
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
kmax = kmax[tracer_rank]
dk = dk[tracer_rank]
pk_fid = pk_fid[tracer_rank]

# determine save folder
if catalogue=='input':
    save_folder = cat_vars.results_input
else:
    save_folder = cat_vars.results_output

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

print('CPU %s loading %s %s data...'%(job_rank, tracer, catalogue), flush=True)

# start by loading the data and random catalogues
if catalogue == 'input':
    target = in_folder + 'input_reduced_' + tracer + '.fits'
    target_rand = in_randoms + 'table_' + tracer + '.fits'
else:
    target = out_folder + 'output_reduced_' + tracer + '.fits'
    target_rand = out_randoms + 'table_' + tracer + '.fits'

source_data = fitsio.read(target)
source_rand = fitsio.read(target_rand)

Ndata = len(source_data)
Nrand = len(source_rand)

# shift the cartesian coords so none of them are negative / when painted to a box
# the bottom corner of the box will be at [0., 0., 0.]
# source_data['xpos'] -= np.amin(source_data['xpos'])
# source_data['ypos'] -= np.amin(source_data['ypos'])
# source_data['zpos'] -= np.amin(source_data['zpos'])
# source_rand['xpos'] -= np.amin(source_rand['xpos'])
# source_rand['ypos'] -= np.amin(source_rand['ypos'])
# source_rand['zpos'] -= np.amin(source_rand['zpos'])

# next, we read in this source data to nbodykits catalogue class
cat_data = ArrayCatalog(source_data)
cat_rand = ArrayCatalog(source_rand)

# finally in this section we add in a 3-vector position column
cat_data['Position'] = cat_data['xpos'][:, None] * [1, 0, 0] + cat_data['ypos'][:, None] * [0, 1, 0] + cat_data['zpos'][:, None] * [0, 0, 1]
cat_rand['Position'] = cat_rand['xpos'][:, None] * [1, 0, 0] + cat_rand['ypos'][:, None] * [0, 1, 0] + cat_rand['zpos'][:, None] * [0, 0, 1]

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

print('Computing the n(z)...')
# The next step is to get the n(z) of the data which we will use for FKP weights

# compute n(z) from the randoms (taking into consideration the sky fraction)
zhist = RedshiftHistogram(cat_rand, cat_vars.fsky, cosmo_nbody, redshift='Z')

# re-normalize to the total size of the data catalog
alpha = 1.0 * cat_data.csize / cat_rand.csize

# add n(z) from randoms to the FKP source
nofz = InterpolatedUnivariateSpline(zhist.bin_centers, alpha*zhist.nbar)
cat_data['NZ'] = nofz(cat_data['REDSHIFT_ESTIMATE'])
cat_rand['NZ'] = nofz(cat_rand['Z'])


# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# finally, we initialise an FKP catalogue object, paint to mesh, and compute
# the P(k) multipoles

# initialize the FKP source
fkp = FKPCatalog(cat_data, cat_rand)

# print out the columns
print("columns in FKPCatalog = ", fkp.columns)

# add the n(z) columns to the FKPCatalogue
fkp['data/NZ'] = nofz(cat_data['REDSHIFT_ESTIMATE'])
fkp['randoms/NZ'] = nofz(cat_rand['Z'])

# add the FKP weights to the FKPCatalogue
fkp['data/FKPWeight'] = 1.0 / (1.0 + fkp['data/NZ'] * pk_fid)
fkp['randoms/FKPWeight'] = 1.0 / (1.0 + fkp['randoms/NZ'] * pk_fid)

# paint to mesh
print('Painting to mesh...')
mesh = fkp.to_mesh(Nmesh=Nmesh, window=assignment, compensated=compensation, interlaced=interlacing, nbar='NZ', fkp_weight='FKPWeight')
BoxSize = np.product(mesh.attrs['BoxSize'])

# compute the power spectrum multipoles in linear bins
print('Calculating the power...')
r = ConvolvedFFTPower(mesh, poles=multipoles, dk=dk, kmin=kmin)

# extract the info from the poles
poles = r.poles
nmodes = poles['modes']
Pk2 = None
Pk4 = None
if 0 in multipoles:
    P = poles['power_%d' %0].real # the measured P_0(k)
    Pk0 = P - poles.attrs['shotnoise'] # only subtract shot noise for monopole
    err0 = np.where(nmodes>0, (1./np.sqrt(nmodes))*(Pk0+(BoxSize/Ndata)), -1.)
if 2 in multipoles:
    Pk2 = poles['power_%d' %2].real # the measured P_2(k)
if 4 in multipoles:
    Pk4 = poles['power_%d' %4].real # the measured P_2(k)


# add the results to the class and save
results = results_powerspec(poles['k'], Pk0, err0, nmodes, Ndata, Nrand,
                    dk, Nmesh, assignment, interlacing,
                    compensation, Pk2=Pk2, Pk4=Pk4)
save_object(results, save_folder + 'results_' + tracer + '_pk_' + str(Nmesh) + '_' + assignment + custom_name)

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# End messages and shutdown

# peak memory print
mem = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2**30, 2) # in GiB
# mem_tot = comm.allreduce(mem, op=MPI.SUM)
print('Peak memory = %s GiB'%(mem), flush=True)

# print('Peak total memory = %s'%(mem_tot), flush=True)
end = time.time()
elapsed = end - start
print('Powerspec measurements complete. Total elapsed time = %s minutes.'%(round(elapsed/60,1)), flush=True)
