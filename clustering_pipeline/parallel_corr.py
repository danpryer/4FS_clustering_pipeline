# Main module to run for calculating correlation functions in parallel.
# Uses MPI to divide out the tasks so that each task does a specific catalogue +
# tracer combo (e.g. task zero does input BG, task one does input LRG etc..).
# This could be run as an array job instead of using MPI, in which case you would
# replace the 'this_rank = comm.rank' with 'this_rank = <array task id>'' and just run
# as an array for e.g. [0-7].
# TreeCorr and CorrFunc use openmp under the hood to parallelise the calcs, and
# so the CPUs per task flag in the submit script should be set to the maximum
# number of CPUs on the node / machine.


# imports here
# from imports import *
# from data_io import *
# from twopointcorr import *
import numpy as np
import pandas as pd
from numba import jit, njit
from mpi4py import MPI
import resource # track memory usage
import time
import sys
import os
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from structures import results_2pcf, results_2pcf_jackknife, full_4FS_results
from data_io import read_to_df, load_object, save_object
from aux import rotate_field
from twopointcorr import wtheta_treecorr, wtheta_jackknife, xi_treecorr, xi_jackknife, wp_corrfunc, wp_jackknife

# ******************************************************************************

# start MPI
comm = MPI.COMM_WORLD
root = 0
ntasks = MPI.COMM_WORLD.Get_size()
this_rank = comm.rank

# get some variables from command line
real_or_redshift=str(sys.argv[1]) # working in real or redshift space coords ['real', 'redshift']
njack = int(sys.argv[2])

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# *** Run variables, change these below as necessary

# full path to the master variables for the catalogues to generate randoms for
cat_vars_path = 'catalogue_vars/2021_05_catalogue_vars'

# specify if working in 'real' or 'redshift' space
# real_or_redshift = 'real'

# number of threads that openmp will use in treecorr/corrfunc
nthreads = 28

# no of jackknife regions to divide the data up into, of equal width in RA
# njack = 20

# If one wants to add a custom string to the end of the results filenames, this can
# be specified below, and should start with an underscore as it will be tagged
# on to the end of an existing string. For example if you want to do a run with
# finer bins this could be '_finebins' etc.
custom_name = ''

# Change the array below to say which statistics you would like to calculate,
# out of 'wtheta', 'xi', and 'wp'.
statistics = ['wtheta', 'xi', 'wp']

# Variables and bin parameters for the measurements.
# These are given as arrays and should match your catalogues variable
# i.e. if your catalogues = ['BG', 'LRG', 'QSO', 'LyA'] then the 0th element in
# the arrays below is for the BGs, the 1st is for LRG etc..
nbins_arr = [20, 20, 20, 20] # number of bins

deg_min_arr = [0.01, 0.01, 0.01, 0.01]  # min and max of bins for w(theta) measurement
deg_max_arr = [10, 10, 10, 10]

r_xi_min_arr = [0.1, 0.1, 0.1, 0.1]  # min and max of bins for Xi(r) measurement
r_xi_max_arr = [50, 50, 50, 50]

r_wp_min_arr = [0.1, 0.1, 0.1, 0.1]  # min and max of bins for w_p(r_p) measurement
r_wp_max_arr = [50, 50, 50, 50]
rpimax_arr = [50, 50, 50, 50] # pi_max for w_p(r_p) measurement

# Verbosity of treecorr and corrfunc in terms of printing output from these libraries
verbose_treecorr = 0 # [0,1,2,3] where 0 is no print progress from treecorr and 4 is debugging level
verbose_corrfunc = False # [True or False]

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

# If we have 2 catalogue types and 4 tracers, we run on 2*4 = 8 tasks so,
# to get the catalogue and tracer that each task is responsible for:
if this_rank < n_tracers:
    catalogue = catalogues[0] # e.g. 'input'
    tracer = tracers[this_rank]
    tracer_rank = this_rank
else:
    catalogue = catalogues[1] # e.g. 'output'
    tracer_rank = this_rank - n_tracers
    tracer = tracers[tracer_rank]


if catalogue=='input':
    save_folder = cat_vars.results_input
else:
    save_folder = cat_vars.results_output


# get the bin variables based on the tracer
nbins = nbins_arr[tracer_rank]
deg_min = deg_min_arr[tracer_rank]
deg_max = deg_max_arr[tracer_rank]
r_xi_min = r_xi_min_arr[tracer_rank]
r_xi_max = r_xi_max_arr[tracer_rank]
r_wp_min = r_wp_min_arr[tracer_rank]
r_wp_max = r_wp_max_arr[tracer_rank]
rpimax = rpimax_arr[tracer_rank]

# determine if working in real or redshift space, making adjustments that QSO and LyA samples dont have the redshift space option at present
xcol = 'xpos'
ycol = 'ypos'
zcol = 'zpos'
Z_col = 'REDSHIFT_ESTIMATE'
Z_col_r = 'Z'
space_name = ''
if (real_or_redshift == 'redshift') and (tracer in ['BG', 'LRG']):
    xcol = 'xpos_S'
    ycol = 'ypos_S'
    zcol = 'zpos_S'
    Z_col = 'redshift_S'
    Z_col_r = 'Z_S'
    space_name = '_zspace'

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# start messages, load catalogues and separate out into fields 1 and 2

# if on the root note then start clock
if comm.rank == root:
    print('Sys platform = %s.'%(sys.platform))
    start = time.time()

# On linux, getrusage returns in kiB
# On Mac systems, getrusage returns in B
scale = 1.0
if 'linux' in sys.platform:
    scale = 2**10

print('Cpu %s running on the %s tracers in the %s catalogue'%(this_rank, tracer, catalogue))

comm.Barrier()

# start by loading the data and random catalogues
if catalogue == 'input':
    target = in_folder + 'input_reduced_' + tracer + '.fits'
    target_rand = in_randoms + 'table_' + tracer + '.fits'
else:
    target = out_folder + 'output_reduced_' + tracer + '.fits'
    target_rand = out_randoms + 'table_' + tracer + '.fits'
    # also need to load the input catalogues to do an initial w(theta) calc to get fibre collision correction
    parent = in_folder + 'input_reduced_' + tracer + '.fits'
    parent_rand = in_randoms + 'table_' + tracer + '.fits'

# load in the data from the .fits files
df_data = read_to_df(target)
df_rand = read_to_df(target_rand)
if catalogue=='output':
    df_parent = read_to_df(parent)
    df_parent_rand = read_to_df(parent_rand)
N_data = len(df_data)
N_rand = len(df_rand)
print('There are %s data points and %s random points in the %s %s catalogue.'%(N_data, N_rand, tracer, catalogue), flush=True)
sys.stdout.flush()

comm.Barrier()

# To save computing time, we don't want to calculate the correlation function "across fields" so we break the data/randoms down into two fields
df_data_1 = df_data[(df_data.RA >= 275) | (df_data.RA <= 125)].copy()
df_data_2 = df_data[(df_data.RA <= 275) & (df_data.RA >= 125)].copy()
df_rand_1 = df_rand[(df_rand.RA >= 275) | (df_rand.RA <= 125)].copy()
df_rand_2 = df_rand[(df_rand.RA <= 275) & (df_rand.RA >= 125)].copy()

if catalogue=='output':
    df_p_1 = df_parent[(df_parent.RA >= 275) | (df_parent.RA <= 125)].copy()
    df_p_2 = df_parent[(df_parent.RA <= 275) & (df_parent.RA >= 125)].copy()
    df_pr_1 = df_parent_rand[(df_parent_rand.RA >= 275) | (df_parent_rand.RA <= 125)].copy()
    df_pr_2 = df_parent_rand[(df_parent_rand.RA <= 275) & (df_parent_rand.RA >= 125)].copy()

# now we will rotate field 1 so it doesnt sit across the 360 deg -> 0 deg boundary
df_data_1['RA'] = rotate_field(df_data_1['RA'].values)
df_rand_1['RA'] = rotate_field(df_rand_1['RA'].values)

if catalogue=='output':
    df_p_1['RA'] = rotate_field(df_p_1['RA'].values)
    df_pr_1['RA'] = rotate_field(df_pr_1['RA'].values)

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************
if 'wtheta' in statistics:
    # now for outputs do an initial w(theta) calc on input and output then calculate the fibre collision correction
    if catalogue=='output':
        res_in_1 = wtheta_treecorr(nbins, deg_min, deg_max, df_p_1['RA'].values, df_p_1['DEC'].values, df_pr_1['RA'].values, df_pr_1['DEC'].values, nthreads, verbose=verbose_treecorr)
        res_out_1 = wtheta_treecorr(nbins, deg_min, deg_max, df_data_1['RA'].values, df_data_1['DEC'].values, df_rand_1['RA'].values, df_rand_1['DEC'].values, nthreads, verbose=verbose_treecorr)
        res_in_2 = wtheta_treecorr(nbins, deg_min, deg_max, df_p_2['RA'].values, df_p_2['DEC'].values, df_pr_2['RA'].values, df_pr_2['DEC'].values, nthreads, verbose=verbose_treecorr)
        res_out_2 = wtheta_treecorr(nbins, deg_min, deg_max, df_data_2['RA'].values, df_data_2['DEC'].values, df_rand_2['RA'].values, df_rand_2['DEC'].values, nthreads, verbose=verbose_treecorr)
        # fibre colision correction = (1 + w_input) / (1 + w_output), where here input = parent
        arr_fc_1 = (1 + np.array(res_in_1.result)) / (1 + np.array(res_out_1.result))
        arr_fc_2 = (1 + np.array(res_in_2.result)) / (1 + np.array(res_out_2.result))
        res_out_1_uncorr = res_out_1.result
        res_out_2_uncorr = res_out_2.result
    else:
        res_out_1_uncorr = None
        res_out_2_uncorr = None
        arr_fc_1 = [1] # on input catalogue dont want to do any weighted correction
        arr_fc_2 = [1]


# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# Now using the jackknife method, calculate the angular, 3d and projected
# clustering and covariance of both fields
if 'wtheta' in statistics:
    print('Calculating w(theta) for %s %s...'%(tracer, catalogue), flush=True)
    sys.stdout.flush()

    results_wtheta_1 = wtheta_jackknife(nbins, deg_min, deg_max, df_data_1['RA'].values, df_data_1['DEC'].values,
                             df_rand_1['RA'].values, df_rand_1['DEC'].values, nthreads, njack, pair_weights=arr_fc_1, verbose=verbose_treecorr)
    results_wtheta_2 = wtheta_jackknife(nbins, deg_min, deg_max, df_data_2['RA'].values, df_data_2['DEC'].values,
                             df_rand_2['RA'].values, df_rand_2['DEC'].values, nthreads, njack, pair_weights=arr_fc_2, verbose=verbose_treecorr)


    # average results from both fields using inverse variance weighting
    wtheta_avg_res = np.zeros(nbins)
    wtheta_avg_var = np.zeros(nbins)
    wtheta_avg_err = np.zeros(nbins)
    for i in range(nbins):
        wtheta_avg_var[i] = 1 / ((1/np.diag(results_wtheta_1.cov)[i])+(1/np.diag(results_wtheta_2.cov)[i]))
        wtheta_avg_res[i] = ((results_wtheta_1.result[i]/ np.diag(results_wtheta_1.cov)[i])+(results_wtheta_2.result[i]/ np.diag(results_wtheta_2.cov)[i])) * wtheta_avg_var[i]
    wtheta_avg_err = np.sqrt(wtheta_avg_var)
    wtheta_avg_cov = 0.5*(results_wtheta_1.cov + results_wtheta_2.cov)


    # add the results to a class and save as a pickle file
    res_wtheta_full = full_4FS_results(results_wtheta_1.bin_mid, results_wtheta_1.result, results_wtheta_2.result, wtheta_avg_res,
                                    results_wtheta_1.err, results_wtheta_2.err, wtheta_avg_err, results_wtheta_1.cov, results_wtheta_2.cov,
                                    wtheta_avg_cov, results_wtheta_1.DD_counts, results_wtheta_2.DD_counts, results_wtheta_1.RR_counts,
                                    results_wtheta_2.RR_counts, results_wtheta_1.DR_counts, results_wtheta_2.DR_counts, res_out_1_uncorr,
                                    res_out_2_uncorr, arr_fc_1, arr_fc_2)
    save_object(res_wtheta_full, save_folder + 'results_' + tracer + '_wtheta' + space_name + custom_name)

# ********************************

if 'xi' in statistics:
    print('Calculating Xi(r) for %s %s...'%(tracer, catalogue), flush=True)
    sys.stdout.flush()

    results_xi_1 = xi_jackknife(nbins, r_xi_min, r_xi_max, df_data_1[xcol].values, df_data_1[ycol].values, df_data_1[zcol].values,
                             df_rand_1['xpos'].values, df_rand_1['ypos'].values, df_rand_1['zpos'].values, nthreads, njack, verbose=verbose_treecorr)
    results_xi_2 = xi_jackknife(nbins, r_xi_min, r_xi_max, df_data_2[xcol].values, df_data_2[ycol].values, df_data_2[zcol].values,
                             df_rand_2['xpos'].values, df_rand_2['ypos'].values, df_rand_2['zpos'].values, nthreads, njack, verbose=verbose_treecorr)


    # average results
    xi_avg_res = np.zeros(nbins)
    xi_avg_var = np.zeros(nbins)
    xi_avg_err = np.zeros(nbins)
    for i in range(nbins):
        xi_avg_var[i] = 1 / ((1/np.diag(results_xi_1.cov)[i])+(1/np.diag(results_xi_2.cov)[i]))
        xi_avg_res[i] = ((results_xi_1.result[i]/ np.diag(results_xi_1.cov)[i])+(results_xi_2.result[i]/ np.diag(results_xi_2.cov)[i])) * xi_avg_var[i]
    xi_avg_err = np.sqrt(xi_avg_var)
    xi_avg_cov = 0.5*(results_xi_1.cov + results_xi_2.cov)

    # add the results to a class and save as a pickle file
    res_xi_full = full_4FS_results(results_xi_1.bin_mid, results_xi_1.result, results_xi_2.result, xi_avg_res,
                                    results_xi_1.err, results_xi_2.err, xi_avg_err, results_xi_1.cov, results_xi_2.cov,
                                    xi_avg_cov, results_xi_1.DD_counts, results_xi_2.DD_counts, results_xi_1.RR_counts,
                                    results_xi_2.RR_counts, results_xi_1.DR_counts, results_xi_2.DR_counts)
    save_object(res_xi_full, save_folder + 'results_' + tracer + '_xi' + space_name + custom_name)


# ********************************
if 'wp' in statistics:
    print('Calculating w_p(r_p) for %s %s...'%(tracer, catalogue), flush=True)
    sys.stdout.flush()

    results_wp_1 = wp_jackknife(nbins, r_wp_min, r_wp_max, rpimax, df_data_1[xcol].values, df_data_1[ycol].values, df_data_1[zcol].values,
                             df_rand_1['xpos'].values, df_rand_1['ypos'].values, df_rand_1['zpos'].values, nthreads, njack, verbose=verbose_corrfunc)
    results_wp_2 = wp_jackknife(nbins, r_wp_min, r_wp_max, rpimax, df_data_2[xcol].values, df_data_2[ycol].values, df_data_2[zcol].values,
                             df_rand_2['xpos'].values, df_rand_2['ypos'].values, df_rand_2['zpos'].values, nthreads, njack, verbose=verbose_corrfunc)

    # average results
    wp_avg_res = np.zeros(nbins)
    wp_avg_var = np.zeros(nbins)
    wp_avg_err = np.zeros(nbins)
    for i in range(nbins):
        wp_avg_var[i] = 1 / ((1/np.diag(results_wp_1.cov)[i])+(1/np.diag(results_wp_2.cov)[i]))
        wp_avg_res[i] = ((results_wp_1.result[i]/ np.diag(results_wp_1.cov)[i])+(results_wp_2.result[i]/ np.diag(results_wp_2.cov)[i])) * wp_avg_var[i]
    wp_avg_err = np.sqrt(wp_avg_var)
    wp_avg_cov = 0.5*(results_wp_1.cov + results_wp_2.cov)

    # add the results to a class and save as a pickle file
    res_wp_full = full_4FS_results(results_wp_1.bin_mid, results_wp_1.result, results_wp_2.result, wp_avg_res,
                                    results_wp_1.err, results_wp_2.err, wp_avg_err, results_wp_1.cov, results_wp_2.cov,
                                    wp_avg_cov, results_wp_1.DD_counts, results_wp_2.DD_counts, results_wp_1.RR_counts,
                                    results_wp_2.RR_counts, results_wp_1.DR_counts, results_wp_2.DR_counts)
    save_object(res_wp_full, save_folder + 'results_' + tracer + '_wp' + space_name + custom_name)

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# End messages and shutdown
comm.Barrier()

# peak memory print
mem = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2**30, 2) # in GiB
mem_tot = comm.allreduce(mem, op=MPI.SUM)
print('Peak memory on CPU %s = %s GiB'%(this_rank, mem), flush=True)

comm.Barrier()

if comm.rank == root:
    print('Peak total memory = %s'%(mem_tot), flush=True)
    end = time.time()
    elapsed = end - start
    print('Random gens complete. Total elapsed time = %s minutes.'%(round(elapsed/60,1)), flush=True)
