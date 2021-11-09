# main module to run for generating random catalogues in parallel
# Run on 16 tasks (4 tracers * 2 catalogues each {input and output} * 2 fields on the sky)
# Produces 8 random catalogues (4 tracers * 2 catalogues)

# imports here
# from imports import *
# from data_io import *
# from surv_footprint import *
import numpy as np
import pandas as pd
import healpy as hp
from scipy import interpolate
from numba import jit, njit
from mpi4py import MPI
import resource # track memory usage
import time
import sys
import os
import pickle
import astropy.coordinates
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from data_io import read_to_df, save_df_fits, load_object
from surv_footprint import fourmost_get_s8foot
from aux import rotate_field, gen_fast_map, completeness_sample

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# run variables, change these below as necessary

# full path to the master variables for the catalogues to generate randoms for
cat_vars_path = 'catalogue_vars/2021_05_catalogue_vars'

# set the seed for the random number generator (can just leave this as is)
seed = 98765

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# pull variables from the cat_vars object
cat_vars = load_object(cat_vars_path)
catalogues = cat_vars.catalogues
tracers = cat_vars.tracers
cosmo = cat_vars.cosmo

in_folder = cat_vars.data_input
out_folder = cat_vars.data_output
mask_path = cat_vars.mask_path

rand_multi = cat_vars.rand_multi

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# initialise MPI, start time, pull command line args, and initialise the rng

# start MPI
comm = MPI.COMM_WORLD
root = 0
ntasks = MPI.COMM_WORLD.Get_size()
this_rank = comm.rank

# if on the root note then start clock
if comm.rank == root:
    print('Sys platform = %s.'%(sys.platform), flush=True)
    start = time.time()

# # get the random multiplier as a command line arg when the prog is run
# rand_multi = int(sys.argv[1])

# On linux, getrusage (for memory monitoring) returns in kiB
# On Mac systems, getrusage returns in B
scale = 1.0
if 'linux' in sys.platform:
    scale = 2**10

# below sets up the random number generator to ensure independence between processes
rng = np.random.default_rng(seed) # create the RNG that you want to pass around
ss = rng.bit_generator._seed_seq # get the SeedSequence of the passed RNG
child_states = ss.spawn(ntasks) # create initial independent states (one for each process)
rng = np.random.default_rng(child_states[this_rank]) # each process takes one of these child_states


# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# Next we load the data and assign a catalogue/tracer/field to each core

# run on 16 tasks, use the below to get the catalogue, tracer and field
# get the field (1 or 2)
fields = [1,2]
taskrank = this_rank
if taskrank > 7:
    field = 2
    taskrank -=8
else:
    field = 1
# get the catalogue (in or out) and the tracer (BG, LRG etc..)
if taskrank < 4:
    catalogue = catalogues[0]
    tracer = tracers[taskrank]
else:
    catalogue = catalogues[1]
    tracer = tracers[taskrank - 4]

print('Cpu %s generating randoms for the %s tracers in field %s of the %s catalogue'%(this_rank, tracer, field, catalogue), flush=True)

comm.Barrier()

if this_rank == root:
    print('Loading data...', flush = True)

# now load the data files - for the input cats we only load the input,
# for the output cats we need to load both input and output to get the completeness
df_in_main = read_to_df(in_folder + 'input_reduced_' + tracer + '.fits')
df_out_main = None
if catalogue == 'output':
    df_out_main = read_to_df(out_folder + 'output_reduced_' + tracer + '.fits')


# cut out the relevant field for this task, and rotate field 1
if field == 1:
    df_in = df_in_main[(df_in_main.RA >= 275) | (df_in_main.RA <= 125)].copy()
    # rotate the field so it doesnt cross the 360->0 boundary
    df_in.RA = rotate_field(df_in['RA'].values)
    if catalogue == 'output':
        df_out = df_out_main[(df_out_main.RA >= 275) | (df_out_main.RA <= 125)].copy()
        # rotate the field so it doesnt cross the 360->0 boundary
        df_out.RA = rotate_field(df_out['RA'].values)
    field_lim = [32, 192, -71.5, -14.5] # RA min/max, DEC min/max
else:
    df_in = df_in_main[(df_in_main.RA <= 275) & (df_in_main.RA >= 125)].copy()
    if catalogue == 'output':
        df_out = df_out_main[(df_out_main.RA <= 275) & (df_out_main.RA >= 125)].copy()
    field_lim = [149, 239, -35.5, 4.5] # RA min/max, DEC min/max

# get the number of input and output gals
N_input = len(df_in)
if catalogue == 'output':
    N_output = len(df_out)


# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# next we look to generate randoms:
# - for the input catalogues, there is no completeness factor to consider, so
#   we just generate randoms uniformly on the sphere and then use the mangle files
#   to keep only points in field.
# - for the output catalogues we need to do the above but also consider the completness
#   factor, so we will additionally use healpy to downsample areas of the field
#   based on this factor

# first lets generate randoms uniformly on the sphere and then use pymangle to
# keep only those in field

if catalogue == 'output': # need to upscale the overall randoms as we will be taking more out
    completeness = N_output / N_input
    rand_multi = int(rand_multi / completeness)
    this_rand_req = rand_multi * N_output
else:
    this_rand_req = rand_multi * N_input


# containers to take the master random data
master_array_RA = np.empty(0)
master_array_DEC = np.empty(0)

# how many randoms to generate in each batch. gets changed in the loop
batch_size = int(this_rand_req)

if this_rank == root:
    print('Generating randoms..', flush = True)

# get the field limits
RA_min = field_lim[0]
RA_max = field_lim[1]
DEC_min = field_lim[2]
DEC_max = field_lim[3]

# generate randoms
this_rand_gen = 0
while this_rand_gen < this_rand_req:

    # generate the randoms
    rand_RA = rng.uniform(RA_min, RA_max, batch_size)
    rand_sinDEC = rng.uniform(np.sin(DEC_min*np.pi / 180), np.sin(DEC_max*np.pi / 180), batch_size)
    rand_DEC = np.arcsin(rand_sinDEC) * 180 / np.pi
    del rand_sinDEC

    # check if masked or not - this not_masked is a True / False array
    if field==1:
        rand_RA = rotate_field(rand_RA, -90)
    not_masked = fourmost_get_s8foot(rand_RA * u.deg, rand_DEC * u.deg, mask_path)[0]
    if field==1:
        rand_RA = rotate_field(rand_RA, 90)

    # drop masked points
    del_ind = np.where(not_masked==False)
    rand_RA = np.delete(rand_RA, del_ind)
    rand_DEC = np.delete(rand_DEC, del_ind)

    # append the remaining points to the master arrays
    master_array_RA = np.append(master_array_RA, rand_RA)
    master_array_DEC = np.append(master_array_DEC, rand_DEC)
    del rand_RA, rand_DEC
    this_rand_gen = len(master_array_RA)
    if comm.rank == root:
        print('Root CPU has generated %s / %s randoms.'%(this_rand_gen, this_rand_req), flush=True)

    batch_size = this_rand_req - this_rand_gen
    # # now append to master arrays
    # master_array_RA = np.append(master_array_RA, field_array_RA)
    # master_array_DEC = np.append(master_array_DEC, field_array_DEC)
    # del field_array_RA, field_array_DEC


print('Cpu %s finished generating %s / %s randoms for the %s %s field %s catalogue'%(this_rank, this_rand_gen, this_rand_req, catalogue, tracer, field), flush=True)


# now for the output catalogues use a healpix map to downweight certain pixels
# based on the completeness
if catalogue == 'output':
    print('Cpu %s down weighting output %s randoms by completeness factor...'%(this_rank, tracer), flush=True)
    healpy_n = 2 # choose healpy resolution (smaller number = less pixels that are larger)
    # healpy_n of 2 corresponds to a pixel with radius of ~ 0.7 deg and area of ~ 1.5 square deg
    nside = 12 * pow(healpy_n, 2)
    npixels = 12 * pow(nside, 2)

    # get pixel numbers (indices) of the data
    pixel_vals_in = hp.ang2pix(nside, df_in["RA"].values, df_in["DEC"].values, lonlat=True)
    pixel_vals_out = hp.ang2pix(nside, df_out["RA"].values, df_out["DEC"].values, lonlat=True)

    # create healpy map of the data - the map is a 1d array where the index = pixel number,
    # and index value = points (i.e. galaxies) inside that pixel
    map_in = gen_fast_map(pixel_vals_in, npixels)
    map_out = gen_fast_map(pixel_vals_out, npixels)

    # get the completeness factor of each pixel
    map_completeness = np.divide(map_out, map_in, out=np.zeros_like(map_out, dtype=np.float64), where=map_in!=0)
    # print('Completeness map len = %s'%(len(map_completeness)), flush=True)

    # get the pixel numbers that each random point lies in
    pixel_vals_rand = hp.ang2pix(nside, master_array_RA, master_array_DEC, lonlat=True)
    del_ind = np.empty(0, dtype=np.int)

    # loop over each pixel now. i is the pixel number
    len_map = len(map_completeness)
    for i in range(len_map):

        # if we have some completeness factor to consider for this pixel:
        if map_completeness[i] != 0.:
            del_ind = completeness_sample(i, map_completeness[i], del_ind, pixel_vals_rand)

    # now delete all of these excess random points
    master_array_RA = np.delete(master_array_RA, del_ind)
    master_array_DEC = np.delete(master_array_DEC, del_ind)

    N_rand_final = len(master_array_RA)
    print('Cpu %s completeness weighting finished for %s %s field %s. N_rand after = %s, rand_multi = %s'%(this_rank, catalogue, tracer, field, N_rand_final, N_rand_final / N_output), flush=True)


# if field 1 then rotate back
if field == 1:
    master_array_RA = rotate_field(master_array_RA, 270)

# now we generate randoms in the radial direction by taking samples of the redshifts
# from the data. Take multiple copies, shuffle, and then discard any excess.
# get N copies of the data array
rand_Z = np.empty(0)
for i in range(rand_multi+1):
    if catalogue == 'input':
        rand_Z = np.append(rand_Z, df_in['REDSHIFT_ESTIMATE'].values)
    else:
        rand_Z = np.append(rand_Z, df_out['REDSHIFT_ESTIMATE'].values)

# # check if we have redshift space coords
# have_redshift_S = False
# if catalogue=='input':
#     if 'redshift_S' in df_in.columns:
#         have_redshift_S=True
#         rand_Z_S = np.empty(0)
#         for i in range(rand_multi+1):
#             rand_Z_S = np.append(rand_Z_S, df_in['redshift_S'].values)
# else:
#     if 'redshift_S' in df_out.columns:
#         have_redshift_S=True
#         rand_Z_S = np.empty(0)
#         for i in range(rand_multi+1):
#             rand_Z_S = np.append(rand_Z_S, df_out['redshift_S'].values)


# # now shuffle them
# # function to do a dual shuffle if necessary
# def unison_shuffled_copies(a, b):
#     assert len(a) == len(b)
#     p = np.random.permutation(len(a))
#     return a[p], b[p]
rng.shuffle(rand_Z)
# if have_redshift_S == False:
#     rng.shuffle(rand_Z)
# else:
#     rand_Z, rand_Z_S = unison_shuffled_copies(rand_Z, rand_Z_S)

# now discard any excess
n_discard = len(rand_Z) - len(master_array_RA)
rand_Z = rand_Z[:-n_discard]
# if have_redshift_S==True:
#     rand_Z_S = rand_Z_S[:-n_discard]

# finally we add the arrays to a dataframe and then save to a fits file
df_rand = pd.DataFrame()
df_rand['RA'] = master_array_RA
df_rand['DEC'] = master_array_DEC
df_rand['Z'] = rand_Z
# if have_redshift_S==True:
#     df_rand['Z_S'] = rand_Z_S

# save each field file to the respective randoms folder
if this_rank == root:
    print('Saving random catalogues..', flush=True)

field_string = '_field' + str(field)
if catalogue == 'output':
    save_df_fits(df_rand, cat_vars.randoms_output + 'table_' + tracer + field_string)
else:
    save_df_fits(df_rand, cat_vars.randoms_input + 'table_' + tracer + field_string)

comm.Barrier()


# now we combine the fields on root into one file and delete the separate
if this_rank == root:

    for catalogue in catalogues:
        if catalogue == 'input':
            folder = cat_vars.randoms_input
        else:
            folder = cat_vars.randoms_output

        for tracer in tracers:
            # load field 1 and 2
            df_field_1 = read_to_df(folder + 'table_' + tracer + '_field1.fits')
            df_field_2 = read_to_df(folder + 'table_' + tracer + '_field2.fits')

            # combine
            RA_1 = df_field_1['RA'].values
            RA_2 = df_field_2['RA'].values
            DEC_1 = df_field_1['DEC'].values
            DEC_2 = df_field_2['DEC'].values
            Z_1 = df_field_1['Z'].values
            Z_2 = df_field_2['Z'].values
            # if 'Z_S' in df_field_1.columns:
            #     Z_S_1 = df_field_1['Z_S'].values
            #     Z_S_2 = df_field_2['Z_S'].values

            RA = np.append(RA_1, RA_2)
            DEC = np.append(DEC_1, DEC_2)
            Z = np.append(Z_1, Z_2)
            # if 'Z_S' in df_field_1.columns:
            #     Z_S = np.append(Z_S_1, Z_S_2)

            # add to fresh df
            df = pd.DataFrame()
            df['RA'] = RA
            df['DEC'] = DEC
            df['Z'] = Z
            # if 'Z_S' in df_field_1.columns:
            #     df['Z_S'] = Z_S

            # create a z -> comov spline
            z_spl = np.linspace(0, np.amax(df['Z'].values)*1.05, 10000)
            comov_spl = [cosmo.comoving_distance(z).value for z in z_spl] # get values for spline
            spl_z_to_r = interpolate.interp1d(z_spl, comov_spl) # create spline using scipy package

            # now add on comoving distance and cartesian x,y,z
            df['comov_dist'] = spl_z_to_r(df['Z'].values)
            x,y,z = astropy.coordinates.spherical_to_cartesian(df['comov_dist'].values*u.Mpc, np.radians(df['DEC'].values), np.radians(df['RA'].values))
            # # shift the cartesian coords so none of them are negative / when painted to a box
            # # the bottom corner of the box will be at [0., 0., 0.]
            # x -= np.amin(x)
            # y -= np.amin(y)
            # z -= np.amin(z)
            # add as column to df
            df['xpos'] = x.value
            df['ypos'] = y.value
            df['zpos'] = z.value

            # # and for redshift space coords
            # if 'Z_S' in df_field_1.columns:
            #     df['comov_dist_S'] = spl_z_to_r(df['Z_S'].values)
            #     x,y,z = astropy.coordinates.spherical_to_cartesian(df['comov_dist_S'].values*u.Mpc, np.radians(df['DEC'].values), np.radians(df['RA'].values))
            #     df['xpos_S'] = x.value
            #     df['ypos_S'] = y.value
            #     df['zpos_S'] = z.value


            save_df_fits(df, folder + 'table_' + tracer)
            # delete individual field files
            os.remove(folder + 'table_' + tracer + '_field1.fits')
            os.remove(folder + 'table_' + tracer + '_field2.fits')


# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# end messages etc

# peak memory print
mem = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2**30, 2) # in GiB
mem_tot = comm.allreduce(mem, op=MPI.SUM)
print('Peak memory on CPU %s = %s GiB'%(this_rank, mem))

comm.Barrier()

if comm.rank == root:
    print('Peak total memory = %s'%(mem_tot), flush=True)
    end = time.time()
    elapsed = end - start
    print('Random gens complete. Total elapsed time = %s minutes.'%(round(elapsed/60,1)), flush=True)
