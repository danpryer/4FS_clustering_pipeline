# main module to run for generating random catalogues in parallel
# currently only generates points in angular space (RA, DEC), no radial points
# Run on 16 tasks (4 tracers * 2 catalogues each {input and output} * 2 fields on the sky)
# Produces 8 random catalogues (4 tracers * 2 catalogues)

# imports here
from imports import *
from data_io import *
from surv_footprint import *

# function to rotate points in RA by a specificed amount in degrees
@jit
def rotate_field(RA_vals, rotation=90):
    RA_vals += rotation
    for i in range(len(RA_vals)):
        if RA_vals[i] > 360:
            RA_vals[i] -= 360
        if RA_vals[i] < 0:
            RA_vals[i] += 360
    return RA_vals

# start MPI
comm = MPI.COMM_WORLD
root = 0
ntasks = MPI.COMM_WORLD.Get_size()
this_rank = comm.rank

# if on the root note then start clock
if comm.rank == root:
    print('Sys platform = %s.'%(sys.platform), flush=True)
    start = time.time()

comm.Barrier()

# On linux, getrusage returns in kiB
# On Mac systems, getrusage returns in B
scale = 1.0
if 'linux' in sys.platform:
    scale = 2**10

# folder paths
datapath = '/cosma6/data/dp004/dc-prye1/4fs_clustering_project/data/'
in_folder = '2021_01_Input/'
out_folder = '2021_01_Output/'
mask_path = '/cosma/home/dp004/dc-prye1/4fs_clustering_project/notebooks/survey_masks/'

catalogues = ['output', 'input']
tracers = ['BG', 'LRG', 'QSO', 'LyA']
fields = [1,2]

# run on 16 tasks, use the below to get the catalogue, tracer and field
taskrank = this_rank
if taskrank > 7:
    field = 2
    taskrank -=8
else:
    field = 1
# run on 8 tasks so..
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
df_in_main = read_to_df(datapath + in_folder + 'input_reduced_' + tracer + '.fits')
df_out_main = None
if catalogue == 'output':
    df_out_main = read_to_df(datapath + out_folder + 'output_reduced_' + tracer + '.fits')



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


# # diagnostic plots
# if this_rank == root:
#     plt.figure(figsize=(14,7))
#     plt.subplot(1,2,1)
#     plt.title('In data', size=16)
#     plt.xlabel('RA', size=14)
#     plt.ylabel('DEC', size=14)
#     plt.xticks(size=14)
#     plt.yticks(size=14)
#     plt.scatter(df_in['RA'].values, df_in['DEC'].values, s=0.01)
#
#     plt.subplot(1,2,2)
#     plt.title('Out data', size=16)
#     plt.xlabel('RA', size=14)
#     plt.ylabel('DEC', size=14)
#     plt.xticks(size=14)
#     plt.yticks(size=14)
#     plt.scatter(df_out['RA'].values, df_out['DEC'].values, s=0.01)
#
#     plt.savefig('test_plot_1')




# next we look to generate randoms:
# - for the input catalogues, there is no completeness factor to consider, so
#   we just generate randoms uniformly on the sphere and then use the mangle files
#   to keep only points in field.
# - for the output catalogues we need to do the above but also consider the completness
#   factor, so we will additionally use healpy to downsample areas of the field
#   based on this factor

# first lets generate randoms uniformly on the sphere and then use pymangle to
# keep only those in field
rand_multi = 50
if catalogue == 'output': # need to upscale the overall randoms as we will be taking more out
    completeness = N_output / N_input
    rand_multi = int(rand_multi / completeness)
    this_rand_req = rand_multi * N_output
else:
    this_rand_req = rand_multi * N_input




# containers to take the master random data
master_array_RA = np.empty(0)
master_array_DEC = np.empty(0)


batch_size = int(this_rand_req) # how many randoms to generate in each batch. gets changed in the loop

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
    rand_RA = np.random.uniform(RA_min, RA_max, batch_size)
    rand_sinDEC = np.random.uniform(np.sin(DEC_min*np.pi / 180), np.sin(DEC_max*np.pi / 180), batch_size)
    rand_DEC = np.arcsin(rand_sinDEC) * 180 / np.pi
    del rand_sinDEC

    # check if masked or not - this not_masked is a True / False array
    # *** this bit is not right if you rotate the field
    if field==1:
        rand_RA = rotate_field(rand_RA, -90)
    not_masked = fourmost_get_s8foot(rand_RA * u.deg, rand_DEC * u.deg, mask_path)[0]
    if field==1:
        rand_RA = rotate_field(rand_RA, 90)

    # drop masked points
    del_ind = np.where(not_masked==False)
    rand_RA = np.delete(rand_RA, del_ind)
    rand_DEC = np.delete(rand_DEC, del_ind)

    # do a small cut on the randoms due to the reduction in survey area but this not being reflected in the mangle mask
    if field == 2:
        del_ind = np.where(((rand_RA > 225) & (rand_DEC > 0)))
        rand_RA = np.delete(rand_RA, del_ind)
        rand_DEC = np.delete(rand_DEC, del_ind)

        del_ind = np.where(((rand_RA < 157) & (rand_DEC > 0)))
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
    healpy_n = 3 # choose healpy resolution (smaller number = less pixels that are larger)
    # healpy_n of 2 corresponds to a pixel with radius of ~ 0.7 deg and area of ~ 1.5 square deg
    nside = 12 * pow(healpy_n, 2)
    npixels = 12 * pow(nside, 2)

    # get pixel numbers (indices) of the data
    pixel_vals_in = hp.ang2pix(nside, df_in["RA"].values, df_in["DEC"].values, lonlat=True)
    pixel_vals_out = hp.ang2pix(nside, df_out["RA"].values, df_out["DEC"].values, lonlat=True)

    # function to generate a healpy map from an array of healpix indices.
    def gen_fast_map(pixel_vals, npixels):
        map_ = np.bincount(pixel_vals, minlength=npixels) # bincount tells you the number of objects in each pixel
        return map_

    # create healpy map of the data - the map is a 1d array where the index = pixel number,
    # and index value = points (i.e. galaxies) inside that pixel
    map_in = gen_fast_map(pixel_vals_in, npixels)
    map_out = gen_fast_map(pixel_vals_out, npixels)

    # get the completeness factor of each pixel
    map_completeness = np.divide(map_out, map_in, out=np.zeros_like(map_out, dtype=np.float64), where=map_in!=0)
    # print('Completeness map len = %s'%(len(map_completeness)), flush=True)

    # function to get the indices of the random points we need to delete, so as to downweight our randoms based on angular completeness
    # @njit
    def completeness_sample(i, map_completeness_i, del_ind, pixel_vals_rand):
        # if we have some completeness factor to consider for this pixel:
        if map_completeness_i != 0.:

            # find the index of the randoms we have generated that are in this pixel
            index = np.where(pixel_vals_rand == i)[0]
            n_rands = len(index) # how many randoms do we have here
            n_remove = int(n_rands*(1-map_completeness_i)) # how many we need to remove
            if n_remove != 0:
                # say we have n_remove = 5, just take the back 5 elements of the
                # index array, and these will be our removal indices (this choice is arbitrary,
                # it shouldnt matter which ever 5 we remove)
                del_ind = np.append(del_ind, np.array(index[-n_remove:]))

        return del_ind

    # get the pixel numbers that each random point lies in
    pixel_vals_rand = hp.ang2pix(nside, master_array_RA, master_array_DEC, lonlat=True)
    del_ind = np.empty(0)

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

# finally we add the arrays to a dataframe and then save to a fits file
df_rand = pd.DataFrame()
df_rand['RA'] = master_array_RA
df_rand['DEC'] = master_array_DEC

# # diagnostic plots
# if this_rank == root:
#     plt.figure(figsize=(14,7))
#     plt.title('rands', size=16)
#     plt.xlabel('RA', size=14)
#     plt.ylabel('DEC', size=14)
#     plt.xticks(size=14)
#     plt.yticks(size=14)
#     plt.scatter(df_rand['RA'].values, df_rand['DEC'].values, s=0.01)
#
#     plt.savefig('test_plot_2')

if this_rank == root:
    print('Saving random catalogues..', flush=True)


field_string = '_field' + str(field)

if catalogue == 'output':
    save_df_fits(df_rand, datapath + out_folder + 'randoms/table_' + tracer + field_string)
else:
    save_df_fits(df_rand, datapath + in_folder + 'randoms/table_' + tracer + field_string)

comm.Barrier()

# combine fields on root
if this_rank == root:

    for catalogue in catalogues:
        if catalogue == 'input':
            folder = in_folder
        else:
            folder = out_folder

        for tracer in tracers:
            # load field 1 and 2
            df_field_1 = read_to_df(datapath + folder + 'randoms/table_' + tracer + '_field1.fits')
            df_field_2 = read_to_df(datapath + folder + 'randoms/table_' + tracer + '_field2.fits')
            # combine
            RA_1 = df_field_1['RA'].values
            RA_2 = df_field_2['RA'].values
            DEC_1 = df_field_1['DEC'].values
            DEC_2 = df_field_2['DEC'].values
            RA = np.append(RA_1, RA_2)
            DEC = np.append(DEC_1, DEC_2)
            df = pd.DataFrame()
            df['RA'] = RA
            df['DEC'] = DEC
            save_df_fits(df, datapath + folder + 'randoms/table_' + tracer)
            # delete individual field files
            os.remove(datapath + folder + 'randoms/table_' + tracer + '_field1.fits')
            os.remove(datapath + folder + 'randoms/table_' + tracer + '_field2.fits')


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
