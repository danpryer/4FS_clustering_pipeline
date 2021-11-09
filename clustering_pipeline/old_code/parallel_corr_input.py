# main module to run for calculating correlation functions in parallel, for input catalogues

# imports here
from imports import *
from data_io import *
from twopointcorr import *

# start MPI
comm = MPI.COMM_WORLD
root = 0
ntasks = MPI.COMM_WORLD.Get_size()
this_rank = comm.rank

# if on the root note then start clock
if comm.rank == root:
    print('Sys platform = %s.'%(sys.platform))
    start = time.time()

# On linux, getrusage returns in kiB
# On Mac systems, getrusage returns in B
scale = 1.0
if 'linux' in sys.platform:
    scale = 2**10


# some run parameters
nthreads = 28 # cpus per task
njack = 10
verbose = 2

# bin parameters
nbins = 20
# angular
deg_min = 0.01
deg_max = 10
# 3d
r_min = 0.5
r_max = 50
# projected
r_proj_min = 0.1
r_proj_max = 20
rpimax = 10

# catalogues = ['output', 'input']
tracers = ['BG', 'LRG', 'QSO', 'LyA']
catalogue = 'input'

# run on 4 tasks so..
tracer = tracers[this_rank]

datapath = '/cosma6/data/dp004/dc-prye1/4fs_clustering_project/data/'
resultspath = '/cosma6/data/dp004/dc-prye1/4fs_clustering_project/results/'


print('Cpu %s running on the %s tracers in the %s catalogue'%(this_rank, tracer, catalogue))


# start by loading the data and random catalogues
# target = '2020_06_Input/input_reduced_' + tracer + '.fits'
# target_rand = '2020_06_Input/randoms/table_' + tracer + '.fits'
target = '2021_05_Input/input_reduced_' + tracer + '.fits'
target_rand = '2021_05_Input/randoms/table_' + tracer + '.fits'


df_data = read_to_df(datapath + target)
df_rand = read_to_df(datapath + target_rand)
N_data = len(df_data)
N_rand = len(df_rand)
print('There are %s data points and %s random points in this catalogue.'%(N_data, N_rand), flush=True)
sys.stdout.flush()

comm.Barrier()

# To save computing time, we don't want to calculate the correlation function "across fields" so we break the data/randoms down into two fields
df_data_1 = df_data[(df_data.RA >= 275) | (df_data.RA <= 125)].copy()
df_data_2 = df_data[(df_data.RA <= 275) & (df_data.RA >= 125)].copy()
df_rand_1 = df_rand[(df_rand.RA >= 275) | (df_rand.RA <= 125)].copy()
df_rand_2 = df_rand[(df_rand.RA <= 275) & (df_rand.RA >= 125)].copy()

# now we will rotate field 1 so it doesnt sit across the 360 deg -> 0 deg boundary
@jit
def rotate_field(RA_vals, rotation=90):
    RA_vals += rotation
    for i in range(len(RA_vals)):
        if RA_vals[i] > 360:
            RA_vals[i] -= 360
    return RA_vals
df_data_1['RA'] = rotate_field(df_data_1['RA'].values)
df_rand_1['RA'] = rotate_field(df_rand_1['RA'].values)


arr_fc_1 = [1] # on input catalogue dont want to do any weighted correction
arr_fc_2 = [1]


# now using the jackknife method, calculate the angular, 3d and projected clustering of both fields
if this_rank==root:
    print('Calculating w(theta)...')
    sys.stdout.flush()

results_wtheta_1 = wtheta_jackknife(nbins, deg_min, deg_max, df_data_1['RA'].values, df_data_1['DEC'].values,
                         df_rand_1['RA'].values, df_rand_1['DEC'].values, nthreads, njack, pair_weights=arr_fc_1, verbose=verbose)
results_wtheta_2 = wtheta_jackknife(nbins, deg_min, deg_max, df_data_2['RA'].values, df_data_2['DEC'].values,
                         df_rand_2['RA'].values, df_rand_2['DEC'].values, nthreads, njack, pair_weights=arr_fc_2, verbose=verbose)

print('Calculating Xi(r)...')
sys.stdout.flush()
results_xi_1 = xi_jackknife(nbins, r_min, r_max, df_data_1['xpos'].values, df_data_1['ypos'].values, df_data_1['zpos'].values,
                         df_rand_1['xpos'].values, df_rand_1['ypos'].values, df_rand_1['zpos'].values, nthreads, njack, verbose=verbose)
results_xi_2 = xi_jackknife(nbins, r_min, r_max, df_data_2['xpos'].values, df_data_2['ypos'].values, df_data_2['zpos'].values,
                         df_rand_2['xpos'].values, df_rand_2['ypos'].values, df_rand_2['zpos'].values, nthreads, njack, verbose=verbose)

print('Calculating w_p(r_p)...')
sys.stdout.flush()
results_wp_1 = wp_jackknife(nbins, r_proj_min, r_proj_max, rpimax, df_data_1['xpos'].values, df_data_1['ypos'].values, df_data_1['zpos'].values,
                         df_rand_1['xpos'].values, df_rand_1['ypos'].values, df_rand_1['zpos'].values, nthreads, njack, verbose=True)
results_wp_2 = wp_jackknife(nbins, r_min, r_max, rpimax, df_data_2['xpos'].values, df_data_2['ypos'].values, df_data_2['zpos'].values,
                         df_rand_2['xpos'].values, df_rand_2['ypos'].values, df_rand_2['zpos'].values, nthreads, njack, verbose=True)

# now average the results (inverse variance weighting) over the two fields:
# for w(theta)
wtheta_avg_res = np.zeros(nbins)
wtheta_avg_var = np.zeros(nbins)
wtheta_avg_err = np.zeros(nbins)
for i in range(nbins):
    wtheta_avg_var[i] = 1 / ((1/np.diag(results_wtheta_1.cov)[i])+(1/np.diag(results_wtheta_2.cov)[i]))
    wtheta_avg_res[i] = ((results_wtheta_1.result[i]/ np.diag(results_wtheta_1.cov)[i])+(results_wtheta_2.result[i]/ np.diag(results_wtheta_2.cov)[i])) * wtheta_avg_var[i]
wtheta_avg_err = np.sqrt(wtheta_avg_var)
wtheta_avg_cov = 0.5*(results_wtheta_1.cov + results_wtheta_2.cov)

# for xi(r)
xi_avg_res = np.zeros(nbins)
xi_avg_var = np.zeros(nbins)
xi_avg_err = np.zeros(nbins)
for i in range(nbins):
    xi_avg_var[i] = 1 / ((1/np.diag(results_xi_1.cov)[i])+(1/np.diag(results_xi_2.cov)[i]))
    xi_avg_res[i] = ((results_xi_1.result[i]/ np.diag(results_xi_1.cov)[i])+(results_xi_2.result[i]/ np.diag(results_xi_2.cov)[i])) * xi_avg_var[i]
xi_avg_err = np.sqrt(xi_avg_var)
xi_avg_cov = 0.5*(results_xi_1.cov + results_xi_2.cov)

# for w_p(r_p)
wp_avg_res = np.zeros(nbins)
wp_avg_var = np.zeros(nbins)
wp_avg_err = np.zeros(nbins)
for i in range(nbins):
    wp_avg_var[i] = 1 / ((1/np.diag(results_wp_1.cov)[i])+(1/np.diag(results_wp_2.cov)[i]))
    wp_avg_res[i] = ((results_wp_1.result[i]/ np.diag(results_wp_1.cov)[i])+(results_wp_2.result[i]/ np.diag(results_wp_2.cov)[i])) * wp_avg_var[i]
wp_avg_err = np.sqrt(wp_avg_var)
wp_avg_cov = 0.5*(results_wp_1.cov + results_wp_2.cov)

# save the results to fits files for the separate fields and the average of them
if this_rank == root:
    print('Saving results...')
    sys.stdout.flush()
df_wtheta = pd.DataFrame()
df_wtheta['theta_mid'] = results_wtheta_1.bin_mid

df_wtheta['res_field_1'] = results_wtheta_1.result
df_wtheta['err_field_1'] = results_wtheta_1.err
df_wtheta['DD_1'] = results_wtheta_1.DD_counts
df_wtheta['RR_1'] = results_wtheta_1.RR_counts
df_wtheta['DR_1'] = results_wtheta_1.DR_counts


df_wtheta['res_field_2'] = results_wtheta_2.result
df_wtheta['err_field_2'] = results_wtheta_2.err
df_wtheta['DD_2'] = results_wtheta_2.DD_counts
df_wtheta['RR_2'] = results_wtheta_2.RR_counts
df_wtheta['DR_2'] = results_wtheta_2.DR_counts

df_wtheta['res_avg'] = wtheta_avg_res
df_wtheta['err_avg'] = wtheta_avg_err



df_xi = pd.DataFrame()
df_xi['r_mid'] = results_xi_1.bin_mid
df_xi['res_field_1'] = results_xi_1.result
df_xi['err_field_1'] = results_xi_1.err
df_xi['res_field_2'] = results_xi_2.result
df_xi['err_field_2'] = results_xi_2.err
df_xi['res_avg'] = xi_avg_res
df_xi['err_avg'] = xi_avg_err

df_wp = pd.DataFrame()
df_wp['r_mid'] = results_wp_1.bin_mid
df_wp['res_field_1'] = results_wp_1.result
df_wp['err_field_1'] = results_wp_1.err
df_wp['res_field_2'] = results_wp_2.result
df_wp['err_field_2'] = results_wp_2.err
df_wp['res_avg'] = wp_avg_res
df_wp['err_avg'] = wp_avg_err

resultspath2 = resultspath + catalogue + '/results_' + tracer
save_df_fits(df_wtheta, resultspath2 + '_wtheta')
np.save(resultspath + catalogue + '/covariance_' + tracer + '_wtheta', wtheta_avg_cov)
save_df_fits(df_xi, resultspath2 + '_xi')
np.save(resultspath + catalogue + '/covariance_' + tracer + '_xi', xi_avg_cov)
save_df_fits(df_wp, resultspath2 + '_wp')
np.save(resultspath + catalogue + '/covariance_' + tracer + '_wp', wp_avg_cov)
