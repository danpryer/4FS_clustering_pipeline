from project_libs import *
import allvars as av
import data_io
import init
import power_spec as ps
import correlation_func as cf
import funcs
import window
import gridding
import cosmology as cmg

# defines the targets and overall catalogue group name
av.gtype = 'LyA'
av.cat_group = '2019_11_MultiDark_MD10'

data_io.set_filepaths() # paths for reading/writing data
init.initialise_target_vars() # sets variables specific to the type of galaxy/AGN

# set cube resolution for 3D P(k) measurement
av.box_points = 512

# load up the data frame and display first 5 rows. angular scatter plot
print('Reading in raw data... '); sys.stdout.flush()
df = data_io.read_fits(av.path_data + av.gtype + '.fits', print_df=True)

# do a data cut here if necessary (e.g. a redshift cut)
df = funcs.cut_raw_data(df)

# initialise some global variables based on the data (min/max redshift, number of targets etc)
init.initialise_data_vars(df)
print('There are %s targets in the %s data.'%(av.N_gal, av.gtype))

# histograms and splines for number counts and densities of the data
print('Initialising splines of radial data...', end = ' '); sys.stdout.flush()
init.initialise_radial_splines(df)
print('done.')

# calculate angular clustering of data in several redshift bins, and fit bias model
print('Calculating angular clustering and fitting bias model...', end = ' '); sys.stdout.flush()
cf.calculate_angular_clustering(df)
print('done.')

# initialise variables for the gridded boxes used in the 3D P(k) measurement
init.initialise_box_vars()

# use NGP to assign data to grid box
print('Assigning targets to grid...', end = ' '); sys.stdout.flush()
av.datgrid, edges = np.histogramdd(np.vstack([df['grid_x'].values, df['grid_y'].values, df['grid_z'].values]).transpose(),
                                        bins=(av.nx,av.ny,av.nz),range=((av.x_min ,av.x_max),
                                                               (av.y_min ,av.y_max),
                                                               (av.z_min ,av.z_max)))
print('done.')

# set up relevant variables etc. for calculating angular window function
# using healpy
print('Generating healpy maps for angular selection function...', end = ' '); sys.stdout.flush()
window.get_angular_selection(df)
print('done.')

# apply radial and angluar selection to the window function (with visuals)
print('Applying radial and angular selection functions to the window grid...', end = ' '); sys.stdout.flush()
window.create_window_grid()
print('done.')

# sample window function to test number counts/densities


# do an initial calculation of the power spectrum
print('Estimating the 3D power spectrum...', end = ' '); sys.stdout.flush()
ps.calculate_pkest()
print('done.')

# now some cosmology calcuations for D(a), f, bias etc.etc.
init.initialise_cosmology() # initialise splines for cosmology

# apply correction for aliasing caused by NGP assignment
print('Calculating NGP gridding correction...', end=' '); sys.stdout.flush()
ps.correct_pkest()
print ('done.')

# 'proper' calc of errors - use with 3 diff weights (non, fkp, new)
print('Estimating power spectrum error...', end=' '); sys.stdout.flush()
av.pkerr = ps.getpkerr()
print ('done.')

# scale calculated power spectrum accordingly
av.pk_dm_lin = np.zeros(len(av.k_bins))
av.pkerr_dm_lin = np.zeros(len(av.k_bins))
av.pk_dm_nonlin = np.zeros(len(av.k_bins))
av.pkerr_dm_nonlin = np.zeros(len(av.k_bins))
for i in range(len(av.k_bins)):
    av.pk_dm_lin[i] = av.pk_est[i] / ps.pk_scaling_amp(av.k_bins[i], 'linear')
    av.pkerr_dm_lin[i] = av.pkerr[i] / ps.pk_scaling_amp(av.k_bins[i], 'linear')

    av.pk_dm_nonlin[i] = av.pk_est[i] / ps.pk_scaling_amp(av.k_bins[i], 'nonlin')
    av.pkerr_dm_nonlin[i] = av.pkerr[i] / ps.pk_scaling_amp(av.k_bins[i], 'nonlin')


# get model P(k) from camb, lin and NL, at z=0.0
print('Getting model P(k) from CAMB...', end =' '); sys.stdout.flush()
ps.get_CAMB_power([0.0])
print('done.')
# k_test = np.logspace(np.log10(av.khmin), np.log10(av.khmax), 200)
# plt.figure(figsize=(15,10))
# plt.loglog(k_test, av.spl_model_pk(k_test))
# plt.loglog(k_test, av.spl_model_pk_nl(k_test))
# plt.show()


# plot results
ps.plot_pk_results()

# save data to a file
