from project_libs import *

# define cosmological parameters (from MDPL2 - https://www.cosmosim.org/cms/simulations/mdpl2/)
cosmo = FlatLambdaCDM(
H0=100.0 * u.km / u.s / u.Mpc,
Om0=0.307115 , Ob0=0.048206)
hubble_param = 0.6777

# filepaths and target type - to be set
global path_data, path_output, cat_group, gtype

# data variables
global N_gal, redshift_min, redshift_max, comov_min, comov_max, z_eff
global x_min, x_max, y_min, y_max, z_min, z_max # comoving dist
hist_bins = 100
global N_rand
rand_multi = 10 # multiplier for how many more randoms than data do you want

# box variables for 3D power spectrum measurement
global box_points # number of points on a side
global nx, ny, nz, N_cells
global Lx, Ly, Lz, vol, lx, ly, lz, x0, y0, z0
global k_nyqx, k_nyqy, k_nyqz

# grids
global datgrid, wingrid, weigrid

# variables for 3D power spectrum measurement
global khmin, khmax, nkbin, k_bins, kspec, nmodes, pk0_fid, pkerr, pk_est, pkspec
global pk_dm_lin, pkerr_dm_lin, pk_dm_nonlin, pkerr_dm_nonlin # inferred dark matter power spectrum

#some CAMB params and splines
CAMB_kmax = 2.0
CAMB_minkh = 1e-4 # max and min kh to get the spectra for, and number of points
CAMB_maxkh = 1
CAMB_npoints = 200
global spl_model_pk, spl_model_pk_nl

# healpix variables
global healpy_n, nside, npixels, pixel_area, pixel_radius

# EBV related variables for angular selection
global map_EBV, bins_EBV, hist_pix_density

# splines for number counts, densities
global spl_data_nr # number density of data w.r.t. comoving dist
global spl_data_nz # number density of data w.r.t. redshift
global spl_data_Nr # as above but number counts
global spl_data_Nz # as above but number counts
#splines for cosmology
global spl_r_to_z # comoving dist to redshift
global spl_z_to_r # redshift to comoving dist
global spl_hubble
global spl_growth_fact
global spl_growth_rate
global spl_bias
global spl_RSD_lin

# bias measurement from angular 2pcf
global bias_redshift_bins
global bias_measurements
# bias parameters, obtained from fitting to the above measurements
global bias_0, bias_1, bias_alpha, bias_sigma
