from project_libs import *
import allvars as av
import funcs
import cosmology as cmg

# class to store the properties of the data. takes a pandas data frame.
class data_vars:
    def __init__(self, df):
        self.N_gal = len(df)
        self.Z_min = min(df['Z'])
        self.Z_max = max(df['Z'])
        self.Z_eff = round(df['Z'].values.mean(), 3)
        self.comov_min = min(df['comov_dist'])
        self.comov_max = max(df['comov_dist'])

        self.x_min = min(df['grid_x'])
        self.x_max = max(df['grid_x'])
        self.y_min = min(df['grid_y'])
        self.y_max = max(df['grid_y'])
        self.z_min = min(df['grid_z'])
        self.z_max = max(df['grid_z'])


# variables and splines based on the data
def initialise_data_vars(df):
    # total number of gals
    av.N_gal = len(df)
    # min, max, avg redshift
    av.redshift_min = min(df['Z'])
    av.redshift_max = max(df['Z'])
    av.z_eff = round(df['Z'].values.mean(), 3)
    # comoving dist min and max (radial, x, y, z)
    av.comov_min = min(df['comov_dist'])
    av.comov_max = max(df['comov_dist'])
    av.x_min = min(df['grid_x'])
    av.x_max = max(df['grid_x'])
    av.y_min = min(df['grid_y'])
    av.y_max = max(df['grid_y'])
    av.z_min = min(df['grid_z'])
    av.z_max = max(df['grid_z'])
    # splines to convert from comoving dist to redshift and vice versa
    z_range = np.linspace(0, av.redshift_max*1.1, 2000)
    r_range = [av.cosmo.comoving_distance(x).value for x in z_range]
    av.spl_r_to_z = funcs.create_spline(r_range, z_range)
    av.spl_z_to_r = funcs.create_spline(z_range, r_range)


# set up the box dimensions for the data and window grid
def initialise_box_vars():
    # number of points on a side, and total number of cells
    av.nx = av.box_points
    av.ny = av.box_points
    av.nz = av.box_points
    av.N_cells = av.nx * av.ny * av.nz
    # box dimensions
    av.Lx = av.x_max - av.x_min
    av.Ly = av.y_max - av.y_min
    av.Lz = av.z_max - av.z_min
    av.vol = av.Lx * av.Ly * av.Lz
    # cell dimensions
    av.lx = av.Lx / av.nx
    av.ly = av.Ly / av.ny
    av.lz = av.Lz / av.nz
    # origin
    av.x0 = 0.
    av.y0 = 0.
    av.z0 = 0.
    # nyquist freq
    av.k_nyqx = np.pi / av.lx
    av.k_nyqy = np.pi / av.ly
    av.k_nyqz = np.pi / av.lz




# initialise variables that are specific to the target type
def initialise_target_vars():
    if(av.gtype == 'AGN_IR'):
        av.khmin = 0.05
        av.khmax = 0.15
        av.nkbin = 20
        av.healpy_n = 5

    if(av.gtype == 'AGN_WIDE'):
        av.khmin = 0.02
        av.khmax = 0.2
        av.nkbin = 20
        av.healpy_n = 5

    if(av.gtype == 'BG'):
        av.khmin = 0.1
        av.khmax = 0.6
        av.nkbin = 20
        av.healpy_n = 5

    if(av.gtype == 'ELG'):
        av.khmin = 0.1
        av.khmax = 0.4
        av.nkbin = 20
        av.healpy_n = 5

    if(av.gtype == 'LRG'):
        av.khmin = 0.08
        av.khmax = 0.35
        av.nkbin = 20
        av.healpy_n = 5

    if(av.gtype == 'LyA'):
        av.khmin = 0.04
        av.khmax = 0.2
        av.nkbin = 20
        av.healpy_n = 5

    if (av.gtype == 'QSO'):
        av.khmin = 0.009
        av.khmax = 0.25
        av.nkbin = 20
        av.healpy_n = 5

    # check that CAMB is calculating model pk over appropriate k range given the above
    if ((av.CAMB_maxkh < av.khmax) or (av.CAMB_minkh > av.khmin)):
        print('Trying to calculate P(k) for a given k outside the range specified for CAMB.')
        print('Please extend CAMB_minkh or CAMB_maxkh.')
        print('Ending run.')
        sys.exit(0)

    initialise_healpy_vars()

    if (av.gtype == ('LRG' or 'ELG' or 'BG')):
        av.bias_0 = 0.79
        av.bias_1 = 0.57
        av.bias_alpha = 2.23
        av.bias_sigma = 3
    else:
        av.bias_0 = 0.742
        av.bias_1 = 0
        av.bias_alpha = 1.63
        av.bias_sigma = 3





# define some healpy variables based on healpy_n value
def initialise_healpy_vars():
    av.nside = 12 * pow(av.healpy_n, 2)
    av.npixels = 12 * pow(av.nside, 2)
    av.pixel_area = hp.nside2pixarea(av.nside, degrees = True)
    av.pixel_radius = pow(av.pixel_area / np.pi, 0.5)





# initialise splines for radial number densities and counts
def initialise_radial_splines(df):

    # radial redshift counts
    hist_redshift, bins_redshift = np.histogram(df['Z'], bins=av.hist_bins)
    bins_redshift_mid = (bins_redshift[1:] + bins_redshift[:-1]) * .5

    # radial comoving counts
    hist_comov, bins_comov = np.histogram(df['comov_dist'], bins=av.hist_bins)
    bins_comov_mid = (bins_comov[1:] + bins_comov[:-1]) * .5

    # calc n(z)
    hist_nz = np.zeros(len(hist_redshift))
    bin_vol_redshift = np.zeros(len(hist_nz))
    for i in range(len(hist_nz)):
        bin_vol_redshift[i] = (4.0/3)*np.pi*((bins_redshift[i+1]**3 - bins_redshift[i]**3))
        hist_nz[i] = hist_redshift[i] / bin_vol_redshift[i]

    # calc n(r)
    hist_nr = np.zeros(len(hist_comov))
    bin_vol_comov = np.zeros(len(hist_nr))
    for i in range(len(hist_nr)):
        bin_vol_comov[i] = (4.0/3)*np.pi*((bins_comov[i+1]**3 - bins_comov[i]**3))
        hist_nr[i] = hist_comov[i] / bin_vol_comov[i]

    # initialise the splines
    av.spl_data_nr = funcs.create_spline(bins_comov_mid, hist_nr)
    av.spl_data_nz = funcs.create_spline(bins_redshift_mid, hist_nz)
    av.spl_data_Nr = funcs.create_spline(bins_comov_mid, hist_comov)
    av.spl_data_Nz = funcs.create_spline(bins_redshift_mid, hist_redshift)

    # plots to check radial distributions and that splines are working as expected
    plt.figure(figsize=(20,13))

    plt.subplot(2,2,1) # N(z)
    plt.xlabel('Redshift', fontsize = 14)
    plt.ylabel('Counts', fontsize = 14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlim(av.redshift_min, av.redshift_max)
    plt.title('Radial distribution of data N(z)', fontsize= 14)
    plt.plot(bins_redshift_mid, hist_redshift, color='black', label='number counts')
    plt.plot(bins_redshift_mid, av.spl_data_Nz(bins_redshift_mid), color='red', label='spline fit',  linestyle='--')
    plt.legend()

    plt.subplot(2,2,2) # n(z)
    plt.xlabel('Redshift', fontsize = 14)
    plt.ylabel('n(z)', fontsize = 14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlim(av.redshift_min, av.redshift_max)
    plt.title('Number density n(z)', fontsize= 14)
    plt.semilogy(bins_redshift_mid, hist_nz, color='black', label='number density')
    plt.semilogy(bins_redshift_mid, av.spl_data_nz(bins_redshift_mid), color='red', label='spline fit', linestyle='--')
    plt.legend()

    plt.subplot(2,2,3) # N(r)
    plt.xlabel('comoving dist', fontsize = 14)
    plt.ylabel('Counts', fontsize = 14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlim(av.comov_min, av.comov_max)
    plt.title('Radial distribution of data N(r)', fontsize= 14)
    plt.plot(bins_comov_mid, hist_comov, color='black', label='number counts')
    plt.plot(bins_comov_mid, av.spl_data_Nr(bins_comov_mid), color='red', label='spline fit',  linestyle='--')
    plt.legend()

    plt.subplot(2,2,4) # n(r)
    plt.xlabel('Redshift', fontsize = 14)
    plt.ylabel('n(r) $[h^3 Mpc^{-3}]$', fontsize = 16)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlim(av.redshift_min, av.redshift_max)
    plt.title('Number density n(r)', fontsize= 14)
    plt.semilogy(bins_redshift_mid, hist_nr, color='black', label='number density')
    plt.semilogy(bins_redshift_mid, av.spl_data_nr(bins_comov_mid), color='red', label='spline fit', linestyle='--')
    plt.legend()

    plt.savefig(av.path_output + 'data_radial_plots.pdf', bbox_inches='tight')
    plt.close()


# initialise splines for H(z), D(z), f(z), and bias/RSD factors
def initialise_cosmology():

    redshift_range = np.linspace(av.redshift_min, av.redshift_max, 500)

    # spline H(z)
    av.spl_hubble = funcs.create_spline(redshift_range, cmg.hubble_param(redshift_range, av.cosmo.H0.value, av.cosmo.Om0))

    # spline linear growth factor D(z)
    growth_array = [cmg.growth_factor(x, av.cosmo.H0.value, av.cosmo.Om0) for x in redshift_range]
    av.spl_growth_fact = funcs.create_spline(redshift_range, growth_array)

    # spline growth rate f(a(z))
    scalefactor_range = [pow(1 + x, -1) for x in redshift_range]
    log_scalefactor = np.log(scalefactor_range)
    log_growth_factor = [np.log(av.spl_growth_fact(x)) for x in redshift_range]
    growth_rate_array = np.diff(log_growth_factor) / np.diff(log_scalefactor)
    r_minus_1 = redshift_range[:-1]
    av.spl_growth_rate = funcs.create_spline(r_minus_1, growth_rate_array)

    # spline bias
    av.spl_bias = funcs.create_spline(redshift_range, cmg.bias_model(redshift_range))

    # linear RSD
    av.spl_RSD_lin = funcs.create_spline(redshift_range, cmg.RSD_linear(redshift_range))
