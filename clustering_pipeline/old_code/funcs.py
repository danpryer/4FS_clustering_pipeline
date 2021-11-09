# General functions in here

from project_libs import *
import allvars as av




# function to create splines
def create_spline(x, fx):
    return interpolate.interp1d(x, fx, kind='cubic', fill_value="extrapolate")


# weight function for FKP weights
def weight_func(z):
    return pow(1 + (av.pk0_fid * av.spl_data_nz(z)), -1)


# function to slice dataframe depending on target type
def cut_raw_data(df):
    if (av.gtype == 'AGN_IR'):
        print('Redshift cut applied to data.')
        df = df[df.Z <= 3.2]
        df = df[df.Z >= 1.0]

    if (av.gtype == 'AGN_WIDE'):
        print('Redshift cut applied to data.')
        df = df[df.Z <= 3.2]

    if (av.gtype == 'LyA'):
        print('Redshift cut applied to data.')
        df = df[df.Z <= 3.2]

    return df




# function to generate a healpy map from an array of healpix indices.
# the map can then be displayed with the hp.visfunc.mollview(map) function
def gen_fast_map(ip_, nside):
    npixel  = hp.nside2npix(nside)
    map_ = np.bincount(ip_,minlength=npixel)
    return map_
