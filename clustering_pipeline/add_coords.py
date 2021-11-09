# code to add radial position 'comov_dist' and cartesian coords
# 'xpos', 'ypos', 'zpos' to the reduced catalogues that were created in TOPCAT

# imports
import numpy as np
import pandas as pd
import astropy.coordinates
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy import interpolate

from structures import catalogue_variables
from data_io import save_df_fits, read_to_df, load_object

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# run variables, change these below as necessary

# full path to the master variables for the catalogues to generate randoms for
cat_vars_path = 'catalogue_vars/2021_05_catalogue_vars'

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************






# load the cat_vars object
cat_vars = load_object(cat_vars_path)


for catalogue in cat_vars.catalogues:
    for tracer in cat_vars.tracers:

        print('Adding coords for %s %s'%(tracer, catalogue), flush=True)

        # load the data
        df_in = read_to_df(cat_vars.data_input + 'input_reduced_' + tracer + '.fits')
        df_out = read_to_df(cat_vars.data_output + 'output_reduced_' + tracer + '.fits')

        # *************************************

        # First we turn a redshift into a comoving distance.
        # Create a spline to do this for speed
        Z_max = np.amax(df_in['REDSHIFT_ESTIMATE'].values)*2
        z_arr = np.linspace(0, Z_max, 10000)
        comov_arr = [cat_vars.cosmo.comoving_distance(z).value for z in z_arr]
        spl_z_to_r = interpolate.interp1d(z_arr, comov_arr)

        df_in['comov_dist'] = spl_z_to_r(df_in['REDSHIFT_ESTIMATE'].values)
        df_out['comov_dist'] = spl_z_to_r(df_out['REDSHIFT_ESTIMATE'].values)
        
        if 'redshift_S' in df_in.columns:
            df_in['comov_dist_S'] = spl_z_to_r(df_in['redshift_S'].values)
        if 'redshift_S' in df_out.columns:
            df_out['comov_dist_S'] = spl_z_to_r(df_out['redshift_S'].values)


        # *************************************

        # next do the cartesian coords
        x,y,z = astropy.coordinates.spherical_to_cartesian(df_in['comov_dist'].values*u.Mpc, np.radians(df_in['DEC'].values), np.radians(df_in['RA'].values))
        # # shift the cartesian coords so none of them are negative / when painted to a box
        # # the bottom corner of the box will be at [0., 0., 0.]
        # x -= np.amin(x)
        # y -= np.amin(y)
        # z -= np.amin(z)
        # add as columns to the df
        df_in['xpos'] = x.value
        df_in['ypos'] = y.value
        df_in['zpos'] = z.value

        # and the redshift space ones
        if 'comov_dist_S' in df_in.columns:
            x,y,z = astropy.coordinates.spherical_to_cartesian(df_in['comov_dist_S'].values*u.Mpc, np.radians(df_in['DEC'].values), np.radians(df_in['RA'].values))
            df_in['xpos_S'] = x.value
            df_in['ypos_S'] = y.value
            df_in['zpos_S'] = z.value


        x,y,z = astropy.coordinates.spherical_to_cartesian(df_out['comov_dist'].values*u.Mpc, np.radians(df_out['DEC'].values), np.radians(df_out['RA'].values))
        # x -= np.amin(x)
        # y -= np.amin(y)
        # z -= np.amin(z)
        df_out['xpos'] = x.value
        df_out['ypos'] = y.value
        df_out['zpos'] = z.value

        # and the redshift space ones
        if 'comov_dist_S' in df_out.columns:
            x,y,z = astropy.coordinates.spherical_to_cartesian(df_out['comov_dist_S'].values*u.Mpc, np.radians(df_out['DEC'].values), np.radians(df_out['RA'].values))
            df_out['xpos_S'] = x.value
            df_out['ypos_S'] = y.value
            df_out['zpos_S'] = z.value

        # *************************************

        # now save over the originals
        save_df_fits(df_in, cat_vars.data_input + 'input_reduced_' + tracer)
        save_df_fits(df_out, cat_vars.data_output + 'output_reduced_' + tracer)

print('Adding coords complete.')
