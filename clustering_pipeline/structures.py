# file to store all the various classes used in the program

# imports
import numpy as np
import pandas as pd
import os
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# Define the class object to store the overall variables of a run as a pickle file.
# Includes all the key file paths as well as the cosmological parameters and the tracers
# to calculate for.
class catalogue_variables:
    def __init__(self, vars_name, root_path, in_folder, out_folder, cosmo, hubble_h,
                catalogues, tracers, rand_multi, raw_input_file, raw_output_file, fsky):

        # Overall 'master' name for this run of the pipleline
        self.vars_name = vars_name

        # Can optionally keep track of the names of the raw input and output
        # files from 4FS that were used to create the reduced input and output
        # catalogues that are fed into the pipeline
        self.raw_input_file = raw_input_file
        self.raw_output_file = raw_output_file

        # folder paths
        self.root_path = root_path # root directory
        self.data_input = root_path + 'data/' + in_folder # input catalogues
        self.data_output = root_path + 'data/' + out_folder # output catalogues
        self.mask_path = root_path + 'data/survey_masks/' # poly files for survey footprints
        self.randoms_input = self.data_input + 'randoms/'  # input random cats
        self.randoms_output = self.data_output + 'randoms/'  # output random cats
        self.results_input = root_path + 'results/' + in_folder  # store the cluster results for input
        self.results_output = root_path + 'results/' + out_folder # cluster results for output
        self.plots_validation = root_path + 'plots_validation/' + vars_name + '/'
        self.plots_results = root_path + 'plots_results/' + vars_name + '/'

        self.catalogues = catalogues
        self.tracers = tracers # list of the tracers to calculate for. eg ['BG', 'LRG', ..]

        self.cosmo = cosmo # astropy cosmo object containing the cosmological parameters
        self.hubble_h = hubble_h # hubble parameter (e.g. 0.677)

        self.rand_multi = rand_multi # multiplier for how many more random points than data points you want

        self.fsky = fsky # fraction of the sky covered by the survey - to be properly calculated and update later

    def make_paths(self):
        os.makedirs('catalogue_vars/', exist_ok=True) # succeeds even if directory exists.
        os.makedirs(self.root_path, exist_ok=True)
        os.makedirs(self.data_input, exist_ok=True)
        os.makedirs(self.randoms_input, exist_ok=True)
        os.makedirs(self.data_output, exist_ok=True)
        os.makedirs(self.randoms_output, exist_ok=True)
        os.makedirs(self.mask_path, exist_ok=True)
        os.makedirs(self.results_input, exist_ok=True)
        os.makedirs(self.results_output, exist_ok=True)
        os.makedirs(self.plots_validation, exist_ok=True)
        os.makedirs(self.plots_results, exist_ok=True)

    # (*** TODO: add functions to load specific input and output data / random files?)


# ******************************************************************************
# ******************************************************************************
# ******************************************************************************
# Classes below to store results of the clustering statistics produced by
# TreeCorr and CorrFunc

# Store the results of a measurement
class results_2pcf:
    def __init__(self, bin_mid, result, var, sig, DD_counts=None, RR_counts=None, DR_counts=None):
        self.bin_mid = bin_mid
        self.result = result
        self.var = var
        self.sig = sig
        self.DD_counts = DD_counts
        self.RR_counts = RR_counts
        self.DR_counts = DR_counts

# Store the results of a measurement, where the cov has been calculated via jackknife
class results_2pcf_jackknife:
    def __init__(self, bin_mid, result, cov, DD_counts=None, RR_counts=None, DR_counts=None):
        self.bin_mid = bin_mid
        self.result = result
        self.cov = cov # covariance matrix
        self.err = np.sqrt(np.diagonal(cov))
        self.DD_counts = DD_counts
        self.RR_counts = RR_counts
        self.DR_counts = DR_counts


# Store the full results from a jackknife 2pcf measurement of field 1
# and 2 in the 4MOST catalogues
class full_4FS_results:
    def __init__(self, bin_mid, res_1, res_2, res_avg, err_1, err_2, err_avg, cov1=None,
                    cov2=None, cov_avg=None, DD1=None, DD2=None, RR1=None, RR2=None,
                    DR1=None, DR2=None, res_1_uncorr=None, res_2_uncorr=None, FC1=None, FC2=None):

        # bin mid of measurement, and measurement results
        self.bin_mid = bin_mid
        self.res_1 = res_1
        self.res_2 = res_2
        self.res_avg = res_avg

        # error and covariance
        self.err_1 = err_1
        self.err_2 = err_2
        self.err_avg = err_avg
        self.cov1 = cov1
        self.cov2 = cov2
        self.cov_avg=cov_avg

        # pair counts
        self.DD1 = DD1
        self.DD2 = DD2
        self.RR1 = RR1
        self.RR2 = RR2
        self.DR1 = DR1
        self.DR2 = DR2

        # fibre collision correction for w(theta) output catalogues
        # and uncorrected results
        self.res_1_uncorr = res_1_uncorr
        self.res_2_uncorr = res_2_uncorr
        self.fibre_1_corr = FC1
        self.fibre_2_corr = FC2

    # function to print the results. Add to a dataframe and display
    def print_results(self):

        df = pd.DataFrame()

        df['bin_mid'] = self.bin_mid
        df['res_1'] = self.res_1
        df['err_1'] = self.err_1
        df['res_2'] = self.res_2
        df['err_2'] = self.err_2
        df['res_avg'] = self.res_avg
        df['err_avg'] = self.err_avg

        if self.res_1_uncorr is not None:
            df['res_1_uncorr'] = self.res_1_uncorr
            df['fibre_1_corr'] = self.fibre_1_corr
            df['res_2_uncorr'] = self.res_2_uncorr
            df['fibre_2_corr'] = self.fibre_2_corr

        df['DD1'] = self.DD1
        df['DD2'] = self.DD2
        df['RR1'] = self.RR1
        df['RR2'] = self.RR2
        df['DR1'] = self.DR1
        df['DR2'] = self.DR2

        display(df)
        del df

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# Classes related to power power spectrum

# # class to store the output of a P(k) measurement from nbodykit
# # Store the results of a measurement
# class results_powerspec:
#     def __init__(self, bin_mid, Pk0, error, Nmodes, Ndata, Nrand,
#                         dk, Pk_fid, Nmesh, window, interlaced, tracer,
#                         catalogue, compensated, Pk2=None, Pk4=None,
#                         Nbins_Z=None, Z_edges=None, Z_bins_mid=None):
#
#         # main results
#         self.bin_mid = bin_mid
#         self.Pk0 = Pk0 # monopole
#         self.Pk0_err = error
#         self.Nmodes = Nmodes
#         self.Ndata = Ndata
#         self.Nrand = Nrand
#
#         # higher poles
#         self.Pk2 = Pk2
#         self.Pk4 = Pk4
#
#         # bin params
#         self.dk = dk
#         self.Nbins = len(self.bin_mid)
#         self.Pk_fid = Pk_fid
#
#         # z-bin parameters
#         self.Nbins_Z = Nbins_Z
#         self.Z_edges = Z_edges
#         self.Z_bins_mid = Z_bins_mid
#
#         # run parameters
#         self.tracer = tracer
#         self.catalogue = catalogue
#         self.Nmesh = Nmesh
#         self.window = window
#         self.interlaced = interlaced
#         self.compensated = compensated
#

# Container class to store results of a powerspectrum measurement. Should contain
# the variables that apply to the power spectrum measurement and data as a whole.
# Will be used to store instances of the powerspec_measurment class, which contain specifics
# of an individual measurement. Can store multiple instances
# which is handy if e.g. doing a measurement in redshift bins which have differing
# kbins, redshift ranges, pk_fid etc. These instances are added using:
# 'setattr(<class name>, <attr name>, <value or object>)'
# so for measurements in redshift bin you would add on instances of the
# pioweerspec_measurement class with attribute name bin1, bin2, bin3 etc.
class results_powerspec:
    def __init__(self, tracer, catalogue, window, Nmesh, interlaced, compensated,
                    Ndata, Nrand, Nbins_Z=None, Z_edges=None, Z_bins_mid=None):

        self.tracer = tracer             # eg. 'LRG'
        self.catalogue = catalogue       # 'input' or 'output'
        self.window = window             # mass assignment scheme eg. 'tsc'
        self.Nmesh = Nmesh               # 3D mesh resolution for FFT
        self.interlaced = interlaced     # True or False
        self.compensated = compensated   # True or False

        self.Ndata = Ndata               # Number of data points in full sample
        self.Nrand = Nrand               # Number of random points in full sample

        self.Nbins_Z = Nbins_Z           # Number of redshift bins to do the measurements over,
                                         # will just be = 1 for a full survey measurement.
        self.Z_edges = Z_edges           # Redshift bin edges.
        self.Z_bins_mid = Z_bins_mid     # Mid points of the redshift bin(s).


# class to store the powerspec_measurements. Actual measurements of the P_l(k)
# multipoles and their errors will be dynamically added to the class, to allow
# for arbitrary order of multipole to be added and stored. can be added using
# 'setattr(<class name>, <attr name>, <value or object>)' and should be named
# Pk0, Pk2, Pk4, Pk0_err, Pk2_err, Pk4_err etc..
class powerspec_measurement:
    def __init__(self, Pk_fid, kmin, dk, multipoles, Nmodes, Ndata, Nrand, Z_min, Z_max, Z_mid):

        self.Pk_fid = Pk_fid             # fiducial P(k) for FKP weight

        self.bin_mid = None              # mid point of the k bins for the measurements, initialised after measurement
        self.kmin = kmin                 # minimum k value
        self.dk = dk                     # linear bin spacing
        self.Nbins = None                # number of k bins, initialised after measurement

        self.multipoles = multipoles     # multipoles contained in the measurement, e.g [0,2,4]

        self.Nmodes = Nmodes             # number of k modes
        self.Ndata = Ndata               # number of data points in this measurement
        self.Nrand = Nrand               # number of random points in this measurement

        self.Z_min = Z_min               # minimum redshift of the measurement
        self.Z_max = Z_max               # maximum redshift of the measurement
        self.Z_mid = Z_mid               # mid point of the Z bins *** TODO: note this is not a good measure,
                                         # should calculate the effective redshift or average, based
                                         # on numberdensities and growth factors etc


# class to store camb lin and nl model power
class CAMB_power:
    def __init__(
        self, k_range, redshift, lin_power, nonlin_power, spl_CAMB_lin, spl_CAMB_nonlin
    ):
        self.k_range = k_range
        self.points = len(k_range)
        self.k_min = min(self.k_range)
        self.k_min = max(self.k_range)

        self.redshift = redshift

        self.lin_power = lin_power
        self.nonlin_power = nonlin_power

        self.spl_pk_lin = spl_CAMB_lin
        self.spl_pk_nonlin = spl_CAMB_nonlin
