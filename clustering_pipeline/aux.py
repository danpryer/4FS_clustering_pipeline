# python file for any auxiliary functions

import numpy as np
from numba import jit, njit
import camb  # model power spectra
from camb import model, initialpower
import scipy.interpolate as interpolate
from structures import CAMB_power


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



# function to generate a healpy map from an array of healpix indices.
def gen_fast_map(pixel_vals, npixels):
    map_ = np.bincount(pixel_vals, minlength=npixels) # bincount tells you the number of objects in each pixel
    return map_



# function to get the indices of the random points we need to delete,
# so as to downweight our randoms based on angular completeness
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


# get the actual pair counts in a bin from a DD/DR/RR_counts object produced by CorrFunc
def get_wp_paircounts(counts_obj):
    # The counts object has columns of data: 'r['rmin'], r['rmax'], r['rpavg'], r['pimax'], r['npairs'], r['weightavg']'
    # and the number of rows will be nbins*rpi_max
    # So we want to sum up all the rows for a given bin (as each bin will have rpimax rows)

    bin_lower = np.unique(counts_obj['rmin']) # create a unique array of the bin lower edge (could also use rmax here instead)

    actual_counts = np.zeros(len(bin_lower))

    # iterate over these unique bins
    for i in range(len(bin_lower)):
        # get the index of each row of the counts object which matches this bin
        ind = np.where(counts_obj['rmin']==bin_lower[i])

        # sum the npairs column of this filtered array, filtering by the rows that match ind,
        # and assign it to our actual counts array
        actual_counts[i] = np.sum(counts_obj['npairs'][ind])

    return actual_counts


# function to get camb power for a designated cosmology, k range and redshift
# returns a class, as defined above
def get_CAMB_power(redshift, cosmo, camb_h, kh_min=1e-4, kh_max=10, points=500, ns=0.9667, As= 2.142e-9):

    h = camb_h  # little h parameter

    print("Running CAMB for redshift = %s" % (np.round(redshift, 3)))

    # Now get matter power spectra and sigma8 at specified redshifts and with desired cosmology
    # pars = camb.CAMBparams(max_l = 2000, max_l_tensor = 1500, max_eta_k = 10000, max_eta_k_tensor = 3000,
    #                       num_nu_massless = 0.3040000E+01, Want_CMB_lensing = False)


    pars = camb.CAMBparams()
    pars.max_l = 2000
    pars.max_l_tensor = 1500
    pars.max_eta_k = 10000
    pars.max_eta_k_tensor = 3000
    pars.num_nu_massless = 0.3040000E+01
    pars.Want_CMB_lensing = False

    # pars.Reion.use_optical_depth=True
    # pars.Reion.redshift=11.
    # pars.Reion.optical_depth=0.9520000E-01
    # pars.Reion.delta_redshift=1.5
    # pars.Recomb.RECFAST_fudge = 0.1140000E+01

    pars.set_cosmology(
        H0=cosmo.H0.value * h, TCMB=0.27255000E+01, mnu=0.0, YHe = 0.2490000E+00,
        ombh2=cosmo.Ob0 * pow(h, 2),
        omch2=cosmo.Om0 * pow(h, 2) - cosmo.Ob0 * pow(h, 2),
    )

    pars.InitPower.set_params(ns=ns, As=As)

    # Note non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=[redshift], kmax=100.0)

    # Linear spectra
    print("Getting linear and nonlinear spectra...", end=" ")
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(
        minkh=kh_min, maxkh=kh_max, npoints=points
    )
    s8 = np.array(results.get_sigma8())

    # Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(
        minkh=kh_min, maxkh=kh_max, npoints=points
    )
    print("done.")
    # spline the linear and nonlinear power
    spl_CAMB_lin = interpolate.interp1d(kh, pk[0, :])
    spl_CAMB_nonlin = interpolate.interp1d(kh, pk_nonlin[0, :])

    # store in a class
    CAMB_power_class = CAMB_power(
        kh, redshift, pk[0, :], pk_nonlin[0, :], spl_CAMB_lin, spl_CAMB_nonlin
    )
    return CAMB_power_class
