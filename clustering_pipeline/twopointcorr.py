# module for things related to calculating correlation functions

# from imports import *
import numpy as np
import pandas as pd
import astropy.coordinates
import treecorr
from Corrfunc.theory import DDrppi
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from structures import results_2pcf, results_2pcf_jackknife, full_4FS_results
from aux import get_wp_paircounts



# simple function to calculate the angular galaxy-galaxy clustering using treecorr, using the LS estimator
def wtheta_treecorr(nbins, degmin, degmax, data_RA, data_DEC, rand_RA, rand_DEC, nthreads, weights=None, pair_weights=[1], cutoff=100, verbose=1):

    data_cat = treecorr.Catalog(ra=data_RA, dec=data_DEC, ra_units='deg', dec_units='deg', w = weights)
    rand_cat = treecorr.Catalog(ra=rand_RA, dec=rand_DEC, ra_units='deg', dec_units='deg', w = weights)

    # config = {'min_sep':degmin, 'max_sep':degmax}

    dd = treecorr.NNCorrelation(min_sep=degmin, max_sep=degmax, nbins=nbins, sep_units='degrees', verbose=verbose, num_threads=nthreads)
    dd.process(data_cat)
    DD_counts = dd.npairs

    # get the bin's mean separation
    # theta = np.exp(dd.meanlogr)
    theta = dd.meanr

    # multiply the dd counts contribution (dd.weight) by the pair weights, to account for fibre collisions, below a certain scale (cutoff, in deg)
    if len(pair_weights) != 1:
        pair_weights = np.where(theta <= cutoff, pair_weights, 1)
        # print(pair_weights)
        dd.weight = np.multiply(dd.weight, pair_weights)

    rr = treecorr.NNCorrelation(min_sep=degmin, max_sep=degmax, nbins=nbins, sep_units='degrees', verbose=verbose, num_threads=nthreads)
    rr.process(rand_cat)
    RR_counts = rr.npairs

    dr = treecorr.NNCorrelation(min_sep=degmin, max_sep=degmax, nbins=nbins, sep_units='degrees', verbose=verbose, num_threads=nthreads)
    dr.process(data_cat, rand_cat)
    DR_counts = dr.npairs

    wtheta, var = dd.calculateXi(rr=rr, dr=dr)

    sig = np.sqrt(var)


    results = results_2pcf(theta, wtheta, var, sig, DD_counts, RR_counts, DR_counts)

    return results





# simple function to calculate the 3d galaxy-galaxy clustering using treecorr, using the LS estimator
def xi_treecorr(nbins, r_min, r_max, data_x, data_y, data_z, rand_x, rand_y, rand_z, nthreads, verbose=1):

    data_cat = treecorr.Catalog(x=data_x, y=data_y, z=data_z)#, ra_units='deg', dec_units='deg')
    rand_cat = treecorr.Catalog(x=rand_x, y=rand_y, z=rand_z)#, ra_units='deg', dec_units='deg')

    dd = treecorr.NNCorrelation(min_sep=r_min, max_sep=r_max, nbins=nbins, verbose=verbose, num_threads=nthreads)
    dd.process(data_cat)
    DD_counts = dd.npairs

    rr = treecorr.NNCorrelation(min_sep=r_min, max_sep=r_max, nbins=nbins, verbose=verbose, num_threads=nthreads)
    rr.process(rand_cat)
    RR_counts = rr.npairs

    dr = treecorr.NNCorrelation(min_sep=r_min, max_sep=r_max, nbins=nbins, verbose=verbose, num_threads=nthreads)
    dr.process(data_cat, rand_cat)
    DR_counts = dr.npairs

    xi, var = dd.calculateXi(rr, dr)

    sig = np.sqrt(var)
    r = dd.meanr

    results = results_2pcf(r, xi, var, sig, DD_counts, RR_counts, DR_counts)

    return results


# function to use the corrfunc package to get the projected angular clustering wp
def wp_corrfunc(nbins, r_min, r_max, rpimax, data_x, data_y, data_z, rand_x, rand_y, rand_z, nthreads, verbose=True):

    # Setup the bins
    bins = np.logspace(np.log10(r_min), np.log10(r_max), nbins + 1)
    N = len(data_x)
    rand_N = len(rand_x)

    # Auto pair counts in DD
    autocorr=1
    DD_counts = DDrppi(autocorr, nthreads, rpimax, bins, X1=data_x, Y1=data_y, Z1=data_z,
                       periodic=False, verbose=verbose)


    # Cross pair counts in DR
    autocorr=0
    DR_counts = DDrppi(autocorr, nthreads, rpimax, bins, data_x, data_y, data_z,
                        X2=rand_x, Y2=rand_y, Z2=rand_z,
                        periodic=False, verbose=verbose)

    # Auto pairs counts in RR
    autocorr=1
    RR_counts = DDrppi(autocorr, nthreads, rpimax, bins, rand_x, rand_y, rand_z,
                        periodic=False, verbose=verbose)


    # All the pair counts are done, get the correlation function
    wp = convert_rp_pi_counts_to_wp(N, N, rand_N, rand_N,
                                     DD_counts, DR_counts,
                                     DR_counts, RR_counts, nbins, rpimax)

    # get actual pair counts in each bin
    DD_actual = get_wp_paircounts(DD_counts)
    DR_actual = get_wp_paircounts(DR_counts)
    RR_actual = get_wp_paircounts(RR_counts)

    return bins, wp, DD_actual, RR_actual, DR_actual





# calculate the 2d 2pcf w(theta), and also the covariance matrix using the jackknife method
def wtheta_jackknife(nbins, degmin, degmax, data_RA, data_DEC, rand_RA, rand_DEC, nthreads, njack, weights=None, pair_weights=[1], cutoff=100, verbose=1):

    # get 2pcf off full field
    # print('Calculating results for full field..')
    results_full = wtheta_treecorr(nbins, degmin, degmax, data_RA, data_DEC, rand_RA, rand_DEC, nthreads, weights, pair_weights, cutoff, verbose)

    # now start removing regions and calculating for the jackknife measurement

    # this is assuming you are running on a rectangular field
    RA_min = np.amin(data_RA)
    RA_max = np.amax(data_RA)
    DEC_min = np.amin(data_DEC)
    DEC_max = np.amax(data_DEC)

    df_data = pd.DataFrame()
    df_rand = pd.DataFrame()
    df_data['RA'] = data_RA
    df_data['DEC'] = data_DEC
    df_rand['RA'] = rand_RA
    df_rand['DEC'] = rand_DEC

    # container for jackknife results
    results_arr = np.zeros((nbins, njack))
    avg_arr = np.zeros(nbins) # store the average value of the corr func for a given bin, over all the jackknife measurements
    bin_mid_arr = np.zeros(nbins)
    cov_arr = np.zeros((nbins, nbins))

    # print('%s jacknife regions to be sampled.'%(njack))
    for i in range(njack):
        # print ('Calculating results for jacknife region %s'%(i+1))

        # divide the data up into strips in RA, remove one strip at a time and calculate
        min_RA = np.amin(data_RA)
        max_RA = np.amax(data_RA)
        RA_thickness = (max_RA - min_RA)/njack
        min_RA_jack = min_RA + (i*RA_thickness)
        max_RA_jack = min_RA + ((i+1)*RA_thickness)

        # cut out the jackknife region
        df_data_jack = pd.DataFrame()
        df_rand_jack = pd.DataFrame()
        df_data_jack = df_data[(df_data.RA <= min_RA_jack) | (df_data.RA > max_RA_jack)]
        df_rand_jack = df_rand[(df_rand.RA <= min_RA_jack) | (df_rand.RA > max_RA_jack)]

        # check the cut has worked as expected
        # plt.figure()
        # plt.xlim(min_RA, max_RA)
        # plt.scatter(df_data_jack['RA'], df_data_jack['DEC'], s=0.05)
        # plt.show()

        # calculate the 2pcf on the jackknife region
        results_jack = wtheta_treecorr(nbins, degmin, degmax, df_data_jack['RA'].values, df_data_jack['DEC'].values,
                                       df_rand_jack['RA'].values, df_rand_jack['DEC'].values, nthreads, weights, pair_weights, cutoff, verbose)


        for j in range(nbins):
            if i == 0: # assign bins
                bin_mid_arr[j] = results_jack.bin_mid[j]

            results_arr[j][i] = results_jack.result[j] # assign result


    # average the 2pcf results
    for i in range(nbins):
        avg_arr[i] = np.sum(results_arr[i]) / njack


    # print('Calculating covariance matrix...')
    norm = (njack-1)/njack

    for i in range(nbins):
        for j in range(nbins):
            res = 0
            for k in range(njack):
                res += (results_arr[i][k] - avg_arr[i])*(results_arr[j][k] - avg_arr[j])
            cov_arr[i][j] = res*norm

    # assign results to results class and return
    results = results_2pcf_jackknife(results_jack.bin_mid, results_full.result, cov_arr, results_full.DD_counts, results_full.RR_counts, results_full.DR_counts)

    return results





# calculate the 3d 2pcf xi(r), and also the covariance matrix using the jackknife method
def xi_jackknife(nbins, r_min, r_max, data_x, data_y, data_z, rand_x, rand_y, rand_z, nthreads, njack, verbose=1):

    # get 2pcf off full field
    # print('Calculating results for full field..')
    results_full = xi_treecorr(nbins, r_min, r_max, data_x, data_y, data_z, rand_x, rand_y, rand_z, nthreads, verbose)

    # now start removing regions and calculating for the jackknife measurement

    # start by calculating the spherical coords from the cartesian
    data_r, data_DEC, data_RA = astropy.coordinates.cartesian_to_spherical(data_x, data_y, data_z)
    data_RA = np.rad2deg(data_RA.value)
    data_DEC = np.rad2deg(data_DEC.value)

    rand_r, rand_DEC, rand_RA = astropy.coordinates.cartesian_to_spherical(rand_x, rand_y, rand_z)
    rand_RA = np.rad2deg(rand_RA.value)
    rand_DEC = np.rad2deg(rand_DEC.value)

    # this is assuming you are running on a rectangular field
    RA_min = np.amin(data_RA)
    RA_max = np.amax(data_RA)
    DEC_min = np.amin(data_DEC)
    DEC_max = np.amax(data_DEC)
    # print(RA_min, RA_max, DEC_min, DEC_max)

    df_data = pd.DataFrame()
    df_rand = pd.DataFrame()
    df_data['RA'] = data_RA
    df_data['DEC'] = data_DEC
    df_data['xpos'] = data_x
    df_data['ypos'] = data_y
    df_data['zpos'] = data_z

    df_rand['RA'] = rand_RA
    df_rand['DEC'] = rand_DEC
    df_rand['xpos'] = rand_x
    df_rand['ypos'] = rand_y
    df_rand['zpos'] = rand_z

    # container for jackknife results
    results_arr = np.zeros((nbins, njack))
    avg_arr = np.zeros(nbins) # store the average value of the corr func for a given bin, over all the jackknife measurements
    bin_mid_arr = np.zeros(nbins)
    cov_arr = np.zeros((nbins, nbins))

    # print('%s jacknife regions to be sampled.'%(njack))
    for i in range(njack):
        # print ('Calculating results for jacknife region %s'%(i+1))

        # divide the data up into strips in RA, remove one strip at a time and calculate
        min_RA = np.amin(data_RA)
        max_RA = np.amax(data_RA)
        RA_thickness = (max_RA - min_RA)/njack
        min_RA_jack = min_RA + (i*RA_thickness)
        max_RA_jack = min_RA + ((i+1)*RA_thickness)

        # cut out the jackknife region
        df_data_jack = pd.DataFrame()
        df_rand_jack = pd.DataFrame()
        df_data_jack = df_data[(df_data.RA <= min_RA_jack) | (df_data.RA > max_RA_jack)]
        df_rand_jack = df_rand[(df_rand.RA <= min_RA_jack) | (df_rand.RA > max_RA_jack)]

        # check the cut has worked as expected
        # plt.figure()
        # plt.xlim(min_RA, max_RA)
        # plt.scatter(df_data_jack['RA'], df_data_jack['DEC'], s=0.05)
        # plt.show()

        # calculate the 2pcf on the jackknife region
        results_jack = xi_treecorr(nbins, r_min, r_max, df_data_jack['xpos'].values, df_data_jack['ypos'].values, df_data_jack['zpos'].values,
                                   df_rand_jack['xpos'].values, df_rand_jack['xpos'].values, df_rand_jack['xpos'].values, nthreads, verbose)


        for j in range(nbins):
            if i == 0: # assign bins
                bin_mid_arr[j] = results_jack.bin_mid[j]

            results_arr[j][i] = results_jack.result[j] # assign result


    # average the 2pcf results
    for i in range(nbins):
        avg_arr[i] = np.sum(results_arr[i]) / njack


    # print('Calculating covariance matrix...')
    norm = (njack-1)/njack

    for i in range(nbins):
        for j in range(nbins):
            res = 0
            for k in range(njack):
                res += (results_arr[i][k] - avg_arr[i])*(results_arr[j][k] - avg_arr[j])
            cov_arr[i][j] = res*norm

    # assign results to results class and return
    results = results_2pcf_jackknife(results_jack.bin_mid, results_full.result, cov_arr, results_full.DD_counts, results_full.RR_counts, results_full.DR_counts)

    return results

# calculate the projected 2pcf w_p(r_p), and also the covariance matrix using the jackknife method
def wp_jackknife(nbins, r_min, r_max, rpimax, data_x, data_y, data_z, rand_x, rand_y, rand_z, nthreads, njack, verbose=True):

    # get 2pcf off full field
    # print('Calculating results for full field..')
    bins, wp_full, DD_counts, RR_counts, DR_counts = wp_corrfunc(nbins, r_min, r_max, rpimax, data_x, data_y, data_z, rand_x, rand_y, rand_z, nthreads, verbose=verbose)
    # bin_mid = (bins[1:] + bins[:-1]) * .5
    bin_mid = np.exp(0.5*(np.log(bins[1:]) + np.log(bins[:-1])))

    # now start removing regions and calculating for the jackknife measurement

    # start by calculating the spherical coords from the cartesian
    data_r, data_DEC, data_RA = astropy.coordinates.cartesian_to_spherical(data_x, data_y, data_z)
    data_RA = np.rad2deg(data_RA.value)
    data_DEC = np.rad2deg(data_DEC.value)

    rand_r, rand_DEC, rand_RA = astropy.coordinates.cartesian_to_spherical(rand_x, rand_y, rand_z)
    rand_RA = np.rad2deg(rand_RA.value)
    rand_DEC = np.rad2deg(rand_DEC.value)

    # this is assuming you are running on a rectangular field
    RA_min = np.amin(data_RA)
    RA_max = np.amax(data_RA)
    DEC_min = np.amin(data_DEC)
    DEC_max = np.amax(data_DEC)
    # print(RA_min, RA_max, DEC_min, DEC_max)

    df_data = pd.DataFrame()
    df_rand = pd.DataFrame()
    df_data['RA'] = data_RA
    df_data['DEC'] = data_DEC
    df_data['xpos'] = data_x
    df_data['ypos'] = data_y
    df_data['zpos'] = data_z

    df_rand['RA'] = rand_RA
    df_rand['DEC'] = rand_DEC
    df_rand['xpos'] = rand_x
    df_rand['ypos'] = rand_y
    df_rand['zpos'] = rand_z

    # container for jackknife results
    results_arr = np.zeros((nbins, njack))
    avg_arr = np.zeros(nbins) # store the average value of the corr func for a given bin, over all the jackknife measurements
    cov_arr = np.zeros((nbins, nbins))

    # print('%s jacknife regions to be sampled.'%(njack))
    for i in range(njack):
        # print ('Calculating results for jacknife region %s'%(i+1))

        # divide the data up into strips in RA, remove one strip at a time and calculate
        min_RA = np.amin(data_RA)
        max_RA = np.amax(data_RA)
        RA_thickness = (max_RA - min_RA)/njack
        min_RA_jack = min_RA + (i*RA_thickness)
        max_RA_jack = min_RA + ((i+1)*RA_thickness)

        # cut out the jackknife region
        df_data_jack = pd.DataFrame()
        df_rand_jack = pd.DataFrame()
        df_data_jack = df_data[(df_data.RA <= min_RA_jack) | (df_data.RA > max_RA_jack)]
        df_rand_jack = df_rand[(df_rand.RA <= min_RA_jack) | (df_rand.RA > max_RA_jack)]

        # check the cut has worked as expected
        # plt.figure()
        # plt.xlim(min_RA, max_RA)
        # plt.scatter(df_data_jack['RA'], df_data_jack['DEC'], s=0.05)
        # plt.show()

        # calculate the 2pcf on the jackknife region
        bins_jack, wp_jack, DD_jack, RR_jack, DR_jack = wp_corrfunc(nbins, r_min, r_max, rpimax, df_data_jack['xpos'].values, df_data_jack['ypos'].values, df_data_jack['zpos'].values,
                                   df_rand_jack['xpos'].values, df_rand_jack['ypos'].values, df_rand_jack['zpos'].values, nthreads, verbose=verbose)

        for j in range(nbins):
            results_arr[j][i] = wp_jack[j] # assign result

    # average the 2pcf results
    for i in range(nbins):
        avg_arr[i] = np.sum(results_arr[i]) / njack


    # print('Calculating covariance matrix...')
    norm = (njack-1)/njack

    for i in range(nbins):
        for j in range(nbins):
            res = 0
            for k in range(njack):
                res += (results_arr[i][k] - avg_arr[i])*(results_arr[j][k] - avg_arr[j])
            cov_arr[i][j] = res*norm

    # assign results to results class and return
    results = results_2pcf_jackknife(bin_mid, wp_full, cov_arr, DD_counts, RR_counts, DR_counts)

    return results
