# python file for the function to calculate the survey-window convolved power spectrum
# using the nbodykit library

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from nbodykit.lab import *

from structures import results_powerspec, powerspec_measurement

def calc_convolved_FFT_power(results, cat_data, cat_rand, cosmo, cat_vars, real_or_redshift, tracer, catalogue,
                                Z_min, Z_max, Z_mid, bin_no, Nmesh=512, window='tsc', compensated=True, interlaced=True,
                                multipoles=[0,2,4], kmin=0, dk=0.005, Pk_fid=1e4, nhist_bins=None):

    # determine if working in real or redshift space, making adjustments that QSO and LyA samples dont have the redshift space option at present
    xcol = 'xpos'
    ycol = 'ypos'
    zcol = 'zpos'
    Z_col = 'REDSHIFT_ESTIMATE'
    if (real_or_redshift == 'redshift') and (tracer in ['BG', 'LRG']):
        xcol = 'xpos_S'
        ycol = 'ypos_S'
        zcol = 'zpos_S'
        Z_col = 'redshift_S'

    print('Dataframe cols:')

    # create 3-vector position columns
    cat_data['Position'] = cat_data[xcol][:, None] * [1, 0, 0] + cat_data[ycol][:, None] * [0, 1, 0] + cat_data[zcol][:, None] * [0, 0, 1]
    cat_rand['Position'] = cat_rand['xpos'][:, None] * [1, 0, 0] + cat_rand['ypos'][:, None] * [0, 1, 0] + cat_rand['zpos'][:, None] * [0, 0, 1]


    # ******************************************************************************
    # ******************************************************************************

    print('Computing the n(z)...')
    # The next step is to get the n(z) of the data which we will use for FKP weights

    # compute n(z) from the randoms (taking into consideration the sky fraction)
    if nhist_bins==None:
        zhist = RedshiftHistogram(cat_rand, cat_vars.fsky, cosmo, redshift='Z')
    else:
        zhist = RedshiftHistogram(cat_rand, cat_vars.fsky, cosmo, redshift='Z', bins=nhist_bins)

    # re-normalize to the total size of the data catalog
    Ndata = cat_data.csize
    Nrand = cat_rand.csize
    alpha = 1.0 * Ndata / Nrand

    # add n(z) from randoms to the FKP source
    nofz = InterpolatedUnivariateSpline(zhist.bin_centers, alpha*zhist.nbar)
    cat_data['NZ'] = nofz(cat_data[Z_col])
    cat_rand['NZ'] = nofz(cat_rand['Z'])


    # ******************************************************************************
    # ******************************************************************************

    # initialise an FKP catalogue object, paint to mesh, and compute the P(k) multipoles

    # initialize the FKP source
    fkp = FKPCatalog(cat_data, cat_rand, BoxPad=0.02)

    # add the n(z) columns to the FKPCatalogue
    fkp['data/NZ'] = nofz(cat_data[Z_col])
    fkp['randoms/NZ'] = nofz(cat_rand['Z'])

    # add the FKP weights to the FKPCatalogue
    fkp['data/FKPWeight'] = 1.0 / (1.0 + fkp['data/NZ'] * Pk_fid)
    fkp['randoms/FKPWeight'] = 1.0 / (1.0 + fkp['randoms/NZ'] * Pk_fid)

    # paint to mesh
    print('Painting to mesh...')
    mesh = fkp.to_mesh(Nmesh=Nmesh, window=window, compensated=compensated, interlaced=interlaced, nbar='NZ', fkp_weight='FKPWeight')
    BoxSize = np.product(mesh.attrs['BoxSize'])

    # compute the power spectrum multipoles in linear bins
    print('Calculating the power...', flush=True)
    r = ConvolvedFFTPower(mesh, poles=multipoles, dk=dk, kmin=kmin)

    # extract the info from the poles
    poles = r.poles
    Nmodes = poles['modes']

    # initialise class to take powerspec results
    measurement = powerspec_measurement(Pk_fid, kmin, dk, multipoles, Nmodes, Ndata, Nrand, Z_min, Z_max, Z_mid)
    measurement.bin_mid = poles['k']
    measurement.Nbins = len(poles['k'])

    # loop over multipoles, get Pk and error and add to class
    for ell in multipoles:
        Pk = poles['power_%d' %ell].real #
        if ell==0:
            Pk = Pk - poles.attrs['shotnoise']
        Pk_err = np.where(Nmodes>0, (np.sqrt(2./Nmodes))*(Pk+(BoxSize/Ndata)), -1)

        attr_name = 'Pk' + str(ell)
        setattr(measurement, attr_name, Pk)
        attr_name = 'Pk' + str(ell) + '_err'
        setattr(measurement, attr_name, Pk_err)




    # if 0 in multipoles:
    #     P = poles['power_%d' %0].real # the measured P_0(k)
    #     Pk0 = P - poles.attrs['shotnoise'] # only subtract shot noise for monopole
    #     err0 = np.where(Nmodes>0, (1./np.sqrt(Nmodes))*(Pk0+(BoxSize/Ndata)), -1.)
    # if 2 in multipoles:
    #     Pk2 = poles['power_%d' %2].real # the measured P_2(k)
    # if 4 in multipoles:
    #     Pk4 = poles['power_%d' %4].real # the measured P_4(k)


    # add the powerspec measurement class to the results class and save
    attr_name = 'bin' + str(bin_no)
    setattr(results, attr_name, measurement)
    # results = results_powerspec(poles['k'], Pk0, err0, Nmodes, Ndata, Nrand,
    #                     dk, Pk_fid, Nmesh, window, interlaced, tracer, catalogue,
    #                     compensated, Pk2=Pk2, Pk4=Pk4)

    print('P(k) calculation done...', flush=True)

    return results

    # ******************************************************************************
    # ******************************************************************************
