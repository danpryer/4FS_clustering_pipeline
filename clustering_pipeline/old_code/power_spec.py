from project_libs import *
import allvars as av
import funcs
import cosmology as cmg


# function to get model P(k) linear and NL from CAMB,
# creates a spline of both, stores as global object
def get_CAMB_power(redshifts):
    # # placeholder vars to be returned
    # model_pk = np.zeros(len(k))
    # model_pk_NL = np.zeros(len(k))

    # set the cosmology
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=av.cosmo.H0.value*av.hubble_param,
                        ombh2=av.cosmo.Ob0*pow(av.hubble_param, 2),
                        omch2=av.cosmo.Om0*pow(av.hubble_param, 2)-av.cosmo.Ob0*pow(av.hubble_param, 2))
    pars.InitPower.set_params(ns=0.965)

    #Note non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=redshifts, kmax=av.CAMB_kmax)

    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=av.CAMB_minkh,
                                maxkh=av.CAMB_maxkh, npoints = av.CAMB_npoints)
    s8 = np.array(results.get_sigma8())

    #Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh_nl, z_nl, pk_nl = results.get_matter_power_spectrum(minkh=av.CAMB_minkh,
                                maxkh=av.CAMB_maxkh, npoints = av.CAMB_npoints)

    # spline linear and nonlinear power
    av.spl_model_pk = funcs.create_spline(kh, pk[0])
    av.spl_model_pk_nl = funcs.create_spline(kh_nl, pk_nl[0])


# ******************************************************************************
# ***************** Functions to calculate NGP correction **********************

# function to get the power from the pk_est list, given an input k
def get_scalar_power(k):
    p = av.pk_est[np.digitize(k, av.k_bins)-1]
    return p

# function that returns the model redshift space power spectrum for a given k, mu
def modelpk(k, mu):
    beta = av.spl_growth_rate(av.z_eff) / av.spl_bias(av.z_eff) # beta = f/b
    P_rs = get_scalar_power(k)*((1+(beta*mu**2))**2)*pow(1+pow(k*av.bias_sigma*mu,2), -1)
    return P_rs

# function to correct for NGP aliasing
def getngpcorr(nx,ny,nz,lx,ly,lz):
    nmax = 2
    nyqx,nyqy,nyqz = nx*np.pi/lx,ny*np.pi/ly,nz*np.pi/lz
    kx = 2.*np.pi*np.fft.fftfreq(nx,d=lx/nx)
    ky = 2.*np.pi*np.fft.fftfreq(ny,d=ly/ny)
    kz = 2.*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1]
    ngpcorrspec = dongpcorrsum(nmax,nyqx,nyqy,nyqz,kx[:,np.newaxis,np.newaxis],ky[np.newaxis,:,np.newaxis],kz[np.newaxis,np.newaxis,:])
    return ngpcorrspec



# NGP aliasing correction, called by getngpcorr above.
def dongpcorrsum(nmax,nyqx,nyqy,nyqz,kx,ky,kz):
    k = np.sqrt((kx**2)+(ky**2)+(kz**2))
    mu = np.divide(kx,k,out=np.zeros_like(k),where=k!=0.)
    pk = modelpk(k,mu)
    sum1 = 0.
    for ix in range(-nmax,nmax+1):
        for iy in range(-nmax,nmax+1):
            for iz in range(-nmax,nmax+1):
                kx1 = kx + 2.*nyqx*ix
                ky1 = ky + 2.*nyqy*iy
                kz1 = kz + 2.*nyqz*iz
                k1 = np.sqrt((kx1**2)+(ky1**2)+(kz1**2))
                mu1 = np.divide(kx1,k1,out=np.zeros_like(k1),where=k1!=0.)
                pk1 = modelpk(k1,mu1)
                qx1,qy1,qz1 = (np.pi*kx1)/(2.*nyqx),(np.pi*ky1)/(2.*nyqy),(np.pi*kz1)/(2.*nyqz)
                wx = np.divide(np.sin(qx1),qx1,out=np.ones_like(qx1),where=qx1!=0.)
                wy = np.divide(np.sin(qy1),qy1,out=np.ones_like(qy1),where=qy1!=0.)
                wz = np.divide(np.sin(qz1),qz1,out=np.ones_like(qz1),where=qz1!=0.)
                ww = wx*wy*wz
                sum1 += (ww**2)*pk1
    return sum1/pk


# apply NGP correction to pk_est
def correct_pkest():
    correction_spec = getngpcorr(av.nx,av.ny,av.nz,av.lx,av.ly,av.lz)
    # now divide the original pspec grid by this correction spec and recalculate the pk_est array
    pkspec_c = av.pkspec / correction_spec

    # Bin in which each mode belongs
    ikbin = np.digitize(av.kspec,np.linspace(av.khmin,av.khmax,av.nkbin+1))

    # Average power spectrum in k-bins
    av.nmodes, av.pk_est = np.zeros(av.nkbin,dtype=int),np.full(av.nkbin,-1.)
    # print nmodes, pk
    for ik in range(av.nkbin):
        av.nmodes[ik] = int(np.sum(np.array([ikbin == ik+1])))
        if (av.nmodes[ik] > 0):
            av.pk_est[ik] = np.mean(pkspec_c[ikbin == ik+1])



# ******************************************************************************
# ****************** Functions to calculate P(k) w/ error **********************

# function for calculating the error on the power spectrum estimate
def getpkerr():
    if (np.absolute(np.sum(av.wingrid)-1.) > 0.01):
        print ('Sum W(x) =',np.sum(av.wingrid))
        print ('Window function is unnormalized!!')
        sys.exit()
    vc = av.vol/av.N_cells
    n2w4 = vc*np.sum(((av.wingrid*av.N_gal/vc)**2)*(av.weigrid**4))
    n3w4 = vc*np.sum(((av.wingrid*av.N_gal/vc)**3)*(av.weigrid**4))
    n4w4 = vc*np.sum(((av.wingrid*av.N_gal/vc)**4)*(av.weigrid**4))
    n2w2 = vc*np.sum(((av.wingrid*av.N_gal/vc)**2)*(av.weigrid**2))
#     print('len (nmodes, pk) = (%s, %s)'%(len(nmodes), len(pk)))
    pkerr = np.where(av.nmodes>0,np.sqrt((av.vol/av.nmodes)*(n4w4*(av.pk_est**2)+2.*n3w4*av.pk_est+n2w4)/(n2w2**2)),-1.)
    return pkerr



# function for calculating the power spectrum
def calculate_pkest():
    # weight function for FKP weights
    av.pk0_fid = 7000
    av.weigrid = 1./(1.+((av.wingrid*av.N_gal*av.pk0_fid*av.N_cells)/av.vol))

    # Determine shot noise factor Sum w(x)^2 N(x)
    sgal = np.sum((av.weigrid**2)*av.datgrid)
    # print ("Shot noise =", sgal)

    # Determine normalization Sum W(x)^2 w(x)^2
    sumwsq = av.N_cells*np.sum((av.wingrid*av.weigrid)**2)

    # Determine FFT[w(x)*N(x)] - N*FFT[w(x)*W(x)]
    datspec = np.fft.rfftn(av.weigrid*av.datgrid) - av.N_gal*np.fft.rfftn(av.weigrid*av.wingrid)

    # Determine P(k) estimator
    av.pkspec = np.real(datspec)**2 + np.imag(datspec)**2

    # Correct P(k) for shot noise and normalisation
    av.pkspec = (av.pkspec-sgal)*av.vol/(sumwsq*(av.N_gal**2))

    # k-value for every mode
    kx = 2.*np.pi*np.fft.fftfreq(av.nx,d=av.Lx/av.nx)
    ky = 2.*np.pi*np.fft.fftfreq(av.ny,d=av.Ly/av.ny)
    kz = 2.*np.pi*np.fft.fftfreq(av.nz,d=av.Lz/av.nz)[:int(av.nz/2)+1]
    av.kspec = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)

    # Bin in which each mode belongs
    ikbin = np.digitize(av.kspec,np.linspace(av.khmin,av.khmax,av.nkbin+1))

    # 1d P(k) bins
    av.k_bins = np.linspace(av.khmin, av.khmax, av.nkbin)

    # Average power spectrum in k-bins
    av.nmodes, av.pk_est = np.zeros(av.nkbin,dtype=int),np.full(av.nkbin,-1.)
    # print nmodes, pk
    for ik in range(av.nkbin):
        av.nmodes[ik] = int(np.sum(np.array([ikbin == ik+1])))
        if (av.nmodes[ik] > 0):
            av.pk_est[ik] = np.mean(av.pkspec[ikbin == ik+1])

    # print ("Nmodes =", nmodes)
    #Approximation to power spectrum error in each bin:
    # av.pkerr = np.where(nmodes>0, (1./np.sqrt(nmodes))*(av.pk_est+(av.vol/av.N_gal)), -1.)



# function to scale the power spectrum amplitude, turning a measurement of the
# galaxy power spectrum into a measurement of the underlying matter P(k)
def pk_scaling_amp(k, regime):

    def numerator_integrand(z, k):
        if (regime == 'linear'):
            return pow(funcs.weight_func(z) * av.spl_bias(z) * av.spl_growth_fact(z) * av.spl_data_nz(z), 2) * av.spl_RSD_lin(z)
        else:
            return pow(funcs.weight_func(z) * av.spl_bias(z) * av.spl_growth_fact(z) * av.spl_data_nz(z), 2) * cmg.RSD_nonlin(z, k)


    def denominator_integrand(z):
        return pow(funcs.weight_func(z) * av.spl_data_nz(z), 2)

    numerator = integrate.quad(numerator_integrand, av.redshift_min, av.redshift_max,
                        args=(k), limit = 10000, epsrel=1e-3)[0]
    denominator = integrate.quad(denominator_integrand, av.redshift_min, av.redshift_max,
                        limit = 10000, epsrel=1e-3)[0]

    pk_norm = pow(av.spl_growth_fact(av.redshift_min), 2) * numerator / (denominator * 1.)
    return pk_norm


def plot_pk_results():

    plt.figure(figsize=(14,10))

    # plot model power spectra from CAMB
    kh=np.logspace(np.log10(av.CAMB_minkh), np.log10(av.CAMB_maxkh), 200)
    plt.plot(kh, av.spl_model_pk(kh), color='k', label = 'linear matter power (CAMB, z=0.0)')
    plt.plot(kh, av.spl_model_pk_nl(kh), color='blue', linestyle = '-', label = 'nonlinear matter power (CAMB, z=0.0)')

    # plot estimated power from data
    plt.errorbar(av.k_bins, av.pk_dm_lin, yerr = av.pkerr_dm_lin, color='red', linestyle='',
             label='Inferred real space matter power at z = 0.0', capsize=5, elinewidth=3)
    plt.errorbar(av.k_bins, av.pk_dm_nonlin, yerr = av.pkerr_dm_nonlin, color='purple', linestyle ='',
             label = 'P(k) NL est', capsize=5, elinewidth=3)
    plt.xlabel('$k \; \; [h \; \mathrm{Mpc}^{-1}]$', size=18)
    plt.ylabel('$P(k) \; \; [h^{-3} \mathrm{Mpc}^3]$', size=18)

    plt.title('Power spectrum from 4MOST simulated QSO data (input)', fontsize=20)

    plt.xlim(av.khmin*0.9, av.khmax*1.05)
    plt.ylim(min(av.pk_dm_lin)*0.9, max(av.pk_dm_lin)*1.15)

    plt.yscale('log')
    plt.xscale('log')

    plt.xticks(size=14)
    plt.yticks(size=14)

    plt.legend(fontsize = 15)

    plt.savefig(av.path_output + 'powerspec_est_plot.pdf', bbox_inches='tight')

    plt.close()
