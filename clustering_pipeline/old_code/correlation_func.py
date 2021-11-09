from project_libs import *
import allvars as av
import funcs



# Generate random (R.A., Dec.) distribution sub-sampled by healpix map
def genhpranradec(nrangen,rmin,rmax,dmin,dmax,nside,hpcomp):
    # Generate random (R.A., Dec.) in the range (rmin-rmax, dmin-dmax)
    fact = np.pi/180.
    smin,smax = np.sin(fact*dmin),np.sin(fact*dmax)
    ras = rmin + (rmax-rmin)*np.random.rand(nrangen)
    dec = np.arcsin(smin + (smax-smin)*np.random.rand(nrangen))/fact
    # Determine healpix pixel of each point
    phi = np.radians(ras)
    theta = np.radians(90.-dec)
    ipix = hp.ang2pix(nside,theta,phi)
    # Sub-sample points according to probability in angular mask
    probkeep = hpcomp[ipix]
    ran = np.random.rand(len(ras))
    keep = (ran < probkeep)
    ras,dec = ras[keep],dec[keep]
    return ras,dec



# get the angular clustering using corrfunc code
def get_wtheta(RA, DEC, rand_RA, rand_DEC, bins):

    rand_N = len(rand_RA)
    N = len(RA)

    # Number of threads to use
    nthreads = 32

    # Auto pair counts in DD
    autocorr=1
    DD_counts = DDtheta_mocks(autocorr, nthreads, bins,
                            RA, DEC, verbose=True)

    # Cross pair counts in DR
    autocorr=0
    DR_counts = DDtheta_mocks(autocorr, nthreads, bins,
                            RA, DEC,
                            RA2=rand_RA, DEC2=rand_DEC, verbose=True)

    # Auto pairs counts in RR
    autocorr=1
    RR_counts = DDtheta_mocks(autocorr, nthreads, bins,
                            rand_RA, rand_DEC, verbose=True)

    # All the pair counts are done, get the angular correlation function
    wtheta = convert_3d_counts_to_cf(N, N, rand_N, rand_N,
                                    DD_counts, DR_counts,
                                    DR_counts, RR_counts)

    return wtheta






# function to calculate the angular clustering of the data ( w(theta ) )
# in various redshift slices and fit a bias model
def calculate_angular_clustering(df):

    # determine redshift bins for calculation.
    # ideally want several redshift bins of thickness 0.1 across the data
    n_bins = 5
    redshift_interval = (av.redshift_max - av.redshift_min) / (n_bins*1.) # space between bin lower vals

    # print('z_min = %s and z_max = %s'%(av.redshift_min, av.redshift_max))
    if (av.redshift_max - av.redshift_min < (0.1*n_bins)): # if too thin to have n * 0.1 thickness bins
        bin_thickness = redshift_interval
    else:
        bin_thickness = 0.1

    # define lower bin edges
    bin_start = np.zeros(n_bins)
    for i in range(n_bins):
        bin_start[i] = av.redshift_min + i*redshift_interval
    # print('z_bins = %s'%(bin_start))

    # calculate angular clustering for several redshift slices
    for i in range(n_bins):

        # slice dataframe to required redshift range
        df_sliced = pd.DataFrame()
        df_sliced = df[(df.Z < (bin_start[i]+bin_thickness)) & (df.Z >= bin_start[i])]
        av.N_rand = len(df_sliced) * av.rand_multi # determine number of randoms
        # print('N gals = %s, n_rand = %s'%(len(df_sliced), av.N_rand))

        # create healpy indices and pixel numbers of input data
        m_data = hp.ang2pix(av.nside, df_sliced['RA'], df_sliced['DEC'], lonlat=True)
        ang_map = funcs.gen_fast_map(m_data, av.nside)

        # use the genhpranradec function above to generate ~ N_rand points
        rand_ra = []
        rand_dec = []
        while (len(rand_ra) < av.N_rand):
            temp_ra, temp_dec = genhpranradec(av.N_rand,min(df_sliced['RA']),max(df_sliced['RA']),min(df_sliced['DEC']),max(df_sliced['DEC']),av.nside,ang_map)
            rand_ra = np.append(rand_ra, temp_ra)
            rand_dec = np.append(rand_dec, temp_dec)
        # print(i, av.N_rand, len(rand_ra), len(rand_dec))

        # #scatter plot of randoms to check they match the data in distribution
        # plt.figure(figsize=(15, 7))
        # plt.subplot(1,2,1)
        # plt.title('data slice')
        # plt.scatter(df_sliced['RA'], df_sliced['DEC'], s=0.2)
        # plt.subplot(1,2,2)
        # plt.title('randoms')
        # plt.scatter(rand_ra, rand_dec, s=0.2)
        # plt.show()

        # Setup the bins for wtheta calc
        nbins = 30
        bins = np.logspace(np.log10(0.01), np.log10(10.0), nbins+1) # note the +1 to nbins
                # get w(theta) from corrfunc code using function above
        wtheta = get_wtheta(df_sliced['RA'].values, df_sliced['DEC'].values, rand_ra, rand_dec, bins)
        print('bin \t w(theta)')
        for i in range(nbins):
            print(round(bins[i],2), '\t', wtheta[i])
