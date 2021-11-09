from project_libs import *
import allvars as av
import funcs

# function to create healpy maps for angular selection function
def get_angular_selection(df):

    # create healpy indices and pixel numbers of input data
    m_data = hp.ang2pix(av.nside, df['RA'], df['DEC'], lonlat=True)

    # assign pixel numbers as new column in dataframe
    df = df.assign(hp_pixel = hp.ang2pix(av.nside, df['RA'], df['DEC'], lonlat=True))

    # create healpy map of the data - the map is a 1d array where the index = pixel number,
    # and index value = points (i.e. galaxies) inside that pixel
    map_data = funcs.gen_fast_map(m_data, av.nside)

    # now want to find the average EBV value per pixel and make a map of this
    avg_EBV_df = df.groupby('hp_pixel').EBV.mean() # can use the group by function with the mean function
    avg_EBV_df = avg_EBV_df.reindex(range(av.npixels), fill_value=0) # fill missing dataframe index rows (caused by empty pixels) with zeros
    # now create the new map
    av.map_EBV = np.zeros(len(map_data))
    for i in range(len(avg_EBV_df)):
        av.map_EBV[i] = avg_EBV_df[i]

    # plots of the maps
    plt.figure(figsize=(15,9))
    plt.subplot(2,1,1)
    hp.mollview(map_data, title='Healpy map of data.', hold=True)
    plt.subplot(2,1,2)
    hp.mollview(av.map_EBV, title='Healpy map of average EBV value per pixel.', hold=True)
    plt.savefig(av.path_output + 'healpix_maps.pdf', bbox_inches='tight')
    plt.close()

    # histogram the average EBV data
    av.bins_EBV = np.linspace(min(av.map_EBV), max(av.map_EBV), 300)
    hist_EBV = np.histogram(av.map_EBV, bins=av.bins_EBV)[0]

    # now get a histogram of the average density of gals per pixel, as a function of avg EBV val
    av.hist_pix_density = np.histogram(av.map_EBV, weights = map_data, bins=av.bins_EBV)[0]
    for i in range(len(av.hist_pix_density)):
        if(hist_EBV[i] == 0):
            av.hist_pix_density[i] = 0
        else:
            av.hist_pix_density[i] = av.hist_pix_density[i] / hist_EBV[i]

    # plotting
    plt.figure(figsize=(14,7))
    plt.subplot(1,2,1)
    plt.hist(av.map_EBV, bins=av.bins_EBV, histtype='bar')
    plt.xlabel('average EBV value per pixel', size=14)
    plt.ylabel('number of pixels', size=14)

    plt.subplot(1,2,2)
    plt.plot(av.bins_EBV[:-1], av.hist_pix_density)
    plt.xlabel('average EBV value per pixel', size=14)
    plt.ylabel('average density of galaxies per pixel', size=14)

    plt.savefig(av.path_output + 'EBV_hists.pdf', bbox_inches='tight')
    plt.close()



# applies angular and radial selection functions to the window grid
def create_window_grid():

    # create arrays for the grid points
    comov_x_arr = np.linspace(av.x_min+(av.lx/2), av.x_max-(av.lx/2), av.nx)
    comov_y_arr = np.linspace(av.y_min+(av.ly/2), av.y_max-(av.ly/2), av.ny)
    comov_z_arr = np.linspace(av.z_min+(av.lz/2), av.z_max-(av.lz/2), av.nz)

    # create a window function grid
    av.wingrid = np.ones((av.nx, av.ny, av.nz))

    # create a grid of comoving distances
    distgrid = np.sqrt(comov_x_arr[:,np.newaxis,np.newaxis]**2 + comov_y_arr[np.newaxis,:,np.newaxis]**2
                       + comov_z_arr[np.newaxis,np.newaxis,:]**2)

   # apply radial selection
    av.wingrid = np.where((distgrid <= av.comov_max) & (distgrid >= av.comov_min), 1., 0.)
    av.wingrid *= av.spl_data_nr(distgrid)

    # create a RA (phi) grid and a DEC (theta) grid
    phigrid = -1 * (np.arctan2(comov_x_arr[:,np.newaxis,np.newaxis], comov_y_arr[np.newaxis,:,np.newaxis] + 0.0*comov_z_arr[np.newaxis,np.newaxis,:]) - np.pi/2)
    thetagrid = np.arccos((0.*comov_x_arr[:,np.newaxis,np.newaxis] + 0.*comov_y_arr[np.newaxis,:,np.newaxis] + comov_z_arr[np.newaxis,np.newaxis,:]) / distgrid)
    del distgrid

    # now get the pixel value for each grid point
    pixelgrid = hp.ang2pix(av.nside, thetagrid, phigrid, lonlat=False)
    del thetagrid, phigrid

    # then get the corresponding EBV value from the map
    def get_EBV_val(grid):
        EBV_val = av.map_EBV[grid]
        return EBV_val
    EBVgrid = get_EBV_val(pixelgrid)
    del pixelgrid

    # now get the appropriate number density from the pixel ebv value and also apply this to the win grid
    av.wingrid *= np.interp(EBVgrid, av.bins_EBV[:-1], av.hist_pix_density)
    del EBVgrid
    gc.collect()

    # finally, normalise the window grid
    av.wingrid /= np.sum(av.wingrid)

    # side by side slices of window and data grids for visual comparisons
    varz = int(av.box_points/2); # choose grid slice
    # print('Slice of window, data grids through z cord %s/%s'%(varz, av.box_points))
    plt.figure(figsize=(15,8))

    plt.subplot(1,2,1)
    plt.title('Slice of window function grid')
    # plt.xlabel('y')
    # plt.ylabel('x')
    plt.imshow(av.wingrid[:, :, varz])
    plt.colorbar()


    plt.subplot(1,2,2)
    plt.title('Slice of data grid')
    plt.imshow(av.datgrid[:, :, varz])
    plt.colorbar()

    plt.savefig(av.path_output + 'data_window_comparison.pdf', bbox_inches='tight')
    plt.close()





# test the window grid to see if you can recover the correct radial distribution
def test_window_grid():

    return 1
