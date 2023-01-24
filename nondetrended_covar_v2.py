import matplotlib.pyplot as plt
from matplotlib import colors
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import bitstring as bs
import multiprocessing
import fast_pca as fpca
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

plt.rcParams.update({'font.size': 22})

mode = 'monthly' # options are "monthly" or "seasonal"
method = 'nondetrended_covar' # options are detrended/nondetrended_covar/svd

save = False
display = False # display the plots on the screen

## DO NOT CHANGE UNLESS YOU KNOW WHAT YOURE DOING
data_path = "./Data/data/"
figs_path = "./figs/"+mode+"_figs/"+method+"/"
logs_path = "./logs/"+method+"/"+mode+"/"
tabl_path = "./tables/"+mode+"_tables/"+method+"/"
fig_ext = "jpg"

m_rng = range(1, 13) # range of months to plot. Always add one to final month so jan-dec is (1, 13), just march is (3, 4), etc.

plt_rng = 3 # how many PCs to plot
## Keep in mind that figs arent deleted with each code execution, only overwritten. So if this number is reduced between executions, there will be some old figs left over
data_dim = (720, 720) # shape of all of the data files
start_year = 1979
end_year = 2020

# Heatmap boundaries
# sorted list of tuples of tuples(month, (min. longitude (degrees EAST), max. longitude,
# min. latitude (degrees NORTH), max. latitude))
hmaps = [(1, (80, 90, 40, 45)),
        (4, (65, 80, 50, 55)),
        (10, (90, 120, 60, 67))] # note for later - may try to define a coordinate Class
# remove months that aren't in m_rng (there are better ways to do this once I get around to refactoring)
hmaps = [tup for tup in hmaps if tup[0] in m_rng]
hmaps.sort(key=lambda tup: tup[0]) # sorted ascending by months

# Dictionary of how months are organized into seasons
seasons = {'winter' : [12, 1, 2],
           'spring' : [3, 4, 5],
           'summer' : [6, 7, 8],
           'fall'   : [9, 10, 11]}

# Dictionary for a months number and its abbreviated name (I could just import datetime to do this, but this works too)
m_names = {1 : 'Jan', 2 : 'Feb', 3 : 'Mar', 4 : 'Apr',
           5 : 'May', 6 : 'Jun', 7 : 'Jul', 8 : 'Aug',
           9 : 'Sep', 10 : 'Oct', 11 : 'Nov', 12 : 'Dec'}


'''
    Generalizing to seasonal analysis is still a work in progress
    But monthly analysis works

    Only looking at snow for now not sea ice

    -> TODO:

        - Fourier analysis and extracting continents
        - Countour plots and mapping significance (fix these)

        - for seasonal mode:
            - if december load data from previous year
            - for simplicity, just make a seperate load function. Then I can combine later if necessary
            - will need to adjust the sum years function as well
            - finish seasonal computation

        - some of the lmap plots still look slightly off as latitude decreases
            - check cartopy transform
        - Standardize colorbar/adjust gradient
        - change logs path. Doesn't need to be split up by method, only mode

        - simplify/generalize saving tables
            - save after every calculation instead of once at the end
        - for dfs, update append to concat

        - update SVD (not a priority)

        - Clean up and abstract code
            - lots of refactoring, get rid of unneccessary functions and imports
            - safety checks (try, catch, except, etc.)
            - put different methods in their own functions
            - make generalized plotting function (or plotting class)
            - create master code for all methods
            - change some variable names (Im looking at you, dtype)
            - optional command line arguments for global variables
            - Lots more comments and documentation (and a README)
            - requirements.txt + instructions and resources for cartopy/proj

'''


###### LOADING DATA

def masking(dmatrix, dtype='snow'):
    '''
        Given a data matrix

        Masks vals >5, 4, 0

        Returns maksed matrix
    '''

    if dtype=='snow':
            dmatrix = np.ma.masked_greater(dmatrix, 5)
            dmatrix = np.ma.masked_equal(dmatrix, 4)
            dmatrix = np.ma.masked_equal(dmatrix, 3)
            dmatrix = np.ma.masked_equal(dmatrix, 2)
            dmatrix = np.ma.masked_equal(dmatrix, 0)
    else: # sea ice
            dmatrix = np.ma.masked_greater(dmatrix, 3)
            dmatrix = np.ma.masked_less(dmatrix, 2)

    # returns as an int and swaps false->1, true->0
    return (dmatrix.mask*1 + 1)%2

def seasonal_load_data():
    pass

def monthly_load_data(month, path=data_path, dtype='snow', quiet=False):
    '''
        Loads all data files from the
        specified path, for the given month

        Returns dictionary of files where the key is the start date
        and the value is the data matrix file (weekly matrix)


    '''

    files = {} # dict with fname as key and data as value
    for dirpath, dirnames, fnames in os.walk(path):
        for fname in [f for f in fnames if f.endswith(".bin")]:

            fpath = os.path.join(dirpath, fname)
            with open(fpath, 'rb') as f:  # open function read binary file

                # getting the folder names for the key
                dirnames = dirpath.split('/')
                # if month is not the one specified, continue
                if int(dirnames[-1]) != month:
                    continue

                # make key from starting date
                key = dirnames[-2] + "-" + dirnames[-1] + "-" + fname[-19:-17]

                # Gets annoying real fast when you have a lot of files
                if not quiet:
                    print("Loading "+fname)

#                 hdr = f.read(300)  # Reading 300 byte header
                ice = np.fromfile(f, dtype=np.uint8)  # Unsigned 8 bit Integer (0-2^7)

                ## Processing
                # make the matrix of 448/304
                # scale by 250, masking out land values
                ice = ice.reshape(data_dim[0], data_dim[1])




                files[key] = masking(ice, dtype)
            continue
        else:
            continue

    return files



def basic_info(dlist):
    '''
        Expects a dict of data files
        Prints
            - The number of files,
            - Dimensions of each file
            - Total number of elements in each file
    '''

    leng = len(dlist.values())
    print("Total no. files loaded: " + str(leng))
    for f, i in zip(dlist.values(), range(leng)):
        print("File ["+str(i)+"]:")
        print("\t Shape: "+str(f.shape))
        print("\t Total entries: "+str(f.size))
        print("-------------------------------")


def load_latlon(cpath="./Data/lat_lon/", dim=data_dim):
    # loading data into dict
    coords = []
    for fname in os.listdir(cpath):
            if fname.endswith(".double"):
                print("Loading "+fname)
                with open(cpath+fname, 'rb') as f:  # open function read binary file
                    file = np.fromfile(f, dtype=float)  # Unsigned 8 bit Integer (0-2^7)
                    file = file.reshape(dim[0], dim[1]) # convert to matrix

                    coords.append(file)

    # lon is in [0], lat is in [1]
    return coords[1], coords[0]



#### PLOTTING

def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

def plotter(ice_d, coord, dx=25000, dy=25000, marg=50000, min_lat=30, contour=False,
            cmap=plt.cm.Blues, cb_tix=True, cb_marg=1, cb_range=None,
            bbox=None, month=None, cbar_label=None, save_as=None):
    '''
        Plot given list of data matricies

        TODO: rewrite to be more versatile, plot only single instance
        or different time periods

        Figure out how to standardize colors/colorbar
    '''
    # plt.rcParams["font.family"] = "Times New Roman"


    lons = sorted(set(coord[1].flatten()))
    min_lon = lons[1] # minimum non -999 val
    max_lon = lons[-1]

    for ice in ice_d.items():

        # set dimensions
        sz_x = ice[1].shape[0] * dx / 2
        sz_y= ice[1].shape[1] * dy / 2
        margin = marg

        x = np.arange(margin-sz_x, sz_x+margin, +dx)
        y = np.arange(sz_y+margin, margin-sz_y, -dy)


        plt.get_backend()

        fig = plt.figure(figsize=(12, 12))
        ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))

        cs = ax.coastlines(resolution='110m', linewidth=0.8)

        ax.gridlines()
        ax.set_extent([-180, 180, min_lat, 90], crs=ccrs.PlateCarree())

        kw = dict(central_latitude=90, central_longitude=0, true_scale_latitude=70)


        if cb_tix:
            cs = ax.pcolormesh(x, y, ice[1], cmap=cmap,
                       transform=ccrs.Stereographic(**kw), zorder=1)
            cb = fig.colorbar(cs, ax=ax, ticks=[0, 1, 2, 3, 4])
        else: # should be the case where I need a diverging color bar

            if cb_range:
                divnorm=colors.TwoSlopeNorm(vmax=cb_range[1], vcenter=0., vmin=cb_range[0])
            else:
                divnorm=colors.TwoSlopeNorm(vmax=ice[1].max()+cb_marg, vcenter=0., vmin=ice[1].min()-cb_marg)

            if contour:
                cs = ax.contourf(x, y, ice[1],
                    transform=ccrs.Stereographic(**kw), cmap=cmap, norm=divnorm)
                ax.contour(x, y, ice[1], levels=cs.levels,
                           transform=ccrs.Stereographic(**kw), colors='black')

            else:
                cs = ax.pcolormesh(x, y, ice[1], cmap=cmap, norm=divnorm,
                       transform=ccrs.Stereographic(**kw), zorder=1)

            cb = fig.colorbar(cs, ax=ax)

        if bbox:
            ax.plot([bbox[0], bbox[1], bbox[1], bbox[0], bbox[0]],
                    [bbox[3], bbox[3], bbox[2], bbox[2], bbox[3]],
                    c='black', zorder=10,
                    transform=ccrs.PlateCarree())

        if month:
            ax.set_title(m_names.get(month), fontsize=28, pad=15)
        if cbar_label:
            cb.set_label(cbar_label, rotation=270, labelpad=30, fontsize=24)

        if save_as:
            if contour: # adjusting filename if contour is true
                save_as = save_as[:-4] + "_contour." + fig_ext

            plt.savefig(save_as, format=fig_ext, dpi=300, bbox_inches = 'tight')

        if display:
            plt.show()

        # clear up ram to avoid memory issues when dealing with a lot of figs
        fig.clear()
        plt.close(fig)

        print(ice[0]) # Print filename



###### OTHER FUNCTIONS


def change_by_index(ice, indicies, sub=-999.0):
    '''
        indicies is list of tuples containing coordinates to swap
    '''
    new_dict = ice
    for m in new_dict.values():
        transplant = np.array([sub]*len(indicies))
        r, c = zip(*indicies)
        m[r, c] = transplant
    return new_dict

def get_head(d):
    '''
        Getting top key value pair from a dict
    '''
    ## This is probably an awful way of doing this but it works
    return dict([list(d.items())[0]])


def sum_years(d, start_y=start_year, end_y=end_year, dim=data_dim):
    '''
        Sum all the weeks belonging to the same year
    '''
    year_d = {}
    for y in range(start_y, end_y+1):
        year = str(y)
        year_d[year] = np.zeros(dim)

        for key, matrix in d.items():
            if key.startswith(year):
                year_d[year] += matrix
    return year_d


def combine_years(years):
    '''
        Takes the dictionary of years and creates a single matrix
    '''

    # Empty dataframe for easier appending
    df = pd.DataFrame()
    for y in sorted(years):
        df[y] = years[y].flatten()

    # return as np matrix
    return df.values


def remove_rows(m, suffix='', dtype='snow'):
    '''
        drop rows from combined matrix where var =0
        return m

        Todo: optionally delete file, add another file name to use ice
    '''

    # check if std file already exists
    file = logs_path + dtype+"_stds_"+suffix+".txt"
    if not Path(file).exists():
        with open(file, 'a') as f: # write to file if it doesnt already exist
            for row in range(m.shape[0]):
                s = m[row].std()
                if (s!=0.0):
                    f.write(str(row)) # write row index to file
                    f.write('\n')
    return cut_rows(m, file)


def cut_rows(m, fname):
    '''
        Function to actually drop the rows from m
        that have 0 std
    '''
    # reading file of indexes
    indicies = [int(line.strip()) for line in open(fname, 'r')]

    # get np matrix of indicies
    return m[np.ix_(indicies)], indicies


def reinsert_rows(vec, suffix='', dtype="snow", dim=data_dim):
    '''
        Read the list of indicies then insert zeros,
        then return vector

        fix this: may need to reinsert rows from vec 1 by 1
    '''

    # list on indicies not removed from original vector
    file = logs_path + dtype+"_stds_"+suffix+".txt"
    with open(file) as f:
        lines = [int(line.rstrip()) for line in f]

    z = np.zeros(dim[0]*dim[1])

    # replacing with values from vec
    z[lines] = vec

    return z

def write_evr(evr, save_as):
    with pd.ExcelWriter(tabl_path+save_as) as writer:
        evr.to_excel(writer, index=False)

def write_timeseries(ts, save_as):
    with pd.ExcelWriter(tabl_path+save_as) as writer:
        for m, t, in ts.items():
            t.to_excel(writer, sheet_name = str(m), index=False)

def get_box_data(data, lat, lon, box):
    '''
    Return a subset of the data as defined by a box of coordinates.
    The data variable must be the non abbreviated, aggregated dataset (i.e. its shape is 720^2 rows and  42 columns)
    The lat/lon variables are still in their flattend form
    And the box variable is a tuple that goes (lon min, lon max, lat min, lat max)
    '''

    # Get the indicies of longitude and latitude that are within the range
    lon_idx = np.where( (lon >= box[0]) & (lon <= box[1]) )
    lat_idx = np.where( (lat >= box[2]) & (lat <= box[3]) )

    # The indicies of pixels within the box are in the intersection of lon and lat indicies within the range
    return data[np.intersect1d(lon_idx, lat_idx, assume_unique=True), :]


def get_regression_coeffs(data, domain, N, two_tailed=True):

    # threshold based on 95% confidence interval and 40 degrees of freedom
    dof = data.shape[1] - 2 # 40
    tval = 2.021
    if not two_tailed:
        tval = 1.684

    # finding regression coefficients
    rcoeffs = np.zeros(N)
    significant = [] # contains indicies of statistically significant pixels
    for pixel in range(N):
        lreg = LinearRegression().fit(domain, data[pixel,:])
        # slopes
        rcoeffs[pixel] = lreg.coef_

        ### t-test
        # residuals
        ypred = lreg.predict(domain)
        resid = data[pixel,:] - ypred

        # variance of residuals
        resid_var = np.square(resid).sum() / dof

        # standard error of rcoeff
        s_a = resid_var / np.square( data[pixel,:] - data[pixel,:].mean() ).sum()

        # t
        t = np.abs(rcoeffs[pixel]) / np.sqrt(s_a)
        # print(t)
        if t > tval:
            significant.append(pixel)






    return rcoeffs, significant









#################################################

def seasonal_data_handler(season):
    # load all files for each month and concatenate them together
    #
    pass

def monthly_data_handler(month):
    '''
        Fairly simple, just a call to the monthly load data function

        Probably don't need a seperate function just for this but it will help with abstraction later on
    '''
    data = monthly_load_data(month=month, dtype='snow', quiet=True)
    return data

def seasonal_computation_handler(season, lat, lon):
    pass

def monthly_computation_handler(month, lat, lon):
    # Same methodology as in monthly but with a couple slight differences to the code

    tseries = pd.DataFrame()

    print("Loading/prepping data...")
    ### LOADING/PREPPING DATA
    snow_d = monthly_data_handler(month)

    year_snow_d = sum_years(snow_d)
    snow_combined = combine_years(year_snow_d)

    # sets max value as 4 instead of 5
    # given the format of this matrix, you would want to apply it to all the columns,
    # but given that the function only works on individual elements for a single month, applying it across all elements row-wise will also work
    snow_combined = np.vectorize(lambda x: min(x, 4)/4)(snow_combined)

    # getting the abbreviated dataset as well as their indicies in the orgiginal dataset
    snow_comb, indicies = remove_rows(snow_combined, suffix=str(month))
    x_time = np.arange(0, snow_comb.shape[1], 1).reshape(-1, 1)

    ### SETTING DATA MATRIX, X

    ## NO DETRENDING
    # print(snow_comb[:10])
    # print(snow_comb.shape)

    X = StandardScaler().fit_transform(snow_comb.T).T

    # print(X.shape)
    # print(snow_comb.shape)



    ########### PCA ############
    ## COVARIANCE CALCULATION
    print("Starting PCA...")
    snow_fpca = fpca.FastPCA(quiet=True)
    snow_fpca.calculate(X) # transpose to put features in rows
    pc = snow_fpca.getPCs()
    eof = snow_fpca.getEOFs()
    evr = snow_fpca.getExpVar()


    ######## PLOTTING #########
    print("Plotting...")
    xlocs = np.array([-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 48])
    xlbls = ['', '1979', '1984', '1989', '1994', '1999', '2004', '2009', '2014', '2019', '']

    time_save_as = figs_path + str(month) + "/snow_time/pc_timeseries"
    lmap_save_as = figs_path + str(month) + "/snow_lmap/map_loading"
    rmap_save_as = figs_path + str(month) + "/snow_rmap/map_reg_coeff"

    # finding regression coefficients
    rcoeffs = np.zeros(X.shape[0])
    for pixel in range(X.shape[0]):
        rcoeffs[pixel] = LinearRegression().fit(x_time, snow_comb[pixel,:]).coef_

    #### Plot Regression coefficient map

    rcoeffs, significant = get_regression_coeffs(snow_comb, x_time, X.shape[0]) # add return value here, also need to track indicies when rows are inserted
    print(significant)

    # reinserting and plotting
    rcoeffs_map = reinsert_rows(rcoeffs, suffix=str(month)).reshape(data_dim[0], data_dim[1])
    # turn into dict and plot
    plotter({"Regression coefficient map" : rcoeffs_map}, coord=(lat, lon), dx=25000, dy=25000,
            marg=0, min_lat=30, cmap=plt.cm.get_cmap('coolwarm_r'), cb_tix=False, cb_marg=0.005,
            month=month, cbar_label="No. weeks / year", save_as=rmap_save_as+"."+fig_ext)

    # Plot heatmaps over specified boundary for a given month
    while(len(hmaps)>0 and month==hmaps[0][0]): # check if current month has boundary box
        bbox = hmaps.pop(0)[1] # tuple of coordinates

        # file names
        rmap_box_save_as = figs_path + str(month) + "/snow_rmap/map_rcoeff_bbox_" + str(bbox[0]) + "_" + str(bbox[1]) + "_"+ str(bbox[2]) + "_" + str(bbox[3])
        hmap_save_as = figs_path + str(month) + "/snow_rmap/hmap_" + str(bbox[0]) + "_" + str(bbox[1]) + "_"+ str(bbox[2]) + "_" + str(bbox[3])


        # replot regression coefficient map with bounding box
        plotter({"Regression coefficient map with bbox" : rcoeffs_map}, coord=(lat, lon), dx=25000, dy=25000,
            marg=0, min_lat=30, cmap=plt.cm.get_cmap('coolwarm_r'), cb_tix=False, cb_marg=0.005, bbox=bbox,
            month=month, cbar_label="No. weeks / year", save_as=rmap_box_save_as+"."+fig_ext)

        # get specified pixes based on coordinates
        hmap_data = get_box_data(snow_combined, lat.flatten(), lon.flatten(), bbox)

        # plot/save "heatmap"
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(hmap_data, interpolation='nearest', cmap=plt.cm.get_cmap('coolwarm_r'), aspect='auto')
        plt.xticks(xlocs[1::2], xlbls[1::2])
        plt.xlabel("years", fontsize=22)
        plt.ylabel("pixels", fontsize=22)

        if save:
            plt.savefig(hmap_save_as+"."+fig_ext, format=fig_ext, dpi=300, bbox_inches="tight")
        if display:
            plt.show()

        fig.clear()
        plt.close(fig)



    ## PLOT PC TIMESERIES
    for y in range(plt_rng):

        snow = StandardScaler().fit_transform(pc[y].reshape(-1,1))
        fig = plt.figure(figsize=(12, 8))
        trendline = LinearRegression().fit(x_time, snow).predict(x_time)
        plt.plot(snow, label="PC "+str(y+1))
        plt.scatter(x_time, snow)
        plt.plot(trendline, c='r', label='trendline')
        plt.xticks(xlocs, xlbls)
        plt.legend(loc='best')
        plt.grid()

        tseries["PC_"+str(y+1)] = snow.reshape(-1).tolist() # collection of all standardized pcs

        if save:
            plt.savefig(time_save_as+str(y+1)+"."+fig_ext, format=fig_ext, dpi=300, bbox_inches = 'tight')

        if display:
            plt.show()

        fig.clear()
        plt.close(fig)


    ## PLOT LOADING MAPS
    # rows contain eigenvectors of X^T X
    for y in range(plt_rng):
        nvec = reinsert_rows(eof[:,y], suffix=str(month))

        # reshape
        nvec = nvec.reshape(data_dim[0], data_dim[1])

        # turn into dict and plot
        plotter({"Loadingvector "+str(y):nvec}, coord=(lat, lon), dx=25000, dy=25000,
            marg=0, min_lat=30, contour=False, cmap=plt.cm.get_cmap('coolwarm_r'), cb_tix=False,
            month=month, save_as=lmap_save_as+str(y+1)+"."+fig_ext)

    return tseries, evr


def main():
    evr = pd.DataFrame() # explained var
    timeseries = {}

    # loading lat/lon
    lat, lon = load_latlon()

    if mode=='monthly':
        for m in m_rng:
            print()
            print("Month No. " + str(m))
            ts, ev = monthly_computation_handler(m, lat, lon)
            evr[str(m)] = ev
            timeseries[m] = ts
        if save:
            print("\nWriting results to spreadsheet.")
            write_evr(evr, save_as="explained_var_ratio.xlsx")
            write_timeseries(timeseries, "pc_timeseries.xlsx")
    elif mode=="seasonal":
        for s in seasons.keys():
            print(s)
            seasonal_computation_handler(s, lat, lon)


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting.")
        sys.exit(0)
