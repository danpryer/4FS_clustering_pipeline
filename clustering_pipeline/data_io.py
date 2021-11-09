# python file for data input/output

# from imports import *
import pandas as pd
from astropy.table import Table
import pickle
import gc


# function to read a structured table file (eg a .fits file) and load into a dataframe
def read_to_df(path):
    df = pd.DataFrame()

    t = Table.read(path)  # read file into table

    df = Table.to_pandas(t)  # send table to df
    del t
    gc.collect()

    return df


# function to save a pd dataframe to a fits file
def save_df_fits(df, path):
    t = Table.from_pandas(df)  # to astropy table
    t.write(path + ".fits", "w")  # to fits file




# save an object (like a class) to a pickle file
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# load the above saved object (note, filename will need to inclue the file extension, e.g. .pkl)
def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)
