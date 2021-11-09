# fuctions for reading / writing .fits files, and other io things
# data handled with dataframes
from project_libs import *
import allvars as av

# set the filepaths
def set_filepaths():
    av.path_data = '../data/' + av.cat_group + '/cleaned/'
    av.path_output = '../output/' + av.cat_group + '/' + av.gtype + '/'
    av.path_output_pk = '../output/' + av.cat_group + '/' + av.gtype + '/powerspec/'
    av.path_output_cf = '../output/' + av.cat_group + '/' + av.gtype + '/corrfunc/'

    # if output directory doesnt exist then make it
    if not os.path.exists(av.path_output):
        os.mkdir(av.path_output)
    if not os.path.exists(av.path_output_pk):
        os.mkdir(av.path_output_pk)
    if not os.path.exists(av.path_output_cf):
        os.mkdir(av.path_output_cf)



# read a .fits file into a dataframe and look at angular scatter of data
def read_fits(path, print_df):
    t = Table.read(path) # read file into table
    temp_df = Table.to_pandas(t) # send table to df
    del t
    if (print_df == True):
        print(temp_df[:5])

    # anuglar scatter plot
    plt.figure(figsize=(10,5))
    plt.xlabel('RA', size=15)
    plt.ylabel('DEC', size=15)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title('Angular scatter of data', size=15)
    plt.scatter(temp_df['RA'][::100],temp_df['DEC'][::100], color='red', s=0.05)
    plt.savefig(av.path_output + 'data_ang_scatter.pdf', bbox_inches='tight')
    plt.close()
    return temp_df




# write dataframe to a .fits file
def write_fits(df, path):
    return 1
