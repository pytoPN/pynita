import numpy as np
from pynita import *

parser = argparse.ArgumentParser()
parser.add_argument('-config_file', type=str, help='Full path with file name to INI configuration file.')
args = parser.parse_args()

# assign the ini file 
if args.config_file is None:
    ini = 'user_configs.ini'
else:
    ini = args.config_file

# initialize the nita object
nita = nitaObj(ini)

# start logging     
nita.startLog()
    
# load points csv
# use info_column to specify the column containing descriptions in the csv, if no such column exists, use nita.loadPts() 
nita.loadPts(info_column='Name')

###
# view the loaded table
nita.pts
# view the reference table 
nita.ref_pts
# check the amount of unique OBJECTIDs
nita.pts_count
# check the OBJECTIDs
nita.pts_OBJECTIDs
###

# run all points and view as panel plot  
nita.runPts([9999], plot=True, max_plot=25, showdata='fit', colorbar=False, plot_title=True)

# run specific OBJECTID
results_dic = nita.runPts([4], plot=True, showdata='fit', colorbar=True, plot_title=True)    

# run specific OBJECTID with parameter overwrite
results_dic = nita.runPts([4], plot=True, showdata='fit', colorbar=True, plot_title=True, **{'min_complex': 5, 'max_complex': 20})

# stop logging 
nita.stopLog()
