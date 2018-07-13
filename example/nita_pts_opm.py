import numpy as np
from pynita import *

# assign the ini file 
ini = 'user_configs.ini'

# initialize the nita object
nita = nitaObj(ini)

# start logging     
nita.startLog()

nita.loadPts(info_column='Name')

# draw trajectories for selected OBJECTIDs 
nita.drawPts([1, 2, 4], plot_title=True)

# draw trajectories for all OBJECTIDs 
nita.drawPts([9999], plot_title=True)

# set paramters for optmization 
nita.setOpmParams()

# set paramters for optmization  with overwrite
nita.setOpmParams(**{'bail_thresh_set': [1], 'noise_thresh_set': [1], 'penalty_set': [1, 2], 'filt_dist_set': [3, 5], 'pct_set': [70], 'max_complex_set': [10]})

# parameter optimization 
nita.paramOpm()

# stop logging 
nita.stopLog()