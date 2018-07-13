import argparse
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

# load image stack 
nita.loadStack()

# Optional step -- set compute mask
# define your mask in here
#user_mask = 
#nita.setMask(user_mask)

# run stack with parallelazition 
#nita.runStack(parallel=True, workers=2)
# run stack without parallelazition 
nita.runStack(parallel=False)

# run compute metrics with parallelazition 
#nita.computeStackMetrics(parallel=True, workers=2)
# run compute metrics without parallelazition 
nita.computeStackMetrics(parallel=False)

# create metric images 
nita.MI_complexity(plot=False, save=True, fn='complexity.tiff')
nita.MI_distDate(option='middle', plot=False, save=True, fn='distdate.tiff')
nita.MI_distDuration(plot=False, save=True, fn='distduration.tiff')
nita.MI_distMag(plot=False, save=True, fn='distMag.tif')
nita.MI_distSlope(plot=False, save=True, fn='distSlope.tif')
nita.MI_linearError(plot=False, save=True, fn='linerror.tif')
nita.MI_noise(plot=False, save=True, fn='noise.tif')
nita.MI_bailcut(plot=False, save=True, fn='bailcut.tif')
nita.MI_postDistSlope(plot=False, save=True, fn='postdistslope.tif')
nita.MI_postDistMag(plot=False, save=True, fn='postdistmag.tif')
nita.MI_dateValue(2005000, plot=False, save=True, fn='datevalue.tif')
nita.MI_valueChange(start_date=-9999, end_date=9999, option='diff', plot=False, save=True, fn='valuechange1.tif')
nita.MI_valueChange(start_date=2002000, end_date=2016900, option='diff', plot=False, save=True, fn='valuechange2.tif')
nita.MI_recovery(1, option='diff', plot=False, save=True, fn='recovery.tif')
nita.MI_recoveryCmp(1, plot=False, save=True, fn='recoverycmp.tif')

# retrivel nita results by x and y 
#results_dic = nita.getPixelResults([8, 5])

# re-run by x and y 
#results_dic = nita.runPixel([8, 5], use_compute_mask=False, plot=True, showdata='fit', colorbar=True)
# re-run by x and y with parameter overwrite 
#results_dic = nita.runPixel([8, 5], use_compute_mask=False, plot=True, showdata='fit', colorbar=True, **{'value_limits': [-0.5, 1], 'min_complex': 6})

# retrivel metrics by x and y 
#metrics_dic = nita.getPixelMetrics([8, 5])
# compute metrics based on results 
#metrics_dic = nita.computeMetrics(results_dic)
# compute metrics based on results with parameter overwrite
#metrics_dic = nita.computeMetrics(results_dic, **{'run_thresh': 3000})

# stop log 
nita.stopLog()
