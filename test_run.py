#%%
import os 
import numpy as np
import pandas as pd
import math
import utility_funs as uf

#%%
os.chdir(r'C:/Users/feng/Documents/pynita')

#%%
test_data = pd.read_csv('./input/test_data.csv')

px = np.array(test_data.px)
date_vec = np.array(test_data.date_vec)
doy_vec = np.array(test_data.doy_vec)

#%%
value_limits = [0,1];
doy_limits = [[1,100],[260,365]]; 
date_limits = [1995000,2010000];
bail_thresh = 1.2; 
noise_thresh = 1; 
penalty = 1; 
filt_dist = 5; 
pct = 50; 
max_complex = 7;
min_complex = 1;
compute_mask = 1;
filter_opt = 'movcv';

#%%
unq_idx = np.unique(date_vec,return_index=True)[1]
px = px[unq_idx] 
date_vec = date_vec[unq_idx]
doy_vec = doy_vec[unq_idx]

try:
    x = date_vec 
    y = px 
    
    x, y, doy_vec = uf.filterLimits(x, y, doy_vec, value_limits, date_limits, doy_limits)
    
    noise = np.median(np.absolute(np.diff(y)))
    
    diff_holder = np.diff(y)
    non_noise_flags = np.append(np.array([False]),np.absolute(diff_holder) <= noise_thresh)
    x = x[non_noise_flags]
    y = y[non_noise_flags]
    x_len = len(x)
    
    if x_len <= (filt_dist*2):
         raise ValueError('Not enough data pairs!')
    
    first_coeff = np.percentile(y[0:(filt_dist-1)], pct, interpolation='midpoint') # use 'midpoint' to mimic matlab
    last_coeff = np.percentile(y[-filt_dist:],pct, interpolation='midpoint');

    knot_set = np.array([x[0], x[-1]])      
    coeff_set = np.array([first_coeff, last_coeff])
    loc_set = np.array([0, x_len])
    
    
    
    except:
        
