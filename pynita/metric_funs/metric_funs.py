"""
xxx
Created on Jun 5, 2018
@author: Leyang Feng
@email: feng@american.edu
@Project: pynita
License:  
Copyright (c) 
"""

import numpy as np 
from scipy import ndimage
import matplotlib.pyplot as plt

def computeMetrics(results_dic, vi_change_thresh, run_thresh, time_step):
    
    # ---
    # 1. extract information from results_dic
    knots = results_dic['final_knots'];
    coeffs = results_dic['final_coeffs'];
    rises = results_dic['rises'];
    #runs = results_dic['runs'];
    runs_in_days = results_dic['runs_days'];
    
    if knots[0] != -999:
    
        try:
            # ---
            # 2. interpolation 
            knot_first = np.floor(knots[0]/1000)
            knot_last = np.floor(knots[-1]/1000)+1
            all_knots_dis = np.sort(np.unique(np.concatenate((np.arange(knot_first, knot_last+1, time_step) * 1000, knots))))
            all_knots_dis = all_knots_dis[(all_knots_dis >= knots[0]) & (all_knots_dis <= knots[-1])]
            all_coeffs_interp = np.interp(all_knots_dis, knots, coeffs) 
            interp_pts = np.column_stack((all_knots_dis, all_coeffs_interp)) # output 
            
            # ---
            # 3. change percent and slope calculation 
            change_percent = rises/abs(coeffs[0:-1])
            dist_flags = (change_percent<vi_change_thresh) & (runs_in_days<=run_thresh)
            dist_bin = [1 if flag else 0 for flag in dist_flags]
            label, num_features = ndimage.label(dist_bin)
            
            # 4. disturbance detection
            dist_locs = []
            for i in range(num_features):
                label_val = i + 1 
                locs = np.where(label == label_val)[0]
                dist_locs.append((locs[0], locs[-1]+1))
            
            # ---
            # 5. disturbance metric calculation 
            # 5.a no disturbance 
            if len(dist_locs) == 0: 
                num_dist = 0
                cum_mag_dist = np.nan
                dist_date_before = np.nan
                dist_date_nadir = np.nan
                dist_duration = np.nan
                dist_slope = np.nan
                dist_coeff_nadir = np.nan
                post_dist_slope = np.nan
                post_dist_mag = np.nan
                dist_mag = np.nan
                dist_coeff_before = np.nan
            # 5.b get metrics for largest disturbance as default 
            else: 
                num_dist = len(dist_locs) # output 
                
                dist_mags = []
                for dist_loc in dist_locs:
                    coeff_st = coeffs[dist_loc[0]]
                    coeff_ed = coeffs[dist_loc[1]]
                    mag = coeff_st-coeff_ed
                    dist_mags.append(mag)
                cum_mag_dist = sum(dist_mags) # output 
                
                dist_mags = np.array(dist_mags)
                dist_idx = int(np.where(dist_mags==dist_mags.min())[0])            
                dist_loc = dist_locs[dist_idx]
                
                dist_date_before = knots[dist_loc[0]] # output 
                dist_date_nadir = knots[dist_loc[1]] # output 
                dist_duration = dist_date_nadir - dist_date_before # output
                
                dist_coeff_before = coeffs[dist_loc[0]] # output 
                dist_coeff_nadir = coeffs[dist_loc[1]] # output 
                dist_mag = dist_coeff_before - dist_coeff_nadir # output 
                
                dist_slope = -dist_mag/dist_duration # output 
                
                # first 'recovery' or just first segment after default disturbance 
                if dist_loc[1] == (len(knots)-1): # the case that the disturbance is the last sagment 
                    post_dist_slope = np.nan # output 
                    post_dist_mag = np.nan
                else:
                    next_loc = [dist_loc[1], dist_loc[1]+1]
                    post_dist_mag = coeffs[next_loc[1]] - coeffs[next_loc[0]] # output 
                    post_dist_slope = post_dist_mag / (knots[next_loc[1]] - knots[next_loc[0]]) # output 
                      
        except:
            num_dist = np.nan
            cum_mag_dist = np.nan
            dist_date_before = np.nan
            dist_date_nadir = np.nan
            dist_duration = np.nan
            dist_slope = np.nan
            dist_coeff_nadir = np.nan
            post_dist_slope = np.nan
            post_dist_mag =np.nan
            dist_mag = np.nan
            dist_coeff_before = np.nan
            interp_pts = np.nan
    
    else: 
        num_dist = np.nan
        cum_mag_dist = np.nan
        dist_date_before = np.nan
        dist_date_nadir = np.nan
        dist_duration = np.nan
        dist_slope = np.nan
        dist_coeff_nadir = np.nan
        post_dist_slope = np.nan
        post_dist_mag =np.nan
        dist_mag = np.nan
        dist_coeff_before = np.nan
        interp_pts = np.nan
        
    metrics_dic = {'num_dist': num_dist,
                   'cum_mag_dist': cum_mag_dist,
                   'dist_date_before': dist_date_before, 
                   'dist_date_nadir': dist_date_nadir, 
                   'dist_duration': dist_duration, 
                   'dist_slope': dist_slope,
                   'dist_mag': dist_mag,
                   'dist_coeff_before': dist_coeff_before,
                   'dist_coeff_nadir': dist_coeff_nadir,
                   'post_dist_slope': post_dist_slope,
                   'post_dist_mag': post_dist_mag,
                   'interp_pts': interp_pts}
        
    return metrics_dic

#%%
#def stretchMI(MI_1d):

#%%
def plotMI(MI_2d): 
    fig, ax = plt.subplots()  
    mappable = ax.matshow(MI_2d)
    fig.colorbar(mappable)


#%%          
def MI_complexity(results_dics):
    values_1d = np.array([dic['complexity'] for dic in results_dics])
    
    return(values_1d)

#%%
def calDistDate(metrics_dic, option='middle'):
    dist_date_before = metrics_dic['dist_date_before']
    dist_date_nadir = metrics_dic['dist_date_nadir']

    if option == 'beginning':
        dist_date = dist_date_before
    elif option == 'middle':
        dist_date = (dist_date_before + dist_date_nadir) / 2
    elif option == 'end':
        dist_date = dist_date_nadir
    else: 
        raise RuntimeError('ERROR: invalid option!')
    
    return dist_date
    
#%%
def MI_distDate(metrics_dics, option='middle'):
    vals_1d = np.array([calDistDate(metrics_dic, option=option) for metrics_dic in metrics_dics])
    
    return(vals_1d)

#%%
def MI_distDuration(metrics_dics): # in days
    dist_durations = [metrics_dic['dist_duration'] for metrics_dic in metrics_dics] 
    vals_1d = np.array([np.floor(duration/1000)*365+np.mod(duration, 1000)/1000*365 for duration in dist_durations])
    
    return vals_1d
    