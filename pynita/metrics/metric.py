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

def computeMetrics(results_dic, vi_change_thresh, run_thresh, time_step):
    
    # ---
    # 1. extract information from results_dic
    knots = results_dic['final_knots'];
    coeffs = results_dic['final_coeffs'];
    rises = results_dic['rises'];
    runs = results_dic['runs'];
    runs_in_days = results_dic['runs_days'];

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
        dist_flags = (change_percent<vi_change_thresh) & (runs_in_days<=run_thres)
        
        # 4. disturbance detection 
        start_ls = []
        end_ls = [] 
        n = 0
        m = 0
        for i in range(len(dist_flags)):
            if dist_flags[i]:
                n = n + 1
                if m == 1:
                    start_ls.append(i)
                m = 0
            else:
                if n != 0:
                    end_ls.append(i-1)  
                n = 0
                m = 1
                
        dist_locs = [(start_ls[i], end_ls[i]+1) for i in range(len(start_ls))] 
        
        # ---
        # 5. disturbance metric calculation 
        # 5.a no disturbance 
        if len(dist_locs) == 0: 
            num_dist = 0
            cum_mag_dist = -999
            dist_date_before = -999
            dist_date_nadir = -999
            dist_duration = -999
            dist_slope = -999
            dist_coeff_nadir = -999
            post_dist_slope = -999
            post_dist_mag = -999
            dist_mag = -999
            dist_coeff_before = -999
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
                post_dist_slope = -999 # output 
                post_dist_mag = -999 
            else:
                next_loc = [dist_loc[1], dist_loc[1]+1]
                post_dist_mag = coeffs[next_loc[1]] - coeffs[next_loc[0]] # output 
                post_dist_slope = post_dist_mag / (knots[next_loc[1]] - knots[next_loc[0]]) # output 
                  
    except:
        num_dist = -999
        cum_mag_dist = -999
        dist_date_before = -999
        dist_date_nadir = -999
        dist_duration = -999
        dist_slope = -999
        dist_coeff_nadir = -999
        post_dist_slope = -999
        post_dist_mag = -999
        dist_mag = -999
        dist_coeff_before = -999
        interp_pts = -999 
    
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


        
                
                