#%%
import os 
import numpy as np
import pandas as pd
import utility_funs as uf

#%%
os.chdir(r'C:/Users/feng/Documents/pynita')

#%%
test_data = pd.read_csv('./input/test_data.csv')

px = np.array(test_data.px)
date_vec = np.array(test_data.date_vec)
doy_vec = np.array(test_data.doy_vec)

#%%
value_limits = [0,1]
doy_limits = [[1,100],[260,365]] 
date_limits = [1995000,2010000]
bail_thresh = 1.2
noise_thresh = 1
penalty = 1
filt_dist = 5
pct = 50
max_complex = 7
min_complex = 1
compute_mask = 1
filter_opt = 'movcv'

#%%
unq_idx = np.unique(date_vec,return_index=True)[1]
px = px[unq_idx] 
date_vec = date_vec[unq_idx]
doy_vec = doy_vec[unq_idx]

#%%
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
    
    first_coeff = np.percentile(y[0:(filt_dist)], pct, interpolation='midpoint') # use 'midpoint' to mimic matlab
    last_coeff = np.percentile(y[-filt_dist:],pct, interpolation='midpoint')

    knot_set = np.array([x[0], x[-1]])      
    coeff_set = np.array([first_coeff, last_coeff])
    loc_set = np.array([0, x_len - 1])
    
    pts = np.column_stack((x, y))
    
    dist_init = uf.calDistance(knot_set, coeff_set, pts)
    mae_lin = uf.calMae(dist_init)
    
    if (mae_lin/noise > bail_thresh) & compute_mask == 1:
        
        mae_ortho = []
        mae_ortho.append(mae_lin)
        
        for i in range(1, max_complex):
                
            dist = uf.calDistance(knot_set, coeff_set, pts)
            cand_loc, coeff = uf.findCandidate(dist, filt_dist, pct, y, loc_set, filter_opt);
            
            if cand_loc == -999:
                break
            
            knot_set, coeff_set, loc_set = uf.updateknotcoeffSet(knot_set, coeff_set, loc_set, x, cand_loc, coeff)
            dist_new = uf.calDistance(knot_set, coeff_set, pts)
            mae_ortho.append(uf.calMae(dist_new))
            del dist, dist_new, cand_loc, coeff 
            
        complexity_count = len(knot_set)-1
        
        knots_max = knot_set
        coeffs_max = coeff_set
                 
        yinterp1 = np.interp(x, knot_set, coeff_set) # method linear as default
        y_pos_flags = (y - yinterp1) > 0  
             
        if complexity_count < min_complex:
            exit_count = complexity_count
        else: 
            exit_count = min_complex
        
        end_count = min_complex
        
        mae_ortho_holder = []
        bic_remove = []
        knot_storage = []
        coeff_storage = []
        
        knot_storage.append(knot_set)
        coeff_storage.append(coeff_set) 
        dist_init = uf.calDistance(knot_set, coeff_set, pts)
        mae_ortho_holder.append(uf.calMae(dist_init)) 
        ortho_err = dist_init.min(axis=1)
        ortho_err[y_pos_flags] = ortho_err[y_pos_flags] * pct
        ortho_err[~y_pos_flags] = ortho_err[~y_pos_flags] * (100 - pct)
        bic_remove.append(uf.calBIC(ortho_err, knot_set, penalty))
        
        for i in range(1, complexity_count - exit_count + 1):
            keep_loc = uf.genKeepIdx(knot_set, coeff_set, pts, pct, y_pos_flags)
            
            knot_set = knot_set[keep_loc]
            coeff_set = coeff_set[keep_loc]
            knot_storage.append(knot_set)
            coeff_storage.append(coeff_set)
            
            dist = uf.calDistance(knot_set, coeff_set, pts)
            mae_ortho_holder.append(uf.calMae(dist))
            ortho_err = dist.min(axis=1)
            ortho_err[y_pos_flags] = ortho_err[y_pos_flags] * pct
            ortho_err[~y_pos_flags] = ortho_err[~y_pos_flags] * (100 - pct)
            
            bic_remove.append(uf.calBIC(ortho_err, knot_set, penalty))
            
        bic_idx = int(np.where(bic_remove == min(bic_remove))[0])
        knots_final = knot_storage[bic_idx]
        coeffs_final = coeff_storage[bic_idx]
        mae_final = mae_ortho_holder[bic_idx]    
        complexity_final = len(knots_final) - 1

    else:
        knot_final = np.array([x[0], x[-1]])      
        coeff_final = np.array([first_coeff, last_coeff])
        mae_final = mae_lin
        complexity_final = 1
     
    rises = np.diff(coeffs_final)
    runs = np.diff(knots_final)   
    runs_days = runs / 1000 * 365
    
except ValueError as err:
    complexity_final = -999
    knots_final = -999
    coeffs_final = -999
    mae_final = -999
    mae_lin = -999
    noise = -999
    rises = -999
    runs = -999
    runs_days = -999
    pts = -999

results_dic = {'complexity': complexity_final,
               'final_knots': knots_final,
               'final_coeffs': coeffs_final,  
               'mae_linear': mae_lin,
               'mae_final': mae_final,
               'noise': noise,
               'rises': rises, 
               'runs': runs,
               'runs_days': runs_days,
               'pts': pts}

#return results_dic    
        
