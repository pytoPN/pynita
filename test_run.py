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
min_complex = 2
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
            cand_idx, coeff = uf.findCandidate(dist, filt_dist, pct, y, loc_set, filter_opt);
            
            if cand_idx == -999:
                break
            
            knot_set, coeff_set, loc_set = uf.updateknotcoeffSet(knot_set, coeff_set, loc_set, x, cand_idx, coeff)
            dist_new = uf.calDistance(knot_set, coeff_set, pts)
            mae_ortho.append(uf.calMae(dist_new))
            del dist, dist_new, cand_idx, coeff 
            
        complexity_count = len(knot_set)-1
                  
        mae_final = mae_ortho[complexity_count-1] 
                
        
        knots_max = knot_set
        coeffs_max = coeff_set
           
        keep_knots = knot_set
        keep_coeffs = coeff_set
                 
        yinterp1 = np.interp(x, knots_max, coeffs_max) # method linear as default
        y_pos_flags = (y - yinterp1) > 0  
             
             
        keep_idx = []
        mae_ortho_holder = []
        bic_remove = []
        knot_storage = []
        coeff_storage = []
             
        for i in range(0, complexity_count - min_complex + 1)):
            keep_idx_it = uf.genKeepIdx(keep_knots, keep_coeffs, pts, pct, y_pos_flags)
            keep_idx.append(keep_idx_it)
            
            dist = uf.calDistance(keep_knots, keep_coeffs, pts)
            ortho_err = dist.min(axis=1)
            mae_ortho_holder.append(uf.calMae(dist))
                      
            ortho_err[y_pos_flags] = ortho_err[y_pos_flags] * pct
            ortho_err[~y_pos_flags] = ortho_err[~y_pos_flags] * (100 - pct)
            
            bic_remove_it = uf.calBIC(ortho_err, keep_knots, penalty)
            bic_remove.append(bic_remove_it)  
        
            if i == 0:
                knot_storage.append(knots_max)
                coeff_storage.append(coeffs_max)
            else: 
                knot_storage.append(keep_knots[keep_idx[i]])
                coeff_storage.append(keep_coeffs[keep_idx[i]])
                     
                keep_coeffs.append(keep_idx[i])
                keep_knots.append(keep_idx[i])
            
        bic_idx = np.where(bic_remove == bic_remove.min)[0]
        keep_coeffs = coeff_storage[bic_idx]
        keep_knots = knot_storage[bic_idx]
        mae_final = mae_ortho_holder[bic_idx]    
                     
        complexity = len(keep_knots) - 1
        final_knots = keep_knots
        final_coeffs = keep_coeffs
                
        mae_final_ortho = mae_final
        mae_linear = mae_lin
        noise_out = noise
    else:
        complexity = 1
        final_knots = [x.min, x.max ]
        final_coeffs = [first_coeff, last_coeff]
        mae_final_ortho = mae_lin
        mae_linear = mae_lin
        noise_out = noise
     
    rises = np.diff(final_coeffs)
    runs = np.diff(final_knots)   
    runs_days = runs / 1000 * 365
    
except ValueError as err:
        complexity = -999
        final_knots = -999
        final_coeffs = -999
        mae_final_ortho = -999
        mae_linear = -999
        noise_out = -999
        rises = -999
        runs = -999
        runs_days = -999
        pts = -999

results_dic = {'complexity': complexity,
                'final_knots': final_knots,
                'final_coeffs': final_coeffs,  
                'mae_linear': mae_linear,
                'mae_final_ortho': mae_final_ortho,
                'noise_out': noise_out,
                'rises': rises, 
                'runs': runs,
                'runs_days': runs_days,
                'pts': pts}

#return results_dic    
        
