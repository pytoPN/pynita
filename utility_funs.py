import numpy as np
import pandas as pd
import math
from scipy import stats  

#%%
def filterLimits(x, y, doy_vec, value_limits, date_limits, doy_limits):
   
    # ---
    # (1)
    if sum(date_limits) == 0:
        x = x
        y = y
        doy_vec = doy_vec
    else:
        if date_limits[0] == -9999 & date_limits[1] != -9999:
            start_date = x[0]
            end_date = date_limits[1]
        elif date_limits[0] != -9999 & date_limits[1] == 9999:
            start_date = date_limits[0]
            end_date = x[-1]
        else:
            start_date = date_limits[0]
            end_date = date_limits[-1]
    
    date_flags = (x >= start_date) & (x <= end_date)
    
    x = x[date_flags]
    y = y[date_flags]
    doy_vec = doy_vec[date_flags]
    
    # ---
    # (2)
    doy_flags = np.full(doy_vec.shape, False, dtype=bool)
    for doy_limit in doy_limits:
        doy_flag_it = (doy_vec >= doy_limit[0]) & (doy_vec <= doy_limit[1])
        doy_flags = doy_flags | doy_flag_it
    
    x = x[doy_flags]
    y = y[doy_flags]
    doy_vec = doy_vec[doy_flags]
    
    # ---
    # (3)
    non_nan_flags = ~np.isnan(y)
    x = x[non_nan_flags]
    y = y[non_nan_flags]
    doy_vec = doy_vec[non_nan_flags]
    
    # ---
    # (4)
    value_flags = np.logical_and(y >= value_limits[0], y <= value_limits[1]) 
    x = x[value_flags]
    y = y[value_flags]
    doy_vec = doy_vec[value_flags]
    
    return x, y, doy_vec

#%%
def distancePointEdge(points, edge):
    # Python implementation of MATLAB function distancePointEdge created by David Legland 
    
    # edge shoud be in the format of [x0, y0, x1, y1]
    # pts should be in the format of [[x1, y1],
    #                                 [x2, y2],
    #                                   ...   
    #                                 [xn,yn]]
    
    edge = np.array(edge).reshape((4,1))
    points = np.array(points) 
    
    if edge.shape != (4,1):
        raise ValueError('in-valid edge shape!')
     
    if points.shape[1] != 2:
        raise ValueError('in-valid points shape!')
        
    # direction vector of each edge
    dx = edge[2] - edge[0]
    dy = edge[3] - edge[1]

    # compute position of points projected on the supporting line
    # (Size of tp is the max number of edges or points)   
    delta = dx * dx + dy * dy
    tp = ((points[:, 0] - edge[0]) * dx + (points[:, 1] - edge[1]) *dy) / delta


    # change position to ensure projected point is located on the edge
    tp[tp < 0] = 0;
    tp[tp > 1] = 1;

    # coordinates of projected point
    p0 = np.column_stack((edge[0] + tp * dx, edge[1] + tp * dy))

    # compute distance between point and its projection on the edge
    dist = np.sqrt((points[:,0] - p0[:,0]) ** 2 + (points[:,1] - p0[:,1]) ** 2);

    return dist 

#%%
def calDistance(knot_set, coeff_set, pts):
    
    pts = np.array(pts) 
    
    dist_mat = np.empty((pts.shape[0],0))
    for i in range(0,len(knot_set)-1):
        edge = [knot_set[i], coeff_set[i], knot_set[i+1], coeff_set[i+1]]
        dist_mat_i = distancePointEdge(pts, edge)
        dist_mat = np.column_stack((dist_mat, dist_mat_i))
    
    return dist_mat

#%%
def calMae(dist):
    dist = dist.min(axis=1)
    mae = dist.mean()
    return mae

#%%
def findCandidate(dist, filt_dist, pct, 
                  y, loc_set, filter_opt='movcv'):
    
    dist = dist.min(axis=1)
    
    #mov_mean = pd.Series(list(dist)).rolling(window=filt_dist, closed='both').mean()    
    
    mov_mean = pd.rolling_mean(dist, filt_dist, min_periods=1)
    mov_std = pd.rolling_std(dist, filt_dist, min_periods=1)
    
    if filter_opt == 'movcv':
        search_series = (mov_mean / mov_std)
        
    invalid_ss_idx = list(set(list(range(0,filt_dist)) + 
                              list(range(len(search_series) - filt_dist, len(search_series))) + 
                              list(loc_set)))
    search_series_inner = np.delete(search_series, invalid_ss_idx, None) # the N-1 search_series got flatten in here
    
    if len(search_series_inner) == 0:
        cand_loc = -999
        coeff = -999 
    else:
        cand_loc = int(np.where(search_series.flatten() == search_series_inner.max())[0])
        cand_loc_filt = list(range(int(cand_loc - ((filt_dist - 1) / 2)), int(cand_loc + ((filt_dist - 1) / 2) + 1)))
        coeff = np.percentile(y[cand_loc_filt], pct, interpolation='midpoint')
        
    return cand_loc, coeff

#%%
def updateknotcoeffSet(knot_set, coeff_set, loc_set, x, cand_loc, coeff):
    knot_val = x[cand_loc]
    knot_set = np.unique(np.append(knot_set, knot_val))
    new_knot_loc = int(np.where(knot_set == knot_val)[0])
    coeff_set = np.insert(coeff_set, new_knot_loc, coeff)
    loc_set = np.unique(np.append(loc_set, cand_loc))

    return knot_set, coeff_set, loc_set 

#%%
def genKeepIdx(keep_knots, keep_coeffs, pts, pct, y_pos_flags):
    if len(keep_knots) - 2 > 0:
        mae_iter_ortho = maeEvaforKnotRemoval(keep_knots, keep_coeffs, pts, y_pos_flags, pct)
        mae_iter_ortho = np.array(mae_iter_ortho)
        remove_idx = mae_iter_ortho.argmin() + 1 # jump ahead since mae_iter_ortho only coresponding to inner knots 
        keep_idx = list(range(0, len(keep_knots)))
        removed = keep_idx.pop(remove_idx)
    else:
        keep_idx = [0, len(keep_knots)-1]

    return keep_idx

#%%
def maeEvaforKnotRemoval(knot_set, coeff_set, pts, y_pos_flags, pct):
    mae_iter_ortho = []
    for i in range(1, len(knot_set) - 1): # evaluating only inner knots
        remove_idx = i
        new_knot_set = np.delete(knot_set, remove_idx)
        new_coeff_set = np.delete(coeff_set, remove_idx)
        
        dist = calDistance(new_knot_set, new_coeff_set, pts)
        
        ortho_err = dist.min(axis=1)
        ortho_err[y_pos_flags] = ortho_err[y_pos_flags] * pct
        ortho_err[~y_pos_flags] = ortho_err[~y_pos_flags] * (100 - pct) 
        mae_iter_ortho.append(ortho_err.mean())
    
    return mae_iter_ortho

#%%
def calBIC(ortho_err, knot_set, penalty):
    # BIC acccumulation
    positive_flags = ortho_err > 0 # in case a value is exactly 0
    
    
    pars = stats.lognorm.fit(ortho_err[positive_flags])
    loglik = -1 * stats.lognorm.nnlf(pars, ortho_err[positive_flags])

    num_segs = len(knot_set)-1

    bic_remove = -2 * loglik + penalty * num_segs * math.log(len(ortho_err))
 
    return bic_remove
