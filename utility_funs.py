import numpy as np

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
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        