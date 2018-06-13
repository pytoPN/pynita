"""
Created on Jun 4, 2018
@author: Leyang Feng
@email: feng@american.edu
@Project: pynita
License:  
Copyright (c) 
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

from pynita.data_reader.data_loader import DataLoader
from pynita.data_reader.ini_reader import ConfigReader
from pynita.nita_funs import nita_funs as nf
import pynita.utils.logging as lg

ini = './example/user_configs.ini'

class nitaObj:
    
    def __init__(self, ini, log=False):
        
        self.__log = log
        if self.__log:
            self.__logger = lg.setupLogger(self.cfg)
            
        self.cfg = ConfigReader(ini)       
        self.pts = None
        self.stack = None
        self.stack_dates = None
        self.stack_doy = None
        self.stack_prj = None
        self.compute_mask = None 
  
    def stopLog(self):
        if self.__log:
            lg.closeLogger()
            
    def loadPts(self, info_column='none', full_table=False):
        dl = DataLoader(self.cfg)
        self.pts = dl.load_pts(info_column, full_table)
        
        # properties
        self.pts_OBJECTIDs = list(set(self.pts.OBJECTID))
        self.pts_count = len(self.pts_OBJECTIDs)
        self.ref_pts = dl.ref_tb        
        
    def loadStack(self):
        dl = DataLoader(self.cfg)
        self.stack, self.stack_dates, self.stack_doy, self.stack_prj = dl.load_stack()
        
        # properties
        self.stack_shape = '{0} rows {1} columns {2} layers'.format(self.stack.shape[1], self.stack.shape[2], self.stack.shape[0])
        
    def setMask(self, user_mask):
        if type(self.stack).__name__ == 'NoneType':
            print('stack not loaded yet, loading now...')
            self.loadStack()
        
        if type(user_mask).__name__ != 'ndarray':
            print('convert user_mask into numpy array')
            user_mask = np.array(user_mask)
        
        if user_mask.shape == self.stack.shape[1:3]:
            pass
        else:
            raise RuntimeError('ERROR: user_mask dimensions does not match stack dimensions')
            
        self.compute_mask = user_mask
    
    def runPts(self, OBJECTIDs, compute_mask=True, 
               plot=True, max_plot=25, 
               showdata='fit', colorbar=True, plot_title=True):
        
        # check to see if pts are loaded 
        if type(self.pts).__name__ == 'NoneType':
            raise RuntimeError('pts not loaded yet' )
        
        # reload in case anything changed in the ini file 
        # TODO: got to be a better way to arrange this -- 
        # ini is needed in the __init__ for path and etc. but 
        # can change before running points  
        
        self.cfg = ConfigReader(ini)       
        
        user_vi = self.cfg.user_vi
        
        value_limits = self.cfg.value_limits
        doy_limits = self.cfg.doy_limits
        date_limits = self.cfg.date_limits
        bail_thresh = self.cfg.bail_thresh
        noise_thresh = self.cfg.noise_thresh      
        penalty = self.cfg.penalty
        filt_dist = self.cfg.filt_dist
        pct = self.cfg.pct
        max_complex = self.cfg.max_complex
        min_complex = self.cfg.min_complex
        compute_mask = compute_mask
        filter_opt = self.cfg.filter_opt
        
        if OBJECTIDs == [9999]:
            OBJECTIDs = list(set(self.pts['OBJECTID']))
        
        OBJECTIDs = OBJECTIDs[0:25]
        
        subplots_ncol = int(min(np.ceil(np.sqrt(len(OBJECTIDs))),5))
        subplots_nrow = int(min(np.ceil(len(OBJECTIDs)/subplots_ncol),5))
        
        if plot:
            fig, ax = plt.subplots(nrows=subplots_nrow, ncols=subplots_ncol)
            ax = np.array(ax)
        
        for OBJECTID in OBJECTIDs:
            
            i = OBJECTIDs.index(OBJECTID)
                    
            px = self.pts.loc[self.pts['OBJECTID'] == OBJECTID][user_vi].values
            date_vec = self.pts.loc[self.pts['OBJECTID'] == OBJECTID]['date_dist'].values
            doy_vec = self.pts.loc[self.pts['OBJECTID'] == OBJECTID]['doy'].values
            
            if len(px) == 0:
                plt.close()
                raise RuntimeError('in-valid one or more OBJECTID(s)') 
            
            results_dic = nf.nita_px(px, date_vec, doy_vec, 
                                     value_limits, doy_limits, date_limits,
                                     bail_thresh, noise_thresh,
                                     penalty, filt_dist, pct, max_complex, min_complex,
                                     compute_mask, filter_opt)

            if plot:
                if plot_title:
                    info_line = self.ref_pts.loc[self.ref_pts['OBJECTID'] == OBJECTID]
                    title = ''.join([str(item)+' ' for item in list(info_line.values.flatten())])
                else:
                    title = ''
                    
                nf.viewNITA(px, date_vec, doy_vec, results_dic, showdata=showdata, colorbar=colorbar, title = title, fig=fig, ax=ax.flatten()[i])               

        if len(OBJECTIDs) == 1:
            return results_dic
        return  

    def runStack():
        
        
        
        
        
        
nita = nitaObj(ini)
nita.loadPts(info_column='Name')
nita.runPts([9999], compute_mask=True, plot=True, showdata='fit', colorbar=False, plot_title=True)
results_dic = nita.runPts([4], compute_mask=True, plot=True, showdata='fit', colorbar=True, plot_title=True)


