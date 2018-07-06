"""
Created on Jun 4, 2018
@author: Leyang Feng
@email: feng@american.edu
@Project: pynita
License:  
Copyright (c) 
"""
#import sys
import numpy as np
import time  
from multiprocessing import Pool
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_reader.data_loader import DataLoader
from data_reader.ini_reader import ConfigReader
from nita_funs import nita_funs as nf
from metric_funs import metric_funs as mf
import utils.logging as lg


ini = '../example/user_configs.ini'

class nitaObj:
    
    def __init__(self, ini):
        
        self.cfg = ConfigReader(ini)     
        self.pts = None
        self.stack = None
        self.stack_dates = None
        self.stack_doy = None
        self.stack_prj = None
        self.compute_mask = None 
    
    def startLog(self):
        self.log = True
        self.logger = lg.setupLogger(self.cfg)
        self.logger.info('Start logging...')
        
    def stopLog(self):
        if self.log:
            self.logger.info('End logging...')
            lg.closeLogger(self.logger)
            
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
        if type(user_mask).__name__ != 'ndarray':
            print('convert user_mask into numpy array')
            user_mask = np.array(user_mask)    
        
        if sum(np.unique(user_mask)) != 1:
            raise RuntimeError('only accept mask as matrix containing 1 or 0, \nplease re-prepare your mask')
        
        self.compute_mask = user_mask
    
    def runPts(self, OBJECTIDs, compute_mask=True, 
               plot=True, max_plot=25, 
               showdata='fit', colorbar=True, plot_title=True):
        
        # check to see if pts are loaded 
        if type(self.pts).__name__ == 'NoneType':
            raise RuntimeError('pts not loaded yet')
        
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

    def runStack(self, parallel=True, workers=2):
 
        # ---
        # 1.
        # initialize log if wanted
        if self.log:
            self.logger.info('Project Name : {}'.format(self.cfg.ProjectName))
            self.logger.info('Input Folder : {}'.format(self.cfg.InputFolder))
            self.logger.info('Output Folder: {}'.format(self.cfg.OutputFolder))
            self.logger.info('Point Data: {}'.format(self.cfg.ptsFn))
            self.logger.info('Stack Date Data: {}'.format(self.cfg.stackdateFn))
            self.logger.info('Stack Data: {}'.format(self.cfg.stackFn))
        
        # ---
        # 2.
        # check if the stack is loaded  
        if type(self.stack).__name__ == 'NoneType':
            raise RuntimeError('stack not loaded yet')
        
        # ---
        # 3. 
        # check if the mask exists
        # if non-existence (setMask() method never called), assign 'global' compute_mask as True (for each pixel in the stack)
        # if set, check dimension 
        if type(self.compute_mask).__name__ != 'ndarray':
            compute_mask = np.ones(self.stack.shape[1:3])
        else:
            compute_mask = self.compute_mask
            if self.compute_mask.shape != self.stack.shape[1:3]:
                raise RuntimeError('ERROR: user_mask dimensions does not match stack dimensions! \nUse setMask() to reset')
        
        # ---
        # 4.         
        # reduce dimension and generate reference vectors 
        # the read-in image stack is t-n-m so reduce to t-(n*m)
        #                            | | |              |   |
        #                            a b c              a   d
        # along (d), the (b) would become b1, b1, b1,..., b2, b2, b2......, bn, bn, bn,.....
        # while c wile become c1, c2, ..., cn, c1, c2, ..., cn, ......, c1, c2, ..., cn  
        stack_shape = self.stack.shape # (t, n, m)
        stack_2d = self.stack.reshape((stack_shape[0], stack_shape[1]*stack_shape[2]))
        stack_2d_shape = stack_2d.shape
        #t_vec = np.arange(0, stack_shape[0]) 
        #nm_vec = np.arange(0, stack_2d.shape[1])
        #n_vec = np.floor_divide(nm_vec, stack_shape[2])
        #m_vec = np.mod(nm_vec, stack_shape[2])
        
        # reduce dimension for compute mask 
        compute_mask_1d = compute_mask.flatten()
        
        # ---
        # 5. 
        # 5.a
        if self.log:
            FUN_start_time = time.time()
            self.logger.info('Stack start time: {}'.format(time.asctime(time.localtime(FUN_start_time))))
            
        if parallel:
            
            # pack other arguments into a dic 
            param_dic = {}
            param_dic['date_vec'] = self.stack_dates
            param_dic['doy_vec'] = self.stack_doy
            param_dic['value_limits'] = self.cfg.value_limits
            param_dic['doy_limits'] = self.cfg.doy_limits
            param_dic['date_limits'] = self.cfg.date_limits
            param_dic['bail_thresh'] = self.cfg.bail_thresh
            param_dic['noise_thresh'] = self.cfg.noise_thresh       
            param_dic['penalty'] = self.cfg.penalty    
            param_dic['filt_dist'] = self.cfg.filt_dist
            param_dic['pct'] = self.cfg.pct      
            param_dic['max_complex'] = self.cfg.max_complex
            param_dic['min_complex'] = self.cfg.min_complex
            param_dic['filter_opt'] = self.cfg.filter_opt 
        
            iterable = [(stack_2d, compute_mask_1d, param_dic, i) for i in range(stack_2d_shape[1])]
        
            pool = Pool(workers)
            results_dics_1d = pool.starmap(nf.nita_stack_wrapper, iterable)
            pool.close()
            pool.join()
        
        # ---
        # 5.b
        if not parallel:
            
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
             filter_opt = self.cfg.filter_opt
             date_vec = self.stack_dates   
             doy_vec = self.stack_doy
             
             results_dics_1d = []
             for i in tqdm(range(stack_2d_shape[1])):
                 compute_mask_run = compute_mask_1d[i]
                 compute_mask_run = compute_mask_run == 1
                    
                 px = stack_2d[:, i]
                    
                 results_dic = nf.nita_px(px, date_vec, doy_vec, 
                                          value_limits, doy_limits, date_limits,
                                          bail_thresh, noise_thresh,
                                          penalty, filt_dist, pct, max_complex, min_complex,
                                          compute_mask_run, filter_opt)
                 
                 results_dics_1d.append(results_dic)
            
        # ---
        # 6. 
        self.stack_results = results_dics_1d
        
        if self.log:
            FUN_end_time = time.time()
            self.logger.info('Stack end time: {}'.format(time.asctime(time.localtime(FUN_end_time))))
            self.logger.info('Stack running time (seconds): {}'.format(FUN_end_time - FUN_start_time))

    def getPixelResults(self, xy_pair):
        
        # check if self.stack_results exists 
        try:
            type(self.stack_results)
        except AttributeError:
            raise RuntimeError('stack results not calculated yet!')        

        n = xy_pair[1]
        m = xy_pair[0]
        
        stack_shape = self.stack.shape # (t, n, m)
        nm_vec = np.arange(0,  stack_shape[1]*stack_shape[2])
        n_vec = np.floor_divide(nm_vec, stack_shape[2])
        m_vec = np.mod(nm_vec, stack_shape[2])
        
        pixel_idx = int(nm_vec[(n_vec == n) & (m_vec == m)])
        
        results_dic = self.stack_results[pixel_idx]

        return results_dic 
    
    def computeStackMetrics(self, parallel=True, workers=2):
        
        # check if self.stack_results exists 
        try:
            type(self.stack_results)
        except AttributeError:
            raise RuntimeError('stack results not calculated yet!')
        
        vi_change_thresh = self.cfg.vi_change_thresh
        run_thresh = self.cfg.run_thresh
        time_step = self.cfg.time_step
        
        if parallel:
            pass
        
        if not parallel: 
            metrics_dics_1d = []
            for results_dic in self.stack_results:
                metrics_dic = mf.computeMetrics(results_dic, vi_change_thresh, run_thresh, time_step)
                metrics_dics_1d.append(metrics_dic)
        
        self.stack_metrics = metrics_dics_1d
        
 #   def getPixelMetrics(self, xy_pair):
  #      pass
    
   # def computeMetrics(self, results_dic):
    #    pass
    
           
if __name__ == '__main__':
    nita = nitaObj(ini)
    
    nita.startLog()
    
    # tests with points 
    nita.loadPts(info_column='Name')
    #nita.runPts([9999], compute_mask=True, plot=True, showdata='fit', colorbar=False, plot_title=True)
    results_dic = nita.runPts([1], compute_mask=True, plot=True, showdata='fit', colorbar=True, plot_title=True)    
    
    # tests with stack 
    nita.loadStack()

    #nita.runStack(parallel=True, workers=2)
    nita.runStack(parallel=False)
    
    
    #nita.stopLog()