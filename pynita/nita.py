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

from data_reader.data_loader import dataLoader
from data_reader.ini_reader import ConfigReader
from data_writer import data_writer_funs as dw
from nita_funs import nita_funs as nf
from metric_funs import metric_funs as mf
import utils.logging as lg

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
        dl = dataLoader(self.cfg)
        self.pts = dl.load_pts(info_column, full_table)
        
        # properties
        self.pts_OBJECTIDs = list(set(self.pts.OBJECTID))
        self.pts_count = len(self.pts_OBJECTIDs)
        self.ref_pts = dl.ref_tb        
        
        if self.log:
            self.logger.info('Points file ' + self.cfg.ptsFn + ' loaded.')
            
    def loadStack(self):
        dl = dataLoader(self.cfg)
        
        FUN_start_time = time.time()
        self.stack, self.stack_dates, self.stack_doy, self.stack_prj, self.stack_geotransform = dl.load_stack()
        FUN_end_time = time.time() 
        
        # properties
        self.stack_shape = '{0} rows {1} columns {2} layers'.format(self.stack.shape[1], self.stack.shape[2], self.stack.shape[0])
        
        if self.log:
            self.logger.info('Points file ' + self.cfg.ptsFn + ' loaded. {}s used.'.format(round(FUN_end_time - FUN_start_time, 4)))
            
    def setMask(self, user_mask):    
        if type(user_mask).__name__ != 'ndarray':
            print('convert user_mask into numpy array')
            user_mask = np.array(user_mask)    
        
        if sum(np.unique(user_mask)) != 1:
            raise RuntimeError('only accept mask as matrix containing 1 or 0, \nplease re-prepare your mask')
        
        self.compute_mask = user_mask
    
        if self.log:
            self.logger.info('User compute mask set.')
            
    def runPts(self, OBJECTIDs, 
               plot=True, max_plot=25, 
               showdata='fit', colorbar=True, plot_title=True, **param_dic):
        
        # check to see if pts are loaded 
        if type(self.pts).__name__ == 'NoneType':
            raise RuntimeError('pts not loaded yet')
        
        default_param_dic = {}
        default_param_dic['user_vi'] = self.cfg.user_vi
        default_param_dic['value_limits'] = self.cfg.value_limits
        default_param_dic['doy_limits'] = self.cfg.doy_limits
        default_param_dic['date_limits'] = self.cfg.date_limits
        default_param_dic['bail_thresh'] = self.cfg.bail_thresh
        default_param_dic['noise_thresh'] = self.cfg.noise_thresh      
        default_param_dic['penalty'] = self.cfg.penalty
        default_param_dic['filt_dist'] = self.cfg.filt_dist
        default_param_dic['pct'] = self.cfg.pct
        default_param_dic['max_complex'] = self.cfg.max_complex
        default_param_dic['min_complex'] = self.cfg.min_complex
        default_param_dic['filter_opt'] = self.cfg.filter_opt
        compute_mask = True
        
        if param_dic is not None:       
            keys = param_dic.keys()
            wrong_names = [key for key in keys if not key in default_param_dic.keys()]
            
            if len(wrong_names) == 0:
                for key, value in param_dic.items():
                    default_param_dic[key] = value
                    print(key)
            else:
                raise RuntimeError('ERROR: Wrong parameter name!')
        
        if OBJECTIDs == [9999]:
            OBJECTIDs = list(set(self.pts['OBJECTID']))
        
        OBJECTIDs = OBJECTIDs[0:max_plot]
        
        subplots_ncol = int(min(np.ceil(np.sqrt(len(OBJECTIDs))),5))
        subplots_nrow = int(min(np.ceil(len(OBJECTIDs)/subplots_ncol),5))
        
        if plot:
            fig, ax = plt.subplots(nrows=subplots_nrow, ncols=subplots_ncol)
            ax = np.array(ax)
        
        for OBJECTID in OBJECTIDs:
            
            i = OBJECTIDs.index(OBJECTID)
                    
            px = self.pts.loc[self.pts['OBJECTID'] == OBJECTID][default_param_dic['user_vi']].values
            date_vec = self.pts.loc[self.pts['OBJECTID'] == OBJECTID]['date_dist'].values
            doy_vec = self.pts.loc[self.pts['OBJECTID'] == OBJECTID]['doy'].values
            
            if len(px) == 0:
                plt.close()
                raise RuntimeError('in-valid one or more OBJECTID(s)') 
            
            results_dic = nf.nita_px(px, date_vec, doy_vec, 
                                     default_param_dic['value_limits'], default_param_dic['doy_limits'], default_param_dic['date_limits'],
                                     default_param_dic['bail_thresh'], default_param_dic['noise_thresh'],
                                     default_param_dic['penalty'], default_param_dic['filt_dist'], default_param_dic['pct'], default_param_dic['max_complex'], default_param_dic['min_complex'],
                                     compute_mask, default_param_dic['filter_opt'])

            if plot:
                if plot_title:
                    info_line = self.ref_pts.loc[self.ref_pts['OBJECTID'] == OBJECTID]
                    title = ''.join([str(item)+' ' for item in list(info_line.values.flatten())])
                else:
                    title = ''
                    
                nf.viewNITA(px, date_vec, doy_vec, results_dic, showdata=showdata, colorbar=colorbar, title = title, fig=fig, ax=ax.flatten()[i])               

        if self.log:
            self.logger.info('runPts start...')
            self.logger.info('OBJECTIDs run: ' + str(OBJECTIDs))
            if len(param_dic) != 0:
                self.logger.info('Updated parameters used')
            else:
                self.logger.info('Parameters in ini file used')
            for k, v in default_param_dic.items():
                self.logger.info(k + ': ' + str(v))
            self.logger.info('runPts end...')
            
        if len(OBJECTIDs) == 1:
            return results_dic  

    def runStack(self, parallel=True, workers=2):
        
        self.cfg = ConfigReader(ini)     
        
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
            self.logger.info('runStack start...')
            if parallel:
                self.logger.info('Parallelization enabled with {} workers'.format(workers))
            else:
                self.logger.info('No-Parallelization enabled')
            FUN_start_time = time.time()
            self.logger.info('Stack start time: {}'.format(time.asctime(time.localtime(FUN_start_time))))
            self.logger.info('Parameters in ini file used')
        
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
            if compute_mask.shape != self.stack.shape[1:3]:
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
    
    def runPixel(self, xy_pair, 
                 use_compute_mask=False, 
                 plot=True, showdata='fit', colorbar=True, 
                 **nita_parameters):
        
        # get n and m 
        n = xy_pair[1]
        m = xy_pair[0]
        
        # get the compute mask value 
        if use_compute_mask:
            if type(self.compute_mask).__name__ != 'ndarray':
                compute_mask = True
            else:
                compute_mask_mat = self.compute_mask
                if compute_mask_mat.shape != self.stack.shape[1:3]:
                    raise RuntimeError('ERROR: user_mask dimensions does not match stack dimensions! \nUse setMask() to reset')
                else:
                    compute_mask = compute_mask_mat[n, m]
        else:
            compute_mask = True         

        # get the nita parameters
        param_dic = {}
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
        
        if len(nita_parameters) != 0:       
            keys = nita_parameters.keys()
            wrong_names = [key for key in keys if not key in param_dic.keys()]
            
            if len(wrong_names) == 0:
                for key, value in nita_parameters.items():
                    param_dic[key] = value
            else:
                raise RuntimeError('ERROR: Wrong parameter name!')
                   
        # get data and run 
        date_vec = self.stack_dates   
        doy_vec = self.stack_doy
        px = self.stack[:, n, m]
        
        results_dic = nf.nita_px(px, date_vec, doy_vec, 
                                 param_dic['value_limits'], param_dic['doy_limits'], param_dic['date_limits'],
                                 param_dic['bail_thresh'], param_dic['noise_thresh'],
                                 param_dic['penalty'], param_dic['filt_dist'], param_dic['pct'], param_dic['max_complex'], param_dic['min_complex'],
                                 compute_mask, param_dic['filter_opt'])
        
        if plot:
            nf.viewNITA(px, date_vec, doy_vec, results_dic, showdata=showdata, colorbar=colorbar)
        
        if self.log:
            self.logger.info('runPixel start...')
            self.logger.info('Pixel location: x = {0}, y = {1}'.format(xy_pair[0], xy_pair[1]))
            if len(nita_parameters) != 0:
                self.logger.info('Updated parameters used')
            else:
                self.logger.info('Parameters in ini file used')
            for k, v in param_dic.items():
                self.logger.info(k + ': ' + str(v))
            self.logger.info('runPixel end...')
        
        return results_dic    
    
    def computeStackMetrics(self, parallel=True, workers=2):
        
        # check if self.stack_results exists 
        try:
            type(self.stack_results)
        except AttributeError:
            raise RuntimeError('stack results not calculated yet!')
        
        if self.log:
            self.logger.info('computeStackMetrics start...')
            if parallel:
                self.logger.info('Parallelization enabled with {} workers'.format(workers))
            else:
                self.logger.info('No-Parallelization enabled')
            FUN_start_time = time.time()
            self.logger.info('computeStackMetrics start time: {}'.format(time.asctime(time.localtime(FUN_start_time))))
            self.logger.info('Parameters in ini file used')        
        
        vi_change_thresh = self.cfg.vi_change_thresh
        run_thresh = self.cfg.run_thresh
        time_step = self.cfg.time_step
        
        if parallel:
            
            iterable = [(results_dic, vi_change_thresh, run_thresh, time_step) for results_dic in self.stack_results]
        
            pool = Pool(workers)
            metrics_dics_1d = pool.starmap(mf.computeMetrics, iterable)
            pool.close()
            pool.join()
        
        if not parallel: 
            metrics_dics_1d = []
            for results_dic in self.stack_results:
                metrics_dic = mf.computeMetrics(results_dic, vi_change_thresh, run_thresh, time_step)
                metrics_dics_1d.append(metrics_dic)
        
        self.stack_metrics = metrics_dics_1d

        if self.log:
            FUN_end_time = time.time()
            self.logger.info('computeStackMetrics end time: {}'.format(time.asctime(time.localtime(FUN_end_time))))
            self.logger.info('computeStackMetrics running time (seconds): {}'.format(FUN_end_time - FUN_start_time))

    def getPixelMetrics(self, xy_pair):
        
        try:
            type(self.stack_metrics)
        except AttributeError:
            raise RuntimeError('stack metrics not calculated yet!')        

        n = xy_pair[1]
        m = xy_pair[0]
        
        stack_shape = self.stack.shape # (t, n, m)
        nm_vec = np.arange(0,  stack_shape[1]*stack_shape[2])
        n_vec = np.floor_divide(nm_vec, stack_shape[2])
        m_vec = np.mod(nm_vec, stack_shape[2])
        
        pixel_idx = int(nm_vec[(n_vec == n) & (m_vec == m)])
        
        metrics_dic = self.stack_metrics[pixel_idx]

        return metrics_dic 
    
    def computeMetrics(self, results_dic, **metric_parameters):
        
        param_dic = {}
        param_dic['vi_change_thresh'] = self.cfg.vi_change_thresh
        param_dic['run_thresh'] = self.cfg.run_thresh
        param_dic['time_step'] = self.cfg.time_step
        
        if len(metric_parameters) != 0:       
            keys = metric_parameters.keys()
            wrong_names = [key for key in keys if not key in param_dic.keys()]
            
            if len(wrong_names) == 0:
                for key, value in metric_parameters.items():
                    param_dic[key] = value
            else:
                raise RuntimeError('ERROR: Wrong parameter name!')
        
        vi_change_thresh = param_dic['vi_change_thresh']
        run_thresh = param_dic['run_thresh']
        time_step = param_dic['time_step']

        metrics_dic = mf.computeMetrics(results_dic, vi_change_thresh, run_thresh, time_step)
     
        if self.log:
            self.logger.info('computeMetrics start...')
            if len(metric_parameters) != 0:
                self.logger.info('Updated parameters used')
            else:
                self.logger.info('Parameters in ini file used')
            for k, v in param_dic.items():
                self.logger.info(k + ': ' + str(v))
            self.logger.info('computeMetrics end...')
            
        return metrics_dic
    
    def MI_complexity(self, plot=True, save=True, fn='complexity.tif'):
        
        # check if self.stack_results exists 
        try:
            type(self.stack_results)
        except AttributeError:
            raise RuntimeError('stack results not calculated yet!')
        
        vals_1d = mf.MI_complexity(self.stack_results)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)
        
        if self.log:
            self.logger.info('Metrics image - complexity. Filename: {0}. Saved: {1}'.format(fn, str(save)))
    
    def MI_distDate(self, option='middle', plot=True, save=True, fn='distdate.tif'):
        
        # check if self.stack_metrics exists 
        try:
            type(self.stack_metrics)
        except AttributeError:
            raise RuntimeError('stack metrics not calculated yet!')
            
        vals_1d = mf.MI_distDate(self.stack_metrics, option=option)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)               
        if self.log:
            self.logger.info('Metrics image - distrubance date. Filename: {0}. Saved: {1}'.format(fn, str(save)))
     
    def MI_distDuration(self, plot=True, save=True, fn='distduration.tif'):

        # check if self.stack_metrics exists 
        try:
            type(self.stack_metrics)
        except AttributeError:
            raise RuntimeError('stack metrics not calculated yet!')
            
        vals_1d = mf.MI_distDuration(self.stack_metrics)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)           

        if self.log:
            self.logger.info('Metrics image - distrubance duration. Filename: {0}. Saved: {1}'.format(fn, str(save)))
     
    def MI_distMag(self, plot=True, save=True, fn='distMag.tif'):
        
        # check if self.stack_metrics exists 
        try:
            type(self.stack_metrics)
        except AttributeError:
            raise RuntimeError('stack metrics not calculated yet!')
            
        vals_1d = mf.MI_distMag(self.stack_metrics)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)

        if self.log:
            self.logger.info('Metrics image - disturbance magnitude. Filename: {0}. Saved: {1}'.format(fn, str(save)))
    
    def MI_distSlope(self, plot=True, save=True, fn='distSlope.tif'):
        
        # check if self.stack_metrics exists 
        try:
            type(self.stack_metrics)
        except AttributeError:
            raise RuntimeError('stack metrics not calculated yet!')
            
        vals_1d = mf.MI_distSlope(self.stack_metrics)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)

        if self.log:
            self.logger.info('Metrics image - disturbance slope. Filename: {0}. Saved: {1}'.format(fn, str(save)))

    def MI_linearError(self, plot=True, save=True, fn='linerror.tif'):
        
        # check if self.stack_metrics exists 
        try:
            type(self.stack_results)
        except AttributeError:
            raise RuntimeError('stack results not calculated yet!')
            
        vals_1d = mf.MI_linearError(self.stack_results)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)

        if self.log:
            self.logger.info('Metrics image - linear error. Filename: {0}. Saved: {1}'.format(fn, str(save)))

    def MI_noise(self, plot=True, save=True, fn='noise.tif'):
        
        # check if self.stack_metrics exists 
        try:
            type(self.stack_results)
        except AttributeError:
            raise RuntimeError('stack results not calculated yet!')
            
        vals_1d = mf.MI_noise(self.stack_results)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)

        if self.log:
            self.logger.info('Metrics image - noise. Filename: {0}. Saved: {1}'.format(fn, str(save)))

    def MI_bailcut(self, plot=True, save=True, fn='bailcut.tif'):
        
        # check if self.stack_metrics exists 
        try:
            type(self.stack_results)
        except AttributeError:
            raise RuntimeError('stack results not calculated yet!')
            
        vals_1d = mf.MI_bailcut(self.stack_results)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)     

        if self.log:
            self.logger.info('Metrics image - bailcut. Filename: {0}. Saved: {1}'.format(fn, str(save)))

    def MI_postDistSlope(self, plot=True, save=True, fn='postdistslope.tif'):
        
        # check if self.stack_metrics exists 
        try:
            type(self.stack_metrics)
        except AttributeError:
            raise RuntimeError('stack metrics not calculated yet!')
            
        vals_1d = mf.MI_postDistSlope(self.stack_metrics)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)
    
        if self.log:
            self.logger.info('Metrics image - post disturbance slope. Filename: {0}. Saved: {1}'.format(fn, str(save)))    

    def MI_postDistMag(self, plot=True, save=True, fn='postdistmag.tif'):
        
        # check if self.stack_metrics exists 
        try:
            type(self.stack_metrics)
        except AttributeError:
            raise RuntimeError('stack metrics not calculated yet!')
            
        vals_1d = mf.MI_postDistMag(self.stack_metrics)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)

        if self.log:
            self.logger.info('Metrics image - post disturbance magnitude. Filename: {0}. Saved: {1}'.format(fn, str(save)))
            
    def MI_head(self, plot=True, save=True, fn='head.tif'):
        
        # check if self.stack_metrics exists 
        try:
            type(self.stack_results)
        except AttributeError:
            raise RuntimeError('stack results not calculated yet!')
            
        vals_1d = mf.MI_head(self.stack_results)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)    

        if self.log:
            self.logger.info('Metrics image - head. Filename: {0}. Saved: {1}'.format(fn, str(save)))

    def MI_tail(self, plot=True, save=True, fn='tail.tif'):
        
        # check if self.stack_metrics exists 
        try:
            type(self.stack_results)
        except AttributeError:
            raise RuntimeError('stack results not calculated yet!')
            
        vals_1d = mf.MI_tail(self.stack_results)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)                
        if self.log:
            self.logger.info('Metrics image - tail. Filename: {0}. Saved: {1}'.format(fn, str(save)))            

    def MI_dateValue(self, value_date, plot=True, save=True, fn='datevalue.tif'):
        
        # check if self.stack_metrics exists 
        try:
            type(self.stack_metrics)
        except AttributeError:
            raise RuntimeError('stack metrics not calculated yet!')
            
        vals_1d = mf.MI_dateValue(self.stack_metrics, value_date)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)                

        if self.log:
            self.logger.info('Metrics image - date value. Filename: {0}. Saved: {1}'.format(fn, str(save)))
             
    def MI_valueChange(self, start_date=-9999, end_date=9999, option='diff', 
                       plot=True, save=True, fn='valuechange.tif'):
        
        # check if self.stack_metrics exists 
        try:
            type(self.stack_metrics)
        except AttributeError:
            raise RuntimeError('stack metrics not calculated yet!')
            
        vals_1d = mf.MI_valueChange(self.stack_metrics, start_date = start_date, end_date = end_date, option=option)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)                

        if self.log:
            self.logger.info('Metrics image - value change. Filename: {0}. Saved: {1}'.format(fn, str(save)))
     
    def MI_recovery(self, time_passed, option='diff', plot=True, save=True, fn='recovery.tif'):

        # check if self.stack_metrics exists 
        try:
            type(self.stack_metrics)
        except AttributeError:
            raise RuntimeError('stack metrics not calculated yet!')
            
        vals_1d = mf.MI_recovery(self.stack_metrics,time_passed, option=option)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)               
  
        if self.log:
            self.logger.info('Metrics image - recovery. Filename: {0}. Saved: {1}'.format(fn, str(save)))

    def MI_recoveryCmp(self, time_passed, plot=True, save=True, fn='recovery.tif'):

        # check if self.stack_metrics exists 
        try:
            type(self.stack_metrics)
        except AttributeError:
            raise RuntimeError('stack metrics not calculated yet!')
            
        vals_1d = mf.MI_recoveryCmp(self.stack_metrics,time_passed)
        stack_shape = self.stack.shape
        vals_2d = vals_1d.reshape(stack_shape[1:3])
        
        if plot: 
            mf.plotMI(vals_2d)
        
        if save:
            dw.saveMI(vals_2d, self.stack_prj, self.stack_geotransform,
                      self.cfg.OutputFolder, fn)        
           
        if self.log:
            self.logger.info('Metrics image - recovery comparison. Filename: {0}. Saved: {1}'.format(fn, str(save)))            
            
    def setOpmParams(self, **param_dic):
         
        default_param_dic = {'bail_thresh': np.arange(1.3, 2.3, 0.2),
                             'noise_thresh': [1],
                             'penalty': [0.5, 1, 2, 3, 5],
                             'filt_dist': [1, 3, 5, 7],
                             'pct': [50, 70, 90],
                             'max_complex': [5, 7, 10, 15],
                             'min_complex':[1],
                             'filter_opt': ['movcv'],
                             'value_limits': [self.cfg.value_limits],
                             'doy_limits': [self.cfg.doy_limits],
                             'date_limits': [self.cfg.date_limits]}
        
        keys = param_dic.keys() 
        if param_dic is not None:                   
            wrong_names = [key for key in keys if not key in default_param_dic.keys()]
            
            if len(wrong_names) == 0:
                for key, value in param_dic.items():
                    default_param_dic[key] = value
            else:
                raise RuntimeError('ERROR: Wrong parameter name!')
        
        # yes I know this is stupid
        param_combos = []
        for bail_thresh in default_param_dic['bail_thresh']:
            for date_limits in default_param_dic['date_limits']:
                for doy_limits in default_param_dic['doy_limits']:
                    for filt_dist in default_param_dic['filt_dist']:
                        for filter_opt in default_param_dic['filter_opt']:
                            for max_complex in default_param_dic['max_complex']:
                                for min_complex in default_param_dic['min_complex']:
                                    for noise_thresh in default_param_dic['noise_thresh']:
                                        for pct in default_param_dic['pct']:
                                            for penalty in default_param_dic['penalty']:
                                                for value_limits in default_param_dic['value_limits']:
                                                    param_dic_run = {'bail_thresh': bail_thresh,
                                                                     'date_limits': date_limits,
                                                                     'doy_limits': doy_limits,
                                                                     'filt_dist': filt_dist,
                                                                     'filter_opt': filter_opt,
                                                                     'max_complex': max_complex,
                                                                     'min_complex': min_complex,
                                                                     'noise_thresh': noise_thresh,
                                                                     'pct': pct, 
                                                                     'penalty': penalty,
                                                                     'value_limits': value_limits}
                                                    param_combos.append(param_dic_run)
            
        self.opm_params = default_param_dic
        self.opm_paramcombos = param_combos 
        
        if self.log:
            self.logger.info('Set optimization parameters.')
            for k, v in default_param_dic.items():
                self.logger.info(k + ': ' + str(v))
        
    def drawPts(self, OBJECTIDs, plot_title=True):
        
        # check to see if pts are loaded 
        try:
            type(self.pts)
        except AttributeError:
            raise RuntimeError('ERROR: pts not loaded yet')

        if self.log:
            self.logger.info('drawPts start...')
    
        user_vi = self.cfg.user_vi
        
        if OBJECTIDs == [9999]:
            OBJECTIDs = list(set(self.pts['OBJECTID']))
        
        OBJECTIDs = OBJECTIDs[0:25]
        
        handdraw_trajs = []
        for OBJECTID in OBJECTIDs:
            
            fig, ax = plt.subplots()
                    
            plot_y = self.pts.loc[self.pts['OBJECTID'] == OBJECTID][user_vi].values
            plot_x = self.pts.loc[self.pts['OBJECTID'] == OBJECTID]['date_dist'].values
            plot_doy = self.pts.loc[self.pts['OBJECTID'] == OBJECTID]['doy'].values

            if plot_title:
                info_line = self.ref_pts.loc[self.ref_pts['OBJECTID'] == OBJECTID]
                title = ''.join([str(item)+' ' for item in list(info_line.values.flatten())])
            else:
                title = ''
            
            mappable = ax.scatter(plot_x, plot_y, c=plot_doy)
            ax.set_xlim([plot_x.min(), plot_x.max()])
            ax.set_ylim([plot_y.min(), plot_y.max()])
            ax.set_title(title)
            fig.colorbar(mappable)
            ginput_res = plt.ginput(-1)
            handdraw_traj = {'OBJECTID': OBJECTID,
                             'traj': ginput_res}
            handdraw_trajs.append(handdraw_traj)
        
        self.handdraw_trajs = handdraw_trajs
        
        plt.close('all')
      
        if self.log:
            self.logger.info('total {} OBJECTIDs drew: ' + str(OBJECTIDs).format(len(OBJECTIDs)))
            self.logger.info('drawPts end...')
            
    def paramOpm(self):

        # check to see if opm_paramcombos 
        try:
            type(self.opm_paramcombos)
        except AttributeError:
            raise RuntimeError('ERROR: opm param combo not set, use setOpmParams()') 

        # check 
        try:
            type(self.handdraw_trajs)
        except AttributeError:
            raise RuntimeError('ERROR: handdraw_trajs not set, use drawPts()')
        
        if self.log:
            self.logger.info('paramOpt...')
            FUN_start_time = time.time()
            self.logger.info('paramOpt start time: {}'.format(time.asctime(time.localtime(FUN_start_time))))
            
        
        OBJECTIDs = [dic['OBJECTID'] for dic in self.handdraw_trajs]
        user_vi = self.cfg.user_vi
        compute_mask=True
        
        paramcombo_rmse_mean = []
        paramcombo_rmse_median = []
        paramcombo_pct95err_mean = []
        for param_combo in self.opm_paramcombos:
            OBJETID_rmse = []
            OBJECTID_pct95_err = []
            for OBJECTID in OBJECTIDs:
                
                handdraw_traj = [dic['traj'] for dic in self.handdraw_trajs if dic['OBJECTID'] == OBJECTID][0]
                
                px = self.pts.loc[self.pts['OBJECTID'] == OBJECTID][user_vi].values
                date_vec = self.pts.loc[self.pts['OBJECTID'] == OBJECTID]['date_dist'].values
                doy_vec = self.pts.loc[self.pts['OBJECTID'] == OBJECTID]['doy'].values
            
                if len(px) == 0:
                    raise RuntimeError('in-valid one or more OBJECTID(s)') 
            
                results_dic = nf.nita_px(px, date_vec, doy_vec, 
                                         param_combo['value_limits'], param_combo['doy_limits'], param_combo['date_limits'],
                                         param_combo['bail_thresh'], param_combo['noise_thresh'],
                                         param_combo['penalty'], param_combo['filt_dist'], param_combo['pct'], param_combo['max_complex'], param_combo['min_complex'],
                                         compute_mask, param_combo['filter_opt'])
                
                nita_knots = results_dic['final_knots']
                nita_coeffs = results_dic['final_coeffs']
                
                draw_knots = [tp[0] for tp in handdraw_traj]
                draw_coeffs = [tp[1] for tp in handdraw_traj]
                
                common_start = max([nita_knots[0], draw_knots[0]])
                common_end = min([nita_knots[-1], draw_knots[-1]])
                
                nita_interp = np.interp(np.arange(common_start, common_end, 200), nita_knots, nita_coeffs)
                draw_interp = np.interp(np.arange(common_start, common_end, 200), draw_knots, draw_coeffs)
                
                sq_error = (draw_interp - nita_interp)**2
                rmse = np.sqrt(sq_error.mean())
                pct95_err = np.sqrt(np.percentile(sq_error, 95, interpolation='midpoint'))
                
                OBJETID_rmse.append(rmse)
                OBJECTID_pct95_err.append(pct95_err)
            
            paramcombo_rmse_mean.append(np.mean(OBJETID_rmse))
            paramcombo_rmse_median.append(np.median(OBJETID_rmse))
            paramcombo_pct95err_mean.append(np.mean(OBJETID_rmse))
            
        paramcombo_rmse_mean = np.array(paramcombo_rmse_mean)
        paramcombo_rmse_median = np.array(paramcombo_rmse_median)
        paramcombo_pct95err_mean = np.array(paramcombo_pct95err_mean)
            
        # save as att in case
        self.paramcombo_rmse_mean = paramcombo_rmse_mean
        self.paramcombo_rmse_median = paramcombo_rmse_median
        self.paramcombo_pct95err_mean = paramcombo_pct95err_mean

        best_paramcombo = self.opm_paramcombos[paramcombo_pct95err_mean.argmin()]
        self.the_paramcombo = best_paramcombo
        
        for k, v in self.the_paramcombo.items():
            print(k + ': ' + str(v))
        
        if self.log:
            FUN_end_time = time.time()
            self.logger.info('paramOpm end time: {}'.format(time.asctime(time.localtime(FUN_end_time))))
            self.logger.info('paramOpm running time (seconds): {}'.format(FUN_end_time - FUN_start_time))
            self.logger.info('The best parameter combo: ')
            for k, v in self.the_paramcombo.items():
                self.logger.info(k + ': ' + str(v))
                
   def addLog(self, message=''):
       if self.log:
           self.logger.info(message)
       else:
           raise RuntimeError('ERROR: log not started. Use startLog to start.')
            
if __name__ == '__main__asdas':
    
    ini = '../example/user_configs.ini'
    nita = nitaObj(ini)
    
    nita.startLog()
    
    # tests with points 
    nita.loadPts(info_column='Name', full_table=False)
    nita.runPts([9999], plot=True, max_plot=25, showdata='fit', colorbar=False, plot_title=True)
    results_dic = nita.runPts([4], plot=True, showdata='fit', colorbar=True, plot_title=True)    
    results_dic = nita.runPts([4], plot=True, showdata='fit', colorbar=True, plot_title=True, **{'min_complex': 5})
    
    # tests with stack 
    nita.loadStack()
    nita.setMask(np.ones((10,10)))
    nita.runStack(parallel=True, workers=2)
    nita.runStack(parallel=False)
    results_dic = nita.getPixelResults([8, 5])
    results_dic = nita.runPixel([8, 5], use_compute_mask=False, plot=True, showdata='fit', colorbar=True)
    results_dic = nita.runPixel([8, 5], use_compute_mask=False, plot=True, showdata='fit', colorbar=True, **{'value_limits': [-0.5, 1], 'min_complex': 2})
    
    nita.computeStackMetrics(parallel=True, workers=2)
    nita.computeStackMetrics(parallel=False)
    
    metrics_dic = nita.getPixelMetrics([8, 5])
    metrics_dic = nita.computeMetrics(results_dic)
    metrics_dic = nita.computeMetrics(results_dic, **{'run_thresh': 3000})

    nita.MI_complexity(plot=True, save=True, fn='complexity.tiff')
    nita.MI_distDate(option='middle', plot=True, save=True, fn='distdate.tiff')
    nita.MI_distDuration(plot=True, save=True, fn='distduration.tiff')
    nita.MI_distMag(plot=True, save=True, fn='distMag.tif')
    nita.MI_distSlope(plot=True, save=True, fn='distSlope.tif')
    nita.MI_linearError(plot=True, save=True, fn='linerror.tif')
    nita.MI_noise(plot=True, save=True, fn='noise.tif')
    nita.MI_bailcut(plot=True, save=True, fn='bailcut.tif')
    nita.MI_postDistSlope(plot=True, save=True, fn='postdistslope.tif')
    nita.MI_postDistMag(plot=True, save=True, fn='postdistmag.tif')
    nita.MI_dateValue(plot=True, save=True, fn='datevalue.tif')
    nita.MI_valueChange(start_date=-9999, end_date=9999, option='diff', plot=True, save=True, fn='valuechange1.tif')
    nita.MI_valueChange(start_date=2002000, end_date=2016900, option='diff', plot=True, save=True, fn='valuechange2.tif')
    nita.MI_recovery(1, option='diff', plot=True, save=True, fn='recovery.tif')
    nita.MI_recoveryCmp(1, plot=True, save=True, fn='recoverycmp.tif')
    
    nita.drawPts([1, 2, 4], plot_title=True)
    nita.drawPts([9999], plot_title=True)
    nita.setOpmParams()
    nita.setOpmParams(**{'bail_thresh': [1], 'noise_thresh': [1], 'penalty': [1, 2], 'filt_dist': [3, 5], 'pct': [70], 'max_complex': [10]})
    nita.paramOpm()
    
    nita.addLog('log if wanted.')
    
    nita.stopLog()
