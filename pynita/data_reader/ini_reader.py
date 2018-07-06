"""
Read in settings from configuration file *.ini
Created on Jun 4, 2018
@author: Leyang Feng
@email: feng@american.edu
@Project: pynita
License:  
Copyright (c) 
"""

import os
from configobj import ConfigObj

class ConfigReader:

    def __init__(self, ini):

        c = ConfigObj(ini)

        p = c['Project']

        # project dirs
        self.root = p['RootDir']
        self.ProjectName = p['ProjectName']
        self.InputFolder = os.path.join(self.root, p['InputFolder'])
        self.ptsFn = p['ptsFn']
        self.stackdateFn = p['stackdateFn']
        self.stackFn = p['stackFn']
        self.OutDir = self.createDir(os.path.join(self.root, p['OutputFolder']))
        self.OutputFolder = self.createDir(os.path.join(self.OutDir, self.ProjectName))

        # see if modules are in config file
        try:
            v = c['VI']
        except KeyError:
            v = False

        try:
            np = c['NITAParameters']
        except KeyError:
            np = False

        try:
            mp = c['MetricsParameters']
        except KeyError:
            mp = False

        try:
            npd = c['NITAParametersDebug']
        except KeyError:
            npd = False

        # module level settings
        # module VI
        if v is not False:
            self.user_vi = v['user_vi'].lower()
            # TODO: add value check in here 
        else:
            raise RuntimeError('ERROR: VI module not found')
        
        # module NITAParameters
        if np is not False: 
            self.value_limits = [int(item) for item in np['value_limits']]
            self.doy_limits = [int(item) for item in np['doy_limits']]
            self.doy_limits = [self.doy_limits[i:i+2] for i in range(0, len(self.doy_limits), 2)]
            self.date_limits = [int(item) for item in np['date_limits']]
            self.bail_thresh = float(np['bail_thresh'])
            self.noise_thresh = float(np['noise_thresh'])
            self.penalty = float(np['penalty'])
            self.filt_dist = int(np['filt_dist'])
            self.pct = float(np['pct'])
            self.max_complex = int(np['max_complex'])
            self.min_complex = int(np['min_complex'])
            self.filter_opt = np['filter_opt']
            # TODO: add value check in here
        else: 
            raise RuntimeError('ERROR: [NITAParameters] module not found')
    
        # module NITAParametersDebug
        if npd is not False:
            self.value_limits_dg = [int(item) for item in npd['value_limits_dg']]
            self.doy_limits_dg = [int(item) for item in npd['doy_limits_dg']]
            self.doy_limits_dg = [self.doy_limits_dg[1:1+2] for i in range(0, len(self.doy_limits_dg), 2)]
            self.date_limits_dg = [int(item) for item in npd['date_limits_dg']]
            self.bail_thresh_dg =  float(npd['bail_thresh_dg'])
            self.noise_thresh_dg = float(npd['noise_thresh_dg'])
            self.penalty_dg = float(npd['penalty_dg'])
            self.filt_dist_dg = int(npd['filt_dist_dg'])
            self.pct_dg = float(npd['pct_dg'])
            self.max_complex_dg = int(npd['max_complex_dg'])
            self.min_complex_dg = int(npd['min_complex_dg'])
            self.filter_opt_dg = npd['filter_opt_dg']
            # TODO: add value check in here
        else: 
            raise RuntimeError('ERROR: [NITAParametersDebug] module not found')
        
        # module MetricsParameters
        if mp is not False:
            self.vi_change_thresh = float(mp['vi_change_thresh'])
            self.run_thresh = int(mp['run_thresh'])
            self.time_step = float(mp['time_step'])
            # TODO: add value check in here
        else:
            raise RuntimeError('ERROR: [MetricsParameters] module not found')

    def createDir(self, pth):
        """
        Check to see if the target path is exists.
        """
        if os.path.isdir(pth) is False:
            os.mkdir(pth)
        return pth
        