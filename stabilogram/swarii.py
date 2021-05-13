# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:25:45 2016

@author: audiffren
"""

import numpy as np
from scipy.interpolate import interp1d
#´from parsers import parse_wbb_acq_data



        

class Local_SWARII:
    """
    Implementation of the Sliding Windows Weighted Averaged Interpolation method
    
    How To use :
        First instantiate the class with the desired parameters
        Then call resample on the desired signal
        
    """

    def __init__(self, window_size=1, desired_frequency=25, verbose=0,**kwargs):
        """
        Instantiate SWARII 

        Parameters :
            desired_frequency : The frequency desired for the output signal,
                                after the resampling.
            window_size : The size of the sliding window, in seconds.
        """
        self.desired_frequency = desired_frequency
        self.window_size = window_size
        self.verbose= verbose
        self.options = kwargs



    def resample(self, time, signal,interpolate=1):
        """
        Apply the SWARII to resample a given signal.
        
        Input :
            time:   The time stamps of the data point in the signal. A 1-d
                    array of shape n, where n is the number of points in the
                    signal. The unit is seconds.
            signal: The data points representing the signal. A k-d array of
                    shape (n,k), where n is the number of points in the signal,
                    and k is the dimension of the signal (e.g. 2 for a
                    statokinesigram).
            skip_if_missing : will raise an exception if the number of empty windows is larger than 
                              this value (default : + infty)
            interpolate : 0 - last point interpolation
                          1 - linear interpolation
                          -1 - no interpolation, delete missing times (experimental)

            options :
                count_interpolations : if True, will return the number of interpolated poitns
                  
                    
        Output: 
            resampled_time : The time stamps of the signal after the resampling
            resampled_signal : The resampled signal.
        """
        
        a_signal=np.array(signal)
        current_time = max(0.,time[0])
        #print current_time
        output_time=[]
        output_signal = []
        missing_windows=0

        while current_time < time[-1]:

            relevant_times = [t for t in range(len(time)) if abs(
                time[t] - current_time) < self.window_size * 0.5]
            if len(relevant_times) == 0 :
                missing_windows +=1
                if self.verbose == 2:
                    print("Trying to interpolate an empty window ! at time ", current_time)
            else :
                if len(relevant_times) == 1:
                    value = a_signal[relevant_times[0]]
                    
                else :
                    value = 0
                    weight = 0
            
                    for i, t in enumerate(relevant_times):
                        if i == 0 or t==0:
                            left_border = max(
                                time[0], (current_time - self.window_size * 0.5))
                            
                        else:
                            left_border = 0.5 * (time[t] + time[t - 1])
                            
                            
    
                        if i == len(relevant_times) - 1:
                            right_border = min(
                                time[-1], current_time + self.window_size * 0.5)
                        else:
                            right_border = 0.5 * (time[t + 1] + time[t])
                            
                        w = right_border - left_border
                            
    
                        value += a_signal[t] * w
                        weight += w
            
                            
          
                    value /= weight
                output_time.append(current_time)
                output_signal.append(value)
            current_time += 1. / self.desired_frequency
        if missing_windows>0:
            if self.verbose>0:
                    print("There was {} empty windows".format(missing_windows))
            if interpolate>=0:
                interpolation_kind = "linear" if interpolate ==1 else 'previous'
                if self.verbose>0:
                    print("interpolating")
                desired_times = np.arange(output_time[0],output_time[-1],1. / self.desired_frequency)
                func = interp1d(output_time,output_signal,kind=interpolation_kind,axis=0,bounds_error=False)
                desired_signal = func(desired_times)
                output_time, output_signal = desired_times, desired_signal
            else :
                if self.verbose>0 :
                    print("no interpolation")
                    
        if interpolate>=0 and self.options["count_interpolations"]:
            return np.array(output_time),np.array(output_signal), missing_windows
        else :
            return np.array(output_time),np.array(output_signal)


    @staticmethod
    def purge_artefact(time, signal, threshold_up=2, threshold_down=0.5,verbose=0):
        asignal = np.array(signal)
        nsignal=[]
        ntime=[]
        n_artefact=0
        
        for t in range(1,len(time)-1):
            if time[t]<0.1:
                pass
            elif ((len(ntime)>0) and (t>0) and (t<len(time)-1)  and \
                    (np.sum(np.abs(asignal[t+1]-nsignal[-1]))<threshold_down) and \
                    (np.sum(np.abs(asignal[t]-nsignal[-1])) > threshold_up)):
                n_artefact+=1
                pass
            elif ( (len(ntime)>0) and (t>1) and (t<len(time)-2)  and (np.sum(np.abs(asignal[t+2]-nsignal[-1]))<threshold_down) and (np.sum(np.abs(asignal[t]-nsignal[-1])) > threshold_up)):
                n_artefact+=1
                pass
            else :
                ntime.append(time[t])
                nsignal.append(signal[t])
        if n_artefact >0:
            if verbose >0:
                print("skipped", n_artefact, "artefacts"               )
        return ntime,nsignal
            
        


class SWARII :
    @staticmethod
    def resample(data,window_size=0.08,desired_frequency=25,interpolate = True, verbose=0, count_interpolations=False):
        """
        time should be in second
        
        """
        swarii = Local_SWARII(window_size=window_size-1e-6,desired_frequency=desired_frequency, verbose=verbose, count_interpolations=count_interpolations)
        t = data[:,0]
        signal = data[:,1:]
        #y = data.T[2]
        nt,nsignal = Local_SWARII.purge_artefact(time=t,signal=signal, verbose=verbose)   
        
        #t_close,x_close = swarii.resample(t,x)
        #t_close,y_close = swarii.resample(t,y)
        
        if count_interpolations :
            nnt,nnsignal, missing_windows= swarii.resample( time =nt, signal= nsignal, interpolate=interpolate)
            return nnsignal[:,:2], missing_windows
        else :
            nnt, nnsignal= swarii.resample( time =nt, signal= nsignal, interpolate=interpolate)
            return nnsignal[:,:2]

