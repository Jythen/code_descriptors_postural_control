



import numpy as np
from numpy.core.defchararray import upper
from scipy.signal import butter, filtfilt, periodogram

import constants.labels as labels
from stabilogram.swarii import SWARII

from scipy.fft import rfft, rfftfreq


class Stabilogram():
    def __init__(self):


        self.raw_signal = None              # contains the raw signal
        self.signal = None                  # contain the processed signal
        self.frequency = None               # frequency of the signal. Only if uniformly sampled  

        self._sampling_ok = None            # is the signal uniformly sampled ?
        self._frequency_ok = None           # is the frequency of the signal within reasonable bounds ?
        

        # Store the values of signal transformation, to avoid multiple computations
        self._radius = None                 
        self._power_spectrum = None
        self._sway_density = None
        self._diffusion_plot = None
        self._speed = None



    def from_array(self, array, center = True, original_frequency = None, resample = True, resample_frequency = 25, filter = True, filter_lower_bound=0, filter_upper_bound=10, filter_order = 4 ):
        """
        Import an array as a stabilogram.

        array should be a N x d   ndarray. N is the number of sample, d is the dimension (d=2 or 3)
        d = 2 : The columns are ML (cm) and AP (cm). the signal is supposed to be already uniformly sampled.  original_frequency should be provided.
        d = 3 : The columns are Time (s), ML (cm) and AP (cm). the signal can be non uniformly sampled.

        center : center the signal. Necessary for the correct computation of the features

        resample : resample the signal to the values defined in the paper. See the function resample for more details
        filter : resample the signal to the values defined in the paper. See the function filter for more details

        """
        
        signal = np.array(array)

        self.raw_signal = signal

        n_columns = signal.shape[1]

        assert n_columns in [2,3], "invalid number of columns in the array, should be 2 or 3"


            

            

        if n_columns == 2 :
            assert original_frequency is not None, "Need to provide a frequency for the signal (parameter original frequency), or timestamps"
            time = np.arange(len(signal))/original_frequency
            time = time[:,None]

            valid_index = (np.sum(np.isnan(signal),axis=1) == 0)
            time = time[valid_index]
            signal = signal[valid_index]

            mean = np.mean(signal, axis=0, keepdims=True)
            self.mean_value = mean[0]
                        
            if center : 
                signal = signal - mean
                

            signal = np.concatenate([time, signal], axis = 1)

        else : 
            # time start from 0
            time = signal[:,0]
            time = time - time[0]
            time = time[:,None]
            signal[:,0] = time
    
            mean = np.mean(signal[:,1:], axis=0, keepdims=True)
            self.mean_value = mean
            
            #center signal 
            if center : 
                csignal = signal[:,1:]
                csignal = csignal - np.mean(csignal, axis=0, keepdims=True)
                signal[:,1:] = csignal
                

        self.signal = signal
        assert not np.isnan(signal).any(), "error"
        if resample :
            self.resample(target_frequency= resample_frequency)
        else :
            assert original_frequency is not None, "Need to provide a frequency for the signal (parameter original frequency), or timestamps"
            self.signal = signal[:,1:]
            self.resample()
            
        

        if filter :
            self.filter(lower_bound=filter_lower_bound, upper_bound=filter_upper_bound, order= filter_order)
   


    def resample(self, target_frequency=25)-> None:

        """
        Resample the stabilogram using SWARII, using the parameters recommended in the paper
        """

        assert self.signal is not None, "Please provide a signal first"

        signal = np.array(self.signal)
        n_columns = signal.shape[1]

        assert n_columns in [2,3], "invalid number of columns in the array, should be 2 or 3"

        if n_columns == 3 :
            signal = SWARII.resample(data = signal, desired_frequency=target_frequency )

        
        self.signal = signal
        self._sampling_ok = True
        self._frequency_ok = False
        self.frequency = target_frequency


    def filter(self, lower_bound=0, upper_bound=10, order = 4) -> None:
        """
        Filter the stabilogram using a Butterworth filter. Default parameters are the one used in the paper. 
        """


        assert self.raw_signal is not None, "Please provide a signal first"
        assert self._sampling_ok, "Please resample the signal first, using the function resample " 
        assert self.signal is not None,  "Error, please resample the signal again"


        signal = np.array(self.signal)
        dt = 1 / self.frequency
        nyq = 0.5 * self.frequency
        low = lower_bound / nyq
        high = upper_bound / nyq

        if low == 0 :
            b, a = butter(order, high, btype='lowpass')
        elif high == np.inf :
            b, a = butter(order, low, btype='highpass')
        else :
            b, a = butter(order, (low,high), btype='bandpass')

        y = filtfilt(b, a, signal,axis=0)    
        self.signal = y
        self._frequency_ok = True


        
    







#   ===================================================================
#    Methods to compute the transformations of the signals.
#    Should not be called directly, and instead accessed through properties
#   ===================================================================





    def _compute_radius(self)-> None:  
        """
        Compute the radius of the stabilogram (signal is supposed centered). 
        """ 
        self._radius = np.linalg.norm(self.signal, axis=1, keepdims=True) 
        


    def _compute_power_spectrum(self)-> None:  
        """
        Compute the PSD of the stabilogram using the Periodogram method. 
        """ 
        
        

        freqs, psd = periodogram(np.concatenate([self.signal,np.ones((1,2))],axis=0), fs=self.frequency, axis=0)       
        power_fft = np.concatenate( [freqs[:,None], psd], axis=1)
        self._power_spectrum  = power_fft

        print(len(self.signal), power_fft[-1,0])


    def _compute_sway_density(self, radius=0.3)-> None:  
        """
        Sway Density is computed by default for a 3 mm radius.
        """
        signal = np.array(self.signal)
        sway = np.zeros(len(signal)-1)
        
        for starting_point in range(len(signal)-1):
            stopping_point = starting_point+1
            while stopping_point< len(signal):
                if np.linalg.norm(signal[stopping_point] - signal[starting_point])>radius:
                    break
                stopping_point+=1
            sway[starting_point] = stopping_point -starting_point - 1

        self._sway_density = sway/self.frequency


        
    def _compute_diffusion_plot(self, duration_ratio=1/3)-> None:  
        """
        Compute the diffusion plot of the stabilogram. duration_ratio parameter set the limit for the computation, and should only be modified by experts familiar with the diffusion plot 
        """ 

        n = len(self.signal)
        max_ind = int(n * duration_ratio)
        time = np.arange(n)/self.frequency
        msd = [np.array([0,0])] + [np.mean((self.signal[i:,:] - self.signal[:(n-i),:])**2,axis=0) for i in range(1,max_ind+1)]
        diffusion_plot = np.concatenate([time[:max_ind+1,None], np.array(msd)], axis=1)
        self._diffusion_plot = diffusion_plot


    def _compute_speed(self, window_length=5, polyorder=3) -> None:  
        """
        Speed is computed using savgol filter. Default parameters are the one used in the paper. 
        """
        cop = self.signal
        spd_savgol = savgol_filter( x = cop, window_length= window_length,polyorder = polyorder, deriv= 1, axis=0, delta= 1/self.frequency  )
        self._speed = spd_savgol


    def _test_correct_format(self) -> None:
        assert self.raw_signal is not None, "Please provide a signal first"
        assert self._sampling_ok, "Please resample the signal first, using the function resample " 
        assert self._frequency_ok, "Please filter the signal first, using the function filter " 
        assert self.signal is not None,  "Error, please resample and filter the signal again"
            


#   ===================================================================
#    Defines the properties to access the transformations of the signal
#   ===================================================================

    def __len__(self)-> int:
        self._test_correct_format
        return(len(self.signal))
        

    @property
    def medio_lateral(self) -> np.ndarray:
        self._test_correct_format()
        return self.signal[:,0:1]

    @property
    def antero_posterior(self) -> np.ndarray:
        self._test_correct_format()
        return self.signal[:,1:2]
        
    @property
    def sway_density(self) -> np.ndarray:
        self._test_correct_format()
        if self._sway_density is None:
            self._compute_sway_density()
        return self._sway_density


    @property
    def speed(self) -> np.ndarray:
        self._test_correct_format()
        if self._speed is None:
            self._compute_speed()
        return self._speed


    @property
    def power_spectrum(self) -> np.ndarray:
        self._test_correct_format()
        if self._power_spectrum is None:
            self._compute_power_spectrum()
        return self._power_spectrum


    @property
    def radius(self) -> np.ndarray:
        self._test_correct_format()
        if self._radius is None:
            self._compute_radius()
        return self._radius
    
    @property
    def diffusion_plot(self) -> np.ndarray:
        self._test_correct_format()
        if self._diffusion_plot is None:
            self._compute_diffusion_plot()
        return self._diffusion_plot
            


    def get_signal(self, name) -> np.ndarray:
        if name == labels.ML:
            return self.medio_lateral
        if name == labels.AP :
            return self.antero_posterior
        if name == labels.MLAP :
            return self.signal
        if name == labels.RADIUS :
            return self.radius
        if name == labels.SWAY_DENSITY :
            return self.sway_density
        if name == labels.PSD_ML :
            return self.power_spectrum[:,0], self.power_spectrum[:,1]
        if name == labels.PSD_AP :
            return self.power_spectrum[:,0], self.power_spectrum[:,2]
        if name == labels.SPD_ML:
            return self.speed[:,0:1] 
        if name == labels.SPD_AP:
            return self.speed[:,1:2]
        if name == labels.DIFF_ML:
            return self.diffusion_plot[:,0], self.diffusion_plot[:,1]
        if name == labels.DIFF_AP:
            return self.diffusion_plot[:,0], self.diffusion_plot[:,2]
        if name == labels.DIFF_MLAP:
            return self.diffusion_plot[:,0], self.diffusion_plot[:,1]+self.diffusion_plot[:,2]   # is it a sum really ?
        raise NotImplementedError





