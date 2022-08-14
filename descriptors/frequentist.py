import numpy as np
from code_descriptors_postural_control.constants import labels



def total_power(signal, axis = labels.PSD_AP, only_feature=False):
    if not (axis in [labels.PSD_ML, labels.PSD_AP]):
        return {} 
    feature_name = "total_power"

    fmin = 0.15
    fmax = 5
    freqs, powers = signal.get_signal(axis)

    selected_powers = powers[((freqs>=fmin) & (freqs<=fmax))]
    
    feature = np.sum(selected_powers)    
    
    if only_feature:
        return feature
    else:
        return { feature_name+"_"+axis  : feature}



def power_frequency_50(signal, axis = labels.PSD_AP):
    if not (axis in [labels.PSD_ML, labels.PSD_AP]):
        return {}   
    feature_name = "power_frequency_50"

    freqs, powers = signal.get_signal(axis)

    fmin = 0.15
    fmax = 5
    
    selected_freqs = freqs[ ((freqs>=fmin) & (freqs<=fmax)) ]
    selected_powers = powers[ ((freqs>=fmin) & (freqs<=fmax)) ]
    
    cum_power = np.cumsum(selected_powers)

    feature = selected_freqs[ (cum_power >= (cum_power[-1]*0.5)) ][0]

    return { feature_name+"_"+axis  : feature}



def power_frequency_95(signal, axis = labels.PSD_AP):
    if not (axis in [labels.PSD_ML, labels.PSD_AP]):
        return {}
    feature_name = "power_frequency_95"
    
    freqs, powers = signal.get_signal(axis)

    fmin = 0.15
    fmax = 5
    
    selected_freqs = freqs[ ((freqs>=fmin) & (freqs<=fmax)) ]
    selected_powers = powers[ ((freqs>=fmin) & (freqs<=fmax)) ]
    
    cum_power = np.cumsum(selected_powers)

    feature = selected_freqs[ (cum_power >= (cum_power[-1]*0.95)) ][0]

    return { feature_name+"_"+axis  : feature}



def power_mode(signal, axis = labels.PSD_AP):
    if not (axis in [labels.PSD_ML, labels.PSD_AP]):
        return {}
    feature_name = "frequency_mode"

    freqs, powers = signal.get_signal(axis)

    fmin = 0.15
    fmax = 5

    selected_freqs = freqs[(freqs>=fmin) & (freqs<=fmax)]
    selected_powers = powers[(freqs>=fmin) & (freqs<=fmax)]

    mode = np.argmax(selected_powers)

    feature = selected_freqs[mode]

    return { feature_name+"_"+axis  : feature}



def _spectral_moment(signal, axis = labels.PSD_AP, moment=1):
    if not (axis in [labels.PSD_ML, labels.PSD_AP]):
        return {}

    fmin = 0.15
    fmax = 5

    freqs, powers = signal.get_signal(axis)

    selected_freqs = freqs[((freqs>=fmin) & (freqs<=fmax))]
    
    selected_powers = powers[((freqs>=fmin) & (freqs<=fmax))]
    
    feature =   np.sum( (selected_freqs**moment) * selected_powers )
    
    return feature



def centroid_frequency(signal, axis = labels.PSD_AP):
    if not (axis in [labels.PSD_ML, labels.PSD_AP]):
        return {}
    feature_name = "centroid_frequency"
    
    m2 = _spectral_moment(signal, axis=axis, moment=2)
    m0 = _spectral_moment(signal, axis=axis, moment=0)

    feature = np.sqrt( m2 / m0 )
    
    return { feature_name+"_"+axis  : feature}



def frequency_dispersion(signal, axis = labels.PSD_AP):
    if not (axis in [labels.PSD_ML, labels.PSD_AP]):
        return {}
    feature_name = "frequency_dispersion"
    
    m2 = _spectral_moment(signal, axis=axis, moment=2)
    m1 = _spectral_moment(signal, axis=axis, moment=1)
    m0 = _spectral_moment(signal, axis=axis, moment=0)

    feature = np.sqrt(  1  - ( (m1**2) / (m0*m2) )  )
    
    return { feature_name+"_"+axis  : feature}



def energy_content_05(signal, axis = labels.PSD_AP):
    if not (axis in [labels.PSD_ML, labels.PSD_AP]):
        return {}
    feature_name = "energy_content_below_05"

    fmin = 0.15
    fmax = 5
    
    freqs, powers = signal.get_signal(axis)

    selected_powers = powers[ (freqs>0.) & (freqs<=0.5) & (freqs>=fmin) & (freqs<=fmax) ]
    
    feature = np.sum(selected_powers)
    
    return { feature_name+"_"+axis  : feature}



def energy_content_05_2(signal, axis = labels.PSD_AP):
    if not (axis in [labels.PSD_ML, labels.PSD_AP]):
        return {}    
    feature_name = "energy_content_05_2"
    
    fmin = 0.15
    fmax = 5
    
    freqs, powers = signal.get_signal(axis)

    selected_powers = powers[ (freqs>0.5) & (freqs<=2) & (freqs>=fmin) & (freqs<=fmax) ]
    
    feature = np.sum(selected_powers)
    
    return { feature_name+"_"+axis  : feature}



def energy_content_2(signal, axis = labels.PSD_AP):
    if not (axis in [labels.PSD_ML, labels.PSD_AP]):
        return {}    
    feature_name = "energy_content_above_2"
    
    fmin = 0.15
    fmax = 5
    
    freqs, powers = signal.get_signal(axis)

    selected_powers = powers[ (freqs > 2) & (freqs>=fmin) & (freqs<=fmax) ]
    
    feature = np.sum(selected_powers)
    
    return { feature_name+"_"+axis  : feature}



def frequency_quotient(signal, axis = labels.PSD_AP):
    if not (axis in [labels.PSD_ML, labels.PSD_AP]):
        return {}    
    feature_name = "frequency_quotient"
    
    fmin = 0.15
    fmax = 5
    
    freqs, powers = signal.get_signal(axis)

    selected_powers_up = powers[ (freqs>2) & (freqs<=5) & (freqs>=fmin) & (freqs<=fmax) ]  
    selected_powers_down = powers[ (freqs>0) & (freqs<=2) & (freqs>=fmin) & (freqs<=fmax) ]

    feature = np.sum(selected_powers_up) / np.sum(selected_powers_down)
    
    return { feature_name+"_"+axis  : feature}



all_features = [total_power, power_frequency_50, power_frequency_95, \
                power_mode, centroid_frequency, frequency_dispersion, \
                energy_content_05, energy_content_05_2, energy_content_2, \
                frequency_quotient]



to_normalize = []