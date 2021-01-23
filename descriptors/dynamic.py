import numpy as np
from constants import labels
import descriptors.positional as positional



def sway_length(signal, axis = labels.ML,only_value = False):
    if not (axis in [labels.ML, labels.AP,labels.MLAP]):
        return {}
    feature_name = "sway_length"
    
    sig = signal.get_signal(axis)

    dif = np.diff(sig, n=1, axis=0)
    dif = np.linalg.norm(dif, axis=1)
    feature = np.sum(dif)

    if only_value:
        return feature

    return { feature_name+"_"+axis  : feature}



def mean_velocity(signal, axis = labels.ML, only_value = False):
    if not (axis in [labels.ML, labels.AP, labels.MLAP]):
        return {}
    feature_name = "mean_velocity"
    
    sig = signal.get_signal(axis)

    sway = sway_length(signal, axis, only_value=True)

    feature = sway * (signal.frequency / len(sig)) 

    if only_value:
        return feature

    return { feature_name+"_"+axis  : feature}



def sway_area_per_second(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}  
    feature_name = "sway_area_per_second"
    
    sig = signal.get_signal(axis)

    dt = 1/ signal.frequency

    duration = (len(sig) -1) * dt
    assert duration>0

    triangles = np.abs(sig[1:,0] * sig[:-1,1] - sig[1:,1] * sig[:-1,0])

    feature =  np.sum(triangles) / (2*duration)

    return { feature_name+"_"+axis  : feature}



def phase_plane_parameter(signal, axis = labels.ML):
    if not (axis in [labels.ML, labels.AP]):
        return {}
    feature_name = "phase_plane_parameter"

    std_sig = positional.rms(signal, axis=axis, only_value=True)
    
    if axis == labels.ML:
        spd = signal.get_signal(labels.SPD_ML)
    elif axis == labels.AP:
        spd = signal.get_signal(labels.SPD_AP)

    feature = np.sqrt(std_sig**2 + np.var(spd))
    
    return { feature_name+"_"+axis  : feature}



def vfy(signal, axis = labels.SPD_MLAP):
    if not (axis in [labels.SPD_MLAP]):
        return {}
    feature_name = "vfy"

    std = signal.get_signal(axis)

    vdxy = np.var(std)

    muy = signal.mean_value[1]

    if muy == 0:
        muy = 0.0001

    feature = vdxy / muy

    return {feature_name+"_"+axis  : feature}



def length_over_area(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}
    feature_name = "length_over_area"

    length = sway_length(signal, axis = labels.MLAP, only_value = True)
    area = positional.confidence_ellipse_area(signal, axis = labels.MLAP, \
                                   only_value = True)
    
    feature =  length/area

    return { feature_name+"_"+axis  : feature}



def fractal_dimension_ce(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}
    feature_name = "fractal_dimension"

    area = positional.confidence_ellipse_area(signal, axis=labels.MLAP, \
                                              only_value=True)

    d = np.sqrt((area * 4) / np.pi)

    N = len(signal)
    
    sway = sway_length(signal,axis=axis,only_value = True)

    fd = np.log(N) / (np.log(N) + np.log(d) - np.log(sway))

    feature = fd

    return { feature_name+"_"+axis  : feature}



def velocity_peaks(signal, axis=labels.SPD_ML):
    if not (axis in [labels.SPD_ML, labels.SPD_AP]):
        return {}

    sig = signal.get_signal(axis)

    current_peak = 0
    current_peak_index = 0
    past_value = 0
    zero_crossing_index = []
    negative_peaks_index = []
    positive_peaks_index = []
    current_side = np.sign(sig[sig!=0][0])

    for index,value in enumerate(sig) : 

        is_crossing_point = ( (value)*past_value <= 0 ) \
                            and (index != 0) \
                            and ( value != 0 ) \
                            and ( np.sign(value) != current_side )

        if is_crossing_point:

            if len(zero_crossing_index)>0:
                
                if value < 0:
                    positive_peaks_index.append(current_peak_index)

                elif value > 0:
                    negative_peaks_index.append(current_peak_index)

            zero_crossing_index += [index-1, index]
            current_side = np.sign(value)

            current_peak = 0

        if np.abs(value) > np.abs(current_peak) :
                current_peak = value
                current_peak_index = index

        past_value=value
   
    positive_peaks = sig[np.array(positive_peaks_index)]
    negative_peaks = np.abs(sig[np.array(negative_peaks_index)])
    all_peaks = np.abs(sig[np.array(positive_peaks_index + negative_peaks_index)])

    return {'zero_crossing'+'_'+axis : int(len(zero_crossing_index)/2),
            'peak_velocity_pos'+'_'+axis : np.mean(positive_peaks),
            'peak_velocity_neg'+'_'+axis : np.mean(negative_peaks),
            'peak_velocity_all'+'_'+axis : np.mean(all_peaks)}
    
 

def swd_peaks(signal, axis=labels.SWAY_DENSITY):
    if not (axis in [labels.SWAY_DENSITY]):
        return {}

    sig = signal.get_signal(axis)

    rsig = signal.get_signal(labels.MLAP)

    crossing_border = np.median(sig)
    
    #to avoid bugs to crossing_border = 0, when individual moves too much
    if crossing_border == 0:
        crossing_border = 0.0001

    sig = sig - crossing_border

    current_peak = 0
    current_peak_index = 0
    past_value = 0
    zero_crossing_index = []
    positive_peaks_index = []
    current_side = np.sign(sig[sig!=0][0])

    for index,value in enumerate(sig) : 
        
        is_crossing_point = ( (value)*past_value <= 0 ) and (index != 0)\
                            and ( value != 0 ) and ( np.sign(value) != current_side )

        if is_crossing_point:

            if len(zero_crossing_index)>0:
            
                if value < 0:
                    
                    positive_peaks_index.append(current_peak_index)

            zero_crossing_index += [index-1, index]
            current_side = np.sign(value)

            current_peak = 0

        if value > current_peak :
                current_peak = value
                current_peak_index = index

        past_value=value

    positive_peaks = sig[np.array(positive_peaks_index)] + crossing_border

    peak_position = np.array([rsig[u] for u in positive_peaks_index])
    dist = np.diff(peak_position, n=1, axis=0)
    dist = np.linalg.norm(dist, axis=1)

    return {'mean_peak'+'_'+axis : np.mean(positive_peaks),
            'mean_distance_peak'+'_'+axis : np.mean(dist)}



def mean_frequency(signal, axis = labels.ML):
    if not (axis in [labels.ML, labels.AP, labels.MLAP]):
        return {}    
    feature_name = "mean_frequency"

    sig = signal.get_signal(axis)

    spd = np.linalg.norm(signal.frequency * ( np.diff(sig, n=1, axis=0)), axis=1,keepdims=True)
        
    if axis==labels.MLAP:
        dist = positional.mean_distance(signal, axis = labels.RADIUS, \
                                        only_value = True)
        feature =  (1/(2 * np.pi)) * ( np.mean(spd)/dist)
        
    else:
        dist = positional.mean_distance(signal, axis = axis, only_value = True)
        feature =  (1/(4*np.sqrt(2))) * ( np.mean(spd)/dist)
        
    return { feature_name+"_"+axis  : feature}



all_features = [mean_velocity, sway_area_per_second, phase_plane_parameter, 
                vfy, length_over_area, fractal_dimension_ce, velocity_peaks, \
                swd_peaks, mean_frequency]