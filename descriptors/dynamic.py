

import numpy as np
from constants import labels
from sklearn.decomposition import PCA
import descriptors.positional as positional
import matplotlib.pyplot as plt



import scipy

import pandas




def mean_velocity(signal, axis = labels.ML,only_value = False):
    if not (axis in [labels.ML, labels.AP, labels.MLAP]):
        return {}

    feature_name = "mean_velocity"
    sig = signal.get_signal(axis)

    sway = positional.sway_length(signal, axis, only_value=True)

    feature = sway * signal.frequency / len(sig) 

    if only_value:
        return feature

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

        is_crossing_point = ( (value)*past_value <= 0 ) and (index != 0)\
                            and ( value != 0 ) and ( np.sign(value) != current_side )

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

    return {'peak_velocity_all'+'_'+axis : np.mean(all_peaks),
            'peak_velocity_pos'+'_'+axis : np.mean(positive_peaks),
            'peak_velocity_neg'+'_'+axis : np.mean(negative_peaks),
            'zero_crossing'+'_'+axis : len(zero_crossing_index)/2}
    
 



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




def principal_sway_direction(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}
    
    feature_name = "principal_sway_direction"
    sig = signal.get_signal(axis)


    dist = np.diff(sig,1, axis=0)
    pca = PCA(n_components= 2)
    pca.fit(dist)
    main_direction = pca.components_[0]

    angle_rad = np.arccos(np.abs(main_direction[1])/np.linalg.norm(main_direction))
    
    feature = angle_rad*(180/np.pi)

    return { feature_name+"_"+axis  : feature}


def mean_frequency(signal, axis = labels.ML):
    if not (axis in [labels.ML, labels.AP, labels.MLAP]):
        return {}
    
    feature_name = "mean_frequency"

    sig = signal.get_signal(axis)
    spd = np.linalg.norm(signal.frequency * ( np.diff(sig, n=1, axis=0)), axis=1,keepdims=True)

    
  
        
    if axis==labels.MLAP:
        dist = positional.mean_distance(signal, axis = labels.RADIUS,only_value = True)
        feature =  (1/(2 * np.pi)) * ( np.mean(spd)/dist)
        
    else:
        dist = positional.mean_distance(signal, axis = axis,only_value = True)
        feature =  (1/(4*np.sqrt(2))) * ( np.mean(spd)/dist)
        
    return { feature_name+"_"+axis  : feature}

    
def phase_plane_parameters(signal, axis = labels.ML):
    if not (axis in [labels.ML, labels.AP]):
        return {}
    
    feature_name = "phase_plane_parameters"
    sig = signal.get_signal(axis)

    dsig = signal.frequency * ( np.diff(sig, n=1, axis=0))
    feature = np.sqrt( np.var(sig) + np.var(dsig) )
    return { feature_name+"_"+axis  : feature}




def sway_length(signal, axis = labels.ML,only_value = False):
    if not (axis in [labels.ML, labels.AP,labels.MLAP]):
        return {}

    feature_name = "sway_length"
    sig = signal.get_signal(axis)

    dif = np.diff(sig, n=1, axis =0)
    dif = np.linalg.norm(dif, axis=1)
    feature = np.sum(dif)

    if only_value:
        return feature

    return { feature_name+"_"+axis  : feature}


def length_over_area(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}
    
    feature_name = "length_over_area"

    length = sway_length(signal,axis = labels.MLAP, only_value = True)
    area = confidence_ellipse_area(signal, axis = labels.MLAP,only_value = True)
    
    feature =  length/area

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



def fractal_dimension_ce(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}
    
    
    feature_name = "fractal_dimension"

    area = confidence_ellipse_area(signal, axis =labels.MLAP, only_value=True)

    d =  np.sqrt( (area * 8) /(2 * np.pi) )

    N= len(signal)
    sway_path = sway_length(signal,axis=axis,only_value = True)

    fd = np.log(N)  / (np.log(N) + np.log(d) - np.log(sway_path)  )



    feature = fd

    return { feature_name+"_"+axis  : feature}

   


def vfy(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}

    xy = signal.get_signal(labels.MLAP)

    dxy = signal.frequency * np.diff(xy, axis = 0, n=1)
    vdxy = np.var( np.linalg.norm(dxy, axis=1))

    muy = signal.mean_value[1]
    if muy == 0:
        feature = 0

    else :
        feature = vdxy / muy

    
    
    
    feature_name = "VFY"
    

    return { feature_name+"_"+axis  : feature}




all_features = [mean_velocity, length_over_area, fractal_dimension_ce, velocity_peaks, velocity_peaks,
                principal_sway_direction, mean_frequency, \
                phase_plane_parameters, sway_area_per_second, vfy]