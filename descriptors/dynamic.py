

import numpy as np
from constants import labels
from sklearn.decomposition import PCA
import descriptors.positional as positional



def mean_velocity(signal, axis = labels.ML,only_value = False):
    if not (axis in [labels.ML, labels.AP,labels.MLAP]):
        return {}

    feature_name = "mean_velocity"
    sig = signal.get_signal(axis)

    sway = positional.sway_length(signal, axis, only_value=True)

    feature = sway * signal.frequency / len(sig) 

    if only_value:
        return feature

    return { feature_name+"_"+axis  : feature}




def zeroCrossing(signal, axis = labels.SPD_ML):
    if not (axis in [labels.SPD_ML, labels.SPD_AP]):
        return {}
    
    feature_name = "zero_crossing"
    sig = signal.get_signal(axis)

    zeros_cross = (sig[1:] * sig[:-1]) <=0

    feature = np.sum(zeros_cross)

    return { feature_name+"_"+axis  : feature}





    



def peak_velocity_all(signal, axis = labels.SPD_ML):
    if not (axis in [labels.SPD_ML, labels.SPD_AP]):
        return {}

    feature_name = "peak_velocity_all"
    sig = signal.get_signal(axis)


    past_cross = False 
    current_peak = 0
    peaks = []
    past_spd = 0

    for spd in sig :

        if spd == 0 :
            continue
        

        if spd*past_spd >= 0:
            past_spd=spd
            current_peak = max(current_peak, np.abs(spd))
            continue
        

        past_spd=spd
        if past_cross  :
            
            peaks.append(current_peak)
            

        else :
            past_cross = True

        current_peak =  0
        current_peak = max(current_peak, np.abs(spd))
        

    feature = np.mean(peaks)

    return { feature_name+"_"+axis  : feature}
    

def peak_velocity_pos(signal, axis = labels.SPD_ML):
    if not (axis in [labels.SPD_ML, labels.SPD_AP]):
        return {}

    feature_name = "peak_velocity_pos"
    sig = signal.get_signal(axis)


    past_cross = False 
    current_peak = 0
    peaks = []
    past_spd = 0

    for spd in sig :
        if spd == 0 :
            continue
        
        current_peak = max(current_peak, np.abs(spd))

        if spd*past_spd >= 0:
            past_spd=spd
            continue
        

        past_spd=spd
        if past_cross  :

            if spd <0 :
                peaks.append(current_peak)
            current_peak =  0

        else :
            past_cross = True
        

    feature = np.mean(peaks)

    return { feature_name+"_"+axis  : feature}    



def peak_velocity_neg(signal, axis = labels.SPD_ML):
    if not (axis in [labels.SPD_ML, labels.SPD_AP]):
        return {}

    feature_name = "peak_velocity_neg"
    sig = signal.get_signal(axis)


    past_cross = False 
    current_peak = 0
    peaks = []
    past_spd = 0

    for spd in sig :

        if spd == 0 :
            continue
        
        current_peak = max(current_peak, np.abs(spd))

        if spd*past_spd >= 0:
            past_spd=spd
            continue
        

        past_spd=spd
        if past_cross  :

            if spd >0 :
                peaks.append(current_peak)
            current_peak =  0

        else :
            past_cross = True
        

    feature = np.mean(peaks)

    return { feature_name+"_"+axis  : feature}    





def mean_peak_swd(signal, axis = labels.SWAY_DENSITY):
    if not (axis in [labels.SWAY_DENSITY]):
        return {}

    feature_name = "mean_peak"
    sig = signal.get_signal(axis)
    

    median = np.median(sig)

    #to avoid bugs to median = 0, when patient moves too much
    if median == 0:
        median = 0.0001

    past_cross = False 
    current_peak = 0
    current_peak_index = 0
    peaks = []
    past_swd = 0

    sig = sig - median

    for index,swd in enumerate(sig) :

        if swd == 0 :
            continue

        if (swd) *past_swd > 0:
            past_swd=swd

            if swd > current_peak :
                current_peak = swd
                current_peak_index = index
            continue
        

        past_swd=swd 
        if past_cross  :

            if past_swd  <0:            
                peaks.append(current_peak_index)
            

        else :
            past_cross = True
        current_peak =  0
        current_peak_index = 0

        if swd > current_peak :
                current_peak = swd
                current_peak_index = index

    peaks_value = [sig[u] + median for u in peaks]            

    feature = np.mean(peaks_value)
    
    return { feature_name+"_"+axis  : feature}


def mean_distance_peak_swd(signal, axis = labels.SWAY_DENSITY):
    if not (axis in [labels.SWAY_DENSITY]):
        return {}

    feature_name = "mean_distance_peak"
    sig = signal.get_signal(axis)

    rsig = signal.get_signal(labels.MLAP)
    median = np.median(sig)
    #to avoid bugs to median = 0, when patient moves too much
    if median == 0:
        median = 0.0001

    past_cross = False 
    current_peak = 0
    current_peak_index = 0
    peaks = []
    past_swd = 0

    sig = sig - median

    for index,swd in enumerate(sig) :

        if swd == 0 :
            continue

        if (swd) *past_swd > 0:
            past_swd=swd

            if swd > current_peak :
                current_peak = swd
                current_peak_index = index
            continue
        

        past_swd=swd 
        if past_cross  :

            if past_swd  <0:            
                peaks.append(current_peak_index)
            

        else :
            past_cross = True
        current_peak =  0
        current_peak_index = 0

        if swd > current_peak :
                current_peak = swd
                current_peak_index = index
        

    peak_position = [rsig[u] for u in peaks]
    peak_position = np.array(peak_position)

    dist = np.diff(peak_position, n=1, axis=0)
    dist = np.linalg.norm(dist, axis=1)



    feature = np.mean(dist)

    return { feature_name+"_"+axis  : feature}







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

   


def vfy(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}

    xy = signal.get_signal(labels.MLAP)

    dxy = signal.frequency * np.diff(xy, axis = 0, n=1)
    vdxy = np.var( np.linalg.norm(dxy, axis=1))

    y =signal.get_signal(labels.AP)
    muy = np.mean(y)
    if muy == 0:
        feature = 0

    else :
        feature = vdxy / muy

    
    
    
    feature_name = "VFY"
    

    return { feature_name+"_"+axis  : feature}




all_features = [mean_velocity, zeroCrossing, peak_velocity_pos, peak_velocity_neg, peak_velocity_all, \
                mean_peak_swd, mean_distance_peak_swd, principal_sway_direction, mean_frequency, \
                phase_plane_parameters, sway_area_per_second, vfy]