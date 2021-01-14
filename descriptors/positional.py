

import numpy as np
from constants import labels

def mean_value(signal, axis = labels.ML, only_value = False):
    if not (axis in [labels.ML, labels.AP]):
        return {}
    
    feature_name = "mean_value"
    sig = signal.get_signal(axis)
    feature = np.mean(sig)

    if only_value :
        return feature

    return { feature_name+"_"+axis  : feature}





def maximal_value(signal, axis = labels.ML):
    if not (axis in [labels.ML, labels.AP,labels.RADIUS]):
        return {}
    
    feature_name = "maximum_value"
    sig = signal.get_signal(axis)
    feature = np.max(np.abs((sig)))

    return { feature_name+"_"+axis  : feature}



def mean_distance(signal, axis = labels.ML ,only_value = False):
    if not (axis in [labels.ML, labels.AP, labels.RADIUS]):
        return {}
    feature_name = "mean_distance"
    sig = signal.get_signal(axis)

    dif = np.abs( sig )

    feature = np.mean(dif)

    if only_value :
        return feature

    return { feature_name+"_"+axis  : feature}


def rms(signal, axis = labels.ML, only_value = False):
    if not (axis in [labels.ML, labels.AP, labels.RADIUS]):
        return {}
    feature_name = "RMS"
    sig = signal.get_signal(axis)

    feature = np.sqrt(np.mean(sig**2))

    if only_value :
        return feature

    return { feature_name+"_"+axis  : feature}






def amplitude(signal, axis = labels.ML,only_value = False):
    if not (axis in [labels.ML, labels.AP, labels.MLAP]):
        return {}


    feature_name = "amplitude"
    sig = signal.get_signal(axis)

    r = 0
    for i in range(len(sig)):
        d = sig - sig[i]
        dist= 0
        
        if len(sig.shape)==1:
            dist = np.abs(d)
        elif len(sig.shape)>1:
            dist = np.linalg.norm(d, axis=1)
            
        r = max(r, np.max(dist))
    feature = r

    if only_value:
        return feature
    return { feature_name+"_"+axis  : feature}



def quotient_both_direction(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}

    feature_name = "Quotient_both_direction"

    amplitude_ml = amplitude(signal,axis=labels.ML, only_value=True)
    amplitude_ap = amplitude(signal,axis=labels.AP, only_value=True)
    
    feature = amplitude_ml/amplitude_ap

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


def confidence_ellipse_area(signal, axis = labels.MLAP, only_value = False):
    if not (axis in [labels.MLAP]):
        return {}
    
    feature_name = "confidence_ellipse_area"
    
    sig = signal.get_signal(axis)

    cov = (1/len(sig)*np.sum(sig[:,0]*sig[:,1]))

    s_ml = rms(signal, axis=labels.ML, only_value=True)
    s_ap =  rms(signal, axis=labels.AP, only_value=True)

   
    feature =  2 * np.pi * 3 * np.sqrt( (s_ml**2)*(s_ap**2) - cov**2 )

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




def coeff_sway_direction(signal, axis = labels.ML):
    if not (axis in [labels.MLAP]):
        return {}

    feature_name = "Coefficient_sway_direction"

    sig = signal.get_signal(axis)

    cov = np.cov(sig[:,0],sig[:,1],bias=True)[0,1]
    s_ml = rms(signal, axis=labels.ML, only_value=True)
    s_ap =  rms(signal, axis=labels.AP, only_value=True)


    feature = (cov**2) / (s_ml * s_ap)

    return { feature_name+"_"+axis  : feature}



def planar_deviation(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}
    feature_name = "planar_deviation"

    sig = signal.get_signal(axis)

    f = np.var (sig, axis=0)

    feature = np.sqrt(np.sum(f))

    return { feature_name+"_"+axis  : feature}



def fractal_dimension_pd(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}
    
    feature_name = "fractal_dimension_pd"

    d = amplitude(signal, axis = axis,only_value = True)

    N= len(signal)
    sway_path = sway_length(signal,axis=axis,only_value = True)
    fd = np.log(N)  / (np.log(N) + np.log(d) - np.log(sway_path)  )
    feature =  fd
    return { feature_name+"_"+axis  : feature}
    

def fractal_dimension_cc(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}
    
    
    feature_name = "fractal_dimension_cc"
    r = signal.get_signal(labels.RADIUS)

    d = 2 * (np.mean(r) + 1.645 * np.std(r))

    N= len(signal)
    sway_path = sway_length(signal,axis=axis,only_value = True)

    fd = np.log(N)  / (np.log(N) + np.log(d) - np.log(sway_path)  )
    feature = fd


    return { feature_name+"_"+axis  : feature}


def fractal_dimension_ce(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}
    
    
    feature_name = "fractal_dimension_ce"

    area = confidence_ellipse_area(signal, axis =labels.MLAP, only_value=True)

    d =  np.sqrt( (area * 8) /(2 * np.pi) )

    N= len(signal)
    sway_path = sway_length(signal,axis=axis,only_value = True)

    fd = np.log(N)  / (np.log(N) + np.log(d) - np.log(sway_path)  )



    feature = fd

    return { feature_name+"_"+axis  : feature}




all_features = [mean_value, maximal_value, mean_distance, rms, amplitude, \
                quotient_both_direction, sway_length, confidence_ellipse_area, length_over_area, coeff_sway_direction,\
                planar_deviation, fractal_dimension_cc, fractal_dimension_ce, fractal_dimension_pd ]