import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from constants import labels



def mean_value(signal, axis = labels.ML, only_value = False):
    if not (axis in [labels.ML, labels.AP]):
        return {}  
    feature_name = "mean_value"

    if axis==labels.ML:
        feature = signal.mean_value[0]
    else:
        feature = signal.mean_value[1]

    if only_value :
        return feature

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



def maximal_distance(signal, axis = labels.ML):
    if not (axis in [labels.ML, labels.AP,labels.RADIUS]):
        return {} 
    feature_name = "maximal_distance"
    
    sig = signal.get_signal(axis)
    feature = np.max(np.abs((sig)))

    return { feature_name+"_"+axis  : feature}



def rms(signal, axis = labels.ML, only_value = False):
    if not (axis in [labels.ML, labels.AP, labels.RADIUS]):
        return {}
    feature_name = "rms"
    
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
    feature_name = "quotient_both_direction"

    amplitude_ml = amplitude(signal,axis=labels.ML, only_value=True)
    amplitude_ap = amplitude(signal,axis=labels.AP, only_value=True)
    
    feature = amplitude_ml/amplitude_ap

    return { feature_name+"_"+axis  : feature}



def planar_deviation(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}
    feature_name = "planar_deviation"

    s_ml = rms(signal, axis=labels.ML, only_value=True)
    s_ap =  rms(signal, axis=labels.AP, only_value=True)

    feature = np.sqrt(s_ml**2 + s_ap**2)

    return { feature_name+"_"+axis  : feature}

    

def coeff_sway_direction(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}
    feature_name = "coefficient_sway_direction"

    sig = signal.get_signal(axis)

    cov = (1/len(sig)*np.sum(sig[:,0]*sig[:,1]))
    s_ml = rms(signal, axis=labels.ML, only_value=True)
    s_ap =  rms(signal, axis=labels.AP, only_value=True)


    feature = cov / (s_ml * s_ap)

    return { feature_name+"_"+axis  : feature}



def confidence_ellipse_area(signal, axis = labels.MLAP, only_value = False):
    if not (axis in [labels.MLAP]):
        return {}
    feature_name = "confidence_ellipse_area"
    
    sig = signal.get_signal(axis)

    cov = (1/len(sig)*np.sum(sig[:,0]*sig[:,1]))

    s_ml = rms(signal, axis=labels.ML, only_value=True)
    s_ap =  rms(signal, axis=labels.AP, only_value=True)

    confidence = 0.95

    quant = stats.f.ppf(confidence, 2, len(sig)-2)
   
    det = (s_ml**2)*(s_ap**2) - cov**2
    feature =  2 * np.pi * ((len(sig)-1)/(len(sig)-2)) * quant * np.sqrt(det)

    if only_value:
        return feature

    return { feature_name+"_"+axis  : feature}



def principal_sway_direction(signal, axis = labels.MLAP):
    if not (axis in [labels.MLAP]):
        return {}
    feature_name = "principal_sway_direction"
    
    sig = signal.get_signal(axis)

    pca = PCA(n_components= 2)
    pca.fit(sig)
    main_direction = pca.components_[0]

    angle_rad = np.arccos(np.abs(main_direction[1])/np.linalg.norm(main_direction))
    
    feature = angle_rad*(180/np.pi)

    return { feature_name+"_"+axis  : feature}



all_features = [mean_value, mean_distance, maximal_distance, rms, amplitude, \
                quotient_both_direction, planar_deviation, \
                coeff_sway_direction, confidence_ellipse_area, \
                principal_sway_direction]


to_normalize = []