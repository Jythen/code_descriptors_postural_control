from descriptors import positional, dynamic, frequentist, stochastic
from constants import labels


        
functions_with_params = {"swd_peaks": ["sway_density_radius"]}


def compute_all_features(signal, params_dic):
    
    
    domains = [positional, dynamic, frequentist, stochastic]
    
    all_labels = labels.all_labels
    
    features = {}
    
    
    for domain in domains:
        
        for function in domain.all_features:
            
            params = None
            
            for key in functions_with_params:
                
                if key in str(function):
                    
                    params = {param: params_dic[param] for param in functions_with_params[key]}

            for label in all_labels:
                
                
                if params is not None:
                    
                    result = function(signal, **params)
                
                else:
                    result = function(signal, axis=label)
                
                features.update(result)
                
                
                
    return features





