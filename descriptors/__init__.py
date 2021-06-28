from descriptors import positional, dynamic, frequentist, stochastic
from constants import labels


        
def compute_all_features(signal):
	domains = [positional, dynamic, frequentist, stochastic]
	all_labels = labels.all_labels

	features = {}

	for domain in domains :
		for function in domain.all_features :
			for label in all_labels:
				result = function( signal, axis=label)
				features.update(result)
	return features 





def compute_all_normalized_features(signal):
    
    domains = [positional, dynamic, frequentist, stochastic]
    all_labels = labels.all_labels
    
    features = {}
    
    for domain in domains:
        
        for function in domain.all_features:
            if function in domain.to_normalize:
                for label in all_labels:
                    result = function(signal, axis=label, normalized=True)
                    features.update(result)
                    
            else:
                for label in all_labels:
                    result = function(signal, axis=label)
                    features.update(result)
                    
    return features




