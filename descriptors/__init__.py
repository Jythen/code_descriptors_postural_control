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

