from ._load_model import _load_model
from ._construct_network import _construct_network


def _construct_model(**params):
	warm_up = params.get('warm_up')
	log = params.get("log")

	if warm_up:
		try:
			model = _load_model(**params)
			constructed = True
			log.info("\n\n------------\nA trained model is loaded\n------------\n\n")
		except OSError:
			print ("The model is not trained before. No saved models found")
	else:
		# Creating the structure of the neural network
		models_dict = _construct_network(**params)
	
	return models_dict