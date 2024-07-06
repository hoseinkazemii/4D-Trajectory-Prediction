from ._load_model import _load_model
from ._construct_network import _construct_network


def _construct_model(**params):

	warm_up = params.get('warm_up')
	log = params.get("log")

	constructed = False
	if warm_up:
		try:
			model = _load_model(**params)
			constructed = True
			log.info("\n\n------------\nA trained model is loaded\n------------\n\n")
		except OSError:
			print ("The model is not trained before. No saved models found")

	if not constructed:
		# Creating the structure of the neural network
		model_Y, model_XZ = _construct_network(**params)
		

	return model_Y, model_XZ