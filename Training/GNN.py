from .BaseMLModel import BaseMLModel
from ._GNN import _construct_model, _train_and_evaluate_model

class GNN(BaseMLModel):
    def __init__(self, **params):
        super(GNN, self).__init__(**params)
        self.model = None

    def construct_model(self):
        self.model = _construct_model(**self.__dict__)

    def run(self, split_data_dict, scalers_dict, row_counts):
        _train_and_evaluate_model(
            split_data_dict=split_data_dict,
            scalers_dict=scalers_dict,
            row_counts=row_counts,
            **self.__dict__
        )