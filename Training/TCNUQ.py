# TCNUQ.py
from .BaseMLModel import BaseMLModel
from ._TCNUQ import _construct_model_tcnuq, _train_and_evaluate_model_tcnuq

class TCNUQ(BaseMLModel):
    def __init__(self, **params):
        super(TCNUQ, self).__init__(**params)
        self.model = None

    def construct_model(self):
        self.model = _construct_model_tcnuq(**self.__dict__)

    def run(self, split_data_dict, scalers_dict, row_counts):
        _train_and_evaluate_model_tcnuq(
            split_data_dict=split_data_dict,
            scalers_dict=scalers_dict,
            row_counts=row_counts,
            **self.__dict__
        )
