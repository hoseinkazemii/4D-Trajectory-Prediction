from .BaseMLModel import BaseMLModel
from ._TCN import _construct_model_tcn, _train_and_evaluate_model_tcn

class TCN(BaseMLModel):
    def __init__(self, **params):
        super(TCN, self).__init__(**params)
        self.models_dict = None

    def construct_model(self):
        self.models_dict = _construct_model_tcn(**self.__dict__)

    def run(self, split_data_dict, scalers_dict, row_counts):
        _train_and_evaluate_model_tcn(
            split_data_dict=split_data_dict,
            scalers_dict=scalers_dict,
            row_counts=row_counts,
            **self.__dict__
        )
