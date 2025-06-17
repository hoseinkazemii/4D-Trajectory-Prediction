from .BaseMLModel import BaseMLModel
from ._ConvLSTM import _construct_model_convlstm
from ._ConvLSTM import _train_and_evaluate_model_convlstm

class ConvLSTM(BaseMLModel):
    def __init__(self, **params):
        super(ConvLSTM, self).__init__(**params)
        self.models_dict = None

    def construct_model(self):
        self.models_dict = _construct_model_convlstm(**self.__dict__)

    def run(self, split_data_dict, scalers_dict, row_counts):
        _train_and_evaluate_model_convlstm(
            split_data_dict=split_data_dict,
            scalers_dict=scalers_dict,
            row_counts=row_counts,
            **self.__dict__
        )
