from .BaseMLModel import BaseMLModel
from ._Seq2SeqMultiHeadAttention import _construct_model, _train_and_evaluate_model

class Seq2SeqMultiHeadAttention(BaseMLModel):
    def __init__(self, **params):
        super(Seq2SeqMultiHeadAttention, self).__init__(**params)
        self.models_dict = None  # will store { "XZ": model, "Y": model, ... } after construct

    def construct_model(self):
        self.models_dict = _construct_model(**self.__dict__)

    def run(self, split_data_dict, scalers_dict):
        _train_and_evaluate_model(
            split_data_dict=split_data_dict,
            scalers_dict=scalers_dict,
            **self.__dict__
        )
