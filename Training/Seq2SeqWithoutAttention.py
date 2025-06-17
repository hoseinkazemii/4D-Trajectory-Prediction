from .BaseMLModel import BaseMLModel
from ._Seq2SeqWithoutAttention import _construct_model, _train_and_evaluate_model

class Seq2SeqWithoutAttention(BaseMLModel):
    def __init__(self, **params):
        super(Seq2SeqWithoutAttention, self).__init__(**params)
        self.models_dict = None

    def construct_model(self):
        self.models_dict = _construct_model(**self.__dict__)

    def run(self, split_data_dict, scalers_dict, row_counts):
        _train_and_evaluate_model(
            split_data_dict=split_data_dict,
            scalers_dict=scalers_dict,
            row_counts=row_counts,
            **self.__dict__
        )
