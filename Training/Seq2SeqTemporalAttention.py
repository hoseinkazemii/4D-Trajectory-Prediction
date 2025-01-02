from .BaseMLModel import BaseMLModel
from ._Seq2SeqTemporalAttention import _construct_model
from ._Seq2SeqTemporalAttention import _train_and_evaluate_model

class Seq2SeqTemporalAttention(BaseMLModel):
    def __init__(self, **params):
        super(Seq2SeqTemporalAttention, self).__init__(**params)

    def construct_model(self):
        self.model_Y, self.model_XZ = _construct_model(**self.__dict__)

    def run(self, X_train_Y_coordinate, X_val_Y_coordinate, X_test_Y_coordinate, y_train_Y_coordinate, y_val_Y_coordinate, y_test_Y_coordinate, \
            X_train_XZ_coordinate, X_val_XZ_coordinate, X_test_XZ_coordinate, y_train_XZ_coordinate, y_val_XZ_coordinate, y_test_XZ_coordinate, \
            Y_scaler, XZ_scaler):

        _train_and_evaluate_model(X_train_Y_coordinate, X_val_Y_coordinate, X_test_Y_coordinate, y_train_Y_coordinate, y_val_Y_coordinate, y_test_Y_coordinate, \
                                  X_train_XZ_coordinate, X_val_XZ_coordinate, X_test_XZ_coordinate, y_train_XZ_coordinate, y_val_XZ_coordinate, y_test_XZ_coordinate, \
                                  Y_scaler, XZ_scaler, **self.__dict__)