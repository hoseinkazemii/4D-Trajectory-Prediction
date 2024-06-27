# Training/Train.py
from .build_model_with_attention import _build_model_with_attention
from ._train import _train
from utils import _plot_3d_trajectory
from Preprocessing import _inverse_transform

class Train():
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.y_pred = None

    def train_model(self, X_train, X_test, X_val, y_train, y_val, coordinate):
        self.model = _build_model_with_attention(**self.params)
        history = self.model.fit(X_train, y_train, epochs=self.params['num_epochs'], batch_size=self.params['batch_size'], validation_data=(X_val, y_val))
        self.y_pred = self.model.predict(X_test)
        return self.y_pred, history
    
    def test_model(self, y_test, y_pred, scaler, coordinate):
        y_test = _inverse_transform(scaler, y_test, **self.params)
        y_pred = _inverse_transform(scaler, y_pred, **self.params)
        _plot_3d_trajectory(y_test, y_pred, **self.params)