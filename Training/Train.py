from ._build_model import _build_model
from ._train import _train
from utils import _plot_3d_trajectory
from Preprocessing import _inverse_transform

class Train():
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.y_pred = None

    def train_model(self, X_train, X_test, y_train):
        self.model = _build_model(**self.params)
        _train(X_train, y_train, self.model, **self.params)
        self.y_pred = self.model.predict(X_test)
        return self.y_pred
    
    def test_model(self, y_test, y_pred, scaler):
        y_test = _inverse_transform(scaler, y_test, **self.params)
        y_pred = _inverse_transform(scaler, y_pred, **self.params)
        _plot_3d_trajectory(y_test, y_pred, **self.params)