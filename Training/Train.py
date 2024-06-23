from ._build_model import _build_model
from ._train import _train
from utils import _plot_3d_trajectory
from Preprocessing import _inverse_transform


class Train():
    def __init__(self, **params):
        pass

    def train_model(self, X_train, X_test, y_train, **params):
        model = _build_model(**params)
        _train(X_train, y_train, model, **params)
        y_pred = model.predict(X_test)
        return y_pred
    
    def test_model(self, y_test, y_pred, scaler, **params):
        y_test = _inverse_transform(scaler, y_test, **params)
        y_pred = _inverse_transform(scaler, y_pred, **params)
        _plot_3d_trajectory(y_test, y_pred, **params)