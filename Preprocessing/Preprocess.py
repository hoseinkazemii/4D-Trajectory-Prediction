from ._load_data import _load_data
from ._df_to_array import _df_to_array
from ._scale_data import _scale_data
from ._split_data import _split_data


class Preprocess():
    def __init__(self, **params):
        self.params = params

    def preprocess_data(self, **params):
        df = _load_data(**params)
        data = _df_to_array(df, **params)
        data, scaler = _scale_data(data, **params)
        X_train, X_test, y_train, y_test = _split_data(data, **params)


        return X_train, X_test, y_train, y_test, scaler
