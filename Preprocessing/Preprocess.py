from ._load_data import _load_data
from ._df_to_array import _df_to_array
from ._scale_data import _scale_data
from ._split_data import _split_data

class Preprocess():
    def __init__(self, **params):
        self.params = params
        self.df = None
        self.data = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_data(self):
        self.df = _load_data(**self.params)
        self.data = _df_to_array(self.df, **self.params)
        self.data, self.scaler = _scale_data(self.data, **self.params)
        self.X_train, self.X_test, self.y_train, self.y_test = _split_data(self.data, **self.params)
        return self.X_train, self.X_test, self.y_train, self.y_test, self.scaler