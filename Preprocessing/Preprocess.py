# Preprocessing/Preprocess.py
from ._load_data import _load_data
from ._df_to_array import _df_to_array
from ._scale_data import _scale_data
from ._split_data_by_scenario import _split_data_by_scenario

class Preprocess():
    def __init__(self, **params):
        self.params = params
        self.df = None
        self.data_array = None
        self.scaler = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def preprocess_data(self, coordinate):
        self.df, row_counts = _load_data(**self.params)
        self.data_array = _df_to_array(self.df, **self.params)
        
        if coordinate == 'Y':
            # Extract the Y coordinate
            data = self.data_array[:, 1:2]
        else:
            # Extract the combined X and Z coordinates
            data = self.data_array[:, [0, 2]]
        
        # Scale the data
        scaled_data, self.scaler = _scale_data(data, **self.params)
        
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = _split_data_by_scenario(scaled_data, row_counts, combined=(coordinate != 'Y'), **self.params)

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.scaler