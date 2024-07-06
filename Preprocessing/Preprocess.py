from ._load_data import _load_data
from ._df_to_array import _df_to_array
from ._scale_data import _scale_data
from ._split_data_by_scenario import _split_data_by_scenario

class Preprocess():
    def __init__(self, **params):
        self.params = params

    def preprocess_data(self):
        self.df, self.row_counts = _load_data(**self.params)
        self.Y_data_array, self.XZ_data_array = _df_to_array(self.df, **self.params)
        self.Y_data_scaled, self.Y_scaler, self.XZ_data_scaled, self.XZ_scaler = _scale_data(self.Y_data_array, self.XZ_data_array, **self.params)
        
        self.X_train_Y_coordinate, self.X_val_Y_coordinate, self.X_test_Y_coordinate, self.y_train_Y_coordinate, \
        self.y_val_Y_coordinate, self.y_test_Y_coordinate, self.X_train_XZ_coordinate, self.X_val_XZ_coordinate, \
        self.X_test_XZ_coordinate, self.y_train_XZ_coordinate, self.y_val_XZ_coordinate, self.y_test_XZ_coordinate = \
        _split_data_by_scenario(self.Y_data_scaled, self.XZ_data_scaled, self.row_counts, **self.params)       

        return self.X_train_Y_coordinate, self.X_val_Y_coordinate, self.X_test_Y_coordinate, self.y_train_Y_coordinate, \
                self.y_val_Y_coordinate, self.y_test_Y_coordinate, self.X_train_XZ_coordinate, self.X_val_XZ_coordinate, \
                self.X_test_XZ_coordinate, self.y_train_XZ_coordinate, self.y_val_XZ_coordinate, self.y_test_XZ_coordinate, \
                self.Y_scaler, self.XZ_scaler