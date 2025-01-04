from ._load_data import _load_data
from ._df_to_array import _df_to_array
from ._scale_data import _scale_data
from ._split_data_by_scenario import _split_data_by_scenario

class Preprocess():
    def __init__(self, **params):
        self.params = params

    def preprocess_data(self):
        self.df, self.row_counts = _load_data(**self.params)
        self.data_arrays_dict = _df_to_array(self.df, **self.params)
        self.scaled_arrays_dict, self.scalers_dict = _scale_data(self.data_arrays_dict, **self.params)
        self.split_data_dict = _split_data_by_scenario(self.scaled_arrays_dict, self.row_counts, **self.params)

        return self.split_data_dict, self.scalers_dict
