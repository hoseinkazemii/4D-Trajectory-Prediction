from ._load_data import _load_data
from ._df_to_array import _df_to_array
from ._scale_data import _scale_data
from ._split_data_by_scenario import _split_data_by_scenario
# from ._plot_distributions import _plot_distributions

class Preprocess():
    def __init__(self, **params):
        self.params = params

    def preprocess_data(self):
        self.df, self.row_counts = _load_data(**self.params)
        self.data_arrays_dict_unscaled = _df_to_array(self.df, **self.params)
        self.scaled_arrays_dict_scaled, self.scalers_dict = _scale_data(self.data_arrays_dict_unscaled, **self.params)
        # _plot_distributions(self.data_arrays_dict_unscaled["Z"], self.scaled_arrays_dict_scaled["Z"])
        self.split_data_dict = _split_data_by_scenario(self.scaled_arrays_dict_scaled, self.row_counts, **self.params)

        return self.split_data_dict, self.scalers_dict
