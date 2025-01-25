from ._load_data import _load_data
from ._df_to_array import _dfs_to_array   # Now handles a SINGLE DataFrame and returns a dict of arrays
from ._scale_data import _scale_data     # We'll revise to fit a global scaler for each coordinate across ALL scenarios
from ._split_data_by_scenario import _split_data_by_scenario
from ._build_graph_data import _build_graph_data

class Preprocess():
    def __init__(self, **params):
        self.params = params

        # These will be populated in preprocess_data
        self.df_list = None            # list of DataFrames (one per scenario)
        self.row_counts = None         # how many rows each scenario has
        self.arrays_list = None        # list of dicts { coord_str: unscaled_array }
        self.scaled_arrays_list = None # list of dicts { coord_str: scaled_array }
        self.scalers_dict = None       # dict of scalers keyed by coordinate
        self.split_data_dict = None    # final train/val/test splits

    def preprocess_data(self):
        """
        Preprocess the data from multiple scenario CSV files:
        1) Load each file into a separate DataFrame. (A list of dataframes, No concatenation)
        2) Convert each DF to unscaled arrays (dict form).
        3) Fit scalers across all scenarios (per coordinate) & transform them => scaled_arrays_list
        4) Split the scaled arrays into train/val/test by scenario index (no cross-scenario sequences).
        """
        # Load the datasets => list of DataFrames (one per scenario)
        self.df_list, self.row_counts = _load_data(**self.params)
        # e.g. df_list[0], df_list[1], ... each is a separate scenario's DataFrame
        self.arrays_list = _dfs_to_array(self.df_list, **self.params)
        # 3) Scale the arrays across ALL scenarios for each coordinate
        #    so they share the same scale. `_scale_data` will do a "two-pass":
        #    (a) gather data for each coordinate from all scenarios,
        #    (b) fit one scaler per coordinate, (c) transform each scenario's arrays.
        self.scaled_arrays_list, self.scalers_dict = _scale_data(self.arrays_list, **self.params)
        # 4) Split the scaled data into train/val/test sets by scenario index
        #    => no cross-file boundaries
        self.split_data_dict = _split_data_by_scenario(self.scaled_arrays_list, **self.params)

        if self.params.get("use_gnn"):
            from ._build_graph_data import _build_graph_data
            self.graph_data_dict = _build_graph_data(self.split_data_dict, **self.params)
        else:
            self.graph_data_dict = None


        return self.split_data_dict, self.scalers_dict, self.row_counts


# from ._load_data import _load_data
# from ._df_to_array import _df_to_array
# from ._scale_data import _scale_data
# from ._split_data_by_scenario import _split_data_by_scenario
# # from ._plot_distributions import _plot_distributions

# class Preprocess():
#     def __init__(self, **params):
#         self.params = params

#     def preprocess_data(self):
#         self.df, self.row_counts = _load_data(**self.params)
#         self.data_arrays_dict_unscaled = _df_to_array(self.df, **self.params)
#         self.scaled_arrays_dict_scaled, self.scalers_dict = _scale_data(self.data_arrays_dict_unscaled, **self.params)
#         # _plot_distributions(self.data_arrays_dict_unscaled["Z"], self.scaled_arrays_dict_scaled["Z"])
#         self.split_data_dict = _split_data_by_scenario(self.scaled_arrays_dict_scaled, self.row_counts, **self.params)

#         return self.split_data_dict, self.scalers_dict
