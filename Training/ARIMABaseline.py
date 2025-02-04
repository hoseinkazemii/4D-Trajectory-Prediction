from .BaseMLModel import BaseMLModel
from ._ARIMABaseline import _train_and_evaluate_arima

class ARIMABaseline(BaseMLModel):
    def __init__(self, **params):
        """
        ARIMA baseline for 4D Trajectory Prediction.
        
        In addition to the common parameters, the following are used:
          - arima_order: tuple, the (p, d, q) order for the ARIMA model (default: (5,1,0)).
          - coordinates: list of coordinate groups (e.g. ["X", "Y", "Z"] or ["XZ"], etc.)
          - sequence_length: length of input window
          - prediction_horizon: number of future steps to forecast
        """
        super(ARIMABaseline, self).__init__(**params)
        self.arima_order = params.get("arima_order", (5, 1, 0))
        # For ARIMA there is no network to construct; we simply store the order per coordinate group.
        self.models_dict = None

    def construct_model(self):
        # There is no network to build for ARIMA. We just save the ARIMA order
        # for each coordinate group in a dictionary (to mimic your API).
        self.models_dict = {coord: self.arima_order for coord in self.coordinates}

    def run(self, split_data_dict, scalers_dict, row_counts):
        _train_and_evaluate_arima(
            split_data_dict=split_data_dict,
            scalers_dict=scalers_dict,
            row_counts=row_counts,
            **self.__dict__
        )
