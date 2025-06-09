import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
import multiprocessing as mp
import warnings
from statsmodels.tsa.base.tsa_model import ValueWarning

warnings.filterwarnings("ignore", category=ValueWarning)


class Model:
    """Base class for all financial models.

    Provides common functionality and structure for derived model classes.

    Parameters
    ----------
    run_sett : dict
        Dictionary containing all runtime settings and parameters
    clustered_dfs_sets : list
        List of dictionaries of DataFrames containing clustered financial data
    """

    def __init__(
        self,
        run_sett: dict,
        clustered_dfs_train_sets,
        clustered_dfs_test_sets,
        forecasted_dates,
    ):
        self.clustered_dfs_train_sets = clustered_dfs_train_sets
        self.clustered_dfs_test_sets = clustered_dfs_test_sets
        self.forecasted_dates = forecasted_dates
        self._run_sett = run_sett


class UAM(Model):
    """Universal Asset Model (UAM) for multi-cluster financial modeling.

    Manages multiple Cluster Asset Models (CAM) to provide consolidated
    predictions across different asset clusters.

    Parameters
    ----------
    run_sett : dict
        Dictionary containing all runtime settings and parameters
    clustered_dfs : dict
        Dictionary of DataFrames containing clustered financial data
    """

    def __init__(
        self,
        run_sett: dict,
        clustered_dfs_train_sets,
        clustered_dfs_test_sets,
        forecasted_dates,
    ):
        super().__init__(
            run_sett,
            clustered_dfs_train_sets,
            clustered_dfs_test_sets,
            forecasted_dates,
        )
        (
            self.train_data_per_cam,
            self.test_data_per_cam,
        ) = self.create_data_sets_per_cam()
        self.cam_objs = self.combine_cams()
        self.model_sett = self._run_sett["models"]["UAM"]

    def create_data_sets_per_cam(self):
        train_data_per_cam = {}
        test_data_per_cam = {}
        for cluster_label in list(self.clustered_dfs_train_sets[0].keys()):
            train_data_per_cam[cluster_label] = [
                d[cluster_label] for d in self.clustered_dfs_train_sets
            ]
            test_data_per_cam[cluster_label] = [
                d[cluster_label] for d in self.clustered_dfs_test_sets
            ]

        return train_data_per_cam, test_data_per_cam

    def combine_cams(self):
        cam_objs = {}
        for cluster_label in list(self.train_data_per_cam.keys()):
            cluster_train_sets = self.train_data_per_cam[cluster_label]
            cluster_test_sets = self.test_data_per_cam[cluster_label]
            cam_obj = CAM(
                self._run_sett,
                self.clustered_dfs_train_sets,
                self.clustered_dfs_test_sets,
                cluster_train_sets,
                cluster_test_sets,
                self.forecasted_dates,
            )
            cam_objs[cluster_label] = cam_obj

        cam_objs = self.decile_portfolio_trading(cam_objs)

        return cam_objs

    def decile_portfolio_trading(self, cam_objs):
        """
        Concatenate forecast elements of each cam_obj.
        TEST WHETHER THIS IS CORRECT!!!!!!!!
        """

        clusters = [cam_obj.forecasts for cam_obj in cam_objs.values()]
        correct_dim_clusters = []
        for cluster in clusters:
            uniform_arrays = [np.expand_dims(arr, axis=1) if arr.ndim == 1 else arr for arr in cluster]
            correct_dim_clusters.append(uniform_arrays)

        grouped_arrays = list(zip(*correct_dim_clusters))
        concatenated_result = [np.concatenate(arrays, axis=1) for arrays in grouped_arrays]

        percentiles_75 = []
        for universal_forecasts_per_test_data in concatenated_result:
            percentiles_75.append(np.percentile(np.abs(universal_forecasts_per_test_data), 75, axis=1, keepdims=True))

        for cam_obj in cam_objs.values():
            for j in range(len(cam_obj.forecasts)):
                cam_obj.forecasts[j] = np.where(
                    cam_obj.forecasts[j] > percentiles_75[j], 
                    1, 
                    np.where(cam_obj.forecasts[j] < -percentiles_75[j], -1, 0)
                )
        return cam_objs
            


class CAM(Model):
    """Cluster Asset Model (CAM) for single-cluster financial modeling.

    Implements a Vector Autoregression (VAR) model for predicting asset
    movements within a specific cluster using rolling window analysis.

    Parameters
    ----------
    run_sett : dict
        Dictionary containing all runtime settings and parameters
    clustered_dfs : dict
        Dictionary of DataFrames containing clustered financial data
    cluster_data : pd.DataFrame
        Data for the specific cluster being modeled
    """

    def __init__(
        self,
        run_sett: dict,
        clustered_dfs_train_sets,
        clustered_dfs_test_sets,
        cluster_train_sets,
        cluster_test_sets,
        forecasted_dates,
    ):
        super().__init__(
            run_sett,
            clustered_dfs_train_sets,
            clustered_dfs_test_sets,
            forecasted_dates,
        )
        self.model_sett = self._run_sett["models"]["CAM"]
        self.model_lag = self.model_sett["p"]
        self.forecasted_dates = forecasted_dates
        self.cluster_train_data = cluster_train_sets
        self.cluster_test_data = cluster_test_sets
        self.assets = [test_data.index for test_data in self.cluster_test_data]
        self.forecasts = self.calculate()

    def calculate(self):
        """Execute the main calculation pipeline for the CAM model.

        Returns
        -------
        tuple
            - forecasts : list of numpy.ndarray
            - forecasted_dates : pandas.DatetimeIndex
                Dates corresponding to the forecasts
        """

        pool = mp.Pool(mp.cpu_count())
        results = pool.map(
            self.train, [data_set for data_set in self.cluster_train_data]
        )
        forecasts = [
            pool.apply(test, args=(test_data_set, result))
            for test_data_set, result in zip(self.cluster_test_data, results)
        ]
        pool.close()

        return forecasts

    def train(self, data):
        """Train a VAR model on a single window of data.

        Parameters
        ----------
        data : pd.DataFrame
            Training data for the current window

        Returns
        -------
        statsmodels.tsa.vector_ar.var_model.VARResults
            Fitted VAR model results

        Notes
        -----
        Currently limited to first 20 columns for computational efficiency
        """
        adj_train = data.transpose()
        if (
            adj_train.shape[1] == 1
        ):  # probably not necessary when fitting on entire data
            result = AutoReg(adj_train, lags=self.model_lag).fit()
        else:
            model = VAR(adj_train)
            result = model.fit(self.model_lag)

        return result


def test(data, result):
    """Generate predictions using a trained VAR model.

    Parameters
    ----------
    data : pd.DataFrame
        Test data to generate predictions for
    result : statsmodels.tsa.vector_ar.var_model.VARResults
        Fitted VAR model

    Returns
    -------
    numpy.ndarray
        Binary predictions (-1 or 1) for asset movements

    Notes
    -----
    - Currently limited to first 20 columns for computational efficiency
    - Uses 0.00005 as the threshold for positive/negative prediction
    """
    all_forecasts = []
    adj_test = data.transpose()
    if adj_test.shape[1] == 1:
        for step in range(data.shape[1]):
            intercept = result.params[0]
            coefs = result.params[1:]
            num_lags = len(coefs)
            forecast = intercept + np.dot(
                coefs, adj_test.values[step : step + num_lags]
            )
            all_forecasts.append(*forecast)
    else:
        for step in range(data.shape[1]):
            forecast = result.forecast(
                adj_test.values[step : step + result.k_ar], steps=1
            )  # step + 1 := step + p
            all_forecasts.append(*forecast)

    return np.array(all_forecasts)

