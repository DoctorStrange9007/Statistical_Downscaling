
i


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


class LinearMDP:
    """"""

    def __init__(self):
        self.theta = 5
        self.kappa = 6
