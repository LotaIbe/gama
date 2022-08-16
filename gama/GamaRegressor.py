import pandas as pd

from .gama import Gama
from gama.configuration.regression import reg_config
from river.base import Regressor
from gama.configuration.river_regression import reg_config_online


class GamaRegressor(Gama):
    """ Gama with adaptations for regression. """

    def __init__(self, config=None, scoring="neg_mean_squared_error", online_learning=False, *args, **kwargs) -> object:
        """ """
        # Empty docstring overwrites base __init__ doc string.
        # Prevents duplication of the __init__ doc string on the API page.
        self._online_learning = online_learning
        if not config:
            # set offline/online configuration
            if not self._online_learning:
                config = reg_config
            else:
                config = reg_config_online
        super().__init__(*args, **kwargs, config=config,
                         online_learning=online_learning, online_scoring="rmse", scoring=scoring)

    def _predict(self, x: pd.DataFrame):
        """ Predict the target for input X.

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe the same number of columns as that of X of `fit`.

        Returns
        -------
        numpy.ndarray
            Array with predictions of shape (N,) where N is len(X).
        """
        return self.model.predict(x)  # type: ignore
