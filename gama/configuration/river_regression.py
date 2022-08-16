import numpy as np

# regression models


from river.ensemble import AdaptiveRandomForestRegressor, BaggingRegressor, EWARegressor
from river.tree import HoeffdingAdaptiveTreeRegressor, HoeffdingTreeRegressor
from river.linear_model import LinearRegression
from river import linear_model, tree


# preprocessing
from river.preprocessing import (
    AdaptiveStandardScaler,
    Binarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler)

# feature extraction
from river.feature_extraction import PolynomialExtender

reg_config_online = {
    AdaptiveRandomForestRegressor: {
        "n_models": range(15, 40),
        "max_features": [0.2, 0.5, 0.7, 0.12, 1.0, "sqrt", "log2", None],
        "aggregation_method": ["mean", "median"],
        "lambda_value": range(2, 10),
        "grace_period": range(50, 350),
        "split_confidence": [1e-9, 1e-5, 1e-4, 1e-1],
        "tie_threshold": np.arange(0.02, 0.08, 0.01),
        "leaf_prediction": ["mean", "model", "adaptive"],
        "model_selector_decay": [0.2, 0.4, 0.7],
        # "max_size": [16],
        # "stop_mem_management": [True],
        # "remove_poor_attrs": [True],
        # "memory_estimate_period": [1000]
    },
    HoeffdingAdaptiveTreeRegressor: {
        "grace_period": range(50, 350),
        "split_confidence": [1e-9, 1e-7, 1e-4, 1e-2],
        "tie_threshold": np.arange(0.02, 0.08, 0.01),
        "leaf_prediction": ["mean", "model", "adaptive"],
        "model_selector_decay": [0.2, 0.4, 0.7],
        "min_samples_split": [3, 5, 7, 10],
        # "splitter": ["tree.splitter.EBSTSplitter", "t
        # ree.splitter.HistogramSplitter",
        #             "tree.splitter.TEBSTSplitter", "tree.splitter.GaussianSplitter"],
        "bootstrap_sampling": [True, False],
        "drift_window_threshold": range(100, 500, 100),
        "adwin_confidence": [2e-4, 2e-3, 2e-2],
        # "max_size": [16],
        # "memory_estimate_period":[1000],
        # "stop_mem_management": [True],
        # "remove_poor_attrs": [True]
    },
    BaggingRegressor: {
        "model": [linear_model.LinearRegression(), tree.HoeffdingTreeRegressor()],
        "n_models": range(1, 20),
    },
    # EWARegressor: {
    #     "models": [linear_model.LinearRegression(), tree.HoeffdingTreeRegressor()],
    #     "learning_rate": [0.1, 0.5, 0.01, 0.3]
    # },
    LinearRegression: {
        "l2": [0, 1e-5, 1e-7]
    },
    StandardScaler: {},
    AdaptiveStandardScaler: {"alpha": np.arange(0.3, 0.8, 0.1)},
    MaxAbsScaler: {},
    MinMaxScaler: {},
    Normalizer: {
        "order": [1, 2],
    },
    Binarizer: {"threshold": np.arange(0.0, 1.01, 0.05)},
    PolynomialExtender: {
        "degree": [2, 3, 4],
        "interaction_only": [True, False],
        "include_bias": [True, False]
    },
    RobustScaler: {}
}
