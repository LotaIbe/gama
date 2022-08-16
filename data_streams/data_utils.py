from numpy import int64
import pandas as pd
from typing import Union, Tuple, Optional
import arff


# Classification Datasets included in library
classification_datasets = ['data_streams/electricity-normalized.arff',
                           'data_streams/new_airlines.arff',
                           'data_streams/new_IMDB_drama.arff',        # - target at the beginning
                           'data_streams/new_vehicle_sensIT.arff',  # - target at the beginning
                           'data_streams/SEA_Abrubt_5.arff',
                           'data_streams/HYPERPLANE_01.arff',
                           'data_streams/SEA_Mixed_5.arff',
                           'data_streams/Forestcover.arff',      # - for later
                           'data_streams/new_ldpa.arff',      # - for later
                           'data_streams/new_pokerhand-normalized.arff',  # - for later
                           'data_streams/new_Run_or_walk_information.arff',  # - for later
                           'data_streams/SEA_Abrubt_15.arff',
                           'data_streams/SEA_Mixed_15.arff',
                           'data_streams/vehSensIT.arff'
                           ]

# Regression Datasets included in library
regression_datasets = ['data_streams/catalyst_activation.arff',
                       'data_streams/sulfur.arff',
                       'data_streams/debutaniser.arff',
                       'data_streams/bank32nh.arff',
                       'data_streams/2dplanes.arff',
                       'data_streams/friedman.arff'
                       ]


reg_num = {data: i for (i, data) in enumerate(regression_datasets)}
cls_num = {data: i for (i, data) in enumerate(classification_datasets)}


def load_format_data(
    data_loc: str = None,
    task: str = "classification",
    target_last: Optional[bool] = True,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]:
    """ Take (X,y) data and convert it to a (pd.DataFrame, pd.Series) tuple.

    Parameters
    ----------
    data_loc: str (default=None)
              Relative path to dataset

    task: str (default=classification)
          "regression" or "classification" task

    target_last: bool (default=True)
            If True, target is located in last column
            If False, target is located in first column

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame or pandas.Series]
        X and y, where X is formatted as pd.DataFrame and y is formatted as `y_type`.
    """

    is_classification = (task == "classification")
    data = pd.DataFrame(arff.load(open(data_loc, 'r'), encode_nominal=True)["data"])

    data = data[:].convert_dtypes()  # auto-infer and convert column data types

    for column in data:
        if data[column].dtype == "string":  # convert string dtypes to category dtypes ?
            data[column] = data[column].astype('category')

    if data[:].iloc[:, 0:-1].eq(0).any().any():
        print("Data contains zero values. They are not removed but might cause issues with some River learners.")

    if target_last:
        if is_classification:
            data.iloc[:, -1] = data.iloc[:, -1].astype(int64)
        X = data[:].iloc[:, 0:-1]
        y = data[:].iloc[:, -1]
    else:
        if is_classification:
            data.iloc[:, 1:] = data.iloc[:, 1:].astype(int64)
        X = data[:].iloc[:, 1:]
        y = data[:].iloc[:, 0]

    return X, y
