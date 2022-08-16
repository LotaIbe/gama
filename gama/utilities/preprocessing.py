import logging
import numbers
from typing import Optional, Iterator, List, Tuple

from sklearn import preprocessing
import category_encoders as ce
import pandas as pd
from sklearn.base import TransformerMixin
from river.preprocessing import PreviousImputer, OneHotEncoder, StatImputer
from river.compose import Select, SelectType
from river import stats
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


log = logging.getLogger(__name__)


def select_categorical_columns(
    df: pd.DataFrame,
    min_f: Optional[int] = None,
    max_f: Optional[int] = None,
    ignore_nan: bool = True,
) -> Iterator[str]:
    """ Find all categorical columns with at least `min_f` and at most `max_f` factors.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas DataFrame to design the encoder for.
    min_f: int, optional (default=None)
        The inclusive minimum number of unique values the column should have.
    max_f: int, optional (default=None)
        The inclusive maximum number of unique values the column should have.
    ignore_nan: bool (default=True)
        If True, don't count NaN as a unique value.
        If False, count NaN as a unique value (only once).

    Returns
    -------
    An iterator which iterates over the column names that satisfy the criteria.
    """
    for column in df:
        if isinstance(df[column].dtype, pd.core.dtypes.dtypes.CategoricalDtype):
            nfactors = df[column].nunique(dropna=ignore_nan)
            if (min_f is None or min_f <= nfactors) and (
                max_f is None or nfactors <= max_f
            ):
                yield column


def basic_encoding(x: pd.DataFrame, is_classification: bool):
    """ Perform 'basic' encoding of categorical features.

     Specifically, perform:
      - Ordinal encoding for features with 2 or fewer unique values.
      - One hot encoding for features with at most 10 unique values.
      - Ordinal encoding for features with 11+ unique values, if y is categorical.
     """
    ord_features = list(select_categorical_columns(x, max_f=2))
    if is_classification:
        ord_features.extend(select_categorical_columns(x, min_f=11))
    leq_10_features = list(select_categorical_columns(x, min_f=3, max_f=10))

    encoding_steps = [
        ("ord-enc", ce.OrdinalEncoder(cols=ord_features, drop_invariant=True)),
        ("oh-enc", ce.OneHotEncoder(cols=leq_10_features, handle_missing="ignore")),
    ]
    encoding_pipeline = Pipeline(encoding_steps)
    x_enc = encoding_pipeline.fit_transform(x, y=None)  # Is this dangerous?
    return x_enc, encoding_pipeline


def basic_pipeline_extension(
    x: pd.DataFrame, is_classification: bool
) -> List[Tuple[str, TransformerMixin]]:
    """ Define a TargetEncoder and SimpleImputer.

    TargetEncoding is will encode categorical features with more than 10 unique values,
    if y is not categorical. SimpleImputer imputes with the median.
    """
    # These steps need to be in the pipeline because they need to be trained each fold.
    extension_steps = []
    if not is_classification:
        # TargetEncoder is broken with categorical target
        many_factor_features = list(select_categorical_columns(x, min_f=11))
        extension_steps.append(
            ("target_enc", ce.TargetEncoder(cols=many_factor_features))
        )
    extension_steps.append(("imputation", SimpleImputer(strategy="median")))

    return extension_steps


def river_pipeline_extension(
    x: pd.DataFrame, is_classification: bool
) -> List[Tuple[str, TransformerMixin]]:
    """ Define a OneHotEncoder and Previous Imputer.

    OneHotEncoder will encode categorical features. SimpleImputer imputes with the previously seen value.
    """
    # These steps need to be in the pipeline because they need to be trained each fold.
    extension_steps = []

    categorical_cols = list(select_categorical_columns(x))
    non_categorical = [column for column in x.columns if column not in categorical_cols]

    # if categorical_cols:
    #     for (i, col) in enumerate(categorical_cols):
    #         if i == 0:
    #             preprocessing_pipeline = Select(
    #                 col) | OneHotEncoder()
    #         else:
    #             preprocessing_pipeline += Select(
    #                 col) | OneHotEncoder()

    #     for col in non_categorical:
    #         preprocessing_pipeline += Select(col)

    # num_col_selector = SelectType(numbers.Number)
    # num_col_selector.learn_one(x.iloc[0].to_dict())
    # num_cols = num_col_selector.transform_one(x.iloc[0].to_dict())

    # num_p = [(col, stats.Mean()) for col in num_cols]
    # categorical_preprocessing = SelectType(str) | PreviousImputer()
    # numeric_preprocessing = StatImputer(*(num_p))
    # # extension_steps.append(preprocessing_pipeline)

    # river_preprocessing = numeric_preprocessing + categorical_preprocessing
    # extension_steps.append(river_preprocessing)

    extension_steps.append(PreviousImputer())

    return extension_steps
