"""Credit to Eddie Bergman for these functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def toy_classification(
    *,
    has_cat_features: bool = True,
    cat_has_nan: bool = True,
    num_has_nan: bool = True,
    cols: int = 20,
    n_classes: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=500,
        n_features=cols,
        n_informative=10,
        n_classes=n_classes,
        random_state=seed,
    )
    _x = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(cols)])
    _y = pd.Series(y)

    if has_cat_features:
        cats_1 = ["a"] * 25 * 5 + ["b"] * 25 * 5 + ["c"] * 25 * 5 + ["d"] * 25 * 5
        cats_2 = ["x"] * 34 * 5 + ["y"] * 33 * 5 + ["z"] * 33 * 5
        if cat_has_nan:
            cats_1[0] = np.nan
            cats_1[49] = np.nan

        _x = _x.assign(cat_1=pd.Categorical(cats_1), cat_2=pd.Categorical(cats_2))

    if num_has_nan:
        _x.iloc[0, 2] = np.nan
        _x.iloc[0, 3] = np.nan

    return _x, _y


def toy_regression(
    *,
    has_cat_features: bool = False,
    cat_has_nan: bool = False,
    num_has_nan: bool = False,
    cols: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_regression(  # type: ignore
        n_samples=500,
        n_features=cols,
        n_informative=10,
        random_state=seed,
    )
    _x = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(cols)])
    _y = pd.Series(y)

    if has_cat_features:
        cats_1 = ["a"] * 25 * 5 + ["b"] * 25 * 5 + ["c"] * 25 * 5 + ["d"] * 25 * 5
        cats_2 = ["x"] * 34 * 5 + ["y"] * 33 * 5 + ["z"] * 33 * 5
        if cat_has_nan:
            cats_1[0] = np.nan
            cats_1[49] = np.nan
        _x = _x.assign(cat_1=pd.Categorical(cats_1), cat_2=pd.Categorical(cats_2))

    if num_has_nan:
        _x.iloc[0, 2] = np.nan
        _x.iloc[0, 3] = np.nan

    return _x, _y


def preprocess_dummy_data(
    *,
    X,
    y,
    seed,
    stratify,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list, list]:
    """Preprocess dummy data for classification/regression."""
    categorical_features = X.select_dtypes(
        include=["category", "object"],
    ).columns.tolist()
    categorical_features_index = [
        X.columns.get_loc(col) for col in categorical_features
    ]
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if stratify else None,
    )

    # -- Preprocess X_train/X_test
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[categorical_features] = oe.fit_transform(X_train[categorical_features])
    X_test[categorical_features] = oe.transform(X_test[categorical_features])
    X_train[categorical_features] = X_train[categorical_features].astype("category")
    X_test[categorical_features] = X_test[categorical_features].astype("category")
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    if stratify:
        le = LabelEncoder()
        y_train = pd.Series(le.fit_transform(y_train)).reset_index(drop=True)
        y_test = pd.Series(le.transform(y_test)).reset_index(drop=True)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        categorical_features,
        categorical_features_index,
    )
