from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin


class SumFeatures(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """
    A scikit-learn compatible transformer that sums multiple features into a new feature.

    Parameters
    ----------
    features : list[str]
        The list of feature names to be summed.
    new_feature_name : str
        The name of the new feature created by summing the input features.
    drop_original : bool, optional
        Whether to drop the original features after creating the new feature. Default is False.
    weights : list[float], optional
        The weights to be applied to each feature during the summation. If None, all features's weights are 1.
        If provided, the length of weights must be the same as the length of features.

    Examples
    --------

    >>> import pandas as pd
    >>> from pandas_processors.create import SumFeatures
    >>> X = pd.DataFrame({"col1": [1, 2, 3], "col2": [1, 1, 1], "col3": [2, 2, 2]})
    >>> sum_feature = SumFeatures(
    ...             features=["col1", "col2", "col3"],
    ...             new_feature_name="col4",
    ...             weights=[1, 2, 0.5],
    ...         )
    >>> sum_feature.fit_transform(X)
       col1  col2  col3  col4
    0     1     1     2   4.0
    1     2     1     2   5.0
    2     3     1     2   6.0
    """

    def __init__(
        self,
        features: list[str],
        new_feature_name: str,
        drop_original: bool = False,
        weights: Optional[list[float]] = None,
    ) -> None:
        self.features = features
        self.new_feature_name = new_feature_name
        self.drop_original = drop_original
        self.weights = weights

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : pd.DataFrame
            The training dataset.
        y : Optional[pd.Series], optional
            The label, by default None

        Returns
        -------
        self
            Returns the transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by summing the specified features into a new feature.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame containing the data to be processed.

        Returns
        -------
        pd.DataFrame
            The input DataFrame with a new feature created by summing the specified features.
        """
        X = X.copy()
        weights = self._get_weights()
        X[self.new_feature_name] = np.dot(X[self.features], weights)
        if self.drop_original:
            X.drop(columns=self.features, inplace=True)
        return X

    def _get_weights(self) -> list[float]:
        """
        Get the weights to be applied to each feature during the summation.

        Returns
        -------
        list[float]
            The weights to be applied to each feature.
        """
        if self.weights is None:
            weights_ = [1.0] * len(self.features)
        else:
            if len(self.weights) != len(self.features):
                raise ValueError(
                    "The length of weights must be the same as the length of features."
                )
            weights_ = self.weights
        return weights_


class ConditionalFeatures(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """
    A scikit-learn compatible transformer that creates new features based on a condition applied to existing features.

    Parameters
    ----------
    features : str | list[str]
        The name(s) of the feature(s) to apply the condition to.
    condition : callable
        A callable object or function that defines the condition to be applied to each value in the features.
        The condition should return True or False.
    true_value : int | float | str
        The value to assign to the new feature(s) when the condition is True.
    false_value : int | float | str
        The value to assign to the new feature(s) when the condition is False.
    new_feature_names : str | list[str]
        The name(s) of the new feature(s) created by applying the condition.
        If a single name is provided, it will be used for all new features.
        If a list of names is provided, it must have the same length as the features.
    drop_original : bool, optional
        Whether to drop the original features after creating the new features. Default is False.

    Examples
    --------

    >>> import pandas as pd
    >>> from pandas_processors.create import ConditionalFeatures
    >>> X = pd.DataFrame({"col1": [-1, 2, 3], "col2": [4, -5, 6]})
    >>> # Use ConditionalFeatures with one feature
    >>> conditional_feature = ConditionalFeatures(
    ...             features="col1",
    ...             new_feature_names="col3",
    ...             condition=lambda x: x > 0,
    ...             true_value=1,
    ...             false_value=0,
    ...         )
    >>> conditional_feature.fit_transform(X)
       col1  col2  col3
    0    -1     4     0
    1     2    -5     1
    2     3     6     1
    >>> # Use ConditionalFeatures with multiple features
    >>> conditional_feature = ConditionalFeatures(
    ...             features=["col1", "col2"],
    ...             new_feature_names=["col3", "col4"],
    ...             condition=lambda x: x > 0,
    ...             true_value=1,
    ...             false_value=0,
    ...         )
    >>> conditional_feature.fit_transform(X)
       col1  col2  col3  col4
    0    -1     4     0     1
    1     2    -5     1     0
    2     3     6     1     1
    """

    def __init__(
        self,
        features: str | list[str],
        condition: Callable,
        true_value: int | float | str,
        false_value: int | float | str,
        new_feature_names: str | list[str],
        drop_original: bool = False,
    ) -> None:
        if isinstance(features, str):
            features = [features]
        if isinstance(new_feature_names, str):
            new_feature_names = [new_feature_names]
        if len(features) != len(new_feature_names):
            raise ValueError(
                "The length of features and new_feature_names must be the same."
            )
        self.features = features
        self.condition = condition
        self.true_value = true_value
        self.false_value = false_value
        self.new_feature_names = new_feature_names
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : pd.DataFrame
            The training dataset.
        y : Optional[pd.Series], optional
            The label, by default None

        Returns
        -------
        self
            Returns the transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by applying the condition to create new features.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame containing the data to be processed.

        Returns
        -------
        pd.DataFrame
            The input DataFrame with new features created by applying the condition.
        """
        X = X.copy()

        X[self.new_feature_names] = X[self.features].map(
            lambda x: self.true_value if self.condition(x) else self.false_value
        )
        if self.drop_original:
            X.drop(columns=self.features, inplace=True)
        return X


if __name__ == "__main__":
    import doctest

    doctest.testmod()
