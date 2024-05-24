from __future__ import annotations

from typing import Optional

import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax, skew
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin


class SkewedFeatureNormalizer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """
    A scikit-learn compatible transformer for normalizing skewed numerical features.

    Parameters
    ----------
    threshold : float, optional
        The threshold value for skewness to be considered skewed.
        Defaults to 0.5.

    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from pandas_processors.normalize import SkewedFeatureNormalizer
    >>> # Set the seed value
    >>> np.random.seed(123)
    >>> # Create a sample DataFrame with skewed features
    >>> data = {
    ...     'Feature1': np.random.exponential(scale=2, size=1000),
    ...     'Feature2': np.random.lognormal(mean=1, sigma=0.5, size=1000),
    ...     'Feature3': np.random.gamma(shape=2, scale=1, size=1000)
    ... }
    >>> df = pd.DataFrame(data)
    >>> normalizer = SkewedFeatureNormalizer(threshold=0.5)
    >>> normalizer.fit_transform(df)
         Feature1  Feature2  Feature3
    0    0.972661  1.114450  1.450302
    1    0.467319  0.901000  0.917198
    2    0.383601  1.358568  1.388989
    3    0.799976  0.774102  0.646975
    4    1.000851  0.975794  0.831716
    ..        ...       ...       ...
    995  0.437187  1.139626  0.711948
    996  0.965916  1.274143  0.574016
    997  0.548840  0.918681  0.447651
    998  0.008306  1.018361  0.762395
    999  0.479283  1.199449  1.158380
    <BLANKLINE>
    [1000 rows x 3 columns]
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @staticmethod
    def _normalize_a_feature(feature: pd.Series) -> pd.Series:
        """Apply Box-Cox transformation to a given numerical feature to normalize its distribution.

        Parameters
        ----------
        feature : pd.Series
            The numerical feature to be normalized.

        Returns
        -------
        pd.Series
            The normalized numerical feature after applying the Box-Cox transformation.
        """
        return boxcox1p(feature, boxcox_normmax(feature + 1))

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
        --------
        self
            Returns the transformer instance.
        """
        feature_skewness = X.apply(lambda x: skew(x)).sort_values(ascending=False)
        self.skewed_features = feature_skewness[feature_skewness > self.threshold].index
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize skewed numerical features in the DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame containing the numerical features to be normalized.

        Returns
        -------
        pd.DataFrame
            The input DataFrame with skewed numerical features normalized.
        """
        X = X.copy()
        X[self.skewed_features] = X[self.skewed_features].apply(
            self._normalize_a_feature
        )
        return X


if __name__ == "__main__":
    import doctest

    doctest.testmod()
