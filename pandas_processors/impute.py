from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd

from pandas_processors._utils import check_category_columns, check_numerical_columns


class DataFrameImputer(ABC):
    """
    Abstract base class for imputing missing values in a pandas DataFrame.

    Methods
    -------
    impute(df)
        Impute missing values in a DataFrame.

    Notes
    -----
    This class defines the interface for imputing missing values in a
    DataFrame. Subclasses must implement the `impute` method to provide
    a specific imputation strategy.
    """

    @abstractmethod
    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to impute.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with missing values imputed.
        """
        pass


class MeanMedianImputer(DataFrameImputer):
    """
    Class for imputing missing values in numerical columns of a pandas DataFrame.

    Parameters
    ----------
    columns : list of str
        The list of column names to impute.
    imputation_method : Literal["mean", "median"], default "mean"
        The imputation method to use. Must be either "mean" or "median".

    Attributes
    ----------
    columns : list of str
        The list of column names to impute.
    imputation_method : Literal["mean", "median"]
        The imputation method to use.

    Examples
    --------
    >>> import pandas as pd
    >>> from pandas_processors.impute import MeanMedianImputer
    >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
    >>> imputer = MeanMedianImputer(columns=['A', 'B'], imputation_method='mean')
    >>> imputed_df = imputer.impute(df)
    >>> imputed_df
         A    B
    0  1.0  4.0
    1  2.0  5.0
    2  1.5  6.0
    """

    def __init__(
        self, columns: list[str], imputation_method: Literal["mean", "median"] = "mean"
    ) -> None:
        self.columns = columns
        self.imputation_method = imputation_method

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the specified numerical columns of a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to impute.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with missing values imputed.

        Notes
        -----
        If the `imputation_method` is not "mean" or "median", a ValueError
        is raised.

        Non-numerical columns are not imputed and are left unchanged.
        """
        columns_ = check_numerical_columns(df, self.columns)
        if self.imputation_method == "mean":
            impute_value = df[columns_].mean()
        elif self.imputation_method == "median":
            impute_value = df[columns_].median()
        else:
            raise ValueError("imputation_method must be 'mean' or 'median'")

        df[columns_] = df[columns_].fillna(impute_value)

        return df


class ConstantImputer(DataFrameImputer):
    """
    Class for imputing missing values in a pandas DataFrame with a constant value.

    Parameters
    ----------
    columns : list of str
        The list of column names to impute.
    value : float or str
        The constant value to use for imputation.

    Attributes
    ----------
    columns : list of str
        The list of column names to impute.
    value : float or str
        The constant value used for imputation.

    Examples
    --------
    >>> import pandas as pd
    >>> from pandas_processors.impute import ConstantImputer
    >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
    >>> imputer = ConstantImputer(columns=['A', 'B'], value=0)
    >>> imputed_df = imputer.impute(df)
    >>> imputed_df
         A    B
    0  1.0  4.0
    1  2.0  0.0
    2  0.0  6.0
    """

    def __init__(self, columns: list[str], value: float | str) -> None:
        self.columns = columns
        self.value = value

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the specified columns of a DataFrame with a constant value.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to impute.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with missing values imputed.
        """
        df[self.columns] = df[self.columns].fillna(self.value)

        return df


class CategoryImputer(DataFrameImputer):
    """
    Class for imputing missing values in categorical columns of a pandas DataFrame.

    Parameters
    ----------
    columns : list of str
        The list of column names to impute.
    imputation_method : Literal["missing", "mode"], default "missing"
        The imputation method to use. Must be either "missing" or "mode".
    fill_value : str, default "Missing"
        The value to use for imputation when `imputation_method` is "missing".

    Attributes
    ----------
    columns : list of str
        The list of column names to impute.
    imputation_method : Literal["missing", "mode"]
        The imputation method to use.
    fill_value : str
        The value used for imputation when `imputation_method` is "missing".

    Examples
    --------
    >>> import pandas as pd
    >>> from pandas_processors.impute import CategoryImputer
    >>> df = pd.DataFrame({'A': ['a', 'b', None], 'B': ['x', None, 'z']})
    >>> imputer = CategoryImputer(columns=['A', 'B'], imputation_method='missing', fill_value='Missing')
    >>> imputed_df = imputer.impute(df)
    >>> imputed_df
             A        B
    0        a        x
    1        b  Missing
    2  Missing        z
    """

    def __init__(
        self,
        columns: list[str],
        imputation_method: Literal["missing", "mode"] = "missing",
        fill_value: str = "Missing",
    ) -> None:
        self.columns = columns
        self.imputation_method = imputation_method
        self.fill_value = fill_value

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the specified categorical columns of a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to impute.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with missing values imputed.

        Notes
        -----
        If the `imputation_method` is not "missing" or "mode", a ValueError
        is raised.

        Non-categorical columns are not imputed and are left unchanged.
        """
        columns_ = check_category_columns(df, self.columns)
        if self.imputation_method == "missing":
            df[columns_] = df[columns_].fillna(self.fill_value)
        elif self.imputation_method == "mode":
            df[columns_] = df[columns_].fillna(df[columns_].mode().iloc[0])
        else:
            raise ValueError('imputation_method must be "missing" or "mode"')

        return df


if __name__ == "__main__":
    import doctest

    doctest.testmod()
