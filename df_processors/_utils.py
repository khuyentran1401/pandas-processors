import pandas as pd


def check_numerical_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """
    Check if the specified columns in a DataFrame are numerical.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    columns : list of str
        The list of column names to check.

    Returns
    -------
    list of str
        The list of column names that are numerical.

    Raises
    ------
    TypeError
        If any of the specified columns are not numerical.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": ["a", "b", "c"]})
    >>> check_numerical_columns(df, ["A", "B"])
    ['A', 'B']
    >>> check_numerical_columns(df, ["A", "C"])
    Traceback (most recent call last):
    ...
    TypeError: Some of the variables are not numerical. Please cast them as numerical before using this transformer.

    Notes
    -----
    If any non-numerical data types are found, a TypeError is raised. If all the specified columns are numerical, the
    function returns the original list of column names.
    """
    if len(df[columns].select_dtypes(exclude="number").columns) > 0:
        raise TypeError(
            "Some of the variables are not numerical. Please cast them as "
            "numerical before using this transformer."
        )

    return columns


def check_category_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """
    Check if the specified columns in a DataFrame are categorical.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    columns : list of str
        The list of column names to check.

    Returns
    -------
    list of str
        The list of column names that are categorical.

    Raises
    ------
    TypeError
        If any of the specified columns are not categorical.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"], "C": pd.Categorical(["x", "y", "z"])})
    >>> check_category_columns(df, ["B", "C"])
    ['B', 'C']
    >>> check_category_columns(df, ["A", "B"])
    Traceback (most recent call last):
    ...
    TypeError: Some of the variables are not categorical. Please cast them as object or categorical before using this transformer.

    Notes
    -----
    If any other data types are found, a TypeError is raised. If all the specified columns are categorical, the function returns the
    original list of column names.
    """
    if len(df[columns].select_dtypes(exclude=["O", "category"]).columns) > 0:
        raise TypeError(
            "Some of the variables are not categorical. Please cast them as "
            "object or categorical before using this transformer."
        )

    return columns


if __name__ == "__main__":
    import doctest

    doctest.testmod()
