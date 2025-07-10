import numpy as np
import pandas as pd

def standardize(X, return_params=False):
    """
    Standardizes the input data to have zero mean and unit variance.

    This function transforms the features in `X` by subtracting the mean and dividing
    by the standard deviation for each feature (column). If a feature has zero standard
    deviation, its values are left unchanged (to prevent division by zero).

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame or pandas.Series
        Input data to be standardized. Each column is treated as a separate feature.
    
    return_params : bool, optional (default=False)
        If True, the function returns a tuple containing the standardized data,
        the mean vector, and the standard deviation vector.

    Returns
    -------
    X_scaled : numpy.ndarray
        Standardized version of the input data.
    
    mean : numpy.ndarray, optional
        Mean of each feature (returned only if `return_params=True`).

    std : numpy.ndarray, optional
        Standard deviation of each feature (returned only if `return_params=True`).
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.values

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1

    X_scaled = (X - mean) / std
    if return_params:
        return X_scaled, mean, std
    return X_scaled

def unstandardize(X_scaled, mean, std):
    """
    Reverts standardized data back to its original scale.

    This function restores the original values of data that was previously standardized
    by applying the inverse transformation: X_original = X_scaled * std + mean.

    Parameters
    ----------
    X_scaled : numpy.ndarray
        The standardized data.

    mean : numpy.ndarray or float
        The mean(s) used during standardization.

    std : numpy.ndarray or float
        The standard deviation(s) used during standardization.

    Returns
    -------
    X_original : numpy.ndarray
        The data transformed back to its original scale.
    """
    return X_scaled * std + mean
def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Splits the data into training and testing sets.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Feature matrix.
    
    y : numpy.ndarray or pandas.Series
        Target vector.
    
    test_size : float
        Proportion of the dataset to include in the test split.
    
    shuffle : bool
        Whether to shuffle the data before splitting.
    
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy.ndarray
        Split datasets.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.values
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.values

    assert len(X) == len(y), "Mismatched X and y lengths"

    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(indices)

    test_size = int(n_samples * test_size)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def min_max_scale(X, return_params=False):
    """
    Scales features to the [0, 1] range.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame/Series
        Input data.

    return_params : bool
        If True, also returns min and max values used in scaling.

    Returns
    -------
    X_scaled : numpy.ndarray
        Scaled data.

    X_min : numpy.ndarray (optional)
    X_max : numpy.ndarray (optional)
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.values

    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    range_ = X_max - X_min
    range_[range_ == 0] = 1  

    X_scaled = (X - X_min) / range_
    if return_params:
        return X_scaled, X_min, X_max
    return X_scaled
