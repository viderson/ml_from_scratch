import numpy as np

def mse(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE).

    Args:
        y_true (numpy.ndarray): The true target values.
        y_pred (numpy.ndarray): The predicted target values.

    Returns:
        float: The Mean Squared Error.
    """
    assert len(y_true) == len(y_pred), "Shape mismatch between y_true and y_pred"
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE).

    Args:
        y_true (numpy.ndarray): The true target values.
        y_pred (numpy.ndarray): The predicted target values.

    Returns:
        float: The Root Mean Squared Error.
    """
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):
    """
    Calculate the R-squared (R²) coefficient of determination.

    Args:
        y_true (numpy.ndarray): The true target values.
        y_pred (numpy.ndarray): The predicted target values.

    Returns:
        float: The R-squared value.
    """
    assert len(y_true) == len(y_pred), "Shape mismatch between y_true and y_pred"
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot
