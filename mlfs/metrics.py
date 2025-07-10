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

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the accuracy of a classification model.

    Parameters
    ----------
    y_true : numpy.ndarray
        True labels for each data point.
    y_pred : numpy.ndarray
        Predicted labels for each data point.

    Returns
    -------
    float
        Accuracy of the model, i.e., the proportion of correct predictions.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return np.mean(y_true == y_pred)

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the precision of a classification model.

    Precision measures the proportion of true positive predictions among all
    instances predicted as positive by the model.

    Parameters
    ----------
    y_true : numpy.ndarray
        True labels for each data point.
    y_pred : numpy.ndarray
        Predicted labels for each data point.

    Returns
    -------
    float
        Precision score.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp + 1e-8)

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the recall (sensitivity) of a classification model.

    Recall measures the proportion of actual positive instances that were correctly
    identified by the model.

    Parameters
    ----------
    y_true : numpy.ndarray
        True labels for each data point.
    y_pred : numpy.ndarray
        Predicted labels for each data point.

    Returns
    -------
    float
        Recall score.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn + 1e-8)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the F1-score of a classification model.

    The F1-score is the harmonic mean of precision and recall. It provides
    a balance between the two metrics and is especially useful when the
    class distribution is imbalanced.

    Parameters
    ----------
    y_true : numpy.ndarray
        True labels for each data point.
    y_pred : numpy.ndarray
        Predicted labels for each data point.

    Returns
    -------
    float
        F1-score.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p * r) / (p + r + 1e-8)