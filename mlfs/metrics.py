import numpy as np

# ==========================
#   Regression Metrics
# ==========================

def mse(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE).
    """
    assert len(y_true) == len(y_pred), "Shape mismatch between y_true and y_pred"
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE).
    """
    return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    """
    Calculate the R-squared (R²) coefficient of determination.
    """
    assert len(y_true) == len(y_pred), "Shape mismatch between y_true and y_pred"
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ==========================
#   Binary Classification Metrics
# ==========================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the accuracy of a classification model.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return np.mean(y_true == y_pred)

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the precision of a classification model.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp + 1e-8)

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the recall (sensitivity) of a classification model.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn + 1e-8)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the F1-score of a classification model.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p * r) / (p + r + 1e-8)


# ==========================
#   Multi-class Metrics
# ==========================

def balanced_accuracy(y_true, y_pred):
    """
    Calculate the balanced accuracy for a multi-class classification problem.

    Balanced accuracy is the average of sensitivity (recall) and specificity
    across all classes.
    """
    y_pred = np.array(y_pred)
    y_true = y_true.flatten()
    classes = np.unique(y_true)

    sensitivities = []
    specificities = []

    for cls in classes:
        TP = np.sum((y_true == cls) & (y_pred == cls))
        TN = np.sum((y_true != cls) & (y_pred != cls))
        FP = np.sum((y_true != cls) & (y_pred == cls))
        FN = np.sum((y_true == cls) & (y_pred != cls))

        sensitivity = TP / (TP + FN + 1e-8)
        specificity = TN / (TN + FP + 1e-8)

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    avg_sensitivity = np.mean(sensitivities)
    avg_specificity = np.mean(specificities)
    return (avg_sensitivity + avg_specificity) / 2
