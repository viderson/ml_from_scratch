import numpy as np
from mlfs.naive_bayes import NaiveBayes  

def test_basic_binary_classification():
    """
    Tests that the classifier can learn a simple binary dataset and make accurate predictions.
    """
    X = np.array([[1, 2], [2, 1], [10, 10], [11, 9]])
    y = np.array([0, 0, 1, 1])
    model = NaiveBayes()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, y), "Model should correctly classify training samples"


def test_class_priors_sum_to_one():
    """
    Verifies that the learned class priors sum to approximately 1.
    """
    X = np.random.randn(100, 3)
    y = np.random.choice([0, 1], size=100)
    model = NaiveBayes()
    model.fit(X, y)
    total = model.prior_0 + model.prior_1
    assert np.isclose(total, 1.0, atol=1e-6), f"Priors should sum to 1, got {total}"


def test_predict_returns_numpy_array():
    """
    Ensures that the predict method returns a NumPy array of correct shape.
    """
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model = NaiveBayes()
    model.fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray), "Predictions should be a numpy array"
    assert preds.shape == y.shape, "Shape of predictions should match shape of labels"


def test_predict_on_unseen_data():
    """
    Tests that model can make predictions on new data not seen during training.
    """
    X_train = np.array([[1, 2], [1, 3], [10, 10], [11, 9]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[2, 2], [10, 11]])
    model = NaiveBayes()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test), "Should return one prediction per test sample"
    assert all(p in [0, 1] for p in preds), "Predicted classes should be binary (0 or 1)"


def test_variance_never_zero():
    """
    Ensures that variance used in Gaussian density is never exactly zero (numerical stability).
    """
    X = np.array([[1, 1], [1, 1]])
    y = np.array([0, 0])
    model = NaiveBayes()
    model.fit(X, y)
    assert np.all(model.var_0 > 0), "Variance should be adjusted to avoid zero"
