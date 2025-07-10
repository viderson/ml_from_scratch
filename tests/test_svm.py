import numpy as np
import pytest
from mlfs.svm import SVM  

def test_predict_linearly_separable_data():
    """
    Test that the SVM correctly classifies a simple linearly separable dataset.
    """
    X = np.array([[2, 3], [1, 1], [2, 0], [0, 0]])
    y = np.array([1, 1, 0, 0])
    model = SVM(iterations=3000, lr=0.1, lambdaa=0.001)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, got {predictions}"

def test_predict_all_same_class():
    """
    Test that the SVM can fit when all targets are the same class.
    Should still return the same class for all predictions.
    """
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([1, 1, 1])
    model = SVM(iterations=100, lr=0.01, lambdaa=0.01)
    model.fit(X, y)
    predictions = model.predict(X)
    assert all(pred == 1 for pred in predictions)

def test_output_shape():
    """
    Check that the shape of the prediction output matches input.
    """
    X = np.random.rand(10, 2)
    y = np.array([0, 1] * 5)
    model = SVM(iterations=10, lr=0.01, lambdaa=0.01)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (10,)

def test_predict_values_range():
    """
    Ensure predict returns only binary class values (0 or 1).
    """
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)
    model = SVM(iterations=20, lr=0.01, lambdaa=0.01)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})

def test_invalid_input_shapes():
    """
    Check that passing mismatched input shapes raises a ValueError.
    """
    X = np.random.rand(5, 2)
    y = np.array([1, 0, 1])  # wrong shape
    model = SVM()
    with pytest.raises(ValueError):
        model.fit(X, y)
