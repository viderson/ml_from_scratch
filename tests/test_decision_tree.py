import numpy as np
import pytest
from mlfs.decision_tree import DecisionTree  


def test_single_split():
    """
    Tests the correctness of the tree when only a single split is possible.
    Verifies that the predictions are only 0 or 1 and have the correct length.
    """
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTree(min_samples=1, max_depth=1)
    tree.fit(X, y)
    preds = tree.predict(X)
    assert all(p in [0, 1] for p in preds), "Predictions should be 0 or 1"
    assert len(preds) == len(y), "Prediction length mismatch"


def test_pure_leaf():
    """
    Tests the case where all samples belong to a single class.
    The tree should create a leaf node and return that same class for all inputs.
    """
    X = np.array([[1], [1], [1]])
    y = np.array([1, 1, 1])
    tree = DecisionTree(min_samples=1, max_depth=1)
    tree.fit(X, y)
    preds = tree.predict(X)
    assert all(p == 1 for p in preds), "All predictions should be class 1"


def test_impure_leaf_due_to_depth():
    """
    Tests tree behavior when max_depth = 0.
    Even if data could be split, the tree should return the dominant class in the dataset.
    """
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTree(min_samples=1, max_depth=0)
    tree.fit(X, y)
    preds = tree.predict(X)
    assert all(p in [0, 1] for p in preds), "Predictions should still be valid labels"


def test_entropy_reduction():
    """
    Tests the correctness of information gain calculation.
    For a perfect split (pure classes on both sides), gain should be close to 1.0.
    """
    tree = DecisionTree()
    parent = np.array([0, 0, 1, 1])
    left = np.array([0, 0])
    right = np.array([1, 1])
    gain = tree.information_gain(parent, left, right)
    assert np.isclose(gain, 1.0, atol=1e-2), f"Gain should be near 1.0 but got {gain}"


def test_fit_and_predict_consistency():
    """
    Verifies that prediction after fitting works properly:
    - The output is a NumPy array.
    - It has the same shape as the target array y.
    """
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTree(min_samples=1, max_depth=2)
    tree.fit(X, y)
    preds = tree.predict(X)
    assert isinstance(preds, np.ndarray), "Predictions should be numpy array"
    assert preds.shape == y.shape, "Shape mismatch between predictions and labels"


def test_incorrect_shapes():
    """
    Tests whether the fit method raises a ValueError when the shapes of X and y don't match.
    """
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1])
    tree = DecisionTree()
    with pytest.raises(ValueError):
        tree.fit(X, y)
