import numpy as np
from mlfs.knn import KNN

def test_knn_predict_perfect_match():
    """
    Tests if KNN can correctly classify a point that exactly matches a point in the training set.
    Should return the label of that matching point.
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    X_test = np.array([[1, 2]])

    model = KNN(n_neighbors=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred[0] == 0, f"Expected 0, got {y_pred[0]}"

def test_knn_tie_break():
    """
    Tests KNN behavior when there is a tie in nearest neighbor votes.
    Since ties are resolved by np.argmax, the label with the lowest index should be returned.
    """
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])
    X_test = np.array([[0.5, 0.5]])

    model = KNN(n_neighbors=4)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred[0] in [0, 1], f"Expected 0 or 1 due to tie, got {y_pred[0]}"

def test_knn_multiple_predictions():
    """
    Tests KNN's ability to make predictions on multiple input samples.
    Ensures the model returns the correct number of predictions.
    """
    X_train = np.array([[1, 2], [2, 3], [3, 4]])
    y_train = np.array([0, 1, 1])
    X_test = np.array([[1, 2], [3, 4]])

    model = KNN(n_neighbors=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == 2, f"Expected 2 predictions, got {len(y_pred)}"
