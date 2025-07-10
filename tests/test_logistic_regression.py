import numpy as np
from mlfs.logistic_regression import LogisticRegression

def test_fit_predict_perfect_separation():
    """
    Tests if the model can perfectly separate two linearly separable classes.
    """
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [8, 9],
        [9, 10],
        [10, 11],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(Learning_rate = 0.1)
    model.fit(X,y, iterations = 10000, plot_cost = False)
    predictions = model.predict(X)

    assert np.array_equal(pred,y), f"Expected {y}, got {pred}"
def test_predict_new_data():
    """
    Tests if the model can generalize and predict correct class on unseen linearly separable data.
    """
    X_train = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [10, 10],
        [11, 11],
        [12, 12],
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression(learning_rate=0.1)
    model.fit(X_train, y_train, iterations=10000, plot_cost=False)

    X_test = np.array([[0, 0], [13, 13]])
    preds = model.predict(X_test)
    expected = np.array([0, 1])

    assert np.array_equal(preds, expected), f"Expected {expected}, got {preds}"
