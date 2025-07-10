import numpy as np
from mlfs.linear_regression import LinearRegression

def test_fit_predict_simple_line():
    """
    Tests whether the model can fit a perfect linear relationship y = 2x + 3.
    The predicted values should be close to the expected values within a tolerance.
    """
    X = np.array([1, 2, 3, 4, 5])
    y = 2 * X + 3

    model = LinearRegression(learning_rate=0.01, convergence_tol=1e-8)
    model.fit(X, y, iterations=1000, plot_cost=False, verbose=False)
    predictions = model.predict(X)

    assert np.allclose(predictions, y, atol=1.0), f"Expected {y}, got {predictions}"

def test_predict_single_value():
    """
    Tests if the model can predict a single unseen value after training on a simple linear function.
    """
    X = np.array([1, 2, 3])
    y = 2 * X + 1
    model = LinearRegression(learning_rate=0.01)
    model.fit(X, y, iterations=500, plot_cost=False, verbose=False)

    pred = model.predict(np.array([10]))
    expected = 2 * 10 + 1
    assert abs(pred - expected) < 2.0, f"Prediction {pred} not close to expected {expected}"
