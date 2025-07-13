import numpy as np
import pandas as pd
import plotly.express as px
import pickle

class LinearRegression:
    """
    Linear Regression Model with Gradient Descent

    Linear regression is a supervised machine learning algorithm used for modeling the relationship
    between a dependent variable (target) and one or more independent variables (features) by fitting
    a linear equation to the observed data.

    This class implements a linear regression model using gradient descent optimization for training.
    It provides methods for model initialization, training, prediction, and model persistence.

    Parameters:
        learning_rate (float): The learning rate used in gradient descent.
        convergence_tol (float, optional): The tolerance for convergence (stopping criterion). Defaults to 1e-6.

    Attributes:
        weights (numpy.ndarray): Coefficients (weights) for the linear regression model.
        bias (float): Intercept (bias) for the linear regression model.

    Methods:
        _initialize_parameters(n_features): Initialize model parameters.
        _forward(X): Compute the forward pass of the linear regression model.
        _compute_cost(y_true, y_pred): Compute the mean squared error cost.
        _compute_gradients(X, y_true, y_pred): Compute gradients for model parameters.
        fit(X, y, iterations, plot_cost=True, verbose=True): Fit the linear regression model to training data.
        predict(X): Predict target values for new input data.
        save_model(filename='...'): Save the trained model to a file using pickle.
        load_model(filename): Load a trained model from a file using pickle.

    Examples:
        >>> from linear_regression import LinearRegression
        >>> model = LinearRegression(learning_rate=0.01)
        >>> model.fit(X_train, y_train, iterations=1000)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, learning_rate=0.01, convergence_tol=1e-6):
        self.learning_rate = learning_rate
        self.convergence_tol = convergence_tol
        self.weights = None
        self.bias = 0

    def _initialize_parameters(self, n_features):
        """Initialize model parameters with small random values."""
        self.weights = np.random.rand(n_features) * 0.01
        self.bias = 0

    def _forward(self, X):
        """Compute linear predictions."""
        return np.dot(X, self.weights) + self.bias

    def _compute_cost(self, y_true, y_pred):
        """Compute mean squared error cost."""
        n = len(y_true)
        return np.sum((y_pred - y_true) ** 2) / (2 * n)

    def _compute_gradients(self, X, y_true, y_pred):
        """Compute gradients of weights and bias."""
        n = len(y_true)
        error = y_pred - y_true
        dw = np.dot(X.T, error) / n
        db = np.sum(error) / n
        return dw, db

    def fit(self, X, y, iterations=1000, plot_cost=False, verbose=True):
        """
        Fit the linear regression model to the training data.

        Parameters:
            X (numpy.ndarray or pd.DataFrame/Series): Training input data.
            y (numpy.ndarray or pd.DataFrame/Series): Training labels.
            iterations (int): Number of iterations for gradient descent.
            plot_cost (bool): Whether to plot the cost during training.
            verbose (bool): Whether to print progress during training.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "X and y must be numpy arrays"
        assert X.shape[0] == y.shape[0], "Mismatched number of samples between X and y"

        self._initialize_parameters(X.shape[1])
        costs = []

        for i in range(iterations):
            y_pred = self._forward(X)
            cost = self._compute_cost(y, y_pred)
            dw, db = self._compute_gradients(X, y, y_pred)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            costs.append(cost)

            # print cost every 100 iterations or if convergence is reached
            if verbose and i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")

            if i > 0 and abs(costs[-1] - costs[-2]) < self.convergence_tol:
                if verbose:
                    print(f"Convergence reached at iteration {i}")
                break

        if plot_cost:
            fig = px.line(y=costs, title='Cost vs Iteration', template='plotly_dark')
            fig.update_layout(
                title_font_color="#41BEE9",
                xaxis=dict(color="#41BEE9", title="Iterations"),
                yaxis=dict(color="#41BEE9", title="Cost")
            )
            fig.show()

    def predict(self, X):
        """
        Predict target values for new input data.

        Parameters:
            X (numpy.ndarray or pd.DataFrame/Series): Input data.

        Returns:
            np.ndarray: Predicted values.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._forward(X)

    def save_model(self, filename='linear_regression_model.pkl'):
        """
        Save the trained model to a file using pickle.

        Parameters:
            filename (str): Path where the model will be saved.
        """
        with open(filename, 'wb') as f:
            pickle.dump({'weights': self.weights, 'bias': self.bias}, f)

    def load_model(self, filename):
        """
        Load a trained model from a file using pickle.

        Parameters:
            filename (str): Path of the file containing the saved model.
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)
            self.weights = model['weights']
            self.bias = model['bias']
