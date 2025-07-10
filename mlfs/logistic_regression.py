import numpy as np
import plotly.express as px

class LogisticRegression:
    """
    Logistic Regression model.

    Parameters:
        learning_rate (float): Learning rate for the model.

    Methods:
        initialize_parameters(): Initializes the parameters of the model.
        sigmoid(z): Computes the sigmoid activation function for given input z.
        forward(X): Computes forward propagation for given input X.
        compute_cost(predictions): Computes the cost function for given predictions.
        compute_gradient(predictions): Computes the gradients for the model using given predictions.
        fit(X, y, iterations, plot_cost): Trains the model on given input X and labels y for specified iterations.
        predict(X): Predicts the labels for given input X.
    """

    def __init__(self, learning_rate=0.0001):
        np.random.seed(1)
        self.learning_rate = learning_rate

    def initialize_parameters(self):
        """
        Initializes the parameters of the model.
        """
        self.W = np.zeros(self.X.shape[1])
        self.b = 0.0

    def sigmoid(self, z):
        """
        Computes the sigmoid activation function.

        Parameters:
            z (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Sigmoid-transformed values.
        """
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """
        Computes forward propagation for given input X.

        Parameters:
            X (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output probabilities.
        """
        Z = np.dot(X, self.W) + self.b
        A = self.sigmoid(Z)
        return A

    def compute_cost(self, predictions):
        """
        Computes the cost function for given predictions.

        Parameters:
            predictions (numpy.ndarray): Predictions of the model.

        Returns:
            float: Cost of the model.
        """
        m = self.X.shape[0]
        epsilon = 1e-8
        cost = -1 / m * np.sum(
            (self.y * np.log(predictions + epsilon)) + 
            ((1 - self.y) * np.log(1 - predictions + epsilon))
        )
        return cost

    def compute_gradient(self, predictions):
        """
        Computes the gradients for the model using given predictions.

        Parameters:
            predictions (numpy.ndarray): Predictions of the model.
        """
        m = self.X.shape[0]
        self.gradient_w = 1 / m * np.dot(self.X.T, (predictions - self.y))  # returns np.array
        self.gradient_b = 1 / m * np.sum(predictions - self.y)             # returns float

    def fit(self, X, y, iterations, plot_cost=False, epsilon=1e-6):
        """
        Trains the model on given input X and labels y for specified iterations.

        Parameters:
            X (numpy.ndarray): Input features array of shape (n_samples, n_features)
            y (numpy.ndarray): Labels array of shape (n_samples,)
            iterations (int): Maximum number of iterations for training.
            plot_cost (bool): Whether to plot cost over iterations or not.
            epsilon (float): Threshold for early stopping based on gradient norm.
        """
        self.X = X
        self.y = y
        self.initialize_parameters()
        costs = []

        for i in range(iterations):
            prediction = self.forward(X)
            cost = self.compute_cost(prediction)
            costs.append(cost)

            self.compute_gradient(prediction)

            # Check gradient magnitude for early stopping
            grad_norm = np.linalg.norm(np.append(self.gradient_w, self.gradient_b))
            if grad_norm < epsilon:
                print(f"🔻 Early stopping at iteration {i}, gradient norm = {grad_norm:.2e}")
                break

            # Update parameters
            self.W -= self.learning_rate * self.gradient_w
            self.b -= self.learning_rate * self.gradient_b

            if i % 10000 == 0:
                print(f"Cost after iteration {i}: {cost}")

        if plot_cost:
            fig = px.line(y=costs, title="Cost vs Iteration", template="plotly_dark")
            fig.update_layout(
                title_font_color="#41BEE9",
                xaxis=dict(color="#41BEE9", title="Iterations"),
                yaxis=dict(color="#41BEE9", title="Cost")
            )
            fig.show()

    def predict(self, X):
        """
        Predicts the labels for given input X.

        Parameters:
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted labels (0 or 1).
        """
        return (self.forward(X) >= 0.5).astype(int)
