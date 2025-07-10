import numpy as np

class SVM:
    """
    Support Vector Machine (SVM) classifier using stochastic gradient descent.
    """
    def __init__(self, iterations=1000, lr=0.01, lambdaa=0.01):
        """
        Initialize the SVM model.

        Parameters:
        -----------
        iterations : int
            Number of training iterations.
        lr : float
            Learning rate for parameter updates.
        lambdaa : float
            Regularization strength.
        """
        self.iterations = iterations
        self.lambdaa = lambdaa
        self.lr = lr
        self.w = None
        self.b = None

    def initialize_parameters(self, X):
        """
        Initialize weights and bias.

        Parameters:
        -----------
        X : numpy array
            Input feature matrix.
        """
        _, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

    def update_parameters(self, dw, db):
        """
        Apply gradient updates to weights and bias.

        Parameters:
        -----------
        dw : numpy array
            Gradient of the weights.
        db : float
            Gradient of the bias.
        """
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def compute_margin(self, X, y):
        """
        Compute margin values for samples.

        Parameters:
        -----------
        X : numpy array
            Input features.
        y : numpy array
            True labels.

        Returns:
        --------
        numpy array of margins.
        """
        return y * (np.dot(X, self.w) - self.b) >= 1

    def stochastic_gradient_descent(self, X, y):
        """
        Perform one pass of stochastic gradient descent.

        Parameters:
        -----------
        X : numpy array
            Input features.
        y : numpy array
            True labels.
        """
        y_transformed = np.where(y <= 0, -1, 1)
        for i, x in enumerate(X):
            condition = y_transformed[i] * (np.dot(x, self.w) - self.b) >= 1
            if condition:
                dw = 2 * self.lambdaa * self.w
                db = 0
            else:
                dw = 2 * self.lambdaa * self.w - x * y_transformed[i]
                db = -y_transformed[i]
            self.update_parameters(dw, db)

    def fit(self, X, y):
        """
        Train the SVM model.

        Parameters:
        -----------
        X : numpy array
            Training feature matrix.
        y : numpy array
            Training labels.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")
        self.initialize_parameters(X)
        for _ in range(self.iterations):
            self.stochastic_gradient_descent(X, y)

    def predict(self, X):
        """
        Predict labels using the trained SVM model.

        Parameters:
        -----------
        X : numpy array
            Feature matrix to predict.

        Returns:
        --------
        numpy array
            Predicted labels.
        """
        scores = np.dot(X, self.w) - self.b
        raw_preds = np.sign(scores)
        return np.where(raw_preds <= -1, 0, 1)
