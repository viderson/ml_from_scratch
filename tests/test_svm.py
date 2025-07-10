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
        self.w = None  # weights
        self.b = None  # bias

    def initialize_parameters(self, X):
        """
        Initialize weights and bias with zeros.

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

        def stochastic_gradient_descent(self, X, y):
        """
        Perform one pass of stochastic gradient descent with hinge loss.

        Parameters:
        -----------
        X : numpy array
            Input features.
        y : numpy array
            True labels.
        """
        # Transform labels to -1 and 1 for hinge loss
        y_transformed = np.where(y <= 0, -1, 1)
        for i, x in enumerate(X):
            margin = y_transformed[i] * (np.dot(x, self.w) - self.b)
            if margin >= 1:
                # Only regularization gradient
                dw = 2 * self.lambdaa * self.w
                db = 0
            else:
                # Hinge loss gradient: update weights and bias
                dw = 2 * self.lambdaa * self.w - y_transformed[i] * x
                db = y_transformed[i]  # flipped sign to correct bias update
            self.update_parameters(dw, db)

    def fit(self, X, y):(self, X, y):
        """
        Train the SVM model.

        Parameters:
        -----------
        X : numpy array
            Training feature matrix.
        y : numpy array
            Training labels.

        Raises:
        ------
        ValueError:
            If X and y have mismatched lengths.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatched input dimensions between X and y.")

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
            Predicted labels (0 or 1).
        """
        # Decision rule: if score >= 0 predict 1, else 0
        scores = np.dot(X, self.w) - self.b
        return np.where(scores >= 0, 1, 0)
