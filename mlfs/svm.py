import numpy as np

class SVM:
    """
    Support Vector Machine (SVM) classifier using stochastic gradient descent and hinge loss.
    """
    def __init__(self, iterations=1000, lr=0.01, lambdaa=0.01):
        """
        Initialize the SVM model.

        Parameters
        ----------
        iterations : int
            Number of training iterations.
        lr : float
            Learning rate for parameter updates.
        lambdaa : float
            Regularization strength.
        """
        self.iterations = iterations
        self.lr = lr
        self.lambdaa = lambdaa
        self.w = None  
        self.b = None  

    def initialize_parameters(self, X):
        """
        Initialize weights and bias to zeros.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).
        """
        _, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

    def update_parameters(self, dw, db):
        """
        Update weights and bias using gradients.

        Parameters
        ----------
        dw : numpy.ndarray
            Gradient w.r.t. weights.
        db : float
            Gradient w.r.t. bias.
        """
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def stochastic_gradient_descent(self, X, y):
        """
        Perform one epoch of stochastic gradient descent using hinge loss.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.
        y : numpy.ndarray
            Label vector (0 or 1).
        """
        y_mod = np.where(y <= 0, -1, 1)
        indices = np.random.permutation(len(y_mod))
        for i in indices:
            xi, target = X[i], y_mod[i]
            margin = target * (np.dot(xi, self.w) + self.b)
            if margin >= 1:
                dw = 2 * self.lambdaa * self.w
                db = 0.0
            else:
                dw = 2 * self.lambdaa * self.w - target * xi
                db = -target
            self.update_parameters(dw, db)

    def fit(self, X, y):
        """
        Train the SVM model.

        Parameters
        ----------
        X : numpy.ndarray
            Training feature matrix of shape (n_samples, n_features).
        y : numpy.ndarray
            Training labels vector of shape (n_samples,) with values {0,1}.

        Raises
        ------
        ValueError
            If the number of samples in X and y do not match.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be equal.")
        self.initialize_parameters(X)
        for _ in range(self.iterations):
            self.stochastic_gradient_descent(X, y)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : numpy.ndarray
            Input feature matrix of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predicted labels {0,1} of shape (n_samples,).
        """
        scores = np.dot(X, self.w) + self.b
        return np.where(scores >= 0, 1, 0)
