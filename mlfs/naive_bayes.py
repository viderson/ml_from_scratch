import numpy as np

class NaiveBayes:
    """
    Naive Bayes classifier implementation using Gaussian distribution assumption.
    """
    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier to the training data.

        Parameters:
        - X (numpy array): Training feature data of shape (n_samples, n_features).
        - y (numpy array): Target labels of shape (n_samples,).

        Returns:
        - None
        """
        eps = 1e-9
        classes, counts = np.unique(y, return_counts=True)
        if len(counts) < 2:
            raise ValueError("NaiveBayes requires at least two classes in the target variable.")
        
        # Class prior probabilities
        self.prior_0 = counts[0] / np.sum(counts)
        self.prior_1 = counts[1] / np.sum(counts)
        
        # Subsets of X for each class
        X_0 = X[y == 0]
        X_1 = X[y == 1]
        
        # Means and variances for each feature in each class
        self.mean_0 = np.mean(X_0, axis=0)
        self.mean_1 = np.mean(X_1, axis=0)
        self.var_0 = np.var(X_0, axis=0)
        self.var_1 = np.var(X_1, axis=0)
        
        # Avoid zero variance
        self.var_0 = np.where(self.var_0 < eps, eps, self.var_0)
        self.var_1 = np.where(self.var_1 < eps, eps, self.var_1)

    def gaussian_density(self, x, class_index):
        """
        Compute the Gaussian log-probability density for a given sample and class.

        Parameters:
        - x (numpy array): Feature vector of shape (n_features,).
        - class_index (int): 0 or 1 — the class index.

        Returns:
        - numpy array: Log probability densities for each feature.
        """
        if class_index == 0:
            part1 = -0.5 * np.log(2 * np.pi * self.var_0)
            part2 = ((x - self.mean_0) ** 2) / (2 * self.var_0)
            return part1 - part2
        else:
            part1 = -0.5 * np.log(2 * np.pi * self.var_1)
            part2 = ((x - self.mean_1) ** 2) / (2 * self.var_1)
            return part1 - part2

    def log_score(self, x):
        """
        Calculate log-probabilities for both classes and return the predicted class.

        Parameters:
        - x (numpy array): A single sample.

        Returns:
        - int: Predicted class label (0 or 1).
        """
        log_score_0 = np.log(self.prior_0) + np.sum(self.gaussian_density(x, 0))
        log_score_1 = np.log(self.prior_1) + np.sum(self.gaussian_density(x, 1))
        return 0 if log_score_0 > log_score_1 else 1

    def predict(self, X):
        """
        Predict class labels for a dataset.

        Parameters:
        - X (numpy array): Feature data of shape (n_samples, n_features).

        Returns:
        - numpy array: Predicted class labels of shape (n_samples,).
        """
        return np.array([self.log_score(record) for record in X])
