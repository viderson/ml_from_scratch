import numpy as np

class KMeans:
    def __init__(self, n_clusters, iterations=10):
        self.n_clusters = n_clusters
        self.iterations = iterations
        self.inertia_ = None  # Total within-cluster sum of squares

    def initialize_centroids(self, X):
        """
        Initializes the centroids for each cluster by selecting K random points from the dataset.

        Parameters:
        -----------
        X : numpy.ndarray
            Dataset to cluster.
        """
        if self.n_clusters > X.shape[0]:
            raise ValueError(f"Number of clusters ({self.n_clusters}) cannot be greater than number of samples ({X.shape[0]})")
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(X.shape[0])
        selected = indices[:self.n_clusters]
        self.centroids = X[selected]

    def euclidean_distance(self, x1, x2):
        """
        Calculates the Euclidean distance between two vectors.

        Parameters:
        -----------
        x1 : numpy.ndarray
        x2 : numpy.ndarray

        Returns:
        --------
        float
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def compute_distance_matrix(self, X):
        """
        Computes the matrix of distances between points in X and centroids.

        Parameters:
        -----------
        X : numpy.ndarray

        Returns:
        --------
        numpy.ndarray
            Distance matrix of shape (n_samples, n_centroids)
        """
        distance_matrix = []
        for record in X:
            distances = []
            for centroid in self.centroids:
                distances.append(self.euclidean_distance(record, centroid))
            distance_matrix.append(distances)
        return np.array(distance_matrix)

    def assign_clusters(self, X):
        """
        Assigns each point in the dataset to the nearest centroid.

        Parameters:
        -----------
        X : numpy.ndarray

        Returns:
        --------
        numpy.ndarray
            Array containing the index of the nearest centroid for each point.
        """
        distance_matrix = self.compute_distance_matrix(X)
        return np.argmin(distance_matrix, axis=1)

    def compute_new_centroids(self, X, assignments):
        """
        Computes the new centroids based on current cluster assignments.

        Parameters:
        -----------
        X : numpy.ndarray
        assignments : numpy.ndarray

        Returns:
        --------
        numpy.ndarray
            New centroids computed as the mean of assigned points.
        """
        new_centroids = []
        for i in range(self.n_clusters):
            new_centroids.append(np.mean(X[assignments == i], axis=0))
        return np.array(new_centroids)

    def compute_inertia(self, X, assignments):
        """
        Computes the total within-cluster sum of squares (inertia).

        Parameters:
        -----------
        X : numpy.ndarray
        assignments : numpy.ndarray

        Returns:
        --------
        float
            The inertia value.
        """
        inertia = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[assignments == i]
            centroid = self.centroids[i]
            inertia += np.sum((cluster_points - centroid) ** 2)
        return inertia

    def fit(self, X):
        """
        Applies K-Means clustering to the dataset.

        Parameters:
        -----------
        X : numpy.ndarray

        Returns:
        --------
        centroids : numpy.ndarray
            Final centroids after clustering.
        assignments : numpy.ndarray
            Cluster assignment for each point.
        """
        self.initialize_centroids(X)
        for _ in range(self.iterations):
            assignments = self.assign_clusters(X)
            self.centroids = self.compute_new_centroids(X, assignments)
        self.inertia_ = self.compute_inertia(X, assignments)
        return self.centroids, assignments

    def predict(self, X):
        """
        Assigns new data points to the nearest cluster.

        Parameters:
        -----------
        X : numpy.ndarray

        Returns:
        --------
        numpy.ndarray
            Cluster assignments for the new data points.
        """
        return self.assign_clusters(X)
