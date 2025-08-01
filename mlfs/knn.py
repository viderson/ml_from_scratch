import numpy as np
import pickle

class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def euclidean_distance(self, x1, x2):
        """
        Calculate the Euclidean distance between two data points.

        Parameters:
        -----------
        x1 : numpy.ndarray, shape (n_features,)
            A data point in the dataset.

        x2 : numpy.ndarray, shape (n_features,)
            A data point in the dataset.

        Returns:
        --------
        distance : float
            The Euclidean distance between x1 and x2.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X_train, y_train):
        """
        Stores the values of X_train and y_train.

        Parameters:
        -----------
        X_train : numpy.ndarray, shape (n_samples, n_features)
            The training dataset.

        y_train : numpy.ndarray, shape (n_samples,)
            The target labels.
        """
        self.X_train = X_train
        self.y_train = y_train

    def _predict(self, x):
        """
        Predicts the class label for a single example.

        Parameters:
        -----------
        x : numpy.ndarray, shape (n_features,)
            A data point in the test dataset.

        Returns:
        --------
        The predicted class label for x.
        """
        distances = []
        for record in self.X_train:
            distance = self.euclidean_distance(x, record)
            distances.append(distance)
        distances = np.array(distances)

        idx = np.argsort(distances)
        nearest = idx[:self.n_neighbors]

        labels = self.y_train[nearest]
        unique_labels, counts = np.unique(labels, return_counts=True)
        final_label = unique_labels[counts.argmax()]
        return final_label

    def predict(self, X):
        """
        Predicts the class labels for each example in X.

        Parameters:
        -----------
        X : numpy.ndarray, shape (n_samples, n_features)
            The test dataset.

        Returns:
        --------
        predictions : numpy.ndarray, shape (n_samples,)
            The predicted class labels for each example in X.
        """
        predictions = []
        for record in X:
            predict_label = self._predict(record)
            predictions.append(predict_label)
        predictions = np.array(predictions)
        return predictions

    def save_model(self, path):
        """
        Saves the model parameters to a file.

        Parameters:
        -----------
        path : str
            The file path to save the model.
        """
        model_data = {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'n_neighbors': self.n_neighbors
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to {path}")

    def load_model(self, path):
        """
        Loads the model parameters from a file.

        Parameters:
        -----------
        path : str
            The file path to load the model.
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.X_train = model_data['X_train']
        self.y_train = model_data['y_train']
        self.n_neighbors = model_data['n_neighbors']
        print(f"✅ Model loaded from {path}")
