import numpy as np
import pickle


class Node:
    """
    A class representing a node in a decision tree.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        """
        Initializes a new instance of the Node class.

        Parameters
        ----------
        feature : int or None
            Feature index used for splitting.
        threshold : float or None
            Threshold value used for splitting.
        left : Node or None
            Left child node.
        right : Node or None
            Right child node.
        gain : float or None
            Information gain from the split.
        value : object or None
            Value for leaf node prediction.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value


class DecisionTree:
    def __init__(self, min_samples=2, max_depth=2):
        """
        Constructor for DecisionTree class.

        Parameters
        ----------
        min_samples : int
            Minimum number of samples required to split an internal node.
        max_depth : int
            Maximum depth of the decision tree.
        """
        self.min_samples = min_samples
        self.max_depth = max_depth

    def split_data(self, dataset, feature, threshold):
        """
        Splits the dataset into two subsets based on a given feature and threshold.

        Parameters
        ----------
        dataset : np.ndarray
            The full dataset to be split (shape: [n_samples, n_features + 1]).
        feature : int
            Index of the feature used to split the data.
        threshold : float
            The value used to split the feature.

        Returns
        -------
        left_dataset : np.ndarray
            Subset where feature value <= threshold.
        right_dataset : np.ndarray
            Subset where feature value > threshold.
        """
        left_dataset = dataset[dataset[:, feature] <= threshold]
        right_dataset = dataset[dataset[:, feature] > threshold]
        return left_dataset, right_dataset

    def entropy(self, y):
        """
        Calculates the entropy of a distribution of class labels.

        Parameters
        ----------
        y : np.ndarray
            Array of class labels.

        Returns
        -------
        float
            Entropy of the distribution.
        """
        values, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def information_gain(self, parent, left, right):
        """
        Computes the information gain from splitting the parent node into left and right.

        Parameters
        ----------
        parent : np.ndarray
            Labels of the parent node.
        left : np.ndarray
            Labels of the left split.
        right : np.ndarray
            Labels of the right split.

        Returns
        -------
        float
            Information gain resulting from the split.
        """
        weight_l = len(left) / len(parent)
        weight_r = len(right) / len(parent)
        return self.entropy(parent) - (weight_l * self.entropy(left) + weight_r * self.entropy(right))

    def best_split(self, dataset, num_samples, num_features):
        """
        Finds the best feature and threshold to split the data for maximum information gain.

        Parameters
        ----------
        dataset : np.ndarray
            Full dataset including target labels (last column).
        num_samples : int
            Number of samples in the dataset.
        num_features : int
            Number of features in the dataset.

        Returns
        -------
        dict
            Dictionary with best feature index, threshold, gain,
            and resulting left/right datasets.
        """
        best = {'gain': -1, 'feature': None, 'threshold': None}
        for feature in range(num_features):
            thresholds = np.unique(dataset[:, feature])
            for threshold in thresholds:
                left, right = self.split_data(dataset, feature, threshold)
                if len(left) > 0 and len(right) > 0:
                    y_parent = dataset[:, -1]
                    y_left = left[:, -1]
                    y_right = right[:, -1]
                    gain = self.information_gain(y_parent, y_left, y_right)
                    if gain > best['gain']:
                        best.update({
                            'gain': gain,
                            'feature': feature,
                            'threshold': threshold,
                            'left_dataset': left,
                            'right_dataset': right
                        })
        return best

    def calculate_leaf_value(self, y):
        """
        Calculates the most frequent class in a label array.

        Parameters
        ----------
        y : np.ndarray
            Array of target class labels.

        Returns
        -------
        int
            Most frequent class label.
        """
        y = y.astype(int)
        return np.bincount(y).argmax()

    def build_tree(self, dataset, current_depth=0):
        """
        Recursively builds the decision tree using information gain.

        Parameters
        ----------
        dataset : np.ndarray
            Full dataset including labels (last column).
        current_depth : int
            Current depth in the tree (used to stop recursion).

        Returns
        -------
        Node
            The root node of the subtree (or leaf node).
        """
        X = dataset[:, :-1]
        y = dataset[:, -1]
        num_samples, num_features = X.shape

        if num_samples >= self.min_samples and current_depth <= self.max_depth:
            best = self.best_split(dataset, num_samples, num_features)
            if best['gain'] > 0:
                left_subtree = self.build_tree(best['left_dataset'], current_depth + 1)
                right_subtree = self.build_tree(best['right_dataset'], current_depth + 1)
                return Node(best['feature'], best['threshold'], left_subtree, right_subtree, best['gain'])

        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)

    def fit(self, X, y):
        """
        Fits the decision tree model to the given data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Label vector of shape (n_samples,).
        """
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        """
        Predicts class labels for the input samples.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Predicted class labels for each input sample.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverses the decision tree to make a prediction for a single input sample.

        Parameters
        ----------
        x : np.ndarray
            A single input feature vector.
        node : Node
            The current node in the tree.

        Returns
        -------
        int
            Predicted class label.
        """
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    def save_model(self, path):
        """
        Saves the decision tree model to a file.

        Parameters
        ----------
        path : str
            File path to save the model.
        """
        model_data = {
            'root': self.root,
            'min_samples': self.min_samples,
            'max_depth': self.max_depth
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to {path}")

    def load_model(self, path):
        """
        Loads the decision tree model from a file.

        Parameters
        ----------
        path : str
            File path to load the model from.
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.root = model_data['root']
        self.min_samples = model_data['min_samples']
        self.max_depth = model_data['max_depth']
        print(f"✅ Model loaded from {path}")

def compute_tree_depth(node):
    if node is None or node.value is not None:
        return 0
    return 1 + max(compute_tree_depth(node.left), compute_tree_depth(node.right))

def count_leaves(node):
    if node is None:
        return 0
    if node.value is not None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)

def count_nodes(node):
    if node is None:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)