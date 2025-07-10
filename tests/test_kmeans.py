import numpy as np
from mlfs.kmeans import KMeans 

def test_centroids_shape():
    """
    Test whether the shape of the centroids matches the expected shape (n_clusters, n_features).
    """
    X = np.array([[1, 2], [3, 4], [5, 6]])
    model = KMeans(n_clusters=2, iterations=1)
    centroids, _ = model.fit(X)
    assert centroids.shape == (2, 2), f"Expected shape (2, 2), got {centroids.shape}"

def test_assignment_length_matches_input():
    """
    Test whether the number of cluster assignments equals the number of input samples.
    """
    X = np.random.rand(10, 2)
    model = KMeans(n_clusters=3, iterations=3)
    _, assignments = model.fit(X)
    assert len(assignments) == len(X), "Mismatch between number of points and assignments"

def test_predict_matches_fit():
    """
    Test whether predict returns the same assignments as during fit when applied on the same dataset.
    """
    X = np.random.rand(20, 2)
    model = KMeans(n_clusters=3, iterations=5)
    _, assignments_fit = model.fit(X)
    assignments_pred = model.predict(X)
    assert np.array_equal(assignments_fit, assignments_pred), "Predict output doesn't match fit output"

def test_inertia_is_positive():
    """
    Test whether computed inertia is positive after clustering.
    """
    X = np.random.rand(50, 3)
    model = KMeans(n_clusters=4, iterations=10)
    model.fit(X)
    assert isinstance(model.inertia_, float), "Inertia should be a float"
    assert model.inertia_ >= 0, "Inertia should be non-negative"

def test_more_clusters_than_samples():
    """
    Test that initializing more clusters than samples raises a proper error.
    """
    X = np.array([[1, 2], [3, 4]])
    model = KMeans(n_clusters=5, iterations=1)
    try:
        model.fit(X)
    except ValueError:
        assert True
    else:
        assert False, "Should raise ValueError if clusters > number of samples"
