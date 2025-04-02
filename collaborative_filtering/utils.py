import numpy as np

def cosine_similarity(X: np.ndarray, Y: np.ndarray) -> float:
  """Standard cosine similarity function."""
  return (X.T @ Y) / ((X.T @ X)**0.5 * (Y.T @ Y)**0.5)

def euclidean_distance(X: np.ndarray, Y: np.ndarray) -> float:
  """Standard euclidean distance function."""
  return (np.sum((X - Y) ** 2)) ** 0.5

def weighted_cosine_similarity(X: np.ndarray, Y: np.ndarray, weights: np.ndarray) -> float:
  """Weighted cosine similarity function."""
  return (X.T @ Y) / ((X.T @ X)**0.5 * (Y.T @ Y)**0.5) * weights

def weighted_cosine_similarity(X: np.ndarray, Y: np.ndarray, weights: np.ndarray) -> float:
    """Weighted cosine similarity function."""
    weighted_X = X * np.sqrt(weights)
    weighted_Y = Y * np.sqrt(weights)
    
    weighted_dot_product = np.sum(weighted_X * weighted_Y)
    weighted_norm_X = np.sqrt(np.sum(weighted_X ** 2))
    weighted_norm_Y = np.sqrt(np.sum(weighted_Y ** 2))
    
    return weighted_dot_product / (weighted_norm_X * weighted_norm_Y)

if __name__ == "__main__":
    X = np.array([1, 2, 3])
    Y = np.array([8, 2, 6])

    assert weighted_cosine_similarity(X, Y, np.array([1, 1, 1])) == cosine_similarity(X, Y)
    assert weighted_cosine_similarity(X, Y, np.array([3, 1, 1])) < cosine_similarity(X, Y)
    print("All tests passed!")