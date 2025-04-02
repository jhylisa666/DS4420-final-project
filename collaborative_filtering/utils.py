import numpy as np

def cosine_similarity(X: np.ndarray, Y: np.ndarray) -> float:
  """Standard cosine similarity function."""
  return (X.T @ Y) / ((X.T @ X)**0.5 * (Y.T @ Y)**0.5)

def euclidean_distance(X: np.ndarray, Y: np.ndarray) -> float:
  """Standard euclidean distance function."""
  return (np.sum((X - Y) ** 2)) ** 0.5