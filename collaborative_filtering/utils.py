import numpy as np
from sklearn.metrics import jaccard_score


def cosine_similarity(X: np.ndarray, Y: np.ndarray) -> float:
    """Standard cosine similarity function."""
    return (X.T @ Y) / ((X.T @ X) ** 0.5 * (Y.T @ Y) ** 0.5)


def jaccard_similarity(X: np.ndarray, Y: np.ndarray) -> float:
    assert all(x in [0, 1] for x in X) and all(
        y in [0, 1] for y in Y
    ), "X and Y must be binary vectors."
    return jaccard_score(X, Y)


if __name__ == "__main__":
    X = np.array([1, 0, 1])
    Y = np.array([1, 1, 1])
    print("Cosine Similarity:", cosine_similarity(X, Y))
    print("Jaccard Similarity:", jaccard_similarity(X, Y))
