import pandas as pd
from data import get_train_test_split, get_restaurant_similarities, RESTAURANT_FEATURES


class ContentBasedFiltering:
    """Content-based filtering class for restaurant recommendation."""

    def __init__(
        self,
        distance_metric: str = "cosine_similarity",
        weights=[1.0 for _ in range(len(RESTAURANT_FEATURES) - 1)],
    ):
        self.similarities_df = get_restaurant_similarities(distance_metric, weights)

        train_df, test_df = get_train_test_split()
        self.train_df = train_df
        self.test_df = test_df


def main():
    """
    1. Compute pairwise similarities between restaurants based on their attributes.
    2. Given a target restaurant, find the top K most similar restaurants to it.
    3. Get the average rating of each of these top K restaurants and compute the target restaurant
    rating as the weighted average of these ratings based on restaurant similarity.
    """


if __name__ == "__main__":
    main()
