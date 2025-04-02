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
        self.train_df, _ = get_train_test_split()

    def predict(self, restaurant_id: int, k: int = 5) -> float:
        """
        Predicts the rating for a given user and restaurant using content-based filtering.

        Args:
            user_id (int): The ID of the user.
            restaurant_id (int): The ID of the restaurant.
            k (int): The number of similar restaurants to consider.

        Returns:
            float: The predicted rating for the restaurant by the user.
        """

        # Get the top K similar restaurants to the target restaurant
        top_k_similarities_df = self.similarities_df[
            (self.similarities_df["restaurant_1"] == restaurant_id)
            | (self.similarities_df["restaurant_2"] == restaurant_id)
        ].nlargest(k, "similarity")

        # Get the restaurant IDs of the top K similar restaurants
        top_k_similar_restaurant_ids = top_k_similarities_df.apply(
            lambda row: (
                row["restaurant_2"]
                if row["restaurant_2"] != restaurant_id
                else row["restaurant_1"]
            ),
            axis=1,
        ).tolist()

        assert len(set(top_k_similar_restaurant_ids)) == len(top_k_similarities_df)
        assert len(top_k_similar_restaurant_ids) == k

        # Get the average ratings of the top K similar restaurants
        average_ratings = (
            self.train_df[self.train_df["placeID"].isin(top_k_similar_restaurant_ids)]
            .groupby("placeID")["rating"]
            .mean()
            .values
        )

        # Get the similarities of the top K similar restaurants to the target restaurant
        similarities = top_k_similarities_df["similarity"].values

        return average_ratings.dot(similarities) / similarities.sum()


def main():
    """
    1. Compute pairwise similarities between restaurants based on their attributes.
    2. Given a target restaurant, find the top K most similar restaurants to it.
    3. Get the average rating of each of these top K restaurants and compute the target restaurant
    rating as the weighted average of these ratings based on restaurant similarity. -> Get the
    weighted average using the user's rating of these restaurants and the restaurant similarity. Use average rating of user
    if the restaurant was not rated by the user.
    """
    cbf = ContentBasedFiltering(distance_metric="cosine_similarity")
    _, test_df = get_train_test_split()
    print(test_df.head(10))

    for index in range(test_df.shape[0]):
        restaurant_id = test_df.iloc[index]["placeID"]
        predicted_rating = cbf.predict(restaurant_id, k=5)
        print(
            f"Predicted rating for restaurant {restaurant_id}: {predicted_rating:.2f}"
        )
        if index == 10:
            break


if __name__ == "__main__":
    main()
