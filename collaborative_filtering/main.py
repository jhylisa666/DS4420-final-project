import numpy as np
from data import (
    get_train_test_split,
    get_restaurant_similarities,
    get_user_similarities,
)
from sklearn.metrics import classification_report


class HybridFiltering:
    """Hybrid filtering class for restaurant recommendation."""

    def __init__(
        self,
        distance_metric: str = "cosine_similarity",
    ):
        self.restaurant_similarities_df = get_restaurant_similarities(distance_metric)
        self.user_similarities_df = get_user_similarities(distance_metric)
        self.train_df, _ = get_train_test_split()

    def predict(self, user_id: int, restaurant_id: int, k: int = 5) -> float:
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
        top_k_similarities_df = self.restaurant_similarities_df[
            (self.restaurant_similarities_df["restaurant_1"] == restaurant_id)
            | (self.restaurant_similarities_df["restaurant_2"] == restaurant_id)
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

        restaurant_similarities, restaurant_ratings = (
            top_k_similarities_df["similarity"].values,
            [],
        )
        # For each restaurant, get the users who have rated the restaurant
        for restaurant_id in top_k_similar_restaurant_ids:
            user_ratings_df = self.train_df[
                (self.train_df["placeID"] == restaurant_id)
                & (self.train_df["rating"].notna())
            ]
            user_ids = user_ratings_df["userID"].values
            if user_id in user_ids:
                user_ids = np.delete(user_ids, np.where(user_ids == user_id))
            assert (
                user_id not in user_ids
            ), "Cannot include the target user as a similar user."

            user_similarities_df = self.user_similarities_df[
                (
                    (self.user_similarities_df["user_1"] == user_id)
                    & self.user_similarities_df["user_2"].isin(user_ids)
                )
                | (
                    (self.user_similarities_df["user_2"] == user_id)
                    & self.user_similarities_df["user_1"].isin(user_ids)
                )
            ]

            assert len(user_similarities_df) == len(
                user_ids
            ), f"Found {len(user_similarities_df)} similar users for {len(user_ids)} users to {user_id} who rated the restaurant {restaurant_id}."
            user_similarities_df = user_similarities_df.nlargest(k, "similarity")

            # We want to get the ratings of the top K similar users who have rated the restaurant
            user_similarities, user_ratings = (
                user_similarities_df["similarity"].values,
                [],
            )
            for _, row in user_similarities_df.iterrows():
                similar_user_id = (
                    row["user_1"] if row["user_1"] != user_id else row["user_2"]
                )
                user_ratings.append(
                    user_ratings_df[user_ratings_df["userID"] == similar_user_id][
                        "rating"
                    ].values[0]
                )

            user_ratings = np.asarray(user_ratings)
            restaurant_ratings.append(
                user_ratings.dot(user_similarities) / user_similarities.sum()
            )

        restaurant_ratings = np.asarray(restaurant_ratings)
        return (
            restaurant_ratings.dot(restaurant_similarities)
            / restaurant_similarities.sum()
        )


def main():
    """
    1. Compute pairwise similarities between restaurants based on their attributes.
    2. Given a target restaurant, find the top K most similar restaurants to it.
    3. Get the average rating of each of these top K restaurants and compute the target restaurant
    rating as the weighted average of these ratings based on restaurant similarity. -> Get the
    weighted average using the user's rating of these restaurants and the restaurant similarity. Use average rating of user
    if the restaurant was not rated by the user.

    Or, given a target user and item, we can find the top K most similar items to the target item. Then,
    for each item, among the users that have rated that item, we can find the top K most similar users to the target user.
    We can compute the predicted rating of the item as the weighted user similarity of the ratings of the top K similar users.
    Repeat this predicted item rating for all items and then compute the weighted item similarity of these ratings.

    """
    hf = HybridFiltering(distance_metric="jaccard_similarity")
    _, test_df = get_train_test_split()
    preds, gt = [], test_df["rating"].values

    for index in range(test_df.shape[0]):
        user_id = test_df.iloc[index]["userID"]
        restaurant_id = test_df.iloc[index]["placeID"]
        predicted_rating = hf.predict(user_id, restaurant_id, k=3)

        if predicted_rating > 1.3:
            preds.append(2)
        elif predicted_rating > 0.7:
            preds.append(1)
        else:
            preds.append(0)

    print(classification_report(y_true=gt, y_pred=preds, zero_division=0))


if __name__ == "__main__":
    main()
