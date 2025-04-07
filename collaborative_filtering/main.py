import numpy as np
from data import (
    get_train_test_split,
    get_restaurant_similarities,
    get_user_similarities,
)
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


class HybridFiltering:
    """
    Hybrid filtering class for restaurant recommendation.
    
    Steps:
    1. Given a target user and target restaurant, find the top K most similar restaurants to the target restaurant.
    2. For each restaurant, find the users who have rated that restaurant.
    3. For each user, find the top K most similar users to the target user.
    4. Compute the predicted rating of the restaurant as the weighted user similarity of the ratings of the top K similar users.
    5. Compute the predicted rating of the restaurant as the weighted item similarity of these ratings.
    """

    def __init__(
        self,
        distance_metric: str = "cosine_similarity",
    ):
        self.restaurant_similarities_df = get_restaurant_similarities(distance_metric)
        self.user_similarities_df = get_user_similarities(distance_metric)
        self.train_df, _ = get_train_test_split()

    def predict(self, user_id: int, restaurant_id: int, k: int = 3) -> float:
        """
        Predicts the rating for a given user and restaurant using content-based filtering.

        Args:
            - user_id (int): The ID of the user.
            - restaurant_id (int): The ID of the restaurant.
            - k (int): The number of similar restaurants to consider.

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
        average_restaurant_ratings, user_similarity_ranges = [], []
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
            user_similarity_ranges.append(user_similarities[0] - user_similarities[-1])
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
            average_restaurant_ratings.append(np.mean(user_ratings))

        restaurant_ratings = np.asarray(restaurant_ratings)

        return (
            restaurant_ratings.dot(restaurant_similarities)
            / restaurant_similarities.sum(), average_restaurant_ratings, user_similarity_ranges
        )


def main():
    """Main function to test the HybridFiltering class."""

    hf = HybridFiltering(distance_metric="jaccard_similarity")
    _, test_df = get_train_test_split()
    preds, gt = [], test_df["rating"].values

    all_restaurant_average_ratings, all_user_similarity_ranges = [], []
    for index in range(test_df.shape[0]):
        user_id = test_df.iloc[index]["userID"]
        restaurant_id = test_df.iloc[index]["placeID"]
        predicted_rating, average_restaurant_ratings, user_similarity_ranges = hf.predict(user_id, restaurant_id, k=1)
        all_restaurant_average_ratings.extend(average_restaurant_ratings)
        all_user_similarity_ranges.extend(user_similarity_ranges)

        if predicted_rating > 1.3:
            preds.append(2)
        elif predicted_rating > 0.7:
            preds.append(1)
        else:
            preds.append(0)

    print(classification_report(y_true=gt, y_pred=preds, zero_division=0))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].hist(all_restaurant_average_ratings, bins=6, edgecolor='black')
    axs[0].set_title("Histogram of Average Restaurant Ratings", fontsize=16)
    axs[0].set_xlabel("Average Restaurant Ratings", fontsize=16)
    axs[0].set_ylabel("Frequency", fontsize=16)

    axs[1].hist(all_user_similarity_ranges, bins=10, edgecolor='black', color='red')
    axs[1].set_title("Histogram of Top-K User Similarity Ranges", fontsize=16)
    axs[1].set_xlabel("User Similarity Range", fontsize=16)
    axs[1].set_ylabel("Frequency", fontsize=16)

    plt.tight_layout()
    plt.savefig('cf_histograms.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
