import pandas as pd
from utils import cosine_similarity, euclidean_distance, weighted_cosine_similarity
import category_encoders as ce
import numpy as np
from typing import Tuple

RESTAURANT_FEATURES = [
    "placeID",
    "alcohol",
    "smoking_area",
    "dress_code",
    "accessibility",
    "price",
    "Rambience",
    "area",
    "Rcuisine",
    "parking_lot",
]

WEIGHTS = [1.0 for _ in range(len(RESTAURANT_FEATURES) - 1)]


def _get_item_similarities(
    df: pd.DataFrame, distance_metric: str = "cosine_similarity"
) -> pd.DataFrame:
    """
    Computes the pairwise restaurant similarities.

    Returns:
      - similarities (pd.DataFrame): a DataFrame with 3 columns "restaurant_1," "restaurant_2", and "similarity"
    """
    assert distance_metric in [
        "euclidean_distance",
        "cosine_similarity",
        "weighted_cosine_similarity",
    ], f"Distance metric {distance_metric} not supported."

    restaurant_1, restaurant_2, similarities = [], [], []
    for i in range(df.shape[0]):
        for j in range(i + 1, df.shape[0]):
            restaurant_1_id, restaurant_2_id = df.index[i], df.index[j]
            restaurant_1_vector, restaurant_2_vector = (
                df.iloc[i, :].values,
                df.iloc[j, :].values,
            )
            if distance_metric == "euclidean_distance":
                similarity = -euclidean_distance(
                    restaurant_1_vector, restaurant_2_vector
                )
            elif distance_metric == "cosine_similarity":
                similarity = cosine_similarity(restaurant_1_vector, restaurant_2_vector)
            elif distance_metric == "weighted_cosine_similarity":
                similarity = weighted_cosine_similarity(
                    restaurant_1_vector, restaurant_2_vector, WEIGHTS
                )

            restaurant_1.append(restaurant_1_id)
            restaurant_2.append(restaurant_2_id)
            similarities.append(similarity)

    return pd.DataFrame(
        {
            "restaurant_1": restaurant_1,
            "restaurant_2": restaurant_2,
            "similarity": similarities,
        }
    )


def _preprocess_restaurant_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all categorical features to numerical values.
    Uses a memory-efficient, lower-dimensional binary encoding for the categorical features.
    """
    encoder = ce.BinaryEncoder(cols=RESTAURANT_FEATURES[1:])
    df = encoder.fit_transform(df)
    return df


def _load_data() -> pd.DataFrame:
    """
    Loads the restaurant data from CSV files and merges them into a single DataFrame
    containing the relevant features.
    """
    restaurants_df = pd.read_csv("../food_data/geoplaces2.csv")
    cuisine_df = pd.read_csv("../food_data/chefmozcuisine.csv")
    parking_df = pd.read_csv("../food_data/chefmozparking.csv")

    merged_df = pd.merge(
        pd.merge(restaurants_df, cuisine_df, on="placeID"), parking_df, on="placeID"
    )[RESTAURANT_FEATURES]
    merged_df.set_index("placeID", inplace=True)

    return merged_df


def get_train_test_split() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given the ratings data, for each user, it random selects one restaurant to be the test set
    and the rest to be the training set. The selected restaurant's rating is removed from the training set.

    Returns:
        - ratings_df (pd.DataFrame): a DataFrame with the training set ratings
        - test_df (pd.DataFrame): a DataFrame with the test set ratings
    """
    user_ids, restaurant_ids, ratings = [], [], []
    ratings_df = pd.read_csv("../food_data/rating_final.csv")
    ratings_df = ratings_df.pivot(index="userID", columns="placeID", values="rating")

    for index in ratings_df.index:
        # Get all the rated restaurants for the current user
        non_nan_values = ratings_df.loc[index][ratings_df.loc[index].notna()]
        assert len(non_nan_values) > 0, f"User {index} has no rated restaurants."

        # Randomly select one restaurant to be the test set
        random_restaurant = np.random.choice(non_nan_values.index)
        rating = non_nan_values[random_restaurant]

        # Set the rating for the selected restaurant to NaN in the training set
        ratings_df.loc[index, random_restaurant] = np.nan

        user_ids.append(index)
        restaurant_ids.append(random_restaurant)
        ratings.append(rating)

    return ratings_df, pd.DataFrame(
        {"userID": user_ids, "placeID": restaurant_ids, "rating": ratings}
    )


train_df, test_df = get_train_test_split()
print(train_df.head())
print(test_df.head())
