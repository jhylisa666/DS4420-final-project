import pandas as pd
from utils import cosine_similarity, euclidean_distance, weighted_cosine_similarity
import category_encoders as ce
import numpy as np
from typing import Tuple, List

RESTAURANT_FEATURES = [
    "placeID",
    "alcohol",
    "smoking_area",
    "dress_code",
    "accessibility",
    "price",
    "Rambience",
    "area",
]

USER_FEATURES = [
    "userID",
    "smoker",
    "drink_level",
    "dress_preference",
    "ambience",
    "hijos",
    "interest",
    "personality",
    "religion",
    "activity",
    "budget",
]


def _get_similarities(
    df: pd.DataFrame, distance_metric: str, weights: np.ndarray, type: str = "user"
) -> pd.DataFrame:
    """
    Computes the pairwise similarities between items. Assumes that the row index of the DataFrame contains the itemID
    and that each row corresponds to a different item.
    """
    assert type in ["user", "restaurant"], f"Type {type} not supported."
    assert distance_metric in [
        "euclidean_distance",
        "cosine_similarity",
        "weighted_cosine_similarity",
    ], f"Distance metric {distance_metric} not supported."

    item_1, item_2, similarities = [], [], []
    for i in range(df.shape[0]):
        for j in range(i + 1, df.shape[0]):
            item_1_id, item_2_id = df.index[i], df.index[j]
            item_1_vector, item_2_vector = (
                df.iloc[i, :].values,
                df.iloc[j, :].values,
            )
            if distance_metric == "euclidean_distance":
                similarity = -euclidean_distance(item_1_vector, item_2_vector)
            elif distance_metric == "cosine_similarity":
                similarity = cosine_similarity(item_1_vector, item_2_vector)
            elif distance_metric == "weighted_cosine_similarity":
                similarity = weighted_cosine_similarity(
                    item_1_vector, item_2_vector, weights
                )

            item_1.append(item_1_id)
            item_2.append(item_2_id)
            similarities.append(similarity)

    if type == "user":
        return pd.DataFrame(
            {
                "user_1": item_1,
                "user_2": item_2,
                "similarity": similarities,
            }
        )
    elif type == "restaurant":
        return pd.DataFrame(
            {
                "restaurant_1": item_1,
                "restaurant_2": item_2,
                "similarity": similarities,
            }
        )


def _get_user_similarities(
    df: pd.DataFrame,
    distance_metric: str = "cosine_similarity",
    weights: np.ndarray = None,
) -> pd.DataFrame:
    """
    Computes the pairwise user similarities.

    Returns:
      - similarities (pd.DataFrame): a DataFrame with 3 columns "user_1," "user_2", and "similarity"
    """
    if distance_metric == "weighted_cosine_similarity":
        assert (
            weights is not None
        ), "Weights must be provided for weighted cosine similarity."
        assert (
            len(weights) == len(USER_FEATURES) - 1
        ), f"Weights length {len(weights)} does not match number of features {len(USER_FEATURES) - 1}."

    return _get_similarities(df, distance_metric, weights, type="user")


def _get_item_similarities(
    df: pd.DataFrame,
    distance_metric: str = "cosine_similarity",
    weights: np.ndarray = None,
) -> pd.DataFrame:
    """
    Computes the pairwise restaurant similarities.

    Returns:
      - similarities (pd.DataFrame): a DataFrame with 3 columns "restaurant_1," "restaurant_2", and "similarity"
    """
    if distance_metric == "weighted_cosine_similarity":
        assert (
            weights is not None
        ), "Weights must be provided for weighted cosine similarity."
        assert (
            len(weights) == len(RESTAURANT_FEATURES) - 1
        ), f"Weights length {len(weights)} does not match number of features {len(RESTAURANT_FEATURES) - 1}."

    return _get_similarities(df, distance_metric, weights, type="restaurant")


def _preprocess_data(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Converts all categorical features to numerical values.
    Uses a memory-efficient, lower-dimensional binary encoding for the categorical features.
    """
    encoder = ce.BinaryEncoder(cols=features)
    df = encoder.fit_transform(df)
    return df


def _load_restaurant_data() -> pd.DataFrame:
    """
    Loads the restaurant data from CSV files and returns a DataFrame containing the restaurant features.
    """
    restaurants_df = pd.read_csv("../food_data/geoplaces2.csv")[RESTAURANT_FEATURES]
    restaurants_df.set_index("placeID", inplace=True)
    assert restaurants_df.index.is_unique, "Restaurant IDs are not unique."

    return restaurants_df


def _load_user_data() -> pd.DataFrame:
    """
    Loads the user data from CSV files and returns a DataFrame containing the user features.
    """
    users_df = pd.read_csv("../food_data/userprofile.csv")[USER_FEATURES]
    users_df.set_index("userID", inplace=True)
    assert users_df.index.is_unique, "User IDs are not unique."

    return users_df


def get_restaurant_similarities(
    distance_metric: str = "cosine_similarity", weights: np.ndarray = None
) -> pd.DataFrame:
    """
    Loads the data, pre-processes it, and computes the pairwise restaurant similarities.

    Returns:
      - similarities (pd.DataFrame): a DataFrame with 3 columns "restaurant_1," "restaurant_2", and "similarity"
    """
    df = _preprocess_data(_load_restaurant_data(), RESTAURANT_FEATURES[1:])
    return _get_item_similarities(df, distance_metric, weights)


def get_user_similarities(
    distance_metric: str = "cosine_similarity", weights: np.ndarray = None
) -> pd.DataFrame:
    """
    Loads the data, pre-processes it, and computes the pairwise user similarities.

    Returns:
        - similarities (pd.DataFrame): a DataFrame with 3 columns "user_1," "user_2", and "similarity"
    """
    df = _preprocess_data(_load_user_data(), USER_FEATURES[1:])
    return _get_user_similarities(df, distance_metric, weights)


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

    unique_user_ids = ratings_df["userID"].unique()
    for index in unique_user_ids:
        # Get all the rated restaurants for the current user
        non_nan_values = ratings_df[ratings_df["userID"] == index]
        assert len(non_nan_values) > 0, f"User {index} has no rated restaurants."

        # Randomly select one restaurant to be the test set
        random_restaurant = np.random.choice(non_nan_values["placeID"])
        rating = non_nan_values[non_nan_values["placeID"] == random_restaurant][
            "rating"
        ].values[0]

        # Set the rating for the selected restaurant to NaN in the training set
        ratings_df.loc[
            (ratings_df["userID"] == index)
            & (ratings_df["placeID"] == random_restaurant),
            "rating",
        ] = np.nan

        user_ids.append(index)
        restaurant_ids.append(random_restaurant)
        ratings.append(rating)

    return ratings_df, pd.DataFrame(
        {"userID": user_ids, "placeID": restaurant_ids, "rating": ratings}
    )


if __name__ == "__main__":
    import math

    NUM_UNIQUE_RESTAURANTS = 130
    NUM_UNIQUE_USERS = 138
    assert len(get_restaurant_similarities()) == math.comb(NUM_UNIQUE_RESTAURANTS, 2)
    assert len(get_user_similarities()) == math.comb(NUM_UNIQUE_USERS, 2)
    assert get_train_test_split()[0]["placeID"].nunique() == NUM_UNIQUE_RESTAURANTS

    print("All tests passed!")

    print(get_user_similarities().head())
