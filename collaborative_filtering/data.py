import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import category_encoders as ce

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


def _get_item_similarities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the pairwise restaurant similarities.

    Returns:
      - similarities (pd.DataFrame): a DataFrame with 3 columns "restaurant_1," "restaurant_2", and "similarity"
    """
    restaurant_1, restaurant_2, similarities = [], [], []
    for i in range(df.shape[0]):
        for j in range(i + 1, df.shape[0]):
            restaurant_1_id, restaurant_2_id = df.index[i], df.index[j]
            restaurant_1_vector, restaurant_2_vector = (
                df.iloc[i, :].values,
                df.iloc[j, :].values,
            )
            similarity = cosine_similarity(
                restaurant_1_vector.reshape(1, -1), restaurant_2_vector.reshape(1, -1)
            )[0][0]

            restaurant_1.append(restaurant_1_id)
            restaurant_2.append(restaurant_2_id)
            similarities.append(similarity)

    return pd.DataFrame(
        {"restaurant_1": restaurant_1, "restaurant_2": restaurant_2, "similarity": similarities}
    )


def _preprocess_rating_data(df: pd.DataFrame) -> pd.DataFrame:
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


def get_train_test_split(df: pd.DataFrame):
    pass


df = _load_data()
df = _preprocess_rating_data(df)
print(df.head())
similarities = _get_item_similarities(df)
print(similarities.describe())
print(similarities.head(20))
