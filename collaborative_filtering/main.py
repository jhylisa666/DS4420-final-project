import pandas as pd

def main():
    """
    1. Compute pairwise similarities between restaurants based on their attributes.
    2. Given a target restaurant, find the top K most similar restaurants to it.
    3. Get the average rating of each of these top K restaurants and compute the target restaurant 
    rating as the weighted average of these ratings based on restaurant similarity.
    """
    ratings_df = pd.read_csv("../food_data/rating_final.csv")
    print(ratings_df.head())

if __name__ == '__main__':
    main()