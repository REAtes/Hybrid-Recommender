# ### Data Preparation
# 1- Load both the movie and rating datasets.
# 2- Combine movie names and genres from the movie dataset with user ratings from the rating dataset.
# 3- Remove movies with less than 1000 total ratings, as they are considered "rare movies".

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 1500)

movie_df = pd.read_csv("Hybrid-Recommender-System/movie.csv")
rating_df = pd.read_csv("Hybrid-Recommender-System/rating.csv")
df = pd.merge(movie_df, rating_df, how="left", on="movieId")


def check_df(dataframe, head=5):
    print("#################### Shape ####################")
    print(dataframe.shape)
    print("#################### Types ####################")
    print(dataframe.dtypes)
    print("#################### Num of Unique ####################")
    print(dataframe.nunique())
    print("#################### Head ####################")
    print(dataframe.head(head))
    print("#################### Tail ####################")
    print(dataframe.tail(head))
    print("#################### NA ####################")
    print(dataframe.isnull().sum())
    print("#################### Quantiles ####################")
    print(dataframe.describe([0.01, 0.05, 0.75, 0.90, 0.95, 0.99]).T)


check_df(df)
df = df.dropna()

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["count"] <= 1000].index
df = df[~df["title"].isin(rare_movies)]


# ### Identifying Movies Watched by the User
# 1- Randomly select a user.
# 2- Create a user-specific dataframe (random_user_df) containing movies watched by the selected user.
# 3- Identify the movies (the_movies_of_random_user_ID) for which the user has provided ratings.

random_user_ID = 1000

user_based_df = df.pivot_table(index=["userId"],
                               columns=["title"],
                               values="rating")
random_user_df = user_based_df[user_based_df.index == random_user_ID]
the_movies_of_random_user_ID = random_user_df.columns[random_user_df.notna().any()].to_list()
number_of_the_movies_of_random_user_ID = len(the_movies_of_random_user_ID)  # 53


# ### Accessing Data and IDs of Other Users Watching the Same Movies
# 1- Filter the dataframe to include only movies watched by the selected user (movies_watched_df).
# 2- Calculate the number of users who have watched each movie (user_movie_count).
# 3- Select users who have watched more than 80% of the same movies as the chosen user
# (users_watched_same_movies_with_target).

movies_watched_df = user_based_df[the_movies_of_random_user_ID]
number_of_col_movies_watched_df = movies_watched_df.shape[1]  # 53
user_movie_count = movies_watched_df.T.notnull().sum().reset_index()
user_movie_count.columns = ["userID", "number of watched same movies"]
users_watched_same_movies_with_target = user_movie_count[user_movie_count["number of watched same movies"]
                                                         > 60 / 100 * number_of_the_movies_of_random_user_ID]['userID']


# ### Determining Users Most Similar to the Selected User
# 1- Filter the movies_watched_df dataframe to include only users with a correlation of 0.65 or higher with the chosen
# user.
# 2- Calculate the correlation between users and create a new dataframe (corr_df).
# 3- Select the users with the highest correlation and save them in the top_users dataframe.

final_df = movies_watched_df[movies_watched_df.index.isin(users_watched_same_movies_with_target)]

corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

top_users_df = corr_df[(corr_df['user_id_1'] == random_user_ID) &
                       (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]]\
    .sort_values(by='corr', ascending=False)\
    .reset_index(drop=True)
top_users_df.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users_ratings_df = top_users_df.merge(rating_df[["userId", "movieId", "rating"]], how='inner')


# ### Calculating Weighted Average Recommendation Score and Selecting the Top 5 Movies
# 1- Calculate the weighted average rating for each movie and create the recommendation_df dataframe.
# 2- Select movies with a weighted rating higher than 3.5 and sort them by weighted rating.
# 3- The top 5 movies to be recommended are obtained.

top_users_ratings_df["weighted_rating"] = top_users_ratings_df["corr"] * top_users_ratings_df["rating"]
top_users_ratings_df = top_users_ratings_df[top_users_ratings_df["userId"] != random_user_ID]
recommendation_df = top_users_ratings_df.groupby("movieId").agg({"weighted_rating": "mean"}).reset_index()

recommendation_df = recommendation_df[recommendation_df["weighted_rating"] > 3.5]

the_top_5_movies = recommendation_df.sort_values("weighted_rating", ascending=False).head()
the_top_5_movies.merge(movie_df[["movieId", "title", "genres"]], how='inner')


# ### Item-Based Recommendation
# 1- Identify the most recently watched and "5" rated movie by the user (the_movie).
# 2- Filter the user_based_df dataframe based on the selected movie.
# 3- Calculate the correlation between the selected movie and other movies.
# 4- Recommend the top 5 movies, excluding the selected movie itself.

the_movie = df[(df["userId"] == random_user_ID) & (df["rating"] == 5)]\
                .sort_values("timestamp", ascending=False)["title"][0:1].values[0]

the_movie_rating_by_other_users = user_based_df[the_movie]
user_based_df.corrwith(the_movie_rating_by_other_users).sort_values(ascending=False).head(5)