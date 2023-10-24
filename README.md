# Hybrid Recommender System

## Business Problem

The goal of this project is to implement a Hybrid Recommender System that combines item-based and user-based recommendation methods to provide movie recommendations for a specific user. The system aims to offer a total of 10 movie recommendations by using both methods.

## Dataset

### Movie Dataset:
- `movieId`: Movie identifier.
- `title`: Movie title.
- `genre`: Movie genres.

### Rating Dataset:
- `userId`: User identifier.
- `movieId`: Movie identifier.
- `rating`: Rating provided by the user.
- `timestamp`: Timestamp of the rating.

## Task

### Data Preparation
- Load both the movie and rating datasets.
- Combine movie names and genres from the movie dataset with user ratings from the rating dataset.
- Remove movies with less than 1000 total ratings, as they are considered "rare movies".

### Identifying Movies Watched by the User
- Randomly select a user.
- Create a user-specific dataframe (random_user_df) containing movies watched by the selected user.
- Identify the movies (the_movies_of_random_user_ID) for which the user has provided ratings.

### Accessing Data and IDs of Other Users Watching the Same Movies
- Filter the dataframe to include only movies watched by the selected user (movies_watched_df).
- Calculate the number of users who have watched each movie (user_movie_count).
- Select users who have watched more than 60% of the same movies as the chosen user
(users_watched_same_movies_with_target).

### Determining Users Most Similar to the Selected User
- Filter the movies_watched_df dataframe to include only users with a correlation of 0.65 or higher with the chosen
user.
- Calculate the correlation between users and create a new dataframe (corr_df).
- Select the users with the highest correlation and save them in the top_users dataframe.

### Calculating Weighted Average Recommendation Score and Selecting the Top 5 Movies
- Calculate the weighted average rating for each movie and create the recommendation_df dataframe.
- Select movies with a weighted rating higher than 3.5 and sort them by weighted rating.
- The top 5 movies to be recommended are obtained.

### Item-Based Recommendation
- Identify the most recently watched and "5" rated movie by the user (the_movie).
- Filter the user_based_df dataframe based on the selected movie.
- Calculate the correlation between the selected movie and other movies.
- Recommend the top 5 movies, excluding the selected movie itself.

## Usage

To use the Hybrid Recommender System, you can follow these steps:
1. Prepare the dataset by loading and cleaning the movie and rating datasets.
2. Identify the movies watched and rated by the user.
3. Determine similar users based on movies watched.
4. Calculate weighted average recommendation scores and select the top 5 movies.
5. Obtain item-based recommendations based on the user's most recently watched movie.

## Results

The system provides movie recommendations that are a combination of user-based and item-based methods, offering a diverse set of movies that the user may enjoy based on their viewing history and preferences.
