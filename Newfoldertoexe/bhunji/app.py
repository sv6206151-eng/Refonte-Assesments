# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import numpy as np
# import re

# # Page setup
# st.set_page_config(page_title="Movie Recommender", layout="wide")
# st.title("🎥 Movie Recommender System")
# st.markdown("Get movie recommendations based on your favorite titles!")

# # Load data
# @st.cache_data
# def load_data():
#     movies = pd.read_csv("nwfte/movies.csv", nrows=10000).drop_duplicates().dropna()
#     ratings = pd.read_csv("nwfte/ratings.csv", nrows=10000).drop_duplicates().dropna()
#     links = pd.read_csv("nwfte/links.csv", nrows=10000).drop_duplicates().dropna()
#     tags = pd.read_csv("nwfte/tags.csv", nrows=10000).drop_duplicates().dropna()
#     return movies, ratings, links, tags

# movies, ratings, links, tags = load_data()

# # Filter top movies
# top_n = 5000
# top_movie_ids = ratings['movieId'].value_counts().head(top_n).index
# movies = movies[movies['movieId'].isin(top_movie_ids)].reset_index(drop=True)

# # Simplify genres
# def simplify_genres(genres):
#     if pd.isna(genres):
#         return 'unknown'
#     filtered = [g for g in genres.split('|') if g.strip()]
#     return '|'.join(filtered) if filtered else 'unknown'

# movies['genres'] = movies['genres'].apply(simplify_genres)
# movies = movies[movies['genres'].str.strip() != '']
# movies = movies[movies['genres'] != 'unknown'].reset_index(drop=True)

# # Tokenize titles
# def tokenize_title(title):
#     tokens = re.findall(r'\b\w+\b', title.lower())
#     return ' '.join(tokens)

# movies['tokenized_title'] = movies['title'].apply(tokenize_title)

# def search_titles(query):
#     query = query.lower().strip()
#     return movies[movies['tokenized_title'].str.contains(query, regex=False)]

# # Content-Based Filtering
# @st.cache_data
# def compute_similarity_matrix(movies_df):
#     tfidf = TfidfVectorizer(stop_words='english')
#     titles = movies_df['title'].dropna().astype(str)
#     titles = titles[titles.str.strip() != '']
#     tfidf_matrix = tfidf.fit_transform(titles.tolist())
#     return linear_kernel(tfidf_matrix)

# cosine_sim = compute_similarity_matrix(movies)
# title_to_index = pd.Series(movies.index, index=movies['title'])

# def recommend_content(title):
#     try:
#         idx = title_to_index[title]
#         sim_scores = list(enumerate(cosine_sim[idx]))
#         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
#         movie_indices = [i[0] for i in sim_scores]
#         return movies['title'].iloc[movie_indices].tolist()
#     except Exception as e:
#         st.warning(f"Recommendation error: {e}")
#         return []

# # Collaborative Filtering
# @st.cache_resource
# def train_collab_model(ratings_df):
#     user_item = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
#     user_item = user_item.dropna(how='all').fillna(0)
#     svd = TruncatedSVD(n_components=20, random_state=42)
#     reduced = svd.fit_transform(user_item.values)
#     reconstructed = np.dot(reduced, svd.components_)
#     return reconstructed, user_item

# reconstructed_matrix, user_item_matrix = train_collab_model(ratings)

# def recommend_collab(user_id, n=10):
#     if user_id not in user_item_matrix.index:
#         return []
#     user_idx = user_item_matrix.index.get_loc(user_id)
#     user_ratings = reconstructed_matrix[user_idx]
#     rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
#     unrated = [i for i in range(len(user_ratings)) if user_item_matrix.columns[i] not in rated]
#     top_indices = sorted(unrated, key=lambda i: user_ratings[i], reverse=True)[:n]
#     top_ids = [user_item_matrix.columns[i] for i in top_indices]
#     return movies[movies['movieId'].isin(top_ids)]['title'].tolist()

# def evaluate_model(user_item_matrix):
#     actual = user_item_matrix.values
#     svd = TruncatedSVD(n_components=20, random_state=42)
#     reduced = svd.fit_transform(actual)
#     reconstructed = np.dot(reduced, svd.components_)
#     rmse = np.sqrt(np.mean((actual - reconstructed) ** 2))
#     mae = np.mean(np.abs(actual - reconstructed))
#     return {'RMSE': rmse, 'MAE': mae}

# evaluation_results = evaluate_model(user_item_matrix)

# # Sidebar EDA
# with st.sidebar:
#     st.header("📊 Explore Data")
#     if st.checkbox("Show Rating Distribution"):
#         fig, ax = plt.subplots()
#         sns.histplot(ratings['rating'], bins=10, ax=ax)
#         ax.set_title("Rating Distribution")
#         st.pyplot(fig)

#     if st.checkbox("Show Top Rated Movies"):
#         top_movies = ratings['movieId'].value_counts().head(10)
#         top_titles = movies[movies['movieId'].isin(top_movies.index)]['title']
#         st.write("Most Rated Movies:")
#         for title in top_titles:
#             st.write(f"- {title}")

#     if st.checkbox("Show Genre Popularity"):
#         genre_counts = movies['genres'].str.split('|').explode().value_counts()
#         fig, ax = plt.subplots()
#         sns.barplot(y=genre_counts.index[:10], x=genre_counts.values[:10], ax=ax)
#         ax.set_title("Top Genres")
#         st.pyplot(fig)

#     if st.checkbox("Show Model Evaluation"):
#         st.subheader("📈 Collaborative Filtering Evaluation")
#         st.write(f"**RMSE:** {evaluation_results['RMSE']:.4f}")
#         st.write(f"**MAE:** {evaluation_results['MAE']:.4f}")
#         st.caption("Lower RMSE and MAE indicate better prediction accuracy.")

# # Main interface
# movie_list = movies['title'].sort_values().tolist()
# selected_movie = st.selectbox("🎬 Select a movie you like:", movie_list)

# if st.button("🎯 Recommend Similar Movies (Content-Based)"):
#     recommendations = recommend_content(selected_movie)
#     if recommendations:
#         st.success("You might also enjoy:")
#         st.write("\n".join([f"• {rec}" for rec in recommendations]))
#     else:
#         st.warning("No recommendations found. Try another movie.")

# # Search
# st.subheader("🔍 Search Movies by Title or Year")
# search_input = st.text_input("Enter a keyword or year:")
# if search_input:
#     results = search_titles(search_input)
#     if not results.empty:
#         st.write("Matching Movies:")
#         for title in results['title']:
#             st.write(f"• {title}")
#     else:
#         st.warning("No matches found.")

# # Collaborative Filtering
# st.subheader("🔐 Personalized Recommendations (Collaborative Filtering)")
# user_id_input = st.text_input("Enter your User ID (numeric):")
# if user_id_input.isdigit():
#     user_id = int(user_id_input)
#     user_recs = recommend_collab(user_id)
#     if user_recs:
#         st.success(f"Top picks for User {user_id}:")
#         st.write("\n".join([f"• {rec}" for rec in user_recs]))
#     else:
#         st.warning("No recommendations found for this user.")
# else:
#     st.info("Enter a valid User ID to get personalized recommendations.")

# # Footer
# st.markdown("---")
# st.caption("Built with ❤️ using Streamlit, Scikit-learn, and MovieLens data")






import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import re

# Page setup
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎥 Movie Recommender System")
st.markdown("Get movie recommendations based on your favorite titles!")

# Dummy data loader
@st.cache_data
def load_data():
    # movies = pd.DataFrame({
    #     'movieId': [1, 2, 3, 4, 5],
    #     'title': ['Toy Story', 'Jumanji', 'Grumpier Old Men', 'Waiting to Exhale', 'Father of the Bride Part II'],
    #     'genres': ['Adventure|Animation|Children|Comedy|Fantasy', 'Adventure|Children|Fantasy', 'Comedy|Romance', 'Comedy|Drama|Romance', 'Comedy']
    # }).drop_duplicates().dropna()

    movies = pd.read_csv("nwfte/movies.csv", nrows=10000).drop_duplicates().dropna()
    
    
    ratings = pd.DataFrame({
        'userId': [1, 2, 1, 3, 2],
        'movieId': [1, 2, 3, 4, 5],
        'rating': [4.0, 5.0, 3.0, 2.0, 4.5],
        'timestamp': [964982703, 964981247, 964982224, 964983815, 964982931]
    }).drop_duplicates().dropna()
    
    # ratings = pd.read_csv("nwfte/ratings.csv", nrows=10000).drop_duplicates().dropna()

    # links = pd.DataFrame({
    #     'movieId': [1, 2, 3, 4, 5],
    #     'imdbId': [114709, 113497, 113228, 114885, 113041],
    #     'tmdbId': [862, 8844, 15602, 31357, 11862]
    # }).drop_duplicates().dropna()
    
    links = pd.read_csv("nwfte/links.csv", nrows=10000).drop_duplicates().dropna()

    # tags = pd.DataFrame({
    #     'userId': [2, 2],
    #     'movieId': [2, 5],
    #     'tag': ['funny', 'family'],
    #     'timestamp': [964982703, 964982703]
    # }).drop_duplicates().dropna()
    
    tags = pd.read_csv("nwfte/tags.csv", nrows=10000).drop_duplicates().dropna()

    return movies, ratings, links, tags

movies, ratings, links, tags = load_data()

# Genre simplification
def simplify_genres(genres):
    if pd.isna(genres):
        return 'unknown'
    filtered = [g for g in genres.split('|') if g.strip()]
    return '|'.join(filtered) if filtered else 'unknown'

movies['genres'] = movies['genres'].apply(simplify_genres)
movies = movies[movies['genres'].str.strip() != '']
movies = movies[movies['genres'] != 'unknown'].reset_index(drop=True)

# Tokenize titles
def tokenize_title(title):
    tokens = re.findall(r'\b\w+\b', title.lower())
    return ' '.join(tokens)

movies['tokenized_title'] = movies['title'].apply(tokenize_title)

def search_titles(query):
    query = query.lower().strip()
    return movies[movies['tokenized_title'].str.contains(query, regex=False)]

# Content-Based Filtering
@st.cache_data
def compute_similarity_matrix(movies_df):
    tfidf = TfidfVectorizer(stop_words='english')
    titles = movies_df['title'].dropna().astype(str)
    titles = titles[titles.str.strip() != '']
    tfidf_matrix = tfidf.fit_transform(titles.tolist())
    return linear_kernel(tfidf_matrix)

cosine_sim = compute_similarity_matrix(movies)
title_to_index = pd.Series(movies.index, index=movies['title'])

def recommend_content(title):
    try:
        idx = title_to_index[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices].tolist()
    except Exception as e:
        st.warning(f"Recommendation error: {e}")
        return []

# Collaborative Filtering
@st.cache_resource
# def train_collab_model(ratings_df):
#     user_item = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
#     user_item = user_item.dropna(how='all').fillna(0)
#     svd = TruncatedSVD(n_components=2, random_state=42)
#     reduced = svd.fit_transform(user_item.values)
#     reconstructed = np.dot(reduced, svd.components_)
#     return reconstructed, user_item

@st.cache_resource
def train_collab_model(ratings_df):
    user_item = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')

    # Drop users and movies with all NaNs
    user_item = user_item.dropna(how='all', axis=0).dropna(how='all', axis=1)

    # If the matrix is empty, return dummy values
    if user_item.empty or user_item.shape[0] == 0 or user_item.shape[1] == 0:
        st.warning("Collaborative model cannot be trained: no valid user-item data.")
        return np.zeros((1, 1)), pd.DataFrame()

    # Fill missing ratings with 0
    user_item = user_item.fillna(0)

    svd = TruncatedSVD(n_components=min(2, min(user_item.shape)-1), random_state=42)
    reduced = svd.fit_transform(user_item.values)
    reconstructed = np.dot(reduced, svd.components_)
    return reconstructed, user_item


reconstructed_matrix, user_item_matrix = train_collab_model(ratings)

def recommend_collab(user_id, n=5):
    if user_id not in user_item_matrix.index:
        return []
    user_idx = user_item_matrix.index.get_loc(user_id)
    user_ratings = reconstructed_matrix[user_idx]
    rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    unrated = [i for i in range(len(user_ratings)) if user_item_matrix.columns[i] not in rated]
    top_indices = sorted(unrated, key=lambda i: user_ratings[i], reverse=True)[:n]
    top_ids = [user_item_matrix.columns[i] for i in top_indices]
    return movies[movies['movieId'].isin(top_ids)]['title'].tolist()

def evaluate_model(user_item_matrix):
    actual = user_item_matrix.values
    svd = TruncatedSVD(n_components=2, random_state=42)
    reduced = svd.fit_transform(actual)
    reconstructed = np.dot(reduced, svd.components_)
    rmse = np.sqrt(np.mean((actual - reconstructed) ** 2))
    mae = np.mean(np.abs(actual - reconstructed))
    return {'RMSE': rmse, 'MAE': mae}

evaluation_results = evaluate_model(user_item_matrix)

# Sidebar EDA
with st.sidebar:
    st.header("📊 Explore Data")
    if st.checkbox("Show Rating Distribution"):
        fig, ax = plt.subplots()
        sns.histplot(ratings['rating'], bins=5, ax=ax)
        ax.set_title("Rating Distribution")
        st.pyplot(fig)

    if st.checkbox("Show Top Rated Movies"):
        top_movies = ratings['movieId'].value_counts().head(5)
        top_titles = movies[movies['movieId'].isin(top_movies.index)]['title']
        st.write("Most Rated Movies:")
        for title in top_titles:
            st.write(f"- {title}")

    if st.checkbox("Show Genre Popularity"):
        genre_counts = movies['genres'].str.split('|').explode().value_counts()
        fig, ax = plt.subplots()
        sns.barplot(y=genre_counts.index[:5], x=genre_counts.values[:5], ax=ax)
        ax.set_title("Top Genres")
        st.pyplot(fig)

    if st.checkbox("Show Model Evaluation"):
        st.subheader("📈 Collaborative Filtering Evaluation")
        st.write(f"**RMSE:** {evaluation_results['RMSE']:.4f}")
        st.write(f"**MAE:** {evaluation_results['MAE']:.4f}")
        st.caption("Lower RMSE and MAE indicate better prediction accuracy.")

# Main interface
movie_list = movies['title'].sort_values().tolist()
selected_movie = st.selectbox("🎬 Select a movie you like:", movie_list)

if st.button("🎯 Recommend Similar Movies (Content-Based)"):
    recommendations = recommend_content(selected_movie)
    if recommendations:
        st.success("You might also enjoy:")
        st.write("\n".join([f"• {rec}" for rec in recommendations]))
    else:
        st.warning("No recommendations found. Try another movie.")

# Search
st.subheader("🔍 Search Movies by Title or Year")
search_input = st.text_input("Enter a keyword or year:")
if search_input:
    results = search_titles(search_input)
    if not results.empty:
        st.write("Matching Movies:")
        for title in results['title']:
            st.write(f"• {title}")
    else:
        st.warning("No matches found.")

# Collaborative Filtering
st.subheader("🔐 Personalized Recommendations (Collaborative Filtering)")
user_id_input = st.text_input("Enter your User ID (numeric):")
if user_id_input.isdigit():
    user_id = int(user_id_input)
    user_recs = recommend_collab(user_id)
    if user_recs:
        st.success(f"Top picks for User {user_id}:")
        st.write("\n".join([f"• {rec}" for rec in user_recs]))
    else:
        st.warning("No recommendations found for this user.")
else:
    st.info("Enter a valid User ID to get personalized recommendations.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Scikit-learn, and MovieLens-style dummy data")
