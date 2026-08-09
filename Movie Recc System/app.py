import streamlit as st
import pandas as pd
import pickle
import requests
import os

# Set page config
st.set_page_config(
    page_title="CineMatch - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Dark theme & Glow aesthetic */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8F00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        text-align: center;
        color: #A0AAB0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .movie-card {
        background: #1E222D;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(255, 75, 75, 0.3);
    }
    
    .movie-title {
        font-weight: 600;
        font-size: 1rem;
        color: #FFFFFF;
        margin-top: 10px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Custom button styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF763C 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 25px;
        box-shadow: 0 4px 14px rgba(255, 75, 75, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #FF763C 0%, #FF4B4B 100%);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.6);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to fetch movie poster using TMDB API
def fetch_poster(movie_id, api_key="8265a56fc4d9e23d52421ed18001da20"):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        data = requests.get(url, timeout=5).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except Exception:
        pass
    # Fallback placeholder poster image
    return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=60"

# Load movies and similarity matrix
@st.cache_resource
def load_data():
    if not os.path.exists('movies_dict.pkl') or not os.path.exists('similarity.pkl'):
        st.error("Data files not found! Running data prep script...")
        import prepare_data
    
    with open('movies_dict.pkl', 'rb') as f:
        movies_dict = pickle.load(f)
    with open('similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)
        
    movies_df = pd.DataFrame(movies_dict)
    return movies_df, similarity

movies, similarity = load_data()

# Recommendation logic
def recommend(movie_title, top_n=5):
    movie_index = movies[movies['title'] == movie_title].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]
    
    recommended_movies = []
    recommended_posters = []
    
    for i in movies_list:
        m_id = movies.iloc[i[0]]['movie_id']
        title = movies.iloc[i[0]]['title']
        recommended_movies.append(title)
        recommended_posters.append(fetch_poster(m_id))
        
    return recommended_movies, recommended_posters

# App Header
st.markdown('<div class="main-header">🎬 CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Discover your next favorite movie powered by Content-Based Machine Learning</div>', unsafe_allow_html=True)

st.markdown("---")

# Main interface layout
col1, col2 = st.columns([3, 1])

with col1:
    selected_movie_name = st.selectbox(
        "🔍 Select or type a movie you like:",
        movies['title'].values,
        index=int(movies[movies['title'] == 'Spider-Man'].index[0]) if 'Spider-Man' in movies['title'].values else 0
    )

with col2:
    num_recs = st.slider("Number of recommendations:", min_value=3, max_value=10, value=5)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Get Recommendations"):
    with st.spinner("Finding similar movies for you..."):
        names, posters = recommend(selected_movie_name, top_n=num_recs)
    
    st.markdown(f"### 🍿 Top Recommendations for *{selected_movie_name}*:")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display recommendations in responsive grid columns
    cols = st.columns(num_recs)
    for idx, col in enumerate(cols):
        with col:
            st.markdown(f"""
            <div class="movie-card">
                <img src="{posters[idx]}" style="width:100%; border-radius:8px; aspect-ratio: 2/3; object-fit: cover;">
                <div class="movie-title">{names[idx]}</div>
            </div>
            """, unsafe_allow_html=True)

# Footer / Sidebar info
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&auto=format&fit=crop&q=80", use_column_width=True)
    st.header("About CineMatch")
    st.write("""
    This app uses **Cosine Similarity** on high-dimensional text vectors generated from:
    - **Movie Overviews**
    - **Genres & Keywords**
    - **Top Cast Members**
    - **Directors**
    
    Dataset based on TMDB 5000 movies.
    """)
    st.markdown("---")
    st.caption("Built with Streamlit & Python 🐍")
