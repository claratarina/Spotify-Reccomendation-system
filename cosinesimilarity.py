import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
from spotipy.oauth2 import SpotifyOAuth

# Load songs data from CSV file, this is songs from the merged-list.
# Our train data
songs_data = pd.read_csv('playlist_data.csv')

# Get user input for tempo
user_tempo = float(input("Enter desired tempo: "))

# Filter songs based on user input tempo, with a +-10 range
filtered_songs = songs_data[(songs_data['tempo'] >= user_tempo - 10) & (songs_data['tempo'] <= user_tempo + 10)]

# Features for comparison
#First version with PCA features
selected_song_features = filtered_songs[['tempo', 'key', 'loudness']].values
#Second version with DVS features
#selected_song_features = filtered_songs[['danceability', 'valence', 'speechiness']].values

# Fetch Spotify recommended songs using spotipy
scope ='playlist-modify-private'
redirect_uri='https://www.google.com/search?q=%C3%B6vers%C3%A4tt&rlz=1C1GTPM_enSE912SE912&oq=%C3%B6vers%C3%A4tt&gs_lcrp=EgZjaHJvbWUqDggAEEUYJxg7GIAEGIoFMg4IABBFGCcYOxiABBiKBTIGCAEQIxgnMg0IAhAAGIMBGLEDGIAEMg0IAxAAGIMBGLEDGIAEMgcIBBAAGIAEMg0IBRAAGIMBGLEDGIAEMgYIBhBFGD0yBggHEEUYPdIBCDExMTlqMGo5qAIAsAIA&sourceid=chrome&ie=UTF-8'
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='f37ea1a1d5a54cc78cf7ef5899edc292',
                                               client_secret='60d57c2b53e34cbbab7cce785d8d264f',
                                               redirect_uri=redirect_uri,
                                               scope=scope))

# Read csv file, the test data. The list is from spotify 1.2M songs
test_playlist_data = pd.read_csv('tracks_features.csv')
#First version with PCa features
recommended_song_features = test_playlist_data[['tempo', 'key', 'loudness']].values
#Second version with DVS features
#recommended_song_features = test_playlist_data[['danceability', 'valence', 'speechiness']].values


# Convert recommended_song_features to a numpy array
recommended_song_features = np.array(recommended_song_features)
print(recommended_song_features)

# Calculate similarities between selected songs and recommended songs after PCA transformation
similarities = cosine_similarity(selected_song_features, recommended_song_features)
print(similarities)

top_similar_indices = similarities.argsort()[0][-100:][::-1] 

# Filter recommended songs based on BPM
filtered_recommendations = []
for idx in top_similar_indices:
    # Fetch song information from test_playlist_data using indices
    tempo = test_playlist_data.iloc[idx]['tempo']
    # Check if tempo falls within the desired range
    if user_tempo - 10 <= tempo <= user_tempo + 10:
        # Append song details to filtered_recommendations
        filtered_recommendations.append(test_playlist_data.iloc[idx])

# Display filtered recommended songs
for song in filtered_recommendations:
     print(f"{song['name']} - {song['artists']} | Tempo: {song['tempo']} BPM")

#get current user 
user_info = sp.current_user()
user_id = user_info['id']

#the playlist name in the current users spotify
playlist_name = "Cosine 160BPM- T,K,L"
playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=False)

# Add tracks to the playlist
track_uris = [track['id'] for track in filtered_recommendations]  # Assuming filtered_recommendations contains Spotify track URIs

sp.playlist_add_items(playlist_id=playlist['id'], items=track_uris)