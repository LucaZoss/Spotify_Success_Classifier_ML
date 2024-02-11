import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd


def get_track_info(track_name):
    # Set up Spotipy client with your Spotify API credentials
    client_credentials_manager = SpotifyClientCredentials(client_id='89357b1de85a4b758d8bcaed6974a7c5',
                                                          client_secret='d764e749f046434ead473dcd01fb116e')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Search for the track
    results = sp.search(q=track_name, type='track', limit=1)
    if not results['tracks']['items']:
        return 'Track not found'

    track = results['tracks']['items'][0]

    # Retrieve audio features of the track
    audio_features = sp.audio_features(track['id'])[0]

    # Retrieve artist information
    artist = sp.artist(track['artists'][0]['id'])

    # Compile the requested information
    track_info = {
        'year': track['album']['release_date'][:4],
        'track_name': track['name'],
        'track_popularity': track['popularity'],
        'album': track['album']['name'],
        'artist_name': track['artists'][0]['name'],
        'artist_genres': artist['genres'],
        'artist_popularity': artist['popularity'],
        'danceability': audio_features['danceability'],
        'energy': audio_features['energy'],
        'key': audio_features['key'],
        'loudness': audio_features['loudness'],
        'mode': audio_features['mode'],
        'speechiness': audio_features['speechiness'],
        'acousticness': audio_features['acousticness'],
        'instrumentalness': audio_features['instrumentalness'],
        'liveness': audio_features['liveness'],
        'valence': audio_features['valence'],
        'tempo': audio_features['tempo'],
        'duration_ms': audio_features['duration_ms'],
    }

    return track_info
