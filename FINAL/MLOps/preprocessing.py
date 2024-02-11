import re
import pandas as pd
import numpy as np

# internal
from joblib import load

sklearn_pca_final = load('PCAmodel_audiofeatures.joblib')

mapping = {
    "pop": ['pop', 'pop punk', 'dance pop', 'europop', 'girl group', 'pop rap', 'bahamian pop', 'canadian pop', 'pop dance', 'post-teen pop', 'boy band', 'art pop', 'metropopolis', 'barbadian pop', 'viral pop', 'candy pop', 'australian pop', 'folk-pop', 'k-pop', 'bubblegum dance', 'pop dance'],
    "rock": ['permanent wave', 'alternative metal', 'modern rock', 'punk', 'rock', 'socal pop punk', 'alternative rock', 'dance rock', 'pop rock', 'celtic rock', 'irish rock', 'grunge', 'uk pop', 'neo mellow', 'piano rock', 'pov: indie', 'modern alternative rock', 'neon pop punk', 'garage rock', 'indie rock', 'canadian rock', 'british invasion', 'emo', 'neo-psychedelic', 'screamo', 'glam metal', 'beatlesque', 'madchester', 'supergroup'],
    "country": ['contemporary country', 'country', 'country dawn', 'country road'],
    "hip_hop_rap": ['detroit hip hop', 'hip hop', 'rap', 'dirty south rap', 'hip pop', 'east coast hip hop', 'hardcore hip hop', 'gangster rap', 'west coast rap', 'st louis rap', 'trap', 'southern hip hop', 'atl hip hop', 'melodic rap', 'miami hip hop', 'chicago rap', 'canadian hip hop', 'old school atlanta hip hop', 'queens hip hop', 'crunk', 'conscious hip hop', 'trap queen', 'south carolina hip hop', 'new orleans rap', 'G Funk', 'brooklyn drill', 'hyphy', 'atlanta bass'],
    "rnb_soul": ['contemporary r&b', 'r&b', 'urban contemporary', 'british soul', 'neo soul', 'pop soul', 'canadian contemporary r&b', 'soul', 'pluggnb'],
    "house_and_pop_fusion": ['disco house', 'filter house', 'bouncy house', 'electro house', 'slap house', 'tropical house', 'pop dance', 'house', 'progressive house', 'indietronica', 'progressive electro house', 'uk garage', 'big room', 'complextro'],
    "underground_electronic": ['electro', 'big beat', 'downtempo', 'melbourne bounce international', 'electro trash', 'grave wave', 'dark clubbing', 'ukg revival', 'hardcore techno', 'grime', 'instrumental grime', 'dancefloor dnb', 'trance', 'speed garage', 'new wave', 'bounce', 'handsup', 'chicago hardcore', 'rave', 'german techno', 'hamburg electronic', 'happy hardcore', 'industrial', 'techno'],
    "metal": ['nu metal', 'post-grunge', 'rap metal', 'funk metal', 'industrial metal'],
    "jazz": ['jazz', 'smooth jazz', 'bebop', 'swing', 'fusion jazz', 'stomp and holler'],
    "classical": ['classical', 'baroque', 'romantic', 'symphony', 'opera'],
    "reggae": ['reggae', 'dancehall', 'reggae fusion', 'dub', 'roots reggae', 'moombahton'],
    "latin": ['latin', 'reggaeton', 'latin pop', 'salsa', 'bachata', 'merengue', 'tango', 'cumbia', 'trap latino', 'urbano latino', 'mexican pop', 'colombian pop', 'mambo chileno', 'urbano chileno', 'tropical'],
    "singer-songwriter": ['singer-songwriter pop', 'singer-songwriter'],
    "folk": ['folk', 'classic schlager', 'oktoberfest', 'schlager', 'celtic', 'middle earth', 'sertanejo', 'sertanejo universitario'],
    "talent show": ['talent show']
}

# Flatten the mapping for easier access
flat_mapping = {}
for main_genre, subgenres in mapping.items():
    for subgenre in subgenres:
        flat_mapping[subgenre] = main_genre

# When a "new" sub genre is being added with a version of the artist_genre


def categorize_genres(genres_list, mapping):
    genre_counts = {}
    for genre in genres_list:
        for subgenre, main_genre in mapping.items():
            if re.search(r'\b' + re.escape(subgenre) + r'\b', genre, flags=re.IGNORECASE):
                genre_counts[main_genre] = genre_counts.get(main_genre, 0) + 1
    if genre_counts:
        # Return the most frequent main genre, if tie, sorted order decides
        return max(genre_counts, key=genre_counts.get)
    else:
        return "other"  # or "other" if you prefer to label unmatched genres as "other"

# Function to categorize genres


def categorize_genres(genres_list, mapping):
    genre_counts = {}
    for genre in genres_list:
        for subgenre, main_genre in mapping.items():
            if subgenre in genre:
                genre_counts[main_genre] = genre_counts.get(main_genre, 0) + 1
    if genre_counts:
        # Return the most frequent main genre, if tie, sorted order decides
        return max(genre_counts, key=genre_counts.get)
    else:
        return "other"  # or "other" if you prefer to label unmatched genres as "other"


def preprocess_data_final(track_data):
    # Convert track_data into a DataFrame
    # Ensure it's a list of dict if track_data is a single dict
    tdata = pd.DataFrame(track_data)
    tdata = tdata.head(1)

    # Convert 'year' to 'track_age_2024'
    tdata['track_age_2024'] = 2024 - \
        pd.to_numeric(tdata['year'], errors='coerce')

    # Apply genre categorization
    tdata['dominant_genre'] = tdata['artist_genres'].apply(
        lambda x: categorize_genres(x, flat_mapping))

    # Map 'key' and 'mode' to their string representations
    key_mapping = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E',
                   5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
    mode_mapping = {0: 'min', 1: 'Maj'}
    tdata['key'] = tdata['key'].map(key_mapping)
    tdata['mode'] = tdata['mode'].map(mode_mapping)
    tdata['tonality'] = tdata['key'] + "_" + tdata['mode']

    # Drop columns not used in PCA before one-hot encoding to prevent data leakage
    tdata.drop(columns=['year', 'key', 'mode', 'track_name', 'album',
               'artist_name', 'artist_genres', 'track_popularity'], inplace=True)

    # One-hot encode categorical features
    tdata = pd.get_dummies(tdata, columns=['dominant_genre', 'tonality'], prefix=[
                           'genre', 'tone'], drop_first=True)

    # Prepare audio features for PCA
    audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                      'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
    pca_components = sklearn_pca_final.transform(tdata[audio_features])

    # Add PCA components to DataFrame
    pca_feature_names = ['PC1_Energetic_Dynamic', 'PC2_Speech_Tempo',
                         'PC3_Acoustic_Speech', 'PC4_Duration_Mood', 'PC5_Acoustic_Liveness']
    for i, name in enumerate(pca_feature_names):
        tdata[name] = pca_components[:, i]

    # Drop original audio features after PCA transformation
    tdata.drop(columns=audio_features, inplace=True)

    # Ensure the order of columns matches the training dataset
    expected_features = ['track_age_2024', 'genre_country', 'genre_folk', 'genre_hip_hop_rap', 'genre_house_and_pop_fusion', 'genre_jazz', 'genre_latin', 'genre_metal', 'genre_other', 'genre_pop', 'genre_reggae', 'genre_rnb_soul', 'genre_rock', 'genre_singer-songwriter', 'genre_underground_electronic'] + [f'tone_{tone}' for tone in [
        'A#min', 'AMaj', 'Amin', 'BMaj', 'Bmin', 'C#Maj', 'C#min', 'CMaj', 'Cmin', 'D#Maj', 'D#min', 'DMaj', 'Dmin', 'EMaj', 'Emin', 'F#Maj', 'F#min', 'FMaj', 'Fmin', 'G#Maj', 'G#min', 'GMaj', 'Gmin']] + ['PC1_Energetic_Dynamic', 'PC2_Speech_Tempo', 'PC3_Acoustic_Speech', 'PC4_Duration_Mood ', 'PC5_Acoustic_Liveness']

    # Reorder and ensure all expected columns are present
    for feature in expected_features:
        if feature not in tdata.columns:
            tdata[feature] = 0  # Add missing features as zeros
    tdata = tdata[expected_features]

    return tdata
