from spotify_api import get_track_info
from preprocessing import preprocess_data_final
from harmonyseeker import predict_pop


def main():
    track_name = input("Will this track be a Success? ")
    track_features = get_track_info(track_name=track_name)
    print("Finding the track...")
    print("Track Features:\n", track_features)
    if track_features:
        processed_data = preprocess_data_final(track_features)
        print("Doing Data Mining, keep calm...")
        print("Track Data Preprocessed")
        print("Starting Predictions...")
        prediction = predict_pop(processed_data)
        if prediction == 1:
            print("It's a success")
        else:
            print("The key of success is in practice, keep working a bit")

    else:
        print("Track not found or an error occurred.")


if __name__ == "__main__":
    main()
