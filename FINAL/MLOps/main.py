# Import necessary libraries
import streamlit as st
from spotify_api import get_track_info
from preprocessing import preprocess_data_final
from harmonyseeker import predict_pop

# Streamlit app main function


def main():
    st.title("Will this track be a Success?")

    # Replace input() with Streamlit's text input
    track_name = st.text_input("Enter the name of the track:")

    # Button to trigger prediction
    if st.button("Predict Success"):
        if track_name:
            with st.spinner("Finding the track..."):
                track_features = get_track_info(track_name=track_name)
                if track_features:
                    with st.expander("Show Track Features", expanded=False):
                        st.write(track_features)
                    processed_data = preprocess_data_final(track_features)
                    prediction = predict_pop(processed_data)
                    if prediction == 1:
                        st.success("SUCCESS !")
                        st.success(
                            "You will be a billionaire or you’re just another False Positive")
                    else:
                        st.warning("NOT A SUCCESS ...")
                        st.warning(
                            "Hope you’ll do better in the ML Exam or we missed the shot")
                else:
                    st.error("Track not found or an error occurred.")


if __name__ == "__main__":
    main()
