import streamlit as st
from speechbrain.pretrained import SpeakerDiarization
import os

# Set up the SpeechBrain diarization model
@st.cache_resource
def load_diarization_model():
    return SpeakerDiarization.from_hparams(
        source="speechbrain/speaker-diarization",
        savedir="tmp_model_dir"
    )

diarization_model = load_diarization_model()

# Streamlit UI
st.title("🔊 Speaker Diarization Web App")
st.write("Upload an audio file, and this app will perform speaker diarization using SpeechBrain.")

# File uploader
uploaded_file = st.file_uploader("Upload your audio file (e.g., .wav, .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    with open("temp_audio_file.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file, format="audio/wav")
    st.info("Processing audio... This may take a moment.")

    # Perform diarization
    try:
        diarization_output = diarization_model.diarize_file("temp_audio_file.wav")

        # Display diarization results
        st.success("Diarization Complete!")
        st.write("Speaker Diarization Results:")
        st.json(diarization_output["content"])  # Display raw diarization output
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

    # Clean up temporary file
    os.remove("temp_audio_file.wav")
