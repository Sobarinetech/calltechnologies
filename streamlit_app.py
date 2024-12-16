import streamlit as st
import requests

# Get the Hugging Face API token from Streamlit secrets
API_TOKEN = st.secrets["hf_token"]
API_URL = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to query the Hugging Face model for transcription
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

# Streamlit app
def main():
    st.title("English Speech-to-Text with Wav2Vec2")

    # Upload audio file (WAV, MP3, or FLAC formats)
    uploaded_file = st.file_uploader("Upload Audio (WAV/MP3/FLAC)", type=["wav", "mp3", "flac"])

    if uploaded_file:
        # Display audio player for the uploaded file
        st.audio(uploaded_file, format="audio/wav")

        # Save the uploaded file temporarily to pass to the API
        temp_file_path = "temp_audio_file" + uploaded_file.name
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Call the Wav2Vec2 model to transcribe the audio
        st.info("Processing your audio...")
        result = query(temp_file_path)

        # Check if the response contains valid results
        if "error" in result:
            st.error("There was an error processing the audio: " + result["error"])
        else:
            # Display transcribed text
            st.success("Transcription complete!")
            st.header("Transcribed Text")
            st.write(result.get("text", "No transcription available"))

if __name__ == "__main__":
    main()
