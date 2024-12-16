import streamlit as st
import requests

# Get the Hugging Face API token from Streamlit secrets
API_TOKEN = st.secrets["hf_token"]
API_URL = "https://api-inference.huggingface.co/models/nyrahealth/CrisperWhisper"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to query the Hugging Face model for transcription and diarization
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()

    # Add the return_timestamps=True parameter for long audio inputs
    params = {
        "return_timestamps": "true"
    }
    
    response = requests.post(API_URL, headers=headers, data=data, params=params)
    return response.json()

# Streamlit app
def main():
    st.title("Speaker Diarization and Transcription with CrisperWhisper")

    # Upload audio file (WAV, MP3, or FLAC formats)
    uploaded_file = st.file_uploader("Upload Call Recording (WAV/MP3/FLAC)", type=["wav", "mp3", "flac"])

    if uploaded_file:
        # Display audio player for the uploaded file
        st.audio(uploaded_file, format="audio/wav")

        # Save the uploaded file temporarily to pass to the API
        temp_file_path = "temp_audio_file" + uploaded_file.name
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Call the CrisperWhisper model to transcribe and diarize the audio
        st.info("Processing your audio...")
        result = query(temp_file_path)

        # Check if the response contains valid results
        if "error" in result:
            st.error("There was an error processing the audio: " + result["error"])
        else:
            # Display transcribed text and diarization results
            st.success("Transcription and diarization complete!")

            st.header("Transcription with Speaker Labels")
            for segment in result.get("segments", []):
                speaker_name = segment.get("speaker", "Unknown")
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "")
                st.write(f"**{speaker_name}:** {text} (from {start_time}s to {end_time}s)")

if __name__ == "__main__":
    main()
