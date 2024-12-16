import streamlit as st
import requests

# Hugging Face API URL and headers with authorization
API_URL = "https://api-inference.huggingface.co/models/pyannote/speaker-diarization-3.1"
API_TOKEN = st.secrets["hf_token"]  # Use the Hugging Face token stored in Streamlit secrets
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to query the Hugging Face API for speaker diarization
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

# Streamlit app
def main():
    st.title("Speaker Diarization with pyannote")

    # Upload audio file (WAV, MP3, FLAC formats)
    uploaded_file = st.file_uploader("Upload Audio (WAV/MP3/FLAC)", type=["wav", "mp3", "flac"])

    if uploaded_file:
        # Display the audio file for playback
        st.audio(uploaded_file, format="audio/wav")

        # Save the uploaded file temporarily to pass to the API
        temp_file_path = "temp_audio_file" + uploaded_file.name
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Call the speaker diarization API
        st.info("Processing your audio for speaker diarization...")
        result = query(temp_file_path)

        # Check if the response contains valid results
        if "error" in result:
            st.error("There was an error processing the audio: " + result["error"])
        else:
            # Display diarization segments
            st.success("Speaker diarization complete!")
            st.header("Diarization Segments")
            if "segments" in result:
                for segment in result["segments"]:
                    start_time = segment.get("start", "N/A")
                    end_time = segment.get("end", "N/A")
                    speaker = segment.get("speaker", "Unknown")
                    st.write(f"**Speaker {speaker}:** {start_time}s - {end_time}s")
            else:
                st.error("No diarization segments found in the result.")

if __name__ == "__main__":
    main()
