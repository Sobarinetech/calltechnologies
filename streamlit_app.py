import streamlit as st
import requests
from pydub import AudioSegment

# Hugging Face API URLs and Headers
TRANSCRIPTION_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large"
DIARIZATION_API_URL = "https://api-inference.huggingface.co/models/pyannote/speaker-diarization"
HEADERS = {"Authorization": "Bearer hf_your_huggingface_token"}

# Function to upload audio for transcription
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio:
        response = requests.post(TRANSCRIPTION_API_URL, headers=HEADERS, data=audio)
    return response.json()

# Function to upload audio for speaker diarization
def diarize_audio(file_path):
    with open(file_path, "rb") as audio:
        response = requests.post(DIARIZATION_API_URL, headers=HEADERS, data=audio)
    return response.json()

# Map speaker IDs to names
def map_speakers(diarization_result):
    speakers = {}
    for idx, segment in enumerate(diarization_result.get('segments', [])):
        speaker_id = segment['speaker']
        if speaker_id not in speakers:
            if len(speakers) == 0:
                speakers[speaker_id] = "Agent Jack"
            else:
                speakers[speaker_id] = "Mark"
    return speakers

# Streamlit UI
def main():
    st.title("Call Recording Transcription App with Speaker Labeling")
    st.write("Upload an audio file to transcribe and label speakers.")

    uploaded_file = st.file_uploader("Upload Call Recording (WAV/MP3)", type=["wav", "mp3"])

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        
        # Save the uploaded file temporarily
        temp_file = "temp_audio.wav"
        audio = AudioSegment.from_file(uploaded_file)
        audio.export(temp_file, format="wav")
        
        # Call transcription API
        st.info("Transcribing audio...")
        transcription = transcribe_audio(temp_file)
        st.success("Transcription complete!")

        # Call diarization API
        st.info("Performing speaker diarization...")
        diarization = diarize_audio(temp_file)
        st.success("Speaker diarization complete!")
        
        # Map speakers and display results
        speaker_map = map_speakers(diarization)
        st.header("Final Transcription with Speaker Labels")
        for segment in diarization['segments']:
            speaker_name = speaker_map.get(segment['speaker'], "Unknown")
            text = transcription.get('text', '')
            st.write(f"**{speaker_name}:** {text}")

if __name__ == "__main__":
    main()
