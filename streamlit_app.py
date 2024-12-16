import streamlit as st
import librosa
import whisper
from pyannote.audio import Pipeline

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load Pyannote speaker diarization pipeline
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Streamlit App UI
st.title("🎙️ Customer Support Call Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your audio file (e.g., .wav, .flac, .mp3)", type=["wav", "flac", "mp3"])

if uploaded_file is not None:
    # Load audio file
    audio, sample_rate = librosa.load(uploaded_file, sr=None)
    
    # Display audio player
    st.audio(uploaded_file, format="audio/wav")

    # Transcription using Whisper
    transcription_result = whisper_model.transcribe(audio)
    text = transcription_result["text"]
    
    # Display Transcript
    st.write("Transcript:")
    st.write(text)

    # Diarization using pyannote.audio
    diarization_result = diarization_pipeline(uploaded_file.name)
    
    # Display Diarization Segments
    st.write("Diarization Segments:")
    for segment in diarization_result.itertracks(yield_label=True):
        start_time, end_time, speaker = segment
        st.write(f"Speaker {speaker}: {start_time:.2f}s - {end_time:.2f}s")

# Add any other analyses you wish to include...
