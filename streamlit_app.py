import streamlit as st
import librosa
import whisper
import spacy
from textblob import TextBlob
from pyannote.audio import Pipeline

# Load Whisper model
model = whisper.load_model("base")

# Load NLP models
nlp = spacy.load("en_core_web_sm")

# Load Pyannote speaker diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

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
    transcription_result = model.transcribe(audio)
    text = transcription_result["text"]
    
    # Display Transcript
    st.write("Transcript:")
    st.write(text)

    # NLP Analysis using TextBlob
    blob = TextBlob(text)
    
    # Display Summary
    st.write("Summary:")
    st.write(blob.sentences[0])

    # Display Sentiment
    st.write("Sentiment:")
    st.write(blob.sentiment.polarity)

    # Display Entities
    st.write("Entities:")
    for entity in blob.noun_phrases:
        st.write(entity)

    # Display Tokens and POS
    st.write("Tokens and POS:")
    for sentence in blob.sentences:
        for word, pos in sentence.tags:
            st.write(f"{word} ({pos})")

    # Diarization using pyannote.audio
    diarization = pipeline(uploaded_file.name)
    
    # Display Diarization Segments
    st.write("Diarization Segments:")
    for segment in diarization.itertracks(yield_label=True):
        start_time, end_time, speaker = segment
        st.write(f"Speaker {speaker}: {start_time}s - {end_time}s")

# Add any other analyses you wish to include...
