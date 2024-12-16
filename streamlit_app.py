import streamlit as st
import librosa
import speech_recognition as sr
import spacy
from textblob import TextBlob
from pyannote.audio import Pipeline
import requests

# Set up Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/pyannote/speaker-diarization"
API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Streamlit App UI
st.title("🎙️ Customer Support Call Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your audio file (e.g., .wav, .flac, .mp3)", type=["wav", "flac", "mp3"])

if uploaded_file is not None:
    # Load audio file
    audio, sample_rate = librosa.load(uploaded_file, sr=None)
    
    # Display audio player
    st.audio(uploaded_file, format="audio/wav")

    # Diarization using Hugging Face API
    def diarize_audio(file):
        try:
            data = file.read()
            response = requests.post(API_URL, headers=HEADERS, data=data)
            if response.status_code == 200:
                return response.json()  # Return diarization
            else:
                return {"error": f"API Error: {response.status_code} - {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    diarization_result = diarize_audio(uploaded_file)
    if "segments" in diarization_result:
        st.success("Diarization Complete")
        diarization_segments = diarization_result["segments"]
    elif "error" in diarization_result:
        st.error(f"Error: {diarization_result['error']}")
    else:
        st.warning("Unexpected response from the API")

    # Convert audio to text using SpeechRecognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(uploaded_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = "Google Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        text = f"Could not request results from Google Speech Recognition service; {e}"

    # Display Transcript
    st.write("Transcript:")
    st.write(text)

    if text:
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

    # Display diarization segments
    st.write("Diarization Segments:")
    for segment in diarization_segments:
        start_time = segment['start_time']
        end_time = segment['end_time']
        speaker = segment['speaker']
        st.write(f"Speaker {speaker}: {start_time}s - {end_time}s")

    # Add any other analyses you wish to include...
