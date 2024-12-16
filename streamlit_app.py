import streamlit as st
from huggingface_hub import login
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

# Login to Hugging Face using your token (replace 'your_huggingface_token' with the actual token)
login(token="your_huggingface_token")

# Initialize the model and processor from Hugging Face
pipe = pipeline("automatic-speech-recognition", model="nyrahealth/CrisperWhisper")

processor = AutoProcessor.from_pretrained("nyrahealth/CrisperWhisper")
model = AutoModelForSpeechSeq2Seq.from_pretrained("nyrahealth/CrisperWhisper")

# Streamlit UI Setup
st.title("CrisperWhisper Speech Recognition")
st.write("Upload an audio file and get a verbatim transcription with precise word-level timestamps.")

# Audio file upload
audio_file = st.file_uploader("Choose a .wav audio file", type=["wav"])

if audio_file is not None:
    # Save the uploaded audio file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())
    
    st.audio(audio_file, format="audio/wav")

    # Process the audio file
    st.write("Processing the audio...")

    # Load the audio file for transcription
    speech_input = processor("temp_audio.wav", return_tensors="pt", sampling_rate=16000)

    # Get the prediction (the transcribed text)
    with torch.no_grad():
        transcription = model.generate(**speech_input)

    # Decode the transcription and show the result
    transcribed_text = processor.decode(transcription[0], skip_special_tokens=True)

    st.subheader("Transcribed Text")
    st.write(transcribed_text)
