import streamlit as st
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import os
import wave

# Initialize the model and processor from Hugging Face
pipe = pipeline("automatic-speech-recognition", model="nyrahealth/CrisperWhisper")

processor = AutoProcessor.from_pretrained("nyrahealth/CrisperWhisper")
model = AutoModelForSpeechSeq2Seq.from_pretrained("nyrahealth/CrisperWhisper")

# Streamlit UI Setup
st.title("CrisperWhisper Speech Recognition")
st.write("Upload an audio file and get a verbatim transcription with precise word-level timestamps.")

# Audio file upload
audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    # Save the uploaded audio file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())
    
    st.audio(audio_file, format="audio/wav")

    # Process the audio file
    st.write("Processing the audio...")
    
    # Load and process the audio file using the CrisperWhisper model
    # Read audio file
    if audio_file.type == 'audio/wav':
        input_audio = "temp_audio.wav"
    else:
        # Convert mp3 to wav (you can extend this for other formats as well)
        import pydub
        sound = pydub.AudioSegment.from_file(audio_file)
        input_audio = "temp_audio.wav"
        sound.export(input_audio, format="wav")
    
    # Load the audio file for transcription
    speech_input = processor(input_audio, return_tensors="pt", sampling_rate=16000)
    
    # Get the prediction
    with torch.no_grad():
        transcription = model.generate(**speech_input)
    
    # Decode the transcription and show the result
    transcribed_text = processor.decode(transcription[0], skip_special_tokens=True)
    
    st.subheader("Transcribed Text")
    st.write(transcribed_text)
