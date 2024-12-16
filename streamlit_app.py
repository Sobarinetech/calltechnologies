import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import tempfile
import os

# Function to adjust pauses between words
def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    """
    Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.
    """
    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]
        pause_duration = next_start - current_end

        if pause_duration > 0:
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

            # Adjust current chunk end time
            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

            # Adjust next chunk start time
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
    
    pipeline_output["chunks"] = adjusted_chunks

    return pipeline_output

# Streamlit app function
def main():
    st.title("CrisperWhisper ASR Web App")

    # Upload audio file
    uploaded_file = st.file_uploader("Upload Audio File (WAV/MP3/FLAC)", type=["wav", "mp3", "flac"])
    
    if uploaded_file is not None:
        # Show the audio file for playback
        st.audio(uploaded_file, format="audio/wav")

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Run transcription using Hugging Face ASR pipeline
        st.info("Processing audio for transcription...")
        transcription = process_audio_with_hf(tmp_file_path)
        if transcription:
            st.success("Transcription Complete!")
            st.header("Transcription with Word Timestamps")
            # Display the result
            display_transcription(transcription)

        # Clean up temporary file
        os.remove(tmp_file_path)

# Function to process audio with Hugging Face's pipeline
def process_audio_with_hf(audio_file_path):
    # Set device and dtype for CPU (since no GPU is available)
    device = "cpu"
    torch_dtype = torch.float32

    model_id = "nyrahealth/CrisperWhisper"  # Hugging Face model ID for CrisperWhisper

    # Load the model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # Setup pipeline for ASR
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,  # Process audio in 30-second chunks
        batch_size=16,
        return_timestamps='word',  # Return word-level timestamps
        torch_dtype=torch_dtype,
        device=device,
    )

    # Load the audio file and process it
    hf_pipeline_output = pipe(audio_file_path)

    # Adjust pauses to improve transcription naturalness
    crisper_whisper_result = adjust_pauses_for_hf_pipeline_output(hf_pipeline_output)

    return crisper_whisper_result

# Function to display transcription with timestamps
def display_transcription(transcription):
    # Display each chunk with timestamps and text
    for chunk in transcription["chunks"]:
        start_time, end_time = chunk["timestamp"]
        word = chunk["text"]
        st.write(f"**{word}** (Start: {start_time:.2f}s, End: {end_time:.2f}s)")

if __name__ == "__main__":
    main()
