# Import necessary libraries
import streamlit as st
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BartForConditionalGeneration, BartTokenizer

# Load Hugging Face's pre-trained models and processors
@st.cache_resource
def load_models():
    # Load Wav2Vec2 model for speech recognition
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

    # Load BART model for summarization
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    
    return processor, model, tokenizer, summarization_model

# Function to convert audio to text using Wav2Vec2 model
def transcribe_audio(audio_data):
    processor, model, _, _ = load_models()

    # Prepare the audio for the model
    input_values = processor(audio_data, sampling_rate=16000, return_tensors="pt", padding="longest").input_values

    # Generate logits
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the prediction to get the transcription
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

# Function to summarize text using BART model
def summarize_text(text):
    tokenizer, summarization_model = load_models()[2:]

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding="longest")
    
    with torch.no_grad():
        summary_ids = summarization_model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Streamlit app UI setup
st.set_page_config(page_title="Audio Summarization App", page_icon="🎤", layout="wide")

# Custom CSS for appearance
st.markdown("""
    <style>
    .title {
        font-size: 2.5em;
        text-align: center;
        color: #4CAF50;
    }
    .header {
        font-size: 1.5em;
        color: #333;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 0.8em;
        color: #888;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Audio Summarization Application", anchor="app")
st.write("This application allows you to upload an audio file and get a summarized transcription of the content. The accuracy of the transcription and summarization depends on the clarity of the voice in the audio.")

# Option to upload an audio file
uploaded_file = st.file_uploader("Upload your audio file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")  # Play the uploaded audio file
    st.write("Transcribing audio...")

    # Read the audio file with librosa
    audio, sr = librosa.load(uploaded_file, sr=16000)
    audio = audio.astype(np.float32)

    # Transcribe the uploaded audio file
    transcription = transcribe_audio(audio)
    st.write("**Transcription:**")
    st.success(transcription)

    # Summarize the transcribed text
    st.write("Summarizing the transcription...")
    summary = summarize_text(transcription)
    st.write("**Summary:**")
    st.success(summary)

    # Note about accuracy
    st.warning("Note: The accuracy of the transcription and summarization depends on the clarity of the voice in the audio.")
else:
    st.write("Please upload a .wav or .mp3 audio file for transcription and summarization.")

# Add footer
st.markdown("<div class='footer'>Created by Hossam Hussein Gharib</div>", unsafe_allow_html=True)
