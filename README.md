
# Audio Summarization Application 🎤


Overview
The Audio Summarization Application is a web-based tool that allows users to upload audio files and receive a summarized transcription of the content. Built using Streamlit and Hugging Face's state-of-the-art models, this application harnesses the power of speech recognition and natural language processing to provide quick and efficient summaries.

Whether you're a student looking to summarize lectures, a professional needing to condense meeting notes, or anyone who wishes to save time by converting long audio recordings into concise text, this application is designed for you!


## Features

- Audio Upload: Easily upload .wav or .mp3 audio files.
- Speech Recognition: Converts spoken language into written text using Wav2Vec2 model.
- Text Summarization: Summarizes transcribed text with the BART model for concise output.
- User-Friendly Interface: An intuitive design powered by Streamlit for a seamless experience.
- Performance Note: Accuracy depends on the clarity of the audio input.


## Technologies Used

- Streamlit: For building the web application interface.
- Hugging Face Transformers: For implementing state-of-the-art models for speech recognition and text summarization.
- Librosa: For audio processing and manipulation.
- Torch: For running the deep learning models.
## Code Explanation
#### 1. Loading Models
The models for speech recognition (Wav2Vec2) and text summarization (BART) are loaded using Hugging Face Transformers. Caching is implemented to speed up subsequent loads.

```
@st.cache_resource
def load_models():
    # Load Wav2Vec2 model for speech recognition
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

    # Load BART model for summarization
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    
    return processor, model, tokenizer, summarization_model
```
#### 2. Audio Processing
Audio files are uploaded and processed using librosa to convert them into a format suitable for the models. The audio is resampled to 16 kHz to match the model's input requirements.

```
audio, sr = librosa.load(uploaded_file, sr=16000)
audio = audio.astype(np.float32)
```
#### 3. Transcription
The audio is passed through the Wav2Vec2 model to generate the transcription. The model predicts the most likely sequence of words based on the input audio.

```
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
```
### 4. Summarization
The transcribed text is then summarized using the BART model. The summarization is tuned to produce concise output while retaining the essential information.

```
def summarize_text(text):
    tokenizer, summarization_model = load_models()[2:]

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding="longest")
    
    with torch.no_grad():
        summary_ids = summarization_model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
```
### 5. Streamlit Interface
The user interface is built using Streamlit, allowing for easy interactions, such as file uploads and displaying the results.

```
# Streamlit app UI setup
st.set_page_config(page_title="Audio Summarization App", page_icon="🎤", layout="wide")
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
```





## Contact
For any inquiries or feedback, please reach out to:

Hossam Hussein Gharib

Email: hossamhusein83@gmail.com

linkedin: www.linkedin.com/in/hossam-husein-ba252b246