import torch
from transformers import pipeline


# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
  # Initialize the speech-to-text pipeline from Hugging Face Transformers
  # This uses the "openai/whisper-tiny.en" model for automatic speech recognition (ASR)
  # The `chunk_length_s` parameter specifies the chunk length in seconds for processing
  pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30,
  )

  # Transcribe the audio file and return the result
  # The `batch_size=8` parameter indicates how many chunks are processed at a time
  # The result is stored in `transcript` with the key "text" containing the transcribed text
  transcript = pipe(audio_file, batch_size=8)["text"]

  return transcript
