import gradio as gr
from speech2text.speech2text import transcript_audio
from llm_analyzer.llm_analyzer import text_analyzer

# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio_file(audio_file):
  audio_text = transcript_audio(audio_file)
  return text_analyzer(audio_text)

# Set up Gradio interface
audio_input = gr.Audio(sources="upload", type="filepath")  # Audio input
output_text = gr.Textbox()  # Text output

# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(fn=transcript_audio_file, 
                     inputs=audio_input, outputs=output_text, 
                     title="Audio Transcription App",
                     description="Upload the audio file")

# Launch the Gradio app
iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
