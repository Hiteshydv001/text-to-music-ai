# Gradio UI implementation
import gradio as gr
from inference import generate_music
import librosa
import numpy as np

def generate_and_play(text_prompt):
    # Generate music
    output_path = generate_music(text_prompt)
    
    # Load audio for waveform visualization
    audio, sr = librosa.load(output_path)
    waveform = audio  # Raw samples for visualization
    
    # Return audio file and waveform
    return output_path, (sr, waveform)

# Gradio interface
with gr.Blocks(title="Text-to-Music Generator") as demo:
    gr.Markdown("# Text-to-Music Generator")
    gr.Markdown("Enter a description to generate music!")
    
    with gr.Row():
        text_input = gr.Textbox(label="Music Description", placeholder="e.g., 'calm piano melody with soft strings'")
        submit_btn = gr.Button("Generate")
    
    with gr.Row():
        audio_output = gr.Audio(label="Generated Music")
        waveform_output = gr.Plot(label="Waveform")
    
    with gr.Row():
        download_btn = gr.File(label="Download WAV")
    
    # Connect inputs/outputs
    submit_btn.click(
        fn=generate_and_play,
        inputs=text_input,
        outputs=[audio_output, waveform_output]
    )
    audio_output.change(
        fn=lambda x: x,
        inputs=audio_output,
        outputs=download_btn
    )

demo.launch()