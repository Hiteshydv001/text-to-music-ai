import logging
from typing import Tuple, Optional
import gradio as gr
import librosa
import numpy as np
from inference import generate_music

logger = logging.getLogger(__name__)

def generate_and_play(text_prompt: str) -> Tuple[Optional[str], Optional[Tuple[int, np.ndarray]]]:
    """Generate music and prepare for Gradio output."""
    try:
        output_path = generate_music(text_prompt)
        if not output_path:
            return None, None
        
        audio, sr = librosa.load(output_path, sr=None)
        waveform = (sr, audio)
        logger.info(f"Prepared audio for Gradio: {output_path}")
        return output_path, waveform
    except Exception as e:
        logger.error(f"Gradio generation failed: {e}")
        return None, None

with gr.Blocks(title="Text-to-Music Generator") as demo:
    gr.Markdown("# Text-to-Music Generator")
    gr.Markdown("Enter a description to generate music!")
    
    with gr.Row():
        text_input = gr.Textbox(label="Music Description", placeholder="e.g., 'calm piano melody'")
        submit_btn = gr.Button("Generate")
    
    with gr.Row():
        audio_output = gr.Audio(label="Generated Music", type="filepath")
        waveform_output = gr.Plot(label="Waveform")
    
    with gr.Row():
        download_btn = gr.File(label="Download WAV")
    
    submit_btn.click(generate_and_play, inputs=text_input, outputs=[audio_output, waveform_output])
    audio_output.change(lambda x: x, inputs=audio_output, outputs=download_btn)

if __name__ == "__main__":
    demo.launch()