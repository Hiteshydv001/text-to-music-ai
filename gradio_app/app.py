import logging
import torch
import torchaudio
import gradio as gr
import librosa
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from audiocraft.models import MusicGen

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MusicGenerator:
    """Optimized wrapper for MusicGen model."""
    def __init__(self, model_name: str = "facebook/musicgen-small", duration: int = 10):
        try:
            self.model = MusicGen.get_pretrained(model_name)
            self.model.set_generation_params(duration=duration)
            self.sample_rate = self.model.sample_rate
            logger.info(f"Loaded MusicGen model: {model_name} with duration {duration}s")
        except Exception as e:
            logger.error(f"Failed to initialize MusicGen: {e}")
            raise

    def generate(self, text_prompt: str) -> Optional[torch.Tensor]:
        """Generate audio from text prompt."""
        try:
            logger.info(f"Generating audio for prompt: {text_prompt}")
            audio_batch = self.model.generate([text_prompt], progress=True)

            if audio_batch.dim() == 3 and audio_batch.shape[0] == 1:  
                audio = audio_batch.squeeze(0)
            elif audio_batch.dim() == 3 and audio_batch.shape[0] > 1:
                logger.warning("Multiple samples generated, selecting the first one.")
                audio = audio_batch[0]  
            else:
                audio = audio_batch

            return audio
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

    def save_audio(self, audio: torch.Tensor, filepath: str = "output.wav") -> Optional[str]:
        """Save audio tensor to WAV file."""
        try:
            if audio is None:
                raise ValueError("No audio tensor provided")
            
            torchaudio.save(filepath, audio.cpu(), sample_rate=self.sample_rate)
            logger.info(f"Audio saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return None

def generate_music(text_prompt: str):
    """Generate and return audio file path & waveform plot."""
    generator = MusicGenerator()
    audio_path = "output.wav"
    
    audio = generator.generate(text_prompt)
    if audio is None:
        return None, None

    file_path = generator.save_audio(audio, audio_path)
    if file_path is None:
        return None, None

    y, sr = librosa.load(file_path, sr=None)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.linspace(0, len(y) / sr, num=len(y)), y, color='cyan', linewidth=1.5)
    ax.set_facecolor("#222222")
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='both', colors='white')
    ax.set_title("Generated Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    return file_path, fig

# Gradio Interface
with gr.Blocks(title="🎵 AI Music Generator", theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="gray")) as demo:
    gr.Markdown("""
        # 🎼 AI-Powered Music Generator
        Generate unique music from text descriptions using AI-powered MusicGen.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="🎵 Enter Music Description", placeholder="e.g., 'calm piano melody'", lines=2)
        with gr.Column(scale=1):
            submit_btn = gr.Button("🎼 Generate Music", elem_id="generate-btn")

    with gr.Row():
        with gr.Column(scale=2):
            audio_output = gr.Audio(label="🎧 Generated Music", type="filepath")
        with gr.Column(scale=1):
            waveform_output = gr.Plot(label="📊 Waveform")

    gr.Markdown("""
        Developed with ❤️ by Hitesh Kumar
    """)

    submit_btn.click(generate_music, inputs=text_input, outputs=[audio_output, waveform_output])

if __name__ == "__main__":
    demo.launch()
