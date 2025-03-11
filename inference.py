import gradio as gr
import torch
import torchaudio
import logging
import os
from audiocraft.models import MusicGen

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ensure output directory exists
OUTPUT_DIR = "generated_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    def generate(self, text_prompt: str) -> torch.Tensor:
        """Generate audio from text prompt."""
        try:
            logger.info(f"Generating audio for prompt: {text_prompt}")
            audio_batch = self.model.generate([text_prompt], progress=True)

            if not isinstance(audio_batch, torch.Tensor):
                raise ValueError("Generated audio is not a valid tensor.")

            # Ensure correct shape: [channels, samples]
            audio = audio_batch.squeeze(0) if audio_batch.dim() == 3 else audio_batch
            return audio.to(dtype=torch.float32)  # Ensure it's float32
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

    def save_audio(self, audio: torch.Tensor) -> str:
        """Save audio tensor to WAV file and return path."""
        try:
            if audio is None:
                raise ValueError("No valid audio tensor provided.")

            output_path = os.path.join(OUTPUT_DIR, "output.wav")
            torchaudio.save(output_path, audio.cpu(), sample_rate=self.sample_rate)
            logger.info(f"Audio saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return None

# Initialize generator once (avoiding multiple model loads)
generator = MusicGenerator()

def generate_music_gradio(prompt: str):
    """Generate music based on text prompt and return the WAV file."""
    logger.info(f"Received prompt: {prompt}")
    audio = generator.generate(prompt)
    if audio is not None:
        return generator.save_audio(audio)
    else:
        return "Failed to generate music."

# Gradio Interface
gr.Interface(
    fn=generate_music_gradio,
    inputs=gr.Textbox(label="Enter Music Prompt"),
    outputs=gr.Audio(label="Generated Music"),
    title="🎵 Text-to-Music Generation",
    description="Enter a text prompt to generate AI music 🎶",
).launch(share=True)
