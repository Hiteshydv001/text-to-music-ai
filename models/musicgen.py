import logging
from typing import Optional
import torch
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
            audio = self.model.generate([text_prompt], progress=True)
            return audio
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

    def save_audio(self, audio: torch.Tensor, filepath: str = "output.wav") -> Optional[str]:
        """Save audio tensor to WAV file."""
        try:
            if audio is None:
                raise ValueError("No audio tensor provided")
            torch.save(audio.cpu(), filepath, format="wav", sample_rate=self.sample_rate)
            logger.info(f"Audio saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return None