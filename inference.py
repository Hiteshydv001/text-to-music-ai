import logging
from typing import Optional
import torchaudio
from models.musicgen import MusicGenerator

logger = logging.getLogger(__name__)

def generate_music(text_prompt: str, output_path: str = "output.wav") -> Optional[str]:
    """Generate music from text prompt."""
    try:
        generator = MusicGenerator()
        audio = generator.generate(text_prompt)
        if audio is None:
            raise ValueError("Audio generation failed")
        return generator.save_audio(audio, output_path)
    except Exception as e:
        logger.error(f"Music generation failed: {e}")
        return None

if __name__ == "__main__":
    prompt = "A slow, emotional piano piece with deep strings."
    output = generate_music(prompt)
    print(f"Generated music saved to {output}" if output else "Failed to generate music.")