# MusicGen wrapper script
from audiocraft.models import MusicGen
import torch

class MusicGenerator:
    def __init__(self, model_name="facebook/musicgen-small"):
        # Load pretrained MusicGen model
        self.model = MusicGen.get_pretrained(model_name)
        self.model.set_generation_params(duration=10)  # Default 10s audio

    def generate(self, text_prompt):
        # Generate audio from text
        audio = self.model.generate([text_prompt], progress=True)
        return audio  # Tensor of audio samples

    def save_audio(self, audio, filepath="output.wav"):
        # Save audio tensor as WAV
        torchaudio.save(filepath, audio.cpu(), sample_rate=self.model.sample_rate)