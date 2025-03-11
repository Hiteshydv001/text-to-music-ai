import torchaudio  # Add this import
from models.musicgen import MusicGenerator

def generate_music(text_prompt, output_path="output.wav"):
    try:
        # Initialize model
        generator = MusicGenerator()
        
        # Generate audio
        audio = generator.generate(text_prompt)
        
        # Save to WAV
        output_path = generator.save_audio(audio, output_path)
        return output_path
    except Exception as e:
        print(f"Error generating music: {e}")
        return None

if __name__ == "__main__":
    prompt = "A slow, emotional piano piece with deep strings."
    output = generate_music(prompt)
    if output:
        print(f"Generated music saved to {output}")
    else:
        print("Failed to generate music.")