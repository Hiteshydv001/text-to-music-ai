import logging
from typing import Optional, List, Tuple
import pretty_midi
import librosa
import numpy as np

logger = logging.getLogger(__name__)

def midi_to_wav(midi_path: str, output_path: str, sample_rate: int = 44100) -> Optional[str]:
    """Convert MIDI to WAV with error handling."""
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        audio = midi_data.synthesize(fs=sample_rate)
        librosa.output.write_wav(output_path, audio, sample_rate)
        logger.info(f"Converted {midi_path} to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"MIDI to WAV conversion failed: {e}")
        return None

def preprocess_midi(midi_path: str) -> List[Tuple[int, float, float, int]]:
    """Extract MIDI features efficiently."""
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        notes = [(note.pitch, note.start, note.end, note.velocity) 
                 for note in midi_data.instruments[0].notes]
        logger.info(f"Preprocessed {midi_path} with {len(notes)} notes")
        return notes
    except Exception as e:
        logger.error(f"MIDI preprocessing failed: {e}")
        return []