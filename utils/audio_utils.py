# Functions for MIDI to WAV conversion
import pretty_midi
import librosa
import numpy as np

def midi_to_wav(midi_path, output_path, sample_rate=44100):
    # Convert MIDI to WAV using pretty_midi and librosa
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    audio = midi_data.synthesize(fs=sample_rate)
    librosa.output.write_wav(output_path, audio, sample_rate)

def preprocess_midi(midi_path):
    # Extract MIDI features (notes, velocity, timing)
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    notes = [(note.pitch, note.start, note.end) for note in midi_data.instruments[0].notes]
    return notes