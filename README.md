# Text-to-Music Generation Project

A system to generate music from text prompts using MusicGen and Gradio.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download the Lakh MIDI Dataset (see below).
3. Run inference: `python inference.py`
4. Launch UI: `python gradio_app/app.py`

## Dataset
Download the Lakh MIDI Dataset:
```bash
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz -P data/raw/
tar -xvzf data/raw/lmd_full.tar.gz -C data/raw/