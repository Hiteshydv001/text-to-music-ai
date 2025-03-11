# Text-to-Music Generator

## Overview
The Text-to-Music Generator is an AI-powered system that generates music from text descriptions. It leverages Meta's MusicGen model and provides a user-friendly interface via Gradio.

## Features
- Converts text prompts into unique music compositions.
- Uses MusicGen for high-quality music generation.
- Provides an interactive Gradio UI for easy music generation.
- Supports waveform visualization.
- Saves generated music as WAV files.

## Directory Structure
```
└── hiteshydv001-text-to-music-ai/
    ├── README.md
    ├── config.yaml
    ├── inference.py
    ├── requirements.txt
    ├── setup.py
    ├── generated_audio/
    ├── gradio_app/
    │   └── app.py
    ├── models/
    │   ├── musicgen.py
    │   └── train.py
    ├── utils/
    │   ├── audio_utils.py
    │   └── text_utils.py
    └── .gradio/
        └── certificate.pem
```

## Setup Instructions

### 1. Install Dependencies
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset (Optional for Training)
Download and extract the Lakh MIDI Dataset:
```bash
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz -P data/raw/
tar -xvzf data/raw/lmd_full.tar.gz -C data/raw/
```

### 3. Running Inference
Generate music from a text prompt:
```bash
python inference.py
```

### 4. Launching the Gradio Interface
To start the Gradio web interface:
```bash
python gradio_app/app.py
```

## Configuration
Modify `config.yaml` to adjust model and training settings:
```yaml
model:
  name: "facebook/musicgen-small"
  duration: 10

training:
  batch_size: 4
  epochs: 10
  learning_rate: 0.0001

data:
  raw_dir: "data/raw/"
  processed_dir: "data/processed/"
```

## Usage
1. Open the Gradio web interface.
2. Enter a text prompt describing the music.
3. Click "Generate" to create and download the music.

## Dependencies
- `torch`
- `torchaudio`
- `transformers`
- `audiocraft`
- `gradio`
- `numpy`
- `librosa`

## Author
Developed by Hitesh Kumar. [Hugging Face Space](https://huggingface.co/spaces/hitesh-aiml/Text-to-music-generator)

## License
This project is released under the MIT License.

