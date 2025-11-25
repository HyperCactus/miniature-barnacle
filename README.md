# Miniature Barnacle - Local Document to Audio Converter

## About
Proprietary cloud-based TTS services exist that allow you to upload a document and have it converted to high-quality audio, but they are often expensive and lack privacy. **Miniature Barnacle** is a fully free, open-source project that performs high-quality, natural-sounding Text-to-Speech (TTS) on entire documents completely locally.

By leveraging local GPU acceleration, this tool ensures your data never leaves your machine while providing professional-grade audio output. It uses advanced LLMs to clean and format text for optimal speech synthesis and supports voice cloning from short audio samples.

## Features
- **Local & Private:** All processing (text cleaning and audio generation) happens locally on your machine. No data is sent to the cloud.
- **High-Quality TTS:** Uses Chatterbox TTS for natural, expressive speech synthesis.
- **Voice Cloning:** Clone any voice by uploading a short (5-30 second) audio sample.
- **Smart Text Cleaning:** Utilizes a local LLM (Qwen) to normalize text (e.g., converting "1/40" to "one over forty") for smoother reading.
- **Multi-Format Support:** Converts PDF, DOCX, TXT, and Markdown files.
- **Customizable:** Adjust TTS parameters like emotional intensity (exaggeration), stability (CFG weight), and randomness (temperature).
- **User-Friendly UI:** Simple, interactive interface built with Streamlit.

## Install instructions

```bash
git clone https://github.com/HyperCactus/miniature-barnacle.git
cd miniature-barnacle
```

```bash
conda create -n myenv python=3.10 -y
conda activate myenv
```

```bash
pip install -r requirements.txt
```

### For GPU support (recommended)
Find your cuda version `nvcc --version` then find the relevant [install command](https://pytorch.org/get-started/locally/) for `torch==2.6.0` and `torchaudio==2.6.0` with your OS and cuda version, and reinstall with `--upgrade` and `--force-reinstall` commands with pip. For example, for cuda 12.6 on windows:

```bash
pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch==2.6.0 torchaudio==2.6.0
```

## Usage
To start the application, run:

```bash
streamlit run app.py
```

The interface will open in your default web browser. From there, you can:
1. Upload a document.
2. Select a default voice or upload a sample to clone a new voice.
3. Click "Convert to Audio" to process the document.
4. Play or download the generated audio file.

## Disclaimer
Do not clone anyone's voice without their permission and do not use this tool for malicious activity.

## Acknowledgements 
[Chatterbox TTS](https://huggingface.co/ResembleAI/chatterbox)