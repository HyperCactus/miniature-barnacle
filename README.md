# miniature-barnacle


## About
Proprietary cloud based TTS services exist that allow you to upload a document and have it converted to high quality audio, but thy are expensive and lack privacy. This project is fully free, local and open source, and does high quallity, nateral sounding tts on documents

## Install instructions
```bash
git clone https://github.com/HyperCactus/miniature-barnacle.git
cd minature-barnacle
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
