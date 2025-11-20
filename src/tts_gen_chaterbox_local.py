import os
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import torch
from typing import Optional
from contextlib import redirect_stdout, redirect_stderr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatterboxLocal():
    def __init__(self, 
                 ref_audio_path: Optional[os.PathLike] = None, 
                 exaggeration: Optional[str] = None,
                 cfg_weight: Optional[str] = None,
                 temperature: Optional[str] = None,
                 verbose: bool = False
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.ref_audio_path = None
        self.verbose = verbose
        self.exaggeration = exaggeration or 0.6
        self.cfg_weight = cfg_weight or 0.5
        self.temperature = temperature or 0.8
        self.ref_audio_path = ref_audio_path

    def load(self):
        if self.model is None:
            # If chatterbox supports kwargs like dtype, you could pass torch.float32 explicitly
            self.model = ChatterboxTTS.from_pretrained(device=self.device)

    def generate(self, text: str, out_path: str, exaggeration: Optional[float] = None, ref_audio_path: Optional[os.PathLike] = None):
        self.load()
        # If there’s no ref, pass None so chatterbox skips voice cloning path
        ref = ref_audio_path or self.ref_audio_path
        exaggeration = exaggeration or self.exaggeration
        if self.verbose:
            wav = self.model.generate(
                text,
                audio_prompt_path=ref,
                exaggeration=exaggeration,
                cfg_weight=self.cfg_weight,
                temperature=self.temperature
            )
        else:
            # suppress stdout/stderr and reduce logging noise during generation
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                root_logger = logging.getLogger()
                prev_level = root_logger.level
                root_logger.setLevel(logging.CRITICAL)
                try:
                    wav = self.model.generate(
                        text,
                        audio_prompt_path=self.ref_audio_path,
                        exaggeration=self.exaggeration,
                        cfg_weight=self.cfg_weight,
                        temperature=self.temperature
                    )
                finally:
                    root_logger.setLevel(prev_level)

        # Ensure 2D [channels, time] on CPU for torchaudio
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        ta.save(
            out_path,
            wav.to("cpu"),
            self.model.sr,
            encoding="PCM_S",
            bits_per_sample=16
        )

    def unload(self):
        self.model = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
    tts = ChatterboxLocal(narrator_reference="narrator_reference/male1")
    tts.load()
    tts.generate(text, "output.wav")
    tts.unload()