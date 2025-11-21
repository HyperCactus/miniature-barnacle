try:
    from .tts_gen_chaterbox_local import ChatterboxLocal
except (ImportError, SystemError): # for running as __main__ (testing)
    from tts_gen_chaterbox_local import ChatterboxLocal

import os
from typing import Optional
import tempfile
from pydub import AudioSegment
import dspy
import lmstudio as lms

class CleanText(dspy.Signature):
    """Fix any obvious issues in the text and write everything out exactly how it should be pronounced.
    For example, 22 + 1/40 should be written as "twenty two plus one over forty".
    Exclude any text that should not be spoken such as image captions, references etc.
    """
    original_text: str = dspy.InputField()
    cleaned_text: str = dspy.OutputField()

class LLMConfig():
    def __init__(
        self, model_name: str, provider: str, api_key: str = "not needed", 
        temperature: float = 0.2, context_length: int = 2048, base_url: str = None
        ):
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key
        self.temperature = temperature
        self.context_length = context_length
        self.base_url = base_url
        self.lm_studio_model = None
        
        self.load()
        self.lm = dspy.LM(model=f"{self.provider}/{self.model_name}", api_key=self.api_key, temperature=self.temperature, 
                          max_tokens=self.context_length, base_url=self.base_url)
        
    def load(self):
        if self.provider == "lm_studio":
            try:
                self.lm_studio_model = lms.llm(self.model_name, ttl=3600*10)
            except Exception as e:
                print(f"Error loading lm_studio model: {e}")
    
    def unload(self):
        if self.provider == "lm_studio":
            self.lm_studio_model.unload()

def clean_text_with_llm(text: str) -> str:
    """Uses an LLM to clean text and prepare it for TTS processing."""
    lm_config = LLMConfig("qwen/qwen3-4b-2507", "lm_studio", base_url='http://localhost:1234/v1/')
    dspy.configure(lm=lm_config.lm)
    cleaner = dspy.Predict(CleanText)
    chunks = chunk_text(text, max_length=1000)
    cleaned_chunks = []
    for chunk in chunks:
        cleaned_chunk = cleaner(original_text=chunk).cleaned_text
        cleaned_chunks.append(cleaned_chunk)
    
    lm_config.unload()
    return " ".join(cleaned_chunks)

def chunk_text(text: str, max_length: int = 200) -> list:
    chunks = []
    for s in text.split('. '):
        s += '.'
        if len(s) <= max_length:
            chunks.append(s)
        else:
            words = s.split(' ')
            for i in range(0, len(words), max_length // 5):
                chunk = ' '.join(words[i:i + max_length // 5])
                chunks.append(chunk)
    return chunks

def text2audio(
    text: str, 
    out_path: os.PathLike, 
    ref_audio_path: Optional[os.PathLike] = None, 
    exaggeration: Optional[float] = None,
    progress_callback: Optional[callable] = None,
    tts=None
    ) -> str:
    """Chunk text into small segments, generate audio for each segment, and combine into a single audio file.
    """
    if not tts:
        tts = ChatterboxLocal(ref_audio_path=ref_audio_path, exaggeration=exaggeration)
    
    tts.load()
    
    lines = chunk_text(text)
    
    # Create a temporary directory to store chunk audio files
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_files = []
        for i, line in enumerate(lines):
            if progress_callback:
                progress_callback((i + 1) / len(lines))
            chunk_out_path = os.path.join(temp_dir, f"chunk_{i}.wav")
            tts.generate(text=line, out_path=chunk_out_path, ref_audio_path=ref_audio_path, exaggeration=exaggeration)
            chunk_files.append(chunk_out_path)
        
        # Combine chunk audio files into one
        combined = AudioSegment.empty()
        for chunk_file in chunk_files:
            chunk_audio = AudioSegment.from_wav(chunk_file)
            combined += chunk_audio + AudioSegment.silent(duration=500)  # Add 0.5s silence between chunks
        
        # Export combined audio
        combined.export(out_path, format="wav")
        
    return out_path


if __name__ == "__main__":
    sample_text = "Hello w0rld. This is a t est of the text to audio conversion. We are using Cha tterbox TTS. 164,391 + 1/40 equals what?"
    output_path = "output_audio.wav"
    reference_audio = "voices/male_voice_1/sample.wav" 
    
    print("Cleaning text with LLM...")
    cleaned_text = clean_text_with_llm(sample_text)
    print("Cleaned Text:", cleaned_text)
    text2audio(cleaned_text, output_path, ref_audio_path=reference_audio, exaggeration=0.7)
    print(f"Audio generated at: {output_path}")