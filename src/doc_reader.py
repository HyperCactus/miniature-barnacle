try:
    from .tts_gen_chaterbox_local import ChatterboxLocal
except (ImportError, SystemError): # for running as __main__ (testing)
    from tts_gen_chaterbox_local import ChatterboxLocal

import os
from typing import Optional
import tempfile
import nltk
from pydub import AudioSegment
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import gc

# System prompt from the original CleanText docstring
SYSTEM_PROMPT = """Fix any obvious issues in the text and write everything out exactly how it should be pronounced.
For example, 22 + 1/40 should be written as "twenty two plus one over forty".
Exclude any text that should not be spoken such as image captions, references etc.
Reply only with the exact text to be spoken, do not include any additional commentary."""

class QwenTextCleaner:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
    
    def load(self):
        """Load the Qwen model and tokenizer with 8-bit quantization"""
        if not self.is_loaded:
            print(f"Loading Qwen model: {self.model_name}")
            
            # Configure 8-bit quantization to reduce memory usage
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf8"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.is_loaded = True
    
    def unload(self):
        """Unload the model from GPU memory"""
        if self.is_loaded:
            print("Unloading Qwen model from GPU...")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.is_loaded = False
    
    def clean_chunk(self, text_chunk: str) -> str:
        """Clean a single text chunk using the Qwen model"""
        if not self.is_loaded:
            self.load()
        
        # Format the prompt for the model
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text_chunk}
        ]
        
        # Format the input for the model
        formatted_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
            )
        
        # Decode the output
        # Slice the output to get only the new tokens (exclude the prompt)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()

def clean_text_with_llm(text: str, progress_callback: Optional[callable] = None) -> str:
    """Uses Qwen model directly from Hugging Face to clean text for TTS processing."""
    cleaner = QwenTextCleaner()
    
    try:
        cleaner.load()
        chunks = chunk_text(text, max_length=1000)
        cleaned_chunks = []
        
        # Process chunks with progress updates
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback((i + 1) / len(chunks))
        
        cleaned_chunk = cleaner.clean_chunk(chunk)
        cleaned_chunks.append(cleaned_chunk)
        
        # Final progress update
        if progress_callback:
            progress_callback(1.0)
        
        return " ".join(cleaned_chunks)
    
    finally:
        cleaner.unload()

def chunk_text(text: str, max_length: int = 200) -> list:
    """Split text into chunks for processing by the LLM"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

    chunks = []
    sentences = nltk.sent_tokenize(text)
    
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence exceeds max_length, save current chunk and start new one
        # (unless current chunk is empty, then we must take it)
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
                
        # Handle case where a single sentence is longer than max_length
        if len(current_chunk) > max_length:
             # If the current chunk (which might be just one sentence) is too long, split it by words
            words = current_chunk.split(' ')
            temp_chunk = []
            temp_length = 0
            
            for word in words:
                word_length = len(word) + 1
                if temp_length + word_length <= max_length:
                    temp_chunk.append(word)
                    temp_length += word_length
                else:
                    if temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                    temp_chunk = [word]
                    temp_length = word_length
            
            if temp_chunk:
                current_chunk = ' '.join(temp_chunk)
            else:
                current_chunk = ""
                
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def text2audio(
    text: str, 
    out_path: os.PathLike, 
    ref_audio_path: Optional[os.PathLike] = None, 
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
    temperature: Optional[float] = None,
    progress_callback: Optional[callable] = None,
    tts=None
    ) -> str:
    """Chunk text into small segments, generate audio for each segment, and combine into a single audio file.
    """
    if not tts:
        tts = ChatterboxLocal(ref_audio_path=ref_audio_path, 
                              exaggeration=exaggeration,
                              cfg_weight=cfg_weight,
                              temperature=temperature)
    
    tts.load()
    
    lines = chunk_text(text)
    
    # Generate audio chunks in memory and combine them
    segments = []
    silence = AudioSegment.silent(duration=500)
    
    for i, line in enumerate(lines):
        if progress_callback:
            progress_callback((i + 1) / len(lines))
            
        # Generate audio in memory (returns BytesIO)
        audio_buffer = tts.generate(text=line,
                                    out_path=None,
                                    ref_audio_path=ref_audio_path,
                                    exaggeration=exaggeration,
                                    cfg_weight=cfg_weight,
                                    temperature=temperature)
        
        # Create AudioSegment from memory buffer
        chunk_audio = AudioSegment.from_wav(audio_buffer)
        segments.append(chunk_audio + silence)

        if segments:
            combined = sum(segments, AudioSegment.empty())
        else:
            combined = AudioSegment.empty()
        
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