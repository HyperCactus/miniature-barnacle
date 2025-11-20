from .tts_gen_chaterbox_local import ChatterboxLocal
import os
from typing import Optional
import tempfile
from pydub import AudioSegment

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
    
    lines = text.split('. ')
    
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
    sample_text = "Hello world. This is a test of the text to audio conversion. We are using Chatterbox TTS."
    output_path = "output_audio.wav"
    reference_audio = "voices/male_voice_1/sample.wav" 
    
    text2audio(sample_text, output_path, ref_audio_path=reference_audio, exaggeration=0.7)
    print(f"Audio generated at: {output_path}")