import streamlit as st
import torch
from src.document_parser import parse_document
from src.tts_gen_chaterbox_local import ChatterboxLocal
from src.doc_reader import text2audio, clean_text_with_llm
from src.voice_manager import VoiceManager
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="Document to Audio Converter",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'tts_engine' not in st.session_state:
    st.session_state.tts_engine = None
if 'voice_manager' not in st.session_state:
    st.session_state.voice_manager = VoiceManager()
if 'audio_generated' not in st.session_state:
    st.session_state.audio_generated = False
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None

def initialize_tts():
    """Initialize TTS engine if not already done"""
    if st.session_state.tts_engine is None:
        st.session_state.tts_engine = text2audio

def main():
    # Header
    st.title("🔊 Document to Audio Converter")
    st.markdown("Convert your documents to audio using local GPU-accelerated Text-to-Speech")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Voice selection
        st.subheader("Voice Selection")
        available_voices = st.session_state.voice_manager.get_voice_list()
        
        if len(available_voices) > 0:
            selected_voice = st.selectbox(
                "Select Voice",
                options=available_voices,
                help="Choose a voice from your saved samples"
            )
        else:
            st.info("No voices available. Please add a voice sample below.")
            selected_voice = None
        
        # Add new voice
        st.subheader("Add New Voice")
        voice_name = st.text_input("Voice Name", placeholder="e.g., My Voice")
        voice_file = st.file_uploader(
            "Upload Voice Sample (WAV, MP3)",
            type=["wav", "mp3"],
            help="Upload a 5-30 second audio clip of the voice you want to clone"
        )
        
        if st.button("Add Voice") and voice_name and voice_file:
            with st.spinner("Adding voice sample..."):
                success = st.session_state.voice_manager.add_voice(voice_name, voice_file)
                if success:
                    st.success(f"Voice '{voice_name}' added successfully!")
                    st.rerun()
                else:
                    st.error("Failed to add voice. Please try again.")
        
        # Exaggeration slider
        st.subheader("TTS Parameters")
        exaggeration = st.slider(
            "Exaggeration",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Controls the emotional intensity and expressiveness of the speech"
        )
        
        # GPU info
        st.subheader("System Info")
        if torch.cuda.is_available():
            st.success(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠ No GPU detected. Using CPU (slower)")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=["pdf", "docx", "txt", "md"],
            help="Supported formats: PDF, Word Document, Text, Markdown"
        )
        
        if uploaded_file:
            st.success(f"✓ Uploaded: {uploaded_file.name}")
            file_size = uploaded_file.size / 1024  # KB
            st.info(f"File size: {file_size:.2f} KB")
    
    with col2:
        st.header("🎵 Generate Audio")
        
        if uploaded_file and selected_voice:
            if st.button("🎙️ Convert to Audio", type="primary", use_container_width=True):
                try:
                    # Initialize TTS
                    initialize_tts()
                    
                    # Parse document
                    with st.spinner("📖 Extracting text from document..."):
                        text = parse_document(uploaded_file)
                        word_count = len(text.split())
                        st.info(f"Extracted {word_count} words")
                    
                    if not text.strip():
                        st.error("No text found in the document!")
                    else:
                        with st.spinner("Cleaning text with LLM..."):
                            llm_progress_bar = st.progress(0.0)
                            text = clean_text_with_llm(text, progress_callback=lambda p: llm_progress_bar.progress(p))
                            llm_progress_bar.progress(1.0)
                            st.success("✓ Text cleaned successfully!")
                        # Generate audio
                        voice_path = st.session_state.voice_manager.get_voice_path(selected_voice)
                        
                        with st.spinner("Loading TTS model into memory (may take a few minutes)..."):
                            tts = ChatterboxLocal(ref_audio_path=voice_path, exaggeration=exaggeration)
                            tts.load()
                        
                        with st.spinner("🎵 Generating audio... This may take a while for long documents."):
                            progress_bar = st.progress(0.0)
                            
                            # Create a persistent directory for audio files
                            os.makedirs("generated_audio", exist_ok=True)
                            audio_path = os.path.join("generated_audio", f"{uploaded_file.name.rsplit('.', 1)[0]}_audio.wav")
                            
                            st.session_state.tts_engine(
                                text,
                                audio_path,
                                ref_audio_path=voice_path,
                                exaggeration=exaggeration,
                                progress_callback=lambda p: progress_bar.progress(p),
                                tts=tts
                            )
                            
                            st.session_state.audio_path = audio_path
                            st.session_state.audio_generated = True
                            progress_bar.progress(1.0)
                        
                        st.success("✓ Audio generated successfully!")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            if not uploaded_file:
                st.info("👆 Please upload a document first")
            if not selected_voice:
                st.info("👈 Please select or add a voice in the sidebar")
    
    # Audio playback section
    if st.session_state.audio_generated and st.session_state.audio_path:
        st.header("🎧 Playback & Download")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.audio(st.session_state.audio_path, format="audio/wav")
        
        with col2:
            with open(st.session_state.audio_path, "rb") as audio_file:
                st.download_button(
                    label="⬇️ Download Audio",
                    data=audio_file,
                    file_name="converted_audio.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #888;'>
            <p>Powered by Chatterbox TTS • Running locally on your GPU</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
