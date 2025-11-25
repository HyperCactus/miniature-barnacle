import unittest
from unittest.mock import MagicMock, patch, ANY
import io
import os
import torch
from src.tts_gen_chaterbox_local import ChatterboxLocal
from src.doc_reader import text2audio
from pydub import AudioSegment

class TestMemoryTTS(unittest.TestCase):
    
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.sr = 24000
        # Mock generate to return a random tensor
        self.mock_model.generate.return_value = torch.randn(1, 24000) 

    @patch('src.tts_gen_chaterbox_local.ChatterboxTTS')
    @patch('src.tts_gen_chaterbox_local.ta.save')
    def test_chatterbox_local_generate_memory(self, mock_ta_save, mock_chatterbox_cls):
        # Setup mock
        mock_chatterbox_cls.from_pretrained.return_value = self.mock_model
        
        tts = ChatterboxLocal()
        tts.load()
        
        # Test generate with out_path=None
        result = tts.generate("Test text", out_path=None)
        
        # Verify result is BytesIO
        self.assertIsInstance(result, io.BytesIO)
        
        # Verify ta.save was called with the buffer
        mock_ta_save.assert_called_once()
        args, _ = mock_ta_save.call_args
        self.assertIs(args[0], result) # First arg should be the buffer
        
        # Verify buffer position is at 0
        self.assertEqual(result.tell(), 0)

    @patch('src.tts_gen_chaterbox_local.ChatterboxTTS')
    @patch('src.tts_gen_chaterbox_local.ta.save')
    @patch('src.doc_reader.AudioSegment.from_wav')
    @patch('src.doc_reader.AudioSegment.silent')
    @patch('src.doc_reader.AudioSegment.empty')
    def test_doc_reader_memory_flow(self, mock_empty, mock_silent, mock_from_wav, mock_ta_save, mock_chatterbox_cls):
        # Setup mocks
        mock_chatterbox_cls.from_pretrained.return_value = self.mock_model
        
        # Mock AudioSegment
        mock_segment = MagicMock()
        mock_from_wav.return_value = mock_segment
        mock_silent.return_value = MagicMock()
        mock_empty.return_value = MagicMock()
        
        # Mock sum of segments behavior
        # sum() starts with start value and adds items: start + item1 + item2...
        mock_empty.return_value.__add__.return_value = mock_segment
        mock_segment.__add__.return_value = mock_segment
        
        # Run text2audio
        output_path = "test_output.wav"
        # We need to mock chunk_text to return a known list so we know how many times generate is called
        with patch('src.doc_reader.chunk_text', return_value=["Chunk 1", "Chunk 2"]):
            # We also need to mock the export of the final combined audio
            # Since we are mocking AudioSegment instances, we need to ensure the final object has export method
            mock_segment.export = MagicMock()
            
            text2audio("Chunk 1 Chunk 2", output_path)
            
            # Verify ta.save was called with BytesIO objects
            self.assertEqual(mock_ta_save.call_count, 2)
            for call in mock_ta_save.call_args_list:
                args, _ = call
                self.assertIsInstance(args[0], io.BytesIO)
            
            # Verify AudioSegment.from_wav was called with BytesIO objects
            self.assertEqual(mock_from_wav.call_count, 2)
            for call in mock_from_wav.call_args_list:
                args, _ = call
                self.assertIsInstance(args[0], io.BytesIO)

if __name__ == '__main__':
    unittest.main()