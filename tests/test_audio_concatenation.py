import unittest
import os
import tempfile
from unittest.mock import MagicMock, patch
from pydub import AudioSegment
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from doc_reader import text2audio

class TestAudioConcatenation(unittest.TestCase):
    @patch('doc_reader.ChatterboxLocal')
    @patch('doc_reader.AudioSegment')
    def test_audio_concatenation_efficiency(self, MockAudioSegment, MockChatterbox):
        # Setup mocks
        mock_tts = MockChatterbox.return_value
        mock_tts.generate.return_value = None # generate just writes to file, returns nothing
        
        # Mock AudioSegment.from_wav to return a 1-second segment
        mock_segment = MagicMock()
        mock_segment.__len__.return_value = 1000 # 1 second in ms
        mock_segment.__add__.return_value = mock_segment # simplified addition
        
        # We need to handle the += operation and the sum() operation if we change to that.
        # For the current implementation (+=), it calls __add__.
        # For the optimized implementation (sum), it might use __radd__ or start with empty.
        
        # Let's make a more realistic mock for AudioSegment that tracks duration
        class FakeAudioSegment:
            def __init__(self, duration=0):
                self.duration = duration
            
            def __len__(self):
                return self.duration
            
            def __add__(self, other):
                return FakeAudioSegment(self.duration + len(other))
            
            def __radd__(self, other):
                if other == 0: # sum() starts with 0
                    return self
                return FakeAudioSegment(self.duration + len(other))
                
            def export(self, *args, **kwargs):
                pass
                
            @classmethod
            def empty(cls):
                return FakeAudioSegment(0)
                
            @classmethod
            def silent(cls, duration=0):
                return FakeAudioSegment(duration)
                
            @classmethod
            def from_wav(cls, path):
                return FakeAudioSegment(1000) # 1 second dummy audio

        # Apply the FakeAudioSegment to the patch
        MockAudioSegment.empty.side_effect = FakeAudioSegment.empty
        MockAudioSegment.silent.side_effect = FakeAudioSegment.silent
        MockAudioSegment.from_wav.side_effect = FakeAudioSegment.from_wav
        
        # We also need to patch the class itself so that isinstance checks or direct instantiation works if needed,
        # but here we are mostly using class methods.
        # However, the code uses `combined += ...` which relies on the object returned by empty().
        
        # Let's try to run text2audio with a text that produces multiple chunks.
        # chunk_text splits by sentences.
        text = "Sentence one. Sentence two. Sentence three."
        
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = os.path.join(temp_dir, "output.wav")
            
            # We need to mock chunk_text or ensure the text is split correctly.
            # The real chunk_text uses nltk. Let's assume it works or mock it if needed.
            # To be safe and avoid nltk dependency issues in test environment if not set up, let's mock chunk_text too?
            # But the environment seems to have it. Let's try without mocking chunk_text first.
            
            # Actually, let's mock chunk_text to control the number of chunks exactly.
            with patch('doc_reader.chunk_text', return_value=["s1", "s2", "s3"]):
                text2audio(text, out_path, tts=mock_tts)
                
                # Verify that AudioSegment.from_wav was called 3 times
                self.assertEqual(MockAudioSegment.from_wav.call_count, 3)
                
                # Verify export was called
                # We need to capture the 'combined' object to check its duration.
                # Since we can't easily access the local variable 'combined', we can check the arguments to export if we could.
                # But our FakeAudioSegment.export is a method on the instance.
                
                # Wait, since we replaced AudioSegment with a Mock that returns FakeAudioSegment instances,
                # we can't easily spy on the final instance unless we capture it.
                # But we can verify the logic by checking if the final duration is correct.
                # 3 chunks * 1000ms + 3 silences * 500ms = 4500ms?
                # The code does: combined += chunk_audio + silence
                # So for 3 chunks: (1000+500) + (1000+500) + (1000+500) = 4500ms.
                # Wait, does it add silence after the last one? Yes.
                
                # To verify this, we can mock the export method on the FakeAudioSegment to save the duration.
                # But FakeAudioSegment is a local class.
                pass

    def test_concatenation_logic(self):
        # This test specifically targets the optimization logic we plan to implement
        # We want to ensure sum([segments]) works as expected with our FakeAudioSegment
        
        class FakeAudioSegment:
            def __init__(self, duration=0):
                self.duration = duration
            def __len__(self):
                return self.duration
            def __add__(self, other):
                return FakeAudioSegment(self.duration + len(other))
            def __radd__(self, other):
                if other == 0: return self
                return FakeAudioSegment(self.duration + len(other))
        
        segments = [FakeAudioSegment(1000) for _ in range(3)]
        silence = FakeAudioSegment(500)
        
        # Simulate the old loop
        combined_old = FakeAudioSegment(0)
        for seg in segments:
            combined_old += seg + silence
        
        self.assertEqual(len(combined_old), 4500)
        
        # Simulate the new approach (list comprehension + sum)
        # We want to interleave silence.
        # [seg + silence for seg in segments]
        segments_with_silence = [seg + silence for seg in segments]
        combined_new = sum(segments_with_silence, FakeAudioSegment(0))
        
        self.assertEqual(len(combined_new), 4500)

if __name__ == '__main__':
    unittest.main()