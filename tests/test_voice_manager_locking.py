import unittest
import tempfile
import shutil
import os
import threading
import time
import sys
from pathlib import Path
from unittest.mock import MagicMock
from filelock import FileLock

# Add project root to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.voice_manager import VoiceManager

class TestVoiceManagerLocking(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.voice_manager = VoiceManager(voices_dir=self.test_dir)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_lock_file_path(self):
        """Test that the lock file path is correctly defined."""
        expected_lock_path = os.path.join(self.test_dir, "voices.json.lock")
        self.assertEqual(str(self.voice_manager.lock_file), expected_lock_path)

    def test_concurrent_access(self):
        """Test concurrent access to the voice manager."""
        # Create a dummy audio file
        dummy_audio = MagicMock()
        dummy_audio.name = "test.wav"
        dummy_audio.read.return_value = b"fake audio data"
        
        def add_voice_thread(name):
            # Simulate some work and then add a voice
            time.sleep(0.1)
            self.voice_manager.add_voice(name, dummy_audio)

        threads = []
        for i in range(5):
            t = threading.Thread(target=add_voice_thread, args=(f"voice_{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify all voices were added correctly
        voices = self.voice_manager.get_voice_list()
        self.assertEqual(len(voices), 6) # 5 added voices + "Model default"
        for i in range(5):
            self.assertIn(f"voice_{i}", voices)

    def test_lock_contention(self):
        """Test that operations wait for the lock."""
        lock_path = os.path.join(self.test_dir, "voices.json.lock")
        
        # Event to signal that the helper thread has acquired the lock
        lock_acquired = threading.Event()
        
        def hold_lock():
            with FileLock(lock_path):
                lock_acquired.set()
                time.sleep(1.0)
        
        t = threading.Thread(target=hold_lock)
        t.start()
        
        # Wait for the thread to acquire the lock
        lock_acquired.wait()
        
        start_time = time.time()
        
        # This should block until the thread releases the lock
        self.voice_manager._load_metadata()
        
        end_time = time.time()
        t.join()
        
        # Ensure it waited at least approximately 1 second
        # We use 0.9 to account for slight timing variations
        self.assertGreaterEqual(end_time - start_time, 0.9)

if __name__ == '__main__':
    unittest.main()