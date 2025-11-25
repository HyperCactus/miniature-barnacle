import sys
import os
import unittest

# Add src to path so we can import doc_reader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.doc_reader import chunk_text

class TestSentenceSplitting(unittest.TestCase):
    def test_basic_splitting(self):
        text = "Hello world. This is a test."
        chunks = chunk_text(text, max_length=50)
        # Should be split into two sentences if max_length allows, or combined if they fit.
        # With max_length=50, "Hello world. This is a test." (28 chars) fits in one chunk.
        # Let's force a split by using a small max_length
        chunks_small = chunk_text(text, max_length=15)
        self.assertIn("Hello world.", chunks_small)
        self.assertIn("This is a test.", chunks_small)

    def test_abbreviations(self):
        text = "Mr. Smith went to the U.S.A. today."
        chunks = chunk_text(text, max_length=100)
        # Should NOT split at Mr. or U.S.A.
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Mr. Smith went to the U.S.A. today.")

    def test_figures(self):
        text = "Fig. 1 shows the data. This is important."
        chunks = chunk_text(text, max_length=100)
        # Should NOT split at Fig.
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Fig. 1 shows the data. This is important.")
        
    def test_multiple_sentences_with_abbreviations(self):
        text = "Dr. Jones is here. He likes the U.K. very much."
        chunks = chunk_text(text, max_length=30)
        # "Dr. Jones is here." is 18 chars. "He likes the U.K. very much." is 28 chars.
        # Should be two chunks.
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "Dr. Jones is here.")
        self.assertEqual(chunks[1], "He likes the U.K. very much.")

if __name__ == '__main__':
    unittest.main()