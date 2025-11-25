import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.doc_reader import QwenTextCleaner

class TestLLMParsing(unittest.TestCase):
    def setUp(self):
        self.cleaner = QwenTextCleaner()
        # Mock the model and tokenizer
        self.cleaner.model = MagicMock()
        self.cleaner.tokenizer = MagicMock()
        self.cleaner.is_loaded = True
        self.cleaner.device = "cpu"

    def test_clean_chunk_with_assistant_keyword_in_response(self):
        """
        Test that the cleaner correctly parses a response that contains the word 'assistant'.
        Current implementation fails this because it splits by 'assistant' and takes the last part.
        """
        # Setup mocks
        input_text = "Some input text"
        
        # Mock tokenizer behavior
        # apply_chat_template returns the prompt string
        prompt_str = "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\nSome input text<|im_end|>\n<|im_start|>assistant\n"
        self.cleaner.tokenizer.apply_chat_template.return_value = prompt_str
        
        # tokenizer call returns input_ids
        input_ids = torch.tensor([[1, 2, 3]])
        
        # Create a mock object for the tokenizer output that has a .to() method
        tokenizer_output = MagicMock()
        # The result of .to() should be an object with input_ids attribute
        inputs_mock = MagicMock()
        inputs_mock.input_ids = input_ids
        tokenizer_output.to.return_value = inputs_mock
        self.cleaner.tokenizer.return_value = tokenizer_output
        
        # Mock model generation
        # The model returns [input_ids + generated_ids]
        # Input length is 3 (from input_ids above)
        # Let's say generated ids are [4, 5, 6, 7] corresponding to "The assistant is helpful."
        generated_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        self.cleaner.model.generate.return_value = generated_ids
        
        # Mock tokenizer.decode
        # It should be called with the sliced tokens [4, 5, 6, 7]
        def decode_side_effect(tokens, skip_special_tokens=True):
            # Verify we are decoding the correct tokens
            if len(tokens) == 4 and tokens.tolist() == [4, 5, 6, 7]:
                return "The assistant is helpful."
            return "Unexpected tokens"
            
        self.cleaner.tokenizer.decode.side_effect = decode_side_effect
        
        # Execute
        result = self.cleaner.clean_chunk(input_text)
        
        # Verification
        print(f"Result: '{result}'")
        self.assertEqual(result, "The assistant is helpful.")

if __name__ == '__main__':
    unittest.main()