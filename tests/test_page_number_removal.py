import unittest
from src.document_parser import clean_text

class TestPageNumberRemoval(unittest.TestCase):
    def test_preserve_valid_numbers(self):
        """Test that valid numbers in lists or text are preserved."""
        # Case 1: List of numbers
        text = "List:\n10\n20\n30"
        cleaned = clean_text(text)
        self.assertIn("10", cleaned)
        self.assertIn("20", cleaned)
        self.assertIn("30", cleaned)
        # Depending on other cleaning, whitespace might change, but numbers should be there.
        # The original regex replaced \n\d+\n with \n.
        # So "List:\n10\n20\n30" -> "List:\n\n\n" (if they matched exactly \n\d+\n)
        
        # Let's check exact expectation if we remove the regex.
        # clean_text also does:
        # text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # text = re.sub(r'[ \t]+', ' ', text)
        
        # If we input "List:\n10\n20\n30"
        # It should remain mostly the same.
        self.assertEqual(cleaned, "List:\n10\n20\n30")

    def test_preserve_standalone_numbers(self):
        """Test that standalone numbers which might look like page numbers are preserved."""
        text = "Page end.\n\n14\n\nNext page"
        cleaned = clean_text(text)
        self.assertIn("14", cleaned)
        
    def test_preserve_years(self):
        text = "Year:\n2023\nEnd"
        cleaned = clean_text(text)
        self.assertIn("2023", cleaned)

if __name__ == '__main__':
    unittest.main()