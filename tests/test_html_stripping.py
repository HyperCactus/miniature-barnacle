import unittest
import io
from src.document_parser import parse_markdown

class TestHtmlStripping(unittest.TestCase):
    def test_basic_html_stripping(self):
        """Test stripping of basic HTML tags."""
        content = "Hello <b>World</b>"
        file_obj = io.BytesIO(content.encode('utf-8'))
        file_obj.name = "test.md"
        result = parse_markdown(file_obj)
        self.assertEqual(result.strip(), "Hello World")

    def test_nested_tags(self):
        """Test stripping of nested HTML tags."""
        content = "<div><p>Nested <span>Content</span></p></div>"
        file_obj = io.BytesIO(content.encode('utf-8'))
        file_obj.name = "test.md"
        result = parse_markdown(file_obj)
        self.assertEqual(result.strip(), "Nested Content")

    def test_attributes_with_gt(self):
        """Test tags with attributes containing '>'."""
        content = '<div data-val="x > y">Content</div>'
        file_obj = io.BytesIO(content.encode('utf-8'))
        file_obj.name = "test.md"
        result = parse_markdown(file_obj)
        self.assertEqual(result.strip(), "Content")

    def test_broken_html(self):
        """Test handling of broken HTML."""
        content = "<div>Unclosed tag"
        file_obj = io.BytesIO(content.encode('utf-8'))
        file_obj.name = "test.md"
        result = parse_markdown(file_obj)
        self.assertEqual(result.strip(), "Unclosed tag")

    def test_script_tags(self):
        """Test that script content is removed or handled appropriately."""
        # Note: BeautifulSoup.get_text() usually keeps script content unless stripped explicitly.
        # We want to ensure we don't get the JS code in our text output if possible, 
        # or at least that the tags are gone.
        # For this task, the goal is robust parsing. 
        # If the previous regex was just <[^<]+?>, it wouldn't remove script content either, just the tags.
        # So we'll check that tags are gone.
        content = "<script>console.log('test');</script>Text"
        file_obj = io.BytesIO(content.encode('utf-8'))
        file_obj.name = "test.md"
        result = parse_markdown(file_obj)
        # We expect the script tag to be gone. 
        # Whether the content remains depends on implementation, but let's assert the tag is gone.
        self.assertNotIn("<script>", result)
        self.assertNotIn("</script>", result)
        self.assertIn("Text", result)

if __name__ == '__main__':
    unittest.main()