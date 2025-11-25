import unittest
from werkzeug.utils import secure_filename

class TestFilenameSanitization(unittest.TestCase):
    def test_normal_filename(self):
        filename = "normal.pdf"
        sanitized = secure_filename(filename)
        self.assertEqual(sanitized, "normal.pdf")

    def test_filename_with_spaces(self):
        filename = "file with spaces.pdf"
        sanitized = secure_filename(filename)
        self.assertEqual(sanitized, "file_with_spaces.pdf")

    def test_path_traversal(self):
        filename = "../../malicious.pdf"
        sanitized = secure_filename(filename)
        # secure_filename should strip directory traversal
        self.assertNotIn("..", sanitized)
        self.assertNotIn("/", sanitized)
        self.assertNotIn("\\", sanitized)
        self.assertEqual(sanitized, "malicious.pdf")

    def test_complex_malicious_filename(self):
        filename = "../../../etc/passwd"
        sanitized = secure_filename(filename)
        self.assertNotIn("..", sanitized)
        self.assertEqual(sanitized, "etc_passwd")

if __name__ == '__main__':
    unittest.main()