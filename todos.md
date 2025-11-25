### 1. Critical Bugs &amp; Stability

*   **Naive Sentence Splitting (`src/doc_reader.py`)**
    *   **Issue:** The code uses `text.split('. ')` to chunk text. This is fragile and will incorrectly split on abbreviations (e.g., "Mr. Smith", "U.S.A.", "Fig. 1"). This results in fragmented audio where sentences are cut off mid-thought.
    *   **Fix:** Use a robust sentence tokenizer like `nltk.sent_tokenize` or `spacy`.

*   **Fragile LLM Response Parsing (`src/doc_reader.py`)**
    *   **Issue:** In `clean_chunk`, you parse the output by looking for the string `"assistant"`. If the model generates text containing the word "assistant" (e.g., "The assistant manager said..."), your parsing logic might cut off the actual content.
    *   **Fix:** Since you are using `tokenizer.apply_chat_template`, the model should output the response directly. If you need to parse raw output, use specific special tokens (like `&lt;|im_start|&gt;assistant`) or rely on the `messages` structure more strictly.

*   **Unsafe Filename Handling (`app.py`)**
    *   **Issue:** `audio_path` is constructed using `uploaded_file.name`. If a user uploads a file with malicious characters or path traversal sequences (e.g., `../../file.pdf`), it could write files outside the intended directory.
    *   **Fix:** Sanitize filenames using `werkzeug.utils.secure_filename` or similar before using them in file paths.

*   **Race Conditions in Voice Manager (`src/voice_manager.py`)**
    *   **Issue:** Streamlit is multi-threaded. If two users (or two tabs) try to add/delete voices simultaneously, `voices.json` could get corrupted because there is no file locking mechanism.
    *   **Fix:** Use a file lock (e.g., `filelock` library) when reading/writing `voices.json`.

### 2. Performance Improvements

*   **Inefficient Audio Concatenation (`src/doc_reader.py`)**
    *   **Issue:** The loop `combined += chunk_audio` creates a new `AudioSegment` object in memory for every iteration. For long documents with hundreds of chunks, this becomes exponentially slower ($O(N^2)$ complexity).
    *   **Fix:** Collect all segments in a list and use `sum(chunk_files, AudioSegment.empty())` or `reduce`, which is much faster.

*   **Unnecessary Disk I/O (`src/tts_gen_chaterbox_local.py` &amp; `src/doc_reader.py`)**
    *   **Issue:** The TTS engine saves every chunk to disk (`ta.save`), and then `doc_reader` reads it back (`AudioSegment.from_wav`). This adds significant latency, especially for HDD users.
    *   **Fix:** Modify `ChatterboxLocal.generate` to return the audio tensor or bytes directly in memory, and construct the `AudioSegment` from that data without writing to disk.

### 3. Code Quality &amp; Best Practices

*   **Regex HTML Stripping (`src/document_parser.py`)**
    *   **Issue:** Using regex to strip HTML (`re.sub('&lt;[^&lt;]+?&gt;', '', html)`) is notoriously unreliable.
    *   **Fix:** Use `BeautifulSoup` (from `bs4`) which is designed for this and handles edge cases correctly.

*   **Hardcoded Page Number Removal (`src/document_parser.py`)**
    *   **Issue:** `re.sub(r'\n\d+\n', '\n', text)` removes any line that is just a number. This might accidentally delete valid content (e.g., a quantity in a list, a year, or a data point).

* Improve readme!