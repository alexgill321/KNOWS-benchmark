# Shared Eval Utils Directory

**Read this file in full before writing or modifying any evaluator.** It is a comprehensive directory of every shared utility function available in `eval_utils/`. Always reuse existing utilities rather than writing new code.

All imports use the pattern: `from src.browsergym.eval.eval_utils.<module> import <function>`

---

## 1. Scoring Framework (`scoring.py`)

Used by **all** evaluators. Every evaluator returns a `Result` containing `Checkpoint` objects.

### `Checkpoint(total, result, name=None)`
Container for one checkpoint's evaluation steps. Initialize with `result=0`, then call `add_step()` for each evaluation criterion.

### `Checkpoint.add_step(name, success, step_id, details=None, score=None, max_score=1)`
Records a single evaluation step. Automatically increments `self.result` by `score` (defaults to `max_score` on success, 0 on failure). Used in every evaluator to track individual pass/fail criteria — e.g., `sheets_6_investmenttracker/instance_1/evaluator.py:170` adds a step for each matched column.

### `Result(checkpoints, total_execution_time=None)`
Top-level result wrapping all checkpoints. Call `.to_dict()` for API output, `.get_detailed_report()` for step-by-step breakdown. Every evaluator's `grade_checkpoints()` returns this.

### `calculate_percentage_score(success_count, total_count, max_points=10) -> int`
Rounds percentage to nearest 10% and scales to `max_points`. Used for partial-credit checkpoints where you validate N items and want proportional scoring — e.g., checking 5 character slides and scoring based on how many passed.

---

## 2. Google Services Setup (`google_services_utils.py`)

Used by **16/18** evaluators. Provides Google API initialization and document operations.

### `initialize_google_services(service_type=None) -> tuple`
Initializes Google Drive + one document service (Docs/Sheets/Slides). Returns `(DRIVE_SERVICE, TYPE_SERVICE)`. Call with `service_type="sheets"`, `"docs"`, or `"slides"`. Used at the top of nearly every evaluator — e.g., `sheets_38_apartment_finder/instance_1/evaluator.py:52`.

### `extract_text_from_doc(doc_id, service) -> str`
Extracts all text content from a Google Doc. Used in `docs_1_formal_letter` evaluators to get the full document text for validation.

### `extract_structure_from_doc(doc_id, service) -> list`
Returns the document's structural elements (paragraphs, images, tables) in order. Used with the `layout` class in `docs_1_formal_letter` to verify element ordering — e.g., checking that a logo comes before the recipient name.

### `extract_images_from_doc(doc_id, service, output_dir=None) -> dict`
Downloads all images from a Google Doc to a temp directory. Returns dict mapping image IDs to file paths. Used in `docs_1_formal_letter/instance_1/evaluator.py` for logo and signature image extraction.

### `extract_images_from_doc_with_cropping(doc_id, service, output_dir=None) -> dict`
Same as above but also generates document-cropped variants of each image. Used when you need images as they appear in the rendered document (handles background/border differences).

### `download_doc_as_pdf(doc_id, output_file, service) -> str`
Exports a Google Doc as PDF. Used in `docs_1_formal_letter` before OCR — the PDF is then converted to PNGs for text extraction.

### `get_image_dimensions_from_doc(doc_id, image_uri, service) -> dict`
Returns the size of an image as it appears in the document (in points). Used with `extract_image_location_size()` for precise image position detection.

### `find_text_structure(text_to_match, doc_structure) -> dict`
Finds the structural element containing the best match for the given text. Used to locate specific text within a document's structure for ordering checks.

### `get_structural_element_order(doc_structure, texts) -> list`
Orders a list of texts by their appearance in the document structure. Used to verify that document sections appear in the expected sequence.

### `list_drive_folder_files(folder_id, service) -> list`
Lists all files in a Google Drive folder. Used in `slides_17_removeimagesaddplaceholders` to find files that were moved to Drive.

### `download_drive_file_as_image(file_id, service, output_path=None) -> Image`
Downloads a Drive file and returns as PIL Image. Used for extracting images from Drive for comparison.

### `download_drive_image_threadsafe(file_id, access_token) -> Image`
Thread-safe version using raw requests instead of the API client. Required when downloading images in parallel with `parallel_download()` — e.g., `slides_20_Illustrated_Book_Report/instance_1/evaluator.py`.

### `extract_drive_file_id(url) -> str`
Extracts the file ID from a Google Drive URL. Handles various URL formats (drive.google.com/file/d/..., docs.google.com/..., etc.).

### `search_doc(filename, service, folder_id=None) -> tuple`
Searches for a Google Doc by filename. Returns `(status, doc_id)` where status indicates if found in the expected location.

### `find_doc_specified_location(folder_id, filename, service)`
Finds a Google Doc in a specified folder by its filename. More targeted than `search_doc()` when you know the folder.

### `find_file_any(filename, service, file_type=None)`
Finds a Google Drive file by filename, optionally filtering by MIME type. Searches across all of Drive.

### `extract_images_from_doc_extended(service, output_dir=None, document=None, doc_id=None, include_positioned=False)`
Extended image extraction that also handles `positionedObjects` (images placed at specific page positions, e.g., side-by-side layouts). Accepts a pre-fetched document to avoid redundant API calls.

### `extract_hyperlinks_from_doc(doc_id, service, document=None) -> list`
Extracts all hyperlinks and plain text URLs from a Google Doc. Returns structured data with URL and anchor text. Handles both embedded hyperlinks (`textStyle.link.url`) and plain text URLs from paragraphs and tables.

### `download_drive_file_bytes(file_id, service) -> bytes`
Downloads a file from Google Drive and returns as raw bytes.

### `extract_text_colors_from_doc(document) -> Dict[str, Tuple[float, float, float]]`
Extracts text content with background colors as RGB tuples. Parses Google Docs JSON to find color-coded text. Used for tasks that require color-coded text validation.

---

## 3. Text Matching (`text_utils.py`)

Used by **13/18** evaluators. Provides exact, fuzzy, and LLM-based text matching.

### `keywords_match_robust(texts, keywords, model=None, description=None, substring=False) -> Optional[str]`
**Primary entry point for keyword matching.** Two-phase: exact match first, then LLM semantic fallback. Pass a list of candidate texts and keywords. Used extensively for column header matching and content validation — e.g., `sheets_7_running_analysis/instance_1/evaluator.py:681` matches chart axis labels against expected keywords.

### `keyword_exact_match(text, keyword, case_sensitive=False, standalone_line=False, substring=False) -> bool`
Strict exact match after normalization. Use `standalone_line=True` for header matching where trailing punctuation (`.,:;`) should be tolerated. Use `substring=True` for partial matches. Used in `slides_42_personal_none_product_comparison/instance_1/evaluator.py` for checking device names in slide text.

### `keywords_exact_match(text, keywords, ...) -> Optional[str]`
Matches text against a list of keywords. Returns the first matching keyword or `None`. Useful when multiple synonyms are acceptable.

### `keywords_llm_match(texts, keywords, model, description=None) -> Optional[str]`
Single LLM call to find semantic match. Called automatically by `keywords_match_robust()` as fallback — you rarely need to call this directly.

### `text_fuzzy_match_contained_short(query, larger_text) -> Optional[str]`
Fuzzy matching for short texts (a sentence or less). Uses sliding window with `rapidfuzz.WRatio`. Returns the best matching substring or `None`. Used in `docs_1_formal_letter` evaluators for matching name/address fields.

### `text_fuzzy_match_contained_long(target, full_text, threshold=85) -> tuple`
Fuzzy matching for long texts (paragraphs). Returns `(match, score)`. Uses sliding word-window with `partial_ratio`. Good for verifying that a paragraph appears in OCR output.

### `match_text_in_list(text, text_list, threshold=80) -> tuple`
Finds the best fuzzy match from a predefined list. Returns `(matched_item, score)`. Used in `slides_20_Illustrated_Book_Report` to match character names against a known list.

### `numerical_match_with_error(value1, value2, error_percent=5.0) -> tuple`
Compares numbers with percentage tolerance. Works for single values or lists. Used in `sheets_6_investmenttracker/instance_1/evaluator.py` to validate stock prices within 5% of expected values.

### `extract_text_from_pdf(pdf_images_path) -> dict`
OCR text extraction using DocTR. Takes a directory of page PNGs. Returns dict mapping page numbers to lists of lines with text and location info. Used in `docs_1_formal_letter` after `download_doc_as_pdf()` + `convert_pdf_to_pngs()`.

### `extract_text_location(ocr_result, text_to_find) -> location`
Finds the bounding box of specific text in OCR results (from `extract_text_from_pdf()`). Returns a `location` object. Used in `docs_1_formal_letter` to verify text positioning (e.g., address in upper-right corner).

### `fuzzy_match_text(text1, text2, threshold=80) -> tuple`
Simple fuzzy comparison between two texts using `token_sort_ratio`. Returns `(is_match, score)`. Good for comparing two short strings where word order may vary.

### `split_delimited_text(text, delimiters=None) -> list`
Splits text by multiple delimiters (default: `,`, `\n`, ` and `). Used for parsing author lists and comma-separated values.

### `normalize_name(name, remove_suffixes=True) -> str`
Normalizes names: lowercases, removes accents/diacritics, strips suffixes (Jr., III, etc.). Used in `sheets_10_paper_sorting` for author name comparison.

### `get_smallest_x_position(text_ocr) -> float`
Returns the leftmost x-coordinate across all OCR lines. Used in `docs_1_formal_letter` to check text alignment (e.g., verifying left-aligned paragraphs).

### `text_exact_match_contained(src_text, ref_text, standalone_line=False) -> bool`
**DEPRECATED** — use `keyword_exact_match()` instead.

---

## 4. Image Matching & Detection (`image_utils.py`)

Used by **12/18** evaluators. Provides pixel-level, perceptual, and AI-based image comparison.

### `match_image_tiered(candidate_path, gold_path, model=None, description="", hash_threshold=10) -> tuple`
**Primary entry point for image comparison.** Tries exact pixel match, then perceptual hash, then VLM comparison. Returns `(matched: bool, method: str)`. Used in `slides_39_Personal_Lookbook_PaintColors` for comparing paint swatch images.

### `binary_judge_image(model, image_path, text, examples=None) -> Optional[str]`
LLM yes/no classification of an image. Accepts a single image or directory of images. Returns the path of the first image that passes, or `None`. Used in `slides_20_Illustrated_Book_Report/instance_1/evaluator.py:213` to verify book cover images and in `slides_42/instance_1/evaluator.py:191` for device image verification.

### `binary_compare_images(model, image1_path, image2_path, mode="same") -> bool`
Compares two images using VLM. Modes: `"same"` (identical), `"similar"` (same subject), `"replacement"` (reasonable substitute), or a custom question string. Used as the VLM tier in `match_image_tiered()`.

### `image_exact_match(src_image_path, gld_image_path) -> Optional[str]`
Pixel-by-pixel comparison. Both args can be single files or directories. Returns the matched image path or `None`. Fast but strict — won't match compressed or resized versions.

### `perceptual_hash_match(img1_path, img2_path, threshold=10) -> bool`
Compares images using pHash algorithm. Robust to compression, resizing, and minor edits. Threshold 0 = identical, 1-10 = very similar, >20 = different.

### `extract_image_location(image_path, doc_path) -> location`
Finds an image in a document using scale-invariant template matching. `doc_path` is a directory of page PNGs. Returns a `location` object with page number and coordinates. Used in `docs_1_formal_letter` for verifying image positions.

### `extract_image_location_size(image_path, image_size, doc_path) -> location`
Same as above but uses known document dimensions for faster, more accurate matching. `image_size` comes from `get_image_dimensions_from_doc()`.

### `extract_image_location_size_feature_based(image_path, image_size, doc_path) -> location`
Uses SIFT feature matching instead of template matching. More robust for images that have been significantly transformed but slower.

### `convert_pdf_to_pngs(pdf_path, output_dir, dpi=300) -> int`
Converts PDF pages to PNG images using PyMuPDF. Returns number of pages. Used before OCR in `docs_1_formal_letter` — typically chained: `download_doc_as_pdf()` -> `convert_pdf_to_pngs()` -> `extract_text_from_pdf()`.

---

## 5. LLM Models (`models.py`)

Used by **11/18** evaluators. Loads model instances for VLM and text-based evaluation.

### `load_model(model_name) -> callable`
Loads and returns a model callable. Common model IDs: `"gemma-google-ai"` (Google AI API, most common), `"gemini-2.5-flash-google-ai"`, `"gemini-3-flash-google-ai"`. The returned callable accepts the standard message format. Used in every evaluator that needs LLM fallback — e.g., `sheets_6_investmenttracker/instance_1/evaluator.py:40`.

**Standard message format** (works with all models):
```python
messages = [
    {"role": "system", "content": [{"type": "text", "text": "..."}]},
    {"role": "user", "content": [
        {"type": "text", "text": "..."},
        {"type": "image", "image": "/path/to/image.png"}  # optional
    ]}
]
response = model(messages)  # returns str
```

---

## 6. LLM Response Utilities (`llm_utils.py`)

Used by **4+** task evaluators. Shared functions for parsing and extracting structured data from LLM responses.

### `strip_markdown_code_blocks(text) -> str`
Strips markdown code block fencing (` ```...``` `) from LLM response text. Returns the content between fences, or the original text if no code blocks found. Use whenever an LLM might wrap its response in code fences.

### `parse_yes_no(response) -> Optional[bool]`
Parses yes/no from an LLM response. Returns `True` if "yes" is in the response, `False` if "no", `None` if neither. Use for binary LLM judgments. Similar to `image_helpers.parse_response()` but intended as the public API for non-image contexts.

### `extract_json_from_llm_response(response, expect_type="auto") -> Optional[Union[Dict, List]]`
Extracts and parses JSON from raw LLM response text. Handles code block stripping, then applies type-specific extraction:
- `"auto"`: Direct `json.loads` after code block stripping.
- `"array"`: Bracket-matching to find outermost `[...]`.
- `"object"`: Finds outermost `{...}` using `find`/`rfind`.
Use when you have already called the model and need to parse the response.

### `extract_json_with_llm(prompt, model, system_prompt=None, content=None, expect_type="auto") -> Optional[Union[Dict, List]]`
**Full pipeline for LLM-based JSON extraction.** Builds standard messages, calls the model, then parses JSON from the response. Default system prompt instructs JSON-only output. Use `content` parameter for multi-modal messages (e.g., with images). Used in `slides_29_buy_car_pres`, `slides_42_personal_none_product_comparison`, and `sheets_38_apartment_finder`.

### `evaluate_with_llm(prompt, model, return_type="bool", system_prompt=None) -> Optional[Union[bool, str, Any]]`
**Multi-type LLM evaluation.** Sends a prompt to the model with return-type-specific instructions:
- `"bool"`: Returns `True`/`False` based on yes/no detection.
- `"str"`: Returns the raw lowercased response string.
- `"json"`: Parses and returns JSON from the response.
Used in `slides_29_buy_car_pres` and `slides_42_personal_none_product_comparison` for evaluating slide content against source data.

---

## 7. Table & Cell Operations (`table_utils.py`)

Used by **9/18** evaluators. Cell-level operations on Google Sheets raw data.

### `SheetTable` (dataclass)
Wraps a pandas DataFrame with sheet position metadata (`start_col`, `end_col`, `start_row`, `end_row`, `sheet_name`). Properties: `.columns`, `.num_rows`, `.col_letter`. Returned by `extract_sheet_data()` and `extract_tables_from_sheet()`.

### `match_columns(df, required_columns, model=None, strict=True, parallel=False) -> dict`
**Primary entry point for column matching.** Takes a DataFrame and list of `(col_name, keywords)` tuples. Returns dict mapping logical names to actual column names. Uses exact keyword matching first, then LLM fallback. Used in `sheets_6_investmenttracker/instance_1/evaluator.py:161` and `sheets_38_apartment_finder/instance_1/evaluator.py:137` to match spreadsheet columns.

### `get_cell_value(sheet_raw, row_idx, col_idx) -> str`
Gets a cell's formatted value from raw sheet data. Simple accessor for single-cell lookups.

### `get_cell(sheet_tab, row_idx, col_idx) -> dict`
Returns the full cell dict (with formatting, formulas, etc.) from a sheet tab.

### `read_column_values(sheet_tab, col_idx, start_row=0, end_row=None) -> list`
Reads all values in a single column. Useful for extracting a full column of data for validation.

### `get_background_color(sheet_raw, row_idx, col_idx=0) -> dict`
Returns the cell's background color as `{'red': float, 'green': float, 'blue': float}` (0-1 scale).

### `get_row_background_color(sheet_raw, row_idx) -> Optional[Dict]`
Gets background color of the first cell in a row. Alias for `get_background_color(sheet_raw, row_idx, 0)`.

### `get_cell_background_color(sheet_raw, row_idx, col_idx) -> Dict`
Gets background color of a specific cell. Alias for `get_background_color()`.

### `cell_bg_hex(sheet_raw, row_idx, col_idx) -> Optional[str]`
Returns cell background as `"#rrggbb"` hex string, or `None` for white/no fill.

### `colors_are_similar(c1, c2, tolerance=0.05) -> bool`
Compares two color dicts channel-by-channel. Used in `sheets_7_running_analysis` to verify consistent color-coding across rows.

### `colors_are_distinct(colors, tolerance=0.1) -> bool`
Checks that all colors in a list are distinct from each other.

### `classify_row_color(color_dict) -> str`
Classifies a color as `"yellow"`, `"orange"`, `"blue"`, `"green"`, `"red"`, or `"none"`. Designed for common Google Sheets highlight colors.

### `validate_color_grouping(row_colors) -> tuple`
Verifies that same-colored rows are grouped contiguously (not interleaved). Returns `(is_valid, message)`.

### Cell formatting checkers (all take a cell dict from the Sheets API):

| Function | Returns | Purpose |
|----------|---------|---------|
| `is_cell_bold(cell)` | `bool` | Check bold formatting |
| `is_cell_italic(cell)` | `bool` | Check italic formatting |
| `is_cell_centered(cell)` | `bool` | Check horizontal center alignment |
| `has_border(cell, edge="bottom")` | `bool` | Check border on specified edge |
| `has_bottom_border(cell)` | `bool` | Shorthand for `has_border(cell, "bottom")` |
| `has_top_border(cell)` | `bool` | Shorthand for `has_border(cell, "top")` |
| `row_has_border(row, edge="bottom")` | `bool` | Check if any cell in row has border |
| `row_has_bottom_border(row)` | `bool` | Shorthand for `row_has_border(row, "bottom")` |
| `row_has_top_border(row)` | `bool` | Shorthand for `row_has_border(row, "top")` |
| `count_bold_cells_in_row(row)` | `(bold, total)` | Count bold vs total non-empty cells |

### Merged cell utilities:

| Function | Purpose |
|----------|---------|
| `check_merged_cells(sheet_raw, expected_cols, row_start, row_end)` | Verify columns are merged across rows |
| `find_merged_cell_by_text(merges, rows, text_pattern)` | Find a merge containing text |
| `get_merge_column_span(merge)` | Get number of columns a merge spans |

### Text visibility and table comparison:

| Function | Purpose |
|----------|---------|
| `is_text_visible_in_cell(content, col_width, wrap_strategy, row_values, col_idx)` | Check if text is fully visible (not clipped) |
| `check_all_content_visible(sheet_raw_data, start_row, end_row, num_cols)` | Verify all table content is visible |
| `table_exact_match(df1, df2, ignore_case=False)` | Exact DataFrame comparison |
| `table_column_check(df, required_columns)` | Verify required columns exist |

### Image and index utilities:

| Function | Purpose |
|----------|---------|
| `extract_image_url_from_cell(cell_value)` | Extract image URL from cell (handles `=IMAGE()` formulas) |
| `get_image_url_from_raw_sheet_cell(sheet_raw, row_idx, col_idx)` | Extract image URL from raw API response |
| `get_column_index_by_name(df, col_name, matched_columns)` | Get column index from logical name |
| `get_sheet_row_index_from_dataframe_row(df_row, header_rows=1)` | Convert DataFrame row to raw sheet index |

---

## 8. Google Sheets Extraction (`google_sheets_utils.py`)

Used by **8/18** evaluators. Sheet-level data extraction and parsing.

### `extract_sheet_data(sheet_id, service, prefer_table_api=True, sheet_index=0, return_raw=False)`
**Primary entry point for sheet extraction.** Tries formal table API first, falls back to manual extraction. Returns `SheetTable` (or `(SheetTable, raw_dict)` if `return_raw=True`). Used in `sheets_7_running_analysis/instance_1/evaluator.py:100` and `sheets_38_apartment_finder/instance_1/evaluator.py`.

### `get_sheet_content(sheet_id, service) -> dict`
Fetches raw Google Sheets API response with `includeGridData=True`. Returns the complete spreadsheet structure. Used when you need raw cell data (formatting, formulas, colors) beyond what DataFrames provide.

### `get_sheet_by_name(sheet_raw, name) -> Optional[dict]`
Navigates to a specific tab by name (case-insensitive substring match). Returns the sheet-level dict. Use when a spreadsheet has multiple tabs.

### `extract_tables_from_sheet(sheet_id, service) -> list[SheetTable]`
Extracts tables using the formal Google Sheets tables API. Returns list of `SheetTable` objects with position metadata. Called by `extract_sheet_data()` as the preferred method.

### `parse_sheet_to_dataframe(sheet_raw, header_row=None) -> Optional[DataFrame]`
Converts raw sheet data to a pandas DataFrame. Auto-detects header row if not specified. Use when `extract_sheet_data()` isn't appropriate.

### `detect_header_row(rows, max_rows_to_check=10) -> int`
Heuristic detection of the header row. Scores rows by text ratio, density, and position. Called internally by other extraction functions.

### `search_sheet(filename, service, folder_id=None) -> tuple`
Searches for a Google Sheet by filename. Returns `(status, sheet_id)` — status 0=not found, 1=found anywhere, 2=found in specified folder.

### `find_urls_in_sheet(sheet_rows, start_row, num_rows=20, start_col=None, end_col=None) -> list`
Comprehensively finds URLs in cells — checks hyperlinks, formulas (`=HYPERLINK()`), text format runs, and plain text. Used in `sheets_7_running_analysis/instance_1/evaluator.py` to find source URLs.

### `extract_charts_from_sheet(sheet_id, service) -> list`
Extracts all charts with metadata (type, title, series, position, data ranges). Returns list of chart dicts. Used in `sheets_6_investmenttracker/instance_1/evaluator.py:74` and `sheets_7_running_analysis/instance_1/evaluator.py:125`.

### `extract_structure_from_sheet(sheet_id, service) -> list`
Returns cell list with metadata for layout checks. Each element has `row`, `col`, `value`, `format` fields.

---

## 9. Parallel Execution (`parallel_utils.py`)

Used by **7/18** evaluators. Thread-based parallelism for I/O-bound operations.

### `parallel_download(download_tasks, max_workers=3, use_rate_limit=True, max_retries=2) -> dict`
Parallel downloads with Google API rate limiting and retry logic. Tasks are dicts with `id`, `func`, `args`, `kwargs`. Used in `slides_20_Illustrated_Book_Report/instance_1/evaluator.py:493` for parallel URL fetching and in `sheets_38_apartment_finder/instance_1/evaluator.py:235` for Craigslist scraping.

### `parallel_execute(tasks, max_workers=3, semaphore=None) -> dict`
Generic parallel execution. Same task format as `parallel_download()` but without retry logic. Used for parallelizing LLM extraction calls.

### `fast_parallel_vlm_calls(vlm_tasks, model, max_workers=10) -> dict`
High-throughput parallel VLM calls without global semaphore. Tasks are dicts with `id` and `messages`. Returns dict mapping IDs to `bool` (True if response contains "yes"). Used in `slides_20_Illustrated_Book_Report/instance_1/evaluator.py:462` for batch bullet-point validation.

### `parallel_vlm_calls(vlm_tasks, model, max_workers=3) -> dict`
Rate-limited version of `fast_parallel_vlm_calls()` using `VLM_API_SEMAPHORE`. Use when the model/API can't handle high concurrency.

### `parallel_image_match(match_tasks, max_workers=5) -> dict`
Parallel exact + perceptual hash matching. Tasks have `id`, `candidate_path`, `gold_path`. Returns dict mapping IDs to `(matched, method)` tuples. Used in lookbook evaluators for bulk image comparison.

**Global rate limiters:**
- `GOOGLE_API_SEMAPHORE` (limit 2) — for Google API calls
- `VLM_API_SEMAPHORE` (limit 3) — for VLM model calls

---

## 10. Slides Content (`slides_utils.py`)

Used by **5/18** evaluators (all slides tasks). Extracts content from Google Slides API objects.

### `extract_slide_text(slide, separator=" ") -> str`
Extracts all text from shapes and tables on a slide. Used in virtually every slides evaluator — e.g., `slides_20_Illustrated_Book_Report/instance_1/evaluator.py:157` and `slides_42/instance_1/evaluator.py:158`.

### `extract_title_text(slide) -> str`
Extracts text from title placeholder or topmost text element. Checks placeholder types (`TITLE`, `CENTERED_TITLE`, `SUBTITLE`) first, then falls back to position.

### `extract_slide_images(slide, presentation_id, service) -> list`
Returns image metadata (objectId, contentUrl, transform, size) for all images on a slide.

### `extract_slide_links(slide) -> list`
Returns all URLs found in text (both hyperlinks and plain-text URLs). Deduplicates results.

### `extract_slide_links_with_positions(slide) -> list`
Same as above but includes bounding box position of each link. Returns list of `{'url': str, 'bbox': dict}`.

### `download_slide_image(image_url) -> Optional[Image]`
Downloads an image from a URL and returns as PIL Image.

### `extract_text_boxes_from_slide(slide) -> list`
Returns all non-empty text boxes with their bounding boxes. Each entry has `objectId`, `text`, `bbox`, `element`.

### `extract_table_from_slide(slide) -> Optional[dict]`
Extracts structured table data: `headers`, `rows` (as dicts), `cell_colors`, `num_columns`, `num_rows`. Used in `slides_42` for product comparison tables.

### `find_slide_by_title_fuzzy(presentation, title_text, threshold=80) -> Optional[int]`
Finds a slide by fuzzy-matching its text content. Returns 0-based slide index.

### `extract_bullet_point_texts(slide) -> list`
Returns text content of all bullet points on a slide.

### `validate_bullet_points(slide, min_count=3) -> tuple`
Checks if slide has at least `min_count` bullets. Returns `(passes, actual_count)`.

### Slide styling and positioning:

| Function | Purpose |
|----------|---------|
| `get_text_style_from_shape(shape)` | Extract color, font size, bold, italic from a shape |
| `is_text_color(text_style, r, g, b, tolerance=0.25)` | Check if text foreground is within Euclidean distance of target RGB (0-1 range per channel) |
| `is_text_big(text_style, min_pt=18)` | Check if font size >= threshold |
| `get_slide_background_color(slide, presentation=None)` | Get slide background RGB color |
| `colors_are_different(color1, color2, threshold=0.01)` | Check if two colors differ |
| `is_text_in_title_position(slide, text)` | Check if text is in title placeholder or top 20% |
| `is_text_at_bottom(slide, text)` | Check if text is in bottom 25% |
| `is_link_at_bottom(slide)` | Check if any link is in bottom 25% |
| `check_text_vertical_order(slide, text_list)` | Verify texts appear top-to-bottom in order |
| `get_element_bbox(element)` | Convert element transform+size to bbox dict (EMUs) |
| `get_slide_element_positions(slide)` | Get all text elements with y-positions |
| `find_url_below_image(image_bbox, links_with_positions)` | Find URL positioned below an image |
| `get_slide_dimensions(presentation_data)` | Get slide width/height in EMUs |
| `get_image_area_percentage_from_api(slide, ...)` | Calculate % of slide covered by images |
| `extract_image_source_urls(slide)` | Extract source URLs from image ALT text |

---

## 11. Web & URL Utilities (`web_utils.py`)

Used by **5/18** evaluators. URL validation and web content fetching.

### `validate_url_accessible(url, timeout=10, fallback_to_format=True) -> tuple`
Checks if a URL is reachable (HEAD, then GET fallback). Falls back to format validation for sites that block bots. Returns `(is_accessible, details)`. Used in `sheets_10_paper_sorting` and `sheets_45_wedding_planner` for validating hyperlinks.

### `validate_url_format(url) -> tuple`
Validates URL structure without HTTP requests. Returns `(is_valid, details)`. Use when you only need format checking.

### `download_image_from_url(url, temp_dir, timeout=15) -> Optional[str]`
Downloads an image to a temp directory. Returns the file path or `None`. Used in `slides_39_Personal_Lookbook_PaintColors` for downloading paint swatch images from URLs.

### `extract_id_from_url(url, patterns) -> Optional[str]`
Extracts an ID from a URL using regex patterns. Patterns are tried in order. Used in `docs_5_influential_papers` for extracting arXiv paper IDs from URLs.

### `is_url_from_domain(url, domain) -> bool`
Checks if URL belongs to a specific domain (case-insensitive). Used in `sheets_2_personal_recipe_foodcomposition` to verify USDA links.

### `is_unverifiable_url(url) -> bool`
Checks if URL is from a domain known to block programmatic downloads (Wikipedia, Instagram, Pinterest, etc.). Use to skip validation for known-blocked domains.

### `fetch_api_with_retry(url, timeout=10, max_retries=3, headers=None) -> Optional[dict]`
Fetches JSON from an API with exponential backoff on rate limits (429). Used for external API calls like Semantic Scholar.

### `fetch_page_text_content(url, timeout=10, max_chars=15000) -> tuple`
Fetches a URL and converts HTML to clean text (removes scripts, nav, footer). Returns `(text, status)`. Used for extracting content from linked web pages.

### `fetch_page_text_content_playwright(url, max_chars=15000, timeout=10) -> tuple`
Playwright-based version of `fetch_page_text_content()`. Renders JS-heavy pages with headless Chromium before extracting content. Returns `(text, method)`. Use for JavaScript-heavy sites (Khan Academy, LibreTexts, etc.) where basic `requests.get()` returns incomplete content.

### `fetch_with_fallbacks(url, max_chars=15000, timeout=15) -> Tuple[Optional[str], str]`
Fetches URL content with multiple fallback strategies. Tries in order: (1) Playwright with stealth, (2) Playwright retry with longer timeout, (3) Wayback Machine archived snapshot. Returns `(content, method_used)`.

### `fetch_page_title(url, timeout=10) -> Optional[str]`
Extracts the `<title>` or `<h1>` from a web page. Used for verifying that linked pages have expected titles.

### `normalize_url_for_comparison(url) -> str`
Normalizes a URL for consistent comparison (e.g., with browsing history). Removes trailing slashes, query parameters, URL fragments, `www.` prefix, and lowercases the domain.

### `fetch_url_content(url) -> Optional[str]`
Uses Playwright (headless browser) for JavaScript-rendered pages. Returns markdown content (up to 60k chars). Use for JS-heavy sites like Fandom wikis where `fetch_page_text_content()` would fail.

### `download_page_images(url, folder, timeout=10, headers=None)`
Downloads all images from a webpage to a specified folder. Returns list of downloaded filenames.

---

## 12. Chart Validation (`chart_utils.py`)

Used by **2/18** evaluators (investment tracker, running analysis). Validates chart data against expected values.

### `extract_chart_domain_data(chart, table_data) -> list`
Extracts x-axis category labels from a chart using its data range. Used in `sheets_6_investmenttracker` to get stock ticker labels.

### `extract_chart_series_data(chart, table_data) -> list`
Extracts data values for each series in a chart. Returns list of series data.

### `validate_chart_categories_match(chart, expected_categories, ...) -> tuple`
Verifies chart x-axis labels match expected values. Returns `(matches, details)`.

### `validate_chart_values_match(chart, expected_values, ...) -> tuple`
Verifies chart data values match expected values within tolerance. Returns `(matches, details)`.

### `get_chart_type(chart) -> str`
Returns the chart type (e.g., `"LINE"`, `"COLUMN"`, `"PIE"`).

### `identify_series_by_content(chart, rows, keywords, ...) -> Optional[int]`
Finds a specific series by matching its header label or values against keywords. Used in `sheets_7_running_analysis` to identify which series represents pace data.

### `identify_chart_vlm(chart_image_1, chart_image_2, description, model) -> str`
Uses a VLM to identify which of two chart images best matches a given description. Returns the identification result.

### `get_series_source_range(chart, series_index) -> Optional[Dict[str, int]]`
Gets the source data range for a specific series in a chart.

### `find_chart_by_metadata(charts, title_keywords, y_axis_keywords, ...) -> Optional[Dict]`
Finds a chart by matching metadata. Matching order: (1) chart title matches title_keywords, (2) y-axis label matches y_axis_keywords, (3) series data matches column data. Supports LLM-based fallback matching.

### Series metadata functions:

| Function | Purpose |
|----------|---------|
| `get_series_header_label(chart, series_index, rows)` | Get the column header for a series |
| `get_series_column_values(chart, series_index, rows)` | Get all values for a series |
| `get_series_color(chart, series_index)` | Get series color as RGB dict |
| `get_series_line_style(chart, series_index)` | Get line style (SOLID, DASHED, etc.) |
| `validate_constant_series(chart, series_index, rows, ...)` | Verify series values are constant |
| `get_all_series_metadata(chart)` | Get metadata for all series at once |
| `get_chart_axis_labels(chart)` | Get horizontal/vertical axis label text |
| `check_chart_overlap(charts_list)` | Check if any charts spatially overlap |
| `check_point_shape(chart, chart_type)` | Get data point marker style |
| `debug_chart_structure(chart)` | Print chart structure for debugging |

---

## 13. Core Location Utilities (`utils.py`)

Used by **3/18** evaluators (primarily formal letter docs). Bounding box and document layout utilities.

### `location(page_number, x, y, width, height)`
Bounding box class for document elements (assumes 300 DPI, 2550x3300 pixel pages). Methods:
- `is_mostly_inside(other, cutoff=0.6)` — checks area overlap percentage
- `is_upper_left()`, `is_upper_right()`, `is_lower_left()`, `is_lower_right()` — quadrant checks
- `is_upper()`, `is_lower()` — half-page checks
- `is_inside(other)` — strict containment check
- `merge_locations(locations)` (static) — combines multiple locations into one bounding box

### `layout(element, element_type, doc_structure)`
Document structure ordering. `element_type` is `"text"` or `"image"`. Methods:
- `comes_before(other_element, other_type)` — checks ordering
- `comes_after(other_element, other_type)` — checks ordering
- `at_start()`, `at_end()` — checks position in document

Used in `docs_1_formal_letter` to verify that the logo comes before the sender name, the signature comes after the closing, etc.

### `bbox_ratio_to_location(bbox_ratio, page_number, image_width, image_height) -> location`
Converts `[x1, y1, x2, y2]` ratios (0-1) to a `location` with absolute pixel coordinates. Used internally by OCR functions.

### `bbox_overlap_ratio(bbox1, bbox2) -> float`
Calculates what fraction of bbox1 overlaps with bbox2. Works with dict bboxes (`{x, y, width, height}`).

### `is_bbox_mostly_inside(inner_bbox, outer_bbox, threshold=0.6) -> bool`
Checks if inner bbox has sufficient overlap with outer bbox.

---

## Internal Helper Files (Do Not Import Directly)

These files provide internal implementations and should not be imported by evaluators:

- **`image_helpers.py`** — Low-level image processing (`find_template_scale_invariant`, `load_process_images`, `parse_response`). Used internally by `image_utils.py`.
- **`text_helpers.py`** — Text preprocessing (`preprocess_text`). Used internally by `text_utils.py`.
- **`google_services_helpers.py`** — Google auth and low-level API helpers (`authenticate`, `get_scopes`, `get_doc_content`). Used internally by `google_services_utils.py` and `google_sheets_utils.py`.
