"""Utility functions for sheets_10_paper_sorting evaluator.

This module contains functions for:
- URL validation (arXiv, Google Drive)
- Author name handling and comparison
- Figure 1 extraction from arXiv source files
- Row color/formatting detection
- Preprocessing helpers (Google Scholar scraping, arXiv search)
"""

import os
import re
import sys
import tarfile
import gzip
import tempfile
import unicodedata
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urlparse, parse_qs
import requests

# Base path setup
def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    elif os.path.exists("/scratch"):
        return "/scratch/general/vast/USER/Agent-Benchmark/"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

from rapidfuzz import fuzz

# ============================================================================
# URL Validation Functions
# ============================================================================

def extract_arxiv_id_from_url(url: str) -> Optional[str]:
    """Extract arXiv ID from a URL.

    Handles various URL formats:
    - https://arxiv.org/abs/2301.12345
    - https://arxiv.org/pdf/2301.12345.pdf
    - http://arxiv.org/abs/2301.12345v1
    - https://ar5iv.org/abs/2301.12345

    Args:
        url: The arXiv URL to parse.

    Returns:
        The arXiv ID (e.g., "2301.12345") or None if not found.
    """
    if not url:
        return None

    # Patterns for arXiv IDs
    # New format: YYMM.NNNNN (e.g., 2301.12345)
    # Old format: category/YYMMNNN (e.g., cs.CV/0601001)
    patterns = [
        r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})',  # New format
        r'arxiv\.org/(?:abs|pdf)/([\w\-\.]+/\d+)',    # Old format
        r'ar5iv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})',   # ar5iv mirror
        r'(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?$',      # Just the ID
    ]

    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            arxiv_id = match.group(1)
            # Remove version suffix if present
            arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
            return arxiv_id

    return None


def validate_arxiv_url(url: str) -> Tuple[bool, Optional[str], str]:
    """Validate an arXiv URL and extract the ID.

    Args:
        url: The URL to validate.

    Returns:
        Tuple of (is_valid, arxiv_id, message).
    """
    if not url:
        return False, None, "Empty URL"

    arxiv_id = extract_arxiv_id_from_url(url)
    if arxiv_id:
        return True, arxiv_id, f"Valid arXiv URL with ID: {arxiv_id}"
    else:
        return False, None, f"Could not extract arXiv ID from: {url}"


def extract_drive_file_id(url: str) -> Optional[str]:
    """Extract Google Drive file ID from a URL.

    Handles various URL formats:
    - https://drive.google.com/file/d/FILE_ID/view
    - https://drive.google.com/open?id=FILE_ID
    - https://docs.google.com/document/d/FILE_ID/edit

    Args:
        url: The Google Drive URL to parse.

    Returns:
        The file ID or None if not found.
    """
    if not url:
        return None

    # Pattern 1: /d/FILE_ID/
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)

    # Pattern 2: ?id=FILE_ID
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    if 'id' in query_params:
        return query_params['id'][0]

    return None


def validate_drive_url(url: str, expected_folder_id: Optional[str] = None) -> Tuple[bool, Optional[str], str]:
    """Validate a Google Drive URL and optionally check folder membership.

    Args:
        url: The Google Drive URL to validate.
        expected_folder_id: Optional folder ID to verify the file belongs to.

    Returns:
        Tuple of (is_valid, file_id, message).
    """
    if not url:
        return False, None, "Empty URL"

    file_id = extract_drive_file_id(url)
    if not file_id:
        return False, None, f"Could not extract file ID from: {url}"

    # If folder validation is needed, it would require API call
    # For now, just validate the URL format
    return True, file_id, f"Valid Drive URL with file ID: {file_id}"


# ============================================================================
# Author Handling Functions
# ============================================================================

def normalize_author_name(name: str) -> str:
    """Normalize an author name for comparison.

    - Converts to lowercase
    - Removes accents/diacritics
    - Removes extra whitespace
    - Handles common variations (Jr., III, etc.)

    Args:
        name: The author name to normalize.

    Returns:
        Normalized author name.
    """
    if not name:
        return ""

    # Convert to lowercase
    name = name.lower().strip()

    # Remove accents/diacritics
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))

    # Remove common suffixes
    suffixes = [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    # Normalize whitespace
    name = ' '.join(name.split())

    return name


def parse_authors_string(authors_str: str) -> List[str]:
    """Parse a comma-separated author string into a list.

    Handles:
    - Comma-separated names
    - "and" as separator
    - Newlines within the string

    Args:
        authors_str: The author string to parse.

    Returns:
        List of individual author names.
    """
    if not authors_str:
        return []

    # Replace newlines with commas
    authors_str = authors_str.replace('\n', ', ')

    # Replace " and " with comma
    authors_str = re.sub(r'\s+and\s+', ', ', authors_str, flags=re.IGNORECASE)

    # Split by comma
    authors = [a.strip() for a in authors_str.split(',')]

    # Remove empty strings
    authors = [a for a in authors if a]

    return authors


def extract_first_author(authors: List[str]) -> Optional[str]:
    """Extract the first author from a list.

    Args:
        authors: List of author names.

    Returns:
        First author name or None if list is empty.
    """
    if not authors:
        return None
    return authors[0]


def compare_authors_list(user_authors: List[str], gold_authors: List[str],
                         strict: bool = False) -> Tuple[bool, str]:
    """Compare two author lists.

    Args:
        user_authors: List of authors from user's spreadsheet.
        gold_authors: List of gold standard authors.
        strict: If True, requires exact order and count match.

    Returns:
        Tuple of (is_match, details_message).
    """
    if not user_authors and not gold_authors:
        return True, "Both author lists are empty"

    if not user_authors:
        return False, "User author list is empty"

    if not gold_authors:
        return False, "Gold author list is empty"

    # Normalize all names
    user_normalized = [normalize_author_name(a) for a in user_authors]
    gold_normalized = [normalize_author_name(a) for a in gold_authors]

    if strict:
        # Exact match required
        if user_normalized == gold_normalized:
            return True, f"Exact match: {len(user_authors)} authors"
        else:
            return False, f"Authors don't match exactly. User: {user_authors[:3]}..., Gold: {gold_authors[:3]}..."
    else:
        # Check if first author matches and most authors are present
        first_match = user_normalized[0] == gold_normalized[0] if user_normalized and gold_normalized else False

        # Count matches
        matches = sum(1 for u in user_normalized if u in gold_normalized)
        match_ratio = matches / len(gold_normalized) if gold_normalized else 0

        if first_match and match_ratio >= 0.8:
            return True, f"First author matches, {matches}/{len(gold_normalized)} authors found"
        elif first_match:
            return False, f"First author matches but only {matches}/{len(gold_normalized)} authors found"
        else:
            return False, f"First author doesn't match. User: {user_authors[0] if user_authors else 'N/A'}, Gold: {gold_authors[0] if gold_authors else 'N/A'}"


# ============================================================================
# Figure 1 Extraction Functions
# ============================================================================

def download_arxiv_source(arxiv_id: str, output_dir: str, timeout: int = 60) -> Tuple[bool, str, List[str]]:
    """Download arXiv source files (tar.gz) for a paper.

    URL format: https://arxiv.org/e-print/{arxiv_id}

    Args:
        arxiv_id: The arXiv paper ID.
        output_dir: Directory to extract files to.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (success, message, list_of_extracted_files).
    """
    url = f"https://arxiv.org/e-print/{arxiv_id}"

    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')

        # Create output directory
        paper_dir = os.path.join(output_dir, arxiv_id.replace('/', '_'))
        os.makedirs(paper_dir, exist_ok=True)

        extracted_files = []

        if 'application/x-eprint-tar' in content_type or 'application/gzip' in content_type:
            # It's a gzipped tar file
            tar_path = os.path.join(paper_dir, 'source.tar.gz')
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract tar.gz
            try:
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=paper_dir)
                    extracted_files = [os.path.join(paper_dir, m.name) for m in tar.getmembers() if m.isfile()]
            except tarfile.TarError:
                # Maybe it's just gzipped, not tar
                try:
                    with gzip.open(tar_path, 'rb') as gz:
                        content = gz.read()
                        tex_path = os.path.join(paper_dir, 'main.tex')
                        with open(tex_path, 'wb') as f:
                            f.write(content)
                        extracted_files = [tex_path]
                except Exception as e:
                    return False, f"Failed to extract gzip: {e}", []

            # Clean up tar file
            if os.path.exists(tar_path):
                os.remove(tar_path)

        elif 'application/x-tex' in content_type or 'text/plain' in content_type:
            # Single TeX file
            tex_path = os.path.join(paper_dir, 'main.tex')
            with open(tex_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            extracted_files = [tex_path]
        else:
            # Unknown format, save as-is
            raw_path = os.path.join(paper_dir, 'source_raw')
            with open(raw_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            extracted_files = [raw_path]

        return True, f"Downloaded and extracted {len(extracted_files)} files", extracted_files

    except requests.RequestException as e:
        return False, f"Failed to download: {e}", []
    except Exception as e:
        return False, f"Error processing source: {e}", []


def find_tex_files(source_dir: str) -> List[str]:
    """Find all .tex files in a source directory.

    Args:
        source_dir: Directory containing extracted arXiv source.

    Returns:
        List of paths to .tex files, with main.tex first if it exists.
    """
    tex_files = []
    main_tex = None

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.tex'):
                filepath = os.path.join(root, file)
                if file.lower() == 'main.tex':
                    main_tex = filepath
                else:
                    tex_files.append(filepath)

    # Put main.tex first if it exists
    if main_tex:
        tex_files.insert(0, main_tex)

    return tex_files


def find_png_files(source_dir: str) -> List[str]:
    """Find all PNG image files in a source directory.

    Args:
        source_dir: Directory containing extracted arXiv source.

    Returns:
        List of paths to PNG files.
    """
    png_files = []

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))

    return png_files


def parse_latex_for_figure_1(tex_content: str) -> Optional[str]:
    """Parse LaTeX content to find Figure 1 and extract its image filename.

    Looks for patterns like:
    - \\begin{figure} ... \\includegraphics{filename} ... \\label{fig:1} ... \\end{figure}
    - \\begin{figure} ... \\includegraphics{filename} ... \\caption{...Figure 1...} ... \\end{figure}
    - The first figure environment in the document (often Figure 1)

    Args:
        tex_content: Content of a .tex file.

    Returns:
        The image filename referenced in Figure 1, or None if not found.
    """
    # Remove comments (lines starting with %)
    lines = tex_content.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove inline comments but keep the rest
        comment_idx = line.find('%')
        if comment_idx == 0:
            continue  # Skip full comment lines
        elif comment_idx > 0:
            # Check if % is escaped
            if line[comment_idx - 1] != '\\':
                line = line[:comment_idx]
        cleaned_lines.append(line)
    tex_content = '\n'.join(cleaned_lines)

    # Find all figure environments
    # Pattern to match \begin{figure} ... \end{figure} (including figure*)
    figure_pattern = r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}'
    figure_matches = re.findall(figure_pattern, tex_content, re.DOTALL | re.IGNORECASE)

    if not figure_matches:
        return None

    # For each figure, check if it's Figure 1
    for idx, figure_content in enumerate(figure_matches):
        is_figure_1 = False

        # Check for label like \label{fig:1}, \label{fig1}, \label{figure1}
        label_patterns = [
            r'\\label\{fig:1\}',
            r'\\label\{fig1\}',
            r'\\label\{figure1\}',
            r'\\label\{fig:one\}',
            r'\\label\{fig_1\}',
        ]
        for lp in label_patterns:
            if re.search(lp, figure_content, re.IGNORECASE):
                is_figure_1 = True
                break

        # Check caption for "Figure 1" or if this is the first figure
        if not is_figure_1:
            caption_match = re.search(r'\\caption\{([^}]*)\}', figure_content, re.DOTALL)
            if caption_match:
                caption_text = caption_match.group(1)
                # Check if caption explicitly mentions "Figure 1" or similar
                if re.search(r'figure\s*1\b', caption_text, re.IGNORECASE):
                    is_figure_1 = True

        # If this is the first figure environment, assume it's Figure 1
        if idx == 0:
            is_figure_1 = True

        if is_figure_1:
            # Extract includegraphics filename
            # Patterns: \includegraphics{file}, \includegraphics[options]{file}
            includegraphics_pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
            img_match = re.search(includegraphics_pattern, figure_content)

            if img_match:
                filename = img_match.group(1).strip()
                # Remove any path components to get just the filename
                # But keep subdirectory info as it might be needed
                return filename

    return None


def resolve_image_path(image_ref: str, source_dir: str, png_only: bool = True) -> Optional[str]:
    """Resolve an image reference from LaTeX to an actual file path.

    LaTeX \includegraphics often omits the extension, so we need to find
    the actual file.

    Args:
        image_ref: Image reference from LaTeX (may lack extension).
        source_dir: Directory containing extracted arXiv source.
        png_only: If True, only return PNG files.

    Returns:
        Full path to the image file, or None if not found.
    """
    # Clean up the reference
    image_ref = image_ref.strip()

    # Common image extensions to try (in order of preference for PNG-only mode)
    if png_only:
        extensions_to_try = ['.png', '']  # Empty string if ref already has extension
    else:
        extensions_to_try = ['.png', '.jpg', '.jpeg', '.pdf', '.eps', '.svg', '']

    # Get the base path without extension if it has one
    base_ref = image_ref
    for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.eps', '.svg', '.gif']:
        if image_ref.lower().endswith(ext):
            base_ref = image_ref[:-len(ext)]
            # If it's a non-PNG extension and we want PNG only, skip
            if png_only and ext != '.png':
                return None
            break

    # Search for the file
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_lower = file.lower()
            file_base = os.path.splitext(file)[0]
            file_ext = os.path.splitext(file)[1].lower()

            # Check if this file matches the reference
            for ext in extensions_to_try:
                # Try direct match
                if file_lower == (base_ref + ext).lower():
                    if png_only and file_ext != '.png':
                        continue
                    return os.path.join(root, file)

                # Try matching just the filename (without subdirectory in ref)
                ref_basename = os.path.basename(base_ref)
                if file_base.lower() == ref_basename.lower():
                    if png_only and file_ext != '.png':
                        continue
                    return os.path.join(root, file)

    return None


def extract_figure_1_with_latex_parsing(source_dir: str) -> Tuple[bool, Optional[str], str]:
    """Stage 1: Extract Figure 1 by parsing LaTeX files.

    Args:
        source_dir: Directory containing extracted arXiv source.

    Returns:
        Tuple of (success, figure_1_path, message).
    """
    tex_files = find_tex_files(source_dir)

    if not tex_files:
        return False, None, "No .tex files found in source"

    # Try each tex file
    for tex_path in tex_files:
        try:
            with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
                tex_content = f.read()
        except Exception as e:
            continue

        # Parse for Figure 1
        image_ref = parse_latex_for_figure_1(tex_content)

        if image_ref:
            # Try to resolve the image reference to an actual PNG file
            resolved_path = resolve_image_path(image_ref, source_dir, png_only=True)

            if resolved_path:
                return True, resolved_path, f"Found Figure 1 via LaTeX parsing: {os.path.basename(resolved_path)}"
            else:
                # Found reference but couldn't resolve to PNG
                return False, None, f"Found Figure 1 reference '{image_ref}' but no matching PNG file exists"

    return False, None, "Could not find Figure 1 in LaTeX files"


def extract_figure_1_with_llm(source_dir: str, model) -> Tuple[bool, Optional[str], str]:
    """Stage 2: Extract Figure 1 by using LLM to read .tex files.

    The LLM iteratively reads tex files starting with main.tex to find
    Figure 1 and identify its associated image file.

    Args:
        source_dir: Directory containing extracted arXiv source.
        model: LLM model to use for parsing.

    Returns:
        Tuple of (success, figure_1_path, message).
    """
    tex_files = find_tex_files(source_dir)
    png_files = find_png_files(source_dir)

    if not tex_files:
        return False, None, "No .tex files found in source"

    if not png_files:
        return False, None, "No PNG files found in source"

    # Create a summary of available PNG files
    png_basenames = [os.path.basename(p) for p in png_files]
    png_list_str = '\n'.join(f"  - {name}" for name in png_basenames)

    # Read tex files and ask LLM to identify Figure 1
    for tex_path in tex_files[:3]:  # Limit to first 3 tex files
        try:
            with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
                tex_content = f.read()
        except Exception:
            continue

        # Truncate if too long
        if len(tex_content) > 50000:
            tex_content = tex_content[:50000] + "\n... [truncated]"

        tex_filename = os.path.basename(tex_path)

        # Ask LLM to find Figure 1
        prompt = f"""Analyze this LaTeX file and identify the image file used for Figure 1.

Available PNG files in the source:
{png_list_str}

LaTeX file ({tex_filename}):
```latex
{tex_content}
```

Task: Find the \\begin{{figure}} environment that contains Figure 1 (usually the first figure, or one with \\label{{fig:1}} or similar).
Look for the \\includegraphics command inside that figure environment and identify which image file it references.

Important:
- Match the image reference to one of the available PNG files listed above
- The LaTeX reference might omit the .png extension
- If you cannot find Figure 1 or it doesn't use a PNG file, respond with "NOT_FOUND"

Respond with ONLY the PNG filename (e.g., "figure1.png") or "NOT_FOUND". Nothing else."""

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a LaTeX parsing assistant. Extract the requested information precisely."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]

        try:
            response = model(messages)
            response_text = response.strip()

            if response_text and response_text != "NOT_FOUND":
                # Try to match response to an actual PNG file
                response_lower = response_text.lower()

                # Remove any quotes or extra text
                response_clean = re.sub(r'["\']', '', response_lower).strip()

                # Find matching PNG
                for png_path in png_files:
                    png_basename = os.path.basename(png_path).lower()
                    png_noext = os.path.splitext(png_basename)[0]

                    if (response_clean == png_basename or
                        response_clean == png_noext or
                        png_basename in response_clean or
                        response_clean in png_basename):
                        return True, png_path, f"Found Figure 1 via LLM: {os.path.basename(png_path)}"

        except Exception as e:
            continue

    return False, None, "LLM could not identify Figure 1 PNG file"


def extract_figure_1_from_source(source_dir: str, arxiv_id: str,
                                  model=None) -> Tuple[bool, Optional[str], str]:
    """Extract Figure 1 PNG image from arXiv source files using 2-stage approach.

    Stage 1: Automatically parse LaTeX files to find Figure 1 and its image file
    Stage 2: If Stage 1 fails and model provided, use LLM to read .tex files

    Only returns PNG images. Returns None if no PNG for Figure 1 exists.

    Args:
        source_dir: Directory containing extracted arXiv source.
        arxiv_id: The arXiv paper ID (for logging).
        model: Optional LLM model for Stage 2 fallback.

    Returns:
        Tuple of (success, figure_1_path, message).
        success=True and figure_1_path=None means Figure 1 doesn't have a PNG.
    """
    # Stage 1: Automatic LaTeX parsing
    success, fig_path, message = extract_figure_1_with_latex_parsing(source_dir)

    if success and fig_path:
        return True, fig_path, f"[Stage 1] {message}"

    # Stage 2: LLM-based extraction (if model provided)
    if model:
        success, fig_path, message = extract_figure_1_with_llm(source_dir, model)

        if success and fig_path:
            return True, fig_path, f"[Stage 2] {message}"

        # If LLM also failed, return the failure message
        return False, None, f"[Stage 2] {message}"

    # No model provided, return Stage 1 failure
    return False, None, f"[Stage 1] {message}"


# Legacy function kept for compatibility
def find_figure_files(source_dir: str) -> List[str]:
    """Find all image files in a source directory.

    Args:
        source_dir: Directory containing extracted arXiv source.

    Returns:
        List of paths to image files.
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.eps', '.svg', '.gif'}
    image_files = []

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(root, file))

    return image_files


# ============================================================================
# Color/Formatting Functions
# ============================================================================

def get_row_background_color(sheet_raw: Dict, row_idx: int) -> Optional[Dict]:
    """Extract background color from a specific row in raw sheet data.

    Args:
        sheet_raw: Raw sheet data from Google Sheets API.
        row_idx: 0-indexed row number.

    Returns:
        Color dict with 'red', 'green', 'blue' keys (0-1 values), or None.
    """
    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            return None

        rows = sheets[0].get('data', [{}])[0].get('rowData', [])
        if row_idx >= len(rows):
            return None

        row = rows[row_idx]
        cells = row.get('values', [])

        if not cells:
            return None

        # Get color from first cell in the row
        cell = cells[0]
        effective_format = cell.get('effectiveFormat', {})
        bg_color = effective_format.get('backgroundColor', {})

        return bg_color if bg_color else None

    except Exception:
        return None


def classify_row_color(color_dict: Optional[Dict]) -> str:
    """Classify a row color as yellow, blue, or none.

    Args:
        color_dict: Color dictionary with 'red', 'green', 'blue' keys.

    Returns:
        'yellow', 'blue', or 'none'.
    """
    if not color_dict:
        return 'none'

    red = color_dict.get('red', 1)
    green = color_dict.get('green', 1)
    blue = color_dict.get('blue', 1)

    # Yellow: high red, high green, low blue
    if red > 0.8 and green > 0.8 and blue < 0.5:
        return 'yellow'

    # Light yellow (Google Sheets default yellow)
    if red > 0.9 and green > 0.9 and blue > 0.6 and blue < 0.9:
        return 'yellow'

    # Blue: low red, low green, high blue
    if red < 0.5 and green < 0.7 and blue > 0.7:
        return 'blue'

    # Light blue
    if red > 0.6 and red < 0.9 and green > 0.8 and blue > 0.9:
        return 'blue'

    # White or near-white
    if red > 0.95 and green > 0.95 and blue > 0.95:
        return 'none'

    return 'none'


def validate_color_grouping(row_colors: List[str]) -> Tuple[bool, str]:
    """Check if same colors are grouped together (not interleaved).

    Args:
        row_colors: List of color classifications for each row.

    Returns:
        Tuple of (is_valid, message).
    """
    if not row_colors:
        return True, "No rows to check"

    # Track which colors we've seen and finished with
    seen_colors = set()
    finished_colors = set()
    current_color = None

    for i, color in enumerate(row_colors):
        if color == 'none':
            continue

        if current_color is None:
            current_color = color
            seen_colors.add(color)
        elif color != current_color:
            # Color changed
            finished_colors.add(current_color)

            if color in finished_colors:
                # We're seeing a color we already finished - interleaving!
                return False, f"Color '{color}' appears in non-contiguous rows (interleaved at row {i+1})"

            current_color = color
            seen_colors.add(color)

    return True, f"Colors are properly grouped: {seen_colors}"


# ============================================================================
# Text Matching Functions
# ============================================================================

def fuzzy_match_text(text1: str, text2: str, threshold: int = 80) -> Tuple[bool, int]:
    """Perform fuzzy matching between two texts.

    Args:
        text1: First text.
        text2: Second text.
        threshold: Minimum similarity score (0-100).

    Returns:
        Tuple of (is_match, similarity_score).
    """
    if not text1 or not text2:
        return False, 0

    # Normalize texts
    text1 = ' '.join(text1.lower().split())
    text2 = ' '.join(text2.lower().split())

    # Use token_sort_ratio for better matching of reordered text
    score = fuzz.token_sort_ratio(text1, text2)

    return score >= threshold, score


# ============================================================================
# Preprocessing Helper Functions
# ============================================================================

def search_arxiv_by_title(title: str, max_results: int = 5) -> Optional[Dict]:
    """Search arXiv for a paper by title.

    Args:
        title: Paper title to search for.
        max_results: Maximum number of results to return.

    Returns:
        Dict with paper info if found, None otherwise.
    """
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(
            query=f'ti:"{title}"',
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = list(client.results(search))

        if not results:
            return None

        # Find best match by title similarity
        best_match = None
        best_score = 0

        for result in results:
            is_match, score = fuzzy_match_text(title, result.title, threshold=80)
            if score > best_score:
                best_score = score
                best_match = result

        if best_match and best_score >= 80:
            # Extract arXiv ID from entry_id (e.g., "http://arxiv.org/abs/2507.20534v1")
            raw_id = best_match.entry_id.split('/')[-1]
            # Remove version suffix (e.g., "v1", "v2")
            arxiv_id = re.sub(r'v\d+$', '', raw_id)
            return {
                'arxiv_id': arxiv_id,
                'title': best_match.title,
                'authors': [str(a) for a in best_match.authors],
                'abstract': best_match.summary,
                'pdf_url': best_match.pdf_url,
                'match_score': best_score
            }

        return None

    except ImportError:
        print("Warning: arxiv package not installed")
        return None
    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return None


def detect_world_models_in_text(text: str) -> bool:
    """Check if text contains mentions of 'world models' in a relevant context.

    Args:
        text: Full text of the paper (or related works section).

    Returns:
        True if 'world models' is mentioned.
    """
    if not text:
        return False

    text_lower = text.lower()

    # Look for "world model" or "world models"
    patterns = [
        r'world\s+model[s]?',
        r'world-model[s]?',
    ]

    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True

    return False


def detect_world_models_in_pdf(pdf_path: str) -> Tuple[bool, str]:
    """Parse PDF and check for 'world models' in related works section.

    Args:
        pdf_path: Path to PDF file.

    Returns:
        Tuple of (has_world_models, section_text).
    """
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            full_text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + '\n'

        # Try to find related works section
        related_works_patterns = [
            r'related\s+work[s]?',
            r'related\s+literature',
            r'prior\s+work',
            r'background',
        ]

        related_section = ''
        text_lower = full_text.lower()

        for pattern in related_works_patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Extract section (rough heuristic: next 5000 chars or until next section)
                start = match.start()
                end = min(start + 5000, len(full_text))
                related_section = full_text[start:end]
                break

        # Check for world models in related section, or full text if section not found
        search_text = related_section if related_section else full_text
        has_world_models = detect_world_models_in_text(search_text)

        return has_world_models, search_text[:500] if search_text else ''

    except ImportError:
        print("Warning: pdfplumber not installed")
        return False, ''
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return False, ''
