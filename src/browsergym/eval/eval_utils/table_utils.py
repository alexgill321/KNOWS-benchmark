import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Any, List, Dict


# =============================================================================
# SheetTable Class
# =============================================================================

@dataclass
class SheetTable:
    """
    Represents a table extracted from a Google Sheet with position metadata.

    Attributes:
        df: The table data as a pandas DataFrame.
        start_col: 0-indexed starting column position in the sheet.
        end_col: 0-indexed ending column position (exclusive).
        start_row: 0-indexed starting row position in the sheet.
        end_row: 0-indexed ending row position (exclusive).
        sheet_name: Name of the sheet tab containing this table.
    """
    df: pd.DataFrame
    start_col: int = 0
    end_col: int = 0
    start_row: int = 0
    end_row: int = 0
    sheet_name: str = ""

    @property
    def col_letter(self) -> str:
        """Return starting column as Excel-style letter (e.g., 'A', 'K', 'AA')."""
        return self._col_index_to_letter(self.start_col)

    @property
    def end_col_letter(self) -> str:
        """Return ending column as Excel-style letter."""
        return self._col_index_to_letter(self.end_col - 1) if self.end_col > 0 else ""

    @property
    def columns(self) -> List[str]:
        """Return list of column names."""
        return list(self.df.columns)

    @property
    def num_rows(self) -> int:
        """Return number of data rows (excluding header)."""
        return len(self.df)

    @property
    def num_cols(self) -> int:
        """Return number of columns."""
        return len(self.df.columns)

    def __len__(self) -> int:
        """Return number of data rows."""
        return len(self.df)

    @staticmethod
    def _col_index_to_letter(idx: int) -> str:
        """Convert 0-indexed column number to Excel-style letter."""
        result = ""
        while idx >= 0:
            result = chr(idx % 26 + ord('A')) + result
            idx = idx // 26 - 1
        return result

    def __repr__(self) -> str:
        return (f"SheetTable(cols={self.col_letter}:{self.end_col_letter}, "
                f"rows={self.num_rows}, columns={self.columns})")

def table_exact_match(df1: pd.DataFrame, df2: pd.DataFrame, ignore_case: bool = False) -> bool:
    if df1.shape != df2.shape:
        return False

    if ignore_case:
        df1 = df1.applymap(lambda x: str(x).lower() if isinstance(x, str) else x)
        df2 = df2.applymap(lambda x: str(x).lower() if isinstance(x, str) else x)

    return df1.equals(df2)

def table_column_check(df: pd.DataFrame, required_columns: list) -> bool:
    return all(col in df.columns for col in required_columns)

def table_number_tolerance(df: pd.DataFrame, gold_dict: dict, nutrient_cols: list, rel_tol: float = 0.05) -> bool:
    for _, row in df.iterrows():
        ingredient = str(row["Ingredient"]).lower().strip()

        if ingredient not in gold_dict:
            print(f"Ingredient '{ingredient}' not in gold data.")
            return False

        for nutrient in nutrient_cols:
            if nutrient not in row or nutrient not in gold_dict[ingredient]:
                print(f"Missing {nutrient} for {ingredient}")
                return False

            try:
                actual = float(row[nutrient])
                expected = float(gold_dict[ingredient][nutrient])
                if not np.isclose(actual, expected, rtol=rel_tol):
                    print(f"{ingredient} → {nutrient} mismatch: got {actual}, expected {expected}")
                    return False
            except Exception as e:
                print(f"Error comparing {ingredient} → {nutrient}: {e}")
                return False

    return True

def table_partial_match(df: pd.DataFrame, gold_df: pd.DataFrame, key_column: str = "Ingredient") -> bool:
    df_keys = set(df[key_column].str.lower().str.strip())
    gold_keys = set(gold_df[key_column].str.lower().str.strip())

    missing = gold_keys - df_keys
    if missing:
        print(f"Missing values: {missing}")
        return False
    return True


def find_matching_column_or_row(
    df: pd.DataFrame,
    criteria: str,
    model: Any,
    search_type: str = "column",
    example_keywords: Optional[list] = None,
) -> Optional[Tuple[str, Union[int, str]]]:
    """
    Find a column or row in a DataFrame that best matches natural language criteria using an LLM.

    Args:
        df (pd.DataFrame): The DataFrame to search.
        criteria (str): Natural language description of what to look for.
        model (Any): LLM model that takes (prompt) and returns a response.
        search_type (str): Either "column" or "row" to specify search direction.
        example_keywords (Optional[list]): Optional list of example keywords/terms that might match
            the criteria. Helps the LLM better understand what to look for.

    Returns:
        Optional[Tuple[str, Union[int, str]]]: Tuple of (title, location) where location is
        column name for columns or row index for rows, or None if no match found.

    Raises:
        ValueError: If search_type is not "column" or "row".
    """
    if search_type not in ["column", "row"]:
        raise ValueError("search_type must be either 'column' or 'row'")

    if search_type == "column":
        headers = df.columns.tolist()
    else:
        # For rows, use the first column as row identifiers if it contains strings,
        # otherwise use row indices
        if len(df) > 0 and df.iloc[:, 0].dtype == 'object':
            headers = df.iloc[:, 0].tolist()
        else:
            headers = [f"Row {i}" for i in range(len(df))]

    if not headers:
        return None

    # Create prompt for LLM
    headers_text = "\n".join([f"{i+1}. {header}" for i, header in enumerate(headers)])

    # Add example keywords to the prompt if provided
    keywords_section = ""
    if example_keywords:
        keywords_section = f"\nExample keywords that might match the criteria: {', '.join(example_keywords)}"

    prompt = f"""You are analyzing a Google Sheets {search_type} headers to find the one that best matches specific criteria.

Criteria: {criteria}{keywords_section}

Available {search_type} headers:
{headers_text}

Please analyze each header and determine which one best matches the criteria. Consider:
- Exact matches
- Synonyms and semantically similar terms
- Common abbreviations
- Spreadsheet naming conventions

Respond with ONLY the number (1, 2, 3, etc.) of the best matching header, or "NONE" if no header adequately matches the criteria.

Your response should be just the number or "NONE", nothing else."""

    try:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        response = model(messages)
        response = response.strip().upper()

        if response == "NONE":
            return None

        # Try to parse the response as a number
        try:
            header_index = int(response) - 1
            if 0 <= header_index < len(headers):
                header_title = headers[header_index]
                if search_type == "column":
                    location = header_title  # Column name
                else:
                    if df.iloc[:, 0].dtype == 'object':
                        location = header_title  # Row identifier from first column
                    else:
                        location = header_index  # Row index
                return (header_title, location)
        except ValueError:
            pass

        return None

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


# =============================================================================
# Google Sheets Text Visibility Utilities
# =============================================================================

def is_text_visible_in_cell(
    content: str,
    col_width: int,
    wrap_strategy: str,
    row_values: List[Dict],
    col_idx: int,
    char_width: int = 7
) -> bool:
    """
    Check if text is fully visible in a Google Sheets cell.

    Text is visible if:
    - Column width is sufficient for the text, OR
    - Wrap strategy is 'WRAP' (text wraps to multiple lines), OR
    - Wrap strategy is 'OVERFLOW_CELL' and adjacent cells are empty (text overflows)

    Text is NOT visible (truncated/hidden) if:
    - Text width exceeds column width AND wrap strategy is 'CLIP', OR
    - Text width exceeds column width AND wrap strategy is 'OVERFLOW_CELL' but next cell has content

    Args:
        content: The text content of the cell.
        col_width: Width of the column in pixels.
        wrap_strategy: One of 'WRAP', 'OVERFLOW_CELL', or 'CLIP'.
        row_values: List of all cell values in the row (to check adjacent cells).
        col_idx: Column index of this cell.
        char_width: Approximate width per character in pixels (default 7).

    Returns:
        True if text is fully visible, False if truncated/hidden.
    """
    if not content:
        return True

    expected_width = len(content) * char_width

    # If text fits in column, it's visible
    if expected_width <= col_width:
        return True

    # If wrapping is enabled, text is visible (wraps to multiple lines)
    if wrap_strategy == 'WRAP':
        return True

    # If CLIP, text is hidden when it exceeds width
    if wrap_strategy == 'CLIP':
        return False

    # For OVERFLOW_CELL (default), check if next cell blocks the overflow
    if wrap_strategy == 'OVERFLOW_CELL':
        # Check subsequent cells to see if overflow is blocked
        overflow_needed = expected_width - col_width
        current_col = col_idx + 1

        while overflow_needed > 0 and current_col < len(row_values):
            next_cell = row_values[current_col] if current_col < len(row_values) else {}
            next_content = next_cell.get('formattedValue', '') if isinstance(next_cell, dict) else ''

            if next_content:
                # Next cell has content, overflow is blocked - text is hidden
                return False

            # Assume default column width for overflow calculation
            overflow_needed -= 100  # Default column width
            current_col += 1

        # Overflow has room, text is visible
        return True

    # Unknown wrap strategy, assume visible
    return True