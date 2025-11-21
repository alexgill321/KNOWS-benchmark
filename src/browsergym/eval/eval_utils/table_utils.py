import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union, Any

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