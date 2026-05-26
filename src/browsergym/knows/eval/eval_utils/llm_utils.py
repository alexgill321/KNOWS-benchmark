"""Shared utilities for LLM-based evaluation tasks.

Provides common functions for extracting structured data from LLM responses,
including JSON extraction, markdown code block handling, yes/no parsing,
and multi-type evaluation helpers.
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional, Union

_TRAILING_COMMA_RE = re.compile(r',(\s*[\]}])')


def strip_markdown_code_blocks(text: str) -> str:
    """Strip markdown code block fencing from LLM response text.

    Handles responses wrapped in ```json ... ``` or ``` ... ``` fencing
    by extracting only the content between the fences.

    Args:
        text (str): Raw LLM response text that may contain code block fencing.

    Returns:
        str: Text with code block fencing removed. Returns the original text
            unchanged if no code blocks are found.
    """
    if "```" not in text:
        return text

    lines = text.split('\n')
    json_lines = []
    in_code_block = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            json_lines.append(line)

    if json_lines:
        return "\n".join(json_lines)
    return text


def parse_yes_no(response: str) -> Optional[bool]:
    """Parse a yes/no answer from an LLM response.

    Looks for 'yes', 'no', or uncertainty markers in the lowercased response.
    Checks for 'unsure'/'don't know' first so uncertain responses return None
    rather than matching a stray 'no' inside 'don't know'.

    Args:
        response (str): Raw LLM response text.

    Returns:
        Optional[bool]: True if response contains 'yes', False if 'no',
            None if unsure or neither is found.
    """
    response_lower = response.strip().lower()
    if "unsure" in response_lower or "don't know" in response_lower or "do not know" in response_lower:
        return None
    if "yes" in response_lower:
        return True
    elif "no" in response_lower:
        return False
    return None


def extract_json_from_llm_response(
    response: str,
    expect_type: Literal["auto", "array", "object"] = "auto",
) -> Optional[Union[Dict, List]]:
    """Extract and parse JSON from an LLM response string.

    Handles common LLM response patterns: markdown code blocks, leading/trailing
    text around JSON, and bracket-matching for nested structures.

    Args:
        response (str): Raw LLM response text containing JSON.
        expect_type (str): Expected JSON structure type:
            - "auto": Try json.loads directly after stripping code blocks.
            - "array": Find and extract the outermost JSON array [...].
            - "object": Find and extract the outermost JSON object {...}.

    Returns:
        Optional[Union[Dict, List]]: Parsed JSON data, or None if parsing fails.
    """
    response = strip_markdown_code_blocks(response.strip())

    try:
        if expect_type == "array":
            start_idx = response.find('[')
            if start_idx != -1:
                bracket_count = 0
                for i, char in enumerate(response[start_idx:], start_idx):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            response = response[start_idx:i + 1]
                            break

        elif expect_type == "object":
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                response = response[start_idx:end_idx + 1]

        response = _TRAILING_COMMA_RE.sub(r'\1', response)
        return json.loads(response)

    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        print(f"Response was: {response[:500]}...")
        return None
    except Exception as e:
        print(f"Error extracting JSON from LLM response: {e}")
        return None


_DEFAULT_EXTRACTION_SYSTEM_PROMPT = (
    "You are a data extraction assistant. Extract the information "
    "in the provided text and format it as specified.\n\n"
    "Always respond with valid JSON only, no other text."
)


def extract_json_with_llm(
    prompt: str,
    model: Any,
    system_prompt: Optional[str] = None,
    content: Optional[List[Dict]] = None,
    expect_type: Literal["auto", "array", "object"] = "auto",
) -> Optional[Union[Dict, List]]:
    """Extract structured JSON data from text using an LLM.

    Builds a message with the standard format, sends it to the model,
    then parses the JSON response. Supports multi-modal content (images)
    via the content parameter.

    Args:
        prompt (str): The user prompt with extraction instructions. Ignored
            if content is provided.
        model (Any): Callable LLM interface (from models.load_model).
        system_prompt (str): Custom system prompt. Defaults to a generic
            data extraction prompt instructing JSON-only output.
        content (Optional[List[Dict]]): Pre-built user content list for
            multi-modal messages (e.g., including images). When provided,
            replaces the default text-only user content.
        expect_type (str): Expected JSON structure type passed to
            extract_json_from_llm_response.

    Returns:
        Optional[Union[Dict, List]]: Parsed JSON dict or list, or None on failure.
    """
    if content is None:
        content = [{"type": "text", "text": prompt}]

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt or _DEFAULT_EXTRACTION_SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": content
        }
    ]

    try:
        response = model(messages)
        if response is None:
            print("LLM returned None response")
            return None

        return extract_json_from_llm_response(response, expect_type=expect_type)

    except Exception as e:
        print(f"Error in LLM extraction: {e}")
        return None


_RETURN_TYPE_INSTRUCTIONS = {
    "bool": "Respond with ONLY 'yes' or 'no'.",
    "str": "Format your response strictly as specified in the task instructions.",
    "json": "Respond with the JSON format specified in the task instructions.",
}

_DEFAULT_EVALUATION_SYSTEM_PROMPT = (
    "You are a helpful assistant who evaluates whether a text "
    "satisfies the requirements of the task."
)


def evaluate_with_llm(
    prompt: str,
    model: Any,
    return_type: Literal["bool", "str", "json"] = "bool",
    system_prompt: Optional[str] = None,
) -> Optional[Union[bool, str, Any]]:
    """Evaluate text using an LLM and return result in specified format.

    Supports three return modes:
    - "bool": Returns True/False based on yes/no in the response.
    - "str": Returns the raw lowercased response string.
    - "json": Parses and returns JSON from the response.

    Args:
        prompt (str): The text/question to evaluate.
        model (Any): Callable LLM interface.
        return_type (str): Format of returned result: 'bool', 'str', or 'json'.
        system_prompt (str): Custom system prompt. Defaults to an evaluation
            prompt with return-type-specific instructions appended.

    Returns:
        Optional[Union[bool, str, Any]]: Result in the specified format,
            or None on error.
    """
    if return_type not in ["bool", "str", "json"]:
        print(f"Error: return_type must be 'bool', 'str', or 'json'. Got: {return_type}")
        return None

    if system_prompt is None:
        sys_text = f"{_DEFAULT_EVALUATION_SYSTEM_PROMPT}\n\n{_RETURN_TYPE_INSTRUCTIONS[return_type]}"
    else:
        sys_text = system_prompt

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": sys_text}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    try:
        response = model(messages)
        if response is None:
            print("LLM returned None response")
            return None
        response = response.strip().lower()

        if return_type == "bool":
            return "yes" in response
        elif return_type == "str":
            return response
        else:
            return extract_json_from_llm_response(response, expect_type="auto")

    except Exception as e:
        print(f"LLM evaluation failed: {e}")
        return None
