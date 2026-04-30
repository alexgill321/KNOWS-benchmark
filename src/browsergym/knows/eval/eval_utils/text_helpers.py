def preprocess_text(text):
    """
    Preprocess the text by removing leading and trailing whitespace, as well as punctuation or special characters.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    import re

    text = text.lower()  # Convert to lowercase

    text = text.replace('\n', ' ')  # Replace newlines with spaces

    # Remove leading and trailing whitespace
    text = text.strip()

    # Remove punctuation or special characters (if needed)
    text = re.sub(r'[^\w\s]', '', text)

    return text