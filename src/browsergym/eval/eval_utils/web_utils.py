"""Web utilities for fetching and downloading content from URLs."""

import os
import requests
from urllib.parse import urlparse

# Domains known to block programmatic image downloads (anti-hotlinking, bot protection, etc.)
UNVERIFIABLE_DOMAINS = [
    "ndtvimg.com",
    "researchgate.net",
    "wikimedia.org",
    "wikipedia.org",
    "instagram.com",
    "fbcdn.net",  # Facebook CDN
    "pinimg.com",  # Pinterest
    "twimg.com",  # Twitter
]


def is_unverifiable_url(url: str) -> bool:
    """Check if URL is from a domain known to block programmatic downloads.

    Args:
        url: The URL to check.

    Returns:
        True if the domain is known to block downloads, False otherwise.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Check if any unverifiable domain is in the URL's domain
        return any(blocked in domain for blocked in UNVERIFIABLE_DOMAINS)
    except Exception:
        return False


def download_image_from_url(url: str, temp_dir: str, timeout: int = 15) -> str:
    """Download image from URL to temp directory.

    Args:
        url: The URL to download the image from.
        temp_dir: Directory to save the downloaded image.
        timeout: Request timeout in seconds.

    Returns:
        Path to downloaded image, or None if download failed.
    """
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            if content_type.startswith('image/'):
                # Determine extension from content type
                ext = content_type.split('/')[-1].split(';')[0]
                if ext not in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
                    ext = 'png'
                temp_path = os.path.join(temp_dir, f"url_image_{hash(url)}.{ext}")
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                return temp_path
    except Exception as e:
        print(f"Failed to download image from {url}: {e}")
    return None
