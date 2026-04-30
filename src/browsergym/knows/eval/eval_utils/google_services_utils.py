import sys
import os
import re
from typing import Dict, List, Optional, Tuple
sys.path.append(os.getcwd())
from src.browsergym.knows.eval.eval_utils.google_services_helpers import *
import requests
import mimetypes
from urllib.parse import urlparse, parse_qs
from google.oauth2.service_account import Credentials # For Service Account
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from google.cloud import secretmanager
import io
import json
import difflib

# Re-export sheets-related functions from google_sheets_utils for backward compatibility
# These functions have been moved to google_sheets_utils.py but are re-exported here
# to support existing code that uses wildcard imports from this module.
from src.browsergym.knows.eval.eval_utils.google_sheets_utils import (
    get_sheet_content,
    detect_header_row,
    search_sheet,
    extract_tables_from_sheet,
    extract_structure_from_sheet,
    extract_charts_from_sheet,
    parse_sheet_to_dataframe,
    extract_sheet_data,  # New comprehensive wrapper function
)

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID") # e.g., your-project-id
SECRET_ID = os.environ.get("DRIVE_SA_SECRET_ID")   # e.g., doc-eval-service-account-key
SECRET_VERSION_ID = os.environ.get("DRIVE_SA_SECRET_VERSION_ID", "latest")
SERVICE_ACCOUNT_PATH = os.environ.get("SERVICE_ACCOUNT_PATH", os.path.join(os.getcwd(), "auth-data", "service-account.json"))

# Global variable to hold initialized Google API services
# (Initialize them once, not on every request)
DRIVE_SERVICE = None
DOCS_SERVICE = None
SLIDES_SERVICE = None
SHEETS_SERVICE = None


def _get_scopes_for_service_type(service_type):
    """Returns the appropriate scopes for a given service type."""
    if service_type == 'sheets':
        return [
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/spreadsheets'
        ]
    elif service_type == 'docs':
        return [
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/documents'
        ]
    elif service_type == 'slides':
        return [
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/presentations'
        ]
    else:
        return ['https://www.googleapis.com/auth/drive']


def _build_services_from_credentials(credentials, service_type):
    """Builds Google API services from credentials based on service type."""
    global DRIVE_SERVICE, DOCS_SERVICE, SLIDES_SERVICE, SHEETS_SERVICE

    DRIVE_SERVICE = build('drive', 'v3', credentials=credentials)

    if service_type == 'sheets':
        SHEETS_SERVICE = build('sheets', 'v4', credentials=credentials)
        return DRIVE_SERVICE, SHEETS_SERVICE
    elif service_type == 'docs':
        DOCS_SERVICE = build('docs', 'v1', credentials=credentials)
        return DRIVE_SERVICE, DOCS_SERVICE
    elif service_type == 'slides':
        SLIDES_SERVICE = build('slides', 'v1', credentials=credentials)
        return DRIVE_SERVICE, SLIDES_SERVICE
    else:
        return DRIVE_SERVICE, None


def initialize_google_services(service_type=None):
    global DRIVE_SERVICE, DOCS_SERVICE, SLIDES_SERVICE, SHEETS_SERVICE

    # Auto-detect service type if not provided
    if service_type is None:
        import inspect
        caller_path = inspect.stack()[1].filename if len(inspect.stack()) > 1 else ""

        if 'docs_' in caller_path:
            service_type = 'docs'
        elif 'sheets_' in caller_path:
            service_type = 'sheets'
        elif 'slides_' in caller_path:
            service_type = 'slides'
        else:
            service_type = 'docs'  # Default to docs for most common case

    # Priority 1: Local service account JSON file
    if os.path.exists(SERVICE_ACCOUNT_PATH):
        try:
            print(f"Using local service account file: {SERVICE_ACCOUNT_PATH}")
            scopes = _get_scopes_for_service_type(service_type)
            credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH, scopes=scopes)
            return _build_services_from_credentials(credentials, service_type)
        except Exception as e:
            print(f"Error loading service account from file: {e}")
            # Fall through to try other methods

    # Priority 2: Secret Manager (for Cloud Run)
    if GCP_PROJECT_ID and SECRET_ID:
        try:
            return _initialize_from_secret_manager(service_type)
        except Exception as e:
            print(f"Error initializing from Secret Manager: {e}")
            # Fall through to try OAuth

    # Priority 3: OAuth token (requires initial login)
    print("No service account found. Attempting OAuth initialization...")
    try:
        if service_type == 'sheets':
            credentials = authenticate(['DRIVE','SHEETS'])
        elif service_type == 'docs':
            credentials = authenticate(['DRIVE', 'DOCS'])
        elif service_type == 'slides':
            credentials = authenticate(['DRIVE', 'SLIDES'])
        else:
            credentials = authenticate(['DRIVE'])
        return _build_services_from_credentials(credentials, service_type)
    except Exception as e:
        print(f"Error initializing Google API services via OAuth: {e}")
        return None, None


def _initialize_from_secret_manager(service_type):
    """Initialize services using a service account key from Secret Manager."""
    # Create the Secret Manager client
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version
    name = f"projects/{GCP_PROJECT_ID}/secrets/{SECRET_ID}/versions/{SECRET_VERSION_ID}"

    # Access the secret version
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")

    # The payload is the JSON string of your service account key
    service_account_info = json.loads(payload)

    scopes = _get_scopes_for_service_type(service_type)
    credentials = Credentials.from_service_account_info(service_account_info, scopes=scopes)

    print("Successfully initialized Google API services using Service Account from Secret Manager.")
    return _build_services_from_credentials(credentials, service_type)


def extract_drive_file_id(url: str) -> str:
    """Extract Google Drive file ID from a URL.

    Handles various Google Drive and Docs URL formats:
    - https://drive.google.com/file/d/FILE_ID/view
    - https://drive.google.com/open?id=FILE_ID
    - https://docs.google.com/document/d/FILE_ID/edit
    - https://docs.google.com/spreadsheets/d/FILE_ID/edit
    - https://docs.google.com/presentation/d/FILE_ID/edit

    Args:
        url: The Google Drive URL to parse.

    Returns:
        The file ID string, or None if not found.
    """
    if not url:
        return None

    # Pattern 1: /d/FILE_ID/ (most common)
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)

    # Pattern 2: ?id=FILE_ID (legacy format)
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    if 'id' in query_params:
        return query_params['id'][0]

    return None


def search_doc(filename, service, folder_id=None):
    """Search for a Google Doc by its filename.

    Args:
        filename (str): The name of the Google Doc to search for.
        service: The Google Drive service instance.
        folder_id (str, optional): The ID of the Google Drive folder to search in. If None, searches in the entire Drive.

    Returns:
        A tuple (status, doc_id) containing:
            - status (int): 0 if not found, 1 if found in any location, 2 if found in specified location.
            - doc_id (str): The ID of the found Google Doc, or None if not found.
    """
    doc = None
    if folder_id:
        doc = find_doc_specified_location(folder_id, filename, service)
    if doc is None:
        doc = find_file_any(filename, service, 'document')
        if doc is None:
            return 0, None
        return 1, doc
    else:
        return 2, doc

def find_doc_specified_location(folder_id, filename, service):
    """Find a Google Doc in a specified folder by its filename.

    Args:
        folder_id (str): The ID of the Google Drive folder to search in.
        filename (str): The name of the Google Doc to find.

    Returns:
        file_id (str): The ID of the found Google Doc, or None if not found.
    """
    

    query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.document' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name, webViewLink)").execute()
    files = results.get('files', [])

    if not files:
        print("No Google Docs found in the specified directory.")
    else:
        for file in files:
            if filename in file['name']:
                file_id = file['id']
                print(f"Found file: {file['name']} (ID: {file_id})")
                return file_id
        else:
            print("No matching file found.")
            return None

def find_file_any(filename, service, file_type=None):
    """Find a Google Drive file by its filename and optionally by file type.

    Args:
        filename (str): The name of the file to find.
        service: The Google Drive service instance.
        file_type (str, optional): The type of file to search for. Options:
            - 'document' or 'doc': Google Docs
            - 'spreadsheet' or 'sheet': Google Sheets
            - 'presentation' or 'slides': Google Slides
            - None: Search all file types

    Returns:
        str: Single file ID if one match found
        list: List of file IDs if multiple matches found
        None: If no files found
    """
    # Map file types to MIME types
    mime_types = {
        'document': 'application/vnd.google-apps.document',
        'doc': 'application/vnd.google-apps.document',
        'spreadsheet': 'application/vnd.google-apps.spreadsheet',
        'sheet': 'application/vnd.google-apps.spreadsheet',
        'presentation': 'application/vnd.google-apps.presentation',
        'slides': 'application/vnd.google-apps.presentation'
    }

    # Build query
    query = f"name='{filename}' and trashed=false"
    if file_type and file_type.lower() in mime_types:
        mime_type = mime_types[file_type.lower()]
        query += f" and mimeType='{mime_type}'"

    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name, mimeType)',
        pageSize=10
    ).execute()
    items = results.get('files', [])

    if not items:
        file_type_str = f" {file_type}" if file_type else ""
        print(f"No{file_type_str} files found with the specified filename.")
        return None
    elif len(items) > 1:
        file_type_str = f" {file_type}" if file_type else ""
        print(f"Multiple{file_type_str} files found with the specified filename.")
        ids = [item["id"] for item in items]
        return ids
    else:
        file_id = items[0]['id']
        print(f"Found file: '{items[0]['name']}' (ID: {file_id})")
        return file_id
                   
def extract_images_from_doc(doc_id, service, output_dir=None):
    """Extracts images from a Google document.

    Args:
        service: The Google Docs service instance.
        doc_id (str): The ID of the Google Doc to extract images from.
        output_dir (str): The directory to save the extracted images. If None, the extracted images will not be saved.

    Returns:
        list: A list of images extracted from the document. Each image is represented as a byte string.
            Returns None if no images were found or an error occurred.
    """
    if output_dir is not None:
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

    document = service.documents().get(documentId=doc_id).execute()
    inline_objects = document.get("inlineObjects") # Get the dictionary of inline objects

    if not inline_objects:
        print("No inline objects (which could contain images) found in the document.")
        return None
    
    image_count = 0
    # Use the authenticated session from the credentials for downloading
    # This automatically handles adding the Authorization: Bearer token
    authed_session = requests.Session()
    authed_session.headers.update({'Authorization': f'Bearer {service._http.credentials.token}'})

    images = []
    for obj_id, obj_data in inline_objects.items():
        embedded_object = obj_data.get('inlineObjectProperties', {}).get('embeddedObject')
        if embedded_object and embedded_object.get('imageProperties'):
            image_properties = embedded_object['imageProperties']
            content_uri = image_properties.get('contentUri')

            if content_uri:
                image_count += 1
                print(f"Found image {image_count} (Object ID: {obj_id})")

                try:
                    response = authed_session.get(content_uri)
                    response.raise_for_status()  # Raise an error for bad responses

                    content_type = response.headers.get('Content-Type')
                    extension = mimetypes.guess_extension(content_type) if content_type else '.jpg' # Default guess
                    if not extension: # Handle cases like 'image/png; charset=utf-8' or unknown types
                        if 'png' in content_type.lower(): extension = '.png'
                        elif 'jpeg' in content_type.lower() or 'jpg' in content_type.lower(): extension = '.jpg'
                        elif 'gif' in content_type.lower(): extension = '.gif'
                        elif 'webp' in content_type.lower(): extension = '.webp'
                        else: extension = '.img' # Generic if unsure

                    images.append(response.content)

                    if output_dir is not None:
                        # Create a filename
                        filename = f"image_{image_count}_{obj_id}{extension}"
                        filepath = os.path.join(output_dir, filename)

                        # Save the image content
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        print(f" -> Saved image to: {filepath}")
                except requests.exceptions.RequestException as req_err:
                    print(f"  -> Error downloading image {image_count} (Object ID: {obj_id}): {req_err}")
                except IOError as io_err:
                        print(f"  -> Error saving image {image_count} (Object ID: {obj_id}): {io_err}")
                except Exception as e:
                    print(f"  -> An unexpected error occurred processing image {image_count} (Object ID: {obj_id}): {e}")

    if image_count == 0:
        print("No images found in the document.")
    else:
        print(f"Extracted {image_count} images from the document.")
        return images
    return None

def extract_images_from_doc_extended(service, output_dir=None, document=None, doc_id=None, include_positioned=False):
    """Extracts images from a Google document, with support for positioned objects.

    Extends extract_images_from_doc by also handling positionedObjects (images
    placed at specific page positions, e.g. side-by-side at the bottom) and
    accepting a pre-fetched document to avoid redundant API calls.

    Args:
        service: The Google Docs service instance.
        output_dir (str): Directory to save extracted images. Created if missing.
        document (dict): Pre-fetched document JSON (from get_doc_content). If None, fetches via doc_id.
        doc_id (str): The ID of the Google Doc. Required if document is not provided.
        include_positioned (bool): If True, also extracts positioned objects.

    Returns:
        list: A list of image byte strings, or None if no images found.
    """
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    if document is None:
        if doc_id is None:
            raise ValueError("Either document or doc_id must be provided")
        document = service.documents().get(documentId=doc_id).execute()

    inline_objects = document.get("inlineObjects") or {}
    positioned_objects = document.get("positionedObjects") or {} if include_positioned else {}

    if not inline_objects and not positioned_objects:
        print("No images found in the document.")
        return None

    image_count = 0
    authed_session = requests.Session()
    authed_session.headers.update({'Authorization': f'Bearer {service._http.credentials.token}'})

    images = []

    def _download_image(content_uri, obj_id, source_label):
        nonlocal image_count
        image_count += 1
        print(f"Found {source_label} image {image_count} (Object ID: {obj_id})")
        try:
            response = authed_session.get(content_uri)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type')
            extension = mimetypes.guess_extension(content_type) if content_type else '.jpg'
            if not extension:
                if 'png' in content_type.lower(): extension = '.png'
                elif 'jpeg' in content_type.lower() or 'jpg' in content_type.lower(): extension = '.jpg'
                elif 'gif' in content_type.lower(): extension = '.gif'
                elif 'webp' in content_type.lower(): extension = '.webp'
                else: extension = '.img'

            images.append(response.content)

            if output_dir is not None:
                filename = f"image_{image_count}_{obj_id}{extension}"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f" -> Saved {source_label} image to: {filepath}")
        except requests.exceptions.RequestException as req_err:
            print(f"  -> Error downloading {source_label} image {image_count} (Object ID: {obj_id}): {req_err}")
        except IOError as io_err:
            print(f"  -> Error saving {source_label} image {image_count} (Object ID: {obj_id}): {io_err}")
        except Exception as e:
            print(f"  -> Unexpected error processing {source_label} image {image_count} (Object ID: {obj_id}): {e}")

    for obj_id, obj_data in inline_objects.items():
        embedded_object = obj_data.get('inlineObjectProperties', {}).get('embeddedObject')
        if embedded_object and embedded_object.get('imageProperties'):
            content_uri = embedded_object['imageProperties'].get('contentUri')
            if content_uri:
                _download_image(content_uri, obj_id, "inline")

    for obj_id, obj_data in positioned_objects.items():
        embedded_object = obj_data.get('positionedObjectProperties', {}).get('embeddedObject')
        if embedded_object and embedded_object.get('imageProperties'):
            content_uri = embedded_object['imageProperties'].get('contentUri')
            if content_uri:
                _download_image(content_uri, obj_id, "positioned")

    if image_count == 0:
        print("No images found in the document.")
        return None

    print(f"Extracted {image_count} images from the document ({len(inline_objects)} inline, {len(positioned_objects)} positioned).")
    return images


def extract_images_from_doc_with_cropping(doc_id, service, output_dir=None):
    """Extracts images from a Google document and applies document cropping.

    Args:
        doc_id (str): The ID of the Google Doc to extract images from.
        service: The Google Docs service instance.
        output_dir (str): The directory to save the extracted images. If None, images not saved.

    Returns:
        list: A list of tuples containing (image_data, crop_info) for each extracted image.
            Returns None if no images were found or an error occurred.
    """
    from PIL import Image
    from io import BytesIO
    
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

    document = service.documents().get(documentId=doc_id).execute()
    inline_objects = document.get("inlineObjects")

    if not inline_objects:
        print("No inline objects found in the document.")
        return None
    
    image_count = 0
    authed_session = requests.Session()
    authed_session.headers.update({'Authorization': f'Bearer {service._http.credentials.token}'})

    extracted_images = []
    
    for obj_id, obj_data in inline_objects.items():
        embedded_object = obj_data.get('inlineObjectProperties', {}).get('embeddedObject')
        if embedded_object and embedded_object.get('imageProperties'):
            image_properties = embedded_object['imageProperties']
            content_uri = image_properties.get('contentUri')
            crop_properties = image_properties.get('cropProperties', {})

            if content_uri:
                image_count += 1
                print(f"Found image {image_count} (Object ID: {obj_id})")
                
                if crop_properties:
                    print(f"  Crop properties: {crop_properties}")

                try:
                    # Download original image
                    response = authed_session.get(content_uri)
                    response.raise_for_status()
                    
                    # Load image with PIL
                    original_image = Image.open(BytesIO(response.content))
                    width, height = original_image.size
                    
                    # Apply cropping if crop properties exist
                    if crop_properties:
                        offset_top = crop_properties.get('offsetTop', 0.0)
                        offset_bottom = crop_properties.get('offsetBottom', 0.0)
                        offset_left = crop_properties.get('offsetLeft', 0.0)
                        offset_right = crop_properties.get('offsetRight', 0.0)
                        
                        # Calculate crop coordinates
                        # Google Docs crop offsets are ratios of how much to crop from each edge
                        left = int(width * offset_left)
                        top = int(height * offset_top)
                        right = int(width * (1.0 - offset_right))
                        bottom = int(height * (1.0 - offset_bottom))
                        
                        print(f"  Applying crop: left={left}, top={top}, right={right}, bottom={bottom}")
                        print(f"  Original size: {width}x{height}")
                        
                        # Crop the image
                        cropped_image = original_image.crop((left, top, right, bottom))
                        print(f"  Cropped size: {cropped_image.size}")
                        
                        # Convert back to bytes
                        img_buffer = BytesIO()
                        img_format = original_image.format if original_image.format else 'PNG'
                        cropped_image.save(img_buffer, format=img_format)
                        cropped_data = img_buffer.getvalue()
                        
                        image_to_save = cropped_data
                        final_image = cropped_image
                    else:
                        print("  No crop properties found, using original image")
                        image_to_save = response.content
                        final_image = original_image
                    
                    # Store result
                    crop_info = {
                        'original_size': (width, height),
                        'final_size': final_image.size,
                        'crop_applied': bool(crop_properties),
                        'crop_properties': crop_properties
                    }
                    extracted_images.append((image_to_save, crop_info))

                    if output_dir is not None:
                        # Determine file extension
                        content_type = response.headers.get('Content-Type')
                        extension = mimetypes.guess_extension(content_type) if content_type else '.jpg'
                        if not extension:
                            if 'png' in content_type.lower(): extension = '.png'
                            elif 'jpeg' in content_type.lower() or 'jpg' in content_type.lower(): extension = '.jpg'
                            else: extension = '.jpg'
                        
                        # Save cropped image
                        filename = f"image_{image_count}_{obj_id}{extension}"
                        filepath = os.path.join(output_dir, filename)
                        
                        with open(filepath, 'wb') as f:
                            f.write(image_to_save)
                        print(f"  -> Saved cropped image to: {filepath}")

                except Exception as e:
                    print(f"  -> Error processing image {image_count} (Object ID: {obj_id}): {e}")

    if image_count == 0:
        print("No images found in the document.")
        return None
    else:
        print(f"Extracted {image_count} images (with cropping applied) from the document.")
        return extracted_images

def extract_text_from_doc(doc_id, service):
    """Extracts all text content from a Google Document.

    Args:
        doc_id (str): The ID of the Google Doc to extract text from.

    Returns:
        A dictionary containing the extracted text under the key 'text'.
        Returns {'text': ''} if the document has no text content.
        Returns None if an error occurs (e.g., document not found, API error).
    """
    try:
        doc = get_doc_content(doc_id, service)
        doc_content = doc.get('body', {}).get('content', [])
        extracted_text = []
        print("Extracting text from document elements...")
        # Iterate through the structural elements of the document body
        for element in doc_content:
            # Paragraphs are the most common container for text
            if 'paragraph' in element:
                para_elements = element.get('paragraph', {}).get('elements')
                if para_elements:
                    for para_element in para_elements:
                        # Check for a text run within the paragraph element
                        if 'textRun' in para_element:
                            text_run = para_element.get('textRun')
                            # Ensure content exists and is not just whitespace (optional, depends on need)
                            if text_run and 'content' in text_run:
                                extracted_text.append(text_run.get('content'))
            # You could add checks for text within tables here if needed:
            # elif 'table' in element:
            #     # Tables contain rows, which contain cells, which contain content (like paragraphs)
            #     table = element.get('table')
            #     if table and 'tableRows' in table:
            #         for row in table.get('tableRows'):
            #             if 'tableCells' in row:
            #                 for cell in row.get('tableCells'):
            #                     if 'content' in cell:
            #                         # Recursively process cell content (it's like a mini-document body)
            #                         for cell_content_element in cell.get('content'):
            #                             if 'paragraph' in cell_content_element:
            #                                 # ... (similar logic as above for paragraphs) ...
            #                                 pass # Add recursive extraction or flatten logic here


        full_text = "".join(extracted_text)
        print(f"Extraction complete. Total characters extracted: {len(full_text)}")

        # Return the text in the specified dictionary format
        return full_text

    except HttpError as error:
        print(f'An HTTP error occurred while extracting text: {error}')
        # Handle specific errors if needed, e.g., 404 Not Found
        if error.resp.status == 404:
            print(f"Document with ID '{doc_id}' not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during text extraction: {e}")
        return None


def extract_hyperlinks_from_doc(doc_id: str, service, document: dict = None) -> list:
    """Extract all hyperlinks and plain text URLs from a Google Doc.

    Extracts both embedded hyperlinks (textStyle.link.url) and plain text URLs
    from paragraphs and tables. Returns structured data with URL and anchor text.

    Args:
        doc_id: The Google Doc ID. Used to fetch document if document is None.
        service: The Google Docs API service instance.
        document: Optional pre-fetched document JSON. If provided, doc_id is
            ignored and no API call is made.

    Returns:
        List of dicts with 'url' and 'text' keys. Each dict represents one
        hyperlink found in the document. For plain text URLs, 'text' equals
        the URL itself.
    """
    if document is None:
        document = service.documents().get(documentId=doc_id).execute()

    body = document.get('body', {})
    content = body.get('content', [])

    links = []
    url_pattern = re.compile(
        r'https?://[^\s<>"\'}\])\u200b\u00a0]+',
        re.IGNORECASE
    )

    def process_paragraph(paragraph):
        para_links = []
        para_elements = paragraph.get('elements', [])

        for elem in para_elements:
            if 'textRun' in elem:
                text_run = elem['textRun']
                content_text = text_run.get('content', '')
                text_style = text_run.get('textStyle', {})

                # Check for embedded hyperlink
                if 'link' in text_style:
                    url = text_style['link'].get('url', '')
                    if url:
                        para_links.append({
                            'url': url,
                            'text': content_text.strip()
                        })

                # Also check for plain text URLs
                plain_urls = url_pattern.findall(content_text)
                for plain_url in plain_urls:
                    if not any(l['url'] == plain_url for l in para_links):
                        para_links.append({
                            'url': plain_url,
                            'text': plain_url
                        })

        return para_links

    def process_table(table):
        table_links = []
        for row in table.get('tableRows', []):
            for cell in row.get('tableCells', []):
                cell_content = cell.get('content', [])
                for elem in cell_content:
                    if 'paragraph' in elem:
                        table_links.extend(process_paragraph(elem['paragraph']))
        return table_links

    for element in content:
        if 'paragraph' in element:
            links.extend(process_paragraph(element['paragraph']))
        elif 'table' in element:
            links.extend(process_table(element['table']))

    return links


def download_doc_as_pdf(doc_id, output_file, service):
    """Downloads a Google Doc as a PDF and saves it to the specified output directory.

    Args:
        doc_id (str): The ID of the Google Doc to download.
        output_file (str): The path to save the downloaded PDF file.
        service: The Google Drive service instance.

    Returns:
        True if the download was successful, False otherwise.
    """
    try:
        request = service.files().export_media(fileId=doc_id, mimeType='application/pdf')
        fh = io.FileIO(output_file, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        return True
    except HttpError as error:
        print(f'An error occurred during PDF download: {error}')
        # Clean up partially downloaded file if error occurs
        if os.path.exists(output_file):
            os.remove(output_file)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)
        return False
    finally:
        if 'fh' in locals() and not fh.closed:
            fh.close()

def get_image_dimensions_from_doc(doc_id, image_uri, service):
    """Gets the dimensions of an image in a Google Document by its URI.
    
    Args:
        doc_id (str): The ID of the Google Doc containing the image.
        image_uri (str): The URI/filename of the image to find dimensions for.
        service: The Google Docs service instance.
        
    Returns:
        dict: A dictionary containing width and height of the image as it appears in the doc.
            Format: {'width': {'magnitude': float, 'unit': str}, 'height': {'magnitude': float, 'unit': str}}
            Returns None if image not found or error occurs.
    """
    try:
        document = get_doc_content(doc_id, service)
        if not document:
            return None
            
        inline_objects = document.get('inlineObjects', {})
        positioned_objects = document.get('positionedObjects', {})
        
        # Check inline objects first
        for obj_id, obj_data in inline_objects.items():
            embedded_object = obj_data.get('inlineObjectProperties', {}).get('embeddedObject', {})
            if embedded_object and embedded_object.get('imageProperties'):
                image_properties = embedded_object['imageProperties']
                content_uri = image_properties.get('contentUri', '')
                
                # Check if this is the image we're looking for
                if image_uri in content_uri or obj_id in image_uri:
                    size = embedded_object.get('size', {})
                    if size:
                        print(f"Found image dimensions for {image_uri}: {size}")
                        return size
        
        # Check positioned objects
        for obj_id, obj_data in positioned_objects.items():
            pos_obj = obj_data.get('positionedObjectProperties', {})
            embedded_object = pos_obj.get('embeddedObject', {})
            
            if embedded_object and embedded_object.get('imageProperties'):
                image_properties = embedded_object['imageProperties']
                content_uri = image_properties.get('contentUri', '')
                
                # Check if this is the image we're looking for
                if image_uri in content_uri or obj_id in image_uri:
                    size = embedded_object.get('size', {})
                    if size:
                        print(f"Found image dimensions for {image_uri}: {size}")
                        return size
        
        print(f"Image with URI {image_uri} not found in document {doc_id}")
        return None
        
    except Exception as e:
        print(f"Error getting image dimensions: {e}")
        return None

def extract_structure_from_doc(doc_id, service):
    """Extracts the structure of a Google Document as an ordered list of elements.
    
    Parses the document structure and returns an ordered list of elements (text and images)
    with their content and metadata. Elements are ordered as they appear in the document.
    
    Args:
        doc_id (str): The ID of the Google Doc to extract structure from.
        
    Returns:
        list: An ordered list of dictionaries, each representing an element in the document.
            Each element has:
            - 'type': Either 'text' or 'image'
            - 'content': For text, the text string; for images, the image ID
            - 'metadata': Additional information about the element
            Returns None if an error occurs.
    """
    # Use the helper method to get document content
    document = get_doc_content(doc_id, service)
    if not document:
        return None
    
    try:
        # Get the document body content and inline objects
        doc_content = document.get('body', {}).get('content', [])
        inline_objects = document.get('inlineObjects', {})
        positioned_objects = document.get('positionedObjects', {})
        
        if not doc_content:
            print("Document body is empty.")
            return []
        
        # List to store document structure elements in order
        structure = []
        
        # Function to process text elements
        def process_text_element(text_content, element_type="paragraph"):
            if not text_content.strip():
                return None
            
            return {
                'type': 'text',
                'content': text_content,
                'metadata': {
                    'element_type': element_type
                }
            }
        
        # Function to process table rows
        def process_table_rows(table):
            table_elements = []
            rows = table.get('tableRows', [])
            
            for row_idx, row in enumerate(rows):
                cells = row.get('tableCells', [])
                for cell_idx, cell in enumerate(cells):
                    cell_content = cell.get('content', [])
                    for cell_element in cell_content:
                        # Process cell content (recursive)
                        cell_structure = process_structural_element(cell_element)
                        if cell_structure:
                            # Add table position metadata
                            if isinstance(cell_structure, list):
                                for item in cell_structure:
                                    if 'metadata' in item:
                                        item['metadata']['table_position'] = {
                                            'row': row_idx,
                                            'cell': cell_idx
                                        }
                                table_elements.extend(cell_structure)
                            else:
                                if 'metadata' in cell_structure:
                                    cell_structure['metadata']['table_position'] = {
                                        'row': row_idx,
                                        'cell': cell_idx
                                    }
                                table_elements.append(cell_structure)
            
            return table_elements
        
        # Function to process list elements
        def process_list_element(list_item):
            list_elements = []
            content = list_item.get('content', [])
            list_properties = list_item.get('listProperties', {})
            
            for element in content:
                list_structure = process_structural_element(element)
                if list_structure:
                    # Add list metadata
                    if isinstance(list_structure, list):
                        for item in list_structure:
                            if 'metadata' in item:
                                item['metadata']['list_properties'] = list_properties
                        list_elements.extend(list_structure)
                    else:
                        if 'metadata' in list_structure:
                            list_structure['metadata']['list_properties'] = list_properties
                        list_elements.append(list_structure)
            
            return list_elements
        
        # Process different structural elements
        def process_structural_element(element):
            # Process paragraph
            if 'paragraph' in element:
                paragraph = element['paragraph']
                para_elements = paragraph.get('elements', [])
                paragraph_text = ""
                paragraph_items = []
                
                for para_element in para_elements:
                    # Process text run
                    if 'textRun' in para_element:
                        text_run = para_element.get('textRun', {})
                        content = text_run.get('content', '')
                        paragraph_text += content
                    
                    # Process inline image
                    elif 'inlineObjectElement' in para_element:
                        # First add any accumulated text
                        if paragraph_text:
                            text_element = process_text_element(paragraph_text)
                            if text_element:
                                paragraph_items.append(text_element)
                                paragraph_text = ""
                        
                        # Add the inline image
                        inline_obj_id = para_element['inlineObjectElement'].get('inlineObjectId')
                        if inline_obj_id and inline_obj_id in inline_objects:
                            obj_data = inline_objects[inline_obj_id]
                            embedded_obj = obj_data.get('inlineObjectProperties', {}).get('embeddedObject', {})
                            
                            image_element = {
                                'type': 'image',
                                'content': inline_obj_id,
                                'metadata': {
                                    'size': embedded_obj.get('size', {}),
                                    'source': 'inline',
                                    'title': embedded_obj.get('title', '')
                                }
                            }
                            paragraph_items.append(image_element)
                
                # Add any remaining text
                if paragraph_text:
                    text_element = process_text_element(paragraph_text)
                    if text_element:
                        paragraph_items.append(text_element)
                
                return paragraph_items
            
            # Process table
            elif 'table' in element:
                return process_table_rows(element['table'])
            
            # Process list (bullet points, numbered lists)
            elif 'listItem' in element:
                return process_list_element(element['listItem'])
            
            # Other element types can be processed as needed
            return None
        
        # Process the positioned images (not inline with text)
        for obj_id, obj_data in positioned_objects.items():
            pos_obj = obj_data.get('positionedObjectProperties', {})
            embedded_obj = pos_obj.get('embeddedObject', {})
            
            if embedded_obj and 'imageProperties' in embedded_obj:
                image_element = {
                    'type': 'image',
                    'content': obj_id,
                    'metadata': {
                        'size': embedded_obj.get('size', {}),
                        'position': pos_obj.get('positioning', {}),
                        'source': 'positioned', 
                        'title': embedded_obj.get('title', '')
                    }
                }
                structure.append(image_element)
        
        # Process the main document content
        for element in doc_content:
            element_structure = process_structural_element(element)
            if element_structure:
                if isinstance(element_structure, list):
                    structure.extend(element_structure)
                else:
                    structure.append(element_structure)
        
        print(f"Extracted document structure with {len(structure)} elements")
        return structure
        
    except Exception as e:
        print(f"An unexpected error occurred during document structure extraction: {e}")
        import traceback
        traceback.print_exc()
        return None


# NOTE: search_sheet, extract_tables_from_sheet, extract_structure_from_sheet
# have been moved to google_sheets_utils.py and are re-exported at the top of this file.


def find_text_structure(text_to_match, doc_structure):
    """Finds the text structure element containing the highest percentage of the matching text.

    Searches through a document structure (from extract_structure_from_doc) and finds
    which text element contains the largest portion of the text to match. This is useful
    for locating where specific text appears in the document structure.

    Args:
        text_to_match (str): The text string to find in the document structure.
        doc_structure (list): The document structure from extract_structure_from_doc.
            Each element is a dict with 'type' ('text' or 'image'), 'content', and 'metadata'.

    Returns:
        tuple: A tuple containing (structure_element, index) or (None, None) if no match found.
            - structure_element (dict): The structure element containing the highest percentage match.
                Includes:
                - 'type': 'text' or 'image'
                - 'content': The text content or image ID
                - 'metadata': Additional metadata about the element
                - 'match_percentage': The percentage of text_to_match found in this element (0-100)
                - 'matched_length': Number of characters matched
                - 'match_start_index': The starting character index of the match within the element content
            - index (int): The index of this element in the doc_structure list
    """
    if not doc_structure or not text_to_match:
        print("Empty document structure or text to match.")
        return None, None

    text_to_match = text_to_match.strip().lower()
    text_length = len(text_to_match)

    if text_length == 0:
        print("Text to match is empty after stripping.")
        return None, None

    best_match = None
    best_match_index = None
    best_match_percentage = 0
    best_matched_length = 0
    best_match_start_index = float('inf')

    # Iterate through all structure elements
    for idx, element in enumerate(doc_structure):
        # Only check text elements
        if element.get('type') != 'text':
            continue

        content = element.get('content', '').strip().lower()
        if not content:
            continue

        # Use difflib to find the longest contiguous match
        matcher = difflib.SequenceMatcher(None, text_to_match, content)
        match = matcher.find_longest_match(0, len(text_to_match), 0, len(content))
        
        matched_chars = match.size
        current_start_index = match.b # Start index in content
        
        # Calculate percentage match based on the longest contiguous block
        match_percentage = (matched_chars / text_length) * 100

        # Update best match if this is better
        if match_percentage > best_match_percentage:
            best_match_percentage = match_percentage
            best_matched_length = matched_chars
            best_match = element.copy()  # Copy to avoid modifying original
            best_match['match_percentage'] = match_percentage
            best_match['matched_length'] = matched_chars
            best_match['match_start_index'] = current_start_index
            best_match_index = idx
            
    if best_match:
        print(f"Found best match at index {best_match_index} with {best_match_percentage:.1f}% of text matched ({best_matched_length}/{text_length} chars)")
        return best_match, best_match_index
    else:
        print("No matching text structure found.")
        return None, None

def get_structural_element_order(doc_structure, texts):
    """
    Orders a list of texts based on their sequential appearance in the document structure.
    
    If two texts are in the same structural element, they are ordered by their
    position within that element.

    Args:
        doc_structure (list): Document structure list from extract_structure_from_doc.
        texts (list): List of text strings to order.
        
    Returns:
        list: The input texts sorted by their location in the document.
              Texts not found in the structure are excluded.
    """
    matches = []
    for text in texts:
        element, index = find_text_structure(text, doc_structure)
        if element:
            matches.append({
                'text': text,
                'index': index,
                'start_index': element.get('match_start_index', -1)
            })
        else:
             # Log warning for unfound text
             safe_text = text[:50] + "..." if len(text) > 50 else text
             print(f"Warning: Text not found in structure during ordering: '{safe_text}'")
             
    # Sort by structure index (primary) and content start index (secondary)
    matches.sort(key=lambda x: (x['index'], x['start_index']))
    
    return [m['text'] for m in matches]


# NOTE: extract_charts_from_sheet has been moved to google_sheets_utils.py
# and is re-exported at the top of this file.


def list_drive_folder_files(folder_id, service):
    """List all files in a Google Drive folder.

    Args:
        folder_id (str): The ID of the Google Drive folder to list files from.
        service: The Google Drive service instance.

    Returns:
        list: List of file dictionaries, each containing 'id', 'name', and 'mimeType'.
            Returns empty list if folder is empty or error occurs.
    """
    try:
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType, createdTime, modifiedTime)",
            pageSize=1000
        ).execute()
        files = results.get('files', [])
        print(f"Found {len(files)} files in folder {folder_id}")
        return files
    except Exception as e:
        print(f"Error listing files in folder {folder_id}: {e}")
        return []


def download_drive_file_as_image(file_id, service, output_path=None):
    """Download a file from Google Drive and return as PIL Image.

    Args:
        file_id (str): The ID of the file to download.
        service: The Google Drive service instance.
        output_path (str, optional): Path to save the downloaded file.
            If None, the image is only returned as PIL Image without saving.

    Returns:
        PIL.Image.Image: The downloaded image, or None if download failed.
    """
    from PIL import Image

    try:
        # Get file metadata to determine mime type
        file_metadata = service.files().get(fileId=file_id, fields='name, mimeType').execute()
        file_name = file_metadata.get('name', 'unknown')
        mime_type = file_metadata.get('mimeType', '')

        # Download file content
        request = service.files().get_media(fileId=file_id)
        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download {file_name}: {int(status.progress() * 100)}%")

        file_content.seek(0)

        # Open as PIL Image
        image = Image.open(file_content)

        # Save to output path if specified
        if output_path:
            image.save(output_path)
            print(f"Saved image to: {output_path}")

        return image

    except Exception as e:
        print(f"Error downloading file {file_id}: {e}")
        return None


def download_drive_file_bytes(file_id, service):
    """Download a file from Google Drive and return as bytes.

    Args:
        file_id (str): The ID of the file to download.
        service: The Google Drive service instance.

    Returns:
        bytes: The file content as bytes, or None if download failed.
    """
    try:
        request = service.files().get_media(fileId=file_id)
        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        file_content.seek(0)
        return file_content.read()

    except Exception as e:
        print(f"Error downloading file {file_id}: {e}")
        return None


def download_drive_image_threadsafe(file_id, access_token):
    """Download image from Google Drive using requests (thread-safe).

    This function uses the requests library instead of the Google API client,
    making it safe to use in multithreaded environments without SSL issues.

    Args:
        file_id (str): The ID of the file to download.
        access_token (str): OAuth access token for authentication.

    Returns:
        PIL.Image.Image: The downloaded image, or None if download failed.
    """
    from PIL import Image

    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            print(f"Error downloading {file_id}: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Exception downloading {file_id}: {e}")
    return None


# NOTE: parse_sheet_to_dataframe has been moved to google_sheets_utils.py
# and is re-exported at the top of this file.


def extract_text_colors_from_doc(document: dict) -> Dict[str, Tuple[float, float, float]]:
    """Extract text content with background colors as RGB tuples.

    Parses Google Docs JSON to extract text and their background colors.
    Useful for tasks that require color-coded text validation.

    Args:
        document: Google Docs API document JSON from get_doc_content().

    Returns:
        Dict mapping text to RGB color tuple (r, g, b) where values are 0-1 floats.
        Default color (white or no color) is (1.0, 1.0, 1.0).
    """
    from typing import Dict, Tuple

    text_colors = {}

    try:
        for element in document.get('body', {}).get('content', []):
            if 'paragraph' not in element:
                continue

            para = element['paragraph']

            for elem in para.get('elements', []):
                if 'textRun' not in elem:
                    continue

                text_run = elem['textRun']
                text = text_run.get('content', '').strip()

                if not text:
                    continue

                # Extract background color
                text_style = text_run.get('textStyle', {})
                bg_color = text_style.get('backgroundColor', {})
                color_data = bg_color.get('color', {})
                rgb_color = color_data.get('rgbColor', {})

                # Default to white if no color specified
                r = rgb_color.get('red', 1.0)
                g = rgb_color.get('green', 1.0)
                b = rgb_color.get('blue', 1.0)

                text_colors[text] = (r, g, b)

    except Exception as e:
        print(f"Error extracting text colors: {e}")

    return text_colors