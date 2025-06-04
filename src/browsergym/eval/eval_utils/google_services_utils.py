import sys
import os
sys.path.append(os.getcwd())
from src.browsergym.eval.eval_utils.google_services_helpers import *
import requests
import mimetypes
from google.oauth2.service_account import Credentials # For Service Account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.cloud import secretmanager
import io
import json

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID") # e.g., your-project-id
SECRET_ID = os.environ.get("DRIVE_SA_SECRET_ID")   # e.g., doc-eval-service-account-key
SECRET_VERSION_ID = os.environ.get("DRIVE_SA_SECRET_VERSION_ID", "latest")

# Global variable to hold initialized Google API services
# (Initialize them once, not on every request)
DRIVE_SERVICE = None
DOCS_SERVICE = None

def initialize_google_services():
    global DRIVE_SERVICE, DOCS_SERVICE

    if not GCP_PROJECT_ID or not SECRET_ID:
        print("Error: GCP_PROJECT_ID or DRIVE_SA_SECRET_ID environment variables not set.")
        print("Attempting local initialization...")

        # Attempt to initialize without Secret Manager (for local development)
        
        try:
            credentials = authenticate(['DRIVE', 'DOCS'])
            DRIVE_SERVICE = build('drive', 'v3', credentials=credentials)
            DOCS_SERVICE = build('docs', 'v1', credentials=credentials)
            print("Successfully initialized Google API services using local credentials.")
            return DRIVE_SERVICE, DOCS_SERVICE
        except Exception as e:
            print(f"Error initializing Google API services locally: {e}")
            return None, None        

    try:
        # Create the Secret Manager client
        client = secretmanager.SecretManagerServiceClient()

        # Build the resource name of the secret version
        name = f"projects/{GCP_PROJECT_ID}/secrets/{SECRET_ID}/versions/{SECRET_VERSION_ID}"

        # Access the secret version
        response = client.access_secret_version(request={"name": name})
        payload = response.payload.data.decode("UTF-8")
        
        # The payload is the JSON string of your service account key
        service_account_info = json.loads(payload)

        # Define the scopes your application needs
        # Adjust these based on what your 'search_doc', 'extract_images_from_doc', etc. require
        SCOPES = [
            'https://www.googleapis.com/auth/documents', # Example: if only reading
            # 'https://www.googleapis.com/auth/drive', # If needing to write/modify
            'https://www.googleapis.com/auth/drive' # For Google Docs API
            # 'https://www.googleapis.com/auth/documents'
        ]

        # Create credentials from the service account info
        credentials = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)

        # Build the service objects
        DRIVE_SERVICE = build('drive', 'v3', credentials=credentials)
        DOCS_SERVICE = build('docs', 'v1', credentials=credentials)
        print("Successfully initialized Google API services using Service Account from Secret Manager.")
        return DRIVE_SERVICE, DOCS_SERVICE
    except Exception as e:
        print(f"Error initializing Google API services: {e}")
        # Handle the error appropriately (e.g., log it, raise an exception, etc.)
        return None, None


def search_doc(filename, service, folder_id=None):
    """Search for a Google Doc by its filename.

    Args:
        filename (str): The name of the Google Doc to search for.
        folder_id (str, optional): The ID of the Google Drive folder to search in. If None, searches in the entire Drive.

    Returns:
        A tuple (status, doc_id) containing:
            - status (int): 0 if not found, 1 if found in any location, 2 if found in specified location.
            - doc_id (str): The ID of the found Google Doc, or None if not found.
    """
    doc = None
    if folder_id:
        doc =  find_doc_specified_location(folder_id, filename, service)
    if doc is None:
        doc = find_doc_any(filename, service)
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

def find_doc_any(filename, service):
    """Find a Google Doc by its filename.

    Args:
        filename (str): The name of the Google Doc to find.

    Returns:
        The ID of the found Google Doc.
    """
    query = f"name='{filename}' and mimeType='application/vnd.google-apps.document' and trashed=false"
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)',
        pageSize=10 # Look for up to 10 matches
    ).execute()
    items = results.get('files', [])

    if not items:
        print("No Google Docs found with the specified filename.")
        return None
    elif len(items) > 1:
        print("Multiple Google Docs found with the specified filename.")
        ids = [item["id"] for item in items]
        return ids
    else:
        doc_id = items[0]['id']
        print(f"Found document: '{items[0]['name']}' (ID: {doc_id})")
        return doc_id
                   
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

