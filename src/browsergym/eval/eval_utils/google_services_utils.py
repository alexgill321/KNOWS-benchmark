import sys
sys.path.append("C:/Users/alexg/Documents/GitHub/Agent-Benchmark")
from eval.eval_utils.google_services_helpers import *
import requests
import mimetypes
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import io

def search_doc(filename, folder_id=None):
    """
    Search for a Google Doc by its filename.

    Args:
        filename (str): The name of the Google Doc to search for.
        folder_id (str, optional): The ID of the Google Drive folder to search in. If None, searches in the entire Drive.

    Returns:
        tuple: A tuple containing:

            - An integer indicating the search result:
                - 0: No document found
                - 1: Document found in the entire Drive
                - 2: Document found in the specified folder

            - The ID of the found Google Doc, or None if not found.
    """
    doc = None
    if folder_id:
        doc =  find_doc_specified_location(folder_id, filename)
    if doc is None:
        doc = find_doc_any(filename)
        if doc is None:
            return 0, None
        return 1, doc
    else:
        return 2, doc

def find_doc_specified_location(folder_id, filename):
    """
    Find a Google Doc in a specified folder by its filename.

    Args:
        folder_id (str): The ID of the Google Drive folder to search in.
        filename (str): The name of the Google Doc to find.

    Returns:
        file_id (str): The ID of the found Google Doc, or None if not found.
    """
    service = build('drive', 'v3', credentials=authenticate(services=['DRIVE']))

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
        
def find_doc_any(filename):
    """
    Find a Google Doc by its filename.

    Args:
        filename (str): The name of the Google Doc to find.

    Returns:
        file_id (str): The ID of the found Google Doc.
    """
    service = build('drive', 'v3', credentials=authenticate(services=['DRIVE']))

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
                   
def extract_images_from_doc(doc_id, output_dir=None):
    """
    Extracts images from a gooogle document

    args:
        doc_id (str): The ID of the Google Doc to extract images from.
        output_dir (str): The directory to save the extracted images. If None, the extracted images will not be saved
    """
    service = build('docs', 'v1', credentials=authenticate(services=['DOCS']))
    if output_dir is not None:
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

    document = service.documents().get(documentId=doc_id).execute()

    doc_content = document.get("body").get("content")
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
                print(f"Found image {image_count} (Object ID: {obj_id}) at URI: {content_uri}")

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

def extract_text_from_doc(doc_id):
    """
    Extracts all text content from a Google Document.

    Args:
        doc_id (str): The ID of the Google Doc to extract text from.

    Returns:
        dict: A dictionary containing the extracted text under the key 'text'.
              Returns {'text': ''} if the document has no text content.
              Returns None if an error occurs (e.g., document not found, API error).
    """
    try:
        # Build the Docs API service
        service = build('docs', 'v1', credentials=authenticate(services=['DOCS']))

        print(f"Fetching document content for ID: {doc_id}")
        # Retrieve the document content
        document = service.documents().get(documentId=doc_id).execute()
        print("Document content fetched successfully.")

        doc_content = document.get('body', {}).get('content')

        if not doc_content:
            print("Document body or content is empty.")
            # Return empty text if the document structure is there but no content
            return {'text': ''} 

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

def download_doc_as_pdf(doc_id, output_file, service=None):
    """
    Downloads a Google Doc as a PDF and saves it to the specified output directory.

    args:
        doc_id (str): The ID of the Google Doc to download.
        output_file (str): The path to save the downloaded PDF file.
        service: The Google Drive service instance. If None, it will be created.
    """
    if service is None:
        service = build('drive', 'v3', credentials=authenticate(services=['DRIVE']))
    try:
        request = service.files().export_media(fileId=doc_id, mimeType='application/pdf')
        fh = io.FileIO(output_file, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        print(f"Downloading '{doc_id}' as PDF to '{output_file}'...")
        while done is False:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download {int(status.progress() * 100)}%.")
        print("Download complete.")
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
                        
