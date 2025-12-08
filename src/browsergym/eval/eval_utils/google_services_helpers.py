from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os

def authenticate(services):
    creds = None
    scopes = get_scopes(services)
    token_path = os.environ.get('TOKEN_PATH', os.getcwd() + '/auth-data/token.json')
    creds_path = os.environ.get('CLIENT_SECRETS_PATH', os.getcwd() + '/auth-data/credentials.json')

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, scopes)
    
    if not creds or not creds.valid and creds_path is not None:
        flow = InstalledAppFlow.from_client_secrets_file(creds_path, scopes)
        creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    elif creds_path is None:
        raise ValueError("Credentials path is required for the first time authentication.")
    return creds

    

def get_scopes(services):
    """
    Maps a list of service names to their corresponding Google API scopes.

    Args:
        services (list): List of service names (e.g., ['DRIVE', 'DOCS']).
        
    Returns:
        list: List of corresponding Google API scopes.
    """
    scope_mappings = {
        'DRIVE': 'https://www.googleapis.com/auth/drive',
        'DOCS': 'https://www.googleapis.com/auth/documents',
        'SHEETS': 'https://www.googleapis.com/auth/spreadsheets',
        'SLIDES': 'https://www.googleapis.com/auth/presentations',
        # Add more mappings as needed
    }
    
    # Generate the list of scopes
    scopes = [scope_mappings[service] for service in services if service in scope_mappings]
    return scopes
    
def get_doc_content(doc_id, service):
    """Fetches the content of a Google Document.
    
    Args:
        doc_id (str): The ID of the Google Doc to fetch.
        
    Returns:
        dict: The content of the document in JSON format.
    """
    try:
        
        print(f"Fetching document content for ID: {doc_id}")
        # Retrieve the document content
        document = service.documents().get(documentId=doc_id).execute()
        print("Document content fetched successfully.")
        
        return document
    except Exception as e:
        print(f"Error fetching document content: {e}")
        return None

def get_sheet_content(sheet_id, service):
    """
    Fetches the content of a Google Sheet.

    Args:
        sheet_id (str): The ID of the Google Sheet.
        service: The Google Sheets service instance.

    Returns:
        dict: The full spreadsheet data. Returns None if an error occurs.
    """
    try:
        print(f"Fetching sheet content for ID: {sheet_id}")

        # `includeGridData=True` includes values, formatting, and layout info
        sheet = service.spreadsheets().get(spreadsheetId=sheet_id, includeGridData=True).execute()

        print("Sheet content fetched successfully.")
        return sheet

    except Exception as e:
        print(f"Error fetching sheet content: {e}")
        return None


def get_slide_content(slide_id):
    """
    Fetches the content of a Google Slides presentation.

    Args:
        slide_id (str): The ID of the Slides presentation.

    Returns:
        tuple: (presentation, service) where `presentation` is the full slide deck data and `service`
         is the Slides API service. Returns (None, None) if an error occurs.
    """
    try:
        service = build('slides', 'v1', credentials=authenticate(services=['SLIDES']))
        print(f"Fetching slide content for ID: {slide_id}")

        presentation = service.presentations().get(presentationId=slide_id).execute()

        print("Slides content fetched successfully.")
        return presentation, service

    except Exception as e:
        print(f"Error fetching slide content: {e}")
        return None, None
