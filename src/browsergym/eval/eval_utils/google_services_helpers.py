from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
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

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Auto-refresh using stored refresh token (no browser needed)
            creds.refresh(Request())
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        elif creds_path is not None:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, scopes)
            creds = flow.run_local_server(port=0)
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        else:
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


def detect_header_row(rows: list, max_rows_to_check: int = 10) -> int:
    """Detect which row contains the table headers in Google Sheets data.

    Uses heuristics to find the header row:
    1. Row with the most non-empty cells
    2. Row where values look like headers (text, not numbers)
    3. Prefers rows early in the sheet

    Args:
        rows: List of row data from Google Sheets API (rowData from get_sheet_content).
        max_rows_to_check: Maximum number of rows to scan for headers.

    Returns:
        0-indexed row number most likely to be the header row.
    """
    if not rows:
        return 0

    best_row = 0
    best_score = -1

    for row_idx in range(min(len(rows), max_rows_to_check)):
        row = rows[row_idx]
        values = row.get('values', [])

        if not values:
            continue

        # Count non-empty cells
        non_empty_count = 0
        text_count = 0
        total_cells = len(values)

        for cell in values:
            formatted = cell.get('formattedValue', '')
            if formatted:
                non_empty_count += 1
                # Check if it looks like text (not purely numeric)
                try:
                    float(formatted.replace(',', '').replace('$', '').replace('%', ''))
                except ValueError:
                    text_count += 1

        if non_empty_count == 0:
            continue

        # Score: prioritize rows with many non-empty text cells
        # Penalize rows that are too early (row 0 often has titles)
        text_ratio = text_count / non_empty_count if non_empty_count > 0 else 0
        density = non_empty_count / max(total_cells, 1)

        # Score combines: text ratio (headers are text), density (headers fill row),
        # non-empty count (more columns = more likely header)
        score = (text_ratio * 0.4) + (density * 0.3) + (non_empty_count * 0.02)

        # Small bonus for rows 1-3 (common header positions)
        if 1 <= row_idx <= 3:
            score += 0.1

        if score > best_score:
            best_score = score
            best_row = row_idx

    return best_row
