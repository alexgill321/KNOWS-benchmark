from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv

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
        # Add more mappings as needed
    }
    
    # Generate the list of scopes
    scopes = [scope_mappings[service] for service in services if service in scope_mappings]
    return scopes