"""Utility functions for fetching stock market data for the investment tracker task."""

import requests
import os
from typing import List, Dict, Optional, Union
from datetime import datetime
from bs4 import BeautifulSoup
import re
import pandas as pd


def parse_currency_value(value: Union[str, int, float]) -> Optional[float]:
    """Strip currency symbols/commas and convert to float.

    Handles values like "$109.99", "€1,234.56", "-$5.00", "($5.00)".

    Args:
        value: Raw value from a spreadsheet cell.

    Returns:
        float or None if unparseable.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = str(value).strip()
    if not s:
        return None
    # Handle accounting-style negatives: ($5.00) -> -5.00
    negative = False
    if s.startswith('(') and s.endswith(')'):
        negative = True
        s = s[1:-1]
    # Strip currency symbols and commas
    s = re.sub(r'[^\d.\-+]', '', s)
    try:
        result = float(s)
        return -result if negative else result
    except ValueError:
        return None


def load_alpha_vantage_api_key() -> Optional[str]:
    """Load Alpha Vantage API key from auth-data directory."""
    try:
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..', '..', '..', '..')
        auth_file = os.path.join(project_root, 'auth-data', 'alpha_vantage_api.txt')
        
        with open(auth_file, 'r') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading API key: {e}")
        return None


def get_top_tech_stocks_alpha_vantage(api_key: str, count: int = 10) -> List[Dict]:
    """
    Get top tech stocks by market cap using Alpha Vantage API.
    
    Args:
        api_key: Alpha Vantage API key
        count: Number of top stocks to return
        
    Returns:
        List of dictionaries containing stock information
    """
    try:
        # Alpha Vantage doesn't have a direct "top stocks by market cap" endpoint
        # So we'll use their sector performance endpoint to get tech stocks
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'SECTOR',
            'apikey': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # # If API response doesn't provide what we need, fall back to known top tech stocks
        # # but still validate they exist via API
        # top_tech_stocks = [
        #     {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
        #     {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
        #     {"symbol": "GOOGL", "name": "Alphabet Inc. Class A", "sector": "Technology"},
        #     {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Technology"},
        #     {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology"},
        #     {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Technology"},
        #     {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology"},
        #     {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Technology"},
        #     {"symbol": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
        #     {"symbol": "CRM", "name": "Salesforce Inc.", "sector": "Technology"}
        # ]
        
        # Validate stocks exist by checking if we can get their quotes
        validated_stocks = []
        for stock in data[:count]:
            if validate_stock_symbol(stock['symbol'], api_key):
                validated_stocks.append(stock)
                
        return validated_stocks
        
    except Exception as e:
        print(f"Error fetching stocks from Alpha Vantage: {e}")
        # Return fallback list
        return [
            {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
            {"symbol": "GOOGL", "name": "Alphabet Inc. Class A", "sector": "Technology"},
            {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Technology"},
            {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology"},
            {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Technology"},
            {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology"},
            {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Technology"},
            {"symbol": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
            {"symbol": "CRM", "name": "Salesforce Inc.", "sector": "Technology"}
        ][:count]


def validate_stock_symbol(symbol: str, api_key: str) -> bool:
    """
    Validate if a stock symbol exists using Alpha Vantage API.
    
    Args:
        symbol: Stock ticker symbol
        api_key: Alpha Vantage API key
        
    Returns:
        True if symbol is valid, False otherwise
    """
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return 'Global Quote' in data and '01. symbol' in data['Global Quote']
        
    except Exception:
        return False


def scrape_companies_market_cap(url) -> List[Dict]:
    """
    Scrape tech companies data from companiesmarketcap.com
    
    Returns:
        List of dictionaries containing company data
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        print(f"Fetching data from: {url}")
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        print(f"Page loaded, parsing HTML content...")
        
        companies = []
        
        # Try different selectors to find the table
        table_selectors = [
            'table',
            '.table',
            '.marketcap-table',
            '[class*="table"]',
            'tbody tr',
            '.company-list tr'
        ]
        
        rows = []
        for selector in table_selectors:
            elements = soup.select(selector)
            if elements:
                print(f"Found elements with selector: {selector}")
                if selector in ['table', '.table', '.marketcap-table', '[class*="table"]']:
                    # If we found a table, get its rows
                    for table in elements:
                        table_rows = table.find_all('tr')
                        rows.extend(table_rows)
                else:
                    rows.extend(elements)
                break
        
        if not rows:
            # Fallback: try to find any structured data
            print("No table found, trying div-based layout...")
            rows = soup.find_all('div', class_=re.compile(r'company|row|item|entry'))
        
        print(f"Found {len(rows)} potential data rows")
        
        # Process each row
        for i, row in enumerate(rows):
            try:
                company_data = extract_company_data_from_row_improved(row)
                if company_data:
                    companies.append(company_data)
                    print(f"Extracted: {company_data}")
                    
                # Limit to avoid too much output during testing
                if len(companies) >= 100:
                    break
                    
            except Exception as e:
                continue
        
        print(f"Successfully extracted {len(companies)} companies")        
        return companies
        
    except Exception as e:
        print(f"Error scraping companies market cap website: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_company_data_from_row(row) -> Optional[Dict]:
    """
    Extract company data from a table row or div element.
    
    Args:
        row: BeautifulSoup element containing company data
        
    Returns:
        Dictionary with company data or None if extraction fails
    """
    try:
        # Look for text content in the row
        text_content = row.get_text(strip=True)
        
        # Skip if no meaningful content
        if len(text_content) < 10:
            return None
            
        # Look for stock symbols (typically 1-5 uppercase letters)
        symbol_match = re.search(r'\b([A-Z]{1,5})\b', text_content)
        if not symbol_match:
            return None
            
        symbol = symbol_match.group(1)
        
        # Skip common false positives
        if symbol in ['USD', 'CEO', 'INC', 'CORP', 'LTD', 'LLC']:
            return None
            
        # Look for company name (usually before the symbol or in a specific element)
        name_candidates = []
        
        # Try to find name in links or specific elements
        links = row.find_all('a')
        for link in links:
            link_text = link.get_text(strip=True)
            if link_text and len(link_text) > 2 and not link_text.isdigit():
                name_candidates.append(link_text)
                
        # Try to find name in spans or divs
        spans = row.find_all(['span', 'div'])
        for span in spans:
            span_text = span.get_text(strip=True)
            if span_text and len(span_text) > 2 and not span_text.isdigit() and '$' not in span_text:
                name_candidates.append(span_text)
        
        # Choose the best name candidate
        company_name = None
        for candidate in name_candidates:
            if len(candidate) > 5 and candidate != symbol:
                company_name = candidate
                break
                
        if not company_name:
            company_name = f"Company {symbol}"
            
        # Look for market cap (contains $ and B/M/T)
        market_cap_match = re.search(r'\$[\d.,]+\s*[BMT]', text_content)
        market_cap = market_cap_match.group(0) if market_cap_match else "N/A"
        
        return {
            'symbol': symbol,
            'name': company_name.strip(),
            'sector': 'Technology',
            'market_cap': market_cap
        }
        
    except Exception:
        return None


def extract_company_data_from_row_improved(row) -> Optional[Dict]:
    """
    Improved extraction function for company data from table rows.
    
    Args:
        row: BeautifulSoup element containing company data
        
    Returns:
        Dictionary with company data or None if extraction fails
    """
    try:
        # Get all text content from the row
        text_content = row.get_text(separator='|', strip=True)
        
        # Skip if no meaningful content or header rows
        if len(text_content) < 10 or 'Rank' in text_content or 'Name' in text_content or 'Close Ad' in text_content:
            return None
        
        # Look for cells/columns in the row
        cells = row.find_all(['td', 'th'])
        if len(cells) < 3:  # Need at least rank, name, market cap
            return None
        
        rank = None
        company_name = None
        symbol = None
        market_cap = None
        
        # Extract data from each cell
        for i, cell in enumerate(cells):
            cell_text = cell.get_text(strip=True)
            
            # Skip empty cells
            if not cell_text:
                continue
                
            # Rank (usually first column, just a number)
            if i == 0 and cell_text.isdigit():
                rank = int(cell_text)
                continue
            
            # Market cap (contains $ and B/M/T)
            if '$' in cell_text and re.search(r'[\d.,]+\s*[BMT]', cell_text):
                market_cap = cell_text
                continue
            
            # Look for company name and symbol pattern
            # Common patterns: "CompanyName" followed by "SYMBOL" or "CompanyNameSYMBOL"
            if i == 1 or (not company_name and len(cell_text) > 3):
                # Try to separate company name and symbol
                # Pattern 1: Company name with symbol at the end (e.g., "MicrosoftMSFT")
                name_symbol_match = re.match(r'^(.+?)([A-Z]{2,5})$', cell_text)
                if name_symbol_match:
                    potential_name = name_symbol_match.group(1).strip()
                    potential_symbol = name_symbol_match.group(2)
                    
                    # Validate this looks like a company name
                    if len(potential_name) > 2 and potential_symbol not in ['USD', 'CEO', 'INC', 'CORP', 'LTD', 'LLC']:
                        company_name = potential_name
                        symbol = potential_symbol
                        continue
                
                # Pattern 2: Look for symbols in parentheses
                paren_match = re.search(r'^(.+?)\s*\(([A-Z]{1,5})\)', cell_text)
                if paren_match:
                    company_name = paren_match.group(1).strip()
                    symbol = paren_match.group(2)
                    continue
                
                # Pattern 3: Plain company name (symbol might be in another cell)
                if not company_name and len(cell_text) > 3:
                    # Clean the text
                    clean_text = re.sub(r'[^a-zA-Z\s\(\)&\.]', '', cell_text).strip()
                    if len(clean_text) > 2:
                        company_name = clean_text
            
            # Look for standalone symbols
            if not symbol and len(cell_text) <= 5 and cell_text.isupper():
                if cell_text not in ['USD', 'CEO', 'INC', 'CORP', 'LTD', 'LLC', 'THE', 'AND']:
                    symbol = cell_text
        
        # Post-processing: clean up extracted data
        if company_name:
            # Remove common suffixes and clean up
            company_name = re.sub(r'\s+(Inc|Corp|Ltd|LLC|Co)\.?$', '', company_name)
            company_name = company_name.strip()
        
        # Validate we have the essential data
        if symbol and company_name and len(symbol) <= 5:
            return {
                'rank': rank,
                'symbol': symbol,
                'name': company_name,
                'sector': 'Technology',
                'market_cap': market_cap or 'N/A'
            }
        
        return None
        
    except Exception as e:
        return None


def download_csv_data_direct(url: str, save_directory: str = "output") -> Optional[str]:
    """
    Download CSV data directly by appending /?download=csv to the URL.
    
    Args:
        url: Base URL to download CSV from
        save_directory: Directory to save the CSV file
        
    Returns:
        Path to saved CSV file or None if failed
    """
    try:
        # Construct CSV download URL
        csv_url = url.rstrip('/') + '/?download=csv'
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/csv,application/csv,text/plain,*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        print(f"Attempting to download CSV from: {csv_url}")
        response = requests.get(csv_url, headers=headers, timeout=20)
        response.raise_for_status()
        
        # Check if response looks like CSV content
        content_type = response.headers.get('content-type', '').lower()
        content = response.text.strip()
        
        # Validate CSV-like content
        if ('csv' in content_type or 
            content.count(',') > 10 or 
            '\n' in content[:500]):  # Check first 500 chars for newlines
            
            print("Successfully downloaded CSV data")
            
            # Create save directory if it doesn't exist
            os.makedirs(save_directory, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            url_parts = url.split('/')
            filename = f"{url_parts[-1]}_{timestamp}.csv"
            filepath = os.path.join(save_directory, filename)
            
            # Save CSV content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"CSV file saved to: {filepath}")
            return filepath
        else:
            print("Response does not appear to be CSV data")
            print(f"Content-Type: {content_type}")
            print(f"First 200 characters: {content[:200]}")
            return None
            
    except Exception as e:
        print(f"Error downloading CSV from {csv_url}: {e}")
        return None
    

def download_csv_data(url, save_directory: str = "output") -> Optional[str]:
    """
    Download CSV data by finding the download button on the webpage and save to specified directory.
    
    Args:
        save_directory: Directory to save the CSV file
        
    Returns:
        Path to saved CSV file or None if failed
    """
    try:
        # First load the main page to find the CSV download link
        url = "https://companiesmarketcap.com/tech/largest-tech-companies-by-market-cap/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        print(f"Loading page to find CSV download button: {url}")
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for CSV download links/buttons
        csv_download_url = None
        
        # Common patterns for CSV download links
        csv_selectors = [
            'a[href*="csv"]',
            'a[href*="download"]',
            'button[onclick*="csv"]',
            'a[download*="csv"]',
            '.download-csv',
            '.csv-download',
            'a[title*="CSV"]',
            'a[title*="csv"]',
            'a:contains("CSV")',
            'a:contains("csv")',
            'button:contains("CSV")',
            'button:contains("csv")'
        ]
        
        for selector in csv_selectors:
            try:
                if ':contains(' in selector:
                    # Handle text-based selectors differently
                    if 'CSV' in selector:
                        elements = soup.find_all('a', string=re.compile(r'CSV', re.IGNORECASE))
                        elements.extend(soup.find_all('button', string=re.compile(r'CSV', re.IGNORECASE)))
                    else:
                        elements = soup.find_all('a', string=re.compile(r'csv', re.IGNORECASE))
                        elements.extend(soup.find_all('button', string=re.compile(r'csv', re.IGNORECASE)))
                else:
                    elements = soup.select(selector)
                
                for element in elements:
                    href = element.get('href')
                    onclick = element.get('onclick', '')
                    
                    if href:
                        # Make URL absolute if it's relative
                        if href.startswith('/'):
                            csv_download_url = f"https://companiesmarketcap.com{href}"
                        elif href.startswith('http'):
                            csv_download_url = href
                        else:
                            csv_download_url = f"https://companiesmarketcap.com/{href}"
                        break
                    elif onclick and 'csv' in onclick.lower():
                        # Extract URL from onclick if possible
                        url_match = re.search(r'["\']([^"\']*csv[^"\']*)["\']', onclick)
                        if url_match:
                            potential_url = url_match.group(1)
                            if potential_url.startswith('/'):
                                csv_download_url = f"https://companiesmarketcap.com{potential_url}"
                            elif potential_url.startswith('http'):
                                csv_download_url = potential_url
                            break
                
                if csv_download_url:
                    break
                    
            except Exception as e:
                continue
        
        # If we found a CSV download URL, try to download it
        if csv_download_url:
            print(f"Found CSV download link: {csv_download_url}")
            try:
                csv_response = requests.get(csv_download_url, headers=headers, timeout=20)
                csv_response.raise_for_status()
                
                # Check if response looks like CSV
                content_type = csv_response.headers.get('content-type', '').lower()
                if 'csv' in content_type or csv_response.text.strip().count(',') > 10:
                    print("Successfully downloaded CSV data")
                    
                    # Create save directory if it doesn't exist
                    os.makedirs(save_directory, exist_ok=True)
                    
                    # Generate filename with timestamp
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"tech_companies_{timestamp}.csv"
                    filepath = os.path.join(save_directory, filename)
                    
                    # Save CSV content to file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(csv_response.text)
                    
                    print(f"CSV file saved to: {filepath}")
                    return filepath
                            
            except Exception as e:
                print(f"Error downloading from CSV URL {csv_download_url}: {e}")
        
        # If no CSV download found or failed, try alternative approaches
        print("No CSV download button found or download failed")
        
        # Look for any export/download functionality
        export_elements = soup.find_all(['a', 'button'], string=re.compile(r'export|download', re.IGNORECASE))
        if export_elements:
            print(f"Found {len(export_elements)} potential export/download elements")
            for elem in export_elements:
                print(f"Export element: {elem.get_text(strip=True)} - {elem.get('href', elem.get('onclick', 'No action'))}")
        
        return None
        
    except Exception as e:
        print(f"Error in download_csv_data: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_top_tech_stocks_from_web(count: int = 10, csv_file: str = None) -> List[Dict]:
    """
    Get top tech stocks by scraping companiesmarketcap.com
    
    Args:
        count: Number of top stocks to return
        
    Returns:
        List of dictionaries containing stock information
    """
    print("Attempting to download CSV data...")
    csv_filepath = download_csv_data(url="https://companiesmarketcap.com/tech/largest-tech-companies-by-market-cap/", save_directory=csv_file)
    
    companies = []
    if csv_filepath and os.path.exists(csv_filepath):
        print(f"CSV file downloaded to: {csv_filepath}")
        try:
            data = pd.read_csv(csv_filepath)
            data = data.sort_values(by='Rank', ascending=True)
            data = data.head(count)  # Limit to top 'count' companies
            companies = data.to_dict(orient='records')
                            
        except Exception as e:
            print(f"Error parsing downloaded CSV: {e}")

    if companies:
        print(f"Successfully retrieved {len(companies)} companies from web")
        return companies[:count]
    else:
        raise ValueError("Stock data could not be retrieved from the web. Please check the CSV download functionality or the website structure.")


def get_top_tech_stocks(count: int = 10) -> List[Dict]:
    """
    Get top tech stocks by market cap using web scraping as primary method.
    
    Args:
        count: Number of top stocks to return
        
    Returns:
        List of dictionaries containing stock information
    """
    # Try web scraping first for most up-to-date data
    return get_top_tech_stocks_from_web(count)


def get_currency_exchange_rate(from_currency: str, to_currency: str = 'USD') -> Optional[float]:
    """
    Get currency exchange rate using Yahoo Finance FX endpoint.
    
    Args:
        from_currency: Source currency code (e.g., 'KRW', 'EUR')
        to_currency: Target currency code (default 'USD')
        
    Returns:
        Exchange rate or None if failed
    """
    if from_currency == to_currency:
        return 1.0
        
    try:
        # Yahoo Finance FX symbol format: FROMTO=X (e.g., KRWUSD=X, EURUSD=X)
        fx_symbol = f"{from_currency}{to_currency}=X"
        
        import urllib.parse
        encoded_symbol = urllib.parse.quote(fx_symbol)
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json,*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://finance.yahoo.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
            result = data['chart']['result'][0]
            if 'meta' in result and 'regularMarketPrice' in result['meta']:
                return float(result['meta']['regularMarketPrice'])
            elif 'meta' in result and 'previousClose' in result['meta']:
                return float(result['meta']['previousClose'])
                
    except Exception as e:
        print(f"Error fetching exchange rate for {from_currency}/{to_currency}: {e}")
        
    return None


def get_stock_price_yahoo_finance(symbol: str) -> Optional[float]:
    """
    Get current stock price using Yahoo Finance API (unofficial).
    Handles both US and foreign stocks with exchange suffixes.
    Returns prices converted to USD for foreign stocks.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', '005930.KS', 'ASML.AS')
        
    Returns:
        Current stock price in USD or None if failed
    """
    try:
        # URL encode the symbol to handle special characters
        import urllib.parse
        encoded_symbol = urllib.parse.quote(symbol)
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json,*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://finance.yahoo.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
            result = data['chart']['result'][0]
            meta = result.get('meta', {})
            
            # Extract price and currency information
            price = None
            currency = meta.get('currency', 'USD')
            
            if 'regularMarketPrice' in meta:
                price = float(meta['regularMarketPrice'])
            elif 'previousClose' in meta:
                price = float(meta['previousClose'])
            
            if price is not None:
                # Convert to USD if necessary
                if currency != 'USD':
                    print(f"Converting {symbol} price from {currency} to USD: {price:.2f} {currency}")
                    exchange_rate = get_currency_exchange_rate(currency, 'USD')
                    if exchange_rate is not None:
                        usd_price = price * exchange_rate
                        print(f"  Exchange rate: 1 {currency} = {exchange_rate:.6f} USD")
                        print(f"  Converted price: {usd_price:.2f} USD")
                        return usd_price
                    else:
                        print(f"  ⚠️ Currency conversion failed, returning original price in {currency}")
                        return price
                else:
                    # Already in USD
                    return price
                
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        
    return None


def get_stock_price_yahoo_finance_alternative(symbol: str) -> Optional[float]:
    """
    Alternative method using Yahoo Finance summary page for foreign stocks.
    Returns prices converted to USD for foreign stocks.
    
    Args:
        symbol: Stock ticker symbol with exchange suffix
        
    Returns:
        Current stock price in USD or None if failed
    """
    try:
        import urllib.parse
        encoded_symbol = urllib.parse.quote(symbol)
        
        # Try the finance summary API endpoint
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{encoded_symbol}"
        params = {
            'modules': 'price,summaryDetail'
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Referer': 'https://finance.yahoo.com/'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        if 'quoteSummary' in data and 'result' in data['quoteSummary'] and data['quoteSummary']['result']:
            result = data['quoteSummary']['result'][0]
            
            price = None
            currency = 'USD'  # Default assumption
            
            # Try price module first
            if 'price' in result:
                price_module = result['price']
                
                # Extract currency if available
                if 'currency' in price_module:
                    currency = price_module['currency']
                
                # Extract price
                if 'regularMarketPrice' in price_module:
                    price_data = price_module['regularMarketPrice']
                    if isinstance(price_data, dict) and 'raw' in price_data:
                        price = float(price_data['raw'])
                    elif isinstance(price_data, (int, float)):
                        price = float(price_data)
            
            # Try summaryDetail module if price not found
            if price is None and 'summaryDetail' in result and 'previousClose' in result['summaryDetail']:
                close_data = result['summaryDetail']['previousClose']
                if isinstance(close_data, dict) and 'raw' in close_data:
                    price = float(close_data['raw'])
                elif isinstance(close_data, (int, float)):
                    price = float(close_data)
            
            # Convert to USD if necessary
            if price is not None:
                if currency != 'USD':
                    print(f"Converting {symbol} price from {currency} to USD: {price:.2f} {currency} (alternative method)")
                    exchange_rate = get_currency_exchange_rate(currency, 'USD')
                    if exchange_rate is not None:
                        usd_price = price * exchange_rate
                        print(f"  Exchange rate: 1 {currency} = {exchange_rate:.6f} USD")
                        print(f"  Converted price: {usd_price:.2f} USD")
                        return usd_price
                    else:
                        print(f"  ⚠️ Currency conversion failed, returning original price in {currency}")
                        return price
                else:
                    # Already in USD
                    return price
                    
    except Exception as e:
        print(f"Error fetching price for {symbol} (alternative method): {e}")
        
    return None


def get_stock_price_robust(symbol: str, api_key: Optional[str] = None) -> Optional[float]:
    """
    Robust stock price fetching that tries multiple methods for foreign stocks.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', '005930.KS', 'ASML.AS')
        api_key: Optional Alpha Vantage API key
        
    Returns:
        Current stock price or None if all methods fail
    """
    print(f"Fetching price for {symbol}...")
    
    # Method 1: Try standard Yahoo Finance chart API
    try:
        price = get_stock_price_yahoo_finance(symbol)
        if price is not None:
            print(f"✅ Yahoo Finance (chart): {symbol} = ${price:.2f}")
            return price
    except Exception as e:
        print(f"❌ Yahoo Finance (chart) failed for {symbol}: {e}")
    
    # Method 2: Try alternative Yahoo Finance quoteSummary API
    try:
        price = get_stock_price_yahoo_finance_alternative(symbol)
        if price is not None:
            print(f"✅ Yahoo Finance (summary): {symbol} = ${price:.2f}")
            return price
    except Exception as e:
        print(f"❌ Yahoo Finance (summary) failed for {symbol}: {e}")
    
    # Method 3: Try Alpha Vantage if API key provided
    if api_key:
        try:
            price = get_stock_price_alpha_vantage(symbol, api_key)
            if price is not None:
                print(f"✅ Alpha Vantage: {symbol} = ${price:.2f}")
                return price
        except Exception as e:
            print(f"❌ Alpha Vantage failed for {symbol}: {e}")
    
    # Method 4: For foreign stocks, try different symbol formats
    if '.' in symbol:
        # Try without exchange suffix for some APIs
        base_symbol = symbol.split('.')[0]
        try:
            price = get_stock_price_yahoo_finance(base_symbol)
            if price is not None:
                print(f"✅ Yahoo Finance (base symbol): {base_symbol} = ${price:.2f}")
                return price
        except Exception as e:
            print(f"❌ Base symbol lookup failed for {base_symbol}: {e}")
    
    print(f"❌ All methods failed for {symbol}")
    return None


def get_stock_price_alpha_vantage(symbol: str, api_key: str) -> Optional[float]:
    """
    Get current stock price using Alpha Vantage API (requires API key).
    
    Args:
        symbol: Stock ticker symbol
        api_key: Alpha Vantage API key
        
    Returns:
        Current stock price or None if failed
    """
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if 'Global Quote' in data and '05. price' in data['Global Quote']:
            return float(data['Global Quote']['05. price'])
            
    except Exception as e:
        print(f"Error fetching price for {symbol} via Alpha Vantage: {e}")
        
    return None


def get_historical_stock_price(symbol: str, date_str: str) -> Optional[float]:
    """
    Get historical stock price for a specific date.
    
    Args:
        symbol: Stock ticker symbol
        date_str: Date in YYYY-MM-DD format
        
    Returns:
        Stock price on that date or None if failed
    """
    try:
        # Convert date string to timestamp
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        timestamp = int(date_obj.timestamp())
        
        # Yahoo Finance historical data endpoint
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            'period1': timestamp - 86400,  # Day before
            'period2': timestamp + 86400,  # Day after
            'interval': '1d'
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
            result = data['chart']['result'][0]
            if 'indicators' in result and 'quote' in result['indicators']:
                closes = result['indicators']['quote'][0].get('close', [])
                if closes and any(c is not None for c in closes):
                    # Return the first non-null close price
                    for close in closes:
                        if close is not None:
                            return float(close)
                            
    except Exception as e:
        print(f"Error fetching historical price for {symbol} on {date_str}: {e}")
        
    return None


def verify_past_prices_with_web_content(
    failed_stocks: List[Dict],
    cached_url_contents: Dict[str, str], price_source_url: str, model,
) -> Dict[str, Optional[float]]:
    """Batch fallback price extraction from web content via LLM.

    Args:
        failed_stocks: List of dicts with keys 'ticker', 'company_name'.
        cached_url_contents: Dict mapping URL -> markdown page content.
        price_source_urls: List of URLs to check against.
        model: Loaded LLM model callable.

    Returns:
        Dict mapping ticker -> extracted price as float, or None if not found.
    """
    from src.browsergym.knows.eval.eval_utils.llm_utils import extract_json_with_llm

    if not failed_stocks:
        return {}

    tickers = [s['ticker'] for s in failed_stocks]
    stocks_to_evaluate = "\n".join(f"{s['company_name']} ({s['ticker']})" for s in failed_stocks)
    example_json = "{" + ", ".join(f'"{t}": <price>' for t in tickers) + "}"
    results = {t: None for t in tickers}

    if price_source_url:
        content = cached_url_contents.get(price_source_url, '')
        if content:

            prompt = (
                f"From the web page content below, extract the stock price "
                f"for each of these stocks.\n\n"
                f"Stocks:\n{stocks_to_evaluate}\n\n"
                f"Web page content:\n{content}\n\n"
                f"Respond ONLY with a JSON object mapping each ticker to its price as a number. "
                f"Use null if the price is not found:\n{example_json}"
            )
            extracted = extract_json_with_llm(prompt, model, expect_type="object")
            if extracted:
                for t in tickers:
                    if results[t] is None and extracted.get(t) is not None:
                        results[t] = extracted[t]

    return results


def calculate_portfolio_performance(stocks: List[Dict], shares_per_stock: int = 100) -> Dict:
    """
    Calculate portfolio performance metrics.
    
    Args:
        stocks: List of stock dictionaries with current and historical prices
        shares_per_stock: Number of shares owned per stock
        
    Returns:
        Dictionary with portfolio performance metrics
    """
    total_current_value = 0
    total_historical_value = 0
    stock_values = []
    
    for stock in stocks:
        if 'current_price' in stock and 'q2_2023_price' in stock:
            current_value = stock['current_price'] * shares_per_stock
            historical_value = stock['q2_2023_price'] * shares_per_stock
            
            total_current_value += current_value
            total_historical_value += historical_value
            
            stock_values.append({
                'symbol': stock['symbol'],
                'name': stock['name'],
                'current_value': current_value,
                'historical_value': historical_value,
                'gain_loss': current_value - historical_value,
                'gain_loss_percent': ((current_value - historical_value) / historical_value) * 100
            })
    
    # Calculate percentages of total portfolio
    for stock_value in stock_values:
        stock_value['portfolio_percentage'] = (stock_value['current_value'] / total_current_value) * 100
    
    return {
        'stocks': stock_values,
        'total_current_value': total_current_value,
        'total_historical_value': total_historical_value,
        'total_gain_loss': total_current_value - total_historical_value,
        'total_gain_loss_percent': ((total_current_value - total_historical_value) / total_historical_value) * 100
    }


def get_complete_stock_data(api_key: Optional[str] = None) -> List[Dict]:
    """
    Get complete stock data including current and historical prices.
    
    Args:
        api_key: Optional Alpha Vantage API key for more reliable data
        
    Returns:
        List of stock dictionaries with all required data
    """
    stocks = get_top_tech_stocks(10)
    q2_2023_date = "2023-06-30"  # End of Q2 2023
    
    print("Fetching stock data...")
    
    for i, stock in enumerate(stocks):
        symbol = stock['symbol']
        print(f"Processing {symbol} ({i+1}/10)...")
        
        # Get current price
        if api_key:
            current_price = get_stock_price_alpha_vantage(symbol, api_key)
        else:
            current_price = get_stock_price_yahoo_finance(symbol)
            
        # Get historical price
        historical_price = get_historical_stock_price(symbol, q2_2023_date)
        
        stock['current_price'] = current_price
        stock['q2_2023_price'] = historical_price
        
        if current_price is None or historical_price is None:
            print(f"Warning: Could not fetch complete data for {symbol}")
    
    return stocks


if __name__ == "__main__":
    # Example usage
    print("Fetching top tech stocks data...")
    stock_data = get_complete_stock_data()
    
    # Calculate portfolio performance
    portfolio = calculate_portfolio_performance(stock_data)
    
    print("\nPortfolio Performance Summary:")
    print(f"Total Current Value: ${portfolio['total_current_value']:,.2f}")
    print(f"Total Historical Value (Q2 2023): ${portfolio['total_historical_value']:,.2f}")
    print(f"Total Gain/Loss: ${portfolio['total_gain_loss']:,.2f} ({portfolio['total_gain_loss_percent']:.2f}%)")
    
    print("\nIndividual Stock Performance:")
    for stock in portfolio['stocks']:
        print(f"{stock['symbol']}: ${stock['current_value']:,.2f} ({stock['portfolio_percentage']:.1f}% of portfolio)")
        
def calculate_expected_stock_values(df_data, matched_cols, ticker_map, num_shares, as_percentage=False):
    """
    Calculate expected per-stock portfolio values from table data.

    Per-stock total values are derived from the Current Price column multiplied
    by `num_shares` (rather than read from any "Total value" column on the
    sheet), so the result is independent of whatever total-value formula the
    agent used.

    Args:
        df_data (pd.DataFrame): DataFrame containing the stock data
        matched_cols (dict): Dictionary mapping column names to actual column names in df
        ticker_map (dict): Mapping from gold ticker to user ticker
        num_shares (int): Number of shares owned per stock, per task spec
        as_percentage (bool): If True, return each stock's value as a percentage
            of the overall portfolio total. If False (default), return the raw
            total dollar value per stock.

    Returns:
        list: List of expected values for each stock in the order they appear
              in df (either raw totals or percentages depending on
              `as_percentage`), or empty list if required columns not found.
    """
    if not matched_cols or "Current Price" not in matched_cols:
        print("Warning: Cannot calculate expected values - Current Price column not found")
        return []

    if not ticker_map:
        print("Warning: Cannot calculate expected values - no ticker mapping available")
        return []

    try:
        current_price_col = matched_cols["Current Price"]
        ticker_col = matched_cols.get("Ticker Symbol")

        if not ticker_col:
            print("Warning: Ticker Symbol column not found for ordering")
            return []

        # Derive per-stock total values from current price * num_shares
        current_prices = df_data[current_price_col].apply(parse_currency_value)
        if current_prices.isna().any():
            print("Warning: Some current prices could not be parsed")
            return []
        total_values = current_prices * num_shares

        if not as_percentage:
            return total_values.tolist()

        overall_total = total_values.sum()
        if overall_total == 0:
            print("Warning: Overall portfolio total is 0")
            return []

        return (total_values / overall_total * 100).tolist()

    except Exception as e:
        print(f"Error calculating expected stock values: {e}")
        return []