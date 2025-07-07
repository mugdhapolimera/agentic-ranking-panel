import csv
import json
import requests
from collections import defaultdict
import os
from typing import Dict, List, Set
import time
import re

# ADS API configuration
ADS_API_TOKEN = os.getenv('ADS_API_TOKEN')
ADS_API_URL = 'https://api.adsabs.harvard.edu/v1/search/query'

def get_bibcode_from_doi(doi: str) -> str:
    """Get bibcode from ADS API using DOI."""
    headers = {
        'Authorization': f'Bearer {ADS_API_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    # Query ADS using DOI
    query = f'doi:{doi}'
    
    params = {
        'q': query,
        'fl': 'bibcode'
    }
    
    try:
        response = requests.get(ADS_API_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        docs = data.get('response', {}).get('docs', [])
        if docs:
            return docs[0]['bibcode']
        return None
    except Exception as e:
        print(f"Error fetching bibcode for DOI {doi}: {str(e)}")
        return None

def get_ads_results(bibcode: str) -> List[Dict]:
    """Get similar papers from ADS API for a given bibcode."""
    headers = {
        'Authorization': f'Bearer {ADS_API_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    # Construct the query for similar papers
    query = f'similar({bibcode})'
    
    params = {
        'q': query,
        'fl': 'bibcode,title,abstract',
        'rows': 20  # Adjust as needed
    }
    
    try:
        response = requests.get(ADS_API_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('response', {}).get('docs', [])
    except Exception as e:
        print(f"Error fetching results for {bibcode}: {str(e)}")
        return []

def get_paper_details(bibcode: str) -> Dict:
    """Get paper details from ADS API."""
    headers = {
        'Authorization': f'Bearer {ADS_API_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    params = {
        'q': f'bibcode:{bibcode}',
        'fl': 'bibcode,title,abstract'
    }
    
    try:
        response = requests.get(ADS_API_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        docs = data.get('response', {}).get('docs', [])
        return docs[0] if docs else None
    except Exception as e:
        print(f"Error fetching paper details for {bibcode}: {str(e)}")
        return None

def normalize_title(title: str) -> str:
    """Normalize title for comparison by removing extra spaces and converting to lowercase."""
    # Handle case where title is a list
    if isinstance(title, list):
        title = title[0] if title else ""
    return ' '.join(title.lower().split())

def extract_identifier(query: str) -> str:
    """Extract DOI or bibcode from the query string."""
    # Try to extract content inside parentheses
    if '(' in query and ')' in query:
        identifier = query.split('(')[1].split(')')[0]
    else:
        identifier = query.strip()
    return identifier

def is_probable_bibcode(identifier: str) -> bool:
    # Bibcodes are 19 characters, often with dots and numbers
    return bool(re.match(r'^[12][0-9]{3}[A-Za-z0-9\.\&]{15}$', identifier))

def process_judgements(csv_file: str) -> Dict[str, Dict]:
    """Process the judgements CSV file and organize by query bibcode."""
    judgements = defaultdict(lambda: {
        'search_results': [],
        'judgements_by_bibcode': defaultdict(dict),
        'judgements_by_title': defaultdict(dict)
    })
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row['Query']
            title = row['Title']
            score = float(row['Score'])
            note = row['Note']
            
            identifier = extract_identifier(query)
            
            if is_probable_bibcode(identifier):
                bibcode = identifier
            else:
                bibcode = get_bibcode_from_doi(identifier)
            
            if not bibcode:
                print(f"Warning: Could not find bibcode for identifier {identifier}")
                continue
            
            # Try to extract bibcode from the title if it exists
            # Format: "Title (bibcode)"
            if '(' in title and ')' in title:
                source_bibcode = title.split('(')[-1].split(')')[0].strip()
                judgements[bibcode]['judgements_by_bibcode'][source_bibcode] = {
                    'score': score,
                    'note': note
                }
            else:
                # Store by normalized title if no bibcode found
                judgements[bibcode]['judgements_by_title'][normalize_title(title)] = {
                    'score': score,
                    'note': note
                }
    
    return judgements

def create_json_output(judgements: Dict[str, Dict], output_dir: str):
    """Create JSON files for each bibcode with their judgements."""
    for bibcode, data in judgements.items():
        # Create directory structure
        bibcode_dir = os.path.join(output_dir, bibcode, "inputs")
        os.makedirs(bibcode_dir, exist_ok=True)
        
        # Define output file path
        output_file = os.path.join(bibcode_dir, f"{bibcode}_similar_judgements.json")
        
        # Check if file already exists
        if os.path.exists(output_file):
            print(f"Skipping {output_file} - file already exists")
            continue
        
        # Get paper details
        paper_details = get_paper_details(bibcode)
        if not paper_details:
            print(f"Skipping {bibcode} - could not fetch paper details")
            continue
        
        # Get similar papers
        similar_papers = get_ads_results(bibcode)
        
        # Create the output structure
        output = {
            'bibcode': paper_details['bibcode'],
            'title': paper_details['title'][0] if isinstance(paper_details['title'], list) else paper_details['title'],
            'abstract': paper_details.get('abstract', [''])[0] if isinstance(paper_details.get('abstract', ['']), list) else paper_details.get('abstract', ''),
            'search_results': []
        }
        
        # Add judgements to the search results
        for paper in similar_papers:
            # Try to match by bibcode first
            if paper['bibcode'] in data['judgements_by_bibcode']:
                judgement = data['judgements_by_bibcode'][paper['bibcode']]
                paper['sme_judgements'] = judgement['score']
                paper['notes'] = judgement['note']
            else:
                # Fall back to title matching
                paper_title = normalize_title(paper['title'])
                if paper_title in data['judgements_by_title']:
                    judgement = data['judgements_by_title'][paper_title]
                    paper['sme_judgements'] = judgement['score']
                    paper['notes'] = judgement['note']
                else:
                    paper['sme_judgements'] = None
                    paper['notes'] = None
            
            # Ensure title and abstract are strings in the output
            if isinstance(paper['title'], list):
                paper['title'] = paper['title'][0]
            if isinstance(paper.get('abstract', ''), list):
                paper['abstract'] = paper['abstract'][0]
            elif 'abstract' not in paper:
                paper['abstract'] = ''
            
            output['search_results'].append(paper)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Created {output_file}")
        time.sleep(1)  # Be nice to the API

def main():
    if not ADS_API_TOKEN:
        print("Error: ADS_API_TOKEN environment variable not set")
        return
    
    csv_files = ["/Users/mugdhapolimera/Downloads/judgements_query_10_2478_gsr-2021-0001.csv",
                 "/Users/mugdhapolimera/Downloads/judgements_query_2024PSJ_____5__183J.csv",
                 "/Users/mugdhapolimera/Downloads/judgements_query_2023PSJ_____4__214R.csv",
                 "/Users/mugdhapolimera/Downloads/judgements_query_2023arXiv231214211B.csv",
                 "/Users/mugdhapolimera/Downloads/judgements_query_2022Icar__38815237J.csv",
                 "/Users/mugdhapolimera/Downloads/judgements_query_2021pds__data__188J.csv",
                 "/Users/mugdhapolimera/Downloads/judgements_query_2020PhDT________27J.csv",
                 "/Users/mugdhapolimera/Downloads/judgements_query_2020AcAau_170____6J.csv",
                 "/Users/mugdhapolimera/Downloads/judgements_query_2019AcAau_155__131J.csv",
                 "/Users/mugdhapolimera/Downloads/judgements_query_1987JGR____9214969O.csv",
                 "/Users/mugdhapolimera/Downloads/judgements_query_10_2478_gsr-2021-0001.csv",
                 ]
    output_dir = "examples"
    
    for csv_file in csv_files:
        print("Processing judgements...")
        judgements = process_judgements(csv_file)
        
        print("Creating JSON output...")
        create_json_output(judgements, output_dir)
        
        print("Done!")

if __name__ == "__main__":
    main() 