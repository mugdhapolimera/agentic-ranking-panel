import json
import requests
from urllib.parse import quote
import tomli
from dataclasses import dataclass
from typing import List, Optional
import os

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the root directory (parent of script directory)
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Read API keys from secrets.toml
with open(os.path.join(ROOT_DIR, "secrets.toml"), "rb") as f:
    secrets = tomli.load(f)

# Get ADS API token
ads_api_token = secrets["api_keys"]["ads_api_token"]

@dataclass
class SearchResult:
    bibcode: str
    title: str
    abstract: str
    claude_score: Optional[int] = None
    gemini_score: Optional[int] = None
    deepseek_score: Optional[int] = None

@dataclass
class PaperData:
    bibcode: str
    title: str
    abstract: str
    search_results: List[SearchResult]
    claude_relative_score: Optional[int] = None
    gemini_relative_score: Optional[int] = None
    deepseek_relative_score: Optional[int] = None

    def to_json(self, json_path: str) -> None:
        """Save paper data to a JSON file."""
        data = {
            'bibcode': self.bibcode,
            'title': self.title,
            'abstract': self.abstract,
            'search_results': [
                {
                    'bibcode': result.bibcode,
                    'title': result.title,
                    'abstract': result.abstract,
                    'claude_score': result.claude_score,
                    'gemini_score': result.gemini_score,
                    'deepseek_score': result.deepseek_score
                }
                for result in self.search_results
            ],
            'claude_relative_score': self.claude_relative_score,
            'gemini_relative_score': self.gemini_relative_score,
            'deepseek_relative_score': self.deepseek_relative_score
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

def fetch_paper_data(bibcode: str) -> dict:
    """Fetch paper data from ADS API."""
    base_url = "https://api.adsabs.harvard.edu/v1"
    headers = {
        "Authorization": f"Bearer {ads_api_token}",
        "Content-Type": "application/json"
    }
    
    # Fetch paper data
    paper_url = f"{base_url}/search/bigquery?q=bibcode:{quote(bibcode)}&fl=bibcode,title,abstract"
    response = requests.get(paper_url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch paper data: {response.text}")
    
    data = response.json()
    if not data.get("response", {}).get("docs"):
        raise Exception(f"No paper found with bibcode: {bibcode}")
    
    paper = data["response"]["docs"][0]
    return {
        "bibcode": paper["bibcode"],
        "title": paper.get("title", [""])[0],  # ADS returns titles as lists
        "abstract": paper.get("abstract", "")
    }

def create_manual_rankings():
    """Create manually ranked search results file."""
    print("\n=== Creating Manual Rankings File ===")
    
    print("\nStep 1: Reading Bibcodes")
    print("------------------------")
    print("Reading bibcodes from top10_manual.txt...")
    # Read bibcodes from file
    with open(os.path.join(ROOT_DIR, "top10_manual.txt"), "r") as f:
        bibcodes = [line.strip() for line in f if line.strip()]
    
    if not bibcodes:
        raise Exception("No bibcodes found in top10_manual.txt")
    
    print(f"Found {len(bibcodes)} bibcodes")
    
    print("\nStep 2: Fetching Main Paper Data")
    print("--------------------------------")
    # Use the first bibcode as the main paper
    main_paper_bibcode = bibcodes[0]
    print(f"Fetching data for main paper: {main_paper_bibcode}")
    main_paper_data = fetch_paper_data(main_paper_bibcode)
    print(f"Successfully fetched main paper: {main_paper_data['title']}")
    
    print("\nStep 3: Fetching Related Papers")
    print("-------------------------------")
    print(f"Fetching data for {len(bibcodes) - 1} related papers...")
    # Create search results for the remaining papers
    search_results = []
    for i, bibcode in enumerate(bibcodes[1:], 1):
        print(f"Fetching paper {i}/{len(bibcodes) - 1}: {bibcode}")
        paper_data = fetch_paper_data(bibcode)
        search_results.append(SearchResult(
            bibcode=paper_data["bibcode"],
            title=paper_data["title"],
            abstract=paper_data["abstract"]
        ))
    print(f"Successfully fetched {len(search_results)} related papers")
    
    print("\nStep 4: Creating PaperData Object")
    print("--------------------------------")
    print("Creating PaperData object with main paper and search results...")
    # Create PaperData object
    paper_data = PaperData(
        bibcode=main_paper_data["bibcode"],
        title=main_paper_data["title"],
        abstract=main_paper_data["abstract"],
        search_results=search_results
    )
    
    print("\nStep 5: Setting Up Directory Structure")
    print("-------------------------------------")
    print("Creating necessary directories...")
    # Create examples directory if it doesn't exist
    examples_dir = os.path.join(ROOT_DIR, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    print(f"Created/verified examples directory: {examples_dir}")
    
    # Create bibcode directory if it doesn't exist
    bibcode_dir = os.path.join(examples_dir, main_paper_bibcode)
    os.makedirs(bibcode_dir, exist_ok=True)
    print(f"Created/verified bibcode directory: {bibcode_dir}")
    
    # Create inputs directory if it doesn't exist
    inputs_dir = os.path.join(bibcode_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    print(f"Created/verified inputs directory: {inputs_dir}")
    
    print("\nStep 6: Saving Results")
    print("---------------------")
    # Save to JSON file in inputs directory
    output_file = os.path.join(inputs_dir, f"{main_paper_bibcode}_manual.json")
    print(f"Saving results to: {output_file}")
    paper_data.to_json(output_file)
    print("Successfully saved results!")
    
    print("\n=== Process Complete! ===")
    print(f"Created manually ranked search results in {output_file}")
    print(f"Total papers processed: {len(search_results) + 1} (1 main paper + {len(search_results)} related papers)")

if __name__ == "__main__":
    print("\n=== Starting Manual Rankings File Creation ===")
    create_manual_rankings() 