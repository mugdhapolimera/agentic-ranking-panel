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

def fetch_operator_papers(bibcode: str, operator: str = "similar") -> List[dict]:
    """Fetch useful papers from ADS API."""
    base_url = "https://api.adsabs.harvard.edu/v1"
    headers = {
        "Authorization": f"Bearer {ads_api_token}",
        "Content-Type": "application/json"
    }
    
    # Fetch useful papers
    useful_url = f"{base_url}/search/bigquery?q={quote(operator)}(bibcode:{quote(bibcode)})&fl=bibcode,title,abstract&rows=10"
    response = requests.get(useful_url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch useful papers: {response.text}")
    
    data = response.json()
    papers = data.get("response", {}).get("docs", [])
    
    return [
        {
            "bibcode": paper["bibcode"],
            "title": paper.get("title", [""])[0],
            "abstract": paper.get("abstract", "")
        }
        for paper in papers
    ]

def create_papers_file(bibcode: str, operator: str = "similar"):
    """Create a file with useful papers in the same format as the example JSON."""
    print(f"\n=== Creating Papers File for {bibcode} ===")
    print(f"Using operator: {operator}")
    
    print("\nStep 1: Fetching Main Paper Data")
    print("--------------------------------")
    print(f"Fetching data for paper: {bibcode}")
    # Fetch main paper data
    main_paper_data = fetch_paper_data(bibcode)
    print(f"Successfully fetched main paper: {main_paper_data['title']}")
    
    print("\nStep 2: Fetching Related Papers")
    print("-------------------------------")
    print(f"Fetching papers related to {bibcode} using {operator} operator...")
    # Fetch useful papers
    useful_papers = fetch_operator_papers(bibcode, operator)
    print(f"Found {len(useful_papers)} related papers")
    
    print("\nStep 3: Creating Search Results")
    print("-------------------------------")
    print("Converting paper data to search results...")
    # Create search results
    search_results = [
        SearchResult(
            bibcode=paper["bibcode"],
            title=paper["title"],
            abstract=paper["abstract"]
        )
        for paper in useful_papers
    ]
    print(f"Created {len(search_results)} search results")
    
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
    bibcode_dir = os.path.join(examples_dir, bibcode)
    os.makedirs(bibcode_dir, exist_ok=True)
    print(f"Created/verified bibcode directory: {bibcode_dir}")
    
    # Create inputs directory if it doesn't exist
    inputs_dir = os.path.join(bibcode_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    print(f"Created/verified inputs directory: {inputs_dir}")
    
    print("\nStep 6: Saving Results")
    print("---------------------")
    # Save to JSON file in inputs directory
    output_file = os.path.join(inputs_dir, f"{bibcode}_{operator}.json")
    print(f"Saving results to: {output_file}")
    paper_data.to_json(output_file)
    print("Successfully saved results!")
    
    print("\n=== Process Complete! ===")
    print(f"Created useful papers file: {output_file}")
    print(f"Total papers processed: {len(search_results) + 1} (1 main paper + {len(search_results)} related papers)")

if __name__ == "__main__":
    print("\n=== Starting Papers File Creation ===")
    # Test fetching useful papers
    test_bibcode = "2022ApJ...931...44P"

    bibcodes = [
        "2022ApJ...931...44P",
        "2015MNRAS.454.3722S",
        "2025arXiv250507749R",
        "2023PhDT........28P",
        "2022ApJ...927..165R",
        "2020AAS...23520720P",
        "2013ApJ...775..116R",
        "2013ApJ...774..100K",

    ]
    for bibcode in bibcodes:
        create_papers_file(bibcode, operator="similar") 