import tomli
from litellm import completion
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import requests
from urllib.parse import quote
import re
from datetime import datetime
from enum import Enum
import glob
import shutil
import concurrent.futures
import time

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the root directory (parent of script directory)
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Read API keys from secrets.toml
with open(os.path.join(ROOT_DIR, "secrets.toml"), "rb") as f:
    secrets = tomli.load(f)

# Get API keys
gcp_api = secrets["api_keys"]["gcp_api"]
ollama_api = secrets["api_keys"]["ollama_api"]
ads_api_token = secrets["api_keys"]["ads_api_token"]
os.environ["AWS_ACCESS_KEY_ID"] = secrets["api_keys"]["bedrock_access_key"]
os.environ["AWS_SECRET_ACCESS_KEY"] = secrets["api_keys"]["bedrock_secret_access_key"]
os.environ["AWS_REGION_NAME"] = "us-east-1"

class AgentRole(Enum):
    CLAUDE = "claude"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"

@dataclass
class SearchResult:
    bibcode: str
    title: str
    abstract: str
    claude_score: int = None
    gemini_score: int = None
    deepseek_score: int = None

@dataclass
class PaperData:
    bibcode: str
    title: str
    abstract: str
    search_results: List[SearchResult]
    claude_relative_score: int = None
    gemini_relative_score: int = None
    deepseek_relative_score: int = None
    
    @classmethod
    def from_json(cls, json_path: str) -> 'PaperData':
        """Load paper data from a JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Convert search results to SearchResult objects
        search_results = [
            SearchResult(
                bibcode=result['bibcode'],
                title=result['title'],
                abstract=result['abstract'],
                claude_score=result.get('claude_score'),
                gemini_score=result.get('gemini_score'),
                deepseek_score=result.get('deepseek_score')
            )
            for result in data.get('search_results', [])
        ]
            
        return cls(
            bibcode=data['bibcode'],
            title=data['title'],
            abstract=data['abstract'],
            search_results=search_results,
            claude_relative_score=data.get('claude_relative_score'),
            gemini_relative_score=data.get('gemini_relative_score'),
            deepseek_relative_score=data.get('deepseek_relative_score')
        )
    
    def save_to_json(self, json_path: str):
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

class EvaluatorAgent:
    def __init__(self, role: AgentRole):
        self.role = role
        self.name = role.value.capitalize()
        self.evaluation = None  # raw text response
        self.scores = None # parsed scores
        
    def get_completion(self, prompt: str) -> Optional[str]:
        """Send a prompt to the agent's LLM via API."""
        try:
            if self.role == AgentRole.CLAUDE:
                response = completion(
                    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            elif self.role == AgentRole.GEMINI:
                response = completion(
                    model="gemini/gemini-2.0-flash",
                    messages=[{"role": "user", "content": prompt}],
                    api_key=gcp_api,
                    api_type="google",
                    max_tokens=4000,
                    temperature=0.7,
                    model_kwargs={
                        "project": "gen-lang-client-0556674802"
                    }
                )
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    return response.choices[0].message.content
                elif hasattr(response, 'text'):
                    return response.text
            elif self.role == AgentRole.DEEPSEEK:
                response = completion(
                    model="bedrock/us.deepseek.r1-v1:0",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting completion from {self.name}: {e}")
            return None
            
    def judgements(self, paper_data: PaperData, new_paper_data: PaperData, raw_outputs_dir: str) -> bool:
        """Evaluate the two sets of search results."""
        print(f"\n[{self.name}] Generating evaluation prompt...")
        
        # Ensure the output directory exists
        os.makedirs(raw_outputs_dir, exist_ok=True)
        
        prompt = self._generate_evaluation_prompt(paper_data, new_paper_data)
        print(f"[{self.name}] Sending prompt to LLM...")
        response = self.get_completion(prompt)
        
        if response is None:
            print(f"[{self.name}] Failed to get response from LLM")
            return False
            
        # Store the raw text response
        self.evaluation = response
        print(f"[{self.name}] Received response from LLM")

        # Save raw response to file
        raw_output_path = os.path.join(raw_outputs_dir, f"{self.name.lower()}_raw_output.txt")
        with open(raw_output_path, 'w') as f:
            f.write(response)
        print(f"[{self.name}] Saved raw response to {raw_output_path}")
            
        return True
        
    def _parse_scores(self, raw_outputs_dir: str, evaluations_dir: str) -> bool:
        """Parse scores from the raw output file."""
        print(f"[{self.name}] Parsing scores from raw response...")
        
        # Ensure the output directory exists
        os.makedirs(evaluations_dir, exist_ok=True)
        
        try:
            # Read the raw response from file
            raw_output_path = os.path.join(raw_outputs_dir, f"{self.name.lower()}_raw_output.txt")
            with open(raw_output_path, 'r') as f:
                raw_response = f.read()

            # Extract json from the response
            json_str = extract_json(raw_response)
            if json_str is None:
                print(f"[{self.name}] Could not find JSON in response")
                return False    
        
            # Clean up the JSON string
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            # Parse the JSON
            evaluation = json.loads(json_str)
            print(f"[{self.name}] Successfully parsed JSON from response")
            
            # Parse scores from the JSON
            self.scores = {
                "first_set": {
                    "individual_scores": [],
                    "overall_score": evaluation["first_set"].get("overall_score"),
                    "ranking_quality": evaluation["first_set"].get("ranking_quality")
                },
                "second_set": {
                    "individual_scores": [],
                    "overall_score": evaluation["second_set"].get("overall_score"),
                    "ranking_quality": evaluation["second_set"].get("ranking_quality")
                },
                "comparison": {
                    "better_set": evaluation["comparison"].get("better_set"),
                    "relative_score": evaluation["comparison"].get("relative_score")
                }
            }
            
            # Parse individual scores
            for set_key in ["first_set", "second_set"]:
                if "individual_scores" in evaluation[set_key]:
                    for result_num, result_data in evaluation[set_key]["individual_scores"].items():
                        score = result_data.get("score")
                        if score is not None and isinstance(score, int) and 0 <= score <= 3:
                            self.scores[set_key]["individual_scores"].append(score)
            
            # Save parsed evaluation
            evaluation_path = os.path.join(evaluations_dir, f"{self.name.lower()}_evaluation.json")
            with open(evaluation_path, 'w') as f:
                json.dump(self.scores, f, indent=2)
            print(f"[{self.name}] Saved parsed evaluation to {evaluation_path}")
                            
            return True
                
        except Exception as e:
            print(f"[{self.name}] Error parsing scores: {e}")
            return False
            
    def _generate_evaluation_prompt(self, paper_data: PaperData, new_paper_data: PaperData) -> str:
        """Generate the evaluation prompt."""
        return f"""You are {self.name}, an expert in evaluating the relevance of scientific papers. Your task is to evaluate two sets of search results against a main paper and provide a detailed assessment.

MAIN PAPER:
Title: {paper_data.title}
Abstract: {paper_data.abstract}

FIRST SET OF RESULTS:
{self._format_search_results(paper_data.search_results)}

SECOND SET OF RESULTS:
{self._format_search_results(new_paper_data.search_results)}

IMPORTANT: You MUST follow these steps in order. Do not skip any steps.

STEP 1: INDEPENDENT ANALYSIS
First, analyze each set independently. For each set:
- How well does each result match the main paper's topic and methodology?
- How relevant are the results to the main paper's research questions?
- How well are the results ordered by relevance?
- What are the strengths and weaknesses of each set?

STEP 2: COMPARISON
Then, compare the two sets:
- Which set better addresses the main paper's research focus?
- Which set has more relevant and high-quality results?
- Which set has better ordering of results by relevance?
- What makes one set better or worse than the other?

STEP 3: FINAL EVALUATION
Provide your evaluation in the following JSON format:

{{
    "first_set": {{
        "individual_scores": {{
            "1": {{"score": <0-3>, "explanation": "<explanation>"}},
            "2": {{"score": <0-3>, "explanation": "<explanation>"}},
            ...
        }},
        "overall_score": <0-3>,
        "ranking_quality": <0-3>
    }},
    "second_set": {{
        "individual_scores": {{
            "1": {{"score": <0-3>, "explanation": "<explanation>"}},
            "2": {{"score": <0-3>, "explanation": "<explanation>"}},
            ...
        }},
        "overall_score": <0-3>,
        "ranking_quality": <0-3>
    }},
    "comparison": {{
        "better_set": "A" or "B",
        "relative_score": <0-3>,
        "justification": "<detailed explanation>"
    }}
}}

SCORING RUBRICS:

CONTENT SIMILARITY (0-3):
0 = Somewhat relevant but tangential
1 = Relevant but not focused on the main topic
2 = Strongly relevant and focused on the main topic
3 = Perfect match in topic and methodology

RANKING QUALITY (0-3):
0 = Some relevant results are near the top
1 = Most relevant results are near the top
2 = Results are well-ordered by relevance
3 = Results are perfectly ordered by relevance

RELATIVE SCORE (0-3):
0 = Both sets are equally good
1 = One set is slightly better
2 = One set is moderately better
3 = One set is much better

Rules:
1. You MUST complete steps 1 and 2 before providing your evaluation
2. Be thorough in your analysis and justification
3. Use better_set: "A" if the first set is better, "B" if the second set is better
4. Ensure all scores are integers between 0 and 3
5. Provide detailed explanations for each score
6. The JSON response must be valid and complete
"""
    
    def _format_search_results(self, results: List[SearchResult]) -> str:
        """Format search results for display in the prompt."""
        formatted = ""
        for i, result in enumerate(results, 1):
            formatted += f"\nResult {i}:\n"
            formatted += f"Title: {result.title}\n"
            formatted += f"Abstract: {result.abstract}\n"
        return formatted

class ConsensusEvaluator:
    def __init__(self):
        self.agents = {
            AgentRole.CLAUDE: EvaluatorAgent(AgentRole.CLAUDE),
            AgentRole.GEMINI: EvaluatorAgent(AgentRole.GEMINI),
            AgentRole.DEEPSEEK: EvaluatorAgent(AgentRole.DEEPSEEK)
        }
        self.consensus = None
        
    def evaluate(self, paper_data: PaperData, new_paper_data: PaperData, raw_outputs_dir: str, evaluations_dir: str, consensus_dir: str) -> bool:
        """Have all agents evaluate the results and reach a consensus."""
        print("\n=== Starting Evaluation Process ===")
        
        # Ensure all output directories exist
        os.makedirs(raw_outputs_dir, exist_ok=True)
        os.makedirs(evaluations_dir, exist_ok=True)
        os.makedirs(consensus_dir, exist_ok=True)
        
        # Step 1: Process each agent in parallel
        print("\n=== Processing Agents in Parallel ===")
        
        # Create a function to process a single agent
        def process_agent(agent):
            print(f"\n=== Processing {agent.name} ===")
            
            # Generate evaluation prompt
            prompt = agent._generate_evaluation_prompt(paper_data, new_paper_data)
            print(f"[{agent.name}] Sending prompt to LLM...")
            
            # Get response from LLM
            response = agent.get_completion(prompt)
            
            if response is None:
                print(f"[{agent.name}] Failed to get response from LLM")
                return False
                
            # Store the raw response
            agent.evaluation = response
            print(f"[{agent.name}] Received response from LLM")
            
            # Save raw response to individual file
            raw_output_path = os.path.join(raw_outputs_dir, f"{agent.name.lower()}_raw_output.txt")
            with open(raw_output_path, 'w') as f:
                f.write(response)
            print(f"[{agent.name}] Saved raw response to {raw_output_path}")
            
            # Parse scores from the response
            if not agent._parse_scores(raw_outputs_dir, evaluations_dir):
                print(f"[{agent.name}] Failed to parse scores from response")
                return False
                
            return True
        
        # Process agents in parallel
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all agent processing tasks
            future_to_agent = {executor.submit(process_agent, agent): agent for agent in self.agents.values()}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    success = future.result()
                    if not success:
                        print(f"[{agent.name}] Processing failed")
                        return False
                except Exception as e:
                    print(f"[{agent.name}] Processing generated an exception: {e}")
                    return False
        
        end_time = time.time()
        print(f"\n=== All agents processed in {end_time - start_time:.2f} seconds ===")
                
        # Step 3: Generate consensus using Claude
        print("\n=== Generating Consensus ===")
        consensus_prompt = self._generate_consensus_prompt()
        print("[Claude] Sending consensus prompt to LLM...")
        consensus_response = self.agents[AgentRole.CLAUDE].get_completion(consensus_prompt)
        
        if consensus_response is None:
            print("[Claude] Failed to get consensus response from LLM")
            return False
            
        try:
            print("[Claude] Extracting JSON from consensus response...")
            # Extract JSON from consensus response
            json_match = re.search(r'```json\s*(\{[\s\S]*\})\s*```', consensus_response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{[\s\S]*\}', consensus_response)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    print("[Claude] Could not find JSON in consensus response")
                    return False
                    
            # Clean up and parse the JSON
            json_str = json_str.strip()
            consensus_data = json.loads(json_str)
            print("[Claude] Successfully parsed JSON from consensus response")
            
            # Validate the consensus data structure
            print("[Claude] Validating consensus data structure...")
            if "consensus" not in consensus_data:
                print("[Claude] Error: 'consensus' field missing in response")
                print(f"[Claude] Available fields: {list(consensus_data.keys())}")
                return False
                
            consensus = consensus_data["consensus"]
            if "comparison" not in consensus:
                print("[Claude] Error: 'comparison' field missing in consensus")
                print(f"[Claude] Available fields in consensus: {list(consensus.keys())}")
                return False
                
            if "better_set" not in consensus["comparison"]:
                print("[Claude] Error: 'better_set' field missing in comparison")
                print(f"[Claude] Available fields in comparison: {list(consensus['comparison'].keys())}")
                return False
                
            # Store the validated consensus
            self.consensus = consensus
            print("[Claude] Consensus data structure validated successfully")
            
            # Save consensus evaluation
            consensus_path = os.path.join(consensus_dir, "consensus_evaluation.json")
            with open(consensus_path, 'w') as f:
                json.dump(self.consensus, f, indent=2)
            print(f"[Claude] Saved consensus evaluation to {consensus_path}")
                
            # Save consensus summary
            summary_path = os.path.join(consensus_dir, "consensus_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Consensus Evaluation Summary\n")
                f.write(f"==========================\n\n")
                f.write(f"Better Set: {self.consensus['comparison']['better_set']}\n")
                f.write(f"Relative Score: {self.consensus['comparison']['relative_score']}\n\n")
                f.write(f"Justification:\n{self.consensus['comparison']['justification']}\n")
            print(f"[Claude] Saved consensus summary to {summary_path}")
            
            # Update PaperData objects with consensus scores
            print("\n=== Updating PaperData Objects with Scores ===")
            self._update_paper_data_scores(paper_data, new_paper_data)
            print("PaperData objects updated with scores")

            # Save final results
            self.save_final_results(paper_data, new_paper_data, consensus_dir)
            print("Final results saved")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"[Claude] Error parsing consensus evaluation as JSON: {e}")
            print(f"[Claude] JSON string: {json_str}")
            return False
        except KeyError as e:
            print(f"[Claude] Error accessing field in consensus data: {e}")
            print(f"[Claude] Consensus data: {json.dumps(consensus_data, indent=2)}")
            return False
        except Exception as e:
            print(f"[Claude] Error in consensus evaluation: {e}")
            print(f"[Claude] Error type: {type(e).__name__}")
            import traceback
            print(f"[Claude] Traceback: {traceback.format_exc()}")
            return False
    
    def _update_paper_data_scores(self, paper_data: PaperData, new_paper_data: PaperData):
        """Update PaperData objects with consensus scores."""
        print("Updating individual agent scores...")
        # Update individual agent scores
        paper_data.claude_relative_score = self.agents[AgentRole.CLAUDE].scores["comparison"]["relative_score"]
        paper_data.gemini_relative_score = self.agents[AgentRole.GEMINI].scores["comparison"]["relative_score"]
        paper_data.deepseek_relative_score = self.agents[AgentRole.DEEPSEEK].scores["comparison"]["relative_score"]
        
        new_paper_data.claude_relative_score = self.agents[AgentRole.CLAUDE].scores["comparison"]["relative_score"]
        new_paper_data.gemini_relative_score = self.agents[AgentRole.GEMINI].scores["comparison"]["relative_score"]
        new_paper_data.deepseek_relative_score = self.agents[AgentRole.DEEPSEEK].scores["comparison"]["relative_score"]
        
        print("Updating search result scores...")
        # Update search result scores
        for i, result in enumerate(paper_data.search_results):
            if i < len(self.agents[AgentRole.CLAUDE].scores["first_set"]["individual_scores"]):
                result.claude_score = self.agents[AgentRole.CLAUDE].scores["first_set"]["individual_scores"][i]
            if i < len(self.agents[AgentRole.GEMINI].scores["first_set"]["individual_scores"]):
                result.gemini_score = self.agents[AgentRole.GEMINI].scores["first_set"]["individual_scores"][i]
            if i < len(self.agents[AgentRole.DEEPSEEK].scores["first_set"]["individual_scores"]):
                result.deepseek_score = self.agents[AgentRole.DEEPSEEK].scores["first_set"]["individual_scores"][i]
        
        for i, result in enumerate(new_paper_data.search_results):
            if i < len(self.agents[AgentRole.CLAUDE].scores["second_set"]["individual_scores"]):
                result.claude_score = self.agents[AgentRole.CLAUDE].scores["second_set"]["individual_scores"][i]
            if i < len(self.agents[AgentRole.GEMINI].scores["second_set"]["individual_scores"]):
                result.gemini_score = self.agents[AgentRole.GEMINI].scores["second_set"]["individual_scores"][i]
            if i < len(self.agents[AgentRole.DEEPSEEK].scores["second_set"]["individual_scores"]):
                result.deepseek_score = self.agents[AgentRole.DEEPSEEK].scores["second_set"]["individual_scores"][i]
    
    def save_final_results(self, paper_data: PaperData, new_paper_data: PaperData, consensus_dir: str):
        """Save the final PaperData objects with all LLM scores."""
        print("\n=== Saving Final Results ===")
        
        # Ensure the consensus directory exists
        os.makedirs(consensus_dir, exist_ok=True)
        
        # Create a dictionary with both PaperData objects and input information
        final_results = {
            "input_paper": {
                "bibcode": paper_data.bibcode,
                "title": paper_data.title,
                "abstract": paper_data.abstract
            },
            "set_a": {
                "bibcode": paper_data.bibcode,
                "title": paper_data.title,
                "abstract": paper_data.abstract,
                "claude_relative_score": paper_data.claude_relative_score,
                "gemini_relative_score": paper_data.gemini_relative_score,
                "deepseek_relative_score": paper_data.deepseek_relative_score,
                "search_results": [
                    {
                        "bibcode": result.bibcode,
                        "title": result.title,
                        "abstract": result.abstract,
                        "claude_score": result.claude_score,
                        "gemini_score": result.gemini_score,
                        "deepseek_score": result.deepseek_score
                    }
                    for result in paper_data.search_results
                ]
            },
            "set_b": {
                "bibcode": new_paper_data.bibcode,
                "title": new_paper_data.title,
                "abstract": new_paper_data.abstract,
                "claude_relative_score": new_paper_data.claude_relative_score,
                "gemini_relative_score": new_paper_data.gemini_relative_score,
                "deepseek_relative_score": new_paper_data.deepseek_relative_score,
                "search_results": [
                    {
                        "bibcode": result.bibcode,
                        "title": result.title,
                        "abstract": result.abstract,
                        "claude_score": result.claude_score,
                        "gemini_score": result.gemini_score,
                        "deepseek_score": result.deepseek_score
                    }
                    for result in new_paper_data.search_results
                ]
            },
            "consensus": self.consensus,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        final_results_path = os.path.join(consensus_dir, "final_results.json")
        with open(final_results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"Saved final results to {final_results_path}")
            
        # Also save individual PaperData objects
        set_a_path = os.path.join(consensus_dir, "set_a_final.json")
        set_b_path = os.path.join(consensus_dir, "set_b_final.json")
        paper_data.save_to_json(set_a_path)
        new_paper_data.save_to_json(set_b_path)
        print(f"Saved Set A data to {set_a_path}")
        print(f"Saved Set B data to {set_b_path}")

    def _generate_consensus_prompt(self) -> str:
        """Generate the consensus prompt."""
        prompt = """You are Claude, tasked with reaching a consensus between three expert evaluators (Claude, Gemini, and DeepSeek) who have independently evaluated two sets of search results.

Here are their individual evaluations:

"""
        
        for agent in self.agents.values():
            prompt += f"\n{agent.name}'s Evaluation:\n"
            prompt += json.dumps(agent.scores, indent=2)
            prompt += "\n"
            
        prompt += """
Please analyze these evaluations and provide a consensus evaluation that:
1. Takes into account all three perspectives
2. Resolves any disagreements
3. Provides a clear, justified final assessment

Provide your consensus evaluation in the following JSON format:

{
    "consensus": {
        "first_set": {
            "individual_scores": {
                "1": {"score": <0-3>, "explanation": "<explanation>"},
                "2": {"score": <0-3>, "explanation": "<explanation>"},
                ...
            },
            "overall_score": <0-3>,
            "ranking_quality": <0-3>
        },
        "second_set": {
            "individual_scores": {
                "1": {"score": <0-3>, "explanation": "<explanation>"},
                "2": {"score": <0-3>, "explanation": "<explanation>"},
                ...
            },
            "overall_score": <0-3>,
            "ranking_quality": <0-3>
        },
        "comparison": {
            "better_set": "A" or "B",
            "relative_score": <0-3>,
            "justification": "<detailed explanation of how the consensus was reached>"
        }
    },
    "discussion": {
        "agreements": ["<list of points all evaluators agreed on>"],
        "disagreements": ["<list of points where evaluators differed>"],
        "resolution": "<explanation of how disagreements were resolved>"
    }
}

IMPORTANT: Your response MUST include the "consensus" field with a "comparison" subfield that contains a "better_set" field with either "A" or "B" as the value.

Rules:
1. Consider all three evaluations carefully
2. Explain how you resolved any disagreements
3. Provide detailed justifications for the consensus scores
4. Ensure all scores are integers between 0 and 3
5. The JSON response must be valid and complete
6. You MUST include the "better_set" field in the "comparison" section
"""
        return prompt

def extract_json(text: str) -> Optional[str]:
    """Extract JSON object from text using multiple strategies."""
    # Try to find JSON in code blocks
    json_match = re.search(r'```json\s*(\{[\s\S]*\})\s*```', text)
    if json_match:
        return json_match.group(1)
        
    # Try to find JSON between curly braces
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        return json_match.group(0)
        
    # Try to find first { and last }
    start = text.find('{')
    end = text.rfind('}') + 1
    if start != -1 and end != 0:
        return text[start:end]
        
    return None

def create_bibcode_directories(bibcode: str) -> Tuple[str, str, str]:
    """
    Create directory structure for a bibcode.
    
    Args:
        bibcode: The bibcode to create directories for
        
    Returns:
        A tuple containing (base_dir, inputs_dir, outputs_dir)
    """
    print(f"\nCreating directory structure for bibcode: {bibcode}")
    
    # Create base directory for bibcode in examples directory
    base_dir = os.path.join(ROOT_DIR, "examples", bibcode)
    inputs_dir = os.path.join(base_dir, "inputs")
    outputs_dir = os.path.join(base_dir, "outputs")
    
    # Create directories if they don't exist
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    print(f"  - Base directory: {base_dir}")
    print(f"  - Inputs directory: {inputs_dir}")
    print(f"  - Outputs directory: {outputs_dir}")
    
    return base_dir, inputs_dir, outputs_dir

def create_comparison_directories(bibcode: str, file1: str, file2: str) -> Tuple[str, str, str, str]:
    """
    Create directory structure for a specific comparison.
    
    Args:
        bibcode: The bibcode being compared
        file1: First file being compared
        file2: Second file being compared
        
    Returns:
        A tuple containing (comparison_dir, raw_outputs_dir, evaluations_dir, consensus_dir)
    """
    # Get base names without extension and bibcode prefix
    file1_base = os.path.splitext(os.path.basename(file1))[0].replace(bibcode, "").strip("_")
    file2_base = os.path.splitext(os.path.basename(file2))[0].replace(bibcode, "").strip("_")
    
    # Create comparison directory name
    comparison_name = f"{bibcode}_{file1_base}_{file2_base}"
    
    print(f"\nCreating comparison directory: {comparison_name}")
    
    # Create directory structure
    base_dir, inputs_dir, outputs_dir = create_bibcode_directories(bibcode)
    comparison_dir = os.path.join(outputs_dir, comparison_name)
    raw_outputs_dir = os.path.join(comparison_dir, "raw_outputs")
    evaluations_dir = os.path.join(comparison_dir, "evaluations")
    consensus_dir = os.path.join(comparison_dir, "consensus")
    
    # Create directories if they don't exist
    os.makedirs(raw_outputs_dir, exist_ok=True)
    os.makedirs(evaluations_dir, exist_ok=True)
    os.makedirs(consensus_dir, exist_ok=True)
    
    print(f"  - Comparison directory: {comparison_dir}")
    print(f"  - Raw outputs directory: {raw_outputs_dir}")
    print(f"  - Evaluations directory: {evaluations_dir}")
    print(f"  - Consensus directory: {consensus_dir}")
    
    return comparison_dir, raw_outputs_dir, evaluations_dir, consensus_dir

def select_files_to_compare(bibcode: str) -> Tuple[PaperData, PaperData, str, str, str, str]:
    """
    Allow the user to select which two files to compare and which one should be set A and set B.
    
    Args:
        bibcode: The bibcode to use for finding files
        
    Returns:
        A tuple containing (paper_data, new_paper_data, comparison_dir, raw_outputs_dir, evaluations_dir, consensus_dir)
    """
    print("\n=== File Selection ===")
    
    # Create directory structure
    base_dir, inputs_dir, outputs_dir = create_bibcode_directories(bibcode)
    
    # Get all JSON files in the inputs directory
    all_files = glob.glob(os.path.join(inputs_dir, "*.json"))
    
    if len(all_files) < 2:
        print(f"Error: Not enough files found for bibcode {bibcode}")
        print(f"Found files: {all_files}")
        print(f"Please ensure at least two files exist in the {inputs_dir} directory")
        exit(1)
    
    # Display available files
    print("\nAvailable files:")
    for i, file in enumerate(all_files, 1):
        print(f"{i}. {os.path.basename(file)}")
    
    # Ask user to select two files
    while True:
        try:
            print("\nSelect two files to compare (enter numbers separated by space):")
            selections = input("> ").strip().split()
            
            if len(selections) != 2:
                print("Please enter exactly two numbers.")
                continue
                
            file1_idx = int(selections[0]) - 1
            file2_idx = int(selections[1]) - 1
            
            if file1_idx < 0 or file1_idx >= len(all_files) or file2_idx < 0 or file2_idx >= len(all_files):
                print("Invalid selection. Please enter numbers from the list above.")
                continue
                
            if file1_idx == file2_idx:
                print("Please select two different files.")
                continue
                
            file1 = all_files[file1_idx]
            file2 = all_files[file2_idx]
            break
        except ValueError:
            print("Please enter valid numbers.")
    
    # Ask user which file should be set A and which should be set B
    print(f"\nWhich file should be Set A (first set) and which should be Set B (second set)?")
    print(f"1. {os.path.basename(file1)} as Set A, {os.path.basename(file2)} as Set B")
    print(f"2. {os.path.basename(file2)} as Set A, {os.path.basename(file1)} as Set B")
    
    while True:
        try:
            choice = int(input("Enter 1 or 2: "))
            if choice == 1:
                set_a_file = file1
                set_b_file = file2
                break
            elif choice == 2:
                set_a_file = file2
                set_b_file = file1
                break
            else:
                print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nSelected files:")
    print(f"Set A (first set): {os.path.basename(set_a_file)}")
    print(f"Set B (second set): {os.path.basename(set_b_file)}")
    
    # Create comparison directories
    comparison_dir, raw_outputs_dir, evaluations_dir, consensus_dir = create_comparison_directories(bibcode, set_a_file, set_b_file)
    
    # Load the selected files
    print("\nLoading selected files...")
    paper_data = PaperData.from_json(set_a_file)
    new_paper_data = PaperData.from_json(set_b_file)
    print(f"Loaded Set A: {paper_data.bibcode} - {paper_data.title}")
    print(f"Loaded Set B: {new_paper_data.bibcode} - {new_paper_data.title}")
    
    return paper_data, new_paper_data, comparison_dir, raw_outputs_dir, evaluations_dir, consensus_dir

def list_available_bibcodes() -> str:
    """
    List available bibcodes in the examples directory and let user select one.
    
    Returns:
        The selected bibcode
    """
    print("\n=== Available Bibcodes ===")
    
    # Get the examples directory path
    examples_dir = os.path.join(ROOT_DIR, "examples")
    
    # Check if examples directory exists
    if not os.path.exists(examples_dir):
        print(f"Error: Examples directory not found at {examples_dir}")
        print("Please run the input file creation scripts first.")
        exit(1)
    
    # Get all directories in examples
    bibcodes = [d for d in os.listdir(examples_dir) if os.path.isdir(os.path.join(examples_dir, d))]
    
    if not bibcodes:
        print("No bibcode directories found in examples directory.")
        print("Please run the input file creation scripts first.")
        exit(1)
    
    # Display available bibcodes
    print("\nAvailable bibcodes:")
    for i, bibcode in enumerate(bibcodes, 1):
        print(f"{i}. {bibcode}")
    
    # Ask user to select a bibcode
    while True:
        try:
            print("\nSelect a bibcode to analyze (enter number):")
            selection = input("> ").strip()
            
            if not selection:
                print("Please enter a number.")
                continue
                
            idx = int(selection) - 1
            
            if idx < 0 or idx >= len(bibcodes):
                print("Invalid selection. Please enter a number from the list above.")
                continue
                
            selected_bibcode = bibcodes[idx]
            print(f"\nSelected bibcode: {selected_bibcode}")
            return selected_bibcode
            
        except ValueError:
            print("Please enter a valid number.")

def main():
    print("\n=== LLM Comparator ===")
    print("This tool compares two sets of search results using multiple LLMs")
    print("\nInitializing evaluator...")
    
    # Let user select which bibcode to analyze
    test_bibcode = list_available_bibcodes()
    
    print("\nStep 1: File Selection")
    print("----------------------")
    # Allow user to select which files to compare
    paper_data, new_paper_data, comparison_dir, raw_outputs_dir, evaluations_dir, consensus_dir = select_files_to_compare(test_bibcode)
    
    print("\nStep 2: Creating Evaluator Agents")
    print("--------------------------------")
    # Create evaluator agents
    print("Initializing Claude agent...")
    claude_agent = EvaluatorAgent(AgentRole.CLAUDE)
    print("Initializing Gemini agent...")
    gemini_agent = EvaluatorAgent(AgentRole.GEMINI)
    print("Initializing DeepSeek agent...")
    deepseek_agent = EvaluatorAgent(AgentRole.DEEPSEEK)
    
    print("\nStep 3: Running Parallel Evaluations")
    print("-----------------------------------")
    print("Starting parallel evaluation with all three agents...")
    # Run evaluations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        print("Submitting evaluation tasks to agents...")
        future_claude = executor.submit(claude_agent.judgements, paper_data, new_paper_data, raw_outputs_dir)
        future_gemini = executor.submit(gemini_agent.judgements, paper_data, new_paper_data, raw_outputs_dir)
        future_deepseek = executor.submit(deepseek_agent.judgements, paper_data, new_paper_data, raw_outputs_dir)
        
        print("Waiting for all evaluations to complete...")
        # Wait for all evaluations to complete
        concurrent.futures.wait([future_claude, future_gemini, future_deepseek])
        print("All evaluations completed successfully")
    
    print("\nStep 4: Parsing Evaluation Scores")
    print("--------------------------------")
    print("Parsing scores from Claude's evaluation...")
    claude_agent._parse_scores(raw_outputs_dir, evaluations_dir)
    print("Parsing scores from Gemini's evaluation...")
    gemini_agent._parse_scores(raw_outputs_dir, evaluations_dir)
    print("Parsing scores from DeepSeek's evaluation...")
    deepseek_agent._parse_scores(raw_outputs_dir, evaluations_dir)
    
    print("\nStep 5: Generating Consensus")
    print("---------------------------")
    print("Creating consensus evaluator...")
    consensus_evaluator = ConsensusEvaluator()
    print("Generating consensus from all evaluations...")
    consensus_evaluator.evaluate(paper_data, new_paper_data, raw_outputs_dir, evaluations_dir, consensus_dir)
    
    print("\n=== Evaluation Complete! ===")
    print(f"Results have been saved in: {consensus_dir}")
    print("\nYou can find the following files:")
    print(f"- Raw outputs: {raw_outputs_dir}")
    print(f"- Individual evaluations: {evaluations_dir}")
    print(f"- Consensus results: {consensus_dir}")
    print("\nThank you for using the LLM Comparator!")

if __name__ == "__main__":
    main() 