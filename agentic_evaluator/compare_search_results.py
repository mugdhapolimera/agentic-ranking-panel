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
import importlib
import argparse

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
        self.evaluation_history = []  # Store past evaluations
        self.performance_metrics = {}  # Track evaluation quality
        
    def get_completion(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Send a prompt to the agent's LLM via API with retry logic."""
        for attempt in range(max_retries):
            try:
                if self.role == AgentRole.CLAUDE:
                    response = completion(
                        model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=4000,
                        temperature=0.7
                    )
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
                elif self.role == AgentRole.DEEPSEEK:
                    response = completion(
                        model="bedrock/us.deepseek.r1-v1:0",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=4000,
                        temperature=0.7
                    )
                else:
                    print(f"[{self.name}] Unknown agent role: {self.role}")
                    return None
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    return response.choices[0].message.content
                elif hasattr(response, 'text'):
                    return response.text
                else:
                    print(f"[{self.name}] Unexpected response format from {self.name}")
                    print(f"[{self.name}] Response attributes: {dir(response)}")
                    if attempt < max_retries - 1:
                        print(f"[{self.name}] Retrying... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2)
                        continue
                    return None
            except Exception as e:
                print(f"[{self.name}] Error getting completion (attempt {attempt + 1}/{max_retries}): {str(e)}")
                import traceback
                print(f"[{self.name}] Traceback: {traceback.format_exc()}")
                # If it's a Bedrock-specific error, try a different model
                if "bedrock" in str(e).lower() and self.role == AgentRole.DEEPSEEK:
                    print(f"[{self.name}] Switching to alternative DeepSeek model...")
                    try:
                        response = completion(
                            model="deepseek/deepseek-coder-33b-instruct",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=4000,
                            temperature=0.7
                        )
                        if hasattr(response, 'choices') and len(response.choices) > 0:
                            return response.choices[0].message.content
                        elif hasattr(response, 'text'):
                            return response.text
                    except Exception as fallback_error:
                        print(f"[{self.name}] Fallback model also failed: {str(fallback_error)}")
                # For Gemini, after all retries, print a warning and continue
                if self.role == AgentRole.GEMINI and attempt == max_retries - 1:
                    print(f"[{self.name}] WARNING: Gemini failed after {max_retries} attempts. Skipping.")
                    return None
                if attempt < max_retries - 1:
                    print(f"[{self.name}] Retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2)
                    continue
                return None
        print(f"[{self.name}] Failed to get completion after {max_retries} attempts")
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

    def evaluate_individual_results(self, paper_data: PaperData, raw_outputs_dir: str) -> bool:
        """Evaluate individual results in a set of search results."""
        print(f"\n[{self.name}] Generating individual evaluation prompt...")
        
        # Ensure the output directory exists
        os.makedirs(raw_outputs_dir, exist_ok=True)
        
        prompt = self._generate_individual_evaluation_prompt(paper_data)
        print(f"[{self.name}] Sending prompt to LLM...")
        response = self.get_completion(prompt)
        
        if response is None:
            print(f"[{self.name}] Failed to get response from LLM")
            return False
            
        # Store the raw text response
        self.evaluation = response
        print(f"[{self.name}] Received response from LLM")

        # Save raw response to file
        raw_output_path = os.path.join(raw_outputs_dir, f"{self.name.lower()}_individual_evaluation.txt")
        with open(raw_output_path, 'w') as f:
            f.write(response)
        print(f"[{self.name}] Saved raw response to {raw_output_path}")
            
        return True

    def compare_multiple_sets(self, paper_data: PaperData, result_sets: List[PaperData], raw_outputs_dir: str) -> bool:
        """Compare multiple sets of search results."""
        print(f"\n[{self.name}] Generating multi-set comparison prompt...")
        
        # Ensure the output directory exists
        os.makedirs(raw_outputs_dir, exist_ok=True)
        
        prompt = self._generate_multi_set_comparison_prompt(paper_data, result_sets)
        print(f"[{self.name}] Sending prompt to LLM...")
        response = self.get_completion(prompt)
        
        if response is None:
            print(f"[{self.name}] Failed to get response from LLM")
            return False
            
        # Store the raw text response
        self.evaluation = response
        print(f"[{self.name}] Received response from LLM")

        # Save raw response to file
        raw_output_path = os.path.join(raw_outputs_dir, f"{self.name.lower()}_multi_set_comparison.txt")
        with open(raw_output_path, 'w') as f:
            f.write(response)
        print(f"[{self.name}] Saved raw response to {raw_output_path}")
            
        return True

    @staticmethod
    def load_few_shot_examples(max_tokens: int = 24000) -> str:
        import json
        import os
        example_files = [
            os.path.join(ROOT_DIR, "examples/2021GSR.....9....1J/inputs/2021GSR.....9....1J_similar_judgements.json"),
            os.path.join(ROOT_DIR, "examples/2022ApJ...931...44P/inputs/2022ApJ...931...44P_similar_judgements.json"),
            os.path.join(ROOT_DIR, "examples/2023arXiv231214211B/inputs/2023arXiv231214211B_similar_judgements.json"),
            os.path.join(ROOT_DIR, "examples/2019MNRAS.486.2075B/inputs/2019MNRAS.486.2075B_similar_judgements.json"),
        ]
        examples = []
        total_tokens = 0
        # Try to use tiktoken for accurate token counting, else fallback to word count
        try:
            tiktoken = importlib.import_module('tiktoken')
            enc = tiktoken.get_encoding('cl100k_base')
            def count_tokens(text):
                return len(enc.encode(text))
        except Exception:
            def count_tokens(text):
                return len(text.split()) // 0.75  # rough estimate: 0.75 words/token
        for file in example_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                main_title = data['title'] if isinstance(data['title'], str) else data['title'][0]
                main_abstract = data['abstract']
                for result in data['search_results']:
                    res_title = result['title'] if isinstance(result['title'], str) else result['title'][0]
                    res_abstract = result['abstract']
                    score = result.get('sme_judgements')
                    note = result.get('notes', '')
                    if score is not None:
                        try:
                            score_fmt = int(round(float(score)))
                        except Exception:
                            score_fmt = score
                        example = f"""
EXAMPLE:
Main Paper: "{main_title}"
Abstract: "{main_abstract}"

Result to Evaluate:
Title: "{res_title}"
Abstract: "{res_abstract}"

Evaluation:
Score: {score_fmt}
Explanation: "{note}"
"""
                        example = example.strip()
                        tokens = count_tokens(example)
                        if total_tokens + tokens > max_tokens:
                            return '\n\n'.join(examples)
                        examples.append(example)
                        total_tokens += tokens
            except Exception as e:
                print(f"[Few-shot] Could not load example from {file}: {e}")
        return '\n\n'.join(examples)

    def _generate_individual_evaluation_prompt(self, paper_data: PaperData) -> str:
        """Generate the prompt for evaluating individual results, with real few-shot examples."""
        few_shot_examples = self.load_few_shot_examples()
        return f"""You are {self.name}, an expert in evaluating the relevance of scientific papers. Your task is to evaluate each result in a set of search results against a main paper and provide a detailed assessment.

FEW-SHOT EXAMPLES:
{few_shot_examples}

MAIN PAPER:
Title: {paper_data.title}
Abstract: {paper_data.abstract}

SEARCH RESULTS:
{self._format_search_results(paper_data.search_results)}

IMPORTANT: You MUST evaluate each result independently and provide a detailed explanation for your score.

Provide your evaluation in the following JSON format:

{{
    "individual_scores": {{
        "1": {{"score": <0-3>, "explanation": "<detailed explanation>"}},
        "2": {{"score": <0-3>, "explanation": "<detailed explanation>"}},
        ...
    }},
    "overall_analysis": {{
        "strengths": ["<list of strengths in the results>"],
        "weaknesses": ["<list of weaknesses in the results>"],
        "suggestions": ["<list of suggestions for improvement>"]
    }}
}}

SCORING RUBRIC (0-3):
0 = Not relevant to the main paper's topic or methodology
1 = Somewhat relevant but tangential to the main paper
2 = Relevant and focused on the main paper's topic
3 = Highly relevant and closely aligned with the main paper's topic and methodology

Rules:
1. Evaluate each result independently
2. Provide detailed explanations for each score
3. Consider both topic relevance and methodological alignment
4. Ensure all scores are integers between 0 and 3
5. The JSON response must be valid and complete
"""

    def _generate_multi_set_comparison_prompt(self, paper_data: PaperData, result_sets: List[PaperData]) -> str:
        """Generate the prompt for comparing multiple sets of results."""
        prompt = f"""You are {self.name}, an expert in evaluating the relevance of scientific papers. Your task is to compare multiple sets of search results against a main paper and determine which set is best.

MAIN PAPER:
Title: {paper_data.title}
Abstract: {paper_data.abstract}

"""
        
        # Add each set of results
        for i, result_set in enumerate(result_sets, 1):
            prompt += f"\nSET {i}:\n"
            prompt += self._format_search_results(result_set.search_results)
            
        prompt += """
IMPORTANT: You MUST analyze each set independently and then compare them to determine which is best.

Provide your evaluation in the following JSON format:

{
    "set_analyses": {
        "1": {
            "strengths": ["<list of strengths>"],
            "weaknesses": ["<list of weaknesses>"],
            "summary": "<brief summary of this set's quality and relevance>"
        },
        "2": {
            "strengths": ["<list of strengths>"],
            "weaknesses": ["<list of weaknesses>"],
            "summary": "<brief summary of this set's quality and relevance>"
        },
        ...
    },
    "comparison": {
        "best_set": <set number>,
        "justification": "<detailed explanation of why this set is best, including specific examples from the results>",
        "key_differences": ["<list of key differences between the sets>"],
        "improvement_suggestions": ["<list of suggestions for improving the other sets>"]
    }
}

Rules:
1. Analyze each set independently first
2. Compare the sets based on:
   - Relevance to the main paper's topic
   - Quality and depth of the results
   - Coverage of the main paper's methodology
   - Overall usefulness for understanding the main paper
3. Provide detailed explanations with specific examples
4. The JSON response must be valid and complete
"""
        return prompt

    def _parse_individual_scores(self, raw_outputs_dir: str, evaluations_dir: str) -> bool:
        """Parse scores from individual result evaluation."""
        print(f"[{self.name}] Parsing individual evaluation scores...")
        os.makedirs(evaluations_dir, exist_ok=True)
        try:
            raw_output_path = os.path.join(raw_outputs_dir, f"{self.name.lower()}_individual_evaluation.txt")
            with open(raw_output_path, 'r') as f:
                raw_response = f.read()
            json_str = extract_json(raw_response)
            if json_str is None:
                print(f"[{self.name}] Could not find JSON in response")
                # Save raw output for debugging
                debug_path = os.path.join(evaluations_dir, f"{self.name.lower()}_individual_evaluation_debug.txt")
                with open(debug_path, 'w') as dbg:
                    dbg.write(raw_response)
                print(f"[{self.name}] Saved raw output to {debug_path}")
                return False    
            # Try parsing, and if it fails, try to fix common issues
            try:
                evaluation = json.loads(json_str)
            except Exception as e:
                print(f"[{self.name}] Error parsing JSON: {e}")
                # Try to fix more aggressively: remove all trailing commas
                json_str_fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
                try:
                    evaluation = json.loads(json_str_fixed)
                except Exception as e2:
                    print(f"[{self.name}] Still failed to parse JSON: {e2}")
                    debug_path = os.path.join(evaluations_dir, f"{self.name.lower()}_individual_evaluation_debug.txt")
                    with open(debug_path, 'w') as dbg:
                        dbg.write(raw_response)
                    print(f"[{self.name}] Saved raw output to {debug_path}")
                    return False
            print(f"[{self.name}] Successfully parsed JSON from response")
            self.scores = {
                "individual_scores": {},
                "overall_analysis": evaluation.get("overall_analysis", {})
            }
            if "individual_scores" in evaluation:
                for result_num, result_data in evaluation["individual_scores"].items():
                    score = result_data.get("score")
                    explanation = result_data.get("explanation")
                    if score is not None and isinstance(score, int) and 0 <= score <= 3:
                        self.scores["individual_scores"][result_num] = {
                            "score": score,
                            "explanation": explanation
                        }
            evaluation_path = os.path.join(evaluations_dir, f"{self.name.lower()}_individual_evaluation.json")
            with open(evaluation_path, 'w') as f:
                json.dump(self.scores, f, indent=2)
            print(f"[{self.name}] Saved parsed evaluation to {evaluation_path}")
            return True
        except Exception as e:
            print(f"[{self.name}] Error parsing individual scores: {e}")
            return False

    def _parse_multi_set_scores(self, raw_outputs_dir: str, evaluations_dir: str) -> bool:
        """Parse scores from multi-set comparison."""
        print(f"[{self.name}] Parsing multi-set comparison...")
        
        # Ensure the output directory exists
        os.makedirs(evaluations_dir, exist_ok=True)
        
        # Check if raw output file exists
        raw_output_path = os.path.join(raw_outputs_dir, f"{self.name.lower()}_multi_set_comparison.txt")
        if not os.path.exists(raw_output_path):
            print(f"[{self.name}] Error: Raw output file not found at {raw_output_path}")
            print(f"[{self.name}] This may indicate the LLM call failed or was interrupted")
            return False
        
        try:
            # Read the raw response from file
            with open(raw_output_path, 'r') as f:
                raw_response = f.read()

            if not raw_response.strip():
                print(f"[{self.name}] Error: Raw output file is empty")
                return False

            # Extract json from the response
            json_str = extract_json(raw_response)
            if json_str is None:
                print(f"[{self.name}] Could not find JSON in response")
                print(f"[{self.name}] Raw response content: {raw_response[:500]}...")  # Print first 500 chars for debugging
                return False    
        
            # Clean up the JSON string
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            try:
                # Parse the JSON
                evaluation = json.loads(json_str)
                print(f"[{self.name}] Successfully parsed JSON from response")
            except json.JSONDecodeError as e:
                print(f"[{self.name}] Error parsing JSON: {str(e)}")
                print(f"[{self.name}] JSON string content: {json_str[:500]}...")  # Print first 500 chars for debugging
                return False
            
            # Parse the evaluation
            self.scores = {
                "set_analyses": {},
                "comparison": evaluation.get("comparison", {})
            }
            
            # Parse set analyses
            if "set_analyses" in evaluation:
                for set_num, set_data in evaluation["set_analyses"].items():
                    self.scores["set_analyses"][set_num] = {
                        "strengths": set_data.get("strengths", []),
                        "weaknesses": set_data.get("weaknesses", []),
                        "summary": set_data.get("summary", "")
                    }
            
            # Save parsed evaluation
            evaluation_path = os.path.join(evaluations_dir, f"{self.name.lower()}_multi_set_comparison.json")
            with open(evaluation_path, 'w') as f:
                json.dump(self.scores, f, indent=2)
            print(f"[{self.name}] Saved parsed evaluation to {evaluation_path}")
                            
            return True
                
        except Exception as e:
            print(f"[{self.name}] Error parsing multi-set comparison: {str(e)}")
            import traceback
            print(f"[{self.name}] Traceback: {traceback.format_exc()}")
            return False

    def generate_challenge(self, discussion: str) -> Optional[Dict]:
        """Generate a challenge to another agent's evaluation."""
        if not self.scores:
            return None
            
        challenge_prompt = f"""You are {self.name}, reviewing a discussion about search result evaluations.

DISCUSSION:
{discussion}

YOUR EVALUATION:
{json.dumps(self.scores, indent=2)}

Please identify any points where you disagree with the discussion or other evaluations.
Provide your challenge in the following JSON format:

{{
    "challenge": {{
        "point_of_disagreement": "<specific point being challenged>",
        "your_perspective": "<your alternative perspective>",
        "reasoning": "<detailed explanation of why you disagree>",
        "confidence": <0-1>,  # Your confidence in this challenge
        "suggested_resolution": "<how you think this should be resolved>"
    }}
}}

Rules:
1. Only challenge if you have a strong alternative perspective
2. Provide clear reasoning for your disagreement
3. Suggest a specific resolution
4. Include your confidence level
"""
        response = self.get_completion(challenge_prompt)
        if response:
            try:
                json_str = extract_json(response)
                if json_str:
                    return json.loads(json_str)
            except Exception as e:
                print(f"[{self.name}] Error parsing challenge: {str(e)}")
        return None
        
    def respond_to_challenge(self, challenge: Dict) -> Optional[Dict]:
        """Respond to a challenge from another agent."""
        response_prompt = f"""You are {self.name}, responding to a challenge about your evaluation.

CHALLENGE:
{json.dumps(challenge, indent=2)}

YOUR ORIGINAL EVALUATION:
{json.dumps(self.scores, indent=2)}

Please provide your response in the following JSON format:

{{
    "response": {{
        "acknowledgment": "<do you acknowledge the challenge?>",
        "defense": "<defense of your original position>",
        "revision": "<any revisions to your original evaluation>",
        "confidence": <0-1>,  # Your confidence in this response
        "compromise": "<suggested compromise if applicable>"
    }}
}}

Rules:
1. Be open to valid challenges
2. Provide clear reasoning for your position
3. Be willing to revise if the challenge is valid
4. Include your confidence level
"""
        response = self.get_completion(response_prompt)
        if response:
            try:
                json_str = extract_json(response)
                if json_str:
                    return json.loads(json_str)
            except Exception as e:
                print(f"[{self.name}] Error parsing challenge response: {str(e)}")
        return None
        
    def update_evaluation(self, consensus: Dict) -> None:
        """Update evaluation based on consensus discussion."""
        self.evaluation_history.append({
            "original": self.scores,
            "consensus": consensus,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update performance metrics
        if "confidence_metrics" in consensus:
            self.performance_metrics[datetime.now().isoformat()] = {
                "agreement_level": consensus["confidence_metrics"]["agreement_level"],
                "confidence_score": consensus["confidence_metrics"]["confidence_score"]
            }

class ConsensusEvaluator:
    def __init__(self):
        self.agents = {
            AgentRole.CLAUDE: EvaluatorAgent(AgentRole.CLAUDE),
            AgentRole.GEMINI: EvaluatorAgent(AgentRole.GEMINI),
            AgentRole.DEEPSEEK: EvaluatorAgent(AgentRole.DEEPSEEK)
        }
        self.consensus = None
        self.discussion_history = []
        
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
        
        # Step 2: Generate consensus through discussion
        print("\n=== Generating Consensus ===")
        consensus_data = self._facilitate_discussion()
        if not consensus_data:
            print("Failed to generate consensus")
            return False
            
        # Step 3: Update PaperData objects with consensus scores
        print("\n=== Updating PaperData Objects with Scores ===")
        self._update_paper_data_scores(paper_data, new_paper_data)
        print("PaperData objects updated with scores")

        # Step 4: Save final results
        self.save_final_results(paper_data, new_paper_data, consensus_dir)
        print("Final results saved")
        
        return True
    
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
            "discussion_history": self.discussion_history,
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
        
    def _facilitate_discussion(self) -> Dict:
        """Facilitate discussion between agents to reach consensus."""
        print("\n=== Facilitating Discussion ===")
        
        # Step 1: Initial discussion
        print("Generating initial discussion...")
        discussion_prompt = self._generate_discussion_prompt()
        initial_discussion = self.agents[AgentRole.CLAUDE].get_completion(discussion_prompt)
        
        if not initial_discussion:
            print("Failed to generate initial discussion")
            return None
            
        self.discussion_history.append({
            "phase": "initial",
            "content": initial_discussion,
            "timestamp": datetime.now().isoformat()
        })
        
        # Step 2: Collect challenges
        print("\nCollecting challenges from agents...")
        challenges = []
        for agent in self.agents.values():
            if agent.role != AgentRole.CLAUDE:  # Skip Claude as facilitator
                print(f"Getting challenge from {agent.name}...")
                challenge = agent.generate_challenge(initial_discussion)
                if challenge:
                    challenges.append({
                        "agent": agent.name,
                        "challenge": challenge
                    })
        
        if challenges:
            self.discussion_history.append({
                "phase": "challenges",
                "content": challenges,
                "timestamp": datetime.now().isoformat()
            })
        
        # Step 3: Collect responses
        print("\nCollecting responses to challenges...")
        responses = []
        for challenge in challenges:
            print(f"Getting response to {challenge['agent']}'s challenge...")
            response = self.agents[AgentRole.CLAUDE].respond_to_challenge(challenge["challenge"])
            if response:
                responses.append({
                    "challenge": challenge,
                    "response": response
                })
        
        if responses:
            self.discussion_history.append({
                "phase": "responses",
                "content": responses,
                "timestamp": datetime.now().isoformat()
            })
        
        # Step 4: Generate final consensus
        print("\nGenerating final consensus...")
        consensus_prompt = self._generate_consensus_prompt(
            initial_discussion,
            challenges,
            responses
        )
        final_consensus = self.agents[AgentRole.CLAUDE].get_completion(consensus_prompt)
        
        if not final_consensus:
            print("Failed to generate final consensus")
            return None
            
        try:
            consensus_data = json.loads(extract_json(final_consensus))
            self.consensus = consensus_data
            
            # Update all agents with the consensus
            for agent in self.agents.values():
                agent.update_evaluation(consensus_data)
            
            return consensus_data
            
        except Exception as e:
            print(f"Error parsing consensus: {str(e)}")
            return None
    
    def _generate_discussion_prompt(self) -> str:
        """Generate the initial discussion prompt."""
        prompt = """You are Claude, tasked with facilitating a discussion between three expert evaluators (Claude, Gemini, and DeepSeek) who have independently evaluated search results.

Here are their individual evaluations:

"""
        
        for agent in self.agents.values():
            prompt += f"\n{agent.name}'s Evaluation:\n"
            prompt += json.dumps(agent.scores, indent=2)
            prompt += "\n"
            
        prompt += """
Please analyze these evaluations and facilitate a discussion to reach a consensus.

Provide your analysis in the following JSON format:

{
    "discussion": {
        "agreements": ["<list of points all evaluators agreed on>"],
        "disagreements": ["<list of points where evaluators differed>"],
        "key_insights": ["<list of important observations>"],
        "suggested_focus": ["<list of areas that need discussion>"]
    }
}

Rules:
1. Identify both agreements and disagreements
2. Highlight key insights from each evaluation
3. Suggest specific areas that need discussion
4. Be objective and fair in your analysis
"""
        return prompt
    
    def _generate_consensus_prompt(self, initial_discussion: str, challenges: List[Dict], responses: List[Dict]) -> str:
        """Generate the final consensus prompt."""
        challenges_and_responses = json.dumps({"challenges": challenges, "responses": responses}, indent=2)
        return f"""You are Claude, tasked with reaching a final consensus between three expert evaluators.

INITIAL DISCUSSION:
{initial_discussion}

CHALLENGES AND RESPONSES:
{challenges_and_responses}

Please provide a final consensus evaluation in the following JSON format:

{{
    "consensus": {{
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
    }},
    "discussion_summary": {{
        "resolved_points": ["<list of points that were resolved>"],
        "remaining_disagreements": ["<list of points that couldn't be fully resolved>"],
        "resolution_process": "<explanation of how consensus was reached>"
    }},
    "confidence_metrics": {{
        "agreement_level": <0-1>,
        "confidence_score": <0-1>,
        "uncertainty_areas": ["<list of areas with remaining uncertainty>"]
    }}
}}

Rules:
1. Consider all evaluations, challenges, and responses
2. Provide detailed explanations for the consensus
3. Acknowledge any remaining disagreements
4. Include confidence metrics
5. Ensure all scores are integers between 0 and 3
"""

def extract_json(text: str) -> Optional[str]:
    """Extract JSON object from text using multiple strategies and clean up common issues."""
    # Try to find JSON in code blocks
    json_match = re.search(r'```json\s*(\{[\s\S]*\})\s*```', text)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON between curly braces
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Try to find first { and last }
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
            else:
                return None
    # Clean up common JSON issues
    # Remove trailing commas before } or ]
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    # Remove any text after the final closing brace
    last_brace = json_str.rfind('}')
    if last_brace != -1:
        json_str = json_str[:last_brace+1]
    return json_str.strip()

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

def create_comparison_directories(bibcode: str, main_file: str, comparison_files: List[str]) -> Tuple[str, str, str, str]:
    """
    Create directory structure for a specific comparison.
    
    Args:
        bibcode: The bibcode being compared
        main_file: Main file being compared
        comparison_files: List of files to compare against
        
    Returns:
        A tuple containing (comparison_dir, raw_outputs_dir, evaluations_dir, consensus_dir)
    """
    # Get base names without extension and bibcode prefix
    file_bases = [os.path.splitext(os.path.basename(f))[0].replace(bibcode, "").strip("_") for f in [main_file] + comparison_files]
    
    # Create comparison directory name
    comparison_name = f"{bibcode}_" + "_".join(file_bases)
    
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

def select_files_to_compare(bibcode: str) -> Tuple[PaperData, List[PaperData], List[str]]:
    """
    Allow the user to select which files to compare.
    
    Args:
        bibcode: The bibcode to use for finding files
        
    Returns:
        A tuple containing (paper_data, result_sets, selected_files)
    """
    print("\n=== File Selection ===")
    
    # Create directory structure
    base_dir, inputs_dir, outputs_dir = create_bibcode_directories(bibcode)
    
    # Get all JSON files in the inputs directory
    all_files = glob.glob(os.path.join(inputs_dir, "*.json"))
    
    if not all_files:
        print(f"Error: No files found for bibcode {bibcode}")
        print(f"Please ensure at least one file exists in the {inputs_dir} directory")
        exit(1)
    
    # Display available files
    print("\nAvailable files:")
    for i, file in enumerate(all_files, 1):
        print(f"{i}. {os.path.basename(file)}")
    
    # Ask user to select files
    while True:
        try:
            print("\nSelect files to compare (enter numbers separated by space):")
            print("For individual judgements: select 1 file")
            print("For multi-set comparison: select 2 or more files")
            selections = input("> ").strip().split()
            
            if not selections:
                print("Please enter at least one number.")
                continue
                
            # Convert selections to indices and validate
            selected_indices = [int(s) - 1 for s in selections]
            if any(idx < 0 or idx >= len(all_files) for idx in selected_indices):
                print("Invalid selection. Please enter numbers from the list above.")
                continue
                
            if len(set(selected_indices)) != len(selected_indices):
                print("Please select each file only once.")
                continue
                
            selected_files = [all_files[idx] for idx in selected_indices]
            break
        except ValueError:
            print("Please enter valid numbers.")
    
    print(f"\nSelected files:")
    for i, file in enumerate(selected_files, 1):
        print(f"Set {i}: {os.path.basename(file)}")
    
    # Load the selected files
    print("\nLoading selected files...")
    paper_data = PaperData.from_json(selected_files[0])
    result_sets = [PaperData.from_json(f) for f in selected_files[1:]]
    print(f"Loaded main paper: {paper_data.bibcode} - {paper_data.title}")
    for i, result_set in enumerate(result_sets, 1):
        print(f"Loaded Set {i+1}: {result_set.bibcode} - {result_set.title}")
    
    return paper_data, result_sets, selected_files

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

def create_individual_result_directories(bibcode: str, result_file: str) -> Tuple[str, str, str, str]:
    """
    Create directory structure for individual result evaluation.
    
    Args:
        bibcode: The bibcode being evaluated
        result_file: The result file being evaluated
        
    Returns:
        A tuple containing (result_dir, raw_outputs_dir, evaluations_dir, consensus_dir)
    """
    # Get the base name without extension and bibcode prefix
    file_base = os.path.splitext(os.path.basename(result_file))[0]
    if file_base.startswith(bibcode):
        file_base = file_base[len(bibcode):].strip("_")
    
    # Create result directory name
    result_name = f"{bibcode}_{file_base}"
    
    print(f"\nCreating result directory: {result_name}")
    
    # Create directory structure
    base_dir, inputs_dir, outputs_dir = create_bibcode_directories(bibcode)
    result_dir = os.path.join(outputs_dir, result_name)
    raw_outputs_dir = os.path.join(result_dir, "raw_outputs")
    evaluations_dir = os.path.join(result_dir, "evaluations")
    consensus_dir = os.path.join(result_dir, "consensus")
    
    # Create directories if they don't exist
    os.makedirs(raw_outputs_dir, exist_ok=True)
    os.makedirs(evaluations_dir, exist_ok=True)
    os.makedirs(consensus_dir, exist_ok=True)
    
    print(f"  - Result directory: {result_dir}")
    print(f"  - Raw outputs directory: {raw_outputs_dir}")
    print(f"  - Evaluations directory: {evaluations_dir}")
    print(f"  - Consensus directory: {consensus_dir}")
    
    return result_dir, raw_outputs_dir, evaluations_dir, consensus_dir

def merge_scores_for_set(set_dir, input_json_path):
    # Paths
    evaluations_dir = os.path.join(set_dir, 'evaluations')
    consensus_dir = os.path.join(set_dir, 'consensus')
    # LLM evaluation files
    eval_files = {
        'claude': os.path.join(evaluations_dir, 'claude_individual_evaluation.json'),
        'gemini': os.path.join(evaluations_dir, 'gemini_individual_evaluation.json'),
        'deepseek': os.path.join(evaluations_dir, 'deepseek_individual_evaluation.json'),
    }
    # Consensus file
    consensus_file = os.path.join(consensus_dir, 'consensus_results.json')

    # Load input JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # Load LLM evaluations
    llm_scores = {}
    for llm, path in eval_files.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                llm_scores[llm] = json.load(f)
        else:
            llm_scores[llm] = None

    # Load consensus
    consensus = None
    if os.path.exists(consensus_file):
        with open(consensus_file, 'r') as f:
            consensus_data = json.load(f)
            consensus = consensus_data.get('consensus') or consensus_data.get('consensus', {}).get('consensus')
            if consensus is None and 'consensus' in consensus_data:
                consensus = consensus_data['consensus']
    
    # Merge scores into search results
    for idx, result in enumerate(data.get('search_results', []), 1):
        idx_str = str(idx)
        # LLM scores
        for llm in ['claude', 'gemini', 'deepseek']:
            scores = llm_scores.get(llm)
            if scores and 'individual_scores' in scores and idx_str in scores['individual_scores']:
                result[f'{llm}_score'] = scores['individual_scores'][idx_str]['score']
                result[f'{llm}_explanation'] = scores['individual_scores'][idx_str]['explanation']
            else:
                result[f'{llm}_score'] = None
                result[f'{llm}_explanation'] = None
        # Consensus
        if consensus and 'consensus' in consensus:
            consensus_scores = consensus['consensus']
            if 'individual_scores' in consensus_scores.get('first_set', {}):
                if idx_str in consensus_scores['first_set']['individual_scores']:
                    cscore = consensus_scores['first_set']['individual_scores'][idx_str]
                    result['consensus_score'] = cscore.get('score')
                    result['consensus_explanation'] = cscore.get('explanation')
                else:
                    result['consensus_score'] = None
                    result['consensus_explanation'] = None
            else:
                result['consensus_score'] = None
                result['consensus_explanation'] = None
        else:
            result['consensus_score'] = None
            result['consensus_explanation'] = None

    # Output path: outputs/<set_name>_llm_scored_results.json
    outputs_dir = os.path.dirname(set_dir)
    set_name = os.path.basename(set_dir)
    out_path = os.path.join(outputs_dir, f"{set_name}_llm_scored_results.json")
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Wrote merged LLM+consensus scores: {out_path}")

def find_input_json(set_dir):
    parent = os.path.dirname(os.path.dirname(set_dir))
    inputs_dir = os.path.join(parent, 'inputs')
    candidates = glob.glob(os.path.join(inputs_dir, '*.json'))
    if not candidates:
        return None
    set_base = os.path.basename(set_dir)
    for c in candidates:
        if set_base in c:
            return c
    return candidates[0]

def main():
    parser = argparse.ArgumentParser(description="LLM Comparator: Evaluate search results using multiple LLMs.")
    parser.add_argument('--auto', action='store_true', help='Run in automatic mode with a list of bibcodes')
    parser.add_argument('--bibcodes', nargs='*', help='List of bibcodes to process in auto mode')
    parser.add_argument('--all-inputs', action='store_true', help='In auto mode, process all input files in the input directory for each bibcode')
    parser.add_argument('--force-update', action='store_true', help='Force update all output files, even if they already exist')
    args = parser.parse_args()

    if args.auto and args.all_inputs:
        # Determine bibcodes to process
        if args.bibcodes:
            bibcodes = args.bibcodes
        else:
            # Find all bibcodes in the examples directory
            examples_dir = os.path.join(ROOT_DIR, "examples")
            bibcodes = [d for d in os.listdir(examples_dir) if os.path.isdir(os.path.join(examples_dir, d))]
        for test_bibcode in bibcodes:
            print(f"\n=== Processing bibcode: {test_bibcode} ===")
            base_dir, inputs_dir, outputs_dir = create_bibcode_directories(test_bibcode)
            all_files = glob.glob(os.path.join(inputs_dir, "*.json"))
            if not all_files:
                print(f"No input files found for bibcode {test_bibcode}")
                continue
            # Create evaluator agents
            claude_agent = EvaluatorAgent(AgentRole.CLAUDE)
            gemini_agent = EvaluatorAgent(AgentRole.GEMINI)
            deepseek_agent = EvaluatorAgent(AgentRole.DEEPSEEK)
            # Process each input file as an individual judgement
            for i, result_file in enumerate(all_files, 0):
                result_set = PaperData.from_json(result_file)
                result_dir, set_raw_outputs, set_evaluations, set_consensus = create_individual_result_directories(
                    test_bibcode, result_file)
                print(f"\nProcessing Set {i+1}: {os.path.basename(result_dir)}")
                
                # Check if we should skip this set
                consensus_path = os.path.join(set_consensus, "consensus_results.json")
                if not args.force_update and os.path.exists(consensus_path):
                    print(f"Skipping {os.path.basename(result_dir)} - output already exists")
                    continue
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    future_claude = executor.submit(claude_agent.evaluate_individual_results, result_set, set_raw_outputs)
                    future_gemini = executor.submit(gemini_agent.evaluate_individual_results, result_set, set_raw_outputs)
                    future_deepseek = executor.submit(deepseek_agent.evaluate_individual_results, result_set, set_raw_outputs)
                    concurrent.futures.wait([future_claude, future_gemini, future_deepseek])
                print(f"All evaluations completed")
                print(f"\nParsing evaluation scores...")
                claude_agent._parse_individual_scores(set_raw_outputs, set_evaluations)
                gemini_agent._parse_individual_scores(set_raw_outputs, set_evaluations)
                deepseek_agent._parse_individual_scores(set_raw_outputs, set_evaluations)
                print(f"\nGenerating consensus...")
                consensus_evaluator = ConsensusEvaluator()
                consensus_evaluator.agents = {
                    AgentRole.CLAUDE: claude_agent,
                    AgentRole.GEMINI: gemini_agent,
                    AgentRole.DEEPSEEK: deepseek_agent
                }
                consensus_data = consensus_evaluator._facilitate_discussion()
                if consensus_data:
                    with open(consensus_path, 'w') as f:
                        json.dump({
                            "consensus": consensus_data,
                            "discussion_history": consensus_evaluator.discussion_history,
                            "timestamp": datetime.now().isoformat()
                        }, f, indent=2)
                    print(f"Saved consensus results to {consensus_path}")
                # Merge LLM and consensus scores into the input JSON
                input_json = find_input_json(result_dir)
                if input_json:
                    merge_scores_for_set(result_dir, input_json)
                print(f"\nCompleted evaluation for Set {i+1}")
                print(f"Results saved in: {result_dir}")
        print("\n=== Automatic Evaluation Complete! ===")
        return

    if args.auto and args.bibcodes:
        # Automatic mode: process each bibcode in sequence (legacy behavior)
        for test_bibcode in args.bibcodes:
            print(f"\n=== Processing bibcode: {test_bibcode} ===")
            base_dir, inputs_dir, outputs_dir = create_bibcode_directories(test_bibcode)
            all_files = glob.glob(os.path.join(inputs_dir, "*.json"))
            if not all_files:
                print(f"No input files found for bibcode {test_bibcode}")
                continue
            paper_data = PaperData.from_json(all_files[0])
            result_sets = [PaperData.from_json(f) for f in all_files]
            selected_files = all_files
            claude_agent = EvaluatorAgent(AgentRole.CLAUDE)
            gemini_agent = EvaluatorAgent(AgentRole.GEMINI)
            deepseek_agent = EvaluatorAgent(AgentRole.DEEPSEEK)
            for i, (result_set, result_file) in enumerate(zip(result_sets, selected_files), 0):
                result_dir, set_raw_outputs, set_evaluations, set_consensus = create_individual_result_directories(
                    test_bibcode, result_file)
                print(f"\nProcessing Set {i+1}: {os.path.basename(result_dir)}")
                
                # Check if we should skip this set
                consensus_path = os.path.join(set_consensus, "consensus_results.json")
                if not args.force_update and os.path.exists(consensus_path):
                    print(f"Skipping {os.path.basename(result_dir)} - output already exists")
                    continue
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    future_claude = executor.submit(claude_agent.evaluate_individual_results, result_set, set_raw_outputs)
                    future_gemini = executor.submit(gemini_agent.evaluate_individual_results, result_set, set_raw_outputs)
                    future_deepseek = executor.submit(deepseek_agent.evaluate_individual_results, result_set, set_raw_outputs)
                    concurrent.futures.wait([future_claude, future_gemini, future_deepseek])
                print(f"All evaluations completed")
                print(f"\nParsing evaluation scores...")
                claude_agent._parse_individual_scores(set_raw_outputs, set_evaluations)
                gemini_agent._parse_individual_scores(set_raw_outputs, set_evaluations)
                deepseek_agent._parse_individual_scores(set_raw_outputs, set_evaluations)
                print(f"\nGenerating consensus...")
                consensus_evaluator = ConsensusEvaluator()
                consensus_evaluator.agents = {
                    AgentRole.CLAUDE: claude_agent,
                    AgentRole.GEMINI: gemini_agent,
                    AgentRole.DEEPSEEK: deepseek_agent
                }
                consensus_data = consensus_evaluator._facilitate_discussion()
                if consensus_data:
                    with open(consensus_path, 'w') as f:
                        json.dump({
                            "consensus": consensus_data,
                            "discussion_history": consensus_evaluator.discussion_history,
                            "timestamp": datetime.now().isoformat()
                        }, f, indent=2)
                    print(f"Saved consensus results to {consensus_path}")
                input_json = find_input_json(result_dir)
                if input_json:
                    merge_scores_for_set(result_dir, input_json)
                print(f"\nCompleted evaluation for Set {i+1}")
                print(f"Results saved in: {result_dir}")
        print("\n=== Automatic Evaluation Complete! ===")
        return

    # Manual mode (default)
    print("\n=== LLM Comparator ===")
    print("This tool evaluates search results using multiple LLMs")
    print("\nInitializing evaluator...")
    
    # Let user select which bibcode to analyze
    test_bibcode = list_available_bibcodes()
    
    print("\nStep 1: File Selection")
    print("----------------------")
    # Allow user to select which files to compare
    paper_data, result_sets, selected_files = select_files_to_compare(test_bibcode)
    
    print("\nStep 2: Creating Evaluator Agents")
    print("--------------------------------")
    # Create evaluator agents
    print("Initializing Claude agent...")
    claude_agent = EvaluatorAgent(AgentRole.CLAUDE)
    print("Initializing Gemini agent...")
    gemini_agent = EvaluatorAgent(AgentRole.GEMINI)
    print("Initializing DeepSeek agent...")
    deepseek_agent = EvaluatorAgent(AgentRole.DEEPSEEK)
    
    # Ask user which evaluation mode to use
    print("\nSelect evaluation mode:")
    print("1. Individual result evaluation")
    print("2. Multi-set comparison")
    while True:
        try:
            mode = int(input("Enter 1 or 2: "))
            if mode in [1, 2]:
                break
            print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")
    
    if mode == 1:
        print("\nStep 3: Running Individual Result Evaluations")
        print("-------------------------------------------")
        
        # Process each result set separately
        for i, (result_set, result_file) in enumerate(zip([paper_data] + result_sets, selected_files), 0):
            # Create directories for this result set
            result_dir, set_raw_outputs, set_evaluations, set_consensus = create_individual_result_directories(
                test_bibcode, 
                result_file
            )
            
            print(f"\nProcessing Set {i+1}: {os.path.basename(result_dir)}")
            
            # Check if we should skip this set
            consensus_path = os.path.join(set_consensus, "consensus_results.json")
            if not args.force_update and os.path.exists(consensus_path):
                print(f"Skipping {os.path.basename(result_dir)} - output already exists")
                continue
            
            # Step 1: Run individual evaluations by each LLM agent
            print(f"\nStarting parallel evaluation...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                print("Submitting evaluation tasks to agents...")
                future_claude = executor.submit(claude_agent.evaluate_individual_results, result_set, set_raw_outputs)
                future_gemini = executor.submit(gemini_agent.evaluate_individual_results, result_set, set_raw_outputs)
                future_deepseek = executor.submit(deepseek_agent.evaluate_individual_results, result_set, set_raw_outputs)
                
                print("Waiting for all evaluations to complete...")
                concurrent.futures.wait([future_claude, future_gemini, future_deepseek])
                print(f"All evaluations completed")
            
            # Step 2: Parse individual evaluation scores
            print(f"\nParsing evaluation scores...")
            print("Parsing scores from Claude's evaluation...")
            claude_agent._parse_individual_scores(set_raw_outputs, set_evaluations)
            print("Parsing scores from Gemini's evaluation...")
            gemini_agent._parse_individual_scores(set_raw_outputs, set_evaluations)
            print("Parsing scores from DeepSeek's evaluation...")
            deepseek_agent._parse_individual_scores(set_raw_outputs, set_evaluations)
            
            # Step 3: Generate consensus for this set
            print(f"\nGenerating consensus...")
            consensus_evaluator = ConsensusEvaluator()
            consensus_evaluator.agents = {
                AgentRole.CLAUDE: claude_agent,
                AgentRole.GEMINI: gemini_agent,
                AgentRole.DEEPSEEK: deepseek_agent
            }
            
            # Generate consensus through discussion
            consensus_data = consensus_evaluator._facilitate_discussion()
            if consensus_data:
                # Save consensus results
                with open(consensus_path, 'w') as f:
                    json.dump({
                        "consensus": consensus_data,
                        "discussion_history": consensus_evaluator.discussion_history,
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2)
                print(f"Saved consensus results to {consensus_path}")
            
            # Merge LLM and consensus scores into the input JSON
            input_json = find_input_json(result_dir)
            if input_json:
                merge_scores_for_set(result_dir, input_json)
            
            print(f"\nCompleted evaluation for Set {i+1}")
            print(f"Results saved in: {result_dir}")
        
        print("\n=== Individual Result Evaluations Complete! ===")
        print(f"Results have been saved in individual result directories")
    
    else:  # mode == 2
        print("\nStep 3: Running Multi-Set Comparison")
        print("-----------------------------------")
        
        # Create comparison directories
        comparison_dir, raw_outputs_dir, evaluations_dir, consensus_dir = create_comparison_directories(
            test_bibcode,
            selected_files[0],
            selected_files[1:]
        )
        
        # Process each comparison set
        for i, (result_set, result_file) in enumerate(zip(result_sets, selected_files[1:]), 0):
            print(f"\nProcessing Comparison Set {i+1}: {os.path.basename(result_file)}")
            
            # Check if we should skip this set
            set_consensus_dir = os.path.join(consensus_dir, f"set_{i+1}")
            consensus_path = os.path.join(set_consensus_dir, "consensus_results.json")
            if not args.force_update and os.path.exists(consensus_path):
                print(f"Skipping Comparison Set {i+1} - output already exists")
                continue
            
            # Step 1: Get individual judgements from each LLM
            print(f"\nGetting individual judgements...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                print("Submitting comparison tasks to agents...")
                future_claude = executor.submit(claude_agent.compare_multiple_sets, paper_data, [result_set], raw_outputs_dir)
                future_gemini = executor.submit(gemini_agent.compare_multiple_sets, paper_data, [result_set], raw_outputs_dir)
                future_deepseek = executor.submit(deepseek_agent.compare_multiple_sets, paper_data, [result_set], raw_outputs_dir)
                
                print("Waiting for all comparisons to complete...")
                concurrent.futures.wait([future_claude, future_gemini, future_deepseek])
                print(f"All comparisons completed")
            
            # Step 2: Parse individual comparison scores
            print(f"\nParsing comparison scores...")
            print("Parsing scores from Claude's comparison...")
            claude_agent._parse_multi_set_scores(raw_outputs_dir, evaluations_dir)
            print("Parsing scores from Gemini's comparison...")
            gemini_agent._parse_multi_set_scores(raw_outputs_dir, evaluations_dir)
            print("Parsing scores from DeepSeek's comparison...")
            deepseek_agent._parse_multi_set_scores(raw_outputs_dir, evaluations_dir)
            
            # Step 3: Generate consensus for this comparison set
            print(f"\nGenerating consensus for Comparison Set {i+1}...")
            consensus_evaluator = ConsensusEvaluator()
            consensus_evaluator.agents = {
                AgentRole.CLAUDE: claude_agent,
                AgentRole.GEMINI: gemini_agent,
                AgentRole.DEEPSEEK: deepseek_agent
            }
            
            # Generate consensus through discussion
            consensus_data = consensus_evaluator._facilitate_discussion()
            if consensus_data:
                # Save consensus results
                os.makedirs(set_consensus_dir, exist_ok=True)
                with open(consensus_path, 'w') as f:
                    json.dump({
                        "consensus": consensus_data,
                        "discussion_history": consensus_evaluator.discussion_history,
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2)
                print(f"Saved consensus results to {consensus_path}")
            
            # Merge LLM and consensus scores into the input JSON
            input_json = find_input_json(result_dir if 'result_dir' in locals() else comparison_dir)
            if input_json:
                merge_scores_for_set(result_dir if 'result_dir' in locals() else comparison_dir, input_json)
            
            print(f"\nCompleted evaluation for Comparison Set {i+1}")
        
        print("\n=== Multi-Set Comparison Complete! ===")
        print(f"Results have been saved in: {comparison_dir}")
    
    print("\n=== Evaluation Complete! ===")
    print(f"Results have been saved in: {comparison_dir if mode == 2 else 'individual result directories'}")
    print("\nYou can find the following files:")
    if mode == 1:
        for result_file in selected_files[1:]:
            result_dir, _, _, _ = create_individual_result_directories(test_bibcode, result_file)
            print(f"\nFor {os.path.basename(result_dir)}:")
            print(f"- Raw outputs: {os.path.join(result_dir, 'raw_outputs')}")
            print(f"- Individual evaluations: {os.path.join(result_dir, 'evaluations')}")
            print(f"- Consensus results: {os.path.join(result_dir, 'consensus')}")
    else:
        print(f"- Raw outputs: {raw_outputs_dir}")
        print(f"- Individual evaluations: {evaluations_dir}")
        print(f"- Consensus results: {consensus_dir}")
    print("\nThank you for using the LLM Comparator!")

if __name__ == "__main__":
    main() 