# LLM Comparator (Agentic Evaluator)

A tool for comparing two sets of search results using multiple Large Language Models (LLMs) to evaluate their relevance and quality.

## Overview

The LLM Comparator uses three different LLMs (Claude, Gemini, and DeepSeek) to independently evaluate two sets of search results against a main paper. The evaluations are then combined to reach a consensus on which set is better and why.

This tool is particularly useful for:
- Comparing different search algorithms or methodologies
- Evaluating the quality of search results
- Analyzing the relevance of papers in a research context
- Getting multiple perspectives on search result quality

## Features

- **Multi-Agent Evaluation**: Uses three different LLMs (Claude, Gemini, and DeepSeek) to provide diverse perspectives
- **Parallel Processing**: Evaluates search results in parallel for faster execution
- **Structured Output**: Generates detailed evaluations with scores and justifications
- **Consensus Building**: Combines individual evaluations to reach a consensus
- **Organized File Structure**: Creates a clear directory structure for inputs and outputs
- **Interactive File Selection**: Allows users to select which files to compare
- **Comprehensive Reporting**: Saves both raw responses and parsed evaluations

## Requirements

- Python 3.8+
- API keys for:
  - Anthropic Claude (via AWS Bedrock)
  - Google Gemini
  - DeepSeek (via AWS Bedrock)
  - ADS API (for paper metadata)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/search-results-llm-evaluator.git
   cd search-results-llm-evaluator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `secrets.toml` file in the root directory with your API keys:
   ```toml
   [api_keys]
   gcp_api = "your_gcp_api_key"
   ollama_api = "your_ollama_api_key"
   ads_api_token = "your_ads_api_token"
   bedrock_access_key = "your_bedrock_access_key"
   bedrock_secret_access_key = "your_bedrock_secret_access_key"
   ```

## Usage

1. Prepare your input files:
   There are two ways to create input files:

   a. Using the provided scripts in `make_input_files/`:
      - `fetch_papers_from_ADS.py`: Creates input files by fetching papers from ADS
        ```bash
        python make_input_files/fetch_papers_from_ADS.py
        ```
        This will create a file with papers related to the specified bibcode in the `examples/<bibcode>/inputs/` directory.

      - `create_file_from_manual_ranking.py`: Creates input files from a manually ranked list of papers
        ```bash
        python make_input_files/create_file_from_manual_ranking.py
        ```
        This will create a file with manually ranked papers in the `examples/<bibcode>/inputs/` directory.

   b. Manual preparation:
      - Create a directory for your bibcode in the `examples` directory (e.g., `examples/2022ApJ...931...44P`)
      - Place your JSON files in the `inputs` subdirectory
      - Each file should contain paper data in the format expected by the `PaperData` class

2. Run the evaluator:
   ```
   python agentic_evaluator/compare_search_results.py
   ```

3. Follow the interactive prompts:
   - Select which bibcode to analyze from the available options
   - Select which files to compare
   - Choose which file should be Set A and which should be Set B

4. Review the results:
   - Raw outputs from each LLM
   - Individual evaluations with scores
   - Consensus evaluation
   - Final results with all scores

## Workflow and Agent Interaction

The LLM Comparator follows a structured workflow where multiple agents evaluate search results in parallel and then reach a consensus. Here's how it works:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              User Input                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        File Selection Process                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Create Directory Structure                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Load Paper Data (Set A & B)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Parallel Agent Evaluation                         │
└─────────────────────────────────────────────────────────────────────────┘
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
│   Claude Agent  │ │ Gemini Agent│ │ DeepSeek Agent  │
└─────────────────┘ └─────────────┘ └─────────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
│Generate Prompt  │ │Gen. Prompt  │ │Generate Prompt  │
└─────────────────┘ └─────────────┘ └─────────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
│Get LLM Response │ │Get Response │ │Get LLM Response │
└─────────────────┘ └─────────────┘ └─────────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
│Save Raw Response│ │Save Response│ │Save Raw Response│
└─────────────────┘ └─────────────┘ └─────────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
│  Parse Scores   │ │Parse Scores │ │  Parse Scores   │
└─────────────────┘ └─────────────┘ └─────────────────┘
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Generate Consensus Prompt                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Get Consensus Response                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Parse Consensus Data                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Update PaperData Objects                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Save Final Results                                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Format

### Input Files

Input files should be JSON files with the following structure:

```json
{
  "bibcode": "2022ApJ...931...44P",
  "title": "Paper Title",
  "abstract": "Paper Abstract",
  "search_results": [
    {
      "bibcode": "2021ApJ...920...12Q",
      "title": "Result Title",
      "abstract": "Result Abstract"
    },
    ...
  ]
}
```

### Output Files

The tool generates several output files:

- **Raw Outputs**: The complete responses from each LLM
- **Evaluations**: Parsed scores and evaluations from each LLM
- **Consensus**: The combined evaluation with consensus scores
- **Final Results**: Complete data with all scores and evaluations

## Scoring System

The tool uses a 0-3 scoring system:

- **Content Similarity (0-3)**:
  - 0 = Somewhat relevant but tangential
  - 1 = Relevant but not focused on the main topic
  - 2 = Strongly relevant and focused on the main topic
  - 3 = Perfect match in topic and methodology

- **Ranking Quality (0-3)**:
  - 0 = Some relevant results are near the top
  - 1 = Most relevant results are near the top
  - 2 = Results are well-ordered by relevance
  - 3 = Results are perfectly ordered by relevance

- **Relative Score (0-3)**:
  - 0 = Both sets are equally good
  - 1 = One set is slightly better
  - 2 = One set is moderately better
  - 3 = One set is much better

## License

[MIT License](LICENSE)

## Acknowledgements

- [LiteLLM](https://github.com/BerriAI/litellm) for LLM API integration
- [ADS API](https://ui.adsabs.harvard.edu/) for paper metadata 

## File Structure

The tool creates the following directory structure:

```
search-results-llm-evaluator/
├── agentic_evaluator/
│   └── compare_search_results.py
├── make_input_files/
│   ├── fetch_papers_from_ADS.py
│   └── create_file_from_manual_ranking.py
├── examples/
│   └── bibcode/
│       ├── inputs/
│       │   ├── bibcode.json
│       │   ├── bibcode_manual.json
│       │   └── bibcode_useful.json
│       └── outputs/
│           └── bibcode_manual_useful/
│               ├── raw_outputs/
│               │   ├── claude_raw_output.txt
│               │   ├── gemini_raw_output.txt
│               │   └── deepseek_raw_output.txt
│               ├── evaluations/
│               │   ├── claude_evaluation.json
│               │   ├── gemini_evaluation.json
│               │   └── deepseek_evaluation.json
│               └── consensus/
│                   ├── consensus_evaluation.json
│                   ├── consensus_summary.txt
│                   ├── final_results.json
│                   ├── set_a_final.json
│                   └── set_b_final.json
├── llmcomp2/
│   ├── lib/
│   ├── include/
│   └── bin/
├── config.yaml
├── secrets.toml
└── README.md
```
