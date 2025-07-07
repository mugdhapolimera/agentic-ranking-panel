import os
import json
import glob
import argparse

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
            # Try to extract the consensus block
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
                # Try to match by index (assuming single set)
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

    # Output path
    out_path = os.path.splitext(input_json_path)[0] + '_llm_scored_results.json'
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Wrote: {out_path}")

def find_input_json(set_dir):
    # Look for *_similar_judgements.json or *_manual_judgements.json in ../inputs/
    parent = os.path.dirname(os.path.dirname(set_dir))
    inputs_dir = os.path.join(parent, 'inputs')
    candidates = glob.glob(os.path.join(inputs_dir, '*.json'))
    if not candidates:
        return None
    # Prefer file with set name in it
    set_base = os.path.basename(set_dir)
    for c in candidates:
        if set_base in c:
            return c
    # Otherwise just return the first
    return candidates[0]

def process_all_sets(outputs_dir):
    # Find all set directories (those with evaluations/ and consensus/)
    for entry in os.listdir(outputs_dir):
        set_dir = os.path.join(outputs_dir, entry)
        if not os.path.isdir(set_dir):
            continue
        if not (os.path.isdir(os.path.join(set_dir, 'evaluations')) and os.path.isdir(os.path.join(set_dir, 'consensus'))):
            continue
        input_json = find_input_json(set_dir)
        if input_json:
            print(f"Processing set: {set_dir}")
            merge_scores_for_set(set_dir, input_json)
        else:
            print(f"No input JSON found for set: {set_dir}")

def main():
    parser = argparse.ArgumentParser(description="Merge LLM and consensus scores into input JSONs for all sets in a directory.")
    parser.add_argument('outputs_dir', help='Path to the outputs directory (e.g., examples/2022ApJ...931...44P/outputs/)')
    args = parser.parse_args()
    process_all_sets(args.outputs_dir)

if __name__ == "__main__":
    main() 