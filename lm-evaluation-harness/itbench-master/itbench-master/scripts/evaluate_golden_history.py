import argparse
import json
import os
import sys
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path to allow importing leadbench package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from itbench.utils.config import config
from itbench.utils.dataset import DialogueEvaluationDataset
from itbench.model.api_model import APIModel
from itbench.generation.generator import ResponseGenerator
from itbench.evaluation.evaluator import DialogueEvaluator
from itbench.utils.report import ReportGenerator
from itbench.generation.model_configs import get_processor_for_model

def process_sample_pipeline(sample, generator, evaluator):
    """
    Process a single sample through the full pipeline:
    1. Generate response (Candidate Model)
    2. Evaluate response (Judge Model)
    """
    try:
        # Step 1: Generation
        # Only generate if response is missing
        if not sample.get('response'):
            sample = generator.process_sample(sample)
            
        # Step 2: Evaluation
        # Evaluation uses sample['response'] which is the processed response
        return evaluator.evaluate_sample(sample)
    except Exception as e:
        print(f"Error processing sample ID {sample.get('id', 'unknown')}: {e}")
        return None

import shutil

def main():
    parser = argparse.ArgumentParser(description="LeadBench Evaluation Pipeline")
    args = parser.parse_args()

    # Load version info
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'version'), 'r') as f:
            version = f.read().strip()
    except:
        version = "unknown"

    # Initialize Models
    print(f"Initializing Candidate Model: {config.CANDIDATE_MODEL_NAME}...")
    candidate_model = APIModel(
        model_name=config.CANDIDATE_MODEL_NAME,
        api_key=config.CANDIDATE_API_KEY,
        api_base=config.CANDIDATE_API_BASE
    )

    # Get model-specific processor
    print(f"Configuring processor for model: {config.CANDIDATE_MODEL_NAME}...")
    processor = get_processor_for_model(config.CANDIDATE_MODEL_NAME)
    
    print(f"Initializing Response Generator (Max Tokens: {config.CANDIDATE_MAX_OUTPUT_TOKENS})...")
    generator = ResponseGenerator(
        model=candidate_model, 
        processor=processor,
        max_tokens=config.CANDIDATE_MAX_OUTPUT_TOKENS
    )

    print(f"Initializing Judge Model: {config.JUDGE_MODEL_NAME}...")
    judge_model = APIModel(
        model_name=config.JUDGE_MODEL_NAME,
        api_key=config.JUDGE_API_KEY,
        api_base=config.JUDGE_API_BASE,
        temperature=0.01 
    )
    evaluator = DialogueEvaluator(
        model=judge_model,
        enable_thinking=config.JUDGE_ENABLE_THINKING
    )

    # Initialize Dataset
    print(f"Loading dataset from {config.INPUT_FILE}...")
    try:
        dataset = DialogueEvaluationDataset(
            file_path=config.INPUT_FILE,
            rules_file=config.RULES_FILE
        )
        print(f"Loaded {len(dataset)} samples.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Determine subset
    if config.EVALUATION_LIMIT:
        print(f"Limiting evaluation to first {config.EVALUATION_LIMIT} samples.")
        indices = range(min(len(dataset), config.EVALUATION_LIMIT))
    else:
        indices = range(len(dataset))

    # Prepare Output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = config.OUTPUT
    if not base_output_dir:
        base_output_dir = "output"
        
    # Extract input filename for the directory name
    input_filename = os.path.splitext(os.path.basename(config.INPUT_FILE))[0]
    
    # Use CANDIDATE_MODEL_NAME in output directory
    candidate_model_name = config.CANDIDATE_MODEL_NAME.replace("/", "_").replace("\\", "_") # Sanitize
    output_dir = os.path.join(base_output_dir, f"v{version}_{candidate_model_name}_{input_filename}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy Input and Rules files
    if os.path.exists(config.INPUT_FILE):
        shutil.copy2(config.INPUT_FILE, output_dir)
    if os.path.exists(config.RULES_FILE):
        shutil.copy2(config.RULES_FILE, output_dir)
    
    # Save Configuration
    config_save_path = os.path.join(output_dir, "config.json")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        # Better approach: dump specific relevant fields
        final_config = {
            "CANDIDATE_MODEL_NAME": config.CANDIDATE_MODEL_NAME,
            "CANDIDATE_API_BASE": config.CANDIDATE_API_BASE,
            "JUDGE_MODEL_NAME": config.JUDGE_MODEL_NAME,
            "JUDGE_API_BASE": config.JUDGE_API_BASE,
            "JUDGE_ENABLE_THINKING": config.JUDGE_ENABLE_THINKING,
            "INPUT_FILE": config.INPUT_FILE,
            "RULES_FILE": config.RULES_FILE,
            "OUTPUT": config.OUTPUT,
            "CONCURRENCY": config.CONCURRENCY,
            "TIMESTAMP": timestamp
        }
        json.dump(final_config, f, indent=4, ensure_ascii=False)
    
    # Update output file path to be inside the new directory
    output_file_path = os.path.join(output_dir, "evaluation_results.jsonl")
    
    report_gen = ReportGenerator(output_dir, rules_file=config.RULES_FILE, version=version)

    # Run Pipeline
    print(f"Starting pipeline with {config.CONCURRENCY} threads...")
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=config.CONCURRENCY) as executor:
        future_to_sample = {executor.submit(process_sample_pipeline, dataset[i], generator, evaluator): i for i in indices}
        
        for future in tqdm(as_completed(future_to_sample), total=len(indices), desc="Processing"):
            result = future.result()
            if result:
                results.append(result)
                report_gen.add_result(result)
    
    end_time = time.time()
    total_duration = end_time - start_time

    # Save Results
    print(f"Saving {len(results)} results to {output_file_path}...")
    
    # Sort results by final_score descending
    results.sort(key=lambda x: x.get('evaluation', {}).get('final_score', 0), reverse=True)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for res in results:
            # Reorder fields to put final_score after id
            new_res = {}
            if 'id' in res:
                new_res['id'] = res['id']
            
            # Extract final_score
            score = res.get('evaluation', {}).get('final_score', 0)
            new_res['final_score'] = score
            
            # Add other fields
            for k, v in res.items():
                if k != 'id': # final_score is usually inside evaluation, so we don't skip it here
                    new_res[k] = v
            
            f.write(json.dumps(new_res, ensure_ascii=False) + "\n")
            
    report_gen.generate_report(total_duration_seconds=total_duration)

if __name__ == "__main__":
    main()
