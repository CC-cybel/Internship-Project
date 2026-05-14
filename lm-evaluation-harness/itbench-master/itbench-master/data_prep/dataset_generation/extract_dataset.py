import json
import os
import argparse
from pathlib import Path

def extract_golden_history(input_file: str, output_file: str):
    """
    Extract data from generated_golden_history.jsonl and convert it to 
    golden_history_input.jsonl format suitable for testing.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # Create target directory if it doesn't exist
    os.makedirs(output_path.parent, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        return
        
    success_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
         
        for line in f_in:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # Check if it was a successful generation
                if "error" in data.get("system_prompt_generation_result", {}):
                    print(f"Skipping sample {data.get('key')} due to generation error.")
                    continue
                    
                # We need to construct the input format
                # Usually it contains: key, messages (up to the last user message)
                # The generated data has complete multi-turn interaction in "messages"
                # The generator in itbench expects input to have 'messages' that ends with the user's turn
                # The generated data has messages ending with a user turn already.
                
                # Extract essential fields
                extracted_data = {
                    "key": data.get("key"),
                    "messages": data.get("messages", []),
                    "rule_list": data.get("rule_list", [])
                }
                
                f_out.write(json.dumps(extracted_data, ensure_ascii=False) + "\n")
                success_count += 1
                
            except json.JSONDecodeError:
                print("Warning: Skipped a line due to JSON decode error.")
                continue
                
    print(f"Successfully extracted {success_count} records to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract generated golden history to dataset format.")
    parser.add_argument('--domain', type=str, default='psychiatry', help="The domain name to extract (e.g., psychiatry, douyin_dentistry)")
    parser.add_argument('--input_file', type=str, default="", help="Optional generated_golden_history JSONL path")
    parser.add_argument('--output_file', type=str, default="", help="Optional output golden_history_input JSONL path")
    args = parser.parse_args()

    domain = args.domain
    input_file = args.input_file or f"data_prep/data/{domain}/generated_golden_history.jsonl"
    output_file = args.output_file or f"data/dataset/{domain}/golden_history_input.jsonl"
    
    extract_golden_history(input_file, output_file)
