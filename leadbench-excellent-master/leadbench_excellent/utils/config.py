import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Config:
    # Candidate Model Config
    CANDIDATE_MODEL_NAME = os.getenv("CANDIDATE_MODEL_NAME", "candidate-model")
    CANDIDATE_API_KEY = os.getenv("CANDIDATE_API_KEY")
    CANDIDATE_API_BASE = os.getenv("CANDIDATE_API_BASE")
    CANDIDATE_MAX_OUTPUT_TOKENS = int(os.getenv("CANDIDATE_MAX_OUTPUT_TOKENS", "2048"))
    
    # User Simulator Model Config
    USER_SIMULATOR_MODEL_NAME = os.getenv("USER_SIMULATOR_MODEL_NAME", "qwen3.5-397b-a17b")
    USER_SIMULATOR_API_KEY = os.getenv("USER_SIMULATOR_API_KEY")
    USER_SIMULATOR_API_BASE = os.getenv("USER_SIMULATOR_API_BASE")
    
    # Judge Model Config
    JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "judge-model")
    JUDGE_API_KEY = os.getenv("JUDGE_API_KEY")
    JUDGE_API_BASE = os.getenv("JUDGE_API_BASE")
    JUDGE_ENABLE_THINKING = os.getenv("JUDGE_ENABLE_THINKING", "false").lower() == "true"
    
    # Dataset Config
    INPUT_FILE = os.getenv("INPUT_FILE", "./data/dataset/golden_history_input.jsonl")
    RULES_FILE = os.getenv("RULES_FILE", "./data/rules/leadbench_rule.json")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "./output/evaluation_results.jsonl")
    
    # Evaluation Config
    EVALUATION_LIMIT = int(os.getenv("EVALUATION_LIMIT")) if os.getenv("EVALUATION_LIMIT") else None

    # Generation Config
    CONCURRENCY = int(os.getenv("CONCURRENCY", "5"))

config = Config()
