from typing import Dict, List, Any
import json
import os
import re

class DialogueEvaluationDataset:
    def __init__(self, file_path: str, rules_file: str = None):
        self.file_path = file_path
        self.rules_file = rules_file
        self.data = self._load_data()
        self.rules = self._load_rules() if rules_file else []

    def _load_data(self) -> List[Dict[str, Any]]:
        data = []
        if not os.path.exists(self.file_path):
            # If file doesn't exist, return empty list (might be creating it)
            return []
            
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    data.append(sample)
                except json.JSONDecodeError:
                    continue
        return data

    def _load_rules(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.rules_file):
            return []
        with open(self.rules_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Prepare data for evaluation
        # If 'response' is already present (from generation step)
        response = sample.get('response', "")
        
        # Format history string for the prompt
        # Assuming 'messages' is the standard format now
        history_str = ""
        messages = [{"role": msg.get("role"), "content": msg.get("content")} for msg in sample.get('messages', [])]
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            history_str += f"{role}: {content}\n"
        
        # Fallback to 'dialogue' if messages missing (legacy support)
        if not history_str and 'dialogue' in sample:
            for turn in sample['dialogue'][:-1]:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                history_str += f"{role}: {content}\n"

        return {
            "id": sample.get('key', sample.get('id')), # 'key' from golden_history_input
            "history_str": history_str,
            "response": response,
            "rules": self.rules,
            "original_data": sample,
            "messages": messages, # Explicitly include messages for output
        }
