import re
from typing import List, Dict, Any, Callable, Optional

class DialogueProcessor:
    """
    Base class for dialogue processing (pre-processing and post-processing).
    """
    def __init__(self):
        self.pre_processors: List[Callable[[List[Dict[str, str]], Dict[str, Any]], List[Dict[str, str]]]] = []
        self.post_processors: List[Callable[[str, Dict[str, Any]], str]] = []

    def add_pre_processor(self, func: Callable):
        self.pre_processors.append(func)

    def add_post_processor(self, func: Callable):
        self.post_processors.append(func)

    def process_input(self, messages: List[Dict[str, str]], context: Dict[str, Any] = None) -> List[Dict[str, str]]:
        processed_messages = [m.copy() for m in messages]
        for func in self.pre_processors:
            processed_messages = func(processed_messages, context or {})
        return processed_messages

    def process_output(self, response: str, context: Dict[str, Any] = None) -> str:
        processed_response = response
        for func in self.post_processors:
            processed_response = func(processed_response, context or {})
        return processed_response

# --- Specific Processors ---

def add_system_instruction(messages: List[Dict[str, str]], context: Dict[str, Any], instruction: str) -> List[Dict[str, str]]:
    """
    Appends instruction to the system prompt.
    If no system prompt exists, creates one at the beginning.
    """
    new_messages = []
    has_system = False
    for msg in messages:
        if msg['role'] == 'system':
            new_msg = msg.copy()
            new_msg['content'] += "\n" + instruction
            new_messages.append(new_msg)
            has_system = True
        else:
            new_messages.append(msg)
    
    if not has_system:
        new_messages.insert(0, {"role": "system", "content": instruction})
        
    return new_messages

def replace_system_instruction(messages: List[Dict[str, str]], context: Dict[str, Any], instruction: str) -> List[Dict[str, str]]:
    """
    Replaces the system prompt with the given instruction.
    If no system prompt exists, creates one at the beginning.
    """
    new_messages = []
    has_system = False
    for msg in messages:
        if msg['role'] == 'system':
            new_msg = msg.copy()
            new_msg['content'] = instruction
            new_messages.append(new_msg)
            has_system = True
        else:
            new_messages.append(msg)
    
    if not has_system:
        new_messages.insert(0, {"role": "system", "content": instruction})
        
    return new_messages

def append_turn_info(messages: List[Dict[str, str]], context: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Appends [系统信息：当前第X轮] to ALL user messages, counting from 1.
    """
    new_messages = []
    user_turn_count = 0
    
    for msg in messages:
        if msg['role'] == 'user':
            user_turn_count += 1
            new_msg = msg.copy()
            new_msg['content'] += f"【系统信息：当前第{user_turn_count}轮】"
            new_messages.append(new_msg)
        else:
            new_messages.append(msg)
            
    return new_messages

def extract_final_response(response: str, context: Dict[str, Any]) -> str:
    """
    Extracts content between BEGIN_FINAL and END_FINAL.
    If tags are not found, returns the original response.
    """
    pattern = r"BEGIN_FINAL\s*(.*?)\s*END_FINAL"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()
