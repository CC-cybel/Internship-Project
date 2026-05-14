import json
from typing import List, Dict, Any, Optional
from leadbench.model.api_model import APIModel
import logging
from leadbench.generation.processors import DialogueProcessor

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(
        self,
        model: APIModel,
        processor: Optional[DialogueProcessor] = None,
        max_tokens: int = 2048
    ):
        self.model = model
        self.processor = processor
        self.max_tokens = max_tokens

    def generate_response(self, sample: Dict[str, Any]) -> str:
        """
        Generates a response for the given sample using the candidate model.
        The sample is expected to have a 'messages' field which is a list of dicts.
        """
        messages = sample.get('messages', [])
        if not messages:
            return ""

        # Pre-process messages if processor is available
        context = {'turn_id': sample.get('turn_id', 1)} # Default to turn 1 if not present
        
        processed_messages = messages
        if self.processor:
            processed_messages = self.processor.process_input(messages, context)
            
        # Store processed messages in sample for debugging/logging
        sample['processed_messages'] = processed_messages

        # Call model with processed messages
        raw_response = self.model.chat(
            processed_messages,
            max_tokens=self.max_tokens
        )
        
        # Store raw response
        sample['raw_response'] = raw_response

        # Post-process response if processor is available
        final_response = raw_response
        if self.processor:
            final_response = self.processor.process_output(raw_response, context)

        return final_response

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates response and appends it to the sample.
        """
        response = self.generate_response(sample)
        sample['response'] = response
        return sample
