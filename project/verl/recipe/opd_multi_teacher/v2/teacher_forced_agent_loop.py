from __future__ import annotations

import logging
import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("teacher_forced_agent")
class TeacherForcedAgentLoop(AgentLoopBase):
    """Use a pre-generated teacher response as the rollout response.

    This loop is intended for supervised forward-KL/top-k distillation. The data
    row must carry a `teacher_response` field. verl still runs the usual actor
    update and teacher logprob path after this loop returns, but the response
    tokens come from the routed teacher instead of the student rollout model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        del sampling_params
        messages = list(kwargs["raw_prompt"])
        teacher_response = kwargs.get("teacher_response")
        if teacher_response is None:
            extra_info = kwargs.get("extra_info") or {}
            teacher_response = extra_info.get("teacher_response")
        if teacher_response is None:
            raise ValueError("TeacherForcedAgentLoop requires `teacher_response` in the dataset row or extra_info.")

        multi_modal_data = await self.process_vision_info(messages)
        prompt_ids = await self.apply_chat_template(
            messages,
            images=multi_modal_data.get("images"),
            videos=multi_modal_data.get("videos"),
        )

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            response_ids = self.tokenizer.encode(str(teacher_response), add_special_tokens=False)
            if self.tokenizer.eos_token_id is not None:
                response_ids = response_ids + [self.tokenizer.eos_token_id]
            response_ids = response_ids[: self.response_length]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=None,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "teacher_forced": True,
                "teacher_route": kwargs.get("teacher_route", "unknown"),
            },
        )
        return output
