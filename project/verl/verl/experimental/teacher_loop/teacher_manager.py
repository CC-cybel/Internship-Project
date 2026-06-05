# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.nn import functional as F

from verl.experimental.agent_loop import AsyncLLMServerManager
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import DistillationConfig, DistillationLossConfig


def _get_teacher_sampling_params(
    distillation_config: DistillationConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if distillation_config.teacher_model.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    if distillation_loss_config.loss_settings.use_topk:
        if distillation_loss_config.loss_mode == "reverse_kl_student_topk_gather":
            num_logprobs = distillation_loss_config.topk
        else:
            num_logprobs = distillation_loss_config.teacher_prompt_logprobs
            if num_logprobs is None:
                num_logprobs = (
                    -1 if distillation_loss_config.loss_mode == "reverse_kl_student_topk" else distillation_loss_config.topk
                )
    else:
        num_logprobs = 0
    return {
        "max_tokens": 1,
        "temperature": distillation_config.teacher_model.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


def _unpad_teacher_inputs(data: DataProto) -> tuple[list[int], int, int]:
    """Unpad valid sequence ids and prompt/response lengths from a single sample.
    The sample is a left-padded prompt concatenated with a right-padded response.
    TODO(wuxibin): remove padding and use tensordict.
    """
    assert len(data) == 1, "Teacher logprob computation expects a single sample"

    input_ids = data.batch["input_ids"][0]
    attention_mask = data.batch["attention_mask"][0]
    prompt_width = data.batch["prompts"][0].shape[0]
    response_width = data.batch["responses"][0].shape[0]
    assert attention_mask.shape[0] == prompt_width + response_width, (
        "attention_mask sequence length must match prompt and response widths"
    )
    valid_prompt_length = int(attention_mask[:prompt_width].sum().item())
    valid_response_length = int(attention_mask[-response_width:].sum().item())
    prompt_num_padding = prompt_width - valid_prompt_length
    sequence_ids = input_ids[prompt_num_padding : prompt_width + valid_response_length]
    sequence_ids = normalize_token_ids(sequence_ids)
    return sequence_ids, valid_prompt_length, valid_response_length


class AsyncTeacherLLMServerManager(AsyncLLMServerManager):
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        distillation_config: DictConfig | DistillationConfig,
        pad_token_id: int,
        route_name: str | None = None,
    ):
        super().__init__(config=config, servers=servers, load_balancer_handle=load_balancer_handle)
        if isinstance(distillation_config, DistillationConfig):
            self.distillation_config = distillation_config
        else:
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(distillation_config)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.pad_token_id = pad_token_id
        self.route_name = route_name

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        target_topk_ids: Optional[list[list[int]]] = None,
        prompt_length: Optional[int] = None,
        response_length: Optional[int] = None,
        multi_modal_data: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        prompt_target_token_ids = None
        if target_topk_ids is not None:
            if prompt_length is None or response_length is None:
                raise ValueError("target_topk_ids requires prompt_length and response_length.")
            topk = len(target_topk_ids[0]) if target_topk_ids else 0
            dummy_target_ids = list(range(topk))
            prompt_target_token_ids = [dummy_target_ids[:] for _ in range(max(len(sequence_ids) - 1, 0))]
            start = max(prompt_length - 1, 0)
            for j in range(response_length):
                if start + j < len(prompt_target_token_ids):
                    prompt_target_token_ids[start + j] = [int(x) for x in target_topk_ids[j]]
        teacher_output = await self.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(self.distillation_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            prompt_target_token_ids=prompt_target_token_ids,
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        return teacher_ids, teacher_logprobs

    async def compute_teacher_logprobs_batch(self, data: DataProto) -> DataProto:
        """Compute teacher log probabilities for a batch of prompt-response pairs."""
        multi_modal_data_batch = data.non_tensor_batch.get("teacher_multi_modal_data")
        tasks = []
        lengths = []
        prompt_width = data.batch["prompts"].shape[1]
        response_width = data.batch["responses"].shape[1]

        # Compute logprobs for each sample in the batch
        for i in range(len(data)):
            item = data[i : i + 1]
            sequence_ids, prompt_length, response_length = _unpad_teacher_inputs(item)
            multi_modal_data = None if multi_modal_data_batch is None else multi_modal_data_batch[i]
            target_topk_ids = None
            if "student_topk_ids" in data.batch:
                # student_topk_ids is already sliced to response logits by no_padding_2_padding.
                target_topk_ids = data.batch["student_topk_ids"][i, :response_length].tolist()
            lengths.append((prompt_length, response_length))
            tasks.append(
                asyncio.create_task(
                    self.compute_teacher_logprobs_single(
                        sequence_ids=sequence_ids,
                        target_topk_ids=target_topk_ids,
                        prompt_length=prompt_length,
                        response_length=response_length,
                        multi_modal_data=multi_modal_data,
                    )
                )
            )
        outputs = await asyncio.gather(*tasks)

        # Pad the teacher logprobs and ids
        padded_teacher_ids = []
        padded_teacher_logprobs = []
        for (teacher_ids, teacher_logprobs), (prompt_length, response_length) in zip(outputs, lengths, strict=True):
            padded_ids, padded_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_width,
                response_width=response_width,
                prompt_length=prompt_length,
                response_length=response_length,
                pad_token_id=self.pad_token_id,
            )
            padded_teacher_ids.append(padded_ids)
            padded_teacher_logprobs.append(padded_logprobs)

        batch = TensorDict(
            {
                "teacher_ids": torch.cat(padded_teacher_ids),
                "teacher_logprobs": torch.cat(padded_teacher_logprobs),
            },
            batch_size=len(data),
        )
        return DataProto(batch=batch)

    async def compute_teacher_logprobs_batch_by_route(
        self,
        data: DataProto,
        route_to_manager: dict[str, "AsyncTeacherLLMServerManager"],
    ) -> DataProto:
        """Compute teacher logprobs with per-sample teacher routing.

        The input batch must carry non_tensor_batch["teacher_route"], whose value
        selects the named teacher manager for each sample.
        """
        route_values = data.non_tensor_batch.get("teacher_route")
        if route_values is None:
            raise ValueError("Missing non_tensor_batch['teacher_route'] for multi-teacher distillation.")

        tasks = []
        lengths = []
        managers = []
        multi_modal_data_batch = data.non_tensor_batch.get("teacher_multi_modal_data")
        prompt_width = data.batch["prompts"].shape[1]
        response_width = data.batch["responses"].shape[1]

        for i in range(len(data)):
            route_name = str(route_values[i])
            manager = route_to_manager.get(route_name)
            if manager is None:
                known_routes = sorted(route_to_manager)
                raise ValueError(f"Unknown teacher_route={route_name!r}; expected one of {known_routes}.")
            item = data[i : i + 1]
            sequence_ids, prompt_length, response_length = _unpad_teacher_inputs(item)
            multi_modal_data = None if multi_modal_data_batch is None else multi_modal_data_batch[i]
            target_topk_ids = None
            if "student_topk_ids" in data.batch:
                # student_topk_ids is already sliced to response logits by no_padding_2_padding.
                target_topk_ids = data.batch["student_topk_ids"][i, :response_length].tolist()
            lengths.append((prompt_length, response_length))
            managers.append(manager)
            tasks.append(
                asyncio.create_task(
                    manager.compute_teacher_logprobs_single(
                        sequence_ids=sequence_ids,
                        target_topk_ids=target_topk_ids,
                        prompt_length=prompt_length,
                        response_length=response_length,
                        multi_modal_data=multi_modal_data,
                    )
                )
            )

        outputs = await asyncio.gather(*tasks)

        padded_teacher_ids = []
        padded_teacher_logprobs = []
        for (teacher_ids, teacher_logprobs), (prompt_length, response_length), manager in zip(
            outputs, lengths, managers, strict=True
        ):
            padded_ids, padded_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_width,
                response_width=response_width,
                prompt_length=prompt_length,
                response_length=response_length,
                pad_token_id=manager.pad_token_id,
            )
            padded_teacher_ids.append(padded_ids)
            padded_teacher_logprobs.append(padded_logprobs)

        batch = TensorDict(
            {
                "teacher_ids": torch.cat(padded_teacher_ids),
                "teacher_logprobs": torch.cat(padded_teacher_logprobs),
            },
            batch_size=len(data),
        )
        return DataProto(batch=batch)
