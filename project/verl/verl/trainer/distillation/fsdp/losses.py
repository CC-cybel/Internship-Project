# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def reverse_kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute reverse KL on a shared top-k support: KL(q || p)."""
    log_p = log_p.float()
    log_q = log_q.float()
    q = log_q.exp()
    kld = q * (log_q - log_p)
    return kld.sum(dim=-1)


def renormalize_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Renormalize log probabilities over the last dimension."""
    return log_probs.float() - torch.logsumexp(log_probs.float(), dim=-1, keepdim=True)


def compute_topk_alignment_metrics(
    student_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute token-set overlap metrics between student and teacher top-k distributions."""
    with torch.no_grad():
        topk = teacher_topk_ids.shape[-1]
        student_topk_log_probs, student_topk_ids = torch.topk(student_log_probs, k=topk, dim=-1)

        overlap = student_topk_ids.unsqueeze(-1).eq(teacher_topk_ids.unsqueeze(-2))
        overlap_mask = overlap.any(dim=-1)
        overlap_count = overlap_mask.sum(dim=-1)
        overlap_rate = overlap_count.float() / float(topk)

        matched_teacher_log_probs = (
            overlap.to(teacher_topk_log_probs.dtype) * teacher_topk_log_probs.unsqueeze(-2)
        ).sum(dim=-1)

        overlap_float = overlap_mask.to(student_topk_log_probs.dtype)
        student_overlap_probs = student_topk_log_probs.exp() * overlap_float
        teacher_overlap_probs = matched_teacher_log_probs.exp() * overlap_float
        student_overlap_mass = student_overlap_probs.sum(dim=-1)
        teacher_overlap_mass = teacher_overlap_probs.sum(dim=-1)

        has_overlap = overlap_count > 0
        safe_student_mass = student_overlap_mass.clamp_min(torch.finfo(student_overlap_mass.dtype).tiny)
        safe_teacher_mass = teacher_overlap_mass.clamp_min(torch.finfo(teacher_overlap_mass.dtype).tiny)

        student_overlap_log_probs = student_topk_log_probs.float() - safe_student_mass.log().unsqueeze(-1)
        teacher_overlap_log_probs = matched_teacher_log_probs.float() - safe_teacher_mass.log().unsqueeze(-1)
        student_overlap_probs = student_overlap_log_probs.exp() * overlap_float

        advantage_terms = student_overlap_probs * (teacher_overlap_log_probs - student_overlap_log_probs)
        overlap_token_advantage = advantage_terms.sum(dim=-1) / overlap_count.clamp_min(1).float()
        overlap_token_advantage = torch.where(
            has_overlap, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
        )

    return {
        "topk_overlap_rate": overlap_rate,
        "topk_overlap_count": overlap_count.float(),
        "overlap_token_advantage": overlap_token_advantage,
    }


def gather_teacher_log_probs(
    teacher_log_probs: torch.Tensor,
    teacher_ids: torch.Tensor,
    target_ids: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Gather teacher logprobs at target token ids from returned prompt_logprobs."""
    teacher_k = teacher_ids.shape[-1]
    if teacher_k == vocab_size:
        dense_teacher_log_probs = torch.full(
            teacher_log_probs.shape[:2] + (vocab_size,),
            torch.nan,
            dtype=teacher_log_probs.dtype,
            device=teacher_log_probs.device,
        )
        dense_teacher_log_probs.scatter_(dim=-1, index=teacher_ids.long(), src=teacher_log_probs)
        gathered = torch.gather(dense_teacher_log_probs, dim=-1, index=target_ids.long())
    else:
        matches = target_ids.unsqueeze(-1).eq(teacher_ids.unsqueeze(-2))
        gathered = (matches.to(teacher_log_probs.dtype) * teacher_log_probs.unsqueeze(-2)).sum(dim=-1)
        gathered = torch.where(matches.any(dim=-1), gathered, torch.full_like(gathered, torch.nan))

    if torch.isnan(gathered).any():
        raise RuntimeError(
            "Teacher prompt_logprobs did not cover all student top-k tokens required by "
            "reverse_kl_student_topk. Use vLLM prompt_logprobs=-1/max_logprobs=-1 for exact student-top-k "
            "distillation, or increase distillation.distillation_loss.teacher_prompt_logprobs."
        )
    return gathered


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    alignment_metrics = compute_topk_alignment_metrics(
        student_log_probs=student_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
        teacher_topk_ids=teacher_topk_ids,
    )
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    outputs = {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
    }
    outputs.update(alignment_metrics)
    return outputs


def compute_reverse_kl_student_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reverse KL on the student's top-k support."""
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, teacher_k)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, teacher_k)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    loss_config: DistillationLossConfig = config.distillation_loss
    topk = loss_config.topk
    if topk is None:
        raise ValueError("reverse_kl_student_topk requires distillation.distillation_loss.topk to be set.")

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs, student_topk_ids = torch.topk(student_log_probs, k=topk, dim=-1)
    teacher_student_topk_log_probs = gather_teacher_log_probs(
        teacher_log_probs=teacher_topk_log_probs,
        teacher_ids=teacher_topk_ids,
        target_ids=student_topk_ids,
        vocab_size=student_logits.shape[-1],
    )
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_student_topk_log_probs.exp().sum(dim=-1)
    teacher_alignment_k = min(topk, teacher_topk_ids.shape[-1])
    alignment_metrics = compute_topk_alignment_metrics(
        student_log_probs=student_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs[..., :teacher_alignment_k],
        teacher_topk_ids=teacher_topk_ids[..., :teacher_alignment_k],
    )

    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_student_topk_log_probs = teacher_student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    student_topk_log_probs_norm = renormalize_log_probs(student_topk_log_probs)
    teacher_student_topk_log_probs_norm = renormalize_log_probs(teacher_student_topk_log_probs)
    distillation_losses = reverse_kl_divergence(
        log_q=student_topk_log_probs_norm,
        log_p=teacher_student_topk_log_probs_norm,
    )

    outputs = {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
    }
    outputs.update(alignment_metrics)
    return outputs


def compute_reverse_kl_student_topk_gather(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reverse KL on a precomputed student top-k support."""
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids.long())
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    alignment_metrics = compute_topk_alignment_metrics(
        student_log_probs=student_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
        teacher_topk_ids=teacher_topk_ids,
    )

    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    student_topk_log_probs_norm = renormalize_log_probs(student_topk_log_probs)
    teacher_topk_log_probs_norm = renormalize_log_probs(teacher_topk_log_probs)
    distillation_losses = reverse_kl_divergence(
        log_q=student_topk_log_probs_norm,
        log_p=teacher_topk_log_probs_norm,
    )

    outputs = {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
    }
    outputs.update(alignment_metrics)
    return outputs


def compute_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reverse KL distillation loss using teacher top-k log probabilities.

    This keeps v2's top-k support but flips the KL direction from
    KL(teacher || student) to KL(student || teacher) on the teacher top-k ids.
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    alignment_metrics = compute_topk_alignment_metrics(
        student_log_probs=student_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
        teacher_topk_ids=teacher_topk_ids,
    )
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = reverse_kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    outputs = {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
    }
    outputs.update(alignment_metrics)
    return outputs
