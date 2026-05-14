#!/usr/bin/env python3
"""
Merge FSDP sharded checkpoint to single-card HuggingFace format.
"""

import sys
sys.path.insert(0, '/data/chengch/project/verl')

import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

checkpoint_dir = Path('/data1/chengch/verl_outputs/grpo_single_turn/qwen3_8b_4gpu_stage5_mid_reward_20260416_134908/global_step_200/actor')
world_size = 4

print("Loading rank 0 to get keys...")
rank0_state = torch.load(checkpoint_dir / f'model_world_size_{world_size}_rank_0.pt', map_location='cpu', weights_only=False)
keys = list(rank0_state.keys())
print(f"Found {len(keys)} keys")

print("Loading all rank state dicts...")
all_state_dicts = [None] * world_size

def load_rank(rank: int) -> dict:
    path = checkpoint_dir / f'model_world_size_{world_size}_rank_{rank}.pt'
    return torch.load(path, map_location='cpu', weights_only=False)

with ThreadPoolExecutor(max_workers=world_size) as executor:
    futures = {executor.submit(load_rank, r): r for r in range(world_size)}
    for future in tqdm(as_completed(futures), total=world_size, desc="Loading shards"):
        rank = futures[future]
        all_state_dicts[rank] = future.result()

print("Merging sharded tensors...")
merged_state_dict = {}

for key in tqdm(keys, desc="Merging"):
    local_tensors = []
    for rank_state in all_state_dicts:
        tensor = rank_state[key]
        if hasattr(tensor, '_local_tensor'):
            local_tensors.append(tensor._local_tensor)
        else:
            local_tensors.append(tensor)

    first_tensor = all_state_dicts[0][key]
    if hasattr(first_tensor, '_local_tensor'):
        merged_state_dict[key] = torch.cat(local_tensors, dim=0).bfloat16()
    else:
        merged_state_dict[key] = local_tensors[0]

print(f"Merged {len(merged_state_dict)} parameters")

target_dir = Path('/data/chengch/merged_model_qwen3_8b_stage5_mid_reward_step200')
target_dir.mkdir(exist_ok=True)
# Clean up previous run if exists
for f in target_dir.glob('model-*.safetensors'):
    f.unlink()
if (target_dir / 'model.safetensors.index.json').exists():
    (target_dir / 'model.safetensors.index.json').unlink()

print("Loading model config...")
config = AutoConfig.from_pretrained(
    checkpoint_dir / 'huggingface',
    trust_remote_code=True,
    local_files_only=True
)

print("Creating model...")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
model.to_empty(device='cpu')

# Save as single safetensors directly without save_pretrained
from safetensors.torch import save_file
print(f"Saving as single safetensors to {target_dir}...")
target_dir.mkdir(exist_ok=True)
single_path = target_dir / 'model.safetensors'
save_file(merged_state_dict, single_path)
print(f"Saved single safetensors to {single_path}")

import shutil
tokenizer_src = checkpoint_dir / 'huggingface'
for f in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
          'chat_template.jinja', 'generation_config.json', 'config.json', 'vocab.json']:
    src = tokenizer_src / f
    if src.exists():
        shutil.copy2(src, target_dir / f)

print(f"Done! Model saved to: {target_dir}")
