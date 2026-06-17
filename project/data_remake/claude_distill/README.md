# claude_distill

This folder contains directed distillation scripts for rewriting or completing consultation data with stronger teacher models.

## Purpose

The scripts here are used when ordinary rule-based conversion is not enough and a teacher model is needed to:

- rewrite final assistant turns;
- make contact-stage replies more natural;
- enforce slot/value structure;
- create directed variants such as age-aware contact data;
- produce cleaner SFT material for later filtering or RL conversion.

## Main File

- `claude_distill_contact_age_directed.py`: contact-age directed distillation script. It reads source samples, calls a configured teacher endpoint, and writes rewritten outputs for downstream checks and cleaning.

## Configuration

Use environment variables for all model credentials and endpoints. Typical variables are:

```bash
CLAUDE_DISTILL_API_KEY=...
CLAUDE_DISTILL_BASE_URL=...
CLAUDE_DISTILL_MODEL_NAME=...
CLAUDE_DISTILL_TIMEOUT=300
```

Do not hard-code real API keys in this folder. The GitHub/GitLab push protection will block committed secrets.

## Outputs

Generated JSONL outputs, raw logs, and intermediate files should be written under ignored paths such as `data_remake/outputs/`, `data_remake/logs/`, or `data_remake/runs/`.
