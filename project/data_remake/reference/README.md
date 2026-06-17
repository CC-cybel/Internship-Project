# reference

`reference/` stores reference material used by data-generation and rewriting scripts.

## Intended Contents

Use this folder for small, human-readable reference files such as:

- prompt-writing notes;
- schema examples;
- formatting conventions;
- rubric excerpts;
- short examples that help scripts or future maintainers understand the target style.

## What Not To Put Here

Do not put large raw datasets, generated JSONL files, model outputs, or logs here. Those should go into ignored folders such as `raw/`, `outputs/`, `runs/`, `logs/`, or external storage.

## Relationship to `data_remake/`

Scripts in the parent folder may use this directory as a stable source of instructions or examples. Keep files compact and explicit so the data pipeline remains easy to audit.
