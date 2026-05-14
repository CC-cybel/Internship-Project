import argparse
import json
from pathlib import Path


def _build_prompt(item: dict, prompt_key: str) -> list[dict]:
    prompt = item.get(prompt_key)
    if isinstance(prompt, list):
        return prompt
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    raise ValueError(f"Invalid prompt format: {type(prompt)}")


def _build_row(item: dict, prompt_key: str) -> dict:
    prompt = _build_prompt(item, prompt_key)
    meta = item.get("meta", {})
    if not isinstance(meta, dict):
        meta = {"raw_meta": meta}
    # pyarrow cannot write an empty struct to parquet.
    if not meta:
        meta = {"_": ""}
    return {
        "data_source": item.get("data_source", "opd_custom"),
        "prompt": prompt,
        "ability": item.get("ability", "general"),
        "reward_model": {
            "style": "rule",
            "ground_truth": item.get("ground_truth", ""),
        },
        "extra_info": {
            "id": item.get("id"),
            "meta": meta,
        },
    }


def convert(input_path: Path, output_path: Path, prompt_key: str) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`pyarrow` is required for parquet conversion. Please install it first, e.g. `pip install pyarrow`."
        ) from exc

    rows = []
    with input_path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("Each line must be a JSON object.")
            rows.append(_build_row(item, prompt_key=prompt_key))

    table = pa.Table.from_pylist(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OPD jsonl to verl RL parquet format.")
    parser.add_argument("--input", required=True, help="Input .jsonl path")
    parser.add_argument("--output", required=True, help="Output .parquet path")
    parser.add_argument("--prompt-key", default="prompt", help="Prompt field key in input jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert(Path(args.input), Path(args.output), prompt_key=args.prompt_key)
    print(f"[DONE] Converted {count} rows to {args.output}")


if __name__ == "__main__":
    main()
