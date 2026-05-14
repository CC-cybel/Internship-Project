import argparse
import os
import sys
from types import SimpleNamespace


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


STEP_ORDER = ["reverse", "mask", "inject", "rewrite", "judge"]
REQUIRED_FLAG_BY_STEP = {
    "mask": "_reverse_done",
    "inject": "_reverse_done",
    "rewrite": "_reverse_done",
    "judge": "_rewrite_done",
}
DONE_FLAG_BY_STEP = {
    "reverse": "_reverse_done",
    "rewrite": "_rewrite_done",
    "judge": "_judge_done",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run data remake pipeline with resume support.")
    parser.add_argument("--input", required=True, help="Input JSON file for the start step.")
    parser.add_argument("--start", choices=STEP_ORDER, default="reverse", help="Step to start from.")
    parser.add_argument("--stop", choices=STEP_ORDER, default="judge", help="Step to stop at.")
    parser.add_argument("--out-dir", default=None, help="Output directory for all step artifacts.")
    parser.add_argument("--prefix", default=None, help="Prefix for output file names.")
    parser.add_argument("--action-prob", type=float, default=None, help="Injection probability for actions.")
    parser.add_argument("--mask-placeholder", default=None, help="Phone placeholder for masking.")
    parser.add_argument("--mask-roles", default=None, help="Comma-separated roles to mask (e.g. human,user).")
    parser.add_argument("--mask-landline", action="store_true", help="Also mask landline numbers.")
    parser.add_argument(
        "--require-prev-done",
        action="store_true",
        help="Require previous step done flag before proceeding.",
    )
    parser.add_argument(
        "--pipeline-report",
        default=None,
        help="Pipeline report JSON path.",
    )
    parser.add_argument("--max-workers", type=int, default=None, help="Max worker threads.")
    parser.add_argument("--save-every", type=int, default=None, help="Save output every N items.")
    parser.add_argument("--max-retries", type=int, default=None, help="Max retries for API calls.")
    parser.add_argument("--model", default=None, help="Model name override.")
    parser.add_argument("--base-url", default=None, help="Base URL override.")
    parser.add_argument("--api-key", default=None, help="API key override.")
    return parser.parse_args()


def resolve_paths(args):
    input_path = os.path.abspath(args.input)
    prefix = args.prefix or os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(SCRIPT_DIR, "runs", prefix)
    paths = {
        "reverse_out": os.path.join(out_dir, f"{prefix}_reverse.json"),
        "mask_out": os.path.join(out_dir, f"{prefix}_reverse_masked.json"),
        "inject_out": os.path.join(out_dir, f"{prefix}_action.json"),
        "rewrite_out": os.path.join(out_dir, f"{prefix}_rewrite.json"),
        "rewrite_cleaned_out": os.path.join(out_dir, f"{prefix}_rewrite_cleaned.json"),
        "judge_report": os.path.join(out_dir, f"{prefix}_judge_report.json"),
        "judge_bad": os.path.join(out_dir, f"{prefix}_judge_bad.json"),
        "judge_stats": os.path.join(out_dir, f"{prefix}_judge_stats.json"),
        "pipeline_report": os.path.join(out_dir, f"{prefix}_pipeline_report.json"),
        "cache_root": os.path.join(out_dir, f"{prefix}_cache"),
        "reverse_raw": os.path.join(out_dir, f"{prefix}_raw_reverse_prompts.txt"),
        "rewrite_raw": os.path.join(out_dir, f"{prefix}_raw_rewrite.txt"),
    }
    return input_path, out_dir, prefix, paths


def load_items(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        return data, data["items"]
    if isinstance(data, list):
        return None, data
    raise ValueError("Unsupported input format: expected list or dict with 'items'.")


def write_items(path, wrapper, items):
    import json
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if wrapper is not None:
        wrapper["items"] = items
        data = wrapper
    else:
        data = items
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def count_items(path):
    _, items = load_items(path)
    return len(items)


def count_done_flag(path, flag):
    _, items = load_items(path)
    count = 0
    for item in items:
        if isinstance(item, dict) and item.get(flag) is True:
            count += 1
    return count


def filter_items_by_flag(input_path, flag, output_path):
    wrapper, items = load_items(input_path)
    input_count = len(items)
    filtered = [item for item in items if isinstance(item, dict) and item.get(flag) is True]
    output_count = len(filtered)
    dropped = input_count - output_count

    if dropped > 0:
        write_items(output_path, wrapper, filtered)
        return output_path, {
            "input_count": input_count,
            "output_count": output_count,
            "dropped": dropped,
            "flag": flag,
            "filtered_path": output_path,
        }

    return input_path, {
        "input_count": input_count,
        "output_count": output_count,
        "dropped": 0,
        "flag": flag,
        "filtered_path": None,
    }


def write_pipeline_report(path, report):
    import json
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def run_reverse(input_path, output_path, cache_dir, raw_log, args):
    import reverse_prompts_from_dialogues as pr

    pr.INPUT_FILE = input_path
    pr.OUTPUT_FILE = output_path
    pr.RAW_LOG_FILE = raw_log
    pr.CACHE_DIR = cache_dir

    if args.max_workers is not None:
        pr.MAX_WORKERS = args.max_workers
    if args.save_every is not None:
        pr.SAVE_EVERY = args.save_every
    if args.max_retries is not None:
        pr.MAX_RETRIES = args.max_retries
    if args.model:
        pr.MODEL_NAME = args.model
    if args.base_url:
        pr.BASE_URL = args.base_url
    if args.api_key:
        pr.API_KEY = args.api_key

    pr.client = pr.OpenAI(api_key=pr.API_KEY, base_url=pr.BASE_URL)
    pr.main()


def run_inject(input_path, output_path, args):
    import inject_random_actions as inj

    inj.INPUT_FILE = input_path
    inj.OUTPUT_FILE = output_path
    if args.action_prob is not None:
        inj.TURN_INJECTION_PROB = args.action_prob
    inj.main()


def run_mask(input_path, output_path, args):
    import mask_user_phone_numbers as mp

    args_ns = SimpleNamespace(
        input=input_path,
        output=output_path,
        placeholder=args.mask_placeholder or mp.DEFAULT_PLACEHOLDER,
        roles=args.mask_roles or "human,user",
        replace_landline=bool(args.mask_landline),
    )

    original_parse_args = mp.parse_args
    mp.parse_args = lambda: args_ns
    try:
        mp.main()
    finally:
        mp.parse_args = original_parse_args


def run_rewrite(input_path, output_path, cache_dir, raw_log, args):
    import rewrite_dialogues as rw

    args_ns = SimpleNamespace(
        input=input_path,
        output=output_path,
        cache_dir=cache_dir,
        raw_log=raw_log,
        max_workers=args.max_workers or rw.MAX_WORKERS,
        save_every=args.save_every or rw.SAVE_EVERY,
        max_retries=args.max_retries or rw.MAX_RETRIES,
        model=args.model or rw.MODEL_NAME,
        base_url=args.base_url or rw.BASE_URL,
        api_key=args.api_key or rw.API_KEY,
    )

    original_parse_args = rw.parse_args
    rw.parse_args = lambda: args_ns
    try:
        rw.main()
    finally:
        rw.parse_args = original_parse_args


def run_symbol_clean(input_path, output_path):
    import clean_response_symbols as cs

    data = cs.load_data(input_path)
    items = cs.get_items(data)
    changed_items, changed_turns = cs.clean_items(items)
    cs.write_data(output_path, data)

    print("✅ Symbol clean finished")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Changed items: {changed_items}")
    print(f"Changed turns: {changed_turns}")


def run_judge(input_path, report_path, bad_path, stats_path, cache_dir, args):
    import judge_rewrite_quality as jd

    args_ns = SimpleNamespace(
        input=input_path,
        report=report_path,
        bad_case=bad_path,
        stats=stats_path,
        cache_dir=cache_dir,
        max_workers=args.max_workers or jd.MAX_WORKERS,
        save_every=args.save_every or jd.SAVE_EVERY,
        max_retries=args.max_retries or jd.MAX_RETRIES,
        model=args.model or jd.MODEL_NAME,
        base_url=args.base_url or jd.BASE_URL,
        api_key=args.api_key or jd.API_KEY,
    )

    original_parse_args = jd.parse_args
    jd.parse_args = lambda: args_ns
    try:
        jd.main()
    finally:
        jd.parse_args = original_parse_args


def main():
    args = parse_args()

    if args.start not in STEP_ORDER or args.stop not in STEP_ORDER:
        raise ValueError("Invalid start/stop step.")
    if STEP_ORDER.index(args.start) > STEP_ORDER.index(args.stop):
        raise ValueError("start step must not be after stop step.")

    input_path, out_dir, prefix, paths = resolve_paths(args)
    pipeline_report_path = args.pipeline_report or paths["pipeline_report"]

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(paths["cache_root"], exist_ok=True)

    print("🔧 Pipeline config")
    print(f"Start: {args.start}")
    print(f"Stop: {args.stop}")
    print(f"Input: {input_path}")
    print(f"Out dir: {out_dir}")
    print(f"Prefix: {prefix}")
    print(f"Require prev done: {args.require_prev_done}")
    print(f"Report: {pipeline_report_path}")

    current_input = input_path
    start_index = STEP_ORDER.index(args.start)
    stop_index = STEP_ORDER.index(args.stop)
    original_count = count_items(current_input)
    report = {
        "require_prev_done": args.require_prev_done,
        "original_count": original_count,
        "steps": [],
        "output_paths": {
            "reverse_out": paths["reverse_out"],
            "mask_out": paths["mask_out"],
            "inject_out": paths["inject_out"],
            "rewrite_out": paths["rewrite_out"],
            "rewrite_cleaned_out": paths["rewrite_cleaned_out"],
            "judge_report": paths["judge_report"],
            "judge_bad": paths["judge_bad"],
            "judge_stats": paths["judge_stats"],
        },
    }
    final_dataset_path = current_input
    last_dataset_step = None

    for step in STEP_ORDER[start_index:stop_index + 1]:
        if args.require_prev_done and step in REQUIRED_FLAG_BY_STEP:
            required_flag = REQUIRED_FLAG_BY_STEP[step]
            filtered_path = os.path.join(
                out_dir,
                f"{prefix}_{step}_filtered.json",
            )
            current_input, stats = filter_items_by_flag(current_input, required_flag, filtered_path)
            report["steps"].append({
                "step": step,
                "required_flag": required_flag,
                **stats,
            })
            if stats["output_count"] == 0:
                report["final_input_count"] = 0
                report["final_done_count"] = 0
                write_pipeline_report(pipeline_report_path, report)
                raise ValueError(
                    f"No items with required flag '{required_flag}' before step '{step}'."
                )
        if step == "reverse":
            print("\n▶ Step: reverse_prompts_from_dialogues")
            run_reverse(
                current_input,
                paths["reverse_out"],
                os.path.join(paths["cache_root"], "reverse_prompts"),
                paths["reverse_raw"],
                args,
            )
            current_input = paths["reverse_out"]
            final_dataset_path = current_input
            last_dataset_step = step
        elif step == "mask":
            print("\n▶ Step: mask_user_phone_numbers")
            run_mask(current_input, paths["mask_out"], args)
            current_input = paths["mask_out"]
            final_dataset_path = current_input
            last_dataset_step = step
        elif step == "inject":
            print("\n▶ Step: inject_random_actions")
            run_inject(current_input, paths["inject_out"], args)
            current_input = paths["inject_out"]
            final_dataset_path = current_input
            last_dataset_step = step
        elif step == "rewrite":
            print("\n▶ Step: rewrite_dialogues")
            run_rewrite(
                current_input,
                paths["rewrite_out"],
                os.path.join(paths["cache_root"], "rewrite"),
                paths["rewrite_raw"],
                args,
            )
            current_input = paths["rewrite_out"]
            final_dataset_path = current_input
            last_dataset_step = step
        elif step == "judge":
            print("\n▶ Step: clean_response_symbols")
            run_symbol_clean(current_input, paths["rewrite_cleaned_out"])
            current_input = paths["rewrite_cleaned_out"]
            final_dataset_path = current_input

            print("\n▶ Step: judge_rewrite_quality")
            run_judge(
                current_input,
                paths["judge_report"],
                paths["judge_bad"],
                paths["judge_stats"],
                os.path.join(paths["cache_root"], "judge"),
                args,
            )

    report["final_input_count"] = count_items(final_dataset_path)
    done_flag = DONE_FLAG_BY_STEP.get(last_dataset_step)
    if done_flag and os.path.exists(final_dataset_path):
        report["final_done_count"] = count_done_flag(final_dataset_path, done_flag)
    else:
        report["final_done_count"] = report["final_input_count"]
    report["final_dataset_path"] = final_dataset_path
    report["last_dataset_step"] = last_dataset_step
    report["pipeline_report_path"] = pipeline_report_path
    write_pipeline_report(pipeline_report_path, report)

    print("\n✅ Pipeline finished")
    print(f"Last output: {current_input}")
    print(f"Judge report: {paths['judge_report']}")
    print(f"Bad cases: {paths['judge_bad']}")
    print(f"Stats: {paths['judge_stats']}")
    print(f"Original count: {report['original_count']}")
    print(f"Final count: {report['final_input_count']}")
    print(f"Final done count: {report['final_done_count']}")
    print(f"Pipeline report: {pipeline_report_path}")


if __name__ == "__main__":
    main()
