#!/usr/bin/env python3
"""Evaluate ToolFormerMicro on tool-calling benchmarks.

Measures:
- Tool Selection Accuracy (TSA): % of correct tool selections
- Parameter F1: F1 on parameter name-value matching
- Exact Match: % of exactly correct responses
- False Positive Rate: non-tool queries incorrectly generating tool calls
- False Negative Rate: tool queries failing to generate tool calls
- Encode time: time to encode tool schemas into gist vectors
- Generate time: time to generate response

Usage:
  python scripts/eval/eval_tool_former.py --checkpoint checkpoints/tool_former/stage2
  python scripts/eval/eval_tool_former.py --checkpoint checkpoints/tool_former/stage2 --split test_seen
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tool_former import ToolFormerMicro, SYSTEM_PROMPT
from src.tool_former_config import ToolFormerConfig


def parse_function_call(text: str) -> dict | None:
    """Parse a function call from model output.

    Handles formats:
    - <functioncall> {"name": "...", "arguments": '...'} <|endoftext|>
    - <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    - Raw JSON: {"name": "...", ...}
    """
    text = text.strip()

    # Try <functioncall> tag (greedy to handle nested braces)
    match = re.search(r'<functioncall>\s*(\{.*\})', text, re.DOTALL)
    if not match:
        # Try <tool_call> tag
        match = re.search(r'<tool_call>\s*(\{.*\})', text, re.DOTALL)
    if not match:
        # Try raw JSON
        if text.startswith("{") and '"name"' in text:
            match = re.match(r'(\{.*\})', text, re.DOTALL)
    if not match:
        return None

    raw = match.group(1).strip()

    # Try direct JSON parse
    try:
        call = json.loads(raw)
        if isinstance(call, dict) and "name" in call:
            return call
    except json.JSONDecodeError:
        pass

    # Handle Glaive format: single-quoted arguments value
    # Replace 'arguments': '{...}' with "arguments": {...}
    fixed = re.sub(r"""'(\{[^']*\})'""", r'\1', raw)
    try:
        call = json.loads(fixed)
        if isinstance(call, dict) and "name" in call:
            return call
    except json.JSONDecodeError:
        pass

    # Last resort: extract just the name
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
    if name_match:
        return {"name": name_match.group(1), "arguments": {}}

    return None


def parse_arguments(args_str: str) -> dict:
    """Parse arguments string (may be JSON string or dict)."""
    if isinstance(args_str, dict):
        return args_str
    try:
        parsed = json.loads(args_str)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def compute_param_f1(pred_args: dict, gold_args: dict) -> float:
    """Compute F1 on parameter key-value pairs."""
    pred_items = {(k, str(v)) for k, v in pred_args.items()}
    gold_items = {(k, str(v)) for k, v in gold_args.items()}

    if not gold_items and not pred_items:
        return 1.0
    if not gold_items or not pred_items:
        return 0.0

    tp = len(pred_items & gold_items)
    precision = tp / len(pred_items) if pred_items else 0
    recall = tp / len(gold_items) if gold_items else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_value_recall(pred_args: dict, gold_args: dict) -> float:
    """Compute recall on parameter values only (ignoring key names).

    This captures cases where the model uses the correct values but
    different parameter names (e.g., 'birthdate' vs 'date_of_birth').
    """
    pred_vals = {str(v) for v in pred_args.values()}
    gold_vals = {str(v) for v in gold_args.values()}

    if not gold_vals:
        return 1.0
    if not pred_vals:
        return 0.0

    tp = len(pred_vals & gold_vals)
    return tp / len(gold_vals)


def evaluate_split(
    model: ToolFormerMicro,
    tokenizer,
    split_path: Path,
    device: str,
    max_examples: int = 50,
    max_new_tokens: int = 256,
) -> dict:
    """Evaluate on a single split."""
    model.eval()

    examples = []
    with open(split_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    # Determine is_tool_call from gold response (data labels are unreliable)
    for e in examples:
        gold = e.get("assistant_response", "")
        e["is_tool_call"] = (
            "<functioncall>" in gold or "<tool_call>" in gold
            or (gold.strip().startswith("{") and '"name"' in gold)
        )
        # Also extract target_tool from gold if missing
        if e["is_tool_call"] and not e.get("target_tool"):
            gold_call = parse_function_call(gold)
            if gold_call:
                e["target_tool"] = gold_call.get("name")

    # Separate tool-calling and non-tool examples
    tool_examples = [e for e in examples if e.get("is_tool_call")]
    non_tool_examples = [e for e in examples if not e.get("is_tool_call")]

    # Limit to balanced set
    num_tool = min(len(tool_examples), max_examples // 2)
    num_non_tool = min(len(non_tool_examples), max_examples - num_tool)
    examples = tool_examples[:num_tool] + non_tool_examples[:num_non_tool]

    results = []
    correct_tool = 0
    false_positives = 0
    false_negatives = 0
    param_f1_sum = 0.0
    value_recall_sum = 0.0
    exact_match = 0
    encode_times = []
    generate_times = []
    num_tool_calls = 0
    num_non_tool_calls = 0

    for ex in examples:
        is_tool = ex.get("is_tool_call", False)
        if is_tool:
            num_tool_calls += 1
        else:
            num_non_tool_calls += 1

        # Encode tools
        tool_schemas = ex["tool_schemas"]
        t0 = time.perf_counter()

        # Tokenize schemas
        schema_ids_list = []
        schema_mask_list = []
        max_len = model.config.max_schema_tokens
        pad_id = tokenizer.pad_token_id or 0

        for schema_text in tool_schemas:
            ids = tokenizer.encode(
                schema_text, add_special_tokens=False,
                max_length=max_len, truncation=True,
            )
            pad_len = max_len - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [pad_id] * pad_len
            schema_ids_list.append(ids)
            schema_mask_list.append(mask)

        tool_ids = torch.tensor([schema_ids_list], device=device)  # (1, T, S)
        tool_mask = torch.tensor([schema_mask_list], device=device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tool_memory = model.encode_tools(tool_ids, tool_mask)
        encode_time = (time.perf_counter() - t0) * 1000
        encode_times.append(encode_time)

        # Prepare input: system prompt + user query
        sys_ids = tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False)
        user_ids = tokenizer.encode(ex["user_query"], add_special_tokens=False, max_length=512, truncation=True)
        input_ids = torch.tensor([sys_ids + user_ids], device=device)

        # Generate
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_ids = model.generate(
                tool_memory, input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
        generate_time = (time.perf_counter() - t0) * 1000
        generate_times.append(generate_time)

        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

        # Parse prediction
        pred_call = parse_function_call(generated_text)

        # Parse gold
        gold_call = parse_function_call(ex["assistant_response"])

        # Score
        if is_tool:
            if pred_call is None:
                false_negatives += 1
            elif gold_call and pred_call.get("name") == gold_call.get("name"):
                correct_tool += 1
                # Check parameters
                pred_args = parse_arguments(pred_call.get("arguments", {}))
                gold_args = parse_arguments(gold_call.get("arguments", {}))
                pf1 = compute_param_f1(pred_args, gold_args)
                param_f1_sum += pf1
                vr = compute_value_recall(pred_args, gold_args)
                value_recall_sum += vr
                # Exact match: name + all argument key-value pairs match
                if pf1 == 1.0 and set(pred_args.keys()) == set(gold_args.keys()):
                    exact_match += 1
            else:
                false_negatives += 1
        else:
            if pred_call is not None:
                false_positives += 1

        results.append({
            "user_query": ex["user_query"],
            "is_tool_call": is_tool,
            "target_tool": ex.get("target_tool"),
            "gold_response": ex["assistant_response"],
            "predicted": generated_text,
            "pred_call": pred_call,
            "correct": (pred_call is not None and gold_call is not None
                       and pred_call.get("name") == gold_call.get("name")) if is_tool else (pred_call is None),
            "encode_ms": encode_time,
            "generate_ms": generate_time,
        })

    # Compute metrics
    tsa = correct_tool / num_tool_calls if num_tool_calls > 0 else 0
    pf1 = param_f1_sum / correct_tool if correct_tool > 0 else 0
    vr = value_recall_sum / correct_tool if correct_tool > 0 else 0
    em = exact_match / num_tool_calls if num_tool_calls > 0 else 0
    fpr = false_positives / num_non_tool_calls if num_non_tool_calls > 0 else 0
    fnr = false_negatives / num_tool_calls if num_tool_calls > 0 else 0

    summary = {
        "num_examples": len(examples),
        "num_tool_calls": num_tool_calls,
        "num_non_tool": num_non_tool_calls,
        "tool_selection_accuracy": tsa,
        "parameter_f1": pf1,
        "value_recall": vr,
        "exact_match": em,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "correct_tool": correct_tool,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "avg_encode_ms": sum(encode_times) / len(encode_times) if encode_times else 0,
        "avg_generate_ms": sum(generate_times) / len(generate_times) if generate_times else 0,
    }

    return summary, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ToolFormerMicro")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--split", type=str, default="all",
                       choices=["all", "test_seen", "test_held_out", "test_unseen"])
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="eval_results/tool_former")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = ToolFormerMicro.load_checkpoint(args.checkpoint, device="cpu")
    model = model.to(args.device)

    config = model.config
    counts = model.param_count()
    print(f"Model: {counts['total']:,} params ({counts['total'] * 2 / 1e6:.0f} MB fp16)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine splits to evaluate
    split_map = {
        "test_seen": "test_seen_gisting.jsonl",
        "test_held_out": "test_held_out_gisting.jsonl",
        "test_unseen": "test_unseen_gisting.jsonl",
    }
    if args.split == "all":
        splits = list(split_map.keys())
    else:
        splits = [args.split]

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}

    for split_name in splits:
        split_path = ROOT / "data" / "processed" / split_map[split_name]
        if not split_path.exists():
            print(f"\n  SKIP {split_name}: {split_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {split_name}")
        print(f"{'='*60}")

        summary, results = evaluate_split(
            model, tokenizer, split_path, args.device,
            max_examples=args.max_examples,
            max_new_tokens=args.max_new_tokens,
        )

        all_summaries[split_name] = summary

        # Print summary
        print(f"  TSA:  {summary['tool_selection_accuracy']:.3f}")
        print(f"  PF1:  {summary['parameter_f1']:.3f}")
        print(f"  VR:   {summary['value_recall']:.3f}")
        print(f"  EM:   {summary['exact_match']:.3f}")
        print(f"  FPR:  {summary['false_positive_rate']:.3f}")
        print(f"  FNR:  {summary['false_negative_rate']:.3f}")
        print(f"  Encode: {summary['avg_encode_ms']:.0f} ms")
        print(f"  Generate: {summary['avg_generate_ms']:.0f} ms")

        # Save per-example results
        results_path = output_dir / f"results_{split_name}.jsonl"
        with open(results_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Save split summary
        summary_path = output_dir / f"summary_{split_name}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # Save combined summary
    summary_path = output_dir / "summary_all.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
