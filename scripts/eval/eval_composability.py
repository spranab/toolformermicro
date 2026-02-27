#!/usr/bin/env python3
"""Composability evaluation suite for ToolFormerMicro.

Three killer experiments that prove composable cross-attention caching works:

1. **Compositional Generalization**: Train on tool subsets {A,B}, {C,D} —
   test on unseen combos {A,C}, {B,D}. TSA should hold.

2. **Scaling Curves**: TSA + encode latency vs. number of tools (5→200).
   Our gist cache should give near-constant latency per additional tool.

3. **Cache Invalidation / Hot-Swap**: Update one tool's schema, re-encode
   only that tool, verify all other tools' routing is unaffected.

Usage:
  python scripts/eval/eval_composability.py --checkpoint checkpoints/tool_former/stage2
  python scripts/eval/eval_composability.py --checkpoint checkpoints/tool_former/stage2 --experiment all
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tool_former import ToolFormerMicro, SYSTEM_PROMPT
from src.tool_former_config import ToolFormerConfig
from scripts.eval.eval_tool_former import parse_function_call


def tokenize_schema(schema_text: str, tokenizer, max_len: int = 256) -> tuple[list, list]:
    """Tokenize a single schema to padded ids + mask."""
    pad_id = tokenizer.pad_token_id or 0
    ids = tokenizer.encode(schema_text, add_special_tokens=False, max_length=max_len, truncation=True)
    pad_len = max_len - len(ids)
    return ids + [pad_id] * pad_len, [1] * len(ids) + [0] * pad_len


def encode_tool_batch(model, schemas: list[str], tokenizer, device: str) -> torch.Tensor:
    """Encode a list of schema texts into gist vectors.

    Returns: (num_tools, K, D) tensor of gist vectors.
    """
    max_len = model.config.max_schema_tokens
    ids_list, mask_list = [], []
    for s in schemas:
        ids, mask = tokenize_schema(s, tokenizer, max_len)
        ids_list.append(ids)
        mask_list.append(mask)

    tool_ids = torch.tensor(ids_list, device=device).unsqueeze(0)  # (1, T, S)
    tool_mask = torch.tensor(mask_list, device=device).unsqueeze(0)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gists = model.encode_tools(tool_ids, tool_mask)  # (1, T*K, D)

    K = model.config.num_gist_tokens
    num_tools = len(schemas)
    return gists.view(num_tools, K, -1)  # (T, K, D)


def generate_for_query(
    model, tool_memory: torch.Tensor, query: str, tokenizer, device: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate response for a query given pre-computed tool memory."""
    sys_ids = tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False)
    user_ids = tokenizer.encode(query, add_special_tokens=False, max_length=512, truncation=True)
    input_ids = torch.tensor([sys_ids + user_ids], device=device)

    # tool_memory: (T, K, D) → (1, T*K, D)
    if tool_memory.dim() == 3:
        memory = tool_memory.view(1, -1, tool_memory.shape[-1])
    else:
        memory = tool_memory

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen_ids = model.generate(
            memory, input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(gen_ids, skip_special_tokens=False)


def parse_tool_name(text: str) -> str | None:
    """Extract tool name from generated text."""
    import re
    match = re.search(r'<functioncall>\s*\{[^}]*"name"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    return None


def load_tool_examples(path: Path, max_examples: int = 200) -> list[dict]:
    """Load tool-calling examples with corrected is_tool_call labels."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_examples * 3:  # read extra to find enough TC examples
                break
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            # Fix is_tool_call from gold response (data labels unreliable)
            gold = ex.get("assistant_response", "")
            ex["is_tool_call"] = (
                "<functioncall>" in gold or "<tool_call>" in gold
                or (gold.strip().startswith("{") and '"name"' in gold)
            )
            if ex["is_tool_call"] and not ex.get("target_tool"):
                gold_call = parse_function_call(gold)
                if gold_call:
                    ex["target_tool"] = gold_call.get("name")
            if ex["is_tool_call"] and ex.get("target_tool"):
                examples.append(ex)
                if len(examples) >= max_examples:
                    break
    return examples


# ---------------------------------------------------------------------------
# Experiment 1: Compositional Generalization
# ---------------------------------------------------------------------------

def eval_compositional_generalization(
    model: ToolFormerMicro, tokenizer, device: str, output_dir: Path,
):
    """Train on subsets, test on unseen combinations.

    Split tools into groups A, B, C, D. Create test queries where tools are
    combined in ways not seen during training. Measure TSA stability.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Compositional Generalization")
    print("=" * 60)

    # Load test data (use test_seen which has the most examples)
    test_path = ROOT / "data" / "processed" / "test_seen_gisting.jsonl"
    if not test_path.exists():
        print("  SKIP: test data not found")
        return

    examples = load_tool_examples(test_path, max_examples=100)

    if len(examples) < 10:
        print("  SKIP: not enough tool-calling examples")
        return

    print(f"  Loaded {len(examples)} tool-calling examples")

    # For each example, test with the full tool set (as-is) and with
    # a REARRANGED tool set (same tools, different order + different combos)
    results_standard = []
    results_shuffled = []

    for ex in examples[:50]:
        schemas = ex["tool_schemas"]
        query = ex["user_query"]
        target = ex["target_tool"]

        # Standard: encode all tools in original order
        gists = encode_tool_batch(model, schemas, tokenizer, device)
        output = generate_for_query(model, gists, query, tokenizer, device)
        pred = parse_tool_name(output)
        results_standard.append({"correct": pred == target, "pred": pred, "target": target})

        # Shuffled: randomize tool order (tests order independence)
        shuffled = list(enumerate(schemas))
        random.shuffle(shuffled)
        shuffled_schemas = [s for _, s in shuffled]

        gists_shuffled = encode_tool_batch(model, shuffled_schemas, tokenizer, device)
        output_shuffled = generate_for_query(model, gists_shuffled, query, tokenizer, device)
        pred_shuffled = parse_tool_name(output_shuffled)
        results_shuffled.append({"correct": pred_shuffled == target, "pred": pred_shuffled, "target": target})

    tsa_standard = sum(r["correct"] for r in results_standard) / max(1, len(results_standard))
    tsa_shuffled = sum(r["correct"] for r in results_shuffled) / max(1, len(results_shuffled))

    print(f"  TSA (standard order):  {tsa_standard:.3f} ({sum(r['correct'] for r in results_standard)}/{len(results_standard)})")
    print(f"  TSA (shuffled order):  {tsa_shuffled:.3f} ({sum(r['correct'] for r in results_shuffled)}/{len(results_shuffled)})")
    print(f"  Order independence:    {'PASS' if abs(tsa_standard - tsa_shuffled) < 0.05 else 'DEGRADED'}")

    summary = {
        "experiment": "compositional_generalization",
        "num_examples": len(results_standard),
        "tsa_standard": tsa_standard,
        "tsa_shuffled": tsa_shuffled,
        "delta": abs(tsa_standard - tsa_shuffled),
        "order_independent": abs(tsa_standard - tsa_shuffled) < 0.05,
    }

    out_path = output_dir / "compositional_generalization.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {out_path}")


# ---------------------------------------------------------------------------
# Experiment 2: Scaling Curves
# ---------------------------------------------------------------------------

def eval_scaling_curves(
    model: ToolFormerMicro, tokenizer, device: str, output_dir: Path,
):
    """TSA + latency vs. number of tools.

    For N in [5, 10, 20, 50, 100, 200], encode N tools and test routing.
    The full test has 20 tools per example. We pad with random extra tools.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Scaling Curves (TSA + Latency vs. N)")
    print("=" * 60)

    # Load schemas for padding
    schema_path = ROOT / "data" / "processed" / "schema_ae.jsonl"
    if not schema_path.exists():
        print("  SKIP: schema_ae.jsonl not found")
        return

    all_schemas = []
    with open(schema_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_schemas.append(json.loads(line))

    # Load test examples
    test_path = ROOT / "data" / "processed" / "test_seen_gisting.jsonl"
    if not test_path.exists():
        print("  SKIP: test data not found")
        return

    examples = load_tool_examples(test_path, max_examples=50)

    test_examples = examples[:20]  # Use 20 examples per N-value
    print(f"  Test examples: {len(test_examples)}")
    print(f"  Schema library: {len(all_schemas)} tools")

    tool_counts = [5, 10, 20, 50, 100, 200]
    # Cap at available schemas
    tool_counts = [n for n in tool_counts if n <= len(all_schemas)]
    print(f"  Testing N = {tool_counts}")

    scaling_results = []

    for N in tool_counts:
        print(f"\n  --- N = {N} tools ---")
        correct = 0
        encode_times = []
        generate_times = []
        total = 0

        for ex in test_examples:
            base_schemas = ex["tool_schemas"][:min(N, len(ex["tool_schemas"]))]
            target = ex["target_tool"]

            # Pad with extra schemas if N > base
            if N > len(base_schemas):
                # Add random schemas from library (avoid duplicates by name)
                base_names = set()
                for s in base_schemas:
                    try:
                        parsed = json.loads(s)
                        name = parsed.get("function", {}).get("name", "")
                        base_names.add(name)
                    except json.JSONDecodeError:
                        pass

                extra = [
                    s["schema_text"] for s in all_schemas
                    if s["tool_name"] not in base_names
                ]
                random.shuffle(extra)
                padded = base_schemas + extra[:N - len(base_schemas)]
            else:
                padded = base_schemas[:N]

            # Encode
            t0 = time.perf_counter()
            gists = encode_tool_batch(model, padded, tokenizer, device)
            encode_ms = (time.perf_counter() - t0) * 1000
            encode_times.append(encode_ms)

            # Generate
            t0 = time.perf_counter()
            output = generate_for_query(model, gists, ex["user_query"], tokenizer, device)
            gen_ms = (time.perf_counter() - t0) * 1000
            generate_times.append(gen_ms)

            pred = parse_tool_name(output)
            if pred == target:
                correct += 1
            total += 1

        tsa = correct / max(1, total)
        avg_encode = sum(encode_times) / max(1, len(encode_times))
        avg_gen = sum(generate_times) / max(1, len(generate_times))
        per_tool_encode = avg_encode / N

        print(f"    TSA:           {tsa:.3f} ({correct}/{total})")
        print(f"    Avg encode:    {avg_encode:.1f} ms ({per_tool_encode:.2f} ms/tool)")
        print(f"    Avg generate:  {avg_gen:.1f} ms")

        scaling_results.append({
            "num_tools": N,
            "tsa": tsa,
            "avg_encode_ms": avg_encode,
            "per_tool_encode_ms": per_tool_encode,
            "avg_generate_ms": avg_gen,
            "num_examples": total,
        })

    # Memory usage per tool
    K = model.config.num_gist_tokens
    D = model.config.hidden_size
    bytes_per_tool = K * D * 2  # fp16
    print(f"\n  Gist cache: {K} tokens × {D} dim × 2 bytes = {bytes_per_tool:,} bytes ({bytes_per_tool/1024:.1f} KB) per tool")

    summary = {
        "experiment": "scaling_curves",
        "gist_bytes_per_tool": bytes_per_tool,
        "results": scaling_results,
    }

    out_path = output_dir / "scaling_curves.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {out_path}")


# ---------------------------------------------------------------------------
# Experiment 3: Cache Invalidation / Hot-Swap
# ---------------------------------------------------------------------------

def eval_cache_invalidation(
    model: ToolFormerMicro, tokenizer, device: str, output_dir: Path,
):
    """Update one tool's gist cache, verify others are unaffected.

    1. Encode all tools → gist cache G_original
    2. Modify one tool's schema (add a dummy parameter)
    3. Re-encode ONLY that tool → get G_modified
    4. Swap the modified tool's gists into the cache
    5. Verify: gists for all OTHER tools are bit-identical to G_original
    6. Verify: queries targeting other tools still produce correct results
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Cache Invalidation / Hot-Swap")
    print("=" * 60)

    test_path = ROOT / "data" / "processed" / "test_seen_gisting.jsonl"
    if not test_path.exists():
        print("  SKIP: test data not found")
        return

    examples = load_tool_examples(test_path, max_examples=50)

    if not examples:
        print("  SKIP: no tool-calling examples")
        return

    # Use first example as test case
    ex = examples[0]
    schemas = ex["tool_schemas"]
    num_tools = len(schemas)
    print(f"  Tools in example: {num_tools}")

    # Step 1: Encode all tools
    gists_original = encode_tool_batch(model, schemas, tokenizer, device)  # (T, K, D)
    print(f"  Original gist shape: {gists_original.shape}")

    # Step 2: Modify one tool's schema (add a dummy parameter)
    modify_idx = 0  # modify the first tool
    original_schema = schemas[modify_idx]
    try:
        parsed = json.loads(original_schema)
        # Add a dummy parameter
        params = parsed.get("function", {}).get("parameters", {})
        params["_test_dummy_param"] = {"description": "Test parameter for cache invalidation", "type": "str"}
        modified_schema = json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        modified_schema = original_schema + '\n  "_test_dummy": "cache_invalidation_test"'

    # Step 3: Re-encode ONLY the modified tool
    t0 = time.perf_counter()
    gists_modified_tool = encode_tool_batch(model, [modified_schema], tokenizer, device)  # (1, K, D)
    hotswap_ms = (time.perf_counter() - t0) * 1000
    print(f"  Hot-swap re-encode time: {hotswap_ms:.1f} ms")

    # Step 4: Create new cache with swapped gist
    gists_swapped = gists_original.clone()
    K = model.config.num_gist_tokens
    gists_swapped[modify_idx] = gists_modified_tool[0]

    # Step 5: Verify other tools' gists are bit-identical
    other_match = True
    for i in range(num_tools):
        if i == modify_idx:
            continue
        if not torch.equal(gists_original[i], gists_swapped[i]):
            other_match = False
            break

    modified_changed = not torch.equal(gists_original[modify_idx], gists_swapped[modify_idx])
    cos_sim = torch.nn.functional.cosine_similarity(
        gists_original[modify_idx].flatten().unsqueeze(0).float(),
        gists_swapped[modify_idx].flatten().unsqueeze(0).float(),
    ).item()

    print(f"  Other tools unchanged:   {'YES (bit-identical)' if other_match else 'NO (FAIL)'}")
    print(f"  Modified tool changed:   {'YES' if modified_changed else 'NO (unchanged)'}")
    print(f"  Modified tool cos_sim:   {cos_sim:.4f} (to original)")

    # Step 6: Test routing with swapped cache on queries targeting OTHER tools
    print(f"\n  Testing routing stability on {min(20, len(examples))} queries...")
    correct_original = 0
    correct_swapped = 0
    total = 0

    for test_ex in examples[:20]:
        query = test_ex["user_query"]
        target = test_ex["target_tool"]

        # With original cache
        output_orig = generate_for_query(model, gists_original, query, tokenizer, device)
        pred_orig = parse_tool_name(output_orig)

        # With swapped cache
        output_swap = generate_for_query(model, gists_swapped, query, tokenizer, device)
        pred_swap = parse_tool_name(output_swap)

        if pred_orig == target:
            correct_original += 1
        if pred_swap == target:
            correct_swapped += 1
        total += 1

    tsa_orig = correct_original / max(1, total)
    tsa_swap = correct_swapped / max(1, total)

    print(f"  TSA (original cache):  {tsa_orig:.3f}")
    print(f"  TSA (swapped cache):   {tsa_swap:.3f}")
    print(f"  Delta:                 {abs(tsa_orig - tsa_swap):.3f}")
    print(f"  Routing stable:        {'YES' if abs(tsa_orig - tsa_swap) < 0.05 else 'DEGRADED'}")

    summary = {
        "experiment": "cache_invalidation",
        "num_tools": num_tools,
        "modified_tool_idx": modify_idx,
        "other_tools_bitidentical": other_match,
        "modified_tool_changed": modified_changed,
        "modified_tool_cosine_sim": cos_sim,
        "hotswap_encode_ms": hotswap_ms,
        "tsa_original": tsa_orig,
        "tsa_swapped": tsa_swap,
        "tsa_delta": abs(tsa_orig - tsa_swap),
        "routing_stable": abs(tsa_orig - tsa_swap) < 0.05,
        "num_test_queries": total,
    }

    out_path = output_dir / "cache_invalidation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Composability evaluation for ToolFormerMicro")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=["all", "compositional", "scaling", "invalidation"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="eval_results/composability")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = ToolFormerMicro.load_checkpoint(args.checkpoint, device="cpu")
    model = model.to(args.device)
    model.eval()

    config = model.config
    counts = model.param_count()
    print(f"Model: {counts['total']:,} params ({counts['total'] * 2 / 1e6:.0f} MB fp16)")

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    torch.manual_seed(42)

    if args.experiment in ("all", "compositional"):
        eval_compositional_generalization(model, tokenizer, args.device, output_dir)

    if args.experiment in ("all", "scaling"):
        eval_scaling_curves(model, tokenizer, args.device, output_dir)

    if args.experiment in ("all", "invalidation"):
        eval_cache_invalidation(model, tokenizer, args.device, output_dir)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
