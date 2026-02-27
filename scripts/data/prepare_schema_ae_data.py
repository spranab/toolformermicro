#!/usr/bin/env python3
"""Generate schema autoencoding data for Stage 1 (encoder warmup).

For each tool schema, creates augmented variants:
  1. Pretty-printed JSON (indent=2)
  2. Compact JSON (no indent)
  3. Shuffled parameter order
  4. Description-truncated variant
  5. Minimal variant (name + params, no descriptions)

Output format (JSONL):
  {"schema_text": "...", "tool_name": "tool_name"}

The encoder must learn to compress any of these variants into
gist vectors that can reconstruct the original schema.
"""

import argparse
import copy
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_catalog(path: Path) -> list[dict]:
    """Load tool catalog as list of tool entries."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def schema_pretty(schema: dict) -> str:
    """Pretty-printed JSON (indent=2)."""
    return json.dumps(schema, indent=2, ensure_ascii=False)


def schema_compact(schema: dict) -> str:
    """Compact JSON (no whitespace)."""
    return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)


def schema_shuffled_params(schema: dict, rng: random.Random) -> str:
    """Shuffle parameter order within the schema."""
    s = copy.deepcopy(schema)
    func = s.get("function", s)
    params = func.get("parameters", {})

    # Collect actual parameter entries (skip "type", "properties", "required")
    meta_keys = {"type", "properties", "required"}
    param_items = [(k, v) for k, v in params.items() if k not in meta_keys]
    rng.shuffle(param_items)

    # Rebuild parameters dict with shuffled order
    new_params = {}
    for k, v in param_items:
        new_params[k] = v
    # Re-add meta keys
    for k in meta_keys:
        if k in params:
            new_params[k] = params[k]
    func["parameters"] = new_params

    return json.dumps(s, indent=2, ensure_ascii=False)


def schema_truncated_desc(schema: dict, max_desc_len: int = 50) -> str:
    """Truncate descriptions to max_desc_len characters."""
    s = copy.deepcopy(schema)
    func = s.get("function", s)

    # Truncate function description
    if "description" in func:
        desc = func["description"]
        if len(desc) > max_desc_len:
            func["description"] = desc[:max_desc_len].rstrip() + "..."

    # Truncate parameter descriptions
    params = func.get("parameters", {})
    meta_keys = {"type", "properties", "required"}
    for k, v in params.items():
        if k not in meta_keys and isinstance(v, dict) and "description" in v:
            desc = v["description"]
            if len(desc) > max_desc_len:
                v["description"] = desc[:max_desc_len].rstrip() + "..."

    return json.dumps(s, indent=2, ensure_ascii=False)


def schema_minimal(schema: dict) -> str:
    """Minimal variant: keep name and parameter names/types, drop descriptions."""
    s = copy.deepcopy(schema)
    func = s.get("function", s)

    # Remove function description
    func.pop("description", None)

    # Remove parameter descriptions, keep name + type + default
    params = func.get("parameters", {})
    meta_keys = {"type", "properties", "required"}
    for k, v in params.items():
        if k not in meta_keys and isinstance(v, dict):
            v.pop("description", None)

    return json.dumps(s, indent=2, ensure_ascii=False)


def schema_reindented(schema: dict) -> str:
    """Re-indented with 4-space indent."""
    return json.dumps(schema, indent=4, ensure_ascii=False)


def generate_augmented_schemas(
    schema: dict,
    tool_name: str,
    rng: random.Random,
    num_shuffled: int = 3,
) -> list[dict]:
    """Generate all augmented variants for a single tool schema."""
    examples = []

    # 1. Pretty-printed (canonical)
    examples.append({
        "schema_text": schema_pretty(schema),
        "tool_name": tool_name,
    })

    # 2. Compact
    examples.append({
        "schema_text": schema_compact(schema),
        "tool_name": tool_name,
    })

    # 3. Shuffled parameters (multiple variants)
    for _ in range(num_shuffled):
        examples.append({
            "schema_text": schema_shuffled_params(schema, rng),
            "tool_name": tool_name,
        })

    # 4. Truncated descriptions
    examples.append({
        "schema_text": schema_truncated_desc(schema),
        "tool_name": tool_name,
    })

    # 5. Minimal (no descriptions)
    examples.append({
        "schema_text": schema_minimal(schema),
        "tool_name": tool_name,
    })

    # 6. Re-indented (4-space)
    examples.append({
        "schema_text": schema_reindented(schema),
        "tool_name": tool_name,
    })

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Generate schema autoencoding data for Stage 1"
    )
    parser.add_argument(
        "--catalog", type=Path,
        default=ROOT / "data" / "catalogs" / "catalog_100.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "processed" / "schema_ae.jsonl",
    )
    parser.add_argument(
        "--tool-split", type=Path, default=None,
        help="Path to tool_split.json (if exists, uses only train tools)",
    )
    parser.add_argument("--num-shuffled", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repeat-epochs", type=int, default=5,
        help="Number of times to repeat the augmented set (more data for 3000 steps)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load catalog
    print(f"Loading catalog from {args.catalog}")
    catalog = load_catalog(args.catalog)

    # Build name -> schema mapping
    name_to_schema = {}
    for tool in catalog:
        name = tool["schema"]["function"]["name"]
        name_to_schema[name] = tool["schema"]

    # Filter to train tools if split exists
    if args.tool_split and args.tool_split.exists():
        print(f"Loading tool split from {args.tool_split}")
        with open(args.tool_split, encoding="utf-8") as f:
            split = json.load(f)
        train_names = set(split["train_tools"])
        name_to_schema = {k: v for k, v in name_to_schema.items() if k in train_names}
        print(f"  Using {len(name_to_schema)} train tools")
    else:
        # Use first 80 tools (default split)
        all_names = list(name_to_schema.keys())
        rng_split = random.Random(args.seed)
        rng_split.shuffle(all_names)
        train_names = set(all_names[:80])
        name_to_schema = {k: v for k, v in name_to_schema.items() if k in train_names}
        print(f"  Using {len(name_to_schema)} tools (default 80/20 split)")

    # Generate augmented schemas
    all_examples = []
    for tool_name, schema in name_to_schema.items():
        examples = generate_augmented_schemas(schema, tool_name, rng, args.num_shuffled)
        all_examples.extend(examples)

    print(f"  Generated {len(all_examples)} augmented examples from {len(name_to_schema)} tools")

    # Repeat for more training data
    repeated = []
    for epoch in range(args.repeat_epochs):
        epoch_rng = random.Random(args.seed + epoch)
        epoch_examples = list(all_examples)
        epoch_rng.shuffle(epoch_examples)
        repeated.extend(epoch_examples)

    print(f"  After {args.repeat_epochs}x repeat: {len(repeated)} total examples")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for example in repeated:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"  Saved to {args.output}")

    # Stats
    tools_seen = set(ex["tool_name"] for ex in repeated)
    avg_len = sum(len(ex["schema_text"]) for ex in repeated) / len(repeated)
    print(f"\nStats:")
    print(f"  Unique tools: {len(tools_seen)}")
    print(f"  Total examples: {len(repeated)}")
    print(f"  Avg schema text length: {avg_len:.0f} chars")
    print(f"  Examples per tool: {len(repeated) / len(tools_seen):.0f}")


if __name__ == "__main__":
    main()
