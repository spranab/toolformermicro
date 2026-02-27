#!/usr/bin/env python3
"""Three-stage training for ToolFormerMicro.

Stage 1: Schema Auto-Encoding (encoder warmup)
  - Train encoder + gist pooling to compress schemas into K gist vectors
  - Decoder reconstructs schema from gist vectors via cross-attention
  - Only encoder, gist pooling, decoder cross-attention, and decoder norm are trained

Stage 1.5: Contrastive Gist Discrimination
  - InfoNCE loss: query embedding close to correct tool's gist, far from negatives
  - Hard negatives mined by schema token similarity
  - Shapes the gist embedding space for discriminative tool selection

Stage 2: End-to-End Tool Calling
  - ALL parameters trained jointly
  - Input: 20 tool schemas (encoded independently) + user query
  - Output: correct tool call or text response
  - Auxiliary: periodic schema AE + contrastive loss to stabilize encoder

Usage:
  python scripts/train/train_tool_former.py --config configs/tool_former_config.yaml
  python scripts/train/train_tool_former.py --config configs/tool_former_config.yaml --stage 2 --resume checkpoints/tool_former/stage1_5
  python scripts/train/train_tool_former.py --config configs/tool_former_config.yaml --no-contrastive
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tool_former import ToolFormerMicro, SYSTEM_PROMPT
from src.tool_former_config import ToolFormerConfig
from src.tool_former_data import (
    ToolFormerDataset,
    SchemaAEDataset,
    ToolFormerCollator,
    SchemaAECollator,
    ContrastiveDataset,
    ContrastiveCollator,
    build_schema_similarity_matrix,
)


def get_cosine_schedule(optimizer, num_warmup_steps, num_training_steps):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def freeze_for_stage1(model: ToolFormerMicro):
    """Freeze decoder self-attention + MLP for Stage 1.

    Only train: encoder, gist pooling, decoder cross-attention + norms.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze encoder
    for p in model.encoder_layers.parameters():
        p.requires_grad = True
    for p in model.encoder_norm.parameters():
        p.requires_grad = True
    for p in model.gist_pool.parameters():
        p.requires_grad = True

    # Unfreeze decoder cross-attention, its norm, and cross-attention gate
    for layer in model.decoder_layers:
        for p in layer.cross_attn.parameters():
            p.requires_grad = True
        for p in layer.cross_attn_layernorm.parameters():
            p.requires_grad = True
        if hasattr(layer, 'cross_attn_gate'):
            layer.cross_attn_gate.requires_grad = True

    # Unfreeze decoder norm and LM head (needed for reconstruction loss)
    for p in model.decoder_norm.parameters():
        p.requires_grad = True
    if not model.config.tie_word_embeddings:
        for p in model.lm_head.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Stage 1 trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def freeze_for_stage1_5(model: ToolFormerMicro):
    """Freeze for Stage 1.5: same as stage 1 (encoder + gist + cross-attn).

    The contrastive loss only backprops through the encoder, gist pooling, and
    the query path through the encoder (shared weights). Decoder self-attn + MLP
    stay frozen.
    """
    # Same freeze pattern as stage 1
    freeze_for_stage1(model)


def unfreeze_all(model: ToolFormerMicro):
    """Unfreeze all parameters for Stage 2."""
    for p in model.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Stage 2 trainable: {trainable:,} (100%)")


def train_stage1(model: ToolFormerMicro, config: ToolFormerConfig, tokenizer, device: str):
    """Stage 1: Schema Auto-Encoding."""
    print("\n" + "=" * 60)
    print("STAGE 1: Schema Auto-Encoding")
    print("=" * 60)

    # Load schema AE dataset
    ae_path = ROOT / "data" / "processed" / "schema_ae.jsonl"
    if not ae_path.exists():
        print(f"  ERROR: Schema AE data not found at {ae_path}")
        print("  Run: python scripts/data/prepare_schema_ae_data.py")
        return
    dataset = SchemaAEDataset(ae_path)
    print(f"  Dataset: {len(dataset)} schemas")

    collator = SchemaAECollator(tokenizer, config)
    loader = DataLoader(
        dataset, batch_size=config.stage1_batch_size,
        shuffle=True, collate_fn=collator, drop_last=True,
    )

    # Freeze for stage 1
    freeze_for_stage1(model)
    model.to(device)
    model.train()

    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=config.stage1_lr, weight_decay=0.01)
    scheduler = get_cosine_schedule(optimizer, config.stage1_warmup_steps, config.stage1_steps)

    print(f"  Steps: {config.stage1_steps}")
    print(f"  LR: {config.stage1_lr}")
    print(f"  Batch size: {config.stage1_batch_size}")
    print()

    step = 0
    running_loss = 0.0
    start_time = time.time()

    while step < config.stage1_steps:
        for batch in loader:
            if step >= config.stage1_steps:
                break

            # Move to device
            schema_ids = batch["schema_ids"].to(device)
            schema_mask = batch["schema_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config.bf16):
                loss = model.forward_schema_ae(schema_ids, schema_mask, target_ids, labels)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            step += 1

            if step % config.logging_steps == 0:
                avg_loss = running_loss / config.logging_steps
                elapsed = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {step}/{config.stage1_steps} | loss={avg_loss:.4f} | lr={lr:.2e} | {elapsed:.0f}s")
                running_loss = 0.0

            if step % config.save_steps == 0:
                save_path = ROOT / config.output_dir / "stage1" / f"step_{step}"
                model.save_checkpoint(str(save_path))

    # Save final stage 1 checkpoint
    save_path = ROOT / config.output_dir / "stage1"
    model.save_checkpoint(str(save_path))
    elapsed = time.time() - start_time
    print(f"\n  Stage 1 complete in {elapsed:.0f}s. Saved to {save_path}")


def train_stage1_5(model: ToolFormerMicro, config: ToolFormerConfig, tokenizer, device: str):
    """Stage 1.5: Contrastive Gist Discrimination.

    InfoNCE loss with hard negatives mined by schema token similarity.
    Optionally includes auxiliary schema AE loss to maintain reconstruction quality.
    """
    print("\n" + "=" * 60)
    print("STAGE 1.5: Contrastive Gist Discrimination")
    print("=" * 60)

    # Load data
    train_path = ROOT / "data" / "processed" / "train_gisting.jsonl"
    schema_path = ROOT / "data" / "processed" / "schema_ae.jsonl"

    if not train_path.exists() or not schema_path.exists():
        print(f"  ERROR: Data not found at {train_path} or {schema_path}")
        return

    contrastive_ds = ContrastiveDataset(train_path, schema_path, max_examples=20000)
    print(f"  Contrastive examples: {len(contrastive_ds)} (tool-calling only)")
    print(f"  Tool library: {len(contrastive_ds.all_schemas)} unique tools")

    # Build similarity matrix for hard negative mining
    print("  Building schema similarity matrix...")
    sim_matrix = build_schema_similarity_matrix(contrastive_ds.all_schemas, tokenizer)
    contrastive_ds.similarity_matrix = sim_matrix

    # Report hard negative quality
    avg_max_sim = 0
    for i in range(len(contrastive_ds.all_schemas)):
        sims = sim_matrix[i].copy()
        sims[i] = -1
        avg_max_sim += sims.max()
    avg_max_sim /= len(contrastive_ds.all_schemas)
    print(f"  Avg max similarity to nearest tool: {avg_max_sim:.3f}")

    collator = ContrastiveCollator(
        tokenizer, config, contrastive_ds.all_schemas,
        similarity_matrix=sim_matrix,
        num_hard_negatives=config.stage1_5_hard_negatives,
    )
    loader = DataLoader(
        contrastive_ds, batch_size=config.stage1_5_batch_size,
        shuffle=True, collate_fn=collator, drop_last=True,
    )

    # Optional: schema AE dataset for auxiliary loss
    ae_path = ROOT / "data" / "processed" / "schema_ae.jsonl"
    ae_dataset = SchemaAEDataset(ae_path) if ae_path.exists() else None
    ae_collator = SchemaAECollator(tokenizer, config) if ae_dataset else None

    # Freeze
    freeze_for_stage1_5(model)
    model.to(device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=config.stage1_5_lr, weight_decay=0.01)
    scheduler = get_cosine_schedule(optimizer, config.stage1_5_warmup_steps, config.stage1_5_steps)

    print(f"  Steps: {config.stage1_5_steps}")
    print(f"  LR: {config.stage1_5_lr}")
    print(f"  Batch size: {config.stage1_5_batch_size}")
    print(f"  Temperature: {config.stage1_5_temperature}")
    print(f"  Hard negatives/positive: {config.stage1_5_hard_negatives}")
    print(f"  AE auxiliary lam: {config.stage1_5_schema_ae_lambda}")
    print()

    step = 0
    running_loss = 0.0
    running_ae_loss = 0.0
    start_time = time.time()

    while step < config.stage1_5_steps:
        for batch in loader:
            if step >= config.stage1_5_steps:
                break

            query_ids = batch["query_ids"].to(device)
            query_mask = batch["query_mask"].to(device)
            tool_ids = batch["tool_input_ids"].to(device)
            tool_mask = batch["tool_attention_mask"].to(device)
            positive_indices = batch["positive_indices"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config.bf16):
                loss = model.forward_contrastive(
                    query_ids, query_mask,
                    tool_ids, tool_mask,
                    positive_indices,
                    temperature=config.stage1_5_temperature,
                )

            loss.backward()

            # Auxiliary schema AE loss
            if ae_dataset is not None and step % 5 == 0:
                ae_idx = random.sample(range(len(ae_dataset)), min(4, len(ae_dataset)))
                ae_batch = ae_collator([ae_dataset[i] for i in ae_idx])
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config.bf16):
                    ae_loss = model.forward_schema_ae(
                        ae_batch["schema_ids"].to(device),
                        ae_batch["schema_mask"].to(device),
                        ae_batch["target_ids"].to(device),
                        ae_batch["labels"].to(device),
                    )
                    ae_loss = ae_loss * config.stage1_5_schema_ae_lambda
                ae_loss.backward()
                running_ae_loss += ae_loss.item()

            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            step += 1

            if step % config.logging_steps == 0:
                avg_loss = running_loss / config.logging_steps
                avg_ae = running_ae_loss / max(1, config.logging_steps // 5)
                elapsed = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  Step {step}/{config.stage1_5_steps} | "
                    f"contrastive={avg_loss:.4f} ae={avg_ae:.4f} | lr={lr:.2e} | {elapsed:.0f}s"
                )
                running_loss = 0.0
                running_ae_loss = 0.0

            if step % config.save_steps == 0:
                save_path = ROOT / config.output_dir / "stage1_5" / f"step_{step}"
                model.save_checkpoint(str(save_path))

    # Save final checkpoint
    save_path = ROOT / config.output_dir / "stage1_5"
    model.save_checkpoint(str(save_path))
    elapsed = time.time() - start_time
    print(f"\n  Stage 1.5 complete in {elapsed:.0f}s. Saved to {save_path}")


def train_stage2(model: ToolFormerMicro, config: ToolFormerConfig, tokenizer, device: str):
    """Stage 2: End-to-End Tool Calling."""
    print("\n" + "=" * 60)
    print("STAGE 2: End-to-End Tool Calling")
    print("=" * 60)

    # Load training data
    train_path = ROOT / "data" / "processed" / "train_gisting.jsonl"
    if not train_path.exists():
        print(f"  ERROR: Training data not found at {train_path}")
        return
    dataset = ToolFormerDataset(train_path)
    print(f"  Training set: {len(dataset)} examples")

    # Optional: schema AE dataset for auxiliary loss
    ae_path = ROOT / "data" / "processed" / "schema_ae.jsonl"
    ae_dataset = SchemaAEDataset(ae_path) if ae_path.exists() else None
    ae_collator = SchemaAECollator(tokenizer, config) if ae_dataset else None

    collator = ToolFormerCollator(tokenizer, config, system_prompt=SYSTEM_PROMPT)
    loader = DataLoader(
        dataset, batch_size=config.stage2_batch_size,
        shuffle=True, collate_fn=collator, drop_last=True,
    )

    # Unfreeze all for stage 2
    unfreeze_all(model)
    if config.gradient_checkpointing:
        model.gradient_checkpointing = True
    model.to(device)
    model.train()

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.stage2_lr, weight_decay=0.01)

    steps_per_epoch = len(loader) // config.stage2_gradient_accumulation
    total_steps = steps_per_epoch * config.stage2_epochs
    warmup_steps = int(total_steps * config.stage2_warmup_ratio)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)

    print(f"  Epochs: {config.stage2_epochs}")
    print(f"  LR: {config.stage2_lr}")
    print(f"  Batch size: {config.stage2_batch_size} × {config.stage2_gradient_accumulation} = {config.stage2_batch_size * config.stage2_gradient_accumulation}")
    print(f"  Steps/epoch: {steps_per_epoch}, Total: {total_steps}")
    print(f"  Schema AE auxiliary: lam={config.schema_ae_lambda}, freq={config.schema_ae_freq}")
    print()

    global_step = 0
    start_time = time.time()

    for epoch in range(config.stage2_epochs):
        running_loss = 0.0
        running_ae_loss = 0.0
        micro_step = 0

        for batch in loader:
            # Move to device
            tool_ids = batch["tool_input_ids"].to(device)
            tool_mask = batch["tool_attention_mask"].to(device)
            query_ids = batch["query_ids"].to(device)
            query_mask = batch["query_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config.bf16):
                loss, logits = model(tool_ids, tool_mask, query_ids, query_mask, labels)

                # Scale for gradient accumulation
                loss = loss / config.stage2_gradient_accumulation

            loss.backward()
            running_loss += loss.item() * config.stage2_gradient_accumulation
            micro_step += 1

            # Auxiliary schema AE loss
            if (ae_dataset is not None
                    and random.random() < config.schema_ae_freq
                    and micro_step % config.stage2_gradient_accumulation == 0):
                ae_idx = random.sample(range(len(ae_dataset)), min(4, len(ae_dataset)))
                ae_batch = ae_collator([ae_dataset[i] for i in ae_idx])
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config.bf16):
                    ae_loss = model.forward_schema_ae(
                        ae_batch["schema_ids"].to(device),
                        ae_batch["schema_mask"].to(device),
                        ae_batch["target_ids"].to(device),
                        ae_batch["labels"].to(device),
                    )
                    ae_loss = ae_loss * config.schema_ae_lambda
                ae_loss.backward()
                running_ae_loss += ae_loss.item()

            # Optimizer step
            if micro_step % config.stage2_gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.logging_steps == 0:
                    avg_loss = running_loss / config.logging_steps
                    avg_ae = running_ae_loss / max(1, config.logging_steps)
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"  Epoch {epoch+1} Step {global_step}/{total_steps} | "
                        f"loss={avg_loss:.4f} ae={avg_ae:.4f} | lr={lr:.2e} | {elapsed:.0f}s"
                    )
                    running_loss = 0.0
                    running_ae_loss = 0.0

                if global_step % config.save_steps == 0:
                    save_path = ROOT / config.output_dir / "stage2" / f"step_{global_step}"
                    model.save_checkpoint(str(save_path))

        # Save end-of-epoch checkpoint
        save_path = ROOT / config.output_dir / "stage2" / f"epoch_{epoch+1}"
        model.save_checkpoint(str(save_path))
        print(f"  Epoch {epoch+1} complete.")

    # Save final checkpoint
    save_path = ROOT / config.output_dir / "stage2"
    model.save_checkpoint(str(save_path))
    elapsed = time.time() - start_time
    print(f"\n  Stage 2 complete in {elapsed:.0f}s. Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ToolFormerMicro")
    parser.add_argument("--config", type=str, default="configs/tool_former_config.yaml")
    parser.add_argument(
        "--stage", type=str, default="all",
        help="Which stages to run: all, 1, 1.5, 2, or combos like '1+1.5', '1.5+2'",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--no-contrastive", action="store_true",
        help="Skip Stage 1.5 (for ablation: Pipeline A vs Pipeline B)",
    )
    args = parser.parse_args()

    config = ToolFormerConfig.from_yaml(ROOT / args.config)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Parse stages
    if args.stage == "all":
        run_stages = {"1", "1.5", "2"}
    else:
        run_stages = set(args.stage.replace("+", ",").replace(" ", "").split(","))

    if args.no_contrastive:
        run_stages.discard("1.5")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load or create model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = ToolFormerMicro.load_checkpoint(args.resume, device="cpu")
    else:
        print("Initializing from pre-trained Qwen2.5-0.5B...")
        model = ToolFormerMicro.from_pretrained_qwen(config)

    # Print model size
    counts = model.param_count()
    print(f"\nModel size: {counts['total']:,} params ({counts['total'] * 2 / 1e6:.0f} MB fp16)")
    print(f"Stages to run: {sorted(run_stages)}")
    if args.no_contrastive:
        print("  (Pipeline A: no contrastive stage — ablation mode)")

    # Train
    if "1" in run_stages:
        train_stage1(model, config, tokenizer, args.device)

    if "1.5" in run_stages:
        train_stage1_5(model, config, tokenizer, args.device)

    if "2" in run_stages:
        # Clear GPU cache between stages to avoid OOM from fragmentation
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        train_stage2(model, config, tokenizer, args.device)

    print("\nDone!")


if __name__ == "__main__":
    main()
