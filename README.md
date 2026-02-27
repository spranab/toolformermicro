# ToolFormerMicro

**Composable Tool Schema Compression via Gated Cross-Attention**

ToolFormerMicro is a ~428M parameter encoder-decoder model that compresses verbose tool schemas (JSON, 200+ tokens each) into compact 8-token **gist vectors** and uses them for tool-augmented generation via gated cross-attention. Built on Qwen2.5-0.5B weights, it enables composable, cache-friendly tool calling where individual tool representations can be independently cached, reordered, added, or removed without recomputation.

## Key Results

| Metric | Value |
|--------|-------|
| Tool Selection Accuracy (TSA) | **0.818** |
| Parameter F1 (PF1) | **0.759** |
| Value Recall (VR) | **0.917** |
| Exact Match (EM) | **0.580** |
| False Positive Rate (FPR) | **0.000** |
| Gist Encode Time | 18 ms/tool |
| Gist Cache Size | 14 KB/tool |

Evaluated on N=200 examples across 3 splits (seen, held-out, unseen tools). Results identical across all splits.

### Composability Properties (All Verified)

| Property | Result |
|----------|--------|
| Order Independence | TSA 0.860 = 0.860 (delta = 0.000) |
| Scaling (5 → 200 tools) | TSA constant at 0.800, encode sub-linear |
| Cache Invalidation | Bit-identical KV after hot-swap |

## Architecture

```
Input: Tool schema JSON (200+ tokens)
  ↓
[Encoder] 6 Qwen2.5-0.5B layers
  ↓
[GistPooling] K=8 learned queries → 8 gist tokens per tool
  ↓
[Decoder] 12 Qwen2.5-0.5B layers + Gated Cross-Attention
  ↓
Output: Tool call (function name + parameters)
```

**Gated Cross-Attention**: Each decoder layer has a learned gate `g_l` (initialized to 0) that controls how much tool information flows in: `h'' = h' + tanh(g_l) * CrossAttn(LN(h'), M)`. This preserves pre-trained decoder behavior at initialization and lets the model learn to use tool gists during training.

## Quick Start

### Install

```bash
git clone https://github.com/spranab/toolformermicro.git
cd toolformermicro
pip install -r requirements.txt
```

### Training (3-stage curriculum)

```bash
# Stage 1: Schema Auto-Encoding (3K steps)
# Stage 1.5: Contrastive Gist Discrimination (2K steps)
# Stage 2: End-to-End Tool Calling (3 epochs)
python scripts/train/train_tool_former.py --config configs/tool_former_config.yaml
```

Requires 1x GPU with 24GB VRAM (RTX 3090 Ti or similar). Total training time: ~15 hours.

### Data Preparation

```bash
# Download tool-calling datasets
python scripts/data/download_sources.py

# Build unified catalog
python scripts/data/build_catalog.py

# Create train/test splits
python scripts/data/create_splits.py

# Format for training
python scripts/data/format_training_data.py

# Prepare schema auto-encoding pairs
python scripts/data/prepare_schema_ae_data.py
```

### Evaluation

```bash
# Main evaluation (TSA, PF1, EM across 3 splits)
python scripts/eval/eval_tool_former.py --checkpoint checkpoints/tool_former/best

# Composability experiments (order independence, scaling, cache invalidation)
python scripts/eval/eval_composability.py --checkpoint checkpoints/tool_former/best
```

## Project Structure

```
src/                          # Core model code
  tool_former.py              # Encoder-decoder with gated cross-attention
  tool_former_config.py       # Architecture & training hyperparameters
  tool_former_data.py         # Dataset loading & collation
configs/
  tool_former_config.yaml     # Training configuration
scripts/
  train/train_tool_former.py  # 3-stage training loop
  eval/eval_tool_former.py    # Main evaluation
  eval/eval_composability.py  # Composability experiments
  data/                       # Data pipeline
  analysis/                   # Plot generation
eval_results/                 # Paper-ready results (JSON/JSONL)
figures/                      # Paper figures (PDF + PNG)
paper/                        # LaTeX source & compiled PDF
```

## Paper

**ToolFormerMicro: Composable Tool Schema Compression via Gated Cross-Attention**
Pranab Sarkar, 2026

Paper PDF: [paper/main.pdf](paper/main.pdf)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18795100.svg)](https://doi.org/10.5281/zenodo.18795100)

## Citation

```bibtex
@techreport{sarkar2026toolformermicro,
  title={ToolFormerMicro: Composable Tool Schema Compression via Gated Cross-Attention},
  author={Sarkar, Pranab},
  year={2026},
  institution={Zenodo},
  doi={10.5281/zenodo.18795100},
  url={https://doi.org/10.5281/zenodo.18795100},
  doi={10.5281/zenodo.18795100},
  url={https://doi.org/10.5281/zenodo.18795100}
}
```

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
