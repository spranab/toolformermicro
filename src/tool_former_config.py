"""Configuration for ToolFormerMicro.

Architecture defaults match Qwen2.5-0.5B for pre-trained weight initialization.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ToolFormerConfig:
    """Configuration for ToolFormerMicro model.

    Model architecture defaults match Qwen2.5-0.5B (494M params) so we can
    initialize encoder/decoder layers from pre-trained weights.
    """

    # Model architecture (Qwen2.5-0.5B compatible)
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_encoder_layers: int = 6
    num_decoder_layers: int = 12
    num_attention_heads: int = 14
    num_kv_heads: int = 2
    head_dim: int = 64  # hidden_size // num_attention_heads
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    attention_bias: bool = True  # Qwen2.5-0.5B has bias on q/k/v (not o_proj)
    initializer_range: float = 0.02

    # Gisting
    num_gist_tokens: int = 8
    num_context_tools: int = 20
    max_schema_tokens: int = 256

    # Pretrained initialization
    pretrained_model: str = "Qwen/Qwen2.5-0.5B"
    encoder_layer_offset: int = 0   # Qwen layers [0, 6) → encoder
    decoder_layer_offset: int = 6   # Qwen layers [6, 18) → decoder

    # Training Stage 1 (Schema Auto-Encoding)
    stage1_steps: int = 3000
    stage1_lr: float = 2e-4
    stage1_batch_size: int = 8
    stage1_warmup_steps: int = 100

    # Training Stage 1.5 (Contrastive Gist Discrimination)
    stage1_5_steps: int = 2000
    stage1_5_lr: float = 1e-4
    stage1_5_batch_size: int = 8
    stage1_5_warmup_steps: int = 100
    stage1_5_temperature: float = 0.07
    stage1_5_hard_negatives: int = 7  # per positive, within batch
    stage1_5_schema_ae_lambda: float = 0.1  # auxiliary AE loss weight

    # Training Stage 2 (End-to-End Tool Calling)
    stage2_epochs: int = 3
    stage2_lr: float = 1e-4
    stage2_batch_size: int = 4
    stage2_gradient_accumulation: int = 4
    stage2_warmup_ratio: float = 0.05
    schema_ae_lambda: float = 0.1
    schema_ae_freq: float = 0.05
    contrastive_lambda: float = 0.1  # auxiliary contrastive loss weight in stage 2

    # General training
    max_seq_length: int = 2048
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # Output
    output_dir: str = "checkpoints/tool_former"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ToolFormerConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        config = cls()
        for section_name, section in raw.items():
            if isinstance(section, dict):
                for k, v in section.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
            elif hasattr(config, section_name):
                setattr(config, section_name, section)

        return config
