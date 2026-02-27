"""ToolFormerMicro: Small encoder-decoder model with cross-attention for composable tool calling.

Architecture:
- Tool Encoder: 6 transformer layers + gist pooling → K gist vectors per tool
- Query Decoder: 12 transformer layers with cross-attention to tool gists → generates response
- Total: ~420M params, <1GB at fp16

Key property: Each tool is encoded independently by the encoder, producing K gist vectors.
These vectors are composable via cross-attention in the decoder — adding/removing a tool
only requires encoding that one tool. This is the "shared cache" the architecture enables.

Pre-trained initialization: Encoder layers from Qwen2.5-0.5B layers [0,6),
decoder self-attention + MLP from layers [6,18). Cross-attention layers and gist
pooling are randomly initialized and learned during fine-tuning.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tool_former_config import ToolFormerConfig

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a set of tools. "
    "When the user's request requires a tool, respond with the appropriate tool call. "
    "When no tool is needed, respond normally."
)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _create_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create additive causal mask: 0 for attend, -inf for masked.

    Returns: (1, 1, seq_len, seq_len)
    """
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]


def _create_padding_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert padding mask (1=attend, 0=pad) to additive mask for cross-attention.

    Args:
        attention_mask: (batch, seq_kv)

    Returns: (batch, 1, 1, seq_kv) additive mask
    """
    mask = (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min
    return mask[:, None, None, :]


def _create_combined_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Create combined causal + padding mask for self-attention.

    Args:
        attention_mask: (batch, seq_len) with 1 for real, 0 for pad

    Returns: (batch, 1, seq_len, seq_len) additive mask
    """
    seq_len = attention_mask.shape[1]
    device = attention_mask.device

    causal = _create_causal_mask(seq_len, device, dtype)
    padding = _create_padding_mask(attention_mask, dtype)

    # Broadcasting: causal (1,1,S,S) + padding (B,1,1,S) → (B,1,S,S)
    return causal + padding


# ---------------------------------------------------------------------------
# Core modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE), compatible with Qwen2."""

    def __init__(self, dim: int, max_position: int = 32768, theta: float = 1000000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position = max_position

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor):
        """Compute cos/sin for given positions.

        Args:
            position_ids: (batch, seq_len)

        Returns:
            cos: (batch, seq_len, dim)
            sin: (batch, seq_len, dim)
        """
        # inv_freq: (dim//2,) → (1, dim//2, 1)
        inv_freq = self.inv_freq[None, :, None].float()
        # position_ids: (batch, seq_len) → (batch, 1, seq_len)
        pos = position_ids[:, None, :].float()

        freqs = (inv_freq @ pos).transpose(1, 2)  # (batch, seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)     # (batch, seq_len, dim)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        q: (batch, num_heads, seq, head_dim)
        k: (batch, num_kv_heads, seq, head_dim)
        cos: (batch, seq, head_dim)
        sin: (batch, seq, head_dim)
    """
    cos = cos.unsqueeze(1)  # (batch, 1, seq, head_dim)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA).

    Supports both self-attention (with RoPE) and cross-attention (no RoPE).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        is_cross_attention: bool = False,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.is_cross_attention = is_cross_attention

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_q, hidden_size) — queries
            memory: (batch, seq_kv, hidden_size) — keys/values for cross-attn
            cos, sin: RoPE embeddings (only used for self-attention)
            attention_mask: additive mask (batch, 1, seq_q, seq_kv) or None
        """
        batch_size, seq_q, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross-attention: K,V from memory. Self-attention: K,V from hidden_states.
        kv_input = memory if (self.is_cross_attention and memory is not None) else hidden_states
        seq_kv = kv_input.shape[1]

        k = self.k_proj(kv_input).view(batch_size, seq_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).view(batch_size, seq_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to self-attention only
        if not self.is_cross_attention and cos is not None and sin is not None:
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads for GQA: (batch, kv_heads, seq, dim) → (batch, heads, seq, dim)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=(not self.is_cross_attention and attention_mask is None),
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_q, -1)
        return self.o_proj(attn_output)


class MLP(nn.Module):
    """SwiGLU feedforward network."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class EncoderLayer(nn.Module):
    """Transformer encoder layer: self-attention + FFN."""

    def __init__(self, config: ToolFormerConfig):
        super().__init__()
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads,
            config.num_kv_heads, config.head_dim,
            qkv_bias=config.attention_bias,
        )
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos=cos, sin=sin, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DecoderLayer(nn.Module):
    """Transformer decoder layer: self-attention + cross-attention + FFN.

    Uses a learnable gate (tanh) on cross-attention output, initialized to 0,
    so cross-attention starts as a no-op and gradually learns to contribute.
    This prevents random cross-attention from corrupting pre-trained hidden states.
    """

    def __init__(self, config: ToolFormerConfig):
        super().__init__()
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads,
            config.num_kv_heads, config.head_dim,
            qkv_bias=config.attention_bias,
        )
        self.cross_attn = Attention(
            config.hidden_size, config.num_attention_heads,
            config.num_kv_heads, config.head_dim,
            is_cross_attention=True,
            qkv_bias=config.attention_bias,
        )
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.cross_attn_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Learnable gate for cross-attention, initialized to 0 (tanh(0)=0)
        self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, hidden_states: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor,
        tool_memory: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, cos=cos, sin=sin, attention_mask=self_attn_mask,
        )
        hidden_states = residual + hidden_states

        # Cross-attention to tool memory (gated)
        residual = hidden_states
        hidden_states = self.cross_attn_layernorm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states, memory=tool_memory, attention_mask=cross_attn_mask,
        )
        hidden_states = residual + hidden_states * torch.tanh(self.cross_attn_gate)

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GistPooling(nn.Module):
    """Cross-attention pooling: K learnable queries extract gist from encoded tokens.

    This is the Perceiver/Q-Former approach: K trainable query vectors attend to
    the full encoder output and produce K fixed-size gist vectors per tool.
    """

    def __init__(self, config: ToolFormerConfig):
        super().__init__()
        self.num_gist_tokens = config.num_gist_tokens
        self.gist_queries = nn.Parameter(
            torch.randn(1, config.num_gist_tokens, config.hidden_size) * config.initializer_range
        )
        self.cross_attn = Attention(
            config.hidden_size, config.num_attention_heads,
            config.num_kv_heads, config.head_dim,
            is_cross_attention=True,
            qkv_bias=config.attention_bias,
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, encoded_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool encoded tokens to K gist vectors.

        Args:
            encoded_tokens: (batch, seq_len, hidden_size)
            attention_mask: (batch, 1, 1, seq_len) additive padding mask

        Returns:
            gist: (batch, K, hidden_size)
        """
        batch_size = encoded_tokens.shape[0]
        queries = self.gist_queries.expand(batch_size, -1, -1)
        queries = self.norm(queries)
        gist = self.cross_attn(queries, memory=encoded_tokens, attention_mask=attention_mask)
        return gist


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ToolFormerMicro(nn.Module):
    """Small encoder-decoder model for composable tool calling.

    Tools are encoded independently → K gist vectors per tool (cacheable).
    The decoder uses cross-attention to attend to all tools' gist vectors.
    Adding/removing a tool = encoding/removing one tool's gists.

    ~420M params, <1GB at fp16. Initialized from Qwen2.5-0.5B.
    """

    def __init__(self, config: ToolFormerConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        # Shared token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Rotary embeddings (shared)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
        )

        # Tool encoder
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_encoder_layers)]
        )
        self.encoder_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gist_pool = GistPooling(config)

        # Query decoder
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_decoder_layers)]
        )
        self.decoder_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # LM head
        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    def param_count(self) -> dict[str, int]:
        """Return parameter counts by component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "embeddings": _count(self.embed_tokens),
            "encoder": sum(_count(l) for l in self.encoder_layers),
            "encoder_norm": _count(self.encoder_norm),
            "gist_pool": _count(self.gist_pool),
            "decoder": sum(_count(l) for l in self.decoder_layers),
            "decoder_norm": _count(self.decoder_norm),
            "lm_head": 0 if self.config.tie_word_embeddings else _count(self.lm_head),
            "total": _count(self),
        }

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def encode_tool(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode tool schema(s) into gist vectors. CACHEABLE.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) with 1=real, 0=pad

        Returns:
            gist: (batch, K, hidden_size)
        """
        hidden_states = self.embed_tokens(input_ids)
        compute_dtype = hidden_states.dtype

        # Position IDs and RoPE
        position_ids = torch.arange(
            input_ids.shape[1], device=input_ids.device,
        ).unsqueeze(0).expand(input_ids.shape[0], -1)
        cos, sin = self.rotary_emb(position_ids)

        # Masks
        if attention_mask is not None:
            self_attn_mask = _create_combined_mask(attention_mask, compute_dtype)
            pool_mask = _create_padding_mask(attention_mask, compute_dtype)
        else:
            self_attn_mask = None
            pool_mask = None

        # Forward through encoder layers
        for layer in self.encoder_layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, cos, sin, self_attn_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, cos, sin, self_attn_mask)

        hidden_states = self.encoder_norm(hidden_states)

        # Pool to K gist vectors
        gist = self.gist_pool(hidden_states, pool_mask)
        return gist

    def encode_tools(
        self,
        tool_input_ids: torch.Tensor,
        tool_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode multiple tools independently and concatenate gists.

        Args:
            tool_input_ids: (batch, num_tools, max_schema_len)
            tool_attention_mask: (batch, num_tools, max_schema_len)

        Returns:
            tool_memory: (batch, num_tools * K, hidden_size)
        """
        batch_size, num_tools, seq_len = tool_input_ids.shape

        # Flatten for batch encoding: (batch * num_tools, seq_len)
        flat_ids = tool_input_ids.reshape(batch_size * num_tools, seq_len)
        flat_mask = tool_attention_mask.reshape(batch_size * num_tools, seq_len)

        # Encode all tools at once
        gists = self.encode_tool(flat_ids, flat_mask)  # (B*T, K, D)

        # Reshape: (batch, num_tools * K, D)
        K = self.config.num_gist_tokens
        gists = gists.reshape(batch_size, num_tools * K, -1)
        return gists

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def decode(
        self,
        input_ids: torch.Tensor,
        tool_memory: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode with cross-attention to tool memory.

        Args:
            input_ids: (batch, seq_len)
            tool_memory: (batch, memory_len, hidden_size)
            attention_mask: (batch, seq_len) with 1=real, 0=pad

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        hidden_states = self.embed_tokens(input_ids)
        compute_dtype = hidden_states.dtype

        position_ids = torch.arange(
            input_ids.shape[1], device=input_ids.device,
        ).unsqueeze(0).expand(input_ids.shape[0], -1)
        cos, sin = self.rotary_emb(position_ids)

        # Causal self-attention mask
        if attention_mask is not None:
            self_attn_mask = _create_combined_mask(attention_mask, compute_dtype)
        else:
            self_attn_mask = None

        # No cross-attention mask: gist vectors have no padding
        for layer in self.decoder_layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, cos, sin, tool_memory,
                    self_attn_mask, None,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states, cos, sin, tool_memory,
                    self_attn_mask=self_attn_mask, cross_attn_mask=None,
                )

        hidden_states = self.decoder_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    # ------------------------------------------------------------------
    # Training forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        tool_input_ids: torch.Tensor,
        tool_attention_mask: torch.Tensor,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Full forward: encode tools + decode response.

        Args:
            tool_input_ids: (batch, num_tools, max_schema_len)
            tool_attention_mask: (batch, num_tools, max_schema_len)
            query_ids: (batch, seq_len) — [system_prompt + user_query + target]
            query_mask: (batch, seq_len) padding mask
            labels: (batch, seq_len) with -100 for non-target positions

        Returns:
            loss: scalar (if labels provided)
            logits: (batch, seq_len, vocab_size)
        """
        tool_memory = self.encode_tools(tool_input_ids, tool_attention_mask)
        logits = self.decode(query_ids, tool_memory, query_mask)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return loss, logits

    def forward_contrastive(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        tool_input_ids: torch.Tensor,
        tool_attention_mask: torch.Tensor,
        positive_indices: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """Contrastive loss: query embedding should be close to correct tool's gist.

        InfoNCE: -log( exp(sim(q, t+)/τ) / Σ exp(sim(q, ti)/τ) )

        Args:
            query_ids: (batch, seq_len) — user query tokens
            query_mask: (batch, seq_len) — padding mask
            tool_input_ids: (num_tools, max_schema_len) — ALL tool schemas (flat)
            tool_attention_mask: (num_tools, max_schema_len)
            positive_indices: (batch,) — index into num_tools for each query's correct tool
            temperature: InfoNCE temperature

        Returns:
            loss: scalar contrastive loss
        """
        # Encode all tools → gist vectors
        tool_gists = self.encode_tool(tool_input_ids, tool_attention_mask)  # (T, K, D)

        # Pool gists per tool: mean over K gist tokens → (T, D)
        tool_embeddings = tool_gists.mean(dim=1)  # (T, D)
        tool_embeddings = F.normalize(tool_embeddings, dim=-1)

        # Encode queries: embed + run through decoder self-attention only (no cross-attn)
        # Use mean-pooled encoder output as query embedding for efficiency
        hidden_states = self.embed_tokens(query_ids)
        compute_dtype = hidden_states.dtype

        position_ids = torch.arange(
            query_ids.shape[1], device=query_ids.device,
        ).unsqueeze(0).expand(query_ids.shape[0], -1)
        cos, sin = self.rotary_emb(position_ids)

        if query_mask is not None:
            self_attn_mask = _create_combined_mask(query_mask, compute_dtype)
        else:
            self_attn_mask = None

        # Run through encoder layers (reuse encoder for query embedding)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, cos, sin, self_attn_mask)
        hidden_states = self.encoder_norm(hidden_states)

        # Mean pool (masking padding)
        if query_mask is not None:
            mask_expanded = query_mask.unsqueeze(-1).to(hidden_states.dtype)
            query_embeddings = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            query_embeddings = hidden_states.mean(dim=1)

        query_embeddings = F.normalize(query_embeddings, dim=-1)  # (B, D)

        # InfoNCE: logits = query @ tools^T / temperature
        logits = query_embeddings @ tool_embeddings.T / temperature  # (B, T)
        loss = F.cross_entropy(logits, positive_indices)
        return loss

    def forward_schema_ae(
        self,
        schema_ids: torch.Tensor,
        schema_mask: torch.Tensor,
        target_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for schema auto-encoding (Stage 1).

        1. Encode schema → K gist vectors
        2. Decode from gist vectors to reconstruct schema

        Args:
            schema_ids: (batch, schema_len) — input schema tokens
            schema_mask: (batch, schema_len) — padding mask
            target_ids: (batch, target_len) — reconstruction target tokens
            labels: (batch, target_len) — with -100 for prefix positions

        Returns:
            loss: scalar reconstruction loss
        """
        gists = self.encode_tool(schema_ids, schema_mask)  # (batch, K, D)
        logits = self.decode(target_ids, gists)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        tool_memory: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """Autoregressive generation with pre-computed tool memory.

        Uses simple recompute (no KV cache) — acceptable for small model.

        Args:
            tool_memory: (1, memory_len, D)
            input_ids: (1, prefix_len) — system prompt + user query
            max_new_tokens: max tokens to generate
            eos_token_id: stop generation token

        Returns:
            list of generated token IDs
        """
        generated = []
        current_ids = input_ids

        for _ in range(max_new_tokens):
            logits = self.decode(current_ids, tool_memory)
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)

            if next_token == eos_token_id:
                break

            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token]], device=current_ids.device),
            ], dim=1)

        return generated

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        """Save full model checkpoint."""
        from pathlib import Path
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), save_dir / "model.pt")

        import yaml
        with open(save_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(vars(self.config), f)

        print(f"Saved checkpoint to {save_dir}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "ToolFormerMicro":
        """Load a saved checkpoint."""
        from pathlib import Path
        load_dir = Path(path)

        config = ToolFormerConfig.from_yaml(load_dir / "config.yaml")
        model = cls(config)
        state = torch.load(load_dir / "model.pt", map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            gate_keys = [k for k in missing if "cross_attn_gate" in k]
            other_keys = [k for k in missing if "cross_attn_gate" not in k]
            if gate_keys:
                print(f"  Pre-gate checkpoint: setting {len(gate_keys)} gates to pass-through")
                for layer in model.decoder_layers:
                    if hasattr(layer, 'cross_attn_gate'):
                        layer.cross_attn_gate.data.fill_(10.0)
            if other_keys:
                print(f"  Missing keys (using defaults): {other_keys}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")
        return model

    # ------------------------------------------------------------------
    # Pre-trained initialization
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_qwen(
        cls,
        config: ToolFormerConfig,
        model_name: str | None = None,
    ) -> "ToolFormerMicro":
        """Initialize from Qwen2.5-0.5B pre-trained weights.

        Weight mapping:
        - Qwen layers [0, 6)  → Encoder self-attention + MLP
        - Qwen layers [6, 18) → Decoder self-attention + MLP
        - Cross-attention layers → randomly initialized
        - Gist pooling → randomly initialized
        - Embeddings, norms, LM head → from Qwen

        Total: ~420M params. Cross-attention adds ~22M new params.
        """
        from transformers import AutoModelForCausalLM

        model_name = model_name or config.pretrained_model
        print(f"Loading pre-trained weights from {model_name}...")
        qwen = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

        # Create our model (randomly initialized)
        model = cls(config)

        # Copy embeddings
        model.embed_tokens.weight.data.copy_(qwen.model.embed_tokens.weight.data)

        # Copy encoder layers from Qwen layers [offset, offset + num_encoder_layers)
        for i, layer in enumerate(model.encoder_layers):
            src = qwen.model.layers[config.encoder_layer_offset + i]
            _copy_qwen_to_encoder_layer(src, layer)

        # Copy encoder norm
        model.encoder_norm.weight.data.copy_(qwen.model.norm.weight.data)

        # Copy decoder layers from Qwen layers [offset, offset + num_decoder_layers)
        for i, layer in enumerate(model.decoder_layers):
            src = qwen.model.layers[config.decoder_layer_offset + i]
            _copy_qwen_to_decoder_layer(src, layer)

        # Copy decoder norm
        model.decoder_norm.weight.data.copy_(qwen.model.norm.weight.data)

        # LM head (if not tied, copy separately)
        if not config.tie_word_embeddings and hasattr(qwen, "lm_head"):
            model.lm_head.weight.data.copy_(qwen.lm_head.weight.data)

        del qwen
        torch.cuda.empty_cache()

        total = sum(p.numel() for p in model.parameters())
        print(f"Initialized ToolFormerMicro: {total:,} params ({total * 2 / 1e6:.0f} MB fp16)")
        print(f"  Encoder: {config.num_encoder_layers} layers (pre-trained)")
        print(f"  Decoder: {config.num_decoder_layers} layers (self-attn+MLP pre-trained, cross-attn random)")
        print(f"  Gist pooling: {config.num_gist_tokens} tokens (random)")
        return model


# ---------------------------------------------------------------------------
# Weight copy helpers
# ---------------------------------------------------------------------------

def _copy_attn_weights(src_attn, dst_attn):
    """Copy attention weights and biases (q/k/v/o projections)."""
    dst_attn.q_proj.weight.data.copy_(src_attn.q_proj.weight.data)
    dst_attn.k_proj.weight.data.copy_(src_attn.k_proj.weight.data)
    dst_attn.v_proj.weight.data.copy_(src_attn.v_proj.weight.data)
    dst_attn.o_proj.weight.data.copy_(src_attn.o_proj.weight.data)
    # Copy biases if both src and dst have them
    if hasattr(src_attn.q_proj, "bias") and src_attn.q_proj.bias is not None and dst_attn.q_proj.bias is not None:
        dst_attn.q_proj.bias.data.copy_(src_attn.q_proj.bias.data)
    if hasattr(src_attn.k_proj, "bias") and src_attn.k_proj.bias is not None and dst_attn.k_proj.bias is not None:
        dst_attn.k_proj.bias.data.copy_(src_attn.k_proj.bias.data)
    if hasattr(src_attn.v_proj, "bias") and src_attn.v_proj.bias is not None and dst_attn.v_proj.bias is not None:
        dst_attn.v_proj.bias.data.copy_(src_attn.v_proj.bias.data)


def _copy_mlp_weights(src_mlp, dst_mlp):
    """Copy MLP weights (gate/up/down projections)."""
    dst_mlp.gate_proj.weight.data.copy_(src_mlp.gate_proj.weight.data)
    dst_mlp.up_proj.weight.data.copy_(src_mlp.up_proj.weight.data)
    dst_mlp.down_proj.weight.data.copy_(src_mlp.down_proj.weight.data)


def _copy_qwen_to_encoder_layer(src, dst: EncoderLayer):
    """Copy Qwen2DecoderLayer → EncoderLayer (self-attn + MLP + norms)."""
    _copy_attn_weights(src.self_attn, dst.self_attn)
    _copy_mlp_weights(src.mlp, dst.mlp)
    dst.input_layernorm.weight.data.copy_(src.input_layernorm.weight.data)
    dst.post_attention_layernorm.weight.data.copy_(src.post_attention_layernorm.weight.data)


def _copy_qwen_to_decoder_layer(src, dst: DecoderLayer):
    """Copy Qwen2DecoderLayer → DecoderLayer (self-attn + MLP + norms only).

    Cross-attention weights and cross_attn_layernorm stay randomly initialized.
    """
    _copy_attn_weights(src.self_attn, dst.self_attn)
    _copy_mlp_weights(src.mlp, dst.mlp)
    dst.input_layernorm.weight.data.copy_(src.input_layernorm.weight.data)
    dst.post_attention_layernorm.weight.data.copy_(src.post_attention_layernorm.weight.data)
