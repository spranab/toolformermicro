"""Dataset and collator for ToolFormerMicro training.

Handles loading tool-calling examples with per-tool schema encoding.
Each example has 20 tool schemas (encoded independently) + user query + target response.

Includes contrastive training support (Stage 1.5) with hard negative mining.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .tool_former_config import ToolFormerConfig


class ToolFormerDataset(Dataset):
    """Dataset for ToolFormerMicro training.

    Each example:
    - tool_schemas: list of JSON schema strings (20 tools)
    - user_query: the user's request
    - assistant_response: target response (tool call or text)
    - is_tool_call: whether this is a tool-calling example
    """

    def __init__(self, path: str | Path, max_examples: int | None = None):
        self.examples = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SchemaAEDataset(Dataset):
    """Dataset for Stage 1: schema auto-encoding.

    Each example is a single tool schema text, used for encode→reconstruct training.
    """

    def __init__(self, path: str | Path):
        self.examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ToolFormerCollator:
    """Collator for ToolFormerMicro training.

    Tokenizes tool schemas, user queries, and targets.
    Returns padded tensors ready for model.forward().
    """

    def __init__(self, tokenizer, config: ToolFormerConfig, system_prompt: str = ""):
        self.tokenizer = tokenizer
        self.config = config
        self.system_prompt = system_prompt
        self._sys_ids = tokenizer.encode(system_prompt, add_special_tokens=False) if system_prompt else []

    def __call__(self, batch: list[dict]) -> dict:
        tool_ids_batch = []
        tool_mask_batch = []
        query_ids_batch = []
        query_mask_batch = []
        labels_batch = []

        max_query_len = 0
        max_schema_len = self.config.max_schema_tokens

        # First pass: tokenize everything and find max lengths
        tokenized_batch = []
        for example in batch:
            # Tokenize each tool schema
            schemas_ids = []
            schemas_mask = []
            for schema_text in example["tool_schemas"]:
                ids = self.tokenizer.encode(
                    schema_text, add_special_tokens=False,
                    max_length=max_schema_len, truncation=True,
                )
                # Pad to max_schema_len
                pad_len = max_schema_len - len(ids)
                mask = [1] * len(ids) + [0] * pad_len
                ids = ids + [self.tokenizer.pad_token_id or 0] * pad_len
                schemas_ids.append(ids)
                schemas_mask.append(mask)

            # Tokenize query: [system_prompt] [user_query]
            user_ids = self.tokenizer.encode(
                example["user_query"], add_special_tokens=False,
                max_length=512, truncation=True,
            )
            prefix_ids = self._sys_ids + user_ids

            # Tokenize target
            target_ids = self.tokenizer.encode(
                example["assistant_response"], add_special_tokens=False,
                max_length=512, truncation=True,
            )

            # Full sequence: [prefix] [target] [eos]
            eos_id = self.tokenizer.eos_token_id or 0
            full_ids = prefix_ids + target_ids + [eos_id]
            prefix_len = len(prefix_ids)

            # Labels: -100 for prefix, actual IDs for target+eos
            labels = [-100] * prefix_len + target_ids + [eos_id]

            max_query_len = max(max_query_len, len(full_ids))
            tokenized_batch.append({
                "schemas_ids": schemas_ids,
                "schemas_mask": schemas_mask,
                "full_ids": full_ids,
                "labels": labels,
            })

        # Second pass: pad query sequences and build tensors
        for item in tokenized_batch:
            # Tool schemas (already padded to max_schema_len)
            tool_ids_batch.append(torch.tensor(item["schemas_ids"], dtype=torch.long))
            tool_mask_batch.append(torch.tensor(item["schemas_mask"], dtype=torch.long))

            # Query sequence (pad to max_query_len)
            pad_len = max_query_len - len(item["full_ids"])
            pad_id = self.tokenizer.pad_token_id or 0

            q_ids = item["full_ids"] + [pad_id] * pad_len
            q_mask = [1] * len(item["full_ids"]) + [0] * pad_len
            q_labels = item["labels"] + [-100] * pad_len

            query_ids_batch.append(torch.tensor(q_ids, dtype=torch.long))
            query_mask_batch.append(torch.tensor(q_mask, dtype=torch.long))
            labels_batch.append(torch.tensor(q_labels, dtype=torch.long))

        return {
            "tool_input_ids": torch.stack(tool_ids_batch),       # (B, T, S)
            "tool_attention_mask": torch.stack(tool_mask_batch),  # (B, T, S)
            "query_ids": torch.stack(query_ids_batch),            # (B, L)
            "query_mask": torch.stack(query_mask_batch),          # (B, L)
            "labels": torch.stack(labels_batch),                  # (B, L)
        }


class SchemaAECollator:
    """Collator for Stage 1 schema auto-encoding.

    Each batch item: encode schema → reconstruct from gist vectors.
    """

    def __init__(self, tokenizer, config: ToolFormerConfig):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, batch: list[dict]) -> dict:
        schema_ids_batch = []
        schema_mask_batch = []
        target_ids_batch = []
        labels_batch = []

        max_len = self.config.max_schema_tokens

        for example in batch:
            schema_text = example["schema_text"]
            ids = self.tokenizer.encode(
                schema_text, add_special_tokens=False,
                max_length=max_len, truncation=True,
            )

            # Schema input (for encoder)
            pad_len = max_len - len(ids)
            pad_id = self.tokenizer.pad_token_id or 0
            s_ids = ids + [pad_id] * pad_len
            s_mask = [1] * len(ids) + [0] * pad_len

            # Target (for decoder reconstruction) — same tokens, unpadded
            eos_id = self.tokenizer.eos_token_id or 0
            t_ids = ids + [eos_id]
            t_labels = ids + [eos_id]

            schema_ids_batch.append(torch.tensor(s_ids, dtype=torch.long))
            schema_mask_batch.append(torch.tensor(s_mask, dtype=torch.long))
            target_ids_batch.append(torch.tensor(t_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(t_labels, dtype=torch.long))

        # Pad target sequences to same length
        max_target = max(t.shape[0] for t in target_ids_batch)
        pad_id = self.tokenizer.pad_token_id or 0

        for i in range(len(target_ids_batch)):
            pad_len = max_target - target_ids_batch[i].shape[0]
            if pad_len > 0:
                target_ids_batch[i] = torch.cat([
                    target_ids_batch[i],
                    torch.full((pad_len,), pad_id, dtype=torch.long),
                ])
                labels_batch[i] = torch.cat([
                    labels_batch[i],
                    torch.full((pad_len,), -100, dtype=torch.long),
                ])

        return {
            "schema_ids": torch.stack(schema_ids_batch),     # (B, S)
            "schema_mask": torch.stack(schema_mask_batch),   # (B, S)
            "target_ids": torch.stack(target_ids_batch),     # (B, T)
            "labels": torch.stack(labels_batch),             # (B, T)
        }


# ---------------------------------------------------------------------------
# Stage 1.5: Contrastive training with hard negatives
# ---------------------------------------------------------------------------


def build_schema_similarity_matrix(schemas: list[dict], tokenizer) -> np.ndarray:
    """Build pairwise similarity matrix for hard negative mining.

    Uses token-level Jaccard similarity between schema texts as a fast proxy.
    Schemas with high Jaccard but different names are the hardest negatives.

    Args:
        schemas: list of {"schema_text": str, "tool_name": str}
        tokenizer: tokenizer for token-level comparison

    Returns:
        similarity: (N, N) numpy array of pairwise similarities
    """
    # Tokenize all schemas
    token_sets = []
    for s in schemas:
        ids = tokenizer.encode(s["schema_text"], add_special_tokens=False)
        token_sets.append(set(ids))

    n = len(schemas)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            intersection = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            score = intersection / max(union, 1)
            sim[i, j] = score
            sim[j, i] = score

    return sim


class ContrastiveDataset(Dataset):
    """Dataset for Stage 1.5: Contrastive Gist Discrimination.

    Each item: a tool-calling example with its correct tool index.
    The collator handles batching tools + mining hard negatives.

    Loads from train_gisting.jsonl (same format as ToolFormerDataset)
    but only keeps tool-calling examples (is_tool_call=True).
    """

    def __init__(
        self,
        data_path: str | Path,
        schema_path: str | Path,
        max_examples: int | None = None,
    ):
        """
        Args:
            data_path: path to train_gisting.jsonl
            schema_path: path to schema_ae.jsonl (all unique schemas)
            max_examples: optional limit
        """
        # Load all unique schemas (the "tool library")
        self.all_schemas: list[dict] = []
        self.name_to_idx: dict[str, int] = {}
        with open(schema_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    idx = len(self.all_schemas)
                    self.all_schemas.append(entry)
                    self.name_to_idx[entry["tool_name"]] = idx

        # Load training examples (tool-calling only)
        self.examples = []
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_examples and len(self.examples) >= max_examples:
                    break
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if not ex.get("is_tool_call"):
                    continue
                target_tool = ex.get("target_tool")
                if target_tool and target_tool in self.name_to_idx:
                    self.examples.append(ex)

        # Pre-computed similarity will be set externally
        self.similarity_matrix: np.ndarray | None = None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ContrastiveCollator:
    """Collator for Stage 1.5 contrastive training.

    For each batch of queries:
    1. Collect the positive tool for each query
    2. Sample hard negatives (tools with high schema similarity to the positive)
    3. Build a shared tool pool for the batch
    4. Return query tokens, tool tokens, and positive indices

    This is efficient: all tools in the pool are encoded once, shared across queries.
    """

    def __init__(
        self,
        tokenizer,
        config: ToolFormerConfig,
        all_schemas: list[dict],
        similarity_matrix: np.ndarray | None = None,
        num_hard_negatives: int = 7,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.all_schemas = all_schemas
        self.similarity_matrix = similarity_matrix
        self.num_hard_negatives = num_hard_negatives
        self._name_to_idx = {s["tool_name"]: i for i, s in enumerate(all_schemas)}

    def _sample_negatives(self, positive_idx: int, k: int) -> list[int]:
        """Sample k hard negative tool indices for a given positive."""
        n = len(self.all_schemas)
        if self.similarity_matrix is not None:
            # Hard negatives: tools most similar to positive (but not positive itself)
            sims = self.similarity_matrix[positive_idx].copy()
            sims[positive_idx] = -1  # exclude self
            # Top-k most similar
            top_k = min(k * 3, n - 1)  # sample from top candidates
            candidates = np.argsort(sims)[-top_k:]
            candidates = [c for c in candidates if c != positive_idx]
            if len(candidates) >= k:
                return list(np.random.choice(candidates, size=k, replace=False))
            else:
                # Not enough hard negatives, fill with random
                remaining = k - len(candidates)
                all_indices = set(range(n)) - {positive_idx} - set(candidates)
                extra = random.sample(list(all_indices), min(remaining, len(all_indices)))
                return candidates + extra
        else:
            # Random negatives
            all_indices = list(set(range(n)) - {positive_idx})
            return random.sample(all_indices, min(k, len(all_indices)))

    def __call__(self, batch: list[dict]) -> dict:
        max_schema_len = self.config.max_schema_tokens
        pad_id = self.tokenizer.pad_token_id or 0

        # Collect unique tool indices needed for this batch
        tool_pool_indices = []  # ordered list of tool indices in the pool
        tool_pool_set = set()
        positive_indices = []  # for each query, index into tool_pool

        for ex in batch:
            target = ex["target_tool"]
            pos_idx = self._name_to_idx.get(target, 0)

            # Add positive to pool
            if pos_idx not in tool_pool_set:
                tool_pool_set.add(pos_idx)
                tool_pool_indices.append(pos_idx)

            # Add hard negatives to pool
            negs = self._sample_negatives(pos_idx, self.num_hard_negatives)
            for neg_idx in negs:
                if neg_idx not in tool_pool_set:
                    tool_pool_set.add(neg_idx)
                    tool_pool_indices.append(neg_idx)

        # Build index mapping: global_idx → pool_position
        idx_to_pool = {idx: pos for pos, idx in enumerate(tool_pool_indices)}

        # Now build positive indices per query
        for ex in batch:
            target = ex["target_tool"]
            pos_idx = self._name_to_idx.get(target, 0)
            positive_indices.append(idx_to_pool[pos_idx])

        # Tokenize tool pool
        tool_ids_list = []
        tool_mask_list = []
        for global_idx in tool_pool_indices:
            schema_text = self.all_schemas[global_idx]["schema_text"]
            ids = self.tokenizer.encode(
                schema_text, add_special_tokens=False,
                max_length=max_schema_len, truncation=True,
            )
            pad_len = max_schema_len - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [pad_id] * pad_len
            tool_ids_list.append(ids)
            tool_mask_list.append(mask)

        # Tokenize queries
        query_ids_list = []
        query_mask_list = []
        max_query_len = 0

        for ex in batch:
            ids = self.tokenizer.encode(
                ex["user_query"], add_special_tokens=False,
                max_length=512, truncation=True,
            )
            query_ids_list.append(ids)
            max_query_len = max(max_query_len, len(ids))

        # Pad queries
        for i in range(len(query_ids_list)):
            pad_len = max_query_len - len(query_ids_list[i])
            mask = [1] * len(query_ids_list[i]) + [0] * pad_len
            query_ids_list[i] = query_ids_list[i] + [pad_id] * pad_len
            query_mask_list.append(mask)

        return {
            "query_ids": torch.tensor(query_ids_list, dtype=torch.long),          # (B, Q)
            "query_mask": torch.tensor(query_mask_list, dtype=torch.long),         # (B, Q)
            "tool_input_ids": torch.tensor(tool_ids_list, dtype=torch.long),       # (T, S)
            "tool_attention_mask": torch.tensor(tool_mask_list, dtype=torch.long),  # (T, S)
            "positive_indices": torch.tensor(positive_indices, dtype=torch.long),   # (B,)
        }
