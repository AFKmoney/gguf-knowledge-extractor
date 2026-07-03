"""
GGUF Parser
===========
Reads metadata, tokenizer info, hyperparameters, and tensor inventory
from a GGUF file WITHOUT running inference.

Uses the official `gguf` Python package.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import gguf


@dataclass
class TensorInfo:
    """Summary of a single tensor inside the GGUF."""
    name: str
    dtype: str
    shape: List[int]
    n_elements: int
    size_bytes: int


@dataclass
class GGUFMetadata:
    """Structured view of a GGUF file's metadata."""
    path: str
    file_size_bytes: int
    arch: str
    name: Optional[str]
    description: Optional[str]
    author: Optional[str]
    version: Optional[str]
    organization: Optional[str]
    license: Optional[str]
    vocab_size: int
    context_length: int
    embedding_length: int
    block_count: int
    feed_forward_length: int
    attention_head_count: int
    attention_head_count_kv: int
    attention_layer_norm_rms_epsilon: Optional[float]
    rope_dimension_count: Optional[int]
    rope_freq_base: Optional[float]
    quantization: Optional[str]
    tokenizer_model: Optional[str]
    tokenizer_size: int
    eos_token_id: Optional[int]
    bos_token_id: Optional[int]
    pad_token_id: Optional[int]
    ftype: Optional[str]
    general_raw_keys: Dict[str, Any] = field(default_factory=dict)
    tensors: List[TensorInfo] = field(default_factory=list)


class GGUFParser:
    """Wraps gguf.GGUFReader to expose a clean metadata view."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"GGUF file not found: {self.path}")
        self._reader = gguf.GGUFReader(self.path)
        self._fields_cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Low-level field access
    # ------------------------------------------------------------------ #
    @property
    def fields(self) -> Dict[str, Any]:
        """Return all metadata fields as a dict {name: value}."""
        if self._fields_cache is not None:
            return self._fields_cache

        out: Dict[str, Any] = {}
        for field_name in self._reader.fields.keys():
            val = self._read_field(field_name)
            if val is not None:
                out[field_name] = val
        self._fields_cache = out
        return out

    def _read_field(self, name: str) -> Any:
        """Read a single metadata field, converting numpy types to native."""
        try:
            f = self._reader.get_field(name)
        except Exception:
            return None
        if f is None:
            return None

        # f.types indicates field type, f.parts holds the data
        # `f.parts` is a list of numpy arrays.
        if len(f.types) == 1 and f.types[0] == gguf.GGUFValueType.ARRAY:
            # array field
            arr = f.parts[f.data[0]] if f.data else None
            if arr is None:
                return None
            try:
                return arr.tolist()
            except Exception:
                return str(arr)
        else:
            # scalar field
            try:
                v = f.contents()
                if hasattr(v, "tolist"):
                    return v.tolist()
                if hasattr(v, "item"):
                    return v.item()
                return v
            except Exception:
                # fall back to direct numpy access
                try:
                    part = f.parts[f.data[0]] if f.data else None
                    if part is None:
                        return None
                    if part.size == 1:
                        return part.item()
                    return part.tolist()
                except Exception:
                    return None

    # ------------------------------------------------------------------ #
    # High-level metadata
    # ------------------------------------------------------------------ #
    def metadata(self) -> GGUFMetadata:
        f = self.fields
        get = lambda k, default=None: f.get(k, default)

        # Quantization is usually encoded as part of tensor dtype names
        quant = self._infer_quantization()

        return GGUFMetadata(
            path=self.path,
            file_size_bytes=os.path.getsize(self.path),
            arch=get("general.architecture", "unknown"),
            name=get("general.name"),
            description=get("general.description"),
            author=get("general.author"),
            version=get("general.version"),
            organization=get("general.organization"),
            license=get("general.license"),
            vocab_size=get(f"{get('general.architecture', 'llm')}.vocab_size", 0) or 0,
            context_length=get(f"{get('general.architecture', 'llm')}.context_length", 0) or 0,
            embedding_length=get(f"{get('general.architecture', 'llm')}.embedding_length", 0) or 0,
            block_count=get(f"{get('general.architecture', 'llm')}.block_count", 0) or 0,
            feed_forward_length=get(f"{get('general.architecture', 'llm')}.feed_forward_length", 0) or 0,
            attention_head_count=get(f"{get('general.architecture', 'llm')}.attention.head_count", 0) or 0,
            attention_head_count_kv=get(f"{get('general.architecture', 'llm')}.attention.head_count_kv", 0) or 0,
            attention_layer_norm_rms_epsilon=get(f"{get('general.architecture', 'llm')}.attention.layer_norm_rms_epsilon"),
            rope_dimension_count=get(f"{get('general.architecture', 'llm')}.rope.dimension_count"),
            rope_freq_base=get(f"{get('general.architecture', 'llm')}.rope.freq_base"),
            quantization=quant,
            tokenizer_model=get("tokenizer.ggml.model"),
            tokenizer_size=get("tokenizer.ggml.tokens", []) and len(get("tokenizer.ggml.tokens", [])) or 0,
            eos_token_id=get("tokenizer.ggml.eos_token_id"),
            bos_token_id=get("tokenizer.ggml.bos_token_id"),
            pad_token_id=get("tokenizer.ggml.padding_token_id"),
            ftype=get("general.file_type_name"),
            general_raw_keys={
                k: v for k, v in f.items()
                if k.startswith("general.") and not isinstance(v, (list, dict))
            },
            tensors=self._tensor_inventory(),
        )

    def _infer_quantization(self) -> Optional[str]:
        """Infer quantization scheme from tensor dtypes."""
        seen_quant = set()
        all_types = set()
        for t in self._reader.tensors:
            try:
                dt = gguf.GGMLQuantizationType(t.tensor_type).name
                all_types.add(dt)
                if dt not in ("F16", "F32", "F64", "I8", "I16", "I32", "I64"):
                    seen_quant.add(dt)
            except Exception:
                pass
        if seen_quant:
            return "+".join(sorted(seen_quant)) + " (quantized)"
        if all_types == {"F32"}:
            return "F32 (unquantized)"
        if all_types == {"F16"}:
            return "F16 (half precision)"
        if all_types == {"BF16"}:
            return "BF16 (bfloat16)"
        if all_types:
            return "+".join(sorted(all_types)) + " (mixed precision)"
        return None

    def _tensor_inventory(self) -> List[TensorInfo]:
        out: List[TensorInfo] = []
        for t in self._reader.tensors:
            try:
                dt = gguf.GGMLQuantizationType(t.tensor_type).name
            except Exception:
                dt = str(t.tensor_type)
            # GGUF stores shape in C-order (row-major), e.g. [vocab_size, embed_dim]
            raw_shape = list(t.shape) if hasattr(t, "shape") else []
            shape = [int(s) for s in raw_shape]
            n_elements = 1
            for s in shape:
                n_elements *= int(s)
            out.append(TensorInfo(
                name=str(t.name, encoding="utf-8") if isinstance(t.name, bytes) else str(t.name),
                dtype=dt,
                shape=shape,
                n_elements=n_elements,
                size_bytes=int(t.n_bytes) if hasattr(t, "n_bytes") else 0,
            ))
        return out

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        m = self.metadata()
        d = asdict(m)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
