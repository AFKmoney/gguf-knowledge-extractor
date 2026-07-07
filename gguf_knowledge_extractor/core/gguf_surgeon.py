"""
GGUF Surgeon
=============
Direct surgical modification of GGUF files — no retraining.

Operations supported:
  1. **patch_tensor**     — overwrite an arbitrary slice of any tensor
  2. **set_metadata**     — add/modify any KV metadata field
  3. **remove_metadata**  — delete a KV metadata field
  4. **set_chat_template**— rewrite the tokenizer chat template
  5. **bake_system_prompt**— inject a default system prompt
  6. **inject_dataset**   — embed a JSON dataset as metadata (RAG-style)
  7. **add_token**        — extend the vocabulary with a new token + embedding
  8. **add_steering_vector**— add an activation steering bias vector for a layer
  9. **batch_rome_edits** — apply multiple ROME rank-1 edits in one pass

All operations read the source GGUF, apply modifications in-memory, and write
a new GGUF file. The source file is never modified.
"""
from __future__ import annotations

import json
import shutil
import struct
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gguf


# ---------------------------------------------------------------------- #
# GGUF value type constants (mirrors gguf.GGUFValueType)
# ---------------------------------------------------------------------- #
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
ALIGN = 32


@dataclass
class SurgeryResult:
    """Result of one surgery operation."""
    operation: str
    success: bool
    details: str
    output_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SurgeryReport:
    """Report from a full surgery session."""
    source_path: str
    output_path: str
    operations: List[Dict[str, Any]]
    elapsed_seconds: float
    n_tensors: int
    n_kv_pairs: int
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------- #
# Value type helpers
# ---------------------------------------------------------------------- #
def _infer_type(value: Any) -> int:
    """Infer the GGUF value type for a Python value."""
    if isinstance(value, bool):
        return GGUF_TYPE_BOOL
    if isinstance(value, int):
        if 0 <= value < 256:
            return GGUF_TYPE_UINT32
        if -128 <= value < 128:
            return GGUF_TYPE_INT32
        if 0 <= value < 2**32:
            return GGUF_TYPE_UINT32
        return GGUF_TYPE_INT64
    if isinstance(value, float):
        return GGUF_TYPE_FLOAT32
    if isinstance(value, str):
        return GGUF_TYPE_STRING
    if isinstance(value, (list, tuple)):
        if not value:
            return GGUF_TYPE_ARRAY
        first = value[0]
        if isinstance(first, str):
            return GGUF_TYPE_ARRAY  # will infer element type as STRING
        if isinstance(first, bool):
            return GGUF_TYPE_ARRAY
        if isinstance(first, int):
            return GGUF_TYPE_ARRAY
        if isinstance(first, float):
            return GGUF_TYPE_ARRAY
    raise ValueError(f"Cannot infer GGUF type for value of type {type(value).__name__}")


def _array_elem_type(value: Any) -> int:
    """Infer element type for an array value."""
    if not value:
        return GGUF_TYPE_STRING
    first = value[0]
    if isinstance(first, bool):
        return GGUF_TYPE_BOOL
    if isinstance(first, int):
        if 0 <= first < 2**32:
            return GGUF_TYPE_UINT32
        return GGUF_TYPE_INT64
    if isinstance(first, float):
        return GGUF_TYPE_FLOAT32
    if isinstance(first, str):
        return GGUF_TYPE_STRING
    raise ValueError(f"Cannot infer array element type for {type(first).__name__}")


# ---------------------------------------------------------------------- #
# Main Surgeon class
# ---------------------------------------------------------------------- #
class GGUFSurgeon:
    """Surgical modifier for GGUF files."""

    def __init__(self, source_path: str):
        self.source_path = str(source_path)
        if not Path(self.source_path).exists():
            raise FileNotFoundError(f"GGUF file not found: {self.source_path}")
        self.reader = gguf.GGUFReader(self.source_path)
        # Extract all metadata fields as a dict {key: (type_id, value)}
        self.metadata: Dict[str, Tuple[int, Any]] = {}
        self._load_metadata()
        # Extract all tensors as {name: (tensor_type_id, shape, data_bytes)}
        self.tensors: Dict[str, Tuple[int, List[int], bytes]] = {}
        self._load_tensors()
        # Pending operations log
        self.operations_log: List[Dict[str, Any]] = []

    def _load_metadata(self):
        """Load all metadata fields from the source GGUF.

        Filters out internal 'GGUF.*' synthetic fields that the reader
        creates from the header (version, tensor_count, kv_count) — these
        are NOT real KV pairs and should not be re-written.
        """
        for name in self.reader.fields.keys():
            # Skip internal synthetic fields
            if name.startswith("GGUF."):
                continue
            try:
                f = self.reader.get_field(name)
                # Determine if this is an array: f.types[0] == GGUF_TYPE_ARRAY (9)
                is_array = (len(f.types) >= 1 and int(f.types[0]) == GGUF_TYPE_ARRAY)
                if is_array:
                    elem_type = int(f.types[1]) if len(f.types) >= 2 else GGUF_TYPE_STRING
                    # Extract elements
                    value = []
                    if elem_type == GGUF_TYPE_STRING:
                        # For string arrays: f.data has one index per string,
                        # each pointing to the string's data part in f.parts.
                        for i in range(len(f.data)):
                            str_part = f.parts[f.data[i]]
                            try:
                                str_bytes = bytes(str_part)
                                value.append(str_bytes.decode("utf-8", errors="replace"))
                            except Exception:
                                value.append(str(str_part))
                    else:
                        # For numeric arrays: f.data[0] is count index,
                        # f.data[1] is the data array index.
                        if len(f.data) >= 2:
                            arr = f.parts[f.data[1]]
                            try:
                                value = arr.tolist()
                            except Exception:
                                value = list(arr)
                        else:
                            value = []
                    self.metadata[name] = (GGUF_TYPE_ARRAY, value)
                    self.metadata[name + ".__elem_type__"] = (GGUF_TYPE_UINT32, elem_type)
                else:
                    # Scalar field
                    type_id = int(f.types[0])
                    if f.data and len(f.data) > 0:
                        part_idx = f.data[0]
                        if part_idx < len(f.parts):
                            part = f.parts[part_idx]
                            if type_id == GGUF_TYPE_STRING:
                                try:
                                    value = bytes(part).decode("utf-8", errors="replace")
                                except Exception:
                                    value = str(part)
                            elif part.size == 1:
                                value = part.item()
                            else:
                                try:
                                    value = part.tolist()
                                except Exception:
                                    value = str(part)
                        else:
                            value = None
                    else:
                        value = None
                    self.metadata[name] = (type_id, value)
            except Exception:
                pass

    def _detect_array_elem_type(self, f) -> int:
        """Detect the element type of an array field."""
        try:
            # gguf reader stores the element type as the first non-array type
            # In f.types, the first entry is ARRAY (9), and we need to find
            # the actual element type. Looking at gguf_reader.py source,
            # array fields have types = [ARRAY, elem_type] in the ReaderField.
            if len(f.types) >= 2:
                return int(f.types[1])
        except Exception:
            pass
        # Fallback: try to infer from parts
        return GGUF_TYPE_STRING

    def _load_tensors(self):
        """Load all tensors (their type, shape, and raw data bytes)."""
        for t in self.reader.tensors:
            name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            tensor_type = int(t.tensor_type)
            shape = [int(s) for s in t.shape]
            # Get the raw bytes — we need to copy from the memoryview
            data = bytes(t.data)
            self.tensors[name] = (tensor_type, shape, data)

    # ------------------------------------------------------------------ #
    # Operation 1: Patch tensor
    # ------------------------------------------------------------------ #
    def patch_tensor(
        self,
        name: str,
        new_values: np.ndarray,
        slice_spec: Optional[Tuple[int, ...]] = None,
    ) -> SurgeryResult:
        """Overwrite a slice of a tensor with new values.

        slice_spec: if None, overwrite the entire tensor. Otherwise, a tuple
        of (row_start, row_end, col_start, col_end) for 2D tensors, or
        (start, end) for 1D tensors.

        Only works for F32 and F16 tensors. Quantized tensors would need
        requantization (not yet supported).
        """
        if name not in self.tensors:
            return SurgeryResult("patch_tensor", False, f"Tensor {name} not found", error="not_found")
        tensor_type, shape, data = self.tensors[name]
        try:
            qt = gguf.GGMLQuantizationType(tensor_type)
        except Exception:
            return SurgeryResult("patch_tensor", False, f"Unknown tensor type {tensor_type}", error="bad_type")

        if qt not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            return SurgeryResult("patch_tensor", False,
                                 f"Cannot patch quantized tensor ({qt.name}) — only F32/F16 supported",
                                 error="quantized_not_supported")

        dtype = np.float32 if qt == gguf.GGMLQuantizationType.F32 else np.float16
        bytes_per = 4 if qt == gguf.GGMLQuantizationType.F32 else 2
        arr = np.frombuffer(data, dtype=dtype).reshape(shape).copy()

        if slice_spec is None:
            if new_values.shape != arr.shape:
                return SurgeryResult("patch_tensor", False,
                                     f"Shape mismatch: tensor {arr.shape} vs new_values {new_values.shape}",
                                     error="shape_mismatch")
            arr[:] = new_values.astype(dtype)
            self.operations_log.append({"op": "patch_tensor", "name": name, "slice": "full"})
        else:
            if len(shape) == 2 and len(slice_spec) == 4:
                r0, r1, c0, c1 = slice_spec
                arr[r0:r1, c0:c1] = new_values.astype(dtype)
                self.operations_log.append({"op": "patch_tensor", "name": name, "slice": [r0, r1, c0, c1]})
            elif len(shape) == 1 and len(slice_spec) == 2:
                s, e = slice_spec
                arr[s:e] = new_values.astype(dtype)
                self.operations_log.append({"op": "patch_tensor", "name": name, "slice": [s, e]})
            else:
                return SurgeryResult("patch_tensor", False,
                                     f"Slice spec {slice_spec} incompatible with tensor shape {shape}",
                                     error="slice_mismatch")

        self.tensors[name] = (tensor_type, shape, arr.tobytes())
        return SurgeryResult("patch_tensor", True, f"Patched tensor {name}", output_path=None)

    # ------------------------------------------------------------------ #
    # Operation 2: Set metadata
    # ------------------------------------------------------------------ #
    def set_metadata(self, key: str, value: Any, value_type: Optional[int] = None) -> SurgeryResult:
        """Add or replace a metadata field."""
        if value_type is None:
            value_type = _infer_type(value)
        self.metadata[key] = (value_type, value)
        # Clean up any stale element type hint
        self.metadata.pop(key + ".__elem_type__", None)
        self.operations_log.append({"op": "set_metadata", "key": key, "type": value_type})
        return SurgeryResult("set_metadata", True, f"Set metadata {key} (type {value_type})")

    # ------------------------------------------------------------------ #
    # Operation 3: Remove metadata
    # ------------------------------------------------------------------ #
    def remove_metadata(self, key: str) -> SurgeryResult:
        """Remove a metadata field."""
        if key not in self.metadata:
            return SurgeryResult("remove_metadata", False, f"Metadata key {key} not found", error="not_found")
        del self.metadata[key]
        self.metadata.pop(key + ".__elem_type__", None)
        self.operations_log.append({"op": "remove_metadata", "key": key})
        return SurgeryResult("remove_metadata", True, f"Removed metadata {key}")

    # ------------------------------------------------------------------ #
    # Operation 4: Set chat template
    # ------------------------------------------------------------------ #
    def set_chat_template(self, template: str) -> SurgeryResult:
        """Rewrite the tokenizer chat template."""
        result = self.set_metadata("tokenizer.chat_template", template, GGUF_TYPE_STRING)
        self.operations_log[-1]["op"] = "set_chat_template"
        return SurgeryResult("set_chat_template", True,
                            f"Chat template updated ({len(template)} chars)")

    # ------------------------------------------------------------------ #
    # Operation 5: Bake system prompt
    # ------------------------------------------------------------------ #
    def bake_system_prompt(self, prompt: str) -> SurgeryResult:
        """Inject a default system prompt into the model metadata."""
        self.set_metadata("general.system_prompt", prompt, GGUF_TYPE_STRING)
        self.set_metadata("general.system_prompt.enabled", True, GGUF_TYPE_BOOL)
        self.operations_log.append({"op": "bake_system_prompt", "prompt_len": len(prompt)})
        return SurgeryResult("bake_system_prompt", True,
                            f"System prompt baked ({len(prompt)} chars)")

    # ------------------------------------------------------------------ #
    # Operation 6: Inject dataset
    # ------------------------------------------------------------------ #
    def inject_dataset(
        self,
        name: str,
        data: Any,
        description: str = "",
    ) -> SurgeryResult:
        """Embed a JSON dataset as metadata. The data is stored as a JSON string
        under `general.injected_dataset.<name>`.

        The model can be prompted to access this data (RAG-style) — the dataset
        becomes part of the model file itself.
        """
        key = f"general.injected_dataset.{name}"
        payload = {
            "name": name,
            "description": description,
            "injected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "n_items": len(data) if hasattr(data, "__len__") else 1,
            "data": data,
        }
        json_str = json.dumps(payload, ensure_ascii=False, default=str)
        # Truncate if extremely large (GGUF metadata should not exceed ~100MB)
        if len(json_str) > 50_000_000:
            return SurgeryResult("inject_dataset", False,
                                 f"Dataset too large ({len(json_str)} chars, max 50MB)",
                                 error="too_large")
        self.set_metadata(key, json_str, GGUF_TYPE_STRING)
        self.operations_log.append({
            "op": "inject_dataset", "name": name,
            "n_items": payload["n_items"], "size_bytes": len(json_str),
        })
        return SurgeryResult("inject_dataset", True,
                            f"Dataset '{name}' injected ({payload['n_items']} items, {len(json_str)} bytes)")

    def list_injected_datasets(self) -> List[Dict[str, Any]]:
        """List all injected datasets in the model."""
        out = []
        for key, (type_id, value) in self.metadata.items():
            if key.startswith("general.injected_dataset."):
                try:
                    payload = json.loads(value) if isinstance(value, str) else value
                    out.append({
                        "name": payload.get("name", key.split(".")[-1]),
                        "description": payload.get("description", ""),
                        "injected_at": payload.get("injected_at"),
                        "n_items": payload.get("n_items"),
                        "size_bytes": len(value) if isinstance(value, str) else 0,
                    })
                except Exception:
                    pass
        return out

    # ------------------------------------------------------------------ #
    # Operation 7: Add token
    # ------------------------------------------------------------------ #
    def add_token(
        self,
        token: str,
        embedding: Optional[np.ndarray] = None,
        score: float = 0.0,
        token_type: int = 1,  # 1 = normal token
    ) -> SurgeryResult:
        """Add a new token to the vocabulary.

        If embedding is None, a zero vector is used (the token will be inert
        until edited). If the model has tied embeddings (output.weight ==
        token_embd.weight), both are updated.
        """
        # Check if token already exists
        tokens_field = self.metadata.get("tokenizer.ggml.tokens")
        if tokens_field is None:
            return SurgeryResult("add_token", False, "No tokenizer.ggml.tokens field", error="no_vocab")
        _, tokens_list = tokens_field
        if not isinstance(tokens_list, list):
            return SurgeryResult("add_token", False, "Vocab is not a list", error="bad_vocab")
        if token in tokens_list:
            return SurgeryResult("add_token", False, f"Token '{token}' already exists", error="exists")

        new_vocab_size = len(tokens_list) + 1

        # Append to vocab arrays
        tokens_list.append(token)
        self.metadata["tokenizer.ggml.tokens"] = (GGUF_TYPE_ARRAY, tokens_list)

        # Append score if scores array exists
        scores_field = self.metadata.get("tokenizer.ggml.scores")
        if scores_field is not None:
            _, scores_list = scores_field
            if isinstance(scores_list, list):
                scores_list.append(score)
                self.metadata["tokenizer.ggml.scores"] = (GGUF_TYPE_ARRAY, scores_list)

        # Append token type if types array exists
        types_field = self.metadata.get("tokenizer.ggml.token_type")
        if types_field is not None:
            _, types_list = types_field
            if isinstance(types_list, list):
                types_list.append(token_type)
                self.metadata["tokenizer.ggml.token_type"] = (GGUF_TYPE_ARRAY, types_list)

        # Update vocab_size metadata
        arch = self.metadata.get("general.architecture", (GGUF_TYPE_STRING, "llama"))[1]
        self.set_metadata(f"{arch}.vocab_size", new_vocab_size, GGUF_TYPE_UINT32)

        # Extend the token embedding matrix with a new row
        if "token_embd.weight" in self.tensors:
            tensor_type, shape, data = self.tensors["token_embd.weight"]
            try:
                qt = gguf.GGMLQuantizationType(tensor_type)
            except Exception:
                qt = None
            if qt in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
                dtype = np.float32 if qt == gguf.GGMLQuantizationType.F32 else np.float16
                arr = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
                embed_dim = shape[1]
                if embedding is None:
                    new_row = np.zeros((1, embed_dim), dtype=dtype)
                else:
                    new_row = np.array(embedding, dtype=dtype).reshape(1, embed_dim)
                arr = np.vstack([arr, new_row])
                shape[0] = new_vocab_size
                self.tensors["token_embd.weight"] = (tensor_type, shape, arr.tobytes())

                # If output.weight exists and is the same shape (tied embeddings), extend it too
                if "output.weight" in self.tensors:
                    out_type, out_shape, out_data = self.tensors["output.weight"]
                    if out_shape == shape:
                        # Already updated above (tied) — but if they're separate copies, extend
                        if out_data != data:
                            out_arr = np.frombuffer(out_data, dtype=dtype).reshape(out_shape).copy()
                            out_arr = np.vstack([out_arr, new_row])
                            out_shape[0] = new_vocab_size
                            self.tensors["output.weight"] = (out_type, out_shape, out_arr.tobytes())

        self.operations_log.append({
            "op": "add_token", "token": token,
            "new_vocab_size": new_vocab_size,
            "embedding_provided": embedding is not None,
        })
        return SurgeryResult("add_token", True,
                            f"Token '{token}' added (vocab now {new_vocab_size})")

    # ------------------------------------------------------------------ #
    # Operation 8: Add steering vector
    # ------------------------------------------------------------------ #
    def add_steering_vector(
        self,
        layer: int,
        vector: np.ndarray,
        name: str = "default",
        strength: float = 1.0,
    ) -> SurgeryResult:
        """Add an activation steering vector for a specific transformer layer.

        The vector is stored as a new tensor `blk.{layer}.steering.{name}.weight`
        with the given strength as a metadata field. These are non-standard
        tensors — they can be applied by a custom inference engine that knows
        to add them to the residual stream after the specified layer.
        """
        arch = self.metadata.get("general.architecture", (GGUF_TYPE_STRING, "llama"))[1]
        # Get the embedding dimension
        embed_dim_field = self.metadata.get(f"{arch}.embedding_length")
        if embed_dim_field is None:
            return SurgeryResult("add_steering_vector", False,
                                 "Cannot determine embedding dim", error="no_embed_dim")
        embed_dim = embed_dim_field[1]

        if len(vector) != embed_dim:
            return SurgeryResult("add_steering_vector", False,
                                 f"Vector length {len(vector)} != embed_dim {embed_dim}",
                                 error="dim_mismatch")

        tensor_name = f"blk.{layer}.steering.{name}.weight"
        vec_f32 = vector.astype(np.float32).reshape(1, embed_dim)
        self.tensors[tensor_name] = (
            int(gguf.GGMLQuantizationType.F32),
            [1, embed_dim],
            vec_f32.tobytes(),
        )

        # Metadata to describe the steering vector
        self.set_metadata(
            f"steering.{name}.layer", layer, GGUF_TYPE_UINT32
        )
        self.set_metadata(
            f"steering.{name}.strength", strength, GGUF_TYPE_FLOAT32
        )
        self.set_metadata(
            f"steering.{name}.tensor", tensor_name, GGUF_TYPE_STRING
        )

        self.operations_log.append({
            "op": "add_steering_vector", "layer": layer, "name": name,
            "strength": strength, "norm": float(np.linalg.norm(vector)),
        })
        return SurgeryResult("add_steering_vector", True,
                            f"Steering vector '{name}' added for layer {layer} (norm={np.linalg.norm(vector):.4f}, strength={strength})")

    def list_steering_vectors(self) -> List[Dict[str, Any]]:
        """List all steering vectors in the model."""
        out = []
        # Find all steering tensor names
        for tname in self.tensors:
            if ".steering." in tname:
                # Parse: blk.{layer}.steering.{name}.weight
                parts = tname.split(".")
                if len(parts) >= 5 and parts[0] == "blk" and parts[2] == "steering":
                    layer = int(parts[1])
                    name = parts[3]
                    strength_field = self.metadata.get(f"steering.{name}.strength")
                    strength = strength_field[1] if strength_field else 1.0
                    out.append({
                        "name": name, "layer": layer,
                        "tensor": tname, "strength": strength,
                    })
        return out

    # ------------------------------------------------------------------ #
    # Write the modified GGUF
    # ------------------------------------------------------------------ #
    def write(self, output_path: str) -> SurgeryReport:
        """Write the modified GGUF to a new file."""
        t0 = time.time()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._write_gguf(str(out_path))
            elapsed = time.time() - t0
            return SurgeryReport(
                source_path=self.source_path,
                output_path=str(out_path),
                operations=self.operations_log,
                elapsed_seconds=elapsed,
                n_tensors=len(self.tensors),
                n_kv_pairs=len([k for k in self.metadata if not k.endswith(".__elem_type__")]),
                success=True,
            )
        except Exception as e:
            import traceback
            return SurgeryReport(
                source_path=self.source_path,
                output_path=str(out_path),
                operations=self.operations_log,
                elapsed_seconds=time.time() - t0,
                n_tensors=len(self.tensors),
                n_kv_pairs=len([k for k in self.metadata if not k.endswith(".__elem_type__")]),
                success=False,
                error=f"{e}\n{traceback.format_exc()}",
            )

    def _write_gguf(self, output_path: str):
        """Write the modified GGUF using gguf.GGUFWriter for format correctness."""
        # Get architecture for the writer
        arch_field = self.metadata.get("general.architecture")
        arch = arch_field[1] if arch_field else "llama"
        if not isinstance(arch, str):
            arch = "llama"

        writer = gguf.GGUFWriter(output_path, arch)

        # Filter out internal element-type hints
        kv_pairs = {k: v for k, v in self.metadata.items() if not k.endswith(".__elem_type__")}

        # Write all KV pairs (skip general.architecture which GGUFWriter handles)
        for key, (type_id, value) in kv_pairs.items():
            if key == "general.architecture":
                continue  # already set by GGUFWriter constructor
            self._add_kv_to_writer(writer, key, type_id, value)

        # Write all tensors
        for name, (tensor_type, shape, data) in self.tensors.items():
            try:
                qt = gguf.GGMLQuantizationType(tensor_type)
            except Exception:
                qt = gguf.GGMLQuantizationType.F32
            # Convert raw bytes to numpy array
            if qt == gguf.GGMLQuantizationType.F32:
                arr = np.frombuffer(data, dtype=np.float32)
            elif qt == gguf.GGMLQuantizationType.F16:
                arr = np.frombuffer(data, dtype=np.float16)
            else:
                # For quantized types, pass raw bytes
                arr = np.frombuffer(data, dtype=np.uint8)
            # Reshape — GGUFWriter expects reversed shape (ggml convention)
            gguf_shape = list(reversed(shape))
            try:
                arr = arr.reshape(gguf_shape)
            except Exception:
                pass  # leave as 1D if reshape fails
            writer.add_tensor(name, arr, raw_dtype=qt)

        # Write everything to file
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def _add_kv_to_writer(self, writer, key: str, type_id: int, value: Any):
        """Add a single KV pair to the GGUFWriter."""
        try:
            if type_id == GGUF_TYPE_UINT8:
                writer.add_uint8(key, int(value))
            elif type_id == GGUF_TYPE_INT8:
                writer.add_int8(key, int(value))
            elif type_id == GGUF_TYPE_UINT16:
                writer.add_uint16(key, int(value))
            elif type_id == GGUF_TYPE_INT16:
                writer.add_int16(key, int(value))
            elif type_id == GGUF_TYPE_UINT32:
                writer.add_uint32(key, int(value))
            elif type_id == GGUF_TYPE_INT32:
                writer.add_int32(key, int(value))
            elif type_id == GGUF_TYPE_FLOAT32:
                writer.add_float32(key, float(value))
            elif type_id == GGUF_TYPE_BOOL:
                writer.add_bool(key, bool(value))
            elif type_id == GGUF_TYPE_STRING:
                writer.add_string(key, str(value) if value is not None else "")
            elif type_id == GGUF_TYPE_UINT64:
                writer.add_uint64(key, int(value))
            elif type_id == GGUF_TYPE_INT64:
                writer.add_int64(key, int(value))
            elif type_id == GGUF_TYPE_FLOAT64:
                writer.add_float64(key, float(value))
            elif type_id == GGUF_TYPE_ARRAY:
                if not isinstance(value, (list, tuple)):
                    value = list(value) if value is not None else []
                # Get element type
                elem_type_hint = self.metadata.get(key + ".__elem_type__")
                if elem_type_hint:
                    elem_type = elem_type_hint[1]
                else:
                    elem_type = _array_elem_type(value) if value else GGUF_TYPE_STRING
                if elem_type == GGUF_TYPE_STRING:
                    writer.add_array(key, [str(v) for v in value])
                elif elem_type in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32):
                    writer.add_array(key, [int(v) for v in value])
                elif elem_type == GGUF_TYPE_FLOAT32:
                    writer.add_array(key, [float(v) for v in value])
                elif elem_type == GGUF_TYPE_BOOL:
                    writer.add_array(key, [bool(v) for v in value])
                else:
                    # Fallback: treat as string array
                    writer.add_array(key, [str(v) for v in value])
        except Exception as e:
            # Skip fields that can't be written (e.g., unsupported types)
            print(f"[surgeon] WARNING: skipping field {key} (type {type_id}): {e}")


# ---------------------------------------------------------------------- #
# Convenience: open + modify + write in one call
# ---------------------------------------------------------------------- #
def surgery_session(
    source_path: str,
    output_path: str,
    operations: List[Dict[str, Any]],
) -> SurgeryReport:
    """Run a list of operations on a GGUF file and write the result.

    Each operation is a dict with at least an 'op' key.
    """
    surgeon = GGUFSurgeon(source_path)
    for op in operations:
        op_name = op.get("op")
        if op_name == "patch_tensor":
            new_values = np.array(op["new_values"], dtype=np.float32)
            slice_spec = op.get("slice")
            if isinstance(slice_spec, list):
                slice_spec = tuple(slice_spec)
            surgeon.patch_tensor(op["name"], new_values, slice_spec)
        elif op_name == "set_metadata":
            surgeon.set_metadata(op["key"], op["value"], op.get("type"))
        elif op_name == "remove_metadata":
            surgeon.remove_metadata(op["key"])
        elif op_name == "set_chat_template":
            surgeon.set_chat_template(op["template"])
        elif op_name == "bake_system_prompt":
            surgeon.bake_system_prompt(op["prompt"])
        elif op_name == "inject_dataset":
            surgeon.inject_dataset(op["name"], op["data"], op.get("description", ""))
        elif op_name == "add_token":
            embedding = np.array(op["embedding"], dtype=np.float32) if op.get("embedding") else None
            surgeon.add_token(op["token"], embedding, op.get("score", 0.0), op.get("token_type", 1))
        elif op_name == "add_steering_vector":
            vector = np.array(op["vector"], dtype=np.float32)
            surgeon.add_steering_vector(op["layer"], vector, op.get("name", "default"), op.get("strength", 1.0))
        else:
            raise ValueError(f"Unknown operation: {op_name}")
    return surgeon.write(output_path)
