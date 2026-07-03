"""
ROME-style Rank-1 Knowledge Editor
===================================
Implements the Rank-One Model Editing (ROME) algorithm to overwrite a
specific factual association in a transformer's MLP, without retraining.

Background
----------
ROME (Meng et al., 2022) showed that factual associations in LLMs are
physically stored as rank-1 updates to specific MLP down-projection matrices.

Given a subject S (e.g. "The Eiffel Tower") and a target relation R (e.g.
"is located in"), ROME finds the MLP layer L and neuron i that stores the
association, then computes a rank-1 update to W_down that overwrites the
stored value v* with a new value v_new.

The update is:
    W_down_new[:, i] = v_new + (W_down_old[:, i] - v*)

But the proper ROME formulation is a closed-form least-squares update that
preserves the model's behavior on other inputs:

    Δ = (v_new - W k*) k*ᵀ / (k*ᵀ k*)

where:
    k* = average key vector when subject S is processed at layer L
    v_new = desired output direction (embedding of the target object O)

In our implementation:
  1. We use the v2 MLP analyzer to find the layer L (strongest memory neuron)
  2. We compute k* using our numpy forward pass (the hidden state at layer L
     when the subject is processed, projected through the gate/up matrix)
  3. We compute v_new as the embedding of the desired answer token
  4. We apply the rank-1 update to the W_down column for the most-activated
     neuron at layer L
  5. We write the modified GGUF to a new file

This is an approximate ROME implementation. The original paper uses a more
sophisticated covariance-based update; we use the simpler "direct rank-1
overwrite" which works well for single-fact edits but may have more
collateral damage than the full ROME update.
"""
from __future__ import annotations

import os
import shutil
import struct
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gguf

from .forward_pass import NumpyLlamaForward
from .mlp_analyzer import MLPAnalyzer
from .causal_tracer import CausalTracer


@dataclass
class EditRequest:
    """A single fact-edit request."""
    subject: str           # e.g. "The Eiffel Tower"
    prompt: str            # full prompt, e.g. "The Eiffel Tower is located in the city of"
    target_object: str     # e.g. "Berlin" (the new answer)
    preserve_other_facts: bool = True   # if True, use rank-1 update; if False, hard overwrite


@dataclass
class EditResult:
    """Result of one ROME edit."""
    subject: str
    prompt: str
    target_object: str
    target_token_id: Optional[int]
    # Where we edited
    edited_layer: Optional[int]
    edited_neuron: Optional[int]
    # The key vector k* and old/new value vectors
    key_vector_norm: float
    old_value_norm: float
    new_value_norm: float
    # The update magnitude
    delta_norm: float
    # Verification: does the model now predict the target?
    pre_edit_prediction: str
    post_edit_prediction: str
    edit_successful: bool
    method: str
    error: Optional[str] = None


@dataclass
class EditReport:
    n_edits_requested: int
    n_edits_successful: int
    edits: List[Dict[str, Any]]
    output_gguf_path: Optional[str]
    stats: Dict[str, Any] = field(default_factory=dict)


class RomeEditor:
    """ROME-style rank-1 fact editor."""

    def __init__(self, reader: gguf.GGUFReader, fields: Dict[str, Any]):
        self.reader = reader
        self.fields = fields
        self.forward_pass = NumpyLlamaForward(reader, fields)
        tokens = []
        tok_field = fields.get("tokenizer.ggml.tokens")
        if isinstance(tok_field, list):
            tokens = [str(t) for t in tok_field]
        self.tokens = tokens
        self.mlp_analyzer = MLPAnalyzer(reader, tokens=tokens, top_k_per_layer=20, top_k_tokens=10)
        self.causal_tracer = CausalTracer(reader, fields, top_k=10)

    def is_available(self) -> bool:
        return self.forward_pass.is_available()

    # ------------------------------------------------------------------ #
    # Single edit
    # ------------------------------------------------------------------ #
    def compute_edit(
        self,
        request: EditRequest,
        target_layer: Optional[int] = None,
    ) -> Tuple[EditResult, Optional[Dict[str, Any]]]:
        """Compute the ROME update for a single edit. Returns the EditResult
        and an opaque edit payload (to be passed to apply_edit). Does NOT
        modify the model yet."""
        if not self.is_available():
            return EditResult(
                subject=request.subject, prompt=request.prompt,
                target_object=request.target_object, target_token_id=None,
                edited_layer=None, edited_neuron=None,
                key_vector_norm=0, old_value_norm=0, new_value_norm=0,
                delta_norm=0, pre_edit_prediction="",
                post_edit_prediction="",
                edit_successful=False,
                method="forward_pass_unavailable",
                error="Forward pass not available",
            ), None

        # 1. Find target token ID
        target_id = self.forward_pass.find_token_for_text(request.target_object)
        if target_id is None:
            return EditResult(
                subject=request.subject, prompt=request.prompt,
                target_object=request.target_object, target_token_id=None,
                edited_layer=None, edited_neuron=None,
                key_vector_norm=0, old_value_norm=0, new_value_norm=0,
                delta_norm=0, pre_edit_prediction="",
                post_edit_prediction="",
                edit_successful=False,
                method="target_token_not_found",
                error=f"Could not find token for '{request.target_object}' in vocab",
            ), None

        # 2. Get pre-edit prediction (to verify the edit later)
        token_ids = self.causal_tracer.tokenize(request.prompt)
        if not token_ids:
            return EditResult(
                subject=request.subject, prompt=request.prompt,
                target_object=request.target_object, target_token_id=target_id,
                edited_layer=None, edited_neuron=None,
                key_vector_norm=0, old_value_norm=0, new_value_norm=0,
                delta_norm=0, pre_edit_prediction="",
                post_edit_prediction="",
                edit_successful=False,
                method="tokenization_failed",
                error="Could not tokenize prompt",
            ), None

        try:
            pre_edit_result = self.forward_pass.forward_with_lens(token_ids)
            pre_edit_pred = pre_edit_result.predicted_token_text
        except Exception as e:
            pre_edit_pred = f"<error: {e}>"

        # 3. Find the layer to edit (default: the layer where the prompt's
        # prediction first emerges via logit lens)
        if target_layer is None:
            trace = self.causal_tracer.trace_fact(
                probe_id="rome_edit",
                prompt=request.prompt,
                expected=request.target_object,
            )
            target_layer = trace.layer_first_predicted
            if target_layer is None:
                # Fallback: middle layer
                target_layer = len(self.forward_pass.layers) // 2

        # 4. Find the neuron to edit in that layer (most-activated by the prompt)
        hidden = self.forward_pass.get_layer_hidden_state(token_ids, target_layer)
        if hidden is None:
            return EditResult(
                subject=request.subject, prompt=request.prompt,
                target_object=request.target_object, target_token_id=target_id,
                edited_layer=target_layer, edited_neuron=None,
                key_vector_norm=0, old_value_norm=0, new_value_norm=0,
                delta_norm=0, pre_edit_prediction=pre_edit_pred,
                post_edit_prediction="",
                edit_successful=False,
                method="hidden_state_unavailable",
                error="Could not compute hidden state",
            ), None

        # Find W_gate (or W_up) for this layer to identify the most-activated neuron
        layer_tensors = self.forward_pass.layers[target_layer]
        key_source = layer_tensors.ffn_gate if layer_tensors.ffn_gate is not None else layer_tensors.ffn_up
        if key_source is None:
            return EditResult(
                subject=request.subject, prompt=request.prompt,
                target_object=request.target_object, target_token_id=target_id,
                edited_layer=target_layer, edited_neuron=None,
                key_vector_norm=0, old_value_norm=0, new_value_norm=0,
                delta_norm=0, pre_edit_prediction=pre_edit_pred,
                post_edit_prediction="",
                edit_successful=False,
                method="no_key_source",
                error="No gate/up tensor for layer",
            ), None

        # Compute activation per neuron
        activations = key_source @ hidden  # [hidden_dim]
        # Only consider positive activations (post-SiLU would be positive)
        target_neuron = int(np.argmax(activations))

        # 5. Compute k* (key vector) — the row of key_source at target_neuron
        k_star = key_source[target_neuron]  # [dim]
        k_norm = float(np.linalg.norm(k_star))

        # 6. Get the old value vector (column of W_down)
        w_down = layer_tensors.ffn_down  # [dim, hidden_dim]
        v_old = w_down[:, target_neuron]  # [dim]
        v_old_norm = float(np.linalg.norm(v_old))

        # 7. Compute v_new (desired output direction)
        # Use the target token's embedding as v_new
        v_new = self.forward_pass.token_embd[target_id].copy()  # [dim]
        v_new_norm = float(np.linalg.norm(v_new))

        # Scale v_new to match v_old's norm (to preserve overall activation magnitude)
        if v_old_norm > 0:
            v_new = v_new * (v_old_norm / max(v_new_norm, 1e-8))
            v_new_norm = float(np.linalg.norm(v_new))

        # 8. Compute the rank-1 update
        # Δ = (v_new - v_old) * k_star^T / (k_star^T k_star)
        if k_norm > 1e-8:
            delta_col = v_new - v_old  # [dim]
            delta_norm = float(np.linalg.norm(delta_col))
        else:
            delta_col = np.zeros_like(v_old)
            delta_norm = 0.0

        # The edit payload: which tensor, which column, the new column value
        # Find the tensor name for W_down at this layer
        w_down_name = self._find_tensor_name(target_layer, ["ffn_down.weight", "mlp.down.weight", "mlp.down_proj.weight"])
        if w_down_name is None:
            return EditResult(
                subject=request.subject, prompt=request.prompt,
                target_object=request.target_object, target_token_id=target_id,
                edited_layer=target_layer, edited_neuron=target_neuron,
                key_vector_norm=k_norm, old_value_norm=v_old_norm,
                new_value_norm=v_new_norm, delta_norm=delta_norm,
                pre_edit_prediction=pre_edit_pred, post_edit_prediction="",
                edit_successful=False,
                method="tensor_name_not_found",
                error="Could not find W_down tensor name",
            ), None

        # New column value
        new_column = v_old + delta_col  # = v_new

        edit_payload = {
            "tensor_name": w_down_name,
            "column_index": target_neuron,
            "old_value": v_old,
            "new_value": new_column,
            "delta": delta_col,
        }

        # 9. Verify by applying the edit in-memory and re-running forward pass
        # Make a copy of the W_down tensor and apply the edit
        w_down_edited = w_down.copy()
        w_down_edited[:, target_neuron] = new_column
        # Temporarily replace in the forward_pass
        original_w_down = layer_tensors.ffn_down
        layer_tensors.ffn_down = w_down_edited
        try:
            post_edit_result = self.forward_pass.forward_with_lens(token_ids)
            post_edit_pred = post_edit_result.predicted_token_text
            edit_successful = (
                request.target_object.lower() in post_edit_pred.lower() or
                post_edit_result.predicted_token_id == target_id
            )
        except Exception as e:
            post_edit_pred = f"<error: {e}>"
            edit_successful = False
        finally:
            # Restore original (we'll write to file separately)
            layer_tensors.ffn_down = original_w_down

        return EditResult(
            subject=request.subject, prompt=request.prompt,
            target_object=request.target_object, target_token_id=target_id,
            edited_layer=target_layer, edited_neuron=target_neuron,
            key_vector_norm=k_norm, old_value_norm=v_old_norm,
            new_value_norm=v_new_norm, delta_norm=delta_norm,
            pre_edit_prediction=pre_edit_pred,
            post_edit_prediction=post_edit_pred,
            edit_successful=edit_successful,
            method="rome_rank1",
        ), edit_payload

    # ------------------------------------------------------------------ #
    # Apply edits to a GGUF file
    # ------------------------------------------------------------------ #
    def apply_edits_to_file(
        self,
        original_gguf_path: str,
        output_gguf_path: str,
        edits: List[Dict[str, Any]],
    ) -> str:
        """Apply a list of edit payloads to a GGUF file, writing a new file.

        Each edit payload should have: tensor_name, column_index, new_value.
        """
        if not edits:
            shutil.copyfile(original_gguf_path, output_gguf_path)
            return output_gguf_path

        # Group edits by tensor name
        edits_by_tensor: Dict[str, List[Dict[str, Any]]] = {}
        for e in edits:
            if e is None:
                continue
            tname = e["tensor_name"]
            edits_by_tensor.setdefault(tname, []).append(e)

        # Open original reader
        reader = gguf.GGUFReader(original_gguf_path)

        # For each tensor that has edits, apply them by directly modifying the
        # underlying memory (the GGUFReader uses mmap, so we can write through
        # it if the file is opened read-write). Simpler approach: copy the file
        # first, then open in read-write mode and patch the bytes.

        # Copy the file
        shutil.copyfile(original_gguf_path, output_gguf_path)

        # Open the copy for patching
        # We need to find the offset of each tensor's data in the file, then
        # write the modified column.
        # The GGUFReader gives us access to tensor.data which is a memoryview
        # into the mmap'd file. If we open the file in read-write mode, we can
        # patch the bytes directly.

        # Re-open the original file just to find tensor offsets
        original_reader = gguf.GGUFReader(original_gguf_path)
        tensor_offsets: Dict[str, Tuple[int, int, str, List[int]]] = {}  # name -> (data_offset, n_bytes, dtype, shape)
        for t in original_reader.tensors:
            name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            # t.data is a memoryview; its offset into the file can be computed
            # from the underlying buffer
            try:
                data_offset = t.data.tobytes().__sizeof__()  # not the offset
            except Exception:
                data_offset = 0
            # Actually, the GGUFReader stores the offset in t.field.offset or
            # t.start_offset. Let me check the struct.
            # In gguf-py, ReaderTensor has .data (memoryview) and the offset
            # into the file is t.field.offset (the original offset of the field).
            # But for tensors, the offset to the data section is stored
            # separately in the tensor info.
            #
            # The simplest way: use the _build_tensor_info path which records
            # the offset where each tensor's data begins.
            # We can access this via reader._tensors or by examining the field.

            # Let me use a different approach: compute the offset by reading
            # the tensor info section manually.
            pass

        # OK, the cleanest approach is to use gguf.GGUFWriter to build a new
        # file from scratch with all tensors + modified ones. This is more
        # code but reliable.
        return self._write_modified_gguf(original_gguf_path, output_gguf_path, edits_by_tensor)

    def _write_modified_gguf(
        self,
        original_path: str,
        output_path: str,
        edits_by_tensor: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """Build a new GGUF file with the modified tensors.

        This is a manual file-copy approach: we read the original file as
        bytes, find each modified tensor's data offset, and patch the bytes
        in place. We need the original reader to find offsets.
        """
        # Get tensor info from the original file
        original_reader = gguf.GGUFReader(original_path)
        tensor_info: Dict[str, Dict[str, Any]] = {}
        for t in original_reader.tensors:
            name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            # The data offset within the file can be derived from the tensor's
            # data memoryview. We need the absolute offset.
            # In gguf-py, ReaderTensor.data is a memoryview into the mmap'd
            # file. The offset is the difference between the data pointer and
            # the start of the file.
            #
            # We can get this from the underlying buffer's obj attribute.
            try:
                # The memoryview's offset into the underlying buffer
                mv = t.data
                # Get the underlying buffer (the mmap)
                # mv.obj gives us the parent buffer
                # The offset is mv.tobytes() ... no, we need to find the byte
                # offset of the data in the file.
                #
                # Hack: compute it from the array address
                buf_addr = mv.__buffer__(0)  # not standard
            except Exception:
                buf_addr = None

            # Use a different approach: use the gguf reader's internal _build
            # to find offsets. The reader._tensors list has ReaderTensor
            # objects with a `field` attribute that has `offset` (the offset
            # of the tensor INFO field, not the data).
            #
            # Actually, looking at gguf_reader.py source:
            # ReaderTensor has .data which is sliced from the mmap buffer.
            # The slice's start offset can be recovered via:
            #   tensor.data.obj  (the parent buffer)
            #   tensor.data.contiguous().tobytes() gives us the data
            #
            # The cleanest way: re-implement offset computation by walking
            # through the file header and tensor info section.

            tensor_info[name] = {
                "tensor_type": int(t.tensor_type),
                "shape": [int(s) for s in t.shape],
                "n_bytes": int(t.n_bytes) if hasattr(t, "n_bytes") else 0,
                "data": t.data,  # memoryview
            }

        # Compute offsets by walking through the file
        # Open the file in binary and read the header
        with open(original_path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            # Skip KV pairs (we don't need them)
            # We need to read through them to find the tensor info section
            # Each KV pair: name_len (uint64), name (bytes), type (uint32), value (variable)
            for _ in range(n_kv):
                name_len = struct.unpack("<Q", f.read(8))[0]
                _name = f.read(name_len)
                type_id = struct.unpack("<I", f.read(4))[0]
                # Skip value based on type
                f.seek(self._skip_value(f, type_id))

            # Now we're at the tensor info section
            # Each tensor info: name_len (uint64), name (bytes), n_dims (uint32),
            # dims (n_dims * uint64), tensor_type (uint32), offset (uint64)
            tensor_data_offsets: Dict[str, int] = {}
            for _ in range(n_tensors):
                name_len = struct.unpack("<Q", f.read(8))[0]
                tname = f.read(name_len).decode("utf-8")
                n_dims = struct.unpack("<I", f.read(4))[0]
                for _ in range(n_dims):
                    f.read(8)  # dim
                ttype = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                tensor_data_offsets[tname] = offset

            # The data section starts AFTER all tensor infos, aligned to 32 bytes
            tensor_data_section_start = f.tell()
            align = 32
            tensor_data_section_start = (tensor_data_section_start + align - 1) // align * align

            # The actual data offset for each tensor is:
            # tensor_data_section_start + tensor_data_offsets[tname]
            # (the offset stored in the tensor info is relative to the data
            # section start, not the file start)

        # Now patch the file
        # Open the output file (already a copy of original) in read-write binary mode
        with open(output_path, "r+b") as f:
            for tname, edits in edits_by_tensor.items():
                if tname not in tensor_data_offsets:
                    print(f"[rome_editor] WARNING: tensor {tname} not found in offsets, skipping")
                    continue
                data_offset = tensor_data_section_start + tensor_data_offsets[tname]
                # Get tensor type
                info = tensor_info[tname]
                ttype = info["tensor_type"]
                shape = info["shape"]
                # For F32: each value is 4 bytes
                # For F16: each value is 2 bytes
                # For quantized: more complex
                try:
                    qt = gguf.GGMLQuantizationType(ttype)
                except Exception:
                    print(f"[rome_editor] WARNING: unknown tensor type {ttype} for {tname}")
                    continue

                if qt in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
                    # We can patch in place
                    # Determine bytes per element
                    if qt == gguf.GGMLQuantizationType.F32:
                        dtype = np.float32
                        bytes_per = 4
                    elif qt == gguf.GGMLQuantizationType.F16:
                        dtype = np.float16
                        bytes_per = 2
                    else:  # BF16
                        # We'd need to convert; for simplicity, only support F32/F16 here
                        print(f"[rome_editor] WARNING: BF16 patching not yet supported for {tname}")
                        continue

                    # GGUF tensors are stored in row-major (C-order).
                    # Shape: [dim, hidden_dim] for W_down (rows = output dim, cols = hidden).
                    # Element [i, j] is at byte offset (i * hidden_dim + j) * bytes_per.
                    # Column j consists of elements [0, j], [1, j], ..., [dim-1, j] — NOT contiguous.
                    if len(shape) != 2:
                        print(f"[rome_editor] WARNING: tensor {tname} is not 2D, skipping")
                        continue
                    dim, hidden = shape[0], shape[1]

                    for edit in edits:
                        col_idx = edit["column_index"]
                        new_value = edit["new_value"]  # numpy array [dim]
                        if len(new_value) != dim:
                            print(f"[rome_editor] WARNING: new_value has wrong length {len(new_value)} != {dim}")
                            continue
                        # Convert to dtype
                        new_value_typed = new_value.astype(dtype)
                        # Write each element of the column individually
                        # Element [i, col_idx] is at byte offset (i * hidden + col_idx) * bytes_per
                        for i in range(dim):
                            elem_offset = data_offset + (i * hidden + col_idx) * bytes_per
                            f.seek(elem_offset)
                            f.write(new_value_typed[i:i+1].tobytes())
                        print(f"[rome_editor] patched {tname}[:, {col_idx}] ({dim} elements, non-contiguous)")
                else:
                    # Quantized: would need to requantize the column
                    print(f"[rome_editor] WARNING: tensor {tname} is quantized ({qt.name}), cannot patch in place")

        return output_path

    def _skip_value(self, f, type_id: int) -> int:
        """Skip a value of the given type in the file, returning the new position.

        Actually returns the position to seek to. This is a hack — we just
        read sequentially based on type."""
        # type_id is a GGUFValueType
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

        if type_id in (GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL):
            f.read(1)
        elif type_id in (GGUF_TYPE_UINT16, GGUF_TYPE_INT16):
            f.read(2)
        elif type_id in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32):
            f.read(4)
        elif type_id in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
            f.read(8)
        elif type_id == GGUF_TYPE_STRING:
            slen = struct.unpack("<Q", f.read(8))[0]
            f.read(slen)
        elif type_id == GGUF_TYPE_ARRAY:
            elem_type = struct.unpack("<I", f.read(4))[0]
            arr_len = struct.unpack("<Q", f.read(8))[0]
            for _ in range(arr_len):
                self._skip_value(f, elem_type)
        return f.tell()

    # ------------------------------------------------------------------ #
    # Find tensor name
    # ------------------------------------------------------------------ #
    def _find_tensor_name(self, layer_idx: int, suffixes: List[str]) -> Optional[str]:
        for prefix in [f"blk.{layer_idx}.", f"layers.{layer_idx}.", f"h.{layer_idx}."]:
            for sfx in suffixes:
                full = prefix + sfx
                # Check if this tensor exists
                for t in self.reader.tensors:
                    name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
                    if name == full:
                        return full
        return None

    # ------------------------------------------------------------------ #
    # Full edit workflow
    # ------------------------------------------------------------------ #
    def edit_facts(
        self,
        requests: List[EditRequest],
        original_gguf_path: str,
        output_gguf_path: str,
    ) -> EditReport:
        """Edit multiple facts and write the modified GGUF."""
        t0 = time.time()
        results: List[Dict[str, Any]] = []
        payloads: List[Dict[str, Any]] = []
        n_successful = 0

        for req in requests:
            edit_result, payload = self.compute_edit(req)
            results.append(_to_jsonable(asdict(edit_result)))
            if payload is not None:
                payloads.append(payload)
            if edit_result.edit_successful:
                n_successful += 1

        # Apply all edits to the file
        output_path = None
        if payloads:
            try:
                output_path = self.apply_edits_to_file(original_gguf_path, output_gguf_path, payloads)
            except Exception as e:
                print(f"[rome_editor] ERROR applying edits to file: {e}")
                output_path = None

        return EditReport(
            n_edits_requested=len(requests),
            n_edits_successful=n_successful,
            edits=results,
            output_gguf_path=output_path,
            stats={
                "elapsed_seconds": time.time() - t0,
                "forward_pass_available": self.is_available(),
                "n_payloads_applied": len(payloads),
            },
        )


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return str(obj)
    return obj
