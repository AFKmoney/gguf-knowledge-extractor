"""ROME-style rank-1 knowledge editing for GGUF models.

The runtime now separates three scientific stages:
1. extract the actual MLP key k*;
2. optimize an explicit target value v* against the real forward pass;
3. solve and apply the full-matrix rank-1 ROME update.

The target optimizer is currently SPSA-based and remains experimental; this
module therefore does not claim paper-equivalence until the exact paper
optimization and model-level validation contract pass.
"""
from __future__ import annotations

import shutil
import struct
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import gguf

from .forward_pass import NumpyLlamaForward
from .mlp_analyzer import MLPAnalyzer
from .causal_tracer import CausalTracer
from .rome_reference import collect_key_statistics, solve_rome_update
from .rome_target import extract_mlp_key, optimize_target_value


@dataclass
class EditRequest:
    subject: str
    prompt: str
    target_object: str
    preserve_other_facts: bool = True
    calibration_prompts: Optional[List[str]] = None
    target_iterations: int = 24
    target_step_size: float = 0.05
    target_probe_scale: float = 0.01
    target_seed: int = 0


@dataclass
class EditResult:
    subject: str
    prompt: str
    target_object: str
    target_token_id: Optional[int]
    edited_layer: Optional[int]
    edited_neuron: Optional[int]
    key_vector_norm: float
    old_value_norm: float
    new_value_norm: float
    delta_norm: float
    pre_edit_prediction: str
    post_edit_prediction: str
    edit_successful: bool
    method: str
    error: Optional[str] = None
    target_initial_probability: Optional[float] = None
    target_final_probability: Optional[float] = None
    target_initial_loss: Optional[float] = None
    target_final_loss: Optional[float] = None
    calibration_key_count: int = 0
    target_seed: Optional[int] = None


@dataclass
class EditReport:
    n_edits_requested: int
    n_edits_successful: int
    edits: List[Dict[str, Any]]
    output_gguf_path: Optional[str]
    stats: Dict[str, Any] = field(default_factory=dict)


class RomeEditor:
    """ROME-style rank-1 fact editor with explicit target optimization."""

    def __init__(self, reader: gguf.GGUFReader, fields: Dict[str, Any]):
        self.reader = reader
        self.fields = fields
        self.forward_pass = NumpyLlamaForward(reader, fields)
        tokens = [str(t) for t in fields.get("tokenizer.ggml.tokens", [])] if isinstance(fields.get("tokenizer.ggml.tokens"), list) else []
        self.tokens = tokens
        self.mlp_analyzer = MLPAnalyzer(reader, tokens=tokens, top_k_per_layer=20, top_k_tokens=10)
        self.causal_tracer = CausalTracer(reader, fields, top_k=10)

    def is_available(self) -> bool:
        return self.forward_pass.is_available()

    def compute_edit(self, request: EditRequest, target_layer: Optional[int] = None):
        if not self.is_available():
            return self._failed(request, "forward_pass_unavailable", "Forward pass not available"), None

        target_id = self.forward_pass.find_token_for_text(request.target_object)
        if target_id is None:
            return self._failed(request, "target_token_not_found", f"Could not find token for '{request.target_object}' in vocab"), None

        token_ids = self.causal_tracer.tokenize(request.prompt)
        if not token_ids:
            return self._failed(request, "tokenization_failed", "Could not tokenize prompt", target_id), None

        try:
            pre = self.forward_pass.forward_with_lens(token_ids)
            pre_pred = pre.predicted_token_text
        except Exception as e:
            return self._failed(request, "forward_error", str(e), target_id), None

        if target_layer is None:
            trace = self.causal_tracer.trace_fact("rome_edit", request.prompt, request.target_object)
            target_layer = trace.layer_first_correct or trace.layer_first_predicted
            if target_layer is None:
                target_layer = len(self.forward_pass.layers) // 2
        if target_layer < 0 or target_layer >= len(self.forward_pass.layers):
            return self._failed(request, "invalid_layer", f"Layer {target_layer} out of range", target_id), None

        layer = self.forward_pass.layers[target_layer]
        w_down = layer.ffn_down
        if w_down is None:
            return self._failed(request, "missing_mlp_tensors", "W_down unavailable", target_id, target_layer), None

        try:
            k_star = extract_mlp_key(self.forward_pass, token_ids, target_layer)
            if request.calibration_prompts:
                calibration_keys = []
                for calibration_prompt in request.calibration_prompts:
                    ids = self.causal_tracer.tokenize(calibration_prompt)
                    if ids:
                        calibration_keys.append(extract_mlp_key(self.forward_pass, ids, target_layer))
                if not calibration_keys:
                    return self._failed(request, "calibration_failed", "No usable calibration prompts", target_id, target_layer), None
            else:
                # Valid single-key second moment, but explicitly not paper calibration.
                calibration_keys = [k_star]

            _, covariance = collect_key_statistics(calibration_keys)
            target_opt = optimize_target_value(
                self.forward_pass, token_ids, target_layer, k_star, target_id,
                iterations=request.target_iterations,
                step_size=request.target_step_size,
                probe_scale=request.target_probe_scale,
                seed=request.target_seed,
            )
            v_target = target_opt.target_value
            delta_w = solve_rome_update(w_down.astype(np.float32), k_star, v_target, [k for k in calibration_keys])
            # Recompute through the same public solver with the exact covariance
            # is intentionally avoided: solve_rome_update owns covariance formation.
            _ = covariance
        except Exception as e:
            return self._failed(request, "rome_target_optimization_failed", str(e), target_id, target_layer), None

        current_v = w_down.astype(np.float32) @ k_star
        edited_w = w_down.astype(np.float32) + delta_w
        original = layer.ffn_down
        layer.ffn_down = edited_w
        try:
            post = self.forward_pass.forward_with_lens(token_ids)
            post_pred = post.predicted_token_text
            successful = post.predicted_token_id == target_id
        except Exception as e:
            post_pred = f"<error: {e}>"
            successful = False
        finally:
            layer.ffn_down = original

        tensor_name = self._find_tensor_name(target_layer, ["ffn_down.weight", "mlp.down.weight", "mlp.down_proj.weight"])
        if tensor_name is None:
            return self._failed(request, "tensor_name_not_found", "Could not find W_down tensor", target_id, target_layer), None

        payload = {
            "tensor_name": tensor_name,
            "delta_matrix": delta_w.astype(np.float32),
            "column_index": None,
            "old_value": current_v.copy(),
            "new_value": v_target.copy(),
            "rome_target_optimization": asdict(target_opt),
            "calibration_key_count": len(calibration_keys),
        }

        return EditResult(
            subject=request.subject, prompt=request.prompt, target_object=request.target_object,
            target_token_id=target_id, edited_layer=target_layer, edited_neuron=None,
            key_vector_norm=float(np.linalg.norm(k_star)), old_value_norm=float(np.linalg.norm(current_v)),
            new_value_norm=float(np.linalg.norm(v_target)), delta_norm=float(np.linalg.norm(delta_w)),
            pre_edit_prediction=pre_pred, post_edit_prediction=post_pred,
            edit_successful=successful, method="rome_rank1_full_matrix_target_spsa",
            target_initial_probability=target_opt.initial_target_probability,
            target_final_probability=target_opt.final_target_probability,
            target_initial_loss=target_opt.initial_loss,
            target_final_loss=target_opt.final_loss,
            calibration_key_count=len(calibration_keys), target_seed=request.target_seed,
        ), payload

    def apply_edits_to_file(self, original_gguf_path: str, output_gguf_path: str, edits: List[Dict[str, Any]]) -> str:
        if not edits:
            shutil.copyfile(original_gguf_path, output_gguf_path)
            return output_gguf_path
        by_tensor: Dict[str, List[Dict[str, Any]]] = {}
        for e in edits:
            if e is not None:
                by_tensor.setdefault(e["tensor_name"], []).append(e)
        shutil.copyfile(original_gguf_path, output_gguf_path)
        return self._write_modified_gguf(original_gguf_path, output_gguf_path, by_tensor)

    def _write_modified_gguf(self, original_path: str, output_path: str, edits_by_tensor: Dict[str, List[Dict[str, Any]]]) -> str:
        reader = gguf.GGUFReader(original_path)
        info: Dict[str, Dict[str, Any]] = {}
        for t in reader.tensors:
            name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            info[name] = {"tensor_type": int(t.tensor_type), "shape": [int(s) for s in t.shape]}
        offsets: Dict[str, int] = {}
        for t in reader.tensors:
            name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            if hasattr(t, "data_offset"):
                offsets[name] = int(t.data_offset)
            elif hasattr(t, "offset"):
                offsets[name] = int(t.offset)
        if len(offsets) != len(info):
            offsets = self._header_offsets(original_path)
        with open(output_path, "r+b") as f:
            for tname, edits in edits_by_tensor.items():
                if tname not in info or tname not in offsets:
                    raise ValueError(f"Tensor offset unavailable: {tname}")
                ttype = gguf.GGMLQuantizationType(info[tname]["tensor_type"])
                shape = info[tname]["shape"]
                if ttype not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
                    raise ValueError(f"ROME full-matrix patch requires F32/F16/BF16, got {ttype.name}")
                if len(shape) != 2:
                    raise ValueError(f"ROME target is not a matrix: {tname}")
                rows, cols = shape
                item_bytes = 4 if ttype == gguf.GGMLQuantizationType.F32 else 2
                f.seek(offsets[tname])
                raw = f.read(rows * cols * item_bytes)
                if ttype == gguf.GGMLQuantizationType.F32:
                    matrix = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
                elif ttype == gguf.GGMLQuantizationType.F16:
                    matrix = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
                else:
                    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                    matrix = u.view(np.float32).reshape(shape).copy()
                for e in edits:
                    delta = np.asarray(e.get("delta_matrix"), dtype=np.float32)
                    if delta.shape != matrix.shape:
                        raise ValueError(f"Delta shape {delta.shape} != tensor shape {matrix.shape}")
                    matrix += delta
                f.seek(offsets[tname])
                if ttype == gguf.GGMLQuantizationType.F32:
                    f.write(matrix.astype(np.float32).tobytes())
                elif ttype == gguf.GGMLQuantizationType.F16:
                    f.write(matrix.astype(np.float16).tobytes())
                else:
                    bits = matrix.astype(np.float32).view(np.uint32) >> 16
                    f.write(bits.astype(np.uint16).tobytes())
        return output_path

    def _header_offsets(self, path: str) -> Dict[str, int]:
        with open(path, "rb") as f:
            magic, version = struct.unpack("<II", f.read(8))
            if magic != 0x46554747:
                raise ValueError("Not a GGUF file")
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            for _ in range(n_kv):
                name_len = struct.unpack("<Q", f.read(8))[0]
                f.seek(name_len, 1)
                type_id = struct.unpack("<I", f.read(4))[0]
                self._skip_value(f, type_id)
            rel: Dict[str, int] = {}
            for _ in range(n_tensors):
                name_len = struct.unpack("<Q", f.read(8))[0]
                name = f.read(name_len).decode("utf-8")
                n_dims = struct.unpack("<I", f.read(4))[0]
                f.seek(8 * n_dims, 1)
                f.seek(4, 1)
                rel[name] = struct.unpack("<Q", f.read(8))[0]
            start = (f.tell() + 31) & ~31
            return {k: start + v for k, v in rel.items()}

    def _skip_value(self, f, type_id: int):
        sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if type_id in sizes:
            f.seek(sizes[type_id], 1)
        elif type_id == 8:
            f.seek(struct.unpack("<Q", f.read(8))[0], 1)
        elif type_id == 9:
            elem = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                self._skip_value(f, elem)
        else:
            raise ValueError(f"Unknown GGUF metadata type {type_id}")

    def _find_tensor_name(self, layer_idx: int, suffixes: List[str]) -> Optional[str]:
        for prefix in [f"blk.{layer_idx}.", f"layers.{layer_idx}.", f"h.{layer_idx}."]:
            for sfx in suffixes:
                full = prefix + sfx
                for t in self.reader.tensors:
                    name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
                    if name == full:
                        return full
        return None

    def edit_facts(self, requests: List[EditRequest], original_gguf_path: str, output_gguf_path: str) -> EditReport:
        t0 = time.time()
        results: List[Dict[str, Any]] = []
        payloads: List[Dict[str, Any]] = []
        successful = 0
        for req in requests:
            result, payload = self.compute_edit(req)
            results.append(_to_jsonable(asdict(result)))
            if payload is not None:
                payloads.append(payload)
            successful += int(result.edit_successful)
        output = None
        if payloads:
            try:
                output = self.apply_edits_to_file(original_gguf_path, output_gguf_path, payloads)
            except Exception as e:
                output = None
                results.append({"apply_error": str(e)})
        return EditReport(len(requests), successful, results, output, {"elapsed_seconds": time.time() - t0, "forward_pass_available": self.is_available(), "n_payloads_applied": len(payloads)})

    @staticmethod
    def _failed(request, method, error, target_id=None, layer=None, neuron=None):
        return EditResult(request.subject, request.prompt, request.target_object, target_id, layer, neuron, 0, 0, 0, 0, "", "", False, method, error)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict): return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    return obj
