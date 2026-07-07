"""
Model Merger
=============
Merge two GGUF models into one — without retraining.

Algorithms:
  1. **Linear merge** — simple weighted average: W_out = α·W_a + (1-α)·W_b
  2. **SLERP** — spherical linear interpolation, preserves norms better than linear
  3. **TIES** (Trim, Elect, Sign) — resolves conflicts by trimming small diffs,
     electing the dominant sign, and merging. Best for merging many fine-tunes.
  4. **DARE** (Drop And REscale) — randomly drops fine-tune deltas and rescales
     to preserve the original model's behavior on most tasks.

All algorithms work on F32/F16 tensors. Quantized tensors are dequantized,
merged, and re-quantized (with fallback to F16 for K-quants).

Use cases:
  - Merge a base model with a fine-tune to get "best of both"
  - Combine two fine-tunes with different capabilities (code + math)
  - Create a "model soup" from multiple checkpoints
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gguf

from .quant_surgery import QuantSurgeon


@dataclass
class MergeResult:
    """Result of a single tensor merge."""
    tensor_name: str
    algorithm: str
    success: bool
    n_changed: int
    error: Optional[str] = None


@dataclass
class MergeReport:
    """Report from a full model merge."""
    source_a: str
    source_b: str
    output_path: str
    algorithm: str
    alpha: float
    n_tensors_merged: int
    n_tensors_skipped: int
    elapsed_seconds: float
    success: bool
    merge_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class ModelMerger:
    """Merge two GGUF models into one."""

    def __init__(self, source_a: str, source_b: str):
        self.source_a = source_a
        self.source_b = source_b
        self.reader_a = gguf.GGUFReader(source_a)
        self.reader_b = gguf.GGUFReader(source_b)

    # ------------------------------------------------------------------ #
    # Merge algorithms
    # ------------------------------------------------------------------ #
    @staticmethod
    def linear_merge(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Linear interpolation: W_out = α·W_a + (1-α)·W_b"""
        return (alpha * a + (1.0 - alpha) * b).astype(np.float32)

    @staticmethod
    def slerp_merge(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Spherical linear interpolation.

        SLERP finds the shortest path on the unit hypersphere between a and b,
        which preserves norms better than linear interpolation.
        """
        # Flatten for dot product computation
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)

        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)

        if norm_a < 1e-10 or norm_b < 1e-10:
            # One is zero — fall back to linear
            return ModelMerger.linear_merge(a, b, alpha)

        # Normalize
        a_norm = a_flat / norm_a
        b_norm = b_flat / norm_b

        # Dot product (cosine of angle)
        dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
        omega = np.arccos(dot)

        if omega < 1e-6:
            # Nearly parallel — fall back to linear
            return ModelMerger.linear_merge(a, b, alpha)

        sin_omega = np.sin(omega)
        # SLERP formula
        w_a = np.sin((1.0 - alpha) * omega) / sin_omega
        w_b = np.sin(alpha * omega) / sin_omega

        # Interpolate the normalized vectors, then scale by interpolated norm
        merged = w_a * a_norm + w_b * b_norm
        # Scale: interpolate norms linearly (SLERP on the direction, linear on the magnitude)
        out_norm = alpha * norm_a + (1.0 - alpha) * norm_b
        merged = merged * out_norm

        return merged.reshape(a.shape).astype(np.float32)

    @staticmethod
    def ties_merge(
        a: np.ndarray,
        b: np.ndarray,
        alpha: float = 0.5,
        trim_ratio: float = 0.2,
    ) -> np.ndarray:
        """TIES merge: Trim, Elect, Sign.

        1. Trim: remove the smallest `trim_ratio` of changes (by absolute value)
        2. Elect: for each parameter, keep the sign chosen by the majority
        3. Sign: average only the parameters with the elected sign

        Here, we treat 'a' as the base and 'b' as the fine-tune.
        """
        delta = b - a  # the fine-tune delta
        flat_delta = delta.flatten()

        # Step 1: Trim — zero out the smallest trim_ratio of deltas
        abs_delta = np.abs(flat_delta)
        threshold = np.quantile(abs_delta, trim_ratio)
        mask = abs_delta > threshold
        trimmed = flat_delta * mask

        # Step 2: Elect — determine the dominant sign
        # For simplicity, use the sign of the sum of all deltas
        # (In the full TIES paper, this is per-parameter, but we approximate)
        total_sign = np.sign(np.sum(trimmed))
        if total_sign == 0:
            total_sign = 1.0

        # Step 3: Keep only deltas with the elected sign, average them
        sign_mask = np.sign(trimmed) == total_sign
        elected = trimmed * sign_mask
        n_elected = sign_mask.sum()
        if n_elected > 0:
            avg_elected = np.sum(elected) / n_elected
        else:
            avg_elected = 0.0

        # Apply: base + alpha * average_elected_delta (scaled by mask)
        result = a.flatten() + alpha * avg_elected * sign_mask
        return result.reshape(a.shape).astype(np.float32)

    @staticmethod
    def dare_merge(
        a: np.ndarray,
        b: np.ndarray,
        alpha: float = 0.5,
        drop_rate: float = 0.1,
        seed: int = 42,
    ) -> np.ndarray:
        """DARE merge: Drop And REscale.

        1. Compute the delta (b - a)
        2. Randomly drop `drop_rate` fraction of the deltas (set to 0)
        3. Rescale the remaining deltas by 1/(1-drop_rate) to compensate
        4. Apply: W_out = W_a + alpha * rescaled_delta
        """
        rng = np.random.RandomState(seed)
        delta = (b - a).astype(np.float32)
        flat_delta = delta.flatten()

        # Drop mask
        drop_mask = rng.random(len(flat_delta)) < drop_rate
        flat_delta[drop_mask] = 0.0

        # Rescale
        if drop_rate < 1.0:
            flat_delta = flat_delta / (1.0 - drop_rate)

        # Apply
        result = a.flatten() + alpha * flat_delta
        return result.reshape(a.shape).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Full merge
    # ------------------------------------------------------------------ #
    def merge(
        self,
        output_path: str,
        algorithm: str = "linear",  # linear | slerp | ties | dare
        alpha: float = 0.5,
        tensor_filter: Optional[str] = None,  # only merge tensors matching this regex
        **kwargs,
    ) -> MergeReport:
        """Merge the two models into a new GGUF file.

        Args:
            output_path: where to write the merged GGUF
            algorithm: linear, slerp, ties, or dare
            alpha: merge weight (0.0 = all B, 1.0 = all A, 0.5 = equal)
            tensor_filter: regex to filter which tensors to merge (None = all)
            **kwargs: algorithm-specific params (trim_ratio for TIES, drop_rate for DARE)
        """
        t0 = time.time()

        # Verify both models have the same architecture
        arch_a = self._get_arch(self.reader_a)
        arch_b = self._get_arch(self.reader_b)
        if arch_a != arch_b:
            return MergeReport(
                source_a=self.source_a, source_b=self.source_b,
                output_path=output_path, algorithm=algorithm, alpha=alpha,
                n_tensors_merged=0, n_tensors_skipped=0,
                elapsed_seconds=0, success=False,
                error=f"Architecture mismatch: {arch_a} vs {arch_b}",
            )

        # Use reader_a as the base (preserve its metadata)
        # We'll merge each tensor in-place
        import re
        filter_re = re.compile(tensor_filter) if tensor_filter else None

        # Choose the merge function
        merge_fns = {
            "linear": lambda a, b: self.linear_merge(a, b, alpha),
            "slerp": lambda a, b: self.slerp_merge(a, b, alpha),
            "ties": lambda a, b: self.ties_merge(a, b, alpha, kwargs.get("trim_ratio", 0.2)),
            "dare": lambda a, b: self.dare_merge(a, b, alpha, kwargs.get("drop_rate", 0.1), kwargs.get("seed", 42)),
        }
        if algorithm not in merge_fns:
            return MergeReport(
                source_a=self.source_a, source_b=self.source_b,
                output_path=output_path, algorithm=algorithm, alpha=alpha,
                n_tensors_merged=0, n_tensors_skipped=0,
                elapsed_seconds=0, success=False,
                error=f"Unknown algorithm: {algorithm}",
            )
        merge_fn = merge_fns[algorithm]

        # Build a tensor lookup for reader_b
        tensors_b = {}
        for t in self.reader_b.tensors:
            name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            tensors_b[name] = t

        # Merge each tensor from reader_a
        merged_tensors = {}  # name -> (qtype_id, shape, data_bytes)
        n_merged = 0
        n_skipped = 0
        merge_results: List[Dict[str, Any]] = []

        for t_a in self.reader_a.tensors:
            name = t_a.name.decode("utf-8") if isinstance(t_a.name, bytes) else str(t_a.name)

            # Apply filter
            if filter_re and not filter_re.search(name):
                # Keep original
                qt = gguf.GGMLQuantizationType(t_a.tensor_type)
                shape = [int(s) for s in t_a.shape]
                merged_tensors[name] = (int(t_a.tensor_type), shape, bytes(t_a.data))
                n_skipped += 1
                continue

            # Find matching tensor in B
            t_b = tensors_b.get(name)
            if t_b is None:
                # No match in B — keep A's version
                shape = [int(s) for s in t_a.shape]
                merged_tensors[name] = (int(t_a.tensor_type), shape, bytes(t_a.data))
                n_skipped += 1
                merge_results.append({
                    "tensor_name": name, "algorithm": algorithm,
                    "success": False, "n_changed": 0,
                    "error": "not found in model B",
                })
                continue

            # Check shape compatibility
            shape_a = [int(s) for s in t_a.shape]
            shape_b = [int(s) for s in t_b.shape]
            if shape_a != shape_b:
                merged_tensors[name] = (int(t_a.tensor_type), shape_a, bytes(t_a.data))
                n_skipped += 1
                merge_results.append({
                    "tensor_name": name, "algorithm": algorithm,
                    "success": False, "n_changed": 0,
                    "error": f"shape mismatch: {shape_a} vs {shape_b}",
                })
                continue

            # Dequantize both to F32
            qt_a = gguf.GGMLQuantizationType(t_a.tensor_type)
            qt_b = gguf.GGMLQuantizationType(t_b.tensor_type)

            try:
                f32_a = QuantSurgeon.dequantize_tensor(bytes(t_a.data), qt_a, shape_a)
                f32_b = QuantSurgeon.dequantize_tensor(bytes(t_b.data), qt_b, shape_b)

                # Merge
                merged_f32 = merge_fn(f32_a, f32_b)

                # Requantize to A's type (with fallback)
                output_qtype = qt_a
                if qt_a in QuantSurgeon.DEQUANT_ONLY:
                    output_qtype = gguf.GGMLQuantizationType.F16
                merged_bytes = QuantSurgeon.requantize_tensor(merged_f32, output_qtype)

                n_changed = int(np.count_nonzero(merged_f32 != f32_a))
                merged_tensors[name] = (int(output_qtype), shape_a, merged_bytes)
                n_merged += 1
                merge_results.append({
                    "tensor_name": name, "algorithm": algorithm,
                    "success": True, "n_changed": n_changed,
                    "output_qtype": output_qtype.name,
                })
            except Exception as e:
                # Keep A's version on error
                merged_tensors[name] = (int(t_a.tensor_type), shape_a, bytes(t_a.data))
                n_skipped += 1
                merge_results.append({
                    "tensor_name": name, "algorithm": algorithm,
                    "success": False, "n_changed": 0,
                    "error": str(e),
                })

        # Now write the merged GGUF using GGUFWriter
        # Load metadata from reader_a
        from .gguf_surgeon import GGUFSurgeon
        surgeon = GGUFSurgeon(self.source_a)
        surgeon.tensors = merged_tensors

        report = surgeon.write(output_path)

        elapsed = time.time() - t0
        return MergeReport(
            source_a=self.source_a,
            source_b=self.source_b,
            output_path=output_path,
            algorithm=algorithm,
            alpha=alpha,
            n_tensors_merged=n_merged,
            n_tensors_skipped=n_skipped,
            elapsed_seconds=elapsed,
            success=report.success,
            merge_results=merge_results,
            error=report.error,
        )

    @staticmethod
    def _get_arch(reader: gguf.GGUFReader) -> str:
        """Extract the architecture from a GGUF reader."""
        try:
            f = reader.get_field("general.architecture")
            if f and f.data:
                return bytes(f.parts[f.data[0]]).decode("utf-8")
        except Exception:
            pass
        return "unknown"
