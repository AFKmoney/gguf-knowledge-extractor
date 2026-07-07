"""
Smart Quantizer
===============
Quantize GGUF models with optional imatrix guidance.

Without imatrix: standard uniform quantization (every tensor gets the same qtype)
With imatrix: per-tensor quantization based on importance scores
  - Important tensors → higher precision (F16, Q8_0)
  - Unimportant tensors → lower precision (Q4_0)
  - Result: smaller file with better quality than uniform quantization

Supports: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
(K-quants Q4_K etc. are not supported for requantization by gguf-py)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import gguf

from .quant_surgery import QuantSurgeon
from .gguf_surgeon import GGUFSurgeon


@dataclass
class QuantizationResult:
    """Result of quantizing a single tensor."""
    tensor_name: str
    input_qtype: str
    output_qtype: str
    input_bytes: int
    output_bytes: int
    roundtrip_error: float
    success: bool
    error: Optional[str] = None


@dataclass
class QuantizationReport:
    """Full quantization report."""
    source_path: str
    output_path: str
    target_qtype: str
    used_imatrix: bool
    n_tensors_quantized: int
    n_tensors_skipped: int
    n_tensors_kept_high_precision: int  # tensors kept in F16 due to importance
    input_size_bytes: int
    output_size_bytes: int
    compression_ratio: float
    avg_roundtrip_error: float
    max_roundtrip_error: float
    results: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    success: bool = False
    error: Optional[str] = None


class SmartQuantizer:
    """Quantize GGUF models with optional imatrix guidance."""

    # Quantization types that can be output
    OUTPUT_TYPES = {
        "F32": gguf.GGMLQuantizationType.F32,
        "F16": gguf.GGMLQuantizationType.F16,
        "BF16": gguf.GGMLQuantizationType.BF16,
        "Q4_0": gguf.GGMLQuantizationType.Q4_0,
        "Q4_1": gguf.GGMLQuantizationType.Q4_1,
        "Q5_0": gguf.GGMLQuantizationType.Q5_0,
        "Q5_1": gguf.GGMLQuantizationType.Q5_1,
        "Q8_0": gguf.GGMLQuantizationType.Q8_0,
    }

    @staticmethod
    def get_supported_qtypes() -> List[str]:
        return sorted(SmartQuantizer.OUTPUT_TYPES.keys())

    def __init__(
        self,
        source_path: str,
        imatrix_report: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            source_path: path to the source GGUF
            imatrix_report: optional imatrix report dict (from ImatrixComputer)
                           with 'tensor_importances' and 'precision_recommendations'
        """
        self.source_path = source_path
        self.imatrix_report = imatrix_report
        # Build importance lookup
        self.importance_lookup: Dict[str, float] = {}
        self.precision_recs: Dict[str, str] = {}
        if imatrix_report:
            for ti in imatrix_report.get("tensor_importances", []):
                # Map tensor name to importance score
                # The imatrix uses names like "blk.0.attn_q" — we need to
                # match these to actual tensor names like "blk.0.attn_q.weight"
                self.importance_lookup[ti["name"]] = ti["importance_score"]
            self.precision_recs = imatrix_report.get("precision_recommendations", {})

    def _get_target_qtype_for_tensor(
        self,
        tensor_name: str,
        default_qtype: gguf.GGMLQuantizationType,
        use_imatrix: bool,
    ) -> gguf.GGMLQuantizationType:
        """Determine the target quantization type for a tensor.

        If using imatrix, look up the precision recommendation.
        Otherwise, use the default.
        """
        if not use_imatrix or not self.precision_recs:
            return default_qtype

        # Try exact match first
        rec = self.precision_recs.get(tensor_name)
        if rec:
            return self.OUTPUT_TYPES.get(rec, default_qtype)

        # Try matching by stripping ".weight" suffix
        base_name = tensor_name.replace(".weight", "")
        rec = self.precision_recs.get(base_name)
        if rec:
            return self.OUTPUT_TYPES.get(rec, default_qtype)

        return default_qtype

    def quantize(
        self,
        output_path: str,
        target_qtype: str = "Q4_0",
        use_imatrix: bool = False,
        keep_embedding_f16: bool = True,
        keep_norm_f32: bool = True,
    ) -> QuantizationReport:
        """Quantize the model.

        Args:
            output_path: where to write the quantized GGUF
            target_qtype: default quantization type (Q4_0, Q5_0, Q8_0, F16, etc.)
            use_imatrix: if True, use the imatrix to vary precision per tensor
            keep_embedding_f16: keep token_embd and output in F16 (always)
            keep_norm_f32: keep norm tensors in F32 (always)
        """
        t0 = time.time()
        import os

        try:
            default_qtype = self.OUTPUT_TYPES.get(target_qtype)
            if default_qtype is None:
                return QuantizationReport(
                    source_path=self.source_path, output_path=output_path,
                    target_qtype=target_qtype, used_imatrix=use_imatrix,
                    n_tensors_quantized=0, n_tensors_skipped=0, n_tensors_kept_high_precision=0,
                    input_size_bytes=0, output_size_bytes=0, compression_ratio=0,
                    avg_roundtrip_error=0, max_roundtrip_error=0,
                    error=f"Unsupported target qtype: {target_qtype}",
                )

            # Load the source GGUF
            surgeon = GGUFSurgeon(self.source_path)

            input_size = os.path.getsize(self.source_path)
            results: List[Dict[str, Any]] = []
            n_quantized = 0
            n_skipped = 0
            n_kept_high = 0
            errors = []
            max_err = 0.0
            total_err = 0.0
            n_err_samples = 0

            for tname, (tensor_type, shape, data) in list(surgeon.tensors.items()):
                qt_in = gguf.GGMLQuantizationType(tensor_type)
                input_bytes = len(data)

                # Determine target qtype
                if keep_embedding_f16 and (tname in ("token_embd.weight", "output.weight") or
                                            "embed" in tname.lower()):
                    target = gguf.GGMLQuantizationType.F16
                    n_kept_high += 1
                elif keep_norm_f32 and ("norm" in tname.lower() or tname == "output_norm.weight"):
                    target = gguf.GGMLQuantizationType.F32
                    n_kept_high += 1
                else:
                    target = self._get_target_qtype_for_tensor(tname, default_qtype, use_imatrix)
                    if target != default_qtype and use_imatrix:
                        n_kept_high += 1

                # Skip if already at target type
                if qt_in == target:
                    results.append({
                        "tensor_name": tname, "input_qtype": qt_in.name,
                        "output_qtype": target.name, "input_bytes": input_bytes,
                        "output_bytes": input_bytes, "roundtrip_error": 0.0,
                        "success": True, "skipped": True,
                    })
                    n_skipped += 1
                    continue

                # Dequantize
                try:
                    f32 = QuantSurgeon.dequantize_tensor(data, qt_in, shape)
                except Exception as e:
                    results.append({
                        "tensor_name": tname, "input_qtype": qt_in.name,
                        "output_qtype": qt_in.name, "input_bytes": input_bytes,
                        "output_bytes": input_bytes, "roundtrip_error": 0.0,
                        "success": False, "error": f"dequant failed: {e}",
                    })
                    n_skipped += 1
                    continue

                # Requantize
                try:
                    new_bytes = QuantSurgeon.requantize_tensor(f32, target)
                except Exception as e:
                    # Fallback to F16 if target type fails
                    try:
                        target = gguf.GGMLQuantizationType.F16
                        new_bytes = QuantSurgeon.requantize_tensor(f32, target)
                    except Exception as e2:
                        results.append({
                            "tensor_name": tname, "input_qtype": qt_in.name,
                            "output_qtype": qt_in.name, "input_bytes": input_bytes,
                            "output_bytes": input_bytes, "roundtrip_error": 0.0,
                            "success": False, "error": f"requant failed: {e2}",
                        })
                        n_skipped += 1
                        continue

                output_bytes = len(new_bytes)

                # Compute roundtrip error
                try:
                    roundtrip_f32 = QuantSurgeon.dequantize_tensor(new_bytes, target, shape)
                    err = float(np.max(np.abs(roundtrip_f32 - f32))) if roundtrip_f32.size > 0 else 0.0
                except Exception:
                    err = 0.0

                if err > max_err:
                    max_err = err
                total_err += err
                n_err_samples += 1

                # Update the tensor in the surgeon
                surgeon.tensors[tname] = (int(target), shape, new_bytes)
                n_quantized += 1

                results.append({
                    "tensor_name": tname, "input_qtype": qt_in.name,
                    "output_qtype": target.name, "input_bytes": input_bytes,
                    "output_bytes": output_bytes, "roundtrip_error": err,
                    "success": True,
                })

            # Write the quantized GGUF
            write_report = surgeon.write(output_path)
            output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

            avg_err = total_err / n_err_samples if n_err_samples > 0 else 0.0
            compression_ratio = input_size / max(output_size, 1)

            return QuantizationReport(
                source_path=self.source_path,
                output_path=output_path,
                target_qtype=target_qtype,
                used_imatrix=use_imatrix,
                n_tensors_quantized=n_quantized,
                n_tensors_skipped=n_skipped,
                n_tensors_kept_high_precision=n_kept_high,
                input_size_bytes=input_size,
                output_size_bytes=output_size,
                compression_ratio=compression_ratio,
                avg_roundtrip_error=avg_err,
                max_roundtrip_error=max_err,
                results=results,
                elapsed_seconds=time.time() - t0,
                success=write_report.success,
                error=write_report.error,
            )
        except Exception as e:
            import traceback
            return QuantizationReport(
                source_path=self.source_path, output_path=output_path,
                target_qtype=target_qtype, used_imatrix=use_imatrix,
                n_tensors_quantized=0, n_tensors_skipped=0, n_tensors_kept_high_precision=0,
                input_size_bytes=0, output_size_bytes=0, compression_ratio=0,
                avg_roundtrip_error=0, max_roundtrip_error=0,
                error=f"{e}\n{traceback.format_exc()}",
            )
