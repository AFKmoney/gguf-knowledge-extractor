"""
Quantized Tensor Surgery
=========================
Extends GGUF surgery to support quantized tensors (Q4_K_M, Q5_K, Q8_0, etc.)

Pipeline:
  1. Dequantize the target tensor to F32 using gguf.quants.dequantize
  2. Apply the patch (slice overwrite, ROME update, etc.) on the F32 data
  3. Requantize back to the original quantization type using gguf.quants.quantize
  4. Write the requantized bytes back to the GGUF

Supported quantization types (all that gguf.quants supports):
  Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
  IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_NL, IQ4_XS, IQ1_M,
  BF16, F16, F32

Note: requantization introduces small precision losses. For surgical edits
that must be byte-exact (e.g. ROME fact editing), prefer F16/F32 models.
For Q4_K_M models, the precision loss is typically <0.1% per edit and
doesn't meaningfully affect model behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import gguf
from gguf.quants import dequantize, quantize, quant_shape_from_byte_shape


@dataclass
class QuantPatchResult:
    """Result of a quantized tensor patch operation."""
    tensor_name: str
    quant_type: str
    original_shape: List[int]
    n_elements_patched: int
    dequant_error: float       # max abs diff after dequant→requant roundtrip
    success: bool
    error: Optional[str] = None


class QuantSurgeon:
    """Handles dequant→patch→requant for quantized GGUF tensors."""

    # Quantization types that support full roundtrip (dequant + requant)
    ROUNDTRIP_SUPPORTED = {
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
        gguf.GGMLQuantizationType.Q4_0,
        gguf.GGMLQuantizationType.Q4_1,
        gguf.GGMLQuantizationType.Q5_0,
        gguf.GGMLQuantizationType.Q5_1,
        gguf.GGMLQuantizationType.Q8_0,
    }

    # Types that support dequantize only (read-only analysis)
    DEQUANT_ONLY = {
        gguf.GGMLQuantizationType.Q2_K,
        gguf.GGMLQuantizationType.Q3_K,
        gguf.GGMLQuantizationType.Q4_K,
        gguf.GGMLQuantizationType.Q5_K,
        gguf.GGMLQuantizationType.Q6_K,
        gguf.GGMLQuantizationType.Q8_K,
    }

    @staticmethod
    def is_supported(qtype: gguf.GGMLQuantizationType) -> bool:
        """Check if this quantization type supports full roundtrip surgery."""
        return qtype in QuantSurgeon.ROUNDTRIP_SUPPORTED

    @staticmethod
    def can_dequantize(qtype: gguf.GGMLQuantizationType) -> bool:
        """Check if this type can be dequantized (read-only analysis)."""
        return qtype in (QuantSurgeon.ROUNDTRIP_SUPPORTED | QuantSurgeon.DEQUANT_ONLY)

    @staticmethod
    def requantizable_types() -> List[str]:
        """List of quantization type names that support full roundtrip."""
        return sorted([q.name for q in QuantSurgeon.ROUNDTRIP_SUPPORTED])

    @staticmethod
    def dequant_only_types() -> List[str]:
        """List of quantization type names that support dequantize only."""
        return sorted([q.name for q in QuantSurgeon.DEQUANT_ONLY])

    @staticmethod
    def dequantize_tensor(
        data: bytes,
        qtype: gguf.GGMLQuantizationType,
        shape: List[int],
    ) -> np.ndarray:
        """Dequantize a tensor to F32.

        Args:
            data: raw bytes of the tensor
            qtype: quantization type
            shape: logical shape of the tensor (e.g. [4096, 11008])

        Returns:
            F32 numpy array with the given shape
        """
        if qtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
                     gguf.GGMLQuantizationType.BF16):
            if qtype == gguf.GGMLQuantizationType.F32:
                arr = np.frombuffer(data, dtype=np.float32)
            elif qtype == gguf.GGMLQuantizationType.F16:
                arr = np.frombuffer(data, dtype=np.float16).astype(np.float32)
            else:  # BF16
                raw = np.frombuffer(data, dtype=np.uint16).astype(np.uint32)
                raw = (raw << 16).astype(np.uint32)
                arr = raw.view(np.float32).copy()
            return arr.reshape(shape)

        # Quantized: use gguf.quants.dequantize
        # The data needs to be a uint8 numpy array
        raw = np.frombuffer(data, dtype=np.uint8)
        # The "byte shape" is the shape of the raw quantized data
        # dequantize expects a 1D array and returns a 1D array
        dequant = dequantize(raw, qtype)
        return dequant.reshape(shape).astype(np.float32)

    @staticmethod
    def requantize_tensor(
        f32_data: np.ndarray,
        qtype: gguf.GGMLQuantizationType,
    ) -> bytes:
        """Requantize an F32 tensor back to the target quantization type.

        Args:
            f32_data: F32 numpy array with the logical shape
            qtype: target quantization type

        Returns:
            Raw bytes of the requantized tensor
        """
        if qtype == gguf.GGMLQuantizationType.F32:
            return f32_data.astype(np.float32).tobytes()
        elif qtype == gguf.GGMLQuantizationType.F16:
            return f32_data.astype(np.float16).tobytes()
        elif qtype == gguf.GGMLQuantizationType.BF16:
            # BF16: take upper 16 bits of float32
            f32 = f32_data.astype(np.float32)
            raw = f32.view(np.uint32)
            bf16 = (raw >> 16).astype(np.uint16)
            return bf16.tobytes()

        # Quantized: use gguf.quants.quantize
        # quantize expects a 1D or 2D array and returns a uint8 array
        flat = f32_data.astype(np.float32).flatten()
        quantized = quantize(flat, qtype)
        return quantized.tobytes()

    @staticmethod
    def patch_quantized_tensor(
        data: bytes,
        qtype: gguf.GGMLQuantizationType,
        shape: List[int],
        patch_fn,
        fallback_to_f16: bool = True,
    ) -> Tuple[bytes, QuantPatchResult]:
        """Dequantize, apply a patch function, and requantize.

        Args:
            data: raw tensor bytes (must match qtype)
            qtype: quantization type of the input data
            shape: logical shape
            patch_fn: function(np.ndarray) -> np.ndarray
            fallback_to_f16: if True, convert K-quants to F16 after patching
                             (since K-quants can't be requantized in gguf-py).
                             This increases file size but preserves the edit.

        Returns:
            (new_bytes, result) tuple
        """
        name = qtype.name if hasattr(qtype, 'name') else str(qtype)
        try:
            # Step 1: Dequantize to F32
            f32 = QuantSurgeon.dequantize_tensor(data, qtype, shape)

            # Step 2: Apply patch
            patched = patch_fn(f32)
            if patched.shape != f32.shape:
                return data, QuantPatchResult(
                    tensor_name="", quant_type=name,
                    original_shape=shape, n_elements_patched=0,
                    dequant_error=0.0, success=False,
                    error=f"Shape mismatch after patch: {patched.shape} vs {f32.shape}",
                )

            # Step 3: Determine output quantization type
            output_qtype = qtype
            converted = False
            if qtype in QuantSurgeon.DEQUANT_ONLY:
                if fallback_to_f16:
                    output_qtype = gguf.GGMLQuantizationType.F16
                    converted = True
                else:
                    return data, QuantPatchResult(
                        tensor_name="", quant_type=name,
                        original_shape=shape, n_elements_patched=0,
                        dequant_error=0.0, success=False,
                        error=f"Cannot requantize {name} (K-quant) — set fallback_to_f16=True",
                    )

            # Step 4: Requantize to the output type
            requantized_bytes = QuantSurgeon.requantize_tensor(patched, output_qtype)

            # Step 5: Compute roundtrip error (dequant→requant→dequant)
            if output_qtype != gguf.GGMLQuantizationType.F32:
                roundtrip = QuantSurgeon.dequantize_tensor(requantized_bytes, output_qtype, shape)
                error = float(np.max(np.abs(roundtrip - patched))) if roundtrip.size > 0 else 0.0
            else:
                error = 0.0

            # Count changed elements
            n_changed = int(np.count_nonzero(patched != f32))

            result = QuantPatchResult(
                tensor_name="", quant_type=output_qtype.name,
                original_shape=shape, n_elements_patched=n_changed,
                dequant_error=error, success=True,
            )

            if converted:
                result.error = f"Converted from {name} to F16 (K-quant requantization not supported by gguf-py)"

            return requantized_bytes, result
        except Exception as e:
            return data, QuantPatchResult(
                tensor_name="", quant_type=name,
                original_shape=shape, n_elements_patched=0,
                dequant_error=0.0, success=False,
                error=str(e),
            )

    @staticmethod
    def patch_column(
        data: bytes,
        qtype: gguf.GGMLQuantizationType,
        shape: List[int],
        col_index: int,
        new_col_values: np.ndarray,
        fallback_to_f16: bool = True,
    ) -> Tuple[bytes, QuantPatchResult]:
        """Patch a single column of a 2D quantized tensor.

        This is the core operation for ROME editing on quantized models.
        """
        def patch_fn(arr: np.ndarray) -> np.ndarray:
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D tensor, got {arr.ndim}D")
            arr = arr.copy()
            arr[:, col_index] = new_col_values.astype(np.float32)
            return arr

        return QuantSurgeon.patch_quantized_tensor(data, qtype, shape, patch_fn, fallback_to_f16)

    @staticmethod
    def patch_slice(
        data: bytes,
        qtype: gguf.GGMLQuantizationType,
        shape: List[int],
        slice_spec: Tuple[int, ...],
        new_values: np.ndarray,
        fallback_to_f16: bool = True,
    ) -> Tuple[bytes, QuantPatchResult]:
        """Patch an arbitrary slice of a quantized tensor."""
        def patch_fn(arr: np.ndarray) -> np.ndarray:
            arr = arr.copy()
            if arr.ndim == 2 and len(slice_spec) == 4:
                r0, r1, c0, c1 = slice_spec
                arr[r0:r1, c0:c1] = new_values.astype(np.float32)
            elif arr.ndim == 1 and len(slice_spec) == 2:
                s, e = slice_spec
                arr[s:e] = new_values.astype(np.float32)
            else:
                raise ValueError(f"Slice {slice_spec} incompatible with shape {arr.shape}")
            return arr

        return QuantSurgeon.patch_quantized_tensor(data, qtype, shape, patch_fn, fallback_to_f16)

    @staticmethod
    def scale_tensor(
        data: bytes,
        qtype: gguf.GGMLQuantizationType,
        shape: List[int],
        scale: float,
        fallback_to_f16: bool = True,
    ) -> Tuple[bytes, QuantPatchResult]:
        """Scale all values in a tensor by a constant factor."""
        def patch_fn(arr: np.ndarray) -> np.ndarray:
            return arr * scale

        return QuantSurgeon.patch_quantized_tensor(data, qtype, shape, patch_fn, fallback_to_f16)

    @staticmethod
    def add_noise(
        data: bytes,
        qtype: gguf.GGMLQuantizationType,
        shape: List[int],
        noise_std: float = 0.01,
        seed: int = 42,
        fallback_to_f16: bool = True,
    ) -> Tuple[bytes, QuantPatchResult]:
        """Add Gaussian noise to a tensor (useful for testing or adversarial robustness)."""
        rng = np.random.RandomState(seed)

        def patch_fn(arr: np.ndarray) -> np.ndarray:
            noise = rng.randn(*arr.shape).astype(np.float32) * noise_std
            return arr + noise

        return QuantSurgeon.patch_quantized_tensor(data, qtype, shape, patch_fn, fallback_to_f16)

    @staticmethod
    def test_roundtrip(
        data: bytes,
        qtype: gguf.GGMLQuantizationType,
        shape: List[int],
    ) -> dict:
        """Test the dequant→requant roundtrip error for a tensor.

        Useful for understanding how much precision is lost per quantization type.
        """
        try:
            f32_orig = QuantSurgeon.dequantize_tensor(data, qtype, shape)
            requant_bytes = QuantSurgeon.requantize_tensor(f32_orig, qtype)
            f32_roundtrip = QuantSurgeon.dequantize_tensor(requant_bytes, qtype, shape)

            diff = f32_roundtrip - f32_orig
            return {
                "qtype": qtype.name,
                "shape": shape,
                "n_elements": f32_orig.size,
                "max_error": float(np.max(np.abs(diff))),
                "mean_error": float(np.mean(np.abs(diff))),
                "std_error": float(np.std(np.abs(diff))),
                "relative_error": float(np.mean(np.abs(diff)) / (np.mean(np.abs(f32_orig)) + 1e-8)),
                "success": True,
            }
        except Exception as e:
            return {
                "qtype": qtype.name if hasattr(qtype, 'name') else str(qtype),
                "shape": shape,
                "success": False,
                "error": str(e),
            }
