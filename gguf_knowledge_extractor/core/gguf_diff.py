"""
GGUF Diff
==========
Compare two GGUF files at the byte, metadata, and tensor level.

Produces a structured diff showing:
  - Added/removed/modified metadata fields
  - Added/removed/modified tensors
  - Per-tensor byte-level differences (when shapes match)
  - Per-tensor statistical differences (mean, std, norm delta)
  - Summary statistics (how similar the two models are overall)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gguf

from .quant_surgery import QuantSurgeon


@dataclass
class TensorDiff:
    """Diff of a single tensor between two GGUFs."""
    name: str
    status: str  # "same" | "modified" | "added" | "removed" | "shape_changed" | "type_changed"
    shape_a: Optional[List[int]]
    shape_b: Optional[List[int]]
    qtype_a: Optional[str]
    qtype_b: Optional[str]
    n_bytes_a: int
    n_bytes_b: int
    # For modified tensors:
    byte_diff_count: int = 0      # number of differing bytes
    byte_diff_percent: float = 0.0
    f32_mean_abs_diff: float = 0.0  # mean absolute difference in F32 space
    f32_max_abs_diff: float = 0.0   # max absolute difference
    f32_cosine_sim: float = 0.0     # cosine similarity (1.0 = identical direction)
    f32_norm_ratio: float = 0.0     # ||b|| / ||a||


@dataclass
class MetadataDiff:
    """Diff of a single metadata field."""
    key: str
    status: str  # "same" | "modified" | "added" | "removed"
    value_a: Optional[Any]
    value_b: Optional[Any]


@dataclass
class DiffReport:
    """Full diff report between two GGUF files."""
    source_a: str
    source_b: str
    file_size_a: int
    file_size_b: int
    n_tensors_a: int
    n_tensors_b: int
    n_metadata_a: int
    n_metadata_b: int
    metadata_diffs: List[Dict[str, Any]]
    tensor_diffs: List[Dict[str, Any]]
    summary: Dict[str, Any]
    elapsed_seconds: float


class GGUFDiffer:
    """Compare two GGUF files."""

    def __init__(self, source_a: str, source_b: str):
        self.source_a = source_a
        self.source_b = source_b
        self.reader_a = gguf.GGUFReader(source_a)
        self.reader_b = gguf.GGUFReader(source_b)

    def diff(self, sample_size: int = 100000) -> DiffReport:
        """Compute the full diff.

        Args:
            sample_size: for large tensors, sample this many elements for statistical diff
        """
        t0 = time.time()

        import os
        size_a = os.path.getsize(self.source_a)
        size_b = os.path.getsize(self.source_b)

        # Metadata diff
        meta_a = self._load_metadata(self.reader_a)
        meta_b = self._load_metadata(self.reader_b)
        meta_diffs = self._diff_metadata(meta_a, meta_b)

        # Tensor diff
        tensors_a = {self._tensor_name(t): t for t in self.reader_a.tensors}
        tensors_b = {self._tensor_name(t): t for t in self.reader_b.tensors}
        all_names = sorted(set(tensors_a.keys()) | set(tensors_b.keys()))

        tensor_diffs: List[Dict[str, Any]] = []
        n_same = 0
        n_modified = 0
        n_added = 0
        n_removed = 0
        total_cosine_sim = 0.0
        n_cosine = 0

        for name in all_names:
            t_a = tensors_a.get(name)
            t_b = tensors_b.get(name)

            if t_a is not None and t_b is None:
                td = TensorDiff(
                    name=name, status="removed",
                    shape_a=[int(s) for s in t_a.shape], shape_b=None,
                    qtype_a=gguf.GGMLQuantizationType(t_a.tensor_type).name, qtype_b=None,
                    n_bytes_a=int(t_a.n_bytes), n_bytes_b=0,
                )
                n_removed += 1
            elif t_a is None and t_b is not None:
                td = TensorDiff(
                    name=name, status="added",
                    shape_a=None, shape_b=[int(s) for s in t_b.shape],
                    qtype_a=None, qtype_b=gguf.GGMLQuantizationType(t_b.tensor_type).name,
                    n_bytes_a=0, n_bytes_b=int(t_b.n_bytes),
                )
                n_added += 1
            else:
                shape_a = [int(s) for s in t_a.shape]
                shape_b = [int(s) for s in t_b.shape]
                qt_a = gguf.GGMLQuantizationType(t_a.tensor_type)
                qt_b = gguf.GGMLQuantizationType(t_b.tensor_type)

                if shape_a != shape_b:
                    td = TensorDiff(
                        name=name, status="shape_changed",
                        shape_a=shape_a, shape_b=shape_b,
                        qtype_a=qt_a.name, qtype_b=qt_b.name,
                        n_bytes_a=int(t_a.n_bytes), n_bytes_b=int(t_b.n_bytes),
                    )
                    n_modified += 1
                elif qt_a != qt_b:
                    td = TensorDiff(
                        name=name, status="type_changed",
                        shape_a=shape_a, shape_b=shape_b,
                        qtype_a=qt_a.name, qtype_b=qt_b.name,
                        n_bytes_a=int(t_a.n_bytes), n_bytes_b=int(t_b.n_bytes),
                    )
                    n_modified += 1
                else:
                    # Check if data is identical
                    data_a = bytes(t_a.data)
                    data_b = bytes(t_b.data)
                    if data_a == data_b:
                        td = TensorDiff(
                            name=name, status="same",
                            shape_a=shape_a, shape_b=shape_b,
                            qtype_a=qt_a.name, qtype_b=qt_b.name,
                            n_bytes_a=len(data_a), n_bytes_b=len(data_b),
                        )
                        n_same += 1
                    else:
                        # Compute byte diff
                        min_len = min(len(data_a), len(data_b))
                        byte_diff = sum(1 for i in range(min_len) if data_a[i] != data_b[i])

                        # Compute F32 statistical diff
                        try:
                            f32_a = QuantSurgeon.dequantize_tensor(data_a, qt_a, shape_a)
                            f32_b = QuantSurgeon.dequantize_tensor(data_b, qt_b, shape_b)

                            # Sample for large tensors
                            if f32_a.size > sample_size:
                                idx = np.random.choice(f32_a.size, sample_size, replace=False)
                                s_a = f32_a.flatten()[idx]
                                s_b = f32_b.flatten()[idx]
                            else:
                                s_a = f32_a.flatten()
                                s_b = f32_b.flatten()

                            diff = s_a - s_b
                            mean_abs_diff = float(np.mean(np.abs(diff)))
                            max_abs_diff = float(np.max(np.abs(diff)))

                            # Cosine similarity
                            norm_a = np.linalg.norm(s_a) + 1e-10
                            norm_b = np.linalg.norm(s_b) + 1e-10
                            cos_sim = float(np.dot(s_a, s_b) / (norm_a * norm_b))
                            norm_ratio = float(norm_b / norm_a)

                            total_cosine_sim += cos_sim
                            n_cosine += 1

                            td = TensorDiff(
                                name=name, status="modified",
                                shape_a=shape_a, shape_b=shape_b,
                                qtype_a=qt_a.name, qtype_b=qt_b.name,
                                n_bytes_a=len(data_a), n_bytes_b=len(data_b),
                                byte_diff_count=byte_diff,
                                byte_diff_percent=byte_diff / max(min_len, 1) * 100,
                                f32_mean_abs_diff=mean_abs_diff,
                                f32_max_abs_diff=max_abs_diff,
                                f32_cosine_sim=cos_sim,
                                f32_norm_ratio=norm_ratio,
                            )
                        except Exception as e:
                            td = TensorDiff(
                                name=name, status="modified",
                                shape_a=shape_a, shape_b=shape_b,
                                qtype_a=qt_a.name, qtype_b=qt_b.name,
                                n_bytes_a=len(data_a), n_bytes_b=len(data_b),
                                byte_diff_count=byte_diff,
                                byte_diff_percent=byte_diff / max(min_len, 1) * 100,
                            )
                        n_modified += 1

            tensor_diffs.append(asdict(td))

        avg_cosine = total_cosine_sim / n_cosine if n_cosine > 0 else 1.0

        # When no tensors are modified, similarity is 100%
        if n_modified == 0 and n_same > 0:
            avg_cosine = 1.0

        return DiffReport(
            source_a=self.source_a,
            source_b=self.source_b,
            file_size_a=size_a,
            file_size_b=size_b,
            n_tensors_a=len(self.reader_a.tensors),
            n_tensors_b=len(self.reader_b.tensors),
            n_metadata_a=len(meta_a),
            n_metadata_b=len(meta_b),
            metadata_diffs=meta_diffs,
            tensor_diffs=tensor_diffs,
            summary={
                "n_tensors_same": n_same,
                "n_tensors_modified": n_modified,
                "n_tensors_added": n_added,
                "n_tensors_removed": n_removed,
                "avg_cosine_similarity": avg_cosine,
                "overall_similarity_percent": avg_cosine * 100,
            },
            elapsed_seconds=time.time() - t0,
        )

    @staticmethod
    def _tensor_name(t) -> str:
        return t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)

    @staticmethod
    def _load_metadata(reader: gguf.GGUFReader) -> Dict[str, Any]:
        """Load all metadata fields as a dict."""
        out = {}
        for name in reader.fields.keys():
            if name.startswith("GGUF."):
                continue
            try:
                f = reader.get_field(name)
                is_array = (len(f.types) >= 1 and int(f.types[0]) == 9)
                if is_array:
                    elem_type = int(f.types[1]) if len(f.types) >= 2 else 8
                    value = []
                    if elem_type == 8:  # STRING
                        for i in range(len(f.data)):
                            str_part = f.parts[f.data[i]]
                            try:
                                value.append(bytes(str_part).decode("utf-8", errors="replace"))
                            except Exception:
                                value.append(str(str_part))
                    else:
                        if len(f.data) >= 2:
                            arr = f.parts[f.data[1]]
                            try:
                                value = arr.tolist()
                            except Exception:
                                value = list(arr)
                else:
                    type_id = int(f.types[0])
                    if f.data and len(f.data) > 0:
                        part = f.parts[f.data[0]]
                        if type_id == 8:  # STRING
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
                out[name] = value
            except Exception:
                pass
        return out

    @staticmethod
    def _diff_metadata(a: Dict[str, Any], b: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compute metadata diff."""
        all_keys = sorted(set(a.keys()) | set(b.keys()))
        diffs = []
        for key in all_keys:
            in_a = key in a
            in_b = key in b
            if in_a and in_b:
                if a[key] == b[key]:
                    continue  # skip identical
                diffs.append({
                    "key": key, "status": "modified",
                    "value_a": str(a[key])[:200], "value_b": str(b[key])[:200],
                })
            elif in_a and not in_b:
                diffs.append({
                    "key": key, "status": "removed",
                    "value_a": str(a[key])[:200], "value_b": None,
                })
            else:
                diffs.append({
                    "key": key, "status": "added",
                    "value_a": None, "value_b": str(b[key])[:200],
                })
        return diffs
