"""JSON exporter — dump the full KnowledgeReport as JSON."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Union

from ..extractor import KnowledgeReport


def export_json(report: KnowledgeReport, out_path: Union[str, Path]) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(report) if hasattr(report, "__dataclass_fields__") else report.__dict__
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    return str(out_path)
