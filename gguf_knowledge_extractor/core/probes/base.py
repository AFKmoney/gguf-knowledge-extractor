"""
Probe Definitions
=================
A Probe is a single question/prompt sent to the model with metadata about
what kind of knowledge it's testing. A ProbePack is a YAML file containing
a group of probes.

Probe YAML format:
------------------
    name: facts_geography
    description: World geography facts
    category: facts
    domain: geography
    probes:
      - id: capital_france
        prompt: "What is the capital of France? Answer with just the city name."
        expected: "Paris"
        match_mode: contains   # exact | contains | regex | llm_judge | none
        max_tokens: 32
        tags: [europe, capital]
      - id: ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Probe:
    id: str
    prompt: str
    expected: Optional[str] = None
    match_mode: str = "contains"   # exact | contains | regex | llm_judge | none
    max_tokens: int = 256
    temperature: float = 0.0
    system_prompt: Optional[str] = None
    stop: Optional[List[str]] = None
    tags: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbePack:
    name: str
    description: str
    category: str  # facts | concepts | behavioral | calibration
    domain: Optional[str]
    probes: List[Probe]
    path: Optional[str] = None

    def __len__(self) -> int:
        return len(self.probes)


# ---------------------------------------------------------------------- #
# Loader
# ---------------------------------------------------------------------- #
def load_probe_pack(path: str | Path) -> ProbePack:
    """Load a single YAML probe pack file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return _build_pack(data, path=str(path))


def load_probe_packs(directory: str | Path) -> List[ProbePack]:
    """Load all .yaml probe packs in a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    packs = []
    for p in sorted(directory.glob("*.yaml")):
        try:
            packs.append(load_probe_pack(p))
        except Exception as e:
            print(f"[probe_loader] failed to load {p}: {e}")
    return packs


def _build_pack(data: Dict[str, Any], path: Optional[str] = None) -> ProbePack:
    probes = []
    for i, p in enumerate(data.get("probes", [])):
        probes.append(Probe(
            id=p.get("id", f"probe_{i}"),
            prompt=p["prompt"],
            expected=p.get("expected"),
            match_mode=p.get("match_mode", "contains"),
            max_tokens=p.get("max_tokens", 256),
            temperature=p.get("temperature", 0.0),
            system_prompt=p.get("system_prompt"),
            stop=p.get("stop"),
            tags=p.get("tags", []) or [],
            domain=p.get("domain"),
            metadata={k: v for k, v in p.items() if k not in {
                "id", "prompt", "expected", "match_mode", "max_tokens",
                "temperature", "system_prompt", "stop", "tags", "domain",
            }},
        ))
    return ProbePack(
        name=data.get("name", "unnamed"),
        description=data.get("description", ""),
        category=data.get("category", "unknown"),
        domain=data.get("domain"),
        probes=probes,
        path=path,
    )


# ---------------------------------------------------------------------- #
# Matching logic
# ---------------------------------------------------------------------- #
def evaluate_probe(probe: Probe, response: str) -> Dict[str, Any]:
    """Evaluate whether the response matches the probe's expected answer.

    Returns a dict with keys: passed (bool), score (float in [0,1]), reason (str).
    """
    if probe.expected is None or probe.match_mode == "none":
        return {"passed": True, "score": 1.0, "reason": "no expected answer / mode=none"}

    expected = probe.expected.strip()
    actual = response.strip()

    if probe.match_mode == "exact":
        passed = actual.lower() == expected.lower()
        return {"passed": passed, "score": 1.0 if passed else 0.0,
                "reason": "exact match" if passed else f"got '{actual[:80]}', expected '{expected[:80]}'"}

    if probe.match_mode == "contains":
        passed = expected.lower() in actual.lower()
        return {"passed": passed, "score": 1.0 if passed else 0.0,
                "reason": "substring found" if passed else "substring not found"}

    if probe.match_mode == "regex":
        try:
            passed = re.search(expected, actual, re.IGNORECASE | re.MULTILINE) is not None
            return {"passed": passed, "score": 1.0 if passed else 0.0,
                    "reason": "regex matched" if passed else "regex not matched"}
        except re.error as e:
            return {"passed": False, "score": 0.0, "reason": f"invalid regex: {e}"}

    if probe.match_mode == "llm_judge":
        # For LLM-judged probes, we don't auto-evaluate here.
        # The extractor can opt-in to a second-pass judging call.
        return {"passed": True, "score": 1.0, "reason": "llm_judge mode — skipped auto-evaluation"}

    return {"passed": False, "score": 0.0, "reason": f"unknown match_mode: {probe.match_mode}"}


# ---------------------------------------------------------------------- #
# Pack discovery
# ---------------------------------------------------------------------- #
DEFAULT_PACK_DIR = Path(__file__).resolve().parents[2].parent / "probe_packs"


def list_default_packs() -> List[ProbePack]:
    """Load all default probe packs shipped with this package."""
    return load_probe_packs(DEFAULT_PACK_DIR)


def get_pack_by_name(packs: List[ProbePack], name: str) -> Optional[ProbePack]:
    for p in packs:
        if p.name == name:
            return p
    return None
