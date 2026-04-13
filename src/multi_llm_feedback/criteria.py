"""Criteria: load, merge, (optionally) auto-suggest, validate, and lock criteria."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import yaml


MAX_CRITERIA = 10
MIN_DESCRIPTION_LEN = 20


@dataclass
class Criterion:
    id: str
    name: str
    description: str
    scale: str = "1-10"
    weight: float = 1.0
    required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CriteriaSet:
    """A locked, validated set of criteria ready for evaluation."""

    domain: str
    criteria: list[Criterion] = field(default_factory=list)
    locked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "locked": self.locked,
            "criteria": [c.to_dict() for c in self.criteria],
        }

    def ids(self) -> list[str]:
        return [c.id for c in self.criteria]

    def to_prompt_block(self) -> str:
        lines = [f"CRITERIA (domain={self.domain}):"]
        for c in self.criteria:
            lines.append(
                f"- [{c.id}] {c.name} (scale {c.scale}, weight {c.weight}, "
                f"required={c.required})\n    {c.description}"
            )
        return "\n".join(lines)


class CriteriaError(ValueError):
    """Raised when criteria are invalid or cannot be loaded."""


# ---------------------------------------------------------------------------
# Loading + merging
# ---------------------------------------------------------------------------


def _default_template_dir() -> Path:
    """Locate the bundled templates directory."""
    # templates/ lives at the project root, one level above src/
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "templates",  # packaged inside the module
        here.parent.parent.parent / "templates",  # repo root during dev
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise CriteriaError(f"Could not locate templates directory. Tried: {candidates}")


def load_template(domain: str, template_dir: Optional[Path] = None) -> CriteriaSet:
    """Load the base criteria template for a domain."""
    if not domain:
        raise CriteriaError("domain is required")
    template_dir = template_dir or _default_template_dir()
    path = template_dir / f"{domain}.yaml"
    if not path.exists():
        raise CriteriaError(
            f"No template found for domain {domain!r} at {path}. "
            "Provide one under templates/ or pass a custom template_dir."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_criteria = data.get("criteria") or []
    if not isinstance(raw_criteria, list):
        raise CriteriaError(f"{path}: 'criteria' must be a list")

    criteria = [_criterion_from_dict(item, source=str(path)) for item in raw_criteria]
    return CriteriaSet(domain=data.get("domain", domain), criteria=criteria)


def load_overrides(path: Path) -> dict[str, Any]:
    """Load a user override YAML file. Returns an empty dict if path is None."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise CriteriaError(f"Overrides file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise CriteriaError(f"{path}: overrides must be a YAML mapping at the root")
    return data


def _criterion_from_dict(data: dict[str, Any], source: str = "<input>") -> Criterion:
    if not isinstance(data, dict):
        raise CriteriaError(f"{source}: each criterion must be a mapping, got {type(data).__name__}")
    try:
        cid = str(data["id"]).strip()
        name = str(data["name"]).strip()
        description = str(data.get("description", "")).strip()
    except KeyError as exc:
        raise CriteriaError(f"{source}: criterion missing required key {exc}") from exc
    if not cid:
        raise CriteriaError(f"{source}: criterion id must be non-empty")
    return Criterion(
        id=cid,
        name=name,
        description=description,
        scale=str(data.get("scale", "1-10")),
        weight=float(data.get("weight", 1.0)),
        required=bool(data.get("required", False)),
    )


def merge_overrides(base: CriteriaSet, overrides: dict[str, Any]) -> CriteriaSet:
    """Merge user overrides into a base CriteriaSet.

    Override semantics (all optional):
      - `add`:     list of new criteria dicts
      - `update`:  mapping of criterion_id -> partial dict (patch fields)
      - `remove`:  list of criterion ids to drop (required criteria cannot be removed)
      - `replace`: list of criteria dicts that fully replaces the base list
    """
    if not overrides:
        return CriteriaSet(domain=base.domain, criteria=list(base.criteria))

    if "replace" in overrides:
        replaced = [
            _criterion_from_dict(item, source="overrides.replace")
            for item in overrides["replace"]
        ]
        return CriteriaSet(domain=base.domain, criteria=replaced)

    merged: dict[str, Criterion] = {c.id: Criterion(**c.to_dict()) for c in base.criteria}
    order: list[str] = [c.id for c in base.criteria]

    for cid in overrides.get("remove", []) or []:
        cid = str(cid)
        existing = merged.get(cid)
        if existing is None:
            continue
        if existing.required:
            raise CriteriaError(
                f"Cannot remove required criterion {cid!r}. "
                "Use 'update' to modify it or change the template."
            )
        merged.pop(cid, None)
        order = [x for x in order if x != cid]

    for cid, patch in (overrides.get("update") or {}).items():
        cid = str(cid)
        if cid not in merged:
            raise CriteriaError(f"Cannot update unknown criterion {cid!r}")
        if not isinstance(patch, dict):
            raise CriteriaError(f"update for {cid!r} must be a mapping")
        current = merged[cid].to_dict()
        current.update(patch)
        current["id"] = cid  # id is not patchable
        merged[cid] = _criterion_from_dict(current, source=f"overrides.update.{cid}")

    for item in overrides.get("add", []) or []:
        c = _criterion_from_dict(item, source="overrides.add")
        if c.id in merged:
            raise CriteriaError(
                f"Cannot add criterion {c.id!r}: already exists. Use 'update' instead."
            )
        merged[c.id] = c
        order.append(c.id)

    return CriteriaSet(domain=base.domain, criteria=[merged[cid] for cid in order])


# ---------------------------------------------------------------------------
# Optional auto-suggest pass
# ---------------------------------------------------------------------------


AutoSuggestLLM = Callable[[str, str], str]
"""Signature: (system_prompt, user_prompt) -> raw JSON text."""


AUTO_SUGGEST_SYSTEM = """You help refine an evaluation rubric.
Given a brief and the current criteria list, suggest up to 3 *additional* criteria
that would meaningfully change the evaluation. Do not duplicate existing criteria.
Respond with JSON only:
{"suggestions": [{"id": "...", "name": "...", "description": "...", "scale": "1-10", "weight": 1.0, "required": false}]}
"""


def auto_suggest(
    brief_block: str,
    current: CriteriaSet,
    llm: AutoSuggestLLM,
    max_new: int = 3,
) -> list[Criterion]:
    """Ask an LLM for additional criteria. Returns a (possibly empty) list."""
    user_prompt = (
        f"BRIEF:\n{brief_block}\n\n"
        f"CURRENT CRITERIA:\n{current.to_prompt_block()}\n\n"
        f"Suggest at most {max_new} new criteria."
    )
    raw = llm(AUTO_SUGGEST_SYSTEM, user_prompt)
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        data = json.loads(raw[start : end + 1])
    except (ValueError, json.JSONDecodeError) as exc:
        raise CriteriaError(f"auto_suggest response was not valid JSON: {exc}") from exc

    existing_ids = set(current.ids())
    out: list[Criterion] = []
    for item in (data.get("suggestions") or [])[:max_new]:
        c = _criterion_from_dict(item, source="auto_suggest")
        if c.id in existing_ids:
            continue
        out.append(c)
        existing_ids.add(c.id)
    return out


# ---------------------------------------------------------------------------
# Validation + locking
# ---------------------------------------------------------------------------


def validate(criteria: CriteriaSet) -> None:
    """Raises CriteriaError if the set is invalid."""
    if not criteria.criteria:
        raise CriteriaError("Criteria list is empty")
    if len(criteria.criteria) > MAX_CRITERIA:
        raise CriteriaError(
            f"Too many criteria: {len(criteria.criteria)} > max {MAX_CRITERIA}. "
            "Drop some via overrides.remove or disable auto-suggest."
        )

    seen: set[str] = set()
    for c in criteria.criteria:
        if c.id in seen:
            raise CriteriaError(f"Duplicate criterion id: {c.id!r}")
        seen.add(c.id)
        if not c.name:
            raise CriteriaError(f"Criterion {c.id!r} has empty name")
        if len(c.description) < MIN_DESCRIPTION_LEN:
            raise CriteriaError(
                f"Criterion {c.id!r} has no clear description "
                f"(need at least {MIN_DESCRIPTION_LEN} chars, got {len(c.description)})"
            )
        if c.weight <= 0:
            raise CriteriaError(f"Criterion {c.id!r} must have positive weight, got {c.weight}")


def build_criteria(
    domain: str,
    brief_block: Optional[str] = None,
    overrides_path: Optional[Path] = None,
    *,
    template_dir: Optional[Path] = None,
    auto_suggest_llm: Optional[AutoSuggestLLM] = None,
    max_new_suggestions: int = 3,
) -> CriteriaSet:
    """High-level entry point.

    1. Load base template for `domain`.
    2. Merge user overrides (if any).
    3. Optionally run an auto-suggest LLM pass.
    4. Validate and lock the resulting set.
    """
    base = load_template(domain, template_dir=template_dir)
    overrides = load_overrides(overrides_path) if overrides_path else {}
    merged = merge_overrides(base, overrides)

    if auto_suggest_llm is not None and brief_block:
        room = MAX_CRITERIA - len(merged.criteria)
        if room > 0:
            extras = auto_suggest(
                brief_block,
                merged,
                auto_suggest_llm,
                max_new=min(max_new_suggestions, room),
            )
            merged = CriteriaSet(
                domain=merged.domain,
                criteria=merged.criteria + extras,
            )

    validate(merged)
    merged.locked = True
    return merged
