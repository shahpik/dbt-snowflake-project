"""Disagreement analysis: stdev, verdict agreement, embedding-based clustering."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Optional

from .criteria import CriteriaSet
from .evaluators import EvaluationResponse, Verdict


# Verdicts mapped to an ordinal scale so we can measure distance.
VERDICT_ORDINAL: dict[str, int] = {
    "strong_no": 1,
    "no": 2,
    "maybe": 3,
    "yes": 4,
    "strong_yes": 5,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CriterionDiff:
    id: str
    scores: dict[str, float]     # model_label -> score
    mean: float
    stdev: float
    spread: float                # max - min
    classification: str          # consensus / soft_split / hard_split / outlier


@dataclass
class Cluster:
    label: str
    items: list[str]
    members: list[str]           # which model labels contributed


@dataclass
class ThemeClusters:
    risks: list[Cluster] = field(default_factory=list)
    strengths: list[Cluster] = field(default_factory=list)
    assumptions: list[Cluster] = field(default_factory=list)


@dataclass
class DisagreementReport:
    overall: str                 # consensus / soft_split / hard_split / outlier
    score_diffs: list[CriterionDiff]
    verdict_agreement: str       # unanimous / majority / split
    verdict_by_model: dict[str, str]
    verdict_spread: int          # max ordinal - min ordinal
    clusters: ThemeClusters
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "verdict_agreement": self.verdict_agreement,
            "verdict_by_model": self.verdict_by_model,
            "verdict_spread": self.verdict_spread,
            "score_diffs": [asdict(s) for s in self.score_diffs],
            "clusters": {
                "risks": [asdict(c) for c in self.clusters.risks],
                "strengths": [asdict(c) for c in self.clusters.strengths],
                "assumptions": [asdict(c) for c in self.clusters.assumptions],
            },
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Score statistics
# ---------------------------------------------------------------------------


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _classify_criterion(stdev: float, spread: float) -> str:
    # On a 1-10 scale with 3 raters:
    if stdev < 1.0 and spread <= 2.0:
        return "consensus"
    if stdev < 2.0 and spread <= 4.0:
        return "soft_split"
    if spread >= 6.0:
        return "hard_split"
    return "hard_split" if stdev >= 2.0 else "soft_split"


def score_diffs(
    responses: list[EvaluationResponse],
    criteria: CriteriaSet,
) -> list[CriterionDiff]:
    diffs: list[CriterionDiff] = []
    for cid in criteria.ids():
        per_model = {r.model_label: r.score_map().get(cid) for r in responses}
        vals = [v for v in per_model.values() if v is not None]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        sd = _stdev(vals)
        spread = max(vals) - min(vals)
        diffs.append(
            CriterionDiff(
                id=cid,
                scores={k: float(v) for k, v in per_model.items() if v is not None},
                mean=mean,
                stdev=sd,
                spread=spread,
                classification=_classify_criterion(sd, spread),
            )
        )
    return diffs


# ---------------------------------------------------------------------------
# Verdict agreement
# ---------------------------------------------------------------------------


def verdict_agreement(responses: list[EvaluationResponse]) -> tuple[str, dict[str, str], int]:
    by_model = {r.model_label: r.verdict.value for r in responses}
    verdicts = list(by_model.values())
    if not verdicts:
        return "split", {}, 0

    ordinals = [VERDICT_ORDINAL[v] for v in verdicts]
    spread = max(ordinals) - min(ordinals)

    if len(set(verdicts)) == 1:
        agreement = "unanimous"
    else:
        counts: dict[str, int] = {}
        for v in verdicts:
            counts[v] = counts.get(v, 0) + 1
        top = max(counts.values())
        agreement = "majority" if top >= 2 else "split"
    return agreement, by_model, spread


# ---------------------------------------------------------------------------
# Embedding-based clustering
# ---------------------------------------------------------------------------


EmbedFn = Callable[[list[str]], list[list[float]]]
"""Signature: (texts) -> list of embedding vectors."""


def default_embed_fn(model: str = "text-embedding-3-small") -> EmbedFn:
    """Factory that returns a default embedding function backed by OpenAI."""

    def _embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        from openai import OpenAI

        client = OpenAI()
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    return _embed


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def cluster_texts(
    items: list[tuple[str, str]],   # (model_label, text)
    embed_fn: EmbedFn,
    threshold: float = 0.78,
) -> list[Cluster]:
    """Greedy single-link clustering over cosine similarity of embeddings."""
    if not items:
        return []

    texts = [t for _, t in items]
    try:
        vectors = embed_fn(texts)
    except Exception:
        # If embeddings fail, fall back to "every item is its own cluster"
        return [
            Cluster(label=text, items=[text], members=[label])
            for label, text in items
        ]

    if len(vectors) != len(items):
        return [
            Cluster(label=text, items=[text], members=[label])
            for label, text in items
        ]

    clusters: list[dict[str, Any]] = []
    for (label, text), vec in zip(items, vectors):
        placed = False
        for c in clusters:
            sim = max(_cosine(vec, v) for v in c["vectors"])
            if sim >= threshold:
                c["items"].append(text)
                c["members"].append(label)
                c["vectors"].append(vec)
                placed = True
                break
        if not placed:
            clusters.append({"items": [text], "members": [label], "vectors": [vec]})

    out: list[Cluster] = []
    for c in clusters:
        out.append(
            Cluster(
                label=c["items"][0],
                items=c["items"],
                members=sorted(set(c["members"])),
            )
        )
    # Sort so broadly-agreed-upon clusters come first
    out.sort(key=lambda cl: (-len(set(cl.members)), -len(cl.items)))
    return out


def _gather(responses: list[EvaluationResponse], field_name: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for r in responses:
        for t in getattr(r, field_name, []) or []:
            t = str(t).strip()
            if t:
                out.append((r.model_label, t))
    return out


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def _overall_classification(
    diffs: list[CriterionDiff],
    agreement: str,
    verdict_spread: int,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    classes = [d.classification for d in diffs]
    n_hard = classes.count("hard_split")
    n_soft = classes.count("soft_split")

    if agreement == "unanimous" and n_hard == 0 and n_soft <= 1:
        overall = "consensus"
    elif verdict_spread >= 3 or n_hard >= 2:
        overall = "hard_split"
    elif any(
        _is_outlier_criterion(d) for d in diffs
    ):
        overall = "outlier"
        notes.append("At least one model was a consistent outlier on a criterion.")
    elif agreement == "majority" or n_soft >= 1:
        overall = "soft_split"
    else:
        overall = "consensus"

    if n_hard:
        notes.append(f"{n_hard} criterion(s) show hard disagreement.")
    if verdict_spread >= 2:
        notes.append(f"Verdict spread is {verdict_spread} ordinal steps.")
    return overall, notes


def _is_outlier_criterion(d: CriterionDiff) -> bool:
    """One model sits >=3 points away from the other two on this criterion."""
    if len(d.scores) < 3:
        return False
    vals = list(d.scores.values())
    for i, v in enumerate(vals):
        others = [x for j, x in enumerate(vals) if j != i]
        if all(abs(v - o) >= 3.0 for o in others):
            return True
    return False


def compute_diff(
    responses: list[EvaluationResponse],
    criteria: CriteriaSet,
    embed_fn: Optional[EmbedFn] = None,
) -> DisagreementReport:
    """Build a full disagreement report from evaluator responses."""
    diffs = score_diffs(responses, criteria)
    agreement, by_model, vspread = verdict_agreement(responses)

    embed_fn = embed_fn or default_embed_fn()
    clusters = ThemeClusters(
        risks=cluster_texts(_gather(responses, "key_risks"), embed_fn),
        strengths=cluster_texts(_gather(responses, "key_strengths"), embed_fn),
        assumptions=cluster_texts(_gather(responses, "key_assumptions"), embed_fn),
    )

    overall, notes = _overall_classification(diffs, agreement, vspread)
    return DisagreementReport(
        overall=overall,
        score_diffs=diffs,
        verdict_agreement=agreement,
        verdict_by_model=by_model,
        verdict_spread=vspread,
        clusters=clusters,
        notes=notes,
    )
