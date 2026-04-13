"""Synthesis: one orchestrator LLM call that produces a final recommendation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from .criteria import CriteriaSet
from .diff import DisagreementReport
from .evaluators import EvaluationResponse
from .framing import Brief


SYNTHESIS_SYSTEM_PROMPT = """You are the orchestrator of a multi-evaluator panel.
Three independent models (shown as Model A, Model B, Model C - blinded) have
scored an idea against a shared rubric. You also receive a disagreement report
summarizing where they agreed and where they diverged.

Your job is to produce a calibrated final recommendation. Calibrate carefully:
- If the panel reached consensus, your confidence should generally be higher.
- If the panel split hard, your confidence should drop and you should flag it.
- Prefer conservatism: if disagreement is high, recommend gathering more information.
- Do not blindly average. Weight by rationale quality, not by loudness.

Respond with JSON only, matching this schema:
{
  "final_verdict": "strong_yes" | "yes" | "maybe" | "no" | "strong_no",
  "confidence": <float 0.0-1.0>,
  "headline": "<one-sentence bottom line>",
  "why": ["reason 1", "reason 2", "..."],
  "top_risks": ["..."],
  "top_strengths": ["..."],
  "open_questions": ["..."],
  "recommended_next_steps": ["..."],
  "disagreement_commentary": "<one paragraph explaining panel splits, or why you discounted one model>"
}"""


@dataclass
class SynthesisResult:
    final_verdict: str
    confidence: float
    headline: str
    why: list[str] = field(default_factory=list)
    top_risks: list[str] = field(default_factory=list)
    top_strengths: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    recommended_next_steps: list[str] = field(default_factory=list)
    disagreement_commentary: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Blinding
# ---------------------------------------------------------------------------


BLIND_LABELS = ["Model A", "Model B", "Model C", "Model D", "Model E"]


def blind_responses(
    responses: list[EvaluationResponse],
) -> tuple[list[tuple[str, EvaluationResponse]], dict[str, str]]:
    """Assign blinded labels deterministically by sorting on model_label."""
    ordered = sorted(responses, key=lambda r: r.model_label)
    blinded = []
    mapping: dict[str, str] = {}
    for i, r in enumerate(ordered):
        label = BLIND_LABELS[i] if i < len(BLIND_LABELS) else f"Model {i + 1}"
        blinded.append((label, r))
        mapping[label] = r.model_label
    return blinded, mapping


def _format_response_block(label: str, r: EvaluationResponse) -> str:
    lines = [
        f"=== {label} ===",
        f"verdict: {r.verdict.value}",
        f"confidence: {r.confidence:.2f}",
        "scores:",
    ]
    for s in r.scores:
        rationale = f" - {s.rationale}" if s.rationale else ""
        lines.append(f"  - {s.id}: {s.score}{rationale}")
    lines.append("key_risks:")
    lines.extend(f"  - {x}" for x in r.key_risks)
    lines.append("key_strengths:")
    lines.extend(f"  - {x}" for x in r.key_strengths)
    lines.append("key_assumptions:")
    lines.extend(f"  - {x}" for x in r.key_assumptions)
    return "\n".join(lines)


def _format_disagreement(report: DisagreementReport, mapping: dict[str, str]) -> str:
    # Re-label disagreement report entries using blinded labels.
    reverse = {v: k for k, v in mapping.items()}
    lines = [
        f"Overall: {report.overall}",
        f"Verdict agreement: {report.verdict_agreement} (spread={report.verdict_spread})",
        "Verdicts by model:",
    ]
    for model_label, verdict in report.verdict_by_model.items():
        lines.append(f"  - {reverse.get(model_label, model_label)}: {verdict}")

    lines.append("Per-criterion spread:")
    for d in report.score_diffs:
        blinded_scores = {
            reverse.get(k, k): v for k, v in d.scores.items()
        }
        lines.append(
            f"  - {d.id}: mean={d.mean:.1f} stdev={d.stdev:.2f} "
            f"spread={d.spread:.1f} [{d.classification}] scores={blinded_scores}"
        )

    def _theme(name: str, clusters):
        lines.append(f"{name} clusters:")
        if not clusters:
            lines.append("  (none)")
            return
        for c in clusters:
            members = ", ".join(reverse.get(m, m) for m in c.members)
            lines.append(f"  - ({members}) {c.label}")
            for extra in c.items[1:]:
                lines.append(f"      also: {extra}")

    _theme("Risk", report.clusters.risks)
    _theme("Strength", report.clusters.strengths)
    _theme("Assumption", report.clusters.assumptions)

    if report.notes:
        lines.append("Notes:")
        lines.extend(f"  - {n}" for n in report.notes)
    return "\n".join(lines)


def build_synthesis_prompt(
    brief: Brief,
    criteria: CriteriaSet,
    responses: list[EvaluationResponse],
    report: DisagreementReport,
) -> tuple[str, dict[str, str]]:
    blinded, mapping = blind_responses(responses)
    parts = [
        "IDEA BRIEF:",
        brief.to_prompt_block(),
        "",
        criteria.to_prompt_block(),
        "",
        "EVALUATOR RESPONSES (blinded):",
    ]
    for label, r in blinded:
        parts.append(_format_response_block(label, r))
        parts.append("")
    parts.append("DISAGREEMENT REPORT:")
    parts.append(_format_disagreement(report, mapping))
    parts.append("")
    parts.append("Produce the synthesis JSON now.")
    return "\n".join(parts), mapping


# ---------------------------------------------------------------------------
# LLM injection + default
# ---------------------------------------------------------------------------


SynthesisLLM = Callable[[str, str], str]
"""Signature: (system_prompt, user_prompt) -> raw text response."""


def default_synthesis_llm(system_prompt: str, user_prompt: str) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts = []
    for block in msg.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in synthesis response: {text!r}")
    return json.loads(text[start : end + 1])


VALID_VERDICTS = {"strong_yes", "yes", "maybe", "no", "strong_no"}


def synthesize(
    brief: Brief,
    criteria: CriteriaSet,
    responses: list[EvaluationResponse],
    report: DisagreementReport,
    llm: Optional[SynthesisLLM] = None,
) -> SynthesisResult:
    if not responses:
        raise ValueError("Cannot synthesize from empty responses list")

    llm = llm or default_synthesis_llm
    user_prompt, _mapping = build_synthesis_prompt(brief, criteria, responses, report)
    raw = llm(SYNTHESIS_SYSTEM_PROMPT, user_prompt)
    data = _extract_json(raw)

    final_verdict = str(data.get("final_verdict", "maybe")).lower().strip()
    if final_verdict not in VALID_VERDICTS:
        final_verdict = "maybe"

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    # When the panel hard-split, cap the synthesizer's confidence.
    if report.overall == "hard_split":
        confidence = min(confidence, 0.6)

    return SynthesisResult(
        final_verdict=final_verdict,
        confidence=confidence,
        headline=str(data.get("headline", "")).strip(),
        why=[str(x) for x in (data.get("why") or [])],
        top_risks=[str(x) for x in (data.get("top_risks") or [])],
        top_strengths=[str(x) for x in (data.get("top_strengths") or [])],
        open_questions=[str(x) for x in (data.get("open_questions") or [])],
        recommended_next_steps=[
            str(x) for x in (data.get("recommended_next_steps") or [])
        ],
        disagreement_commentary=str(data.get("disagreement_commentary", "")).strip(),
        raw=data,
    )
