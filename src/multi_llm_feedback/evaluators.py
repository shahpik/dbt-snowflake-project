"""Async parallel evaluators for Claude, Gemini, and GPT."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from .criteria import CriteriaSet
from .framing import Brief


class Verdict(str, Enum):
    STRONG_YES = "strong_yes"
    YES = "yes"
    MAYBE = "maybe"
    NO = "no"
    STRONG_NO = "strong_no"


VALID_VERDICTS = {v.value for v in Verdict}


@dataclass
class CriterionScore:
    id: str
    score: float
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResponse:
    """Normalized response from a single evaluator."""

    model_label: str  # e.g. "claude", "gemini", "gpt" (not blinded here)
    model_id: str
    scores: list[CriterionScore]
    confidence: float  # 0.0 - 1.0
    verdict: Verdict
    key_risks: list[str] = field(default_factory=list)
    key_strengths: list[str] = field(default_factory=list)
    key_assumptions: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d

    def score_map(self) -> dict[str, float]:
        return {s.id: s.score for s in self.scores}


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


EVALUATOR_SYSTEM_PROMPT = """You are a rigorous, independent evaluator.
Score the given idea strictly against the provided criteria. Be calibrated:
reserve high scores for genuinely strong attributes, and flag weaknesses openly.
Think step-by-step internally, then respond with JSON only.

Schema:
{
  "scores": [{"id": "<criterion_id>", "score": <1-10 number>, "rationale": "<one sentence>"}],
  "confidence": <float 0.0-1.0>,
  "verdict": "strong_yes" | "yes" | "maybe" | "no" | "strong_no",
  "key_risks": ["..."],
  "key_strengths": ["..."],
  "key_assumptions": ["..."]
}

You MUST provide a score for every criterion id listed. Use the full 1-10 range."""


def build_user_prompt(brief: Brief, criteria: CriteriaSet) -> str:
    return (
        "IDEA BRIEF:\n"
        f"{brief.to_prompt_block()}\n\n"
        f"{criteria.to_prompt_block()}\n\n"
        "Return the JSON response now."
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


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
        raise ValueError(f"No JSON object found in evaluator response: {text!r}")
    return json.loads(text[start : end + 1])


def parse_response(
    text: str,
    criteria: CriteriaSet,
    model_label: str,
    model_id: str,
) -> EvaluationResponse:
    data = _extract_json(text)

    expected_ids = criteria.ids()
    raw_scores = data.get("scores") or []
    if not isinstance(raw_scores, list):
        raise ValueError("'scores' must be a list")

    score_map: dict[str, CriterionScore] = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id", "")).strip()
        if cid not in expected_ids:
            continue
        try:
            val = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        val = max(1.0, min(10.0, val))
        score_map[cid] = CriterionScore(
            id=cid,
            score=val,
            rationale=str(item.get("rationale", "")).strip(),
        )

    missing = [cid for cid in expected_ids if cid not in score_map]
    if missing:
        raise ValueError(
            f"Evaluator {model_label} did not score criteria: {missing}"
        )

    ordered = [score_map[cid] for cid in expected_ids]

    verdict_str = str(data.get("verdict", "maybe")).lower().strip()
    if verdict_str not in VALID_VERDICTS:
        verdict_str = "maybe"

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return EvaluationResponse(
        model_label=model_label,
        model_id=model_id,
        scores=ordered,
        confidence=confidence,
        verdict=Verdict(verdict_str),
        key_risks=[str(x) for x in (data.get("key_risks") or [])],
        key_strengths=[str(x) for x in (data.get("key_strengths") or [])],
        key_assumptions=[str(x) for x in (data.get("key_assumptions") or [])],
        raw=data,
    )


# ---------------------------------------------------------------------------
# Evaluator protocol
# ---------------------------------------------------------------------------


EvaluatorFn = Callable[[str, str], Awaitable[str]]
"""Async callable: (system_prompt, user_prompt) -> raw text response."""


@dataclass
class Evaluator:
    label: str          # canonical label, e.g. "claude"
    model_id: str       # provider-specific model string
    call: EvaluatorFn   # async function producing raw text


# ---------------------------------------------------------------------------
# Default provider adapters (lazy-loaded SDKs)
# ---------------------------------------------------------------------------


def _claude_call_factory(model_id: str) -> EvaluatorFn:
    async def _call(system_prompt: str, user_prompt: str) -> str:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic()
        msg = await client.messages.create(
            model=model_id,
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

    return _call


def _gemini_call_factory(model_id: str) -> EvaluatorFn:
    async def _call(system_prompt: str, user_prompt: str) -> str:
        import google.generativeai as genai

        model = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=system_prompt,
        )
        # SDK is sync; run in a thread to avoid blocking the loop.
        def _sync() -> str:
            resp = model.generate_content(user_prompt)
            return getattr(resp, "text", "") or ""

        return await asyncio.to_thread(_sync)

    return _call


def _gpt_call_factory(model_id: str) -> EvaluatorFn:
    async def _call(system_prompt: str, user_prompt: str) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    return _call


def default_evaluators() -> list[Evaluator]:
    """Return the default 3-model evaluator panel."""
    return [
        Evaluator("claude", "claude-opus-4-6", _claude_call_factory("claude-opus-4-6")),
        Evaluator("gemini", "gemini-1.5-pro", _gemini_call_factory("gemini-1.5-pro")),
        Evaluator("gpt", "gpt-4o", _gpt_call_factory("gpt-4o")),
    ]


# ---------------------------------------------------------------------------
# Parallel orchestration
# ---------------------------------------------------------------------------


@dataclass
class EvaluatorError:
    label: str
    model_id: str
    error: str


async def _run_one(
    ev: Evaluator,
    system_prompt: str,
    user_prompt: str,
    criteria: CriteriaSet,
) -> EvaluationResponse | EvaluatorError:
    try:
        raw = await ev.call(system_prompt, user_prompt)
        return parse_response(raw, criteria, ev.label, ev.model_id)
    except Exception as exc:  # noqa: BLE001 - surface any failure
        return EvaluatorError(label=ev.label, model_id=ev.model_id, error=repr(exc))


async def run_evaluators(
    brief: Brief,
    criteria: CriteriaSet,
    evaluators: Optional[list[Evaluator]] = None,
) -> tuple[list[EvaluationResponse], list[EvaluatorError]]:
    """Dispatch all evaluators in parallel and collect results."""
    evaluators = evaluators or default_evaluators()
    user_prompt = build_user_prompt(brief, criteria)

    results = await asyncio.gather(
        *(_run_one(ev, EVALUATOR_SYSTEM_PROMPT, user_prompt, criteria) for ev in evaluators)
    )
    successes: list[EvaluationResponse] = []
    errors: list[EvaluatorError] = []
    for r in results:
        if isinstance(r, EvaluatorError):
            errors.append(r)
        else:
            successes.append(r)
    return successes, errors
