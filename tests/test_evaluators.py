"""Tests for the async parallel evaluator layer."""

import json

import pytest

from multi_llm_feedback.criteria import build_criteria
from multi_llm_feedback.evaluators import (
    Evaluator,
    EvaluatorError,
    Verdict,
    parse_response,
    run_evaluators,
)


def _payload_for(criteria_ids: list[str], verdict: str = "yes") -> str:
    return json.dumps(
        {
            "scores": [
                {"id": cid, "score": 7, "rationale": "ok"} for cid in criteria_ids
            ],
            "confidence": 0.8,
            "verdict": verdict,
            "key_risks": ["risk"],
            "key_strengths": ["strength"],
            "key_assumptions": ["assumption"],
        }
    )


def test_parse_response_happy(template_dir):
    criteria = build_criteria("business_idea", template_dir=template_dir)
    raw = _payload_for(criteria.ids(), verdict="strong_yes")
    resp = parse_response(raw, criteria, "claude", "claude-test")
    assert resp.verdict is Verdict.STRONG_YES
    assert len(resp.scores) == len(criteria.criteria)
    assert 0.0 <= resp.confidence <= 1.0


def test_parse_response_clamps_scores(template_dir):
    criteria = build_criteria("business_idea", template_dir=template_dir)
    bad = {
        "scores": [
            {"id": cid, "score": 99 if i == 0 else -3}
            for i, cid in enumerate(criteria.ids())
        ],
        "confidence": 2.5,
        "verdict": "not_a_verdict",
    }
    resp = parse_response(json.dumps(bad), criteria, "claude", "c")
    assert resp.scores[0].score == 10.0
    assert resp.scores[1].score == 1.0
    assert resp.confidence == 1.0
    assert resp.verdict is Verdict.MAYBE


def test_parse_response_missing_score_raises(template_dir):
    criteria = build_criteria("business_idea", template_dir=template_dir)
    data = {
        "scores": [{"id": criteria.ids()[0], "score": 5}],
        "confidence": 0.5,
        "verdict": "maybe",
    }
    with pytest.raises(ValueError, match="did not score criteria"):
        parse_response(json.dumps(data), criteria, "claude", "c")


@pytest.mark.asyncio
async def test_run_evaluators_parallel(template_dir, sample_brief, fake_evaluators_factory):
    criteria = build_criteria("business_idea", template_dir=template_dir)
    evaluators = fake_evaluators_factory(criteria.ids(), agree=True)
    responses, errors = await run_evaluators(sample_brief, criteria, evaluators)
    assert errors == []
    assert len(responses) == 3
    labels = {r.model_label for r in responses}
    assert labels == {"claude", "gemini", "gpt"}


@pytest.mark.asyncio
async def test_run_evaluators_collects_errors(template_dir, sample_brief):
    criteria = build_criteria("business_idea", template_dir=template_dir)

    async def good(_s, _u):
        return _payload_for(criteria.ids())

    async def bad(_s, _u):
        raise RuntimeError("boom")

    evaluators = [
        Evaluator("claude", "c", good),
        Evaluator("gemini", "g", bad),
    ]
    responses, errors = await run_evaluators(sample_brief, criteria, evaluators)
    assert len(responses) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], EvaluatorError)
    assert "boom" in errors[0].error
