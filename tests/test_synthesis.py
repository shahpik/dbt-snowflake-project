"""Tests for the synthesis / orchestrator layer."""

import json

import pytest

from multi_llm_feedback.criteria import build_criteria
from multi_llm_feedback.diff import compute_diff
from multi_llm_feedback.evaluators import run_evaluators
from multi_llm_feedback.synthesis import (
    blind_responses,
    build_synthesis_prompt,
    synthesize,
)


@pytest.mark.asyncio
async def test_blind_responses_mapping_stable(
    template_dir, sample_brief, fake_evaluators_factory
):
    criteria = build_criteria("business_idea", template_dir=template_dir)
    evs = fake_evaluators_factory(criteria.ids(), agree=True)
    responses, _ = await run_evaluators(sample_brief, criteria, evs)

    blinded, mapping = blind_responses(responses)
    assert [b[0] for b in blinded] == ["Model A", "Model B", "Model C"]
    # Deterministic: claude -> Model A (alphabetical)
    assert mapping["Model A"] == "claude"
    assert mapping["Model B"] == "gemini"
    assert mapping["Model C"] == "gpt"


@pytest.mark.asyncio
async def test_synthesis_prompt_blinds_ids(
    template_dir, sample_brief, fake_evaluators_factory, embed_fn
):
    criteria = build_criteria("business_idea", template_dir=template_dir)
    evs = fake_evaluators_factory(criteria.ids(), agree=True)
    responses, _ = await run_evaluators(sample_brief, criteria, evs)
    report = compute_diff(responses, criteria, embed_fn=embed_fn)

    prompt, mapping = build_synthesis_prompt(sample_brief, criteria, responses, report)
    assert "Model A" in prompt
    assert "claude" not in prompt
    assert "gpt" not in prompt
    assert "gemini" not in prompt


@pytest.mark.asyncio
async def test_synthesize_runs(
    template_dir,
    sample_brief,
    fake_evaluators_factory,
    embed_fn,
    fake_synthesis_llm,
):
    criteria = build_criteria("business_idea", template_dir=template_dir)
    evs = fake_evaluators_factory(criteria.ids(), agree=True)
    responses, _ = await run_evaluators(sample_brief, criteria, evs)
    report = compute_diff(responses, criteria, embed_fn=embed_fn)
    result = synthesize(sample_brief, criteria, responses, report, llm=fake_synthesis_llm)
    assert result.final_verdict == "yes"
    assert 0 <= result.confidence <= 1
    assert result.headline


@pytest.mark.asyncio
async def test_synthesize_caps_confidence_on_hard_split(
    template_dir,
    sample_brief,
    fake_evaluators_factory,
    embed_fn,
):
    criteria = build_criteria("business_idea", template_dir=template_dir)
    evs = fake_evaluators_factory(criteria.ids(), agree=False)
    responses, _ = await run_evaluators(sample_brief, criteria, evs)
    report = compute_diff(responses, criteria, embed_fn=embed_fn)

    # Force hard_split so the confidence cap actually triggers.
    report.overall = "hard_split"

    def _llm(_s, _u):
        return json.dumps(
            {
                "final_verdict": "yes",
                "confidence": 0.95,
                "headline": "h",
                "why": [],
                "top_risks": [],
                "top_strengths": [],
                "open_questions": [],
                "recommended_next_steps": [],
                "disagreement_commentary": "",
            }
        )

    result = synthesize(sample_brief, criteria, responses, report, llm=_llm)
    assert result.confidence <= 0.6
