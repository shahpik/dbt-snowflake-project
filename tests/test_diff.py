"""Tests for disagreement analysis."""

import pytest

from multi_llm_feedback.criteria import build_criteria
from multi_llm_feedback.diff import (
    CriterionDiff,
    _is_outlier_criterion,
    cluster_texts,
    compute_diff,
    verdict_agreement,
)
from multi_llm_feedback.evaluators import run_evaluators


@pytest.mark.asyncio
async def test_compute_diff_consensus(
    template_dir, sample_brief, fake_evaluators_factory, embed_fn
):
    criteria = build_criteria("business_idea", template_dir=template_dir)
    evs = fake_evaluators_factory(criteria.ids(), agree=True)
    responses, _ = await run_evaluators(sample_brief, criteria, evs)

    report = compute_diff(responses, criteria, embed_fn=embed_fn)
    assert report.verdict_agreement == "unanimous"
    assert report.overall in {"consensus", "soft_split"}
    # All three raters agree → clusters should merge each shared theme into 1.
    assert all(len(c.members) == 3 for c in report.clusters.risks)


@pytest.mark.asyncio
async def test_compute_diff_hard_split(
    template_dir, sample_brief, fake_evaluators_factory, embed_fn
):
    criteria = build_criteria("business_idea", template_dir=template_dir)
    evs = fake_evaluators_factory(criteria.ids(), agree=False)
    responses, _ = await run_evaluators(sample_brief, criteria, evs)

    report = compute_diff(responses, criteria, embed_fn=embed_fn)
    assert report.verdict_agreement in {"split", "majority"}
    assert report.overall in {"hard_split", "soft_split", "outlier"}
    assert report.verdict_spread >= 2


def test_verdict_agreement_unanimous():
    class R:
        def __init__(self, label, verdict):
            self.model_label = label
            class V:
                value = verdict
            self.verdict = V()

    rs = [R("a", "yes"), R("b", "yes"), R("c", "yes")]
    agreement, mapping, spread = verdict_agreement(rs)
    assert agreement == "unanimous"
    assert spread == 0
    assert mapping == {"a": "yes", "b": "yes", "c": "yes"}


def test_is_outlier_criterion():
    d = CriterionDiff(
        id="x",
        scores={"a": 9, "b": 3, "c": 3},
        mean=5.0,
        stdev=3.46,
        spread=6,
        classification="hard_split",
    )
    assert _is_outlier_criterion(d) is True


def test_cluster_texts_merges_similar(embed_fn):
    items = [
        ("claude", "liability exposure if dog is lost"),
        ("gemini", "liability exposure if dog is lost"),
        ("gpt", "completely unrelated regulatory concern about pricing"),
    ]
    clusters = cluster_texts(items, embed_fn)
    # At least two clusters (the identical ones should group).
    assert any(len(c.items) >= 2 for c in clusters)


def test_cluster_texts_empty(embed_fn):
    assert cluster_texts([], embed_fn) == []
