"""Shared test fixtures: fake LLM clients, fake embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from multi_llm_feedback.criteria import CriteriaSet
from multi_llm_feedback.evaluators import Evaluator
from multi_llm_feedback.framing import Brief


REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"


@pytest.fixture
def template_dir() -> Path:
    return TEMPLATES_DIR


@pytest.fixture
def sample_brief() -> Brief:
    return Brief(
        title="Remote dog-walker marketplace",
        domain="business_idea",
        problem="Urban dog owners struggle to find reliable, vetted dog walkers on short notice.",
        proposed_solution="An on-demand marketplace app matching owners with background-checked walkers.",
        scope="US metro areas, single-dog walks, 30-60 minute sessions.",
        assumptions=["demand is inelastic to small price changes"],
        open_questions=["how do we handle lost-dog liability?"],
        raw_idea="dog walker app",
    )


def _fake_framing_llm(_system: str, _user: str) -> str:
    return json.dumps(
        {
            "title": "Dog Walker Marketplace",
            "problem": "Hard to find reliable walkers.",
            "proposed_solution": "On-demand app with vetted walkers.",
            "scope": "US metros.",
            "assumptions": ["demand elastic"],
            "open_questions": ["liability model?"],
        }
    )


@pytest.fixture
def fake_framing_llm():
    return _fake_framing_llm


def _evaluator_payload(
    base_score: float,
    verdict: str,
    confidence: float,
    *,
    risks: list[str],
    strengths: list[str],
    assumptions: list[str],
    criterion_ids: list[str],
) -> str:
    scores = []
    for i, cid in enumerate(criterion_ids):
        # deterministic per-id offset
        offset = (i % 3) - 1
        scores.append(
            {
                "id": cid,
                "score": max(1, min(10, base_score + offset)),
                "rationale": f"rationale for {cid}",
            }
        )
    return json.dumps(
        {
            "scores": scores,
            "confidence": confidence,
            "verdict": verdict,
            "key_risks": risks,
            "key_strengths": strengths,
            "key_assumptions": assumptions,
        }
    )


def make_fake_evaluators(
    criterion_ids: list[str],
    *,
    agree: bool = True,
) -> list[Evaluator]:
    """Build 3 fake evaluators. If agree=True, their scores/verdicts converge."""
    if agree:
        specs = [
            ("claude", 7, "yes", 0.8),
            ("gemini", 7, "yes", 0.75),
            ("gpt", 8, "yes", 0.82),
        ]
        shared_risks = ["liability exposure if dog is lost"]
        shared_strengths = ["large addressable market of dog owners"]
        shared_assumptions = ["vetted walkers can be supplied at scale"]
    else:
        specs = [
            ("claude", 9, "strong_yes", 0.9),
            ("gemini", 4, "no", 0.7),
            ("gpt", 6, "maybe", 0.6),
        ]
        shared_risks = ["liability exposure if dog is lost"]
        shared_strengths = ["big market"]
        shared_assumptions = ["walkers are easy to recruit"]

    evaluators: list[Evaluator] = []
    for label, base, verdict, conf in specs:
        payload = _evaluator_payload(
            base,
            verdict,
            conf,
            risks=shared_risks,
            strengths=shared_strengths,
            assumptions=shared_assumptions,
            criterion_ids=criterion_ids,
        )

        async def _call(_sys: str, _user: str, _p=payload) -> str:
            return _p

        evaluators.append(
            Evaluator(label=label, model_id=f"fake-{label}", call=_call)
        )
    return evaluators


@pytest.fixture
def fake_evaluators_factory():
    return make_fake_evaluators


def fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """A tiny deterministic embedder: bag-of-words in a fixed vocab space."""
    # Build a simple shared vocab from the incoming batch.
    vocab: dict[str, int] = {}
    tokenized = []
    for t in texts:
        tokens = [w.lower() for w in t.split() if w.isalnum() or w.isalpha()]
        tokenized.append(tokens)
        for w in tokens:
            if w not in vocab:
                vocab[w] = len(vocab)
    dim = max(len(vocab), 1)
    out: list[list[float]] = []
    for tokens in tokenized:
        v = [0.0] * dim
        for w in tokens:
            if w in vocab:
                v[vocab[w]] += 1.0
        # normalize
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out


@pytest.fixture
def embed_fn():
    return fake_embed_fn


def _fake_synthesis_llm(_system: str, _user: str) -> str:
    return json.dumps(
        {
            "final_verdict": "yes",
            "confidence": 0.72,
            "headline": "Promising marketplace with solvable liability risk.",
            "why": ["strong demand signals", "clear GTM"],
            "top_risks": ["liability", "supply acquisition"],
            "top_strengths": ["large TAM"],
            "open_questions": ["insurance model?"],
            "recommended_next_steps": ["pilot in one city"],
            "disagreement_commentary": "Panel broadly agreed.",
        }
    )


@pytest.fixture
def fake_synthesis_llm():
    return _fake_synthesis_llm
