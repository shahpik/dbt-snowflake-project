"""Tests for criteria loading, merging, validation, auto-suggest."""

import json
from pathlib import Path

import pytest

from multi_llm_feedback.criteria import (
    CriteriaError,
    CriteriaSet,
    Criterion,
    MAX_CRITERIA,
    build_criteria,
    load_template,
    merge_overrides,
    validate,
)


def test_load_business_template(template_dir):
    cs = load_template("business_idea", template_dir=template_dir)
    assert cs.domain == "business_idea"
    assert 1 <= len(cs.criteria) <= MAX_CRITERIA
    assert "problem_severity" in cs.ids()


def test_load_unknown_domain_raises(template_dir):
    with pytest.raises(CriteriaError, match="No template found"):
        load_template("nonexistent_domain", template_dir=template_dir)


def test_merge_add_update_remove(template_dir):
    base = load_template("business_idea", template_dir=template_dir)
    overrides = {
        "add": [
            {
                "id": "founder_focus",
                "name": "Founder Focus",
                "description": "Is the team fully committed and not juggling multiple projects?",
                "weight": 0.9,
            }
        ],
        "update": {
            "go_to_market": {
                "description": "Is there a credible, affordable path to the first 1000 customers in 90 days?",
                "weight": 1.5,
            }
        },
        "remove": ["timing"],
    }
    merged = merge_overrides(base, overrides)
    assert "founder_focus" in merged.ids()
    assert "timing" not in merged.ids()
    gtm = next(c for c in merged.criteria if c.id == "go_to_market")
    assert gtm.weight == 1.5
    assert "1000 customers" in gtm.description


def test_cannot_remove_required(template_dir):
    base = load_template("business_idea", template_dir=template_dir)
    with pytest.raises(CriteriaError, match="Cannot remove required"):
        merge_overrides(base, {"remove": ["problem_severity"]})


def test_replace_overrides_everything(template_dir):
    base = load_template("business_idea", template_dir=template_dir)
    merged = merge_overrides(
        base,
        {
            "replace": [
                {
                    "id": "only_one",
                    "name": "Only Criterion",
                    "description": "This is the only criterion in the replacement list.",
                    "weight": 1.0,
                    "required": True,
                }
            ]
        },
    )
    assert len(merged.criteria) == 1
    assert merged.ids() == ["only_one"]


def test_validate_enforces_max(template_dir):
    base = load_template("business_idea", template_dir=template_dir)
    extras = [
        Criterion(
            id=f"extra_{i}",
            name=f"Extra {i}",
            description="A sufficiently long description of this extra criterion.",
            weight=1.0,
        )
        for i in range(20)
    ]
    cs = CriteriaSet(domain=base.domain, criteria=base.criteria + extras)
    with pytest.raises(CriteriaError, match="Too many criteria"):
        validate(cs)


def test_validate_rejects_short_description():
    cs = CriteriaSet(
        domain="x",
        criteria=[
            Criterion(id="a", name="A", description="short", weight=1.0),
        ],
    )
    with pytest.raises(CriteriaError, match="no clear description"):
        validate(cs)


def test_build_criteria_locks_result(template_dir):
    cs = build_criteria("business_idea", template_dir=template_dir)
    assert cs.locked is True
    assert len(cs.criteria) <= MAX_CRITERIA


def test_build_criteria_with_overrides_file(template_dir, tmp_path):
    overrides_path = tmp_path / "overrides.yaml"
    overrides_path.write_text(
        "remove:\n  - timing\nadd:\n  - id: regulatory\n"
        "    name: Regulatory Risk\n"
        "    description: How significant is regulatory exposure for this business?\n"
        "    weight: 1.0\n"
    )
    cs = build_criteria("business_idea", overrides_path=overrides_path, template_dir=template_dir)
    assert "timing" not in cs.ids()
    assert "regulatory" in cs.ids()


def test_auto_suggest_adds_new_criterion(template_dir):
    def fake_llm(_system: str, _user: str) -> str:
        return json.dumps(
            {
                "suggestions": [
                    {
                        "id": "data_moat",
                        "name": "Data Moat",
                        "description": "Does the business accumulate proprietary data that strengthens over time?",
                        "weight": 1.1,
                    }
                ]
            }
        )

    cs = build_criteria(
        "business_idea",
        brief_block="dummy brief",
        template_dir=template_dir,
        auto_suggest_llm=fake_llm,
    )
    assert "data_moat" in cs.ids()
    assert len(cs.criteria) <= MAX_CRITERIA
