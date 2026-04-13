"""End-to-end integration test with all three LLM clients mocked out."""

import json

import pytest

from multi_llm_feedback.cli import Workflow, write_outputs


@pytest.mark.asyncio
async def test_end_to_end_workflow(
    tmp_path,
    template_dir,
    fake_framing_llm,
    fake_evaluators_factory,
    embed_fn,
    fake_synthesis_llm,
):
    # We don't know the criteria IDs until after the template is loaded.
    # Build the evaluators using the default business_idea criteria IDs.
    from multi_llm_feedback.criteria import load_template

    criteria_preview = load_template("business_idea", template_dir=template_dir)
    evaluators = fake_evaluators_factory(criteria_preview.ids(), agree=True)

    workflow = Workflow(
        framing_llm=fake_framing_llm,
        evaluators=evaluators,
        embed_fn=embed_fn,
        synthesis_llm=fake_synthesis_llm,
        template_dir=template_dir,
    )

    result = await workflow.run(
        raw_idea="An on-demand dog walking marketplace.",
        domain="business_idea",
    )

    assert result["brief"].title
    assert len(result["responses"]) == 3
    assert result["errors"] == []
    assert result["synthesis"].final_verdict in {"yes", "strong_yes", "maybe"}
    assert result["report"].overall in {"consensus", "soft_split", "hard_split", "outlier"}

    output_dir = tmp_path / "output"
    report_path, raw_dir = write_outputs(result, output_dir)

    assert report_path.exists()
    md = report_path.read_text()
    assert "Multi-LLM Feedback" in md
    assert "## Bottom Line" in md
    assert "## Scores by Criterion" in md

    # Raw per-model files should have been written.
    raw_files = list(raw_dir.glob("*.json"))
    labels = {f.stem.split(".")[-1] for f in raw_files if ".summary" not in f.name}
    assert {"claude", "gemini", "gpt"}.issubset(labels)

    # And a summary bundle with the disagreement report.
    summary_files = list(raw_dir.glob("*.summary.json"))
    assert len(summary_files) == 1
    bundle = json.loads(summary_files[0].read_text())
    assert "brief" in bundle
    assert "report" in bundle
    assert "synthesis" in bundle


@pytest.mark.asyncio
async def test_workflow_with_auto_suggest(
    template_dir,
    fake_framing_llm,
    fake_evaluators_factory,
    embed_fn,
    fake_synthesis_llm,
):
    def fake_auto_suggest(_sys, _user):
        return json.dumps(
            {
                "suggestions": [
                    {
                        "id": "capital_efficiency",
                        "name": "Capital Efficiency",
                        "description": "How capital-efficient is the path to first revenue and profitability?",
                        "weight": 1.0,
                    }
                ]
            }
        )

    from multi_llm_feedback.criteria import load_template

    base = load_template("business_idea", template_dir=template_dir)
    # auto-suggest will add one criterion, so fake evaluators must score it too.
    expected_ids = base.ids() + ["capital_efficiency"]
    evaluators = fake_evaluators_factory(expected_ids, agree=True)

    workflow = Workflow(
        framing_llm=fake_framing_llm,
        evaluators=evaluators,
        embed_fn=embed_fn,
        synthesis_llm=fake_synthesis_llm,
        auto_suggest_llm=fake_auto_suggest,
        template_dir=template_dir,
    )

    result = await workflow.run(
        raw_idea="An on-demand dog walking marketplace.",
        domain="business_idea",
    )

    assert "capital_efficiency" in result["criteria"].ids()
    assert result["criteria"].locked is True
