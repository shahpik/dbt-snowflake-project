"""Entry point for the multi-LLM feedback CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .criteria import CriteriaSet, build_criteria
from .diff import DisagreementReport, compute_diff
from .evaluators import (
    EvaluationResponse,
    Evaluator,
    EvaluatorError,
    default_evaluators,
    run_evaluators,
)
from .framing import Brief, frame_idea
from .synthesis import SynthesisResult, synthesize


# ---------------------------------------------------------------------------
# Workflow object (nice for tests)
# ---------------------------------------------------------------------------


class Workflow:
    """A runnable workflow with injectable LLM clients (for testing)."""

    def __init__(
        self,
        *,
        framing_llm=None,
        evaluators: Optional[list[Evaluator]] = None,
        embed_fn=None,
        synthesis_llm=None,
        auto_suggest_llm=None,
        template_dir: Optional[Path] = None,
    ):
        self.framing_llm = framing_llm
        self.evaluators = evaluators
        self.embed_fn = embed_fn
        self.synthesis_llm = synthesis_llm
        self.auto_suggest_llm = auto_suggest_llm
        self.template_dir = template_dir

    async def run(
        self,
        raw_idea: str,
        domain: str,
        overrides_path: Optional[Path] = None,
    ) -> dict:
        brief = frame_idea(raw_idea, domain, llm=self.framing_llm)
        criteria = build_criteria(
            domain,
            brief_block=brief.to_prompt_block(),
            overrides_path=overrides_path,
            template_dir=self.template_dir,
            auto_suggest_llm=self.auto_suggest_llm,
        )
        responses, errors = await run_evaluators(
            brief, criteria, evaluators=self.evaluators
        )
        if not responses:
            raise RuntimeError(
                f"All evaluators failed. Errors: {[asdict(e) for e in errors]}"
            )
        report = compute_diff(responses, criteria, embed_fn=self.embed_fn)
        synthesis = synthesize(
            brief, criteria, responses, report, llm=self.synthesis_llm
        )
        return {
            "brief": brief,
            "criteria": criteria,
            "responses": responses,
            "errors": errors,
            "report": report,
            "synthesis": synthesis,
        }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return text.strip("-")[:60] or "idea"


def render_markdown(
    brief: Brief,
    criteria: CriteriaSet,
    responses: list[EvaluationResponse],
    errors: list[EvaluatorError],
    report: DisagreementReport,
    synthesis: SynthesisResult,
) -> str:
    lines: list[str] = []
    lines.append(f"# Multi-LLM Feedback: {brief.title}")
    lines.append("")
    lines.append(f"_Domain_: **{brief.domain}**  ")
    lines.append(
        f"_Generated_: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append(
        f"**Verdict:** `{synthesis.final_verdict}` &nbsp; "
        f"**Confidence:** `{synthesis.confidence:.2f}` &nbsp; "
        f"**Panel agreement:** `{report.overall}`"
    )
    lines.append("")
    if synthesis.headline:
        lines.append(f"> {synthesis.headline}")
        lines.append("")

    lines.append("## Brief")
    lines.append("")
    lines.append(f"- **Problem:** {brief.problem}")
    lines.append(f"- **Proposed Solution:** {brief.proposed_solution}")
    lines.append(f"- **Scope:** {brief.scope}")
    if brief.assumptions:
        lines.append("- **Assumptions:**")
        lines.extend(f"  - {a}" for a in brief.assumptions)
    if brief.open_questions:
        lines.append("- **Open Questions:**")
        lines.extend(f"  - {q}" for q in brief.open_questions)
    lines.append("")

    lines.append("## Scores by Criterion")
    lines.append("")
    header = ["Criterion"] + [r.model_label for r in responses] + ["Mean", "Stdev", "Class"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    diff_by_id = {d.id: d for d in report.score_diffs}
    for c in criteria.criteria:
        row = [c.name]
        for r in responses:
            score = r.score_map().get(c.id)
            row.append(f"{score:.1f}" if score is not None else "—")
        d = diff_by_id.get(c.id)
        if d:
            row.extend([f"{d.mean:.1f}", f"{d.stdev:.2f}", d.classification])
        else:
            row.extend(["—", "—", "—"])
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Verdicts")
    lines.append("")
    for label, v in report.verdict_by_model.items():
        lines.append(f"- **{label}**: `{v}`")
    lines.append(
        f"- _Agreement_: {report.verdict_agreement} (spread {report.verdict_spread})"
    )
    lines.append("")

    def _theme_section(title: str, clusters):
        lines.append(f"### {title}")
        if not clusters:
            lines.append("_(none)_")
            lines.append("")
            return
        for c in clusters:
            members = ", ".join(c.members)
            lines.append(f"- **[{members}]** {c.label}")
            for extra in c.items[1:]:
                lines.append(f"  - also: {extra}")
        lines.append("")

    lines.append("## Themes (embedding-clustered)")
    lines.append("")
    _theme_section("Risks", report.clusters.risks)
    _theme_section("Strengths", report.clusters.strengths)
    _theme_section("Assumptions", report.clusters.assumptions)

    lines.append("## Synthesis")
    lines.append("")
    if synthesis.why:
        lines.append("**Why:**")
        lines.extend(f"- {x}" for x in synthesis.why)
        lines.append("")
    if synthesis.top_strengths:
        lines.append("**Top Strengths:**")
        lines.extend(f"- {x}" for x in synthesis.top_strengths)
        lines.append("")
    if synthesis.top_risks:
        lines.append("**Top Risks:**")
        lines.extend(f"- {x}" for x in synthesis.top_risks)
        lines.append("")
    if synthesis.open_questions:
        lines.append("**Open Questions:**")
        lines.extend(f"- {x}" for x in synthesis.open_questions)
        lines.append("")
    if synthesis.recommended_next_steps:
        lines.append("**Recommended Next Steps:**")
        lines.extend(f"- {x}" for x in synthesis.recommended_next_steps)
        lines.append("")
    if synthesis.disagreement_commentary:
        lines.append("**On panel disagreement:**")
        lines.append("")
        lines.append(f"> {synthesis.disagreement_commentary}")
        lines.append("")

    if errors:
        lines.append("## Evaluator Errors")
        for e in errors:
            lines.append(f"- {e.label} ({e.model_id}): {e.error}")
        lines.append("")

    return "\n".join(lines)


def write_outputs(
    result: dict,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write the markdown report and raw per-model JSONs. Returns (report_path, raw_dir)."""
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    brief: Brief = result["brief"]
    criteria: CriteriaSet = result["criteria"]
    responses: list[EvaluationResponse] = result["responses"]
    errors: list[EvaluatorError] = result["errors"]
    report: DisagreementReport = result["report"]
    synthesis: SynthesisResult = result["synthesis"]

    slug = _slugify(brief.title or brief.domain)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}-{slug}"

    md = render_markdown(brief, criteria, responses, errors, report, synthesis)
    report_path = output_dir / f"{base}.md"
    report_path.write_text(md, encoding="utf-8")

    bundle = {
        "brief": brief.to_dict(),
        "criteria": criteria.to_dict(),
        "report": report.to_dict(),
        "synthesis": synthesis.to_dict(),
        "errors": [asdict(e) for e in errors],
    }
    (raw_dir / f"{base}.summary.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8"
    )
    for r in responses:
        (raw_dir / f"{base}.{r.model_label}.json").write_text(
            json.dumps(r.to_dict(), indent=2), encoding="utf-8"
        )
    return report_path, raw_dir


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _read_idea(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.idea:
        return args.idea
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("No idea provided. Pass --file, --idea, or pipe via stdin.")
    return data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="multi-llm-feedback",
        description="Get structured feedback on an idea from Claude, Gemini, and GPT in parallel.",
    )
    p.add_argument("--domain", required=True, help="e.g. business_idea, investment, product_feature")
    p.add_argument("--file", help="Path to a file containing the raw idea.")
    p.add_argument("--idea", help="Inline raw idea text.")
    p.add_argument("--overrides", help="Optional YAML file of criteria overrides.")
    p.add_argument(
        "--output-dir",
        default="output",
        help="Directory for markdown reports and raw/ subfolder.",
    )
    p.add_argument(
        "--auto-suggest",
        action="store_true",
        help="Run an auto-suggest LLM pass on the criteria list.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    raw_idea = _read_idea(args)

    workflow = Workflow()
    if args.auto_suggest:
        from .framing import default_framing_llm

        def _auto_suggest(system: str, user: str) -> str:
            return default_framing_llm(system, user)

        workflow.auto_suggest_llm = _auto_suggest

    result = asyncio.run(
        workflow.run(
            raw_idea=raw_idea,
            domain=args.domain,
            overrides_path=Path(args.overrides) if args.overrides else None,
        )
    )
    report_path, raw_dir = write_outputs(result, Path(args.output_dir))
    print(f"Report: {report_path}")
    print(f"Raw:    {raw_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
