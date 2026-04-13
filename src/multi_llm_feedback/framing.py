"""Framing: transforms a raw idea into a structured brief via one LLM call."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


FRAMING_SYSTEM_PROMPT = """You are an analyst who converts raw ideas into structured briefs.
Given a raw idea and a domain (e.g. business_idea, investment, product_feature),
produce a concise, neutral brief that an evaluator can reason over.

Respond with JSON only, matching this schema exactly:
{
  "title": "short title for the idea (<= 80 chars)",
  "problem": "the core problem or opportunity the idea addresses",
  "proposed_solution": "what is being proposed",
  "scope": "what is in-scope and out-of-scope",
  "assumptions": ["assumption 1", "assumption 2", "..."],
  "open_questions": ["question 1", "question 2", "..."]
}

Keep each field grounded in what is present in the raw idea. Do not invent
facts. If something is unclear, add it to open_questions."""


@dataclass
class Brief:
    """Structured brief used downstream by criteria + evaluators."""

    title: str
    domain: str
    problem: str
    proposed_solution: str
    scope: str
    assumptions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    raw_idea: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt_block(self) -> str:
        lines = [
            f"TITLE: {self.title}",
            f"DOMAIN: {self.domain}",
            f"PROBLEM: {self.problem}",
            f"PROPOSED SOLUTION: {self.proposed_solution}",
            f"SCOPE: {self.scope}",
            "ASSUMPTIONS:",
        ]
        lines.extend(f"  - {a}" for a in self.assumptions)
        lines.append("OPEN QUESTIONS:")
        lines.extend(f"  - {q}" for q in self.open_questions)
        return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Could not find JSON object in framing response: {text!r}")
    return json.loads(text[start : end + 1])


FramingLLM = Callable[[str, str], str]
"""Signature: (system_prompt, user_prompt) -> raw text response."""


def default_framing_llm(system_prompt: str, user_prompt: str) -> str:
    """Default framing LLM implementation using Anthropic Claude."""
    # Imported lazily so tests/mocks don't require a live SDK.
    from anthropic import Anthropic

    client = Anthropic()
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # Anthropic SDK returns content as a list of blocks
    parts = []
    for block in msg.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def frame_idea(
    raw_idea: str,
    domain: str,
    llm: Optional[FramingLLM] = None,
) -> Brief:
    """Produce a structured Brief from a raw idea and domain.

    The LLM is injected so it can be mocked in tests.
    """
    if not raw_idea or not raw_idea.strip():
        raise ValueError("raw_idea must be a non-empty string")
    if not domain or not domain.strip():
        raise ValueError("domain must be a non-empty string")

    llm = llm or default_framing_llm
    user_prompt = (
        f"DOMAIN: {domain}\n\nRAW IDEA:\n{raw_idea.strip()}\n\n"
        "Return the JSON brief now."
    )
    response = llm(FRAMING_SYSTEM_PROMPT, user_prompt)
    data = _extract_json(response)

    required = ["title", "problem", "proposed_solution", "scope"]
    missing = [k for k in required if not data.get(k)]
    if missing:
        raise ValueError(f"Framing response missing required fields: {missing}")

    return Brief(
        title=str(data["title"])[:160],
        domain=domain,
        problem=str(data["problem"]),
        proposed_solution=str(data["proposed_solution"]),
        scope=str(data["scope"]),
        assumptions=[str(a) for a in data.get("assumptions", [])],
        open_questions=[str(q) for q in data.get("open_questions", [])],
        raw_idea=raw_idea.strip(),
    )
