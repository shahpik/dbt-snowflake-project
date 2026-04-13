# multi-llm-feedback

A reusable Python CLI that gets structured feedback on an idea (business,
investment, product feature, or anything you define) from Claude, Gemini, and
GPT **in parallel**, measures their disagreement, and synthesizes a final
calibrated recommendation.

## Workflow

```
  raw idea  ──►  framing.py  ──►  criteria.py
                                      │
                                      ▼
                               evaluators.py
                         (Claude ∥ Gemini ∥ GPT)
                                      │
                                      ▼
                                  diff.py
                      (stdev, verdicts, embedding clusters)
                                      │
                                      ▼
                              synthesis.py
                        (blinded A/B/C → final call)
                                      │
                                      ▼
                           output/<report>.md
                           output/raw/<per-model>.json
```

## Install

```bash
pip install -e ".[dev]"
```

Required env vars when running against real providers:

- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `OPENAI_API_KEY`

## Usage

```bash
# From stdin
echo "an on-demand dog walking marketplace" | multi-llm-feedback --domain business_idea

# From a file, with criteria overrides and LLM-assisted auto-suggest
multi-llm-feedback \
  --domain investment \
  --file ideas/thesis.md \
  --overrides overrides/my_rubric.yaml \
  --auto-suggest
```

Outputs land in `output/` (markdown) and `output/raw/` (per-model JSON +
summary bundle for auditability).

## Domains & templates

Bundled templates live in `templates/`:

- `business_idea.yaml`
- `investment.yaml`
- `product_feature.yaml`

Each criterion has: `id`, `name`, `description`, `scale`, `weight`, `required`.
Max 10 criteria per evaluation.

## Overrides file format

```yaml
add:
  - id: regulatory
    name: Regulatory Risk
    description: How significant is regulatory exposure for this business?
    weight: 1.0
update:
  go_to_market:
    weight: 1.5
remove:
  - timing
# or, to fully replace:
# replace:
#   - {id: ..., name: ..., description: ..., weight: 1.0, required: true}
```

Required criteria cannot be removed.

## Tests

```bash
pytest
```

Tests mock all three LLM clients plus the embedding function, so the suite
runs offline. Includes an end-to-end integration test exercising the full
framing → criteria → evaluators → diff → synthesis → markdown pipeline.
