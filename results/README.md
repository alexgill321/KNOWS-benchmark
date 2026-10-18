# Benchmark results

Machine-readable result dictionaries are grouped by harness:

| File | Coverage |
| --- | --- |
| [codex/gpt-6-astra.json](codex/gpt-6-astra.json) | 110 Astra task results, with checkpoints and individual rubric steps |
| [codex/gpt-6-luna.json](codex/gpt-6-luna.json) | 110 Luna task results, with checkpoints and individual rubric steps |
| [codex/configuration.json](codex/configuration.json) | Recorded Codex campaign settings and source hashes |
| [comet/results.json](comet/results.json) | Existing published Comet aggregate scores and three detailed task examples |
| [CONFIGURATIONS.md](CONFIGURATIONS.md) | Configuration, selection policy, metric definitions, and qualifications |

## Codex snapshot

The dictionaries preserve the **October 8, 2026** reporting snapshot, with one
selected evaluation per task and model. They include saved-output regrades and
reviewed replacement attempts. This is an export of existing results; publishing
these files did not run models or evaluators.

Each file has a `results` dictionary keyed by the full KNOWS task ID:

```python
import json
from pathlib import Path

astra = json.loads(Path("results/codex/gpt-6-astra.json").read_text())
task = astra["results"]["knows.docs_1_formal_letter.2"]
print(task["result"]["final_score"])
print(task["result"]["checkpoints"][0]["step_summary"]["steps"])
```

`result` retains the evaluator's checkpoint and step dictionaries, including
scores, success flags, explanations, categories, and execution times.
`evaluation` retains judge identities and execution-error flags. `run` records
the selected model episode's limits, status, usage, and start time.
`provenance` records source report/episode hashes and evaluator source hashes
where available. Its `codex-results/...` paths identify local archival inputs;
those archives are not part of this publication.

`metrics_from_recorded_scores` contains metrics computed from the saved scores.
`reported_metrics` preserves the existing reporting snapshot, including one
explicit Astra success-rate override. The original checkpoint/step scores remain
unchanged. See [the qualifications](CONFIGURATIONS.md#qualifications) before
using the aggregate numbers.

| Model | Task results | Mean task reward | Provisional source checks | Partial runs | Reports flagging execution errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Astra | 110 | 0.694477 | 0 | 1 | 29 |
| Luna | 110 | 0.439694 | 5 | 0 | 53 |

Mean task reward is the mean of each task's earned/possible points. It is a
different metric from average checkpoint fraction. Counts above describe the
selected reports, including the existing snapshot's inclusions; they do not
assert that every evaluation step completed without errors.

## Comet coverage

[comet/results.json](comet/results.json) organizes the Comet data already
published in this repository: the aggregate table from `docs/index.html`, and
Recipe OCR, Ski Tour Plan, and Illustrated Book Report examples from
`docs/comparisons/`. Example dictionaries retain their original schema and
scores. The source JSON and screenshots remain in `docs/comparisons/` for the
project website; each exported example records its source path and SHA-256.

The repository does **not** contain a complete 110-task Comet result dictionary.
The three examples are not used to reconstruct its aggregate scores. Comet
aggregates are the already published percentages, rounded to one decimal place.

## Validate

From the repository root, with Python 3.10 or later and no extra dependencies:

```bash
python results/validate.py
```

Validation checks task coverage, checkpoint totals, step counts, derived and
reported metrics, annotations, and correspondence with the existing Comet data.
Credentials, browser state, private workspace identifiers, protocol logs, and
raw recovery archives are excluded. Local temporary paths and Google Workspace
artifact URLs in evaluator explanations are replaced with descriptive markers;
scores and step outcomes are unchanged.
