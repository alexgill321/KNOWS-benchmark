# Result configurations

## Codex: Astra and Luna

These settings describe the saved results in this directory. The runnable
[example configuration](../benchmark.example.json) is a template; it is not the
historical run record. Machine-readable settings are in
[codex/configuration.json](codex/configuration.json).

| Setting | Recorded value |
| --- | --- |
| Models | `gpt-6-astra`, `gpt-6-luna` |
| Reasoning effort | `max` for both; individual model execution with delegation disabled |
| Model authentication | Signed-in ChatGPT subscription through `codex app-server` |
| Codex CLI | `0.159.0`, experimental dynamic tools protocol |
| Python | `3.12.13` |
| Browser | Chrome, headless, 1440 × 900 viewport |
| Agent tools | Browser UI operations, screenshots, and available accessibility observations |
| Maximum actions | 150 tool calls per episode, including failed calls and observations |
| Model episode deadline | 1,800 seconds (30 minutes) |
| Configured evaluator deadline | 900 seconds (15 minutes); saved-output grading/recovery is recorded separately |
| Evaluator judge | `gemini-3.8-flash`, billed separately through Google credentials |
| Seed | `0`, a run label; app-server exposes no model sampling seed |
| Benchmark coverage | 22 task templates × 5 instances = 110 tasks per model |
| Artifact counts per model | 25 Docs, 45 Sheets, 40 Slides |
| Repetitions | One selected result per task; reviewed retry provenance is retained |
| Main campaign concurrency | Serial initially, then four workers, reduced to two after browser authentication failures |
| Late wedding-task completion | Serial |
| Base benchmark commit | `4faa99d735355bbf08c7a7a56c732ab3acc3c1d1` |
| Published harness commit | `e3f7171f50fcccf15f0a08e4b5f715894081c5e1` |
| Reporting snapshot | October 8, 2026 |

The agent received the task prompt and fresh workspace through browser tools.
Google Workspace APIs were used by the supervisor for provisioning and grading.
Evaluator credentials, reference answers, shell access, connectors, and subagents
were not exposed as agent tools. Browser accessibility extraction was not always
available; screenshot observations and failures are retained in the local run
records. The exported scores do not imply every observation succeeded.

The original manifest planned Luna, Astra, and Sol. Sol was subsequently
cancelled; this publication contains only Luna and Astra. The main campaign
started October 2, 2026, with subsequent recovery and saved-output grading.

Evaluator fixes were applied during the campaign, so the base Git commit alone
does not identify the exact evaluator used for every report. Individual reports
retain `provenance.evaluator_source` when the harness recorded it; earlier reports
use `null`. The published harness commit packages the implementation and fixes,
but does not claim that every task used that exact final source snapshot.

### Selection and provenance

The export follows the final saved reporting snapshot, matched against its
authoritative summary and each original evaluation report's SHA-256. Reports
are selected by recorded evaluation attempt and hash, not by maximizing reward.
Each selected report is also matched to its original model episode's task,
model, workspace identity, effort, and limits before private workspace identity
fields are omitted from the export.

The selected origins are recorded in each file's `summary.selected_origins` and
each task's `annotations.origin`: main campaign, reviewed infrastructure/capacity
retry, saved Luna wedding retry, Astra wedding retry, or a saved partial output.
Thus one published result per task does not mean every task had exactly one
historical model attempt. The original attempts and recovery evidence remain in
local archives, identified by source paths and hashes.

### Metrics

All Codex fractions are in `[0, 1]`. Overall and per-artifact aggregates are
unweighted means across the included tasks.

| Field | Definition used by the saved reporting snapshot |
| --- | --- |
| `reward` / `mean_reward` | Task earned points divided by possible points / mean task reward |
| `success_rate` (SR) | 1 when every checkpoint earns all possible points, otherwise 0; see the reporting override below |
| `asc` (ASC) | Fraction of checkpoints earning all possible points |
| `average_checkpoint_fraction` (ACF) | Mean of checkpoint earned/possible points within each task |
| `step_fraction` (SF) | Fraction of individual rubric step `success` flags that are true |

The evaluator can reweight checkpoint scores after registering individual steps.
Consequently, a checkpoint's point total need not equal the sum of its stored
step scores. Preserve both representations; do not silently substitute step
counts for point-weighted ACF or task reward. These definitions document the
existing Codex reporting workflow. The Comet website describes its published
metrics in terms of steps, so comparisons require attention to this distinction.

### Qualifications

- **Five provisional Luna results:** all five `sheets_45` wedding-planner
  instances are included in the saved reporting snapshot despite unresolved
  source URL verification. Each retains `provisional_source_checks: true`, its
  grading note, and the unresolved source URLs. Their inclusion is not evidence
  that those URLs were verified.
- **One partial Astra result:** `knows.sheets_45_Personal_WeddingPlanner_weddingcolorpallette.3`
  grades the interrupted, saved output and has reward 0. Its recorded model
  episode status remains `running`; `annotations.partial_run: true` identifies
  the saved partial output, not an active process or completed model turn.
- **One reporting override:** Astra's formal-letter instance 2 has recorded
  checkpoint fractions `[1, 0, 1]` and reward 0.8. An existing user-requested
  reporting adjustment counted it as successful. `reported_metrics.success_rate`
  is 1; `metrics_from_recorded_scores.success_rate` remains 0. Astra's reported
  overall SR is 4/110 (3.6364%); its SR from the recorded scores is 3/110
  (2.7273%). The adjustment does not change the stored rubric scores, ASC, ACF,
  SF, or reward.
- **Execution-error flags:** 53 selected Luna reports and 29 selected Astra
  reports have `has_execution_errors: true`. Their completed report status and
  inclusion in the existing reporting snapshot do not erase those flags.
- **Judge and evaluator differences:** this campaign uses `gemini-3.8-flash`
  and evaluator repairs; it is not an exact reproduction of the paper's original
  evaluator settings. No new source verification or regrading was performed for
  this export.

## Comet

| Setting | Available record |
| --- | --- |
| Harness | Perplexity Comet |
| Agent model | Claude Opus 4.7, as identified by the existing project page |
| Aggregate results | Existing Docs, Sheets, Slides, and Overall SR/ACF percentages |
| Detailed records | Three examples, each for instance 1: Recipe OCR, Ski Tour Plan, Illustrated Book Report |
| Reasoning effort, browser version, action limit, timeouts, seed, judge configuration | Not recorded in the repository's published Comet results |

Comet's aggregate percentages are copied from the existing project-page table,
not recomputed from the three examples. Its detailed examples retain their
original field names (`score`, `checkpoints`, `steps`, and `categories`). The
website's gold comparisons and screenshots remain in `docs/comparisons/` and are
referenced from the organized dictionary. Missing configuration or task records
are not inferred from the Codex run.
