# KNOWS through a ChatGPT/Codex subscription

This harness uses the signed-in `codex app-server`, with explicit model IDs
`gpt-6-luna`, `gpt-6.1-sol`, and `gpt-6-astra`. Model inference uses ChatGPT
subscription usage. KNOWS evaluation separately uses Google Workspace APIs and
its configured Gemini judge. There is no OpenAI API-key fallback.

The benchmark is 110 live Google Workspace tasks. One repetition for each model
is **330 episodes total**. The three smoke episodes are the first three episodes
of this same run, so resuming with the full suite does not add three repetitions.

## Install and configure

Use Python 3.12 on macOS or Linux (the runners use POSIX file locks and process
groups), an installed Chrome browser, and a signed-in Codex CLI supporting
the experimental app-server dynamic tools protocol. The adapter was built against
`codex-cli 0.159.0`; it checks the exact model and effort reported by the server.

```bash
uv venv --python 3.12
uv pip install -r requirements-codex.txt
source .venv/bin/activate
codex login
cp benchmark.example.json benchmark.local.json
```

Edit the local config to point to your Google browser `storage_state`, evaluator
service account, owner OAuth token/client credentials, and judge-key `.env`.
These files and all results are ignored by Git. Existing environment files are
read with a restricted Google/judge key allowlist. The agent process receives
neither those secrets nor evaluator tools.

The saved Google browser state must belong to a dedicated benchmark account.
Workspace files are created in that account and shared with the evaluator via
owner OAuth without sending notification emails. The two families that write
into Drive also need that valid **owner OAuth token** to create
private, fresh per-episode folders and presentation copies. Google OAuth renewal
requires the account owner's consent if the refresh token has expired.

```bash
python -m codex_benchmark.reauth --config benchmark.local.json
```

This saves a separate token in this checkout and leaves other projects' tokens
unchanged. Select the dedicated KNOWS account in the consent screen.

## Run

Run commands from this checkout. Use the same config and run directory to resume.

```bash
python -m codex_benchmark.runner list-tasks
python -m codex_benchmark.runner run --suite smoke --run-dir codex-results/run-001
python -m codex_benchmark.runner run --suite all --run-dir codex-results/run-001
python -m codex_benchmark.runner report codex-results/run-001
```

The full command also runs the smoke episodes first and stops on infrastructure
or smoke-evaluator errors. Already finished episodes are not replayed. An
interrupted episode is preserved and requires inspection; automatic replay could
contaminate the artifact or overwrite evidence. Other tasks get fresh files and
fresh Codex threads. The serial runner remains available. After all three smoke
evaluations, the parallel coordinator can continue the same campaign:

```bash
python -m codex_benchmark.parallel run --run-dir codex-results/run-001 --workers 2
```

Stop and drain any existing serial runner at a saved terminal model checkpoint
before switching. The coordinator holds the account lock, gives each worker a
private browser state and owner OAuth token, and claims tasks transactionally in
`parallel.sqlite`. Tasks run concurrently within a model; all Luna tasks finish
before remaining Astra tasks, then Sol. Completed episodes are retained, and a
saved terminal result needing grading is evaluated without a new model turn.
An interrupted unfinished turn requires inspection instead of automatic replay.
The two-worker example reflects Google cookie-refresh rate limits observed with
four workers during development. Reducing concurrency is a mitigation, not
proof that authentication is stable.

`parallel-status.json` records all worker PIDs, task claims, progress evidence,
and authentication health every ten seconds. Worker logs are under `workers/`.
Actual Google sign-in/challenge navigation interrupts the affected turn and
preserves an infrastructure exclusion; it cannot count as a benchmark zero.
Failures or stalled progress pause new task claims while other active workers
finish. Separate state files prevent concurrent writes, but account-wide Google
session revocation still requires owner sign-in. Monitor saved evidence for all
workers; do not open extra diagnostic browsers during active turns.

Provisioning failures save a sanitized final URL and a screenshot with a
five-second capture deadline. Actual Google login redirects also produce
`auth-health.json`. A failed setup before any model activity can be recovered
under the account lock after all workers drain: validate it with
`setup_recovery.validate_pre_agent_failure`, provision and verify its workspace,
then use `complete_setup_recovery` and `reset_pre_agent_claim`. The helpers
archive the exact original episode with a SHA-256 receipt, reject any existing
model/evaluator evidence, and retain target mappings and the queue pause. Clear
the pause only after every blocking cause is resolved. This starts the task's
first model turn; it never replays a completed turn.

The runner saves `agent-result.json`, history, and the terminal episode status
before shutting down the app-server or browser. Browser login-state export has
a 10-second deadline and replaces the previous private state file atomically
only after a successful export. Cleanup failures are retained in the episode
and browser logs. A terminal sequence of at least three observation timeouts
marks the episode as affected by infrastructure and stops the campaign for
inspection; recovered individual tool timeouts do not trigger this exclusion.

After stopping and inspecting a stalled runner, an explicit recovery command
can reconstruct a stale `running` episode from its unique matching terminal
model result and saved history. It archives the original episode and never
replays model actions:

```bash
python -m codex_benchmark.runner recover --episode PATH/episode.json --infrastructure-reason "Inspected browser failure"
```

Omit the infrastructure reason only when the model's task execution was not
impaired. Resume the original run afterward. For excluded episodes the official
rubric still runs and all evaluation steps are saved, but its result is shown
as `diagnostic_reward`, with `reward: null` in the summary. `evaluated` counts
all completed evaluator reports; `scored` and mean reward include only eligible
episodes. Recovery is recorded separately from full-task or evaluator timeouts.

An authentication-interrupted model turn remains `infrastructure_error` and
excluded. Under the drained account lock, `auth_recovery` can archive its exact
`run_error` snapshot and matching interrupted-turn confirmation, grade its
existing artifact diagnostically, and validate the completed report. Only then
can the queue recognize that reviewed exclusion as evaluated. This path creates
no completed model result and never replays the interrupted turn.

Model order is Luna, Astra, then Sol: smoke first, followed by each model's
remaining 109 tasks in that order. An explicit `judge_model` config setting can
override the repository's Gemini judge. The harness records the override in
every evaluation and the manifest. Results with an overridden judge should not
be described as an exact reproduction of the paper's evaluator settings.
Judge model/billing errors produce no score.

The influential-papers evaluators for instances 1–3 contain a stale PDF-helper
import. The adapter redirects that exact import to the benchmark's current
`eval_utils.image_utils` module in the temporary evaluator copy, leaving that
source file untouched. Each report records original and effective source hashes
plus any compatibility repair in `evaluator_source`.

This checkout also includes evaluator fixes discovered during Codex runs:

- Read embedded hyperlinks and source URLs in Docs, Sheets, and Slides, including
  supported DOI, Craigslist, KBB, and Wikipedia URL variants.
- Resolve chart data against its source sheet and ranges, and bound proportional
  running-analysis scores to the available points.
- Handle annotated travel dates, figure-image files, WebP payloads, and readable
  content-image comparisons.
- Surface incomplete structured extraction or image matching as evaluation
  errors instead of silently interpreting missing results as task failures.

These source changes can affect scores. Record the benchmark commit when
comparing runs; the evaluator is not identical to an unmodified upstream release.

The correction/review modules and `infrastructure_retry.py` and
`post_main_retry.py` retain narrowly scoped validators for the development
campaign. Some intentionally bind specific model/task IDs and dated recovery
archives. They require the matching local manifests and hash-bound evidence;
the repository does not include those private records. Ordinary new runs do not
apply historical corrections when those manifests are absent. Treat these
helpers as campaign audit/recovery support, not a general regrading interface.

## Experimental contract

- One repetition per task. `seed: 0` is a run label; app-server does not expose a
  deterministic sampling seed. Do not report model generation as reproducibly seeded.
- `max` is the highest individual-agent effort shared by these models. The local
  catalog describes `ultra` for Sol/Astra as maximum reasoning plus automatic
  delegation. This controlled individual-model harness rejects Ultra rather than
  silently turning its delegation off.
- All models receive the same UI browser tools, 1440 × 900 viewport, and the task
  text plus its workspace suffix. Defaults: 150 tool calls and 30 minutes per
  episode. Tool failures and observations count toward the tool limit.
- The app-server thread has no execution environment, shell, file access, personal
  instructions, memories, connectors, web-search tool, or subagents. Reference
  answers and the evaluator are available only to the supervisor.
- Normal browser actions operate through Playwright. No arbitrary JavaScript,
  direct Google artifact editing APIs, or network-request tools are exposed to the
  agent. Google APIs are used by the supervisor for provisioning and grading.
- Every main-frame navigation is recorded with its full URL, including popups,
  query strings, and fragments. `slides_17` honors the task-specific copy workflow
  and resolves the actual output deck before evaluation.
- The benchmark commit, model catalog, effort, prompt hash, limits, token usage,
  per-step evaluator categories, screenshots, action log, and Playwright trace are
  saved. Scores with evaluator execution errors remain explicitly flagged.

`summary.json` reports mean task reward over successfully graded tasks, with the
scored/expected denominator and infrastructure errors. A partial mean is not a
full benchmark score. Subscription token usage is not converted to API dollar
costs. Judge API usage is separate.

## Evaluation records

Every episode directory under `codex-results/<run>/<model>/<task>/` keeps:

- `evaluation.json`: the complete latest result, including every checkpoint and
  step with its ID, name, pass/fail, earned/possible points, details, category,
  and execution time, plus judge identities and errors.
- `evaluation-steps.jsonl`: an append-only journal that flushes each completed
  rubric step to disk as the evaluator registers it. Attempt IDs separate retries. Final checkpoint
  snapshots preserve score/category changes made after a step was added.
- `evaluation-attempts/<attempt-id>.json`: a separate final report for each
  grading attempt, so regrading cannot erase earlier results.
- `evaluate.log`: evaluator stdout/stderr. Browser actions, screenshots,
  navigation history, and model events are saved separately alongside it.

The journal retains registered rubric steps if a grading process is terminated.
Internal calculations that have not yet produced a rubric step are not recorded
as completed checks. `evaluation.json` remains
the final score report; intermediate journal events are not additional scores.

```bash
python -m unittest discover -s tests -p 'test_codex_*.py' -v
```

Sources: [KNOWS standalone protocol](https://github.com/alexgill321/KNOWS-benchmark/blob/main/RUNNING_STANDALONE.md),
[Codex app-server](https://learn.chatgpt.com/docs/app-server),
[Codex model availability](https://learn.chatgpt.com/docs/models).
