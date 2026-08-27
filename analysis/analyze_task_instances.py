"""
Analyzes all task instances in Agent-Benchmark and prints dataset statistics.
Output is also saved to analysis_results.md.

Covers:
  - Basic NLP stats on task prompts (word counts, vocab, etc.)
  - Artifact type distribution (docs / sheets / slides)
  - Checkpoint and evaluation step statistics
  - Eval template type distribution
"""

import io
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

TASKS_DIR = Path(__file__).parent.parent / "src" / "browsergym" / "knows" / "eval" / "tasks"
OUTPUT_MD = Path(__file__).parent / "analysis_results.md"

# Keywords used to infer template type from step description text (for Eval Steps: format)
TEMPLATE_INFERENCE_RULES: list[tuple[str, list[str]]] = [
    ("LLM / VLM Judge",      ["vlm", "llm", "language model", "vision model", "relevant", "reasonable substitute",
                               "engaging", "related to", "verif", "reputable", "accurate"]),
    ("Web Visit Check",       ["browsing history", "visited", "visit to"]),
    ("Link Validation",       ["valid url", "valid link", "url pointing", "arXiv url", "drive url",
                               "valid arXiv", "valid drive", "hyperlink"]),
    ("Image Match",           ["image", "figure 1", "figure1", "photo", "picture"]),
    ("Text Fuzzy Match",      ["fuzzy match", "fuzzy"]),
    ("Text Exact Match",      ["exact match", "labeled", "reads ", "correct title", "correct author",
                               "correct abstract", "keyword"]),
    ("Color / Formatting",    ["color", "colour", "highlight", "bold", "italic", "font", "centered",
                               "aligned", "grouped", "format"]),
    ("Numerical Match",       ["tolerance", "within", "score", "rating", "matches the listed value"]),
    ("Structural Check",      ["checkbox", "checked", "column", "row", "overflow", "structure"]),
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def get_artifact_type(task_name: str) -> str:
    for prefix in ("docs", "sheets", "slides"):
        if task_name.lower().startswith(prefix):
            return prefix
    return "unknown"


def word_count(text: str) -> int:
    return len(text.split())


def infer_template_from_step(step_text: str) -> str:
    lower = step_text.lower()
    for template, keywords in TEMPLATE_INFERENCE_RULES:
        if any(kw in lower for kw in keywords):
            return template
    return "Other"


def parse_checkpoints_md(path: Path) -> dict:
    """Return structured info parsed from a checkpoints.md file.

    Handles two checkpoint formats:
      - ### Outcome Evaluation  (bullet lines starting with -)
      - ### Eval Steps[...]:   (numbered items starting with digits)
    """
    if not path.exists():
        return {
            "num_checkpoints": 0,
            "outcome_steps": [],
            "eval_templates": [],
            "total_points": 0,
        }

    text = path.read_text(encoding="utf-8")

    # --- total points ---
    total_pts = 0
    pts_match = re.search(r"(\d+)\s+points? in total", text, re.IGNORECASE)
    if pts_match:
        total_pts = int(pts_match.group(1))
    else:
        # sum pts from checkpoint headers like "## Checkpoint 1 (25 pt):"
        total_pts = sum(int(m) for m in re.findall(r"##\s+Checkpoint[^(]*\((\d+)", text))

    # --- split into checkpoint blocks ---
    # Some files use single # for checkpoint headers (e.g. slides_29, slides_42)
    checkpoint_blocks = re.split(r"(?=^#{1,2} Checkpoint)", text, flags=re.MULTILINE)

    num_checkpoints = 0
    outcome_steps: list[int] = []
    eval_templates: list[str] = []

    for block in checkpoint_blocks:
        if not re.match(r"^#{1,2} Checkpoint", block.strip()):
            continue
        num_checkpoints += 1

        # ---- find the eval section (any variant of the header) ----
        # "Evalutation" is a known typo in slides_51; using Eval\w* to catch any misspelling
        eval_section_match = re.search(
            r"###\s+(?:Outcome Eval\w*|Eval Steps?)[^#\n]*\n(.*?)(?=^#{1,3} |\Z)",
            block, re.DOTALL | re.MULTILINE | re.IGNORECASE,
        )
        steps = 0
        if eval_section_match:
            body = eval_section_match.group(1)

            # Count bullet lines (- ...) or numbered items (1. ...)
            # Exclude --- horizontal rules by requiring the char after [-*] is not another -
            # Also handles files that omit the space after the dash (e.g. "-All topics...")
            # Only match top-level bullets (no leading whitespace) to avoid counting sub-bullets
            bullet_lines = re.findall(r"^[-*](?!-)\s*.+", body, re.MULTILINE)
            numbered_lines = re.findall(r"^\d+\.\s+.+", body, re.MULTILINE)
            step_lines = bullet_lines if bullet_lines else numbered_lines
            steps = len(step_lines)

            # Infer template types from step descriptions
            for line in step_lines:
                line_clean = re.sub(r"^\s*[-*\d.]+\s*", "", line).strip()
                eval_templates.append(infer_template_from_step(line_clean))

        # ---- also parse explicit Eval Template(s) lines if present ----
        explicit_match = re.search(
            r"###\s+Eval Template[s]?\s*\n(.*?)(?=^#{1,2} |\Z)",
            block, re.DOTALL | re.MULTILINE | re.IGNORECASE,
        )
        if explicit_match:
            raw = explicit_match.group(1).strip()
            for line in raw.splitlines():
                line = line.strip().lstrip("-").strip()
                if line:
                    for t in line.split(","):
                        t = t.strip()
                        if t:
                            eval_templates.append(t)

        outcome_steps.append(steps)

    return {
        "num_checkpoints": num_checkpoints,
        "outcome_steps": outcome_steps,
        "eval_templates": eval_templates,
        "total_points": total_pts,
    }


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_instances(tasks_dir: Path) -> list[dict]:
    rows = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        artifact = get_artifact_type(task_dir.name)
        for instance_dir in sorted(task_dir.iterdir()):
            if not instance_dir.is_dir() or not instance_dir.name.startswith("instance_"):
                continue

            task_md = instance_dir / "task.md"
            prompt = task_md.read_text(encoding="utf-8").strip() if task_md.exists() else ""

            id_file = instance_dir / "id.txt"
            task_id = id_file.read_text(encoding="utf-8").strip() if id_file.exists() else ""

            cp_info = parse_checkpoints_md(instance_dir / "checkpoints.md")

            rows.append({
                "task_name": task_dir.name,
                "instance": instance_dir.name,
                "task_id": task_id,
                "artifact": artifact,
                "prompt": prompt,
                "word_count": word_count(prompt),
                "char_count": len(prompt),
                **cp_info,
            })
    return rows


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def five_num(values: list, label: str) -> str:
    if not values:
        return f"  {label}: n/a"
    mn = min(values)
    mx = max(values)
    mean = statistics.mean(values)
    med = statistics.median(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return (
        f"  {label}:\n"
        f"    mean={mean:.1f}  median={med:.1f}  std={sd:.1f}  "
        f"min={mn}  max={mx}  n={len(values)}"
    )


def pct(n: int, total: int) -> str:
    return f"{n} ({100 * n / total:.1f}%)" if total else str(n)


# ---------------------------------------------------------------------------
# Main analysis  (writes to `out`, a file-like object)
# ---------------------------------------------------------------------------

def analyze(rows: list[dict], out: io.TextIOBase) -> None:
    def p(*args, **kwargs):
        print(*args, **kwargs, file=out)

    def section(title: str) -> None:
        width = 64
        p("\n" + "=" * width)
        p(f"  {title}")
        p("=" * width)

    n = len(rows)

    # -----------------------------------------------------------------------
    section("OVERVIEW")
    # -----------------------------------------------------------------------
    task_names = {r["task_name"] for r in rows}
    artifact_counts = Counter(r["artifact"] for r in rows)
    task_per_artifact = defaultdict(set)
    for r in rows:
        task_per_artifact[r["artifact"]].add(r["task_name"])

    p(f"  Total instances      : {n}")
    p(f"  Unique tasks         : {len(task_names)}")
    instances_per_task = Counter(r["task_name"] for r in rows)
    avg_inst = statistics.mean(instances_per_task.values()) if instances_per_task else 0
    p(f"  Avg instances / task : {avg_inst:.1f}")

    # -----------------------------------------------------------------------
    section("ARTIFACT TYPE DISTRIBUTION")
    # -----------------------------------------------------------------------
    for art in ("docs", "sheets", "slides", "unknown"):
        c = artifact_counts[art]
        t = len(task_per_artifact[art])
        if c == 0:
            continue
        p(f"  {art:8s}: {pct(c, n)} instances, {t} unique tasks")

    # -----------------------------------------------------------------------
    section("NLP STATS — TASK PROMPTS")
    # -----------------------------------------------------------------------
    all_wc = [r["word_count"] for r in rows]
    all_cc = [r["char_count"] for r in rows]
    total_words = sum(all_wc)
    all_words = " ".join(r["prompt"] for r in rows).lower()
    tokens = re.findall(r"\b[a-z]+\b", all_words)
    vocab = set(tokens)

    p(f"  Total words (corpus) : {total_words:,}")
    p(f"  Total tokens         : {len(tokens):,}")
    p(f"  Vocabulary size      : {len(vocab):,}")
    if tokens:
        p(f"  Type-token ratio     : {len(vocab)/len(tokens):.3f}")
    p()
    p(five_num(all_wc, "Words per prompt (all)"))
    for art in ("docs", "sheets", "slides"):
        p(five_num([r["word_count"] for r in rows if r["artifact"] == art], f"Words per prompt ({art})"))
    p()
    p(five_num(all_cc, "Chars per prompt (all)"))
    p()
    sent_counts = [len(re.split(r"[.!?]+", r["prompt"])) for r in rows]
    p(five_num(sent_counts, "Sentences per prompt (approx, all)"))
    for art in ("docs", "sheets", "slides"):
        p(five_num(
            [len(re.split(r"[.!?]+", r["prompt"])) for r in rows if r["artifact"] == art],
            f"Sentences per prompt ({art})"
        ))

    # -----------------------------------------------------------------------
    section("CHECKPOINT STATISTICS")
    # -----------------------------------------------------------------------
    cp_counts = [r["num_checkpoints"] for r in rows]
    p(f"  Total checkpoints    : {sum(cp_counts)}")
    p()
    p(five_num(cp_counts, "Checkpoints per instance (all)"))
    for art in ("docs", "sheets", "slides"):
        p(five_num([r["num_checkpoints"] for r in rows if r["artifact"] == art],
                   f"Checkpoints per instance ({art})"))

    # -----------------------------------------------------------------------
    section("EVALUATION STEP STATISTICS")
    # -----------------------------------------------------------------------
    all_steps_per_cp: list[int] = []
    steps_per_instance: list[int] = []
    for r in rows:
        total = sum(r["outcome_steps"])
        steps_per_instance.append(total)
        all_steps_per_cp.extend(r["outcome_steps"])

    p(f"  Total eval steps     : {sum(steps_per_instance)}")
    p()
    p(five_num(steps_per_instance, "Eval steps per instance (all)"))
    for art in ("docs", "sheets", "slides"):
        p(five_num([sum(r["outcome_steps"]) for r in rows if r["artifact"] == art],
                   f"Eval steps per instance ({art})"))
    p()
    p(five_num(all_steps_per_cp, "Eval steps per checkpoint (all)"))

    # -----------------------------------------------------------------------
    section("POINTS STATISTICS")
    # -----------------------------------------------------------------------
    pts_all = [r["total_points"] for r in rows if r["total_points"] > 0]
    p(five_num(pts_all, "Total points per instance (all)"))
    for art in ("docs", "sheets", "slides"):
        p(five_num([r["total_points"] for r in rows if r["artifact"] == art and r["total_points"] > 0],
                   f"Total points per instance ({art})"))

    # -----------------------------------------------------------------------
    section("EVAL TEMPLATE TYPE DISTRIBUTION")
    # -----------------------------------------------------------------------
    # Per-instance template sets (for "# instances using X")
    instance_template_sets: list[set[str]] = [set(r["eval_templates"]) for r in rows]
    all_templates_flat: list[str] = [t for r in rows for t in r["eval_templates"]]
    template_counter = Counter(all_templates_flat)
    # Count instances that use each template at least once
    instance_uses: Counter = Counter()
    for tmpl_set in instance_template_sets:
        for tmpl in tmpl_set:
            instance_uses[tmpl] += 1

    unique_types = len(template_counter)
    total_mentions = len(all_templates_flat)
    p(f"  Unique template types : {unique_types}")
    p(f"  Total step mentions   : {total_mentions}")
    p()
    p(f"  {'Template':<30}  {'Steps':>6}  {'% Steps':>8}  {'Instances':>10}  {'% Inst':>8}")
    p(f"  {'-'*30}  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}")
    for tmpl, cnt in template_counter.most_common():
        inst_cnt = instance_uses[tmpl]
        p(f"  {tmpl:<30}  {cnt:>6}  {100*cnt/total_mentions:>7.1f}%  {inst_cnt:>10}  {100*inst_cnt/n:>7.1f}%")

    # -----------------------------------------------------------------------
    section("TASK-LEVEL SUMMARY")
    # -----------------------------------------------------------------------
    task_data: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        task_data[r["task_name"]].append(r)

    p(f"  {'Task':<52} {'Art':6} {'Inst':5} {'CPs':5} {'Steps':6} {'Pts':6}")
    p(f"  {'-'*52} {'-'*6} {'-'*5} {'-'*5} {'-'*6} {'-'*6}")
    for task_name in sorted(task_data):
        recs = task_data[task_name]
        art = recs[0]["artifact"]
        n_inst = len(recs)
        avg_cps = statistics.mean(r["num_checkpoints"] for r in recs)
        avg_steps = statistics.mean(sum(r["outcome_steps"]) for r in recs)
        avg_pts = statistics.mean(r["total_points"] for r in recs)
        p(f"  {task_name:<52} {art:6} {n_inst:5} {avg_cps:5.1f} {avg_steps:6.1f} {avg_pts:6.1f}")


# ---------------------------------------------------------------------------
# Diagnostic: per-instance table + outlier flagging
# ---------------------------------------------------------------------------

def diagnose(rows: list[dict]) -> None:
    """Print per-instance stats and flag outliers."""
    import statistics as _stats

    all_wc    = [r["word_count"] for r in rows]
    all_cps   = [r["num_checkpoints"] for r in rows]
    all_steps = [sum(r["outcome_steps"]) for r in rows]
    all_pts   = [r["total_points"] for r in rows]

    wc_mean,    wc_sd    = _stats.mean(all_wc),    _stats.stdev(all_wc)
    cp_mean,    cp_sd    = _stats.mean(all_cps),   _stats.stdev(all_cps)
    step_mean,  step_sd  = _stats.mean(all_steps), _stats.stdev(all_steps)
    pts_mean,   pts_sd   = _stats.mean(all_pts),   _stats.stdev(all_pts)

    def flag(val, mean, sd, threshold=2.0) -> str:
        if sd == 0:
            return ""
        z = (val - mean) / sd
        if z > threshold:
            return f"HIGH({z:+.1f}σ)"
        if z < -threshold:
            return f"LOW({z:+.1f}σ)"
        return ""

    print(f"\n{'Task / Instance':<62} {'Words':>6} {'CPs':>4} {'Steps':>6} {'Pts':>5}  Flags")
    print("-" * 110)

    task_data: dict[str, list[dict]] = {}
    for r in rows:
        task_data.setdefault(r["task_name"], []).append(r)

    outliers: list[dict] = []

    for task_name in sorted(task_data):
        task_rows = task_data[task_name]
        # also compute within-task z-scores for steps and checkpoints
        task_steps = [sum(r["outcome_steps"]) for r in task_rows]
        task_cps   = [r["num_checkpoints"] for r in task_rows]
        t_step_mean = _stats.mean(task_steps)
        t_step_sd   = _stats.stdev(task_steps) if len(task_steps) > 1 else 0
        t_cp_mean   = _stats.mean(task_cps)
        t_cp_sd     = _stats.stdev(task_cps)   if len(task_cps)   > 1 else 0

        for r in sorted(task_rows, key=lambda x: x["instance"]):
            steps = sum(r["outcome_steps"])
            pts   = r["total_points"]
            wc    = r["word_count"]
            cps   = r["num_checkpoints"]

            flags = []
            g = flag(wc,    wc_mean,   wc_sd)   ; flags.append(f"words:{g}") if g else None
            g = flag(cps,   cp_mean,   cp_sd)   ; flags.append(f"cps:{g}")   if g else None
            g = flag(steps, step_mean, step_sd) ; flags.append(f"steps:{g}") if g else None
            g = flag(pts,   pts_mean,  pts_sd)  ; flags.append(f"pts:{g}")   if g else None
            # within-task outliers (looser threshold since smaller N)
            g = flag(steps, t_step_mean, t_step_sd, threshold=1.5)
            if g: flags.append(f"within-task steps:{g}")
            g = flag(cps, t_cp_mean, t_cp_sd, threshold=1.5)
            if g: flags.append(f"within-task cps:{g}")
            if steps == 0:
                flags.append("ZERO STEPS")
            if cps == 0:
                flags.append("ZERO CPS")

            flag_str = "  ".join(flags)
            label = f"  {task_name}/{r['instance']}"
            print(f"{label:<62} {wc:>6} {cps:>4} {steps:>6} {pts:>5}  {flag_str}")

            if flags:
                outliers.append({**r, "steps": steps, "flags": flag_str})

    print(f"\n{'='*64}")
    print(f"  OUTLIER SUMMARY ({len(outliers)} flagged instances)")
    print(f"{'='*64}")
    for o in outliers:
        print(f"  {o['task_name']}/{o['instance']}: {o['flags']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    rows = collect_instances(TASKS_DIR)
    if not rows:
        print("No task instances found.")
        return

    # Write to stdout and capture for .md file simultaneously
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            buf.write(s)
            return len(s)

        def flush(self):
            buf.flush()

    tee = Tee()

    import sys
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        analyze(rows, tee)
    finally:
        sys.stdout = old_stdout

    output = buf.getvalue()
    print(output, end="")

    md_content = f"# Agent-Benchmark Dataset Analysis\n\n```\n{output}\n```\n"
    OUTPUT_MD.write_text(md_content, encoding="utf-8")
    print(f"\nSaved to {OUTPUT_MD}")

    print("\n\n" + "=" * 64)
    print("  PER-INSTANCE DIAGNOSTIC")
    print("=" * 64)
    diagnose(rows)


if __name__ == "__main__":
    main()
