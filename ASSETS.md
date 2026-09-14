# External Drive Assets

Six task families reference source documents, spreadsheets, presentations, or Drive folders from their `task.md` prompts. These assets are **hosted, view-only ("anyone with the link"), and referenced directly by the shipped `task.md` files** — no setup is needed to use them: agents read them in place, and they cannot be modified by benchmark runs.

| Family | Instances | Hosted source asset |
|---|---|---|
| `docs_1_formal_letter` | i1–i5 | Drive folder of letter source materials (signature photo among realistic distractor files) |
| `docs_11_personal_recipe_ocr` | i1–i5 | Recipe scan (file or folder) |
| `sheets_7_running_analysis` | i1–i5 | Strava/Garmin activity export (folder for i1–i2, spreadsheet for i3–i5) |
| `sheets_10_paper_sorting` | i1–i5 | Folder of paper PDFs (the *source* folder; the upload destination is provisioned per run in **your** Drive — see below) |
| `slides_17_removeimagesaddplaceholders` | i1–i5 | The original reference presentation (the editable working copy and the image-save folder are provisioned per run in **your** Drive) |
| `slides_30_Work_Wikipedia_Photos` | i1–i5 | Client-list Google Doc (its ID is also the evaluator's `--client_doc_id` argument) |

## Per-run write targets (always yours)

Two families also name Drive locations the **agent writes into**. These can never be shared assets, so their `task.md` prompts carry `{{PLACEHOLDER}}` tokens rather than URLs, and one script provisions the real locations in your own Drive:

```bash
python src/browsergym/knows/eval/tasks/provision_run_targets.py --all
```

| Family | Instances | Tokens | What gets created |
|---|---|---|---|
| `sheets_10_paper_sorting` | i1–i5 | `{{OUTPUT_FOLDER_URL}}` | A run folder containing `pdfs/` and `figures/` subfolders — the prompt tells the agent to upload into those two by name, so **the token points at their parent**, not at either subfolder |
| `slides_17_removeimagesaddplaceholders` | i1 | `{{WORKING_COPY_URL}}`, `{{IMAGES_FOLDER_URL}}` | An editable copy of the source deck, plus an `images/` folder |
| `slides_17_removeimagesaddplaceholders` | i2–i5 | `{{OUTPUT_FOLDER_URL}}`, `{{IMAGES_FOLDER_URL}}` | A `copies/` folder the agent puts its own copy into, plus an `images/` folder |

Each pass creates a fresh `run_NNNN` folder per instance, so a previous run's uploads are never counted against the current one. Everything is created **anyone-with-link writable** so the agent's browser session can write there regardless of which account it is signed into.

The resolved URLs land in `run_targets.json` at the repository root (git-ignored). Both the harness — which substitutes them into the prompt before the agent sees it — and the evaluators read from there, so no file needs hand-editing. Precedence is environment variable first, then the config file:

```bash
# override a single token without touching the config
export KNOWS_TARGET_SLIDES_17_REMOVEIMAGESADDPLACEHOLDERS_INSTANCE_2_IMAGES_FOLDER_URL="https://drive.google.com/drive/folders/…"
```

**Credentials note.** `slides_17` instance 1 copies a presentation, and service accounts have no Drive storage quota, so they cannot own the copy — run that one with `--auth oauth`. Folder-only provisioning works with either credential.

If you run a task without provisioning it, the prompt substitution fails immediately and names both the missing token and the command that creates it, rather than sending the agent at a dead URL.

## Rehosting fallback

If the hosted links ever become unavailable, `knows-assets.zip` on this repository's GitHub Release page mirrors all source assets in a `<family>/instance_<N>/` layout. To rehost: upload each item to your own Drive (converting Office-format exports back to Google formats — Drive: *Open with → Google Docs/Sheets/Slides*), share as anyone-with-link **viewer**, and replace the corresponding source URL in each `instance_N/task.md` (for `slides_30`, also pass your copy's ID as `--client_doc_id`).

All other task families are self-contained: agents research the live public web, and evaluators use only the bundled `data/` gold assets plus public APIs.
