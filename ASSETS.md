# External Drive Assets

Six task families reference source documents, spreadsheets, presentations, or Drive folders from their `task.md` prompts. These assets are **hosted, view-only ("anyone with the link"), and referenced directly by the shipped `task.md` files** — no setup is needed to use them: agents read them in place, and they cannot be modified by benchmark runs.

| Family | Instances | Hosted source asset |
|---|---|---|
| `docs_1_formal_letter` | i1–i5 | Drive folder of letter source materials (signature photo among realistic distractor files) |
| `docs_11_personal_recipe_ocr` | i1–i5 | Recipe scan (file or folder) |
| `sheets_7_running_analysis` | i1–i5 | Strava/Garmin activity export (folder for i1–i2, spreadsheet for i3–i5) |
| `sheets_10_paper_sorting` | i1–i5 | Folder of paper PDFs (the *source* folder; the destination folder is created per run by `setup_run.py` in **your** Drive) |
| `slides_17_removeimagesaddplaceholders` | i1–i5 | The original reference presentation (the editable working copy and the image-save folder are provisioned per run in **your** Drive) |
| `slides_30_Work_Wikipedia_Photos` | i1–i5 | Client-list Google Doc (its ID is also the evaluator's `--client_doc_id` argument) |

## Per-run write targets (always yours)

Some prompts also name Drive locations the **agent writes into**. These are never shared assets — provision them in your own Drive per run and substitute their URLs:

- `sheets_10_paper_sorting`: run `python src/browsergym/knows/eval/tasks/sheets_10_paper_sorting/setup_run.py --instance <N> --parent_folder_id <your folder>` — it creates the destination folder and rewrites `task.md` + the evaluator's `DEST_FOLDER_ID` automatically.
- `slides_17_removeimagesaddplaceholders`: create an empty folder for saved images (put its URL in `task.md` and its ID in the evaluator's `DRIVE_FOLDER_ID` constant), and have the agent (or your harness) work on a copy of the hosted original presentation.

## Rehosting fallback

If the hosted links ever become unavailable, `knows-assets.zip` on this repository's GitHub Release page mirrors all source assets in a `<family>/instance_<N>/` layout. To rehost: upload each item to your own Drive (converting Office-format exports back to Google formats — Drive: *Open with → Google Docs/Sheets/Slides*), share as anyone-with-link **viewer**, and replace the corresponding source URL in each `instance_N/task.md` (for `slides_30`, also pass your copy's ID as `--client_doc_id`).

All other task families are self-contained: agents research the live public web, and evaluators use only the bundled `data/` gold assets plus public APIs.
