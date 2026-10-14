# Which tasks need a browsing history

Evaluators consume the agent's browsing history as a flat list of raw URL strings (see `RUNNING_STANDALONE.md` §3). This table says, per task family, whether the history changes the score. **Option A submissions (we grade your artifacts) must include histories for every family marked *Scores it = yes*; without them those steps are scored as failures.** Families marked *no* can be submitted without a history.

| Family | Accepts `browsing_history` | Scores it | History-dependent steps / total steps (per instance) | What the history is checked against |
|---|---|---|---|---|
| `docs_11_personal_recipe_ocr` | yes | **yes** | 2 / 25 | recipe-tip source URLs visited |
| `docs_1_formal_letter` | no | no | 0 / 13–15 | — |
| `docs_31_education_lesson_plan` | yes | **yes** | 1 / 23 | lesson sources visited (also accepts a Doc of links via `--browsing_history_doc_id`) |
| `docs_37_reference_list` | yes | no | 0 / 18–20 | — (parameter accepted but unused) |
| `docs_5_influential_papers` | yes | **yes** | 4 / 9 | paper / venue pages visited |
| `sheets_10_paper_sorting` | yes | **yes** | 2 / 27 | paper pages visited |
| `sheets_25_skitourplan` | yes | **yes** | 4 / 32–34 | tour / forecast sites visited |
| `sheets_28_personal_travel_planner` | yes | no | 0 / 40 | — (parameter accepted but unused) |
| `sheets_2_personal_recipe_foodcomposition` | yes | **yes** | 4 / 55–70 | recipe URL + nutrition source visited |
| `sheets_38_apartment_finder` | yes | **yes** | 4 / 25–27 | Craigslist listing pages visited |
| `sheets_45_Personal_WeddingPlanner_weddingcolorpallette` | yes | **yes** | 1 / 27–32 | ≥N wedding-colour articles visited (titles fetched, keyword/LLM-matched) |
| `sheets_55_Movie_Recommendation` | yes | **yes** | 1 / 13–16 | movie pages visited |
| `sheets_6_investmenttracker` | yes | **yes** | 2 / 16 | stock-info and price-data sites visited (pages fetched and cached for later steps) |
| `sheets_7_running_analysis` | yes | **yes** | 1–3 / 27–39 | baseline-source URLs (5K / Kipchoge) visited |
| `slides_17_removeimagesaddplaceholders` | no | no | 0 / 12 | — |
| `slides_20_Illustrated_Book_Report` | yes | **yes** | 2 / 15–16 | cited links appear in history |
| `slides_26_basic_educational_slide_deck` | yes | no | 0 / 24 | — (parameter accepted but unused) |
| `slides_29_buy_car_pres` | yes | **yes** | 2 / 20 | car listing / spec pages visited |
| `slides_30_Work_Wikipedia_Photos` | yes | **yes** | 3 / 14 | Wikipedia featured-picture pages visited |
| `slides_39_Personal_Lookbook_PaintColors` | yes | **yes** | 2 / 19 | paint-brand pages visited |
| `slides_42_personal_none_product_comparison` | yes | no | 0 / 41 | — (parameter accepted but unused) |
| `slides_51_event_announcement_poster` | yes | no | 0 / 37–38 | — (parameter accepted but unused) |

**15 of 22 families (75 of 110 instances) score the history**, covering roughly 180 of ~2751 evaluation steps (6.5%). Counted from the evaluators at commit `a216282` (main, 2026-10-12): steps tagged `StepCategory.WEB_VISIT`, plus the history-gated steps in `sheets_45` (article research, categorised deterministic/LLM) and `sheets_6` (two source steps, categorised `execution_error` when the history is missing). Total steps are bullet/numbered items under each checkpoint's evaluation list in `checkpoints.md`, so they are approximate for proportional steps.

Families that need a history:

```
docs_11_personal_recipe_ocr
docs_31_education_lesson_plan
docs_5_influential_papers
sheets_10_paper_sorting
sheets_25_skitourplan
sheets_2_personal_recipe_foodcomposition
sheets_38_apartment_finder
sheets_45_Personal_WeddingPlanner_weddingcolorpallette
sheets_55_Movie_Recommendation
sheets_6_investmenttracker
sheets_7_running_analysis
slides_20_Illustrated_Book_Report
slides_29_buy_car_pres
slides_30_Work_Wikipedia_Photos
slides_39_Personal_Lookbook_PaintColors
```

Five families accept the argument but never read it (`docs_37`, `sheets_28`, `slides_26`, `slides_42`, `slides_51`); two take no argument at all (`docs_1`, `slides_17`).
