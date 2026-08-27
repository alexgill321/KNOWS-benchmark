"""Evaluator for the Personal Travel Planner Google Sheets task."""

import os
import sys
import time
import argparse
from typing import List

BASE_PATH = (
    "/app" if os.path.exists("/app/src")
    else "." if os.path.exists("/scratch")
    else os.getcwd()
)
sys.path.append(BASE_PATH)

from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, StepCategory, calculate_percentage_score
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.google_sheets_utils import (
    get_sheet_content,
    detect_header_row,
    parse_sheet_to_dataframe,
)
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.table_utils import (
    get_cell, get_column_index_by_name, get_text_foreground_color,
)
from src.browsergym.knows.eval.eval_utils.web_utils import is_url_from_domain

from src.browsergym.knows.eval.tasks.sheets_28_personal_travel_planner.utils import (
    REQUIRED_COLUMNS,
    EXPECTED_ACTIVITIES_PER_DAY,
    EXPECTED_FOOD_STOPS_PER_DAY,
    TIME_SLOT_BLOCKS,
    validate_header_row,
    count_rows_per_day,
    run_vlm_batch,
    score_vlm_steps,
    empty_checkpoint,
    format_failures,
    add_fraction_step,
    evaluate_cost_color_coding,
    seed_cp6_route_and_daytype,
    build_cp3_cp4_vlm_results,
    build_task_context,
    find_semantic_duplicates,
)

TASK_DIR = os.path.join(
    BASE_PATH,
    "src/browsergym/knows/eval/tasks/sheets_28_personal_travel_planner/instance_4/",
)
model_id = "gemini-3-flash-google-ai"
GOOGLE_MAPS_DOMAINS = ("google.com/maps", "maps.app.goo.gl", "goo.gl/maps", "maps.google.com")

DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

# --- Module globals populated by setup() ---
sheet_id = sheet_raw = df = header_row_idx = model = None
raw_rows = []
maps_api_key = ""
matched_columns = {}
matched_column_methods = {}
dest_names = {}
trip_details = {}
all_review_urls = {}
url_content = {}
place_cache = {}
directions_cache = {}
alt_directions_cache = {}
pairwise_transit_cache = {}
day_specific_transit_cache = {}
day_return_transit_cache = {}
url_geocode_cache = {}
all_dest_names = []
all_alt_names = []
all_alt_types = []
row_data = []
places_out_of_city = set()


def setup(workspace_doc_id: str):
    """Fetch the sheet and delegate all pre-computation to utils.build_task_context()."""
    global sheet_id, sheet_raw, df, header_row_idx, model
    raw_rows.clear()
    matched_columns.clear(); matched_column_methods.clear(); dest_names.clear(); trip_details.clear()
    all_review_urls.clear(); url_content.clear()
    place_cache.clear(); directions_cache.clear(); alt_directions_cache.clear()
    pairwise_transit_cache.clear(); day_specific_transit_cache.clear()
    day_return_transit_cache.clear()
    all_dest_names.clear(); all_alt_names.clear(); all_alt_types.clear()
    row_data.clear(); places_out_of_city.clear()

    if workspace_doc_id:
        sheet_id = workspace_doc_id

    sheet_raw = get_sheet_content(sheet_id, SHEETS_SERVICE)

    if sheet_raw:
        try:
            sheets = sheet_raw.get("sheets", [])
            if sheets:
                data = sheets[0].get("data") or [{}]
                raw_rows.extend(data[0].get("rowData", []))
                if raw_rows:
                    header_row_idx = validate_header_row(raw_rows, detect_header_row(raw_rows))
            df = parse_sheet_to_dataframe(sheet_raw, header_row=header_row_idx)
        except Exception as e:
            print(f"Error extracting DataFrame: {e}")
            df = None

    if df is None or df.empty:
        return

    if model is None:
        model = load_model(model_id)

    globals().update(build_task_context(
        df=df, sheet_raw=sheet_raw, raw_rows=raw_rows,
        header_row_idx=header_row_idx, task_dir=TASK_DIR, model=model,
        gold_path=os.path.join(TASK_DIR, "task.md"),
    ))


def grade_checkpoint_1():
    """Checkpoint 1: Table Structure and Layout (40 pts, 10 pts each step)."""
    if df is None or df.empty:
        return empty_checkpoint("Table Structure and Layout", 40)

    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=40, result=0, name="Table Structure and Layout")

    # Step 1: All headers match
    step_start = time.time()
    missing = [name for name, _ in REQUIRED_COLUMNS if name not in matched_columns]
    checkpoint.add_step(
        "All Headers Match", not missing, 1,
        details=f"Found {len(matched_columns)}/{len(REQUIRED_COLUMNS)} required columns"
        + (f". Missing: {missing}" if missing else ""),
        score=calculate_percentage_score(len(matched_columns), len(REQUIRED_COLUMNS), 10),
        max_score=10, execution_time=time.time() - step_start,
        category=StepCategory.aggregate([
            (StepCategory.DETERMINISTIC
             if matched_column_methods.get(name) == "keyword"
             else StepCategory.LLM_VLM_JUDGEMENT,
             name in matched_columns)
            for name, _ in REQUIRED_COLUMNS
        ]),
    )

    # Step 2: Header frozen
    step_start = time.time()
    sheets = sheet_raw.get("sheets", [])
    frozen_rows = (
        sheets[0].get("properties", {}).get("gridProperties", {}).get("frozenRowCount", 0)
        if sheets else 0
    )
    is_frozen = frozen_rows >= (header_row_idx + 1) if header_row_idx is not None else frozen_rows >= 1
    checkpoint.add_step(
        "Header Row Frozen", is_frozen, 2,
        details=f"Frozen rows: {frozen_rows}, header at row index {header_row_idx}",
        score=10 if is_frozen else 0, max_score=10,
        execution_time=time.time() - step_start,
        category=StepCategory.DETERMINISTIC,
    )

    # Steps 3 & 4: Row counts per day (Activity/Food via pre-classified row_type from setup)
    step_start = time.time()
    date_col = matched_columns.get("Date")
    row_types = [rd.get("row_type", "activity") for rd in row_data]
    day_counts = count_rows_per_day(df, date_col, row_types) if date_col and row_types else []
    total_days = len(day_counts)

    for step_id, name, key, expected in (
        (3, "3 Activity Rows Per Day", "activity_count", EXPECTED_ACTIVITIES_PER_DAY),
        (4, "2 Food Rows Per Day", "food_count", EXPECTED_FOOD_STOPS_PER_DAY),
    ):
        if not total_days:
            checkpoint.add_step(
                name, False, step_id,
                details="No days found" if date_col and row_types else "Required columns not found",
                score=0, max_score=10, execution_time=time.time() - step_start,
                category=StepCategory.EXECUTION_ERROR,
            )
            continue

        correct = 0
        per_day_reports = []
        failures = []
        for d in day_counts:
            label = d["date"] if d["date"] else "(date missing)"
            issues = []
            if not d["date"]:
                issues.append("date missing")
            elif d["parsed_date"] is None:
                issues.append("date unparseable")
            if d[key] != expected:
                issues.append(f"{d[key]}/{expected} rows")
            if not issues:
                correct += 1
                per_day_reports.append(f"{label}: {d[key]}")
            else:
                per_day_reports.append(f"{label}: {d[key]} ({'; '.join(issues)})")
                failures.append(f"{label} [{'; '.join(issues)}]")
        detail = f"{correct}/{total_days} days correct. [" + ", ".join(per_day_reports) + "]"
        detail += format_failures(failures, limit=None)
        add_fraction_step(checkpoint, name, step_id, correct, total_days, detail, step_start,
                          category=StepCategory.DETERMINISTIC)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """Checkpoint 2: Content Completeness (130 pts, 10 pts each step)."""
    if df is None or df.empty:
        return empty_checkpoint("Content Completeness", 130)

    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=130, result=0, name="Content Completeness")

    total_rows = len(df)
    date_col = matched_columns.get("Date") if matched_columns.get("Date") in df.columns else None

    simple_fields = [
        ("destination", "Destination"),
        ("time_of_day", "Time of Day"),
        ("opening_hours", "Opening Time"),
        ("start_time", "Start Time"),
        ("departure_time", "Departure Time"),
        ("duration", "Duration"),
        ("transport_mode", "Transportation Mode"),
        ("travel_time", "Travel Time"),
        ("cost", "Cost"),
        ("alternative", "Alternative Option"),
    ]
    filled = {k: 0 for k, _ in simple_fields}
    filled.update({"review_link": 0, "cuisine": 0, "date": 0})
    failures = {k: [] for k in filled}
    food_row_count = 0

    for idx in range(total_rows):
        row = df.iloc[idx]
        row_label = f"Row {idx + 1}"

        for key, col_name in simple_fields:
            c = matched_columns.get(col_name)
            if c and str(row.get(c, "")).strip():
                filled[key] += 1
            else:
                failures[key].append(row_label)

        if date_col and str(row.get(date_col, "")).strip():
            filled["date"] += 1
        else:
            failures["date"].append(row_label)

        url = all_review_urls.get(idx, "")
        if url and any(is_url_from_domain(url, d) for d in GOOGLE_MAPS_DOMAINS):
            filled["review_link"] += 1
        else:
            failures["review_link"].append(row_label)

        if idx < len(row_data) and row_data[idx].get("row_type") == "food":
            food_row_count += 1
            cuisine_col = matched_columns.get("Cuisine")
            if cuisine_col and str(row.get(cuisine_col, "")).strip():
                filled["cuisine"] += 1
            else:
                failures["cuisine"].append(row_label)

    steps = [
        (1, "Destination Name", "destination", total_rows),
        (2, "Time of Day", "time_of_day", total_rows),
        (3, "Opening Hours", "opening_hours", total_rows),
        (4, "Start Time", "start_time", total_rows),
        (5, "Departure Time", "departure_time", total_rows),
        (6, "Duration", "duration", total_rows),
        (7, "Google Maps Review Link", "review_link", total_rows),
        (8, "Transportation Mode", "transport_mode", total_rows),
        (9, "Travel Time", "travel_time", total_rows),
        (10, "Cost", "cost", total_rows),
        (11, "Cuisine (Restaurants)", "cuisine", food_row_count),
        (12, "Alternative Option", "alternative", total_rows),
        (13, "Date", "date", total_rows),
    ]

    for step_id, name, key, denom in steps:
        count = filled[key]
        detail = f"{count}/{denom} rows have {name.lower()}"
        detail += format_failures(failures[key], prefix="Missing", sep=", ")
        add_fraction_step(checkpoint, name, step_id, count, denom, detail,
                          category=StepCategory.DETERMINISTIC)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3(vlm_results=None, vlm_reasons=None, vlm_seed_categories=None):
    """Checkpoint 3: Destination Validity and Uniqueness (40 pts, 4 steps)."""
    if df is None or df.empty:
        return empty_checkpoint("Destination Validity and Uniqueness", 40)

    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=40, result=0, name="Destination Validity and Uniqueness")
    vlm_results = vlm_results or {}
    city_name = trip_details.get("city", "the destination city")

    # Steps 1, 2, 4: scored from shared vlm_results (built once in grade_checkpoints)
    score_vlm_steps(checkpoint, vlm_results, [
        (1, "Destinations Exist in City", "dest_", all_dest_names,
         f"{{passed}}/{{total}} destinations verified in {city_name}",
         "No destinations to verify"),
        (2, "Alternatives Exist in City", "alt_", all_alt_names,
         f"{{passed}}/{{total}} alternative options verified in {city_name}",
         "No alternatives to verify"),
        (4, "Open During Visit Hours", "open_", dest_names,
         "{passed}/{total} destinations verified as open during visit time",
         "No visit time data to verify"),
    ], reasons=vlm_reasons, seed_categories=vlm_seed_categories)

    # Step 3: Uniqueness (exact match + LLM semantic dedup)
    step_start = time.time()
    all_names = [n for n in all_dest_names + all_alt_names if n.strip()]
    duplicate_groups = find_semantic_duplicates(all_names, model)

    is_unique = not duplicate_groups
    total_names = len(all_names)
    if is_unique:
        detail = f"All {total_names} names are unique"
    else:
        group_strs = [" / ".join(g) for g in duplicate_groups]
        detail = f"Duplicate groups found: {'; '.join(group_strs)}"
    checkpoint.add_step(
        "All Names Unique", is_unique, 3,
        details=detail,
        score=10 if is_unique
              else calculate_percentage_score(
                  max(0, total_names - sum(len(g) - 1 for g in duplicate_groups)),
                  total_names, 10),
        max_score=10, execution_time=time.time() - step_start,
        category=(StepCategory.LLM_VLM_JUDGEMENT if model is not None
                  else StepCategory.DETERMINISTIC),
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4(vlm_results=None, vlm_reasons=None, vlm_seed_categories=None):
    """Checkpoint 4: Content Accuracy (70 pts, 7 steps scored from shared vlm_results)."""
    if df is None or df.empty:
        return empty_checkpoint("Content Accuracy", 70)

    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=70, result=0, name="Content Accuracy")
    vlm_results = vlm_results or {}

    score_vlm_steps(checkpoint, vlm_results, [
        (1, "Opening Hours Accurate", "hours_", dest_names,
         "{passed}/{total} destinations have accurate opening hours",
         "No opening hours to verify"),
        (2, "Cost Accurate", "cost_", dest_names,
         "{passed}/{total} destinations have accurate cost",
         "No costs to verify"),
        (3, "Transportation Mode Available", "mode_", dest_names,
         "{passed}/{total} rows have available transportation mode",
         "No transportation modes to verify"),
        (4, "Travel Time Realistic", "ttime_", dest_names,
         "{passed}/{total} rows have realistic travel time",
         "No travel times to verify"),
        (5, "Cuisine Accurate", "cuisine_", dest_names,
         "{passed}/{total} restaurants have accurate cuisine type",
         "No cuisine data to verify"),
        (6, "Review Links Valid", "review_", dest_names,
         "{passed}/{total} review links contain reviews for the named place",
         "No review links to verify"),
        (7, "Alternatives Viable", "altv_", dest_names,
         "{passed}/{total} alternatives are viable substitutes within reachable distance",
         "No alternatives to verify"),
    ], reasons=vlm_reasons, seed_categories=vlm_seed_categories)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_5():
    """Checkpoint 5: Conditional Formatting and Styling (20 pts, 2 steps)."""
    if df is None or df.empty:
        return empty_checkpoint("Conditional Formatting and Styling", 20)

    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=20, result=0, name="Conditional Formatting and Styling")

    total_rows = len(df)
    sheet_tab = sheet_raw.get("sheets", [{}])[0]

    cost_col_idx = get_column_index_by_name(df, "Cost", matched_columns)
    alt_col_idx = get_column_index_by_name(df, "Alternative Option", matched_columns)

    # --- Step 1: Cost column color-coded correctly ---
    step_start = time.time()
    if cost_col_idx >= 0:
        correct, checked, failures = evaluate_cost_color_coding(
            sheet_raw, sheet_tab, row_data, header_row_idx, cost_col_idx, total_rows,
        )
        detail = f"{correct}/{checked} cost cells have correct color coding"
        detail += format_failures(failures)
        add_fraction_step(checkpoint, "Cost Color Coding", 1, correct, checked, detail, step_start,
                          category=StepCategory.FUZZY_MATCH)
    else:
        checkpoint.add_step(
            "Cost Color Coding", False, 1, details="Cost column not found",
            score=0, max_score=10, execution_time=time.time() - step_start,
            category=StepCategory.DEPENDENCY_NOT_EVALUATED,
        )

    # --- Step 2: Alternative Option text is blue ---
    step_start = time.time()
    if alt_col_idx >= 0:
        blue_count = non_empty = 0
        failures = []
        for idx in range(total_rows):
            cell = get_cell(sheet_tab, header_row_idx + 1 + idx, alt_col_idx)
            if not cell.get("formattedValue", "").strip():
                continue
            non_empty += 1

            fg = get_text_foreground_color(cell, sheet_raw)
            b_val = fg.get("blue", 0)
            if b_val > 0.4 and b_val > fg.get("red", 0) and b_val > fg.get("green", 0):
                blue_count += 1
            else:
                failures.append(f"Row {idx+1}")

        detail = f"{blue_count}/{non_empty} alternative option cells have blue text"
        detail += format_failures(failures, prefix="Not blue", sep=", ")
        add_fraction_step(
            checkpoint, "Alternative Options Blue Text", 2,
            blue_count, non_empty, detail, step_start,
            category=StepCategory.FUZZY_MATCH,
        )
    else:
        checkpoint.add_step(
            "Alternative Options Blue Text", False, 2,
            details="Alternative Option column not found", score=0, max_score=10,
            execution_time=time.time() - step_start,
            category=StepCategory.DEPENDENCY_NOT_EVALUATED,
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_6():
    """Checkpoint 6: Logical Scheduling and Route Planning (70 pts, 7 steps)."""
    if df is None or df.empty:
        return empty_checkpoint("Logical Scheduling and Route Planning", 70)

    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=70, result=0, name="Logical Scheduling and Route Planning")

    total_rows = len(df)
    city_name = trip_details.get("city", "the destination city")

    day_groups = {}
    for idx, rd in enumerate(row_data):
        if rd["date"]:
            day_groups.setdefault(rd["date"], []).append(idx)
    day_keys = list(day_groups.keys())
    total_days = len(day_keys)
    day_name_map = dict(enumerate(day_keys))

    # Steps 1 & 6: route order + day-aware transit (seeded + VLM fallback)
    cp6_seed_categories: dict = {}
    vlm_results, vlm_tasks, vlm_reasons = seed_cp6_route_and_daytype(
        day_keys, day_groups, row_data, trip_details.get("hotel", ""),
        places_out_of_city, pairwise_transit_cache, day_specific_transit_cache, city_name,
        day_return_transit_cache=day_return_transit_cache,
        seed_categories=cp6_seed_categories,
    )
    vlm_results.update(run_vlm_batch(vlm_tasks, model))

    score_vlm_steps(checkpoint, vlm_results, [
        (1, "Logical Route Order", "route_", day_name_map,
         "{passed}/{total} days have a logical route order",
         "No route data to verify"),
    ], reasons=vlm_reasons, seed_categories=cp6_seed_categories)

    # --- Step 2: Meal placement ---
    step_start = time.time()
    correct_days = 0
    failures = []
    for day, indices in day_groups.items():
        slots = [row_data[i]["classification"] for i in indices]
        tods = [row_data[i]["time_of_day"].lower() for i in indices]

        # Start False if the meal is missing entirely; flip to False if found but mis-placed.
        lunch_ok = any(s == "food" and "lunch" in t for s, t in zip(slots, tods))
        dinner_ok = any(s == "food" and "dinner" in t for s, t in zip(slots, tods))
        for i, (slot, tod) in enumerate(zip(slots, tods)):
            if slot == "food" and "lunch" in tod:
                if not any(s == "activity" and "morning" in t for s, t in zip(slots[:i], tods[:i])):
                    lunch_ok = False
            if slot == "food" and "dinner" in tod:
                if not any(s == "activity" and "afternoon" in t for s, t in zip(slots[:i], tods[:i])):
                    dinner_ok = False

        if lunch_ok and dinner_ok:
            correct_days += 1
        else:
            reasons = []
            if not lunch_ok:
                reasons.append("lunch missing or not after morning")
            if not dinner_ok:
                reasons.append("dinner missing or not after afternoon")
            failures.append(f"{day}: {', '.join(reasons)}")

    detail = f"{correct_days}/{total_days} days have correct meal placement"
    detail += format_failures(failures, limit=None)
    add_fraction_step(checkpoint, "Meal Placement", 2, correct_days, total_days, detail, step_start,
                      category=StepCategory.STRUCTURAL)

    # --- Step 3: Reasonable duration ---
    step_start = time.time()
    reasonable = checked = 0
    failures = []
    for rd in row_data:
        if rd["dur_min"] is None:
            continue
        checked += 1
        hours = rd["dur_min"] / 60
        ok = (0.5 <= hours <= 3.0) if rd.get("row_type") == "food" else (0.5 <= hours <= 5.0)
        if ok:
            reasonable += 1
        else:
            failures.append(f"{rd['dest_name']}: {hours:.1f}h")

    detail = f"{reasonable}/{checked} rows have reasonable duration"
    detail += format_failures(failures)
    add_fraction_step(checkpoint, "Reasonable Durations", 3, reasonable, checked, detail, step_start,
                      category=StepCategory.DETERMINISTIC)

    # --- Step 4: Start time feasible ---
    step_start = time.time()
    feasible = checked = 0
    failures = []
    for idx, rd in enumerate(row_data):
        if rd["start_min"] is None:
            continue
        checked += 1
        reasons = []

        tod_lower = rd["time_of_day"].lower()
        for key, (block_start, block_end) in TIME_SLOT_BLOCKS.items():
            if key in tod_lower:
                if not (block_start <= rd["start_min"] <= block_end):
                    reasons.append(f"outside {key} block")
                break

        if idx > 0 and row_data[idx - 1]["date"] == rd["date"]:
            prev = row_data[idx - 1]
            if (prev["depart_min"] is not None and rd["travel_min"] is not None
                    and rd["start_min"] < prev["depart_min"] + rd["travel_min"] - 5):
                reasons.append("overlaps previous event + travel")

        if not reasons:
            feasible += 1
        else:
            failures.append(f"{rd['dest_name']}: {', '.join(reasons)}")

    detail = f"{feasible}/{checked} rows have feasible start time"
    detail += format_failures(failures)
    add_fraction_step(checkpoint, "Start Time Feasible", 4, feasible, checked, detail, step_start,
                      category=StepCategory.DETERMINISTIC)

    # --- Step 5: Departure time feasible ---
    step_start = time.time()
    feasible = checked = 0
    failures = []
    for idx, rd in enumerate(row_data):
        if rd["depart_min"] is None or rd["start_min"] is None:
            continue
        checked += 1
        reasons = []

        if rd["depart_min"] <= rd["start_min"]:
            reasons.append("departure <= start")

        if rd["dur_min"] is not None:
            gap = rd["depart_min"] - rd["start_min"]
            if abs(gap - rd["dur_min"]) > 30:
                reasons.append(f"gap {gap}min vs duration {rd['dur_min']:.0f}min")

        if idx < total_rows - 1 and row_data[idx + 1]["date"] == rd["date"]:
            next_rd = row_data[idx + 1]
            if next_rd["start_min"] is not None and rd["depart_min"] > next_rd["start_min"]:
                reasons.append("overlaps next event")

        if not reasons:
            feasible += 1
        else:
            failures.append(f"{rd['dest_name']}: {', '.join(reasons)}")

    detail = f"{feasible}/{checked} rows have feasible departure time"
    detail += format_failures(failures)
    add_fraction_step(checkpoint, "Departure Time Feasible", 5, feasible, checked, detail, step_start,
                      category=StepCategory.DETERMINISTIC)

    # Step 6: Weekday/weekend (scored from shared vlm_results built above)
    score_vlm_steps(checkpoint, vlm_results, [
        (6, "Day-Aware Transit Accuracy", "daytype_", day_name_map,
         "{passed}/{total} days have transit times matching the real schedule for that date",
         "No day-aware transit data to verify"),
    ], reasons=vlm_reasons, seed_categories=cp6_seed_categories)

    # --- Step 7: Transit time limits (includes return-to-hotel leg) ---
    step_start = time.time()
    correct_days = 0
    failures = []
    for day, indices in day_groups.items():
        travel_times = [row_data[i]["travel_min"] for i in indices if row_data[i]["travel_min"] is not None]
        if not travel_times:
            failures.append(f"{day}: no travel time data")
            continue

        # Include return-to-hotel leg if available
        return_min = day_return_transit_cache.get(indices[-1])
        all_legs = list(travel_times) + ([return_min] if return_min is not None else [])

        reasons = []
        if max(all_legs) > 30:
            over_30 = [t for t in all_legs if t > 30]
            if return_min is not None and return_min > 30:
                reasons.append(f"return leg {return_min} min > 30")
            if max(travel_times) > 30:
                reasons.append(f"max leg {max(travel_times)} min > 30")
        if sum(all_legs) > 90:
            reasons.append(f"total {sum(all_legs)} min > 90")

        if not reasons:
            correct_days += 1
        else:
            failures.append(f"{day}: {', '.join(reasons)}")

    detail = f"{correct_days}/{total_days} days within transit time limits"
    detail += format_failures(failures, limit=None)
    add_fraction_step(checkpoint, "Transit Time Limits", 7, correct_days, total_days, detail, step_start,
                      category=StepCategory.DETERMINISTIC)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(
    workspace_doc_id: str = None,
    cached_models=None,
    browsing_history: List[str] = None,
):
    """Run setup, build the shared CP3+CP4 VLM results, then score all six checkpoints."""
    total_start = time.time()

    try:
        setup(workspace_doc_id)
    except Exception as e:
        print(f"Setup failed: {e}")
        cp = Checkpoint(total=40, result=0, name="Table Structure and Layout")
        cp.add_step("Setup", False, 1, f"Setup failed: {e}", score=0, max_score=10,
                    category=StepCategory.EXECUTION_ERROR)
        return Result(checkpoints=[cp], total_execution_time=time.time() - total_start)

    cp34_vlm_results: dict = {}
    cp34_vlm_reasons: dict = {}
    cp34_vlm_seed_categories: dict = {}
    if df is not None and not df.empty:
        try:
            cp34_vlm_results, cp34_vlm_reasons = build_cp3_cp4_vlm_results(
            ctx={
                "df": df,
                "matched_columns": matched_columns,
                "dest_names": dest_names,
                "all_dest_names": all_dest_names,
                "all_alt_names": all_alt_names,
                "all_alt_types": all_alt_types,
                "row_data": row_data,
                "place_cache": place_cache,
                "places_out_of_city": places_out_of_city,
                "directions_cache": directions_cache,
                "alt_directions_cache": alt_directions_cache,
                "url_content": url_content,
                "all_review_urls": all_review_urls,
                "trip_details": trip_details,
                "url_geocode_cache": url_geocode_cache,
                "maps_api_key": maps_api_key,
            },
            model=model,
            seed_categories=cp34_vlm_seed_categories,
        )
        except Exception as e:
            print(f"Error in build_cp3_cp4_vlm_results: {e}")

    return Result(
        checkpoints=[
            grade_checkpoint_1(),
            grade_checkpoint_2(),
            grade_checkpoint_3(cp34_vlm_results, cp34_vlm_reasons, cp34_vlm_seed_categories),
            grade_checkpoint_4(cp34_vlm_results, cp34_vlm_reasons, cp34_vlm_seed_categories),
            grade_checkpoint_5(),
            grade_checkpoint_6(),
        ],
        total_execution_time=time.time() - total_start,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate personal travel planner spreadsheet")
    parser.add_argument(
        "--workspace_doc_id", type=str, required=True,
        help="Google Sheets document ID to evaluate",
    )
    args = parser.parse_args()

    start_time = time.time()
    result = grade_checkpoints(workspace_doc_id=args.workspace_doc_id)

    report = result.get_detailed_report()
    print("=== EVALUATION RESULTS ===")
    print(f"Final Score: {report['final_score']}")
    print(f"Total time: {time.time() - start_time:.2f}s")

    for cp in report["checkpoints"]:
        print(f"\n=== {cp['name']} ({cp['score']}) ===")
        for step in cp["steps"]:
            status = "PASS" if step["success"] else "FAIL"
            print(f"  [{status}] {step['name']} ({step['score']}/{step['max_score']}): {step['details']}")
