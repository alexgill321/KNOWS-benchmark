"""Read investment-chart cells using their absolute source-sheet coordinates."""

from browsergym.knows.eval.eval_utils.chart_utils import _parse_numeric_cell


class ChartSourceDataError(RuntimeError):
    """Chart source identity or fetched data cannot be verified for evaluation."""


def _source(chart, role):
    if not isinstance(chart, dict) or not isinstance(chart.get("raw_chart"), dict):
        raise ChartSourceDataError("Raw chart metadata is unavailable")
    spec = chart.get("raw_chart", {}).get("spec", {})
    if not isinstance(spec, dict):
        raise ChartSourceDataError("Chart specification is malformed")
    if "basicChart" in spec:
        config = spec["basicChart"]
        if not isinstance(config, dict):
            raise ChartSourceDataError("Basic chart specification is malformed")
        entries = config.get("domains" if role == "domain" else "series", [])
        if not isinstance(entries, list) or (entries and not isinstance(entries[0], dict)):
            raise ChartSourceDataError("Chart source entries are malformed")
        data = entries[0].get(role, {}) if entries else {}
        # An omitted basic-chart count is automatic in Sheets, not necessarily 1.
        header_count = config.get("headerCount")
    elif "pieChart" in spec:
        config = spec["pieChart"]
        if not isinstance(config, dict):
            raise ChartSourceDataError("Pie chart specification is malformed")
        data = config.get(role, {})
        # PieChartSpec has no headerCount field. Keep the pre-existing investment
        # evaluator's one-header convention; do not add a new inference rule.
        header_count = 1
    else:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("sourceRange"), dict):
        raise ChartSourceDataError("Chart source range is unavailable")
    sources = data["sourceRange"].get("sources", [])
    if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], dict):
        raise ChartSourceDataError("A single unambiguous chart source range is required")
    source = dict(sources[0])
    # ProtoJSON can omit a default-valued int32. Normalize absence only; the
    # fetch below still requires an actual, unique metadata sheet with ID 0.
    # Never substitute the chart's host tab, or accept null/bool/string IDs.
    if "sheetId" not in source:
        source["sheetId"] = 0
    if type(source.get("sheetId")) is not int or source["sheetId"] < 0:
        raise ChartSourceDataError("Chart source sheet identity is missing")
    if type(header_count) is not int or header_count < 0:
        raise ChartSourceDataError("Chart header count is not an explicit nonnegative integer")
    return source, header_count


def _bounds(source, header_count, sheet):
    grid = sheet["grid"]
    start_row, start_col = source.get("startRowIndex", 0), source.get("startColumnIndex", 0)
    end_row = source.get("endRowIndex", grid["rowCount"])
    end_col = source.get("endColumnIndex", grid["columnCount"])
    if (any(type(n) is not int for n in (start_row, start_col, end_row, end_col))
            or not 0 <= start_row < end_row <= grid["rowCount"]
            or not 0 <= start_col < end_col <= grid["columnCount"]
            or start_row + header_count > end_row):
        raise ChartSourceDataError("Chart source coordinates do not fit the identified sheet")
    return start_row + header_count, end_row, start_col, end_col


def fetch_chart_source_data(spreadsheet_id, service, charts, chart_type=None):
    """Fetch formatted rows from the chart's source tabs, never a guessed tab.

    Whole-tab reads begin at A1, preserving absolute row/column offsets and the
    existing formatted-number parser. Missing data/fetch errors must propagate.
    """
    if chart_type is not None:
        accepted = {"BAR", "COLUMN"} if chart_type == "BAR" else {chart_type}
        charts = [chart for chart in charts if chart.get("chart_type", "").upper() in accepted][:1]
    selected = [parsed for chart in charts for role in ("domain", "series")
                if (parsed := _source(chart, role)) is not None]
    if not selected:
        return {}
    try:
        metadata = service.spreadsheets().get(
            spreadsheetId=spreadsheet_id,
            fields="sheets(properties(sheetId,title,gridProperties(rowCount,columnCount)))",
        ).execute()
        sheets = {}
        titles = set()
        for item in metadata["sheets"]:
            props = item["properties"]
            sheet_id, title, grid = props["sheetId"], props["title"], props.get("gridProperties")
            if (type(sheet_id) is not int or sheet_id < 0 or sheet_id in sheets
                    or not isinstance(title, str) or not title or title in titles):
                raise ChartSourceDataError("Sheet metadata is missing or ambiguous")
            sheets[sheet_id] = {"title": title, "grid": grid}
            titles.add(title)
        needed = {source["sheetId"] for source, _ in selected}
        if not needed.issubset(sheets):
            raise ChartSourceDataError("Chart references an unknown source sheet")
        for sheet_id in needed:
            grid = sheets[sheet_id]["grid"]
            if (not isinstance(grid, dict) or any(type(grid.get(k)) is not int or grid[k] <= 0
                                                for k in ("rowCount", "columnCount"))):
                raise ChartSourceDataError("Source-sheet dimensions are unavailable")
        for source, headers in selected:
            _bounds(source, headers, sheets[source["sheetId"]])
        for sheet_id in sorted(needed):
            sheet = sheets[sheet_id]
            quoted_title = "'" + sheet["title"].replace("'", "''") + "'"
            result = service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id, range=quoted_title,
                majorDimension="ROWS", valueRenderOption="FORMATTED_VALUE",
            ).execute()
            returned_tab, separator, returned_cells = result.get("range", "").rpartition("!")
            if (not separator or returned_tab not in (sheet["title"], quoted_title)
                    or returned_cells.split(":", 1)[0] != "A1"):
                raise ChartSourceDataError("Chart values response does not identify the requested whole tab")
            values = result.get("values", [])
            if (result.get("majorDimension", "ROWS") != "ROWS" or not isinstance(values, list)
                    or len(values) > sheet["grid"]["rowCount"]
                    or any(not isinstance(row, list) or len(row) > sheet["grid"]["columnCount"]
                           for row in values)):
                raise ChartSourceDataError("Chart source values are not a complete row-oriented response")
            sheet["values"] = values
        return {sheet_id: sheets[sheet_id] for sheet_id in needed}
    except ChartSourceDataError:
        raise
    except Exception as exc:
        raise ChartSourceDataError("Could not fetch verified chart source sheet data") from exc


def _rows(chart, role, sources):
    parsed = _source(chart, role)
    if parsed is None:
        return []
    source, headers = parsed
    if not isinstance(sources, dict) or source["sheetId"] not in sources:
        raise ChartSourceDataError("Fetched chart source sheet is unavailable")
    sheet = sources[source["sheetId"]]
    start, end, first_col, last_col = _bounds(source, headers, sheet)
    return [row[first_col:last_col] for row in sheet["values"][start:end]]


def extract_chart_domain_data(chart, table_data, full_sheet_data=None):
    """Read labels from the source range; the stock table has unrelated offsets."""
    values = []
    for row in _rows(chart, "domain", full_sheet_data):
        parts = [str(value).strip() for value in row
                 if value is not None and str(value).strip() and str(value).lower() != "nan"]
        if parts:
            values.append(" ".join(parts))
    return values


def extract_chart_series_data(chart, table_data, full_sheet_data=None):
    """Keep existing numeric parsing, using only the specified source cells."""
    return [number for row in _rows(chart, "series", full_sheet_data) for value in row
            if (number := _parse_numeric_cell(value)) is not None]
