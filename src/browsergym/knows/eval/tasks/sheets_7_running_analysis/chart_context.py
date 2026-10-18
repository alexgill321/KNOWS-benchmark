"""Task-local chart access with explicit source-tab and host-tab identities."""
from copy import deepcopy
from types import FunctionType

from browsergym.knows.eval.eval_utils import chart_utils as _shared


class ChartContextError(RuntimeError):
    """Saved sheet metadata does not establish a chart's evaluation context."""


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ChartContextError(f"Invalid {label}")
    return value


def _sheet_rows(sheet):
    """Reconstruct absolute cell coordinates without dropping GridData offsets."""
    blocks = sheet.get('data', [])
    if not isinstance(blocks, list):
        raise ChartContextError('Invalid sheet GridData')
    rows, occupied = [], set()
    grid = sheet['properties']['gridProperties']
    for block in blocks:
        first_row = _integer(block.get('startRow', 0), 'GridData start row')
        first_col = _integer(block.get('startColumn', 0), 'GridData start column')
        data = block.get('rowData', [])
        if not isinstance(data, list) or first_row + len(data) > grid['rowCount']:
            raise ChartContextError('GridData rows exceed identified sheet')
        for offset, row in enumerate(data):
            index = first_row + offset
            values = row.get('values', [])
            if not isinstance(values, list) or first_col + len(values) > grid['columnCount']:
                raise ChartContextError('GridData columns exceed identified sheet')
            while len(rows) <= index:
                rows.append({'values': []})
            dest = rows[index]['values']
            for col, cell in enumerate(values, first_col):
                if (index, col) in occupied or not isinstance(cell, dict):
                    raise ChartContextError('Overlapping or malformed GridData cells')
                occupied.add((index, col))
                while len(dest) <= col:
                    dest.append({})
                dest[col] = cell
    return rows


def bind_chart_context(charts, sheet_raw, table_sheet_id):
    """Bind copied chart objects to one verified, immutable-in-memory sheet read."""
    if not charts:
        return []
    if not isinstance(sheet_raw, dict) or not isinstance(sheet_raw.get('sheets'), list):
        raise ChartContextError('Whole-sheet chart source data unavailable')
    sheets, titles = {}, set()
    for sheet in sheet_raw['sheets']:
        props = sheet.get('properties', {})
        sheet_id = _integer(props.get('sheetId'), 'sheet identity')
        title, grid = props.get('title'), props.get('gridProperties')
        if sheet_id in sheets or not isinstance(title, str) or not title or title in titles:
            raise ChartContextError('Ambiguous source-sheet identity')
        if not isinstance(grid, dict) or any(type(grid.get(k)) is not int or grid[k] <= 0
                                            for k in ('rowCount', 'columnCount')):
            raise ChartContextError('Source-sheet grid dimensions unavailable')
        sheets[sheet_id] = {'title': title, 'grid': grid, 'rows': _sheet_rows(sheet)}
        titles.add(title)
    if _integer(table_sheet_id, 'table sheet identity') not in sheets:
        raise ChartContextError('Table sheet is not in captured metadata')
    result = deepcopy(charts)
    for chart in result:
        host = _integer(chart.get('sheet_id'), 'chart host identity')
        if host not in sheets or chart.get('sheet_name') != sheets[host]['title']:
            raise ChartContextError('Chart host does not match sheet metadata')
        chart['_running_context'] = {'sheets': sheets, 'table_sheet_id': table_sheet_id}
        _bind_overlay_anchor(chart)
        series = chart.get('raw_chart', {}).get('spec', {}).get('basicChart', {}).get('series', [])
        if not isinstance(series, list):
            raise ChartContextError('Malformed chart series list')
        # Validate before any parallel series-identification call can catch an error.
        for index in range(len(series)):
            _source(chart, index)
    return result


def _bind_overlay_anchor(chart):
    """Recover an omitted zero row only from a verified raw overlay anchor."""
    position = chart.get('position', {})
    if position.get('type') != 'overlay':
        return
    raw = chart.get('raw_chart', {}).get('position', {}).get('overlayPosition')
    if not isinstance(raw, dict) or not isinstance(raw.get('anchorCell'), dict):
        raise ChartContextError('Raw overlay anchor unavailable')
    anchor = raw['anchorCell']
    host = chart['sheet_id']
    # Missing zero-valued sheet IDs are valid only for the already verified host 0.
    if _integer(anchor.get('sheetId', 0), 'overlay anchor sheet identity') != host:
        raise ChartContextError('Overlay anchor does not match verified chart host')
    row = _integer(anchor.get('rowIndex', 0), 'overlay anchor row')
    # Validate an absent zero column, but preserve its parsed representation;
    # this repair changes only the proven omitted-row handling.
    col = _integer(anchor.get('columnIndex', 0), 'overlay anchor column')
    grid = _context(chart)['sheets'][host]['grid']
    if row >= grid['rowCount'] or col >= grid['columnCount']:
        raise ChartContextError('Overlay anchor exceeds verified host grid')
    parsed = position.get('anchor_cell')
    expected_col = anchor.get('columnIndex')
    if (not isinstance(parsed, dict)
            or (expected_col is not None and type(parsed.get('col')) is not int)
            or parsed.get('col') != expected_col):
        raise ChartContextError('Parsed overlay anchor differs from raw evidence')
    expected = anchor.get('rowIndex')
    if (expected is not None and type(parsed.get('row')) is not int) or parsed.get('row') != expected:
        raise ChartContextError('Parsed overlay row differs from raw evidence')
    if 'rowIndex' not in anchor:
        parsed['row'] = row


def _context(chart):
    context = chart.get('_running_context')
    if not isinstance(context, dict):
        raise ChartContextError('Chart has no verified sheet context')
    return context


def chart_host_rows(chart):
    context = _context(chart)
    host = _integer(chart.get('sheet_id'), 'chart host identity')
    if host not in context['sheets']:
        raise ChartContextError('Chart host data unavailable')
    return context['sheets'][host]['rows']


def _source(chart, index):
    context = _context(chart)
    config = chart.get('raw_chart', {}).get('spec', {}).get('basicChart', {})
    series = config.get('series', [])
    if type(index) is not int or not 0 <= index < len(series):
        raise ChartContextError('Chart series index is invalid')
    sources = series[index].get('series', {}).get('sourceRange', {}).get('sources', [])
    if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], dict):
        raise ChartContextError('One unambiguous chart series source is required')
    source = sources[0]
    # Absent zero-valued IDs are accepted only against actual metadata ID 0.
    sheet_id = _integer(source.get('sheetId', 0), 'series source sheet identity')
    if sheet_id not in context['sheets']:
        raise ChartContextError('Unknown chart series source sheet')
    sheet = context['sheets'][sheet_id]
    grid = sheet['grid']
    start = _integer(source.get('startRowIndex', 0), 'source start row')
    end = _integer(source.get('endRowIndex', grid['rowCount']), 'source end row')
    col = _integer(source.get('startColumnIndex', 0), 'source start column')
    last = _integer(source.get('endColumnIndex', grid['columnCount']), 'source end column')
    # Preserve the existing one-header convention when Google omits its auto count.
    headers = _integer(config.get('headerCount', 1), 'chart header count')
    if not 0 <= start < end <= grid['rowCount'] or not 0 <= col < last <= grid['columnCount'] or start + headers > end:
        raise ChartContextError('Chart source coordinates exceed identified sheet')
    if last != col + 1:
        raise ChartContextError('A chart series must identify one source column')
    return sheet_id, sheet['rows'], start, end, col, headers


def series_source_is_table(chart, index):
    return _source(chart, index)[0] == _context(chart)['table_sheet_id']


def get_series_header_label(chart, series_index, rows):
    _, source_rows, start, _, col, headers = _source(chart, series_index)
    if headers == 0 or start >= len(source_rows):
        return ''
    values = source_rows[start].get('values', [])
    return values[col].get('formattedValue', '') if col < len(values) else ''


def get_series_column_values(chart, series_index, rows):
    """Keep the original numeric/time parser, supplying only verified source cells."""
    _, source_rows, start, end, col, headers = _source(chart, series_index)
    cells = []
    for row in source_rows[start + headers:end]:
        values = row.get('values', [])
        cells.append({'values': [values[col] if col < len(values) else {}]})
    normalized = deepcopy(chart['raw_chart'])
    normalized['spec']['basicChart']['series'][series_index]['series']['sourceRange']['sources'] = [
        {'startRowIndex': 0, 'endRowIndex': len(cells) + 1, 'startColumnIndex': 0, 'endColumnIndex': 1}]
    return _shared.get_series_column_values({'raw_chart': normalized}, series_index, [{'values': []}] + cells)


def _matching_source_range(chart, index):
    sheet_id, _, start, end, col, _ = _source(chart, index)
    return {'start_row': start, 'end_row': end,
            'start_col': col if sheet_id == _context(chart)['table_sheet_id'] else None,
            'end_col': col + 1}


def _isolated(function, replacements):
    """Use unchanged identification code with task-local readers, never patch globals."""
    result = FunctionType(function.__code__, {**function.__globals__, **replacements},
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


_identify = _isolated(_shared.identify_series_by_content, {
    'get_series_header_label': get_series_header_label,
    'get_series_column_values': get_series_column_values,
    'get_series_source_range': _matching_source_range,
})


def identify_series_by_content(chart, rows, *args, **kwargs):
    # The original routine gates reading on rows being nonempty. A chart may
    # legitimately source another tab even when its host has no cell data.
    _context(chart)
    return _identify(chart, chart_host_rows(chart) or [{'values': []}], *args, **kwargs)


def check_chart_overlap(chart, table_start_row, table_end_row, other_charts,
                        table_start_col=0, table_end_col=None):
    """Compare coordinates only inside the same actual sheet tab."""
    context = _context(chart)
    host = chart['sheet_id']
    same_sheet = [other for other in other_charts if other.get('sheet_id') == host]
    if host != context['table_sheet_id']:
        # The selected data table is on another tab; it cannot overlap this chart.
        table_start_row = table_end_row = table_start_col = table_end_col = 0
    return _shared.check_chart_overlap(chart, table_start_row, table_end_row,
                                       same_sheet, table_start_col, table_end_col)
