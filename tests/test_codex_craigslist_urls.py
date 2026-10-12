"""Offline checks for actual Craigslist listing routes and evaluator dispatch."""
import ast
from pathlib import Path
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import unittest
from urllib.parse import urlsplit
from unittest.mock import Mock

from bs4 import BeautifulSoup
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / 'src/browsergym/knows/eval/tasks/sheets_38_apartment_finder'
SOURCE = FAMILY / 'utils.py'
SAVED_URLS = [
    'https://www.craigslist.org/view/d/salt-lake-city-downtown-1bd-1ba-730/oAGk5h3iJAc2FREtFZRZVj',
    'https://www.craigslist.org/view/d/salt-lake-city-one-bedroom-on-bath/gvGawyp6r3wmuu9t2gPhbd',
    'https://www.craigslist.org/view/d/salt-lake-city-extra-storage-private/obL6oBRq5dzKVmaTB3tkUV',
    'https://www.craigslist.org/view/d/salt-lake-city-tmlp-12-mo-chill-by-the/9sKfibLkx9e8R1aLpwrhrN',
    'https://www.craigslist.org/view/d/west-valley-city-barbecue-areas-laundry/kdP7RQrY5w3imyD8qjWqwR',
]


def definitions(path, names, namespace):
    # Importing the full evaluator initializes Google services. Load its real
    # function definitions only, with all I/O dependencies explicitly stubbed.
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


namespace = dict(re=re, urlsplit=urlsplit, BeautifulSoup=BeautifulSoup,
                 Optional=Optional, Dict=Dict, Any=Any)
definitions(SOURCE, {'is_valid_craigslist_url', 'extract_craigslist_data_structured',
                     '_listing_cell_link_target', 'resolve_listing_url'}, namespace)
valid = namespace['is_valid_craigslist_url']
extract = namespace['extract_craigslist_data_structured']
resolve = namespace['resolve_listing_url']
SHORT_URLS = [f'https://www.craigslist.org/view/{identifier}' for identifier in
              ('1qXHwW-18RGMIrvGTup3Hw', '2Odv8Rus8RG_zfTpYJpXFw', 'fBACdL-p8RG2XOm16vvQyg')]


def grid(rows, **offsets):
    return {'sheets': [{'data': [dict(rowData=[{'values': row} for row in rows], **offsets)]}]}


def cell(label, **fields):
    return dict(formattedValue=label, **fields)


class Checkpoint:
    def __init__(self, **kwargs):
        self.steps = []

    def add_step(self, name, success, step_id, details, **kwargs):
        self.steps.append(dict(name=name, success=success, details=details, **kwargs))


class Categories:
    def __getattr__(self, name):
        return name


def execute(tasks, **kwargs):
    return {t['id']: t['func'](*t.get('args', ())) for t in tasks}


class CraigslistURLTests(unittest.TestCase):
    def test_three_saved_short_routes_are_valid_without_admitting_other_paths(self):
        for url in SHORT_URLS:
            self.assertTrue(valid(url))
            self.assertFalse(valid(url.replace('www.craigslist.org', 'craigslist.org.evil.example')))
        for url in ('https://www.craigslist.org/view/d', 'https://www.craigslist.org/view/search',
                    SHORT_URLS[0] + '/extra', SHORT_URLS[0][:-1],
                    'https://www.craigslist.org/?next=/view/1qXHwW-18RGMIrvGTup3Hw'):
            self.assertFalse(valid(url))

    def test_all_five_saved_raw_listing_urls_are_accepted(self):
        for url in SAVED_URLS:
            with self.subTest(url=url):
                self.assertTrue(valid(url))

    def test_legacy_housing_paths_remain_accepted(self):
        for category in ('apa', 'sub', 'hsw', 'hou', 'roo'):
            for scheme in ('https', 'http'):
                with self.subTest(category=category, scheme=scheme):
                    self.assertTrue(valid(f'{scheme}://saltlakecity.craigslist.org/{category}/d/listing/1234567890.html'))
                    self.assertTrue(valid(f'{scheme}://losangeles.craigslist.org/wst/{category}/d/listing/123.html'))

    def test_valid_route_retains_query_fragment_and_outer_whitespace(self):
        self.assertTrue(valid('  ' + SAVED_URLS[0] + '?lang=en#map  '))

    def test_fake_hosts_credentials_ports_and_embedded_urls_are_rejected(self):
        for value in (
            SAVED_URLS[0].replace('www.craigslist.org', 'craigslist.org.evil.example'),
            SAVED_URLS[0].replace('www.craigslist.org', 'notcraigslist.org'),
            SAVED_URLS[0].replace('www.craigslist.org', 'craigslist.org@evil.example'),
            SAVED_URLS[0].replace('www.craigslist.org', 'user@www.craigslist.org'),
            SAVED_URLS[0].replace('www.craigslist.org', 'www.craigslist.org:8080'),
            'https://evil.example/apa/?url=https://craigslist.org',
            'https://evil.example/#https://craigslist.org/apa/listing',
            'https://www.craigslist.org.evil.example/apa/d/listing/123.html',
            'https://www.craigslist.org\\@evil.example/apa/d/listing/123.html',
            'https://www.craigslist.org:/view/d/slug/id' + '\nextra',
        ):
            with self.subTest(value=value):
                self.assertFalse(valid(value))

    def test_malformed_or_nonlisting_paths_do_not_become_valid(self):
        for value in (None, 7, '', 'Craigslist listing',
                      'https://www.craigslist.org/area/saltlakecity',
                      'https://www.craigslist.org/search/apa',
                      'https://www.craigslist.org/?next=/apa/',
                      'https://www.craigslist.org/#/apa/',
                      'https://www.craigslist.org/view/d/slug',
                      'https://www.craigslist.org/view/d//opaqueid',
                      'https://www.craigslist.org/view/d/slug/',
                      'https://www.craigslist.org/view/d/slug/opaque/id',
                      'https://www.craigslist.org/view/d/slug/../id',
                      'https://www.craigslist.org/view/d/slug/%2F',
                      'file://www.craigslist.org/apa/d/x',
                      'https://[invalid/apa/d/x'):
            with self.subTest(value=value):
                self.assertFalse(valid(value))

    def test_current_listing_dom_is_supported_by_unchanged_structured_parser(self):
        html = '''<span class="price">$1,011</span>
        <p class="attrgroup"><span>1BR / 1Ba</span><span>730ft2</span></p>
        <div class="mapaddress">360 S 200 W</div>
        <p class="attrgroup"><span>cats are OK - purrr</span><span>dogs are OK - wooof</span>
        <span>w/d in unit</span><span>off-street parking</span></p>
        <section id="postingbody">A one-bedroom apartment. Application fee: $50.</section>'''
        data = extract(html)
        self.assertEqual({k: data[k] for k in ('price', 'bedrooms', 'bathrooms', 'sqft',
                                               'address', 'in_unit_laundry', 'pet_friendly')},
                         dict(price=1011, bedrooms=1, bathrooms=1, sqft=730,
                              address='360 S 200 W', in_unit_laundry='Yes', pet_friendly='Yes'))
        self.assertIsNone(extract('<html><body>Service temporarily unavailable</body></html>'))

    def checkpoint(self, instance, fetched_html):
        fetch = Mock(return_value=fetched_html)
        extract_mock = Mock(return_value=None)
        scope = dict(time=time, re=re, Checkpoint=Checkpoint, StepCategory=Categories(),
                     model=object(), matched_columns={'Listing URL': 'URL'},
                     df=pd.DataFrame({'URL': [SAVED_URLS[0]]}),
                     sheet_raw=None, resolve_listing_url=resolve,
                     is_valid_craigslist_url=valid, fetch_craigslist_page=fetch,
                     extract_craigslist_data_with_fallback=extract_mock,
                     parallel_download=execute, parallel_execute=execute)
        definitions(FAMILY / f'instance_{instance}/evaluator.py', {'grade_checkpoint_2'}, scope)
        return scope['grade_checkpoint_2'](), fetch, extract_mock

    def test_all_five_real_evaluators_dispatch_view_routes_to_existing_fetch(self):
        for instance in range(1, 6):
            with self.subTest(instance=instance):
                checkpoint, fetch, extractor = self.checkpoint(instance, '<html>fixture</html>')
                fetch.assert_called_once_with(SAVED_URLS[0], True)
                extractor.assert_called_once()
                self.assertTrue(checkpoint.steps[0]['success'])
                # URL admission alone must never invent listing metadata credit.
                self.assertFalse(any(s['success'] for s in checkpoint.steps[1:]))

    def test_all_five_real_evaluators_keep_failed_fetch_as_execution_error(self):
        for instance in range(1, 6):
            with self.subTest(instance=instance):
                checkpoint, fetch, extractor = self.checkpoint(instance, None)
                fetch.assert_called_once_with(SAVED_URLS[0], True)
                extractor.assert_not_called()
                self.assertFalse(checkpoint.steps[0]['success'])
                self.assertEqual(checkpoint.steps[0]['category'], 'EXECUTION_ERROR')
                self.assertFalse(any(s['success'] for s in checkpoint.steps))


class ListingHyperlinkTests(unittest.TestCase):
    def row(self, label='View First Hill listing'):
        return pd.Series(['First Hill', '1645', label], index=['Listing', 'Rent', 'Link'])

    def raw(self, fields=None, label='View First Hill listing', **offsets):
        return grid([[cell('First Hill'), cell('1645'), cell(label, **(fields or {}))]], **offsets)

    def test_literal_formula_target_requires_exact_decoded_display_label(self):
        for separator in (',', ';'):
            formula = f'=HYPERLINK("{SAVED_URLS[0]}"{separator}"View First Hill listing")'
            self.assertEqual(resolve(self.raw({'userEnteredValue': {'formulaValue': formula}}),
                                     self.row(), 'Link'), SAVED_URLS[0])
        label = 'View "First Hill" listing'
        formula = f'=HYPERLINK("{SAVED_URLS[0]}","View ""First Hill"" listing")'
        self.assertEqual(resolve(self.raw({'userEnteredValue': {'formulaValue': formula}}, label),
                                 self.row(label), 'Link'), SAVED_URLS[0])

    def test_sheet_hyperlink_rich_text_and_spill_metadata_use_actual_target(self):
        for fields in ({'hyperlink': SHORT_URLS[0]},
                       {'effectiveFormat': {'textFormat': {'link': {'uri': SHORT_URLS[0]}}}},
                       {'userEnteredFormat': {'textFormat': {'link': {'uri': SHORT_URLS[0]}}}},
                       {'textFormatRuns': [{'format': {'link': {'uri': SHORT_URLS[0]}}}]}):
            with self.subTest(fields=fields):
                self.assertEqual(resolve(self.raw(fields), self.row(), 'Link'), SHORT_URLS[0])

    def test_offsets_blank_rows_and_other_tabs_preserve_exact_row_mapping(self):
        raw = self.raw({'hyperlink': SAVED_URLS[0]}, startRow=8, startColumn=3)
        raw['sheets'].insert(0, {'data': [{'rowData': [{}, {'values': [cell('Other row')]}]}]})
        self.assertEqual(resolve(raw, self.row(), 'Link'), SAVED_URLS[0])
        raw['sheets'][1]['data'][0]['rowData'].insert(0, {})
        self.assertEqual(resolve(raw, self.row(), 'Link'), SAVED_URLS[0])

    def test_real_table_extraction_preserves_formatted_numbers_and_blank_strings(self):
        headers = ['Listing', 'Rent', 'Sq ft', 'Link']
        cells = [cell('First Hill'), cell('$1,645.00', effectiveValue={'numberValue': 1645}),
                 {}, cell('View listing', hyperlink=SAVED_URLS[0])]
        raw = grid([[], [], [{}, {}] + [cell(h) for h in headers] + [{}, {}, {}, cell('Statistic'), cell('Value')],
                    [{}, {}] + cells + [{}, {}, {}, cell('Average'), cell('$1,645.00')]])
        raw['sheets'][0]['tables'] = [
            {'range': {'startRowIndex': 2, 'endRowIndex': 4, 'startColumnIndex': 2, 'endColumnIndex': 6},
             'columnProperties': [{'columnName': h} for h in headers]},
            {'range': {'startRowIndex': 2, 'endRowIndex': 4, 'startColumnIndex': 9, 'endColumnIndex': 11},
             'columnProperties': [{'columnName': h} for h in ('Statistic', 'Value')]},
        ]
        scope = dict(pd=pd, List=List, SheetTable=SimpleNamespace,
                     get_sheet_content=Mock(return_value=raw))
        definitions(FAMILY.parents[1] / 'eval_utils/google_sheets_utils.py',
                    {'extract_tables_from_sheet'}, scope)
        tables = scope['extract_tables_from_sheet']('fixture', object())
        self.assertEqual(len(tables), 2)
        row = tables[0].df.iloc[0]
        self.assertEqual(list(row), ['First Hill', '$1,645.00', '', 'View listing'])
        self.assertTrue(all(isinstance(value, str) for value in row))
        self.assertFalse(row.isna().any())
        self.assertEqual(resolve(raw, row, 'Link'), SAVED_URLS[0])
        self.assertEqual(resolve(raw, tables[1].df.iloc[0], 'Value'), '$1,645.00')

    def test_duplicate_displayed_rows_and_multiple_links_fail_closed(self):
        raw = self.raw({'hyperlink': SAVED_URLS[0]})
        raw['sheets'].extend(self.raw({'hyperlink': SAVED_URLS[1]})['sheets'])
        self.assertEqual(resolve(raw, self.row(), 'Link'), 'View First Hill listing')
        ambiguous = self.raw({'textFormatRuns': [{'format': {'link': {'uri': u}}} for u in SAVED_URLS[:2]]})
        self.assertEqual(resolve(ambiguous, self.row(), 'Link'), '')

    def test_error_cells_reference_formulas_and_failed_arrays_do_not_infer_links(self):
        for formula in (f'=HYPERLINK(A1,"View First Hill listing")',
                        f'=IF(TRUE,HYPERLINK("{SAVED_URLS[0]}","View First Hill listing"),"")',
                        f'=HYPERLINK("{SAVED_URLS[0]}","Different label")',
                        f'={{HYPERLINK("{SAVED_URLS[0]}","View First Hill listing")}}'):
            raw = self.raw({'userEnteredValue': {'formulaValue': formula}})
            self.assertEqual(resolve(raw, self.row(), 'Link'), 'View First Hill listing')
        raw = self.raw({'hyperlink': SAVED_URLS[0], 'effectiveValue': {'errorValue': {'type': 'REF'}}})
        self.assertEqual(resolve(raw, self.row(), 'Link'), '')

    def test_missing_or_unmatched_data_and_wrong_column_keep_original_fallback(self):
        self.assertEqual(resolve(None, self.row(), 'Link'), 'View First Hill listing')
        raw = self.raw({'hyperlink': SAVED_URLS[0]})
        other = self.row().copy()
        other['Listing'] = 'Different listing'
        self.assertEqual(resolve(raw, other, 'Link'), 'View First Hill listing')
        # The Luna1 title-column selection issue is deliberately not repaired.
        self.assertEqual(resolve(raw, self.row(), 'Listing'), 'First Hill')
        wrong = 'https://example.invalid/not-craigslist'
        self.assertEqual(resolve(self.raw({'hyperlink': wrong}), self.row(), 'Link'), wrong)
        self.assertFalse(valid(wrong))

    def test_all_five_real_checkpoints_use_targets_for_fetch_and_history(self):
        for instance in range(1, 6):
            with self.subTest(instance=instance):
                fetch = Mock(return_value=None)
                raw = self.raw({'hyperlink': SAVED_URLS[0]})
                scope = dict(time=time, re=re, Optional=Optional, List=List,
                             Checkpoint=Checkpoint, StepCategory=Categories(),
                             model=object(), matched_columns={'Listing URL': 'Link'},
                             df=pd.DataFrame([self.row()]), sheet_raw=raw,
                             resolve_listing_url=resolve, is_valid_craigslist_url=valid,
                             fetch_craigslist_page=fetch, extract_craigslist_data_with_fallback=Mock(),
                             parallel_download=execute, parallel_execute=execute)
                definitions(FAMILY / f'instance_{instance}/evaluator.py',
                            {'grade_checkpoint_2', 'grade_checkpoint_3'}, scope)
                scope['grade_checkpoint_2']()
                fetch.assert_called_once_with(SAVED_URLS[0], True)
                history = scope['grade_checkpoint_3']([SAVED_URLS[0]])
                self.assertTrue(history.steps[1]['success'])
                history = scope['grade_checkpoint_3']([SAVED_URLS[1]])
                self.assertFalse(history.steps[1]['success'])


if __name__ == '__main__':
    unittest.main()
