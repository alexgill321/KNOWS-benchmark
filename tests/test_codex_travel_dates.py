"""Offline date parsing regressions for annotated travel-planner Date cells."""
import ast
from datetime import date, datetime
from pathlib import Path
import re
from typing import Dict, List, Optional
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/browsergym/knows/eval/tasks/sheets_28_personal_travel_planner/utils.py'
CONSTANTS = {'_DATE_FORMATS', '_WEEKDAY_PREFIX_RE', '_DAY_TYPE_SUFFIX_RE',
             '_ANNOTATION_WEEKDAY', '_ANNOTATED_DAY_TYPE_SUFFIX_RE', '_YEARLESS_FORMATS'}
FUNCTIONS = {'parse_trip_date', 'count_rows_per_day'}
namespace = dict(re=re, pd=pd, _date=date, _dt=datetime, Optional=Optional, List=List, Dict=Dict)
nodes = [node for node in ast.parse(SOURCE.read_text()).body if
         (isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS) or
         (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in CONSTANTS
                                             for target in node.targets))]
# The rest of the task module imports network helpers. Load only these pure
# parsing/grouping definitions; no evaluator, Google client, or model is run.
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
parse = namespace['parse_trip_date']
count_rows = namespace['count_rows_per_day']


SAVED_DATES = {
    'Mon 07 Dec 2026 · Weekday': date(2026, 12, 7),
    'Tue 08 Dec 2026 · Weekday': date(2026, 12, 8),
    'Wed 09 Dec 2026 · Weekday': date(2026, 12, 9),
    'Thu 10 Dec 2026 · Weekday': date(2026, 12, 10),
    'Fri 11 Dec 2026 · Weekday': date(2026, 12, 11),
    '19 Mar 2027 • Fri • Weekday': date(2027, 3, 19),
    '20 Mar 2027 • Sat • Weekend': date(2027, 3, 20),
    '21 Mar 2027 • Sun • Weekend': date(2027, 3, 21),
    '22 Mar 2027 • Mon • Weekday': date(2027, 3, 22),
}
ADDITIONAL_REVIEWED_DATES = {
    '2027-07-08 Thu • Weekday': date(2027, 7, 8),
    '2027-07-09 Fri • Weekend': date(2027, 7, 9),
    '2027-07-10 Sat • Weekend': date(2027, 7, 10),
    '2027-07-11 Sun • Weekday': date(2027, 7, 11),
    '2027-07-12 Mon • Weekday': date(2027, 7, 12),
    'Fri 5 Mar 2027 — Weekday': date(2027, 3, 5),
    'Sat 6 Mar 2027 — Weekend': date(2027, 3, 6),
    'Sun 7 Mar 2027 — Weekend': date(2027, 3, 7),
    'Mon 8 Mar 2027 — Weekday': date(2027, 3, 8),
    'Tue 11 May 2027 · Weekday': date(2027, 5, 11),
    'Wed 12 May 2027 · Weekday': date(2027, 5, 12),
    'Thu 13 May 2027 · Weekday': date(2027, 5, 13),
    '08 Jul 2027 · Thu · WEEKDAY': date(2027, 7, 8),
    '09 Jul 2027 · Fri · WEEKEND': date(2027, 7, 9),
    '10 Jul 2027 · Sat · WEEKEND': date(2027, 7, 10),
    '11 Jul 2027 · Sun · WEEKDAY': date(2027, 7, 11),
    '12 Jul 2027 · Mon · WEEKDAY': date(2027, 7, 12),
    '02 Aug 2027 | Mon • Weekday': date(2027, 8, 2),
    '03 Aug 2027 | Tue • Weekday': date(2027, 8, 3),
    '04 Aug 2027 | Wed • Weekday': date(2027, 8, 4),
    '05 Aug 2027 | Thu • Weekday': date(2027, 8, 5),
    '06 Aug 2027 | Fri • Weekday': date(2027, 8, 6),
    '07 Aug 2027 | Sat • Weekend': date(2027, 8, 7),
    '08 Aug 2027 | Sun • Weekend': date(2027, 8, 8),
    '09 Aug 2027 | Mon • Weekday': date(2027, 8, 9),
}


class TravelDateTests(unittest.TestCase):
    def test_all_nine_reviewed_saved_dates_parse_to_exact_calendar_day(self):
        for value, expected in SAVED_DATES.items():
            with self.subTest(value=value):
                self.assertEqual(parse(value), expected)

    def test_full_weekday_names_case_and_spacing_are_safe_annotations(self):
        for value in ('19 March 2027 • Friday • Weekday', '2027-03-20 · Saturday · Weekend',
                      ' 19 Mar 2027 • fri. • weekday ', '2026-12-07·Weekday'):
            with self.subTest(value=value):
                self.assertIsInstance(parse(value), date)

    def test_additional_reviewed_separator_and_iso_weekday_forms(self):
        for value, expected in ADDITIONAL_REVIEWED_DATES.items():
            with self.subTest(value=value):
                self.assertEqual(parse(value), expected)

    def test_regional_weekend_annotations_do_not_redefine_the_calendar_date(self):
        for value, expected in (('2027-07-09 Fri • Weekend', date(2027, 7, 9)),
                                ('09 Jul 2027 · Fri · WEEKEND', date(2027, 7, 9)),
                                ('11 Jul 2027 · Sun · WEEKDAY', date(2027, 7, 11))):
            with self.subTest(value=value):
                self.assertEqual(parse(value), expected)

    def test_exact_parser_still_rejects_impossible_calendar_dates(self):
        for value in ('31 Apr 2027 • Fri • Weekday', '29 Feb 2027 · Weekday',
                      '2027-13-01 · Weekday', '2027-00-20 • Weekend', '20/35/2027 • Weekend'):
            with self.subTest(value=value):
                self.assertIsNone(parse(value))
        self.assertEqual(parse('29 Feb 2028 · Weekday'), date(2028, 2, 29))

    def test_ambiguous_dates_unknown_or_stacked_annotations_are_not_stripped(self):
        for value in ('19 Mar 2027 or 20 Mar 2027 • Weekday', '19–20 Mar 2027 • Weekend',
                      '19 Mar 2027 • Fri • Sat • Weekday', '19 Mar 2027 • Weekday • Weekend',
                      '19 Mar 2027 • Weekday • 20 Mar 2027', '19 Mar 2027 • Holiday',
                      '19 Mar 2027 • Fridayish • Weekday', '19 Mar 2027 • Fri • Weekday maybe',
                      '19 Mar 2027 (Weekday) • Fri • Weekday', '19 Mar 2027 • Weekday (Weekend)',
                      'Arrival • 19 Mar 2027 • Weekday', '19 Mar 2027 • Fri',
                      '19 Mar 2027 • • Weekday', '2027-07-08 Thu Fri • Weekday',
                      '2027-07-08 Thu | 2027-07-09 Fri • Weekend',
                      '2027-07-08 | transfer • Weekday', '2027-07-08 — Weekday trip',
                      'TRIP NOTES — May 2027 · Weekday'):
            with self.subTest(value=value):
                self.assertIsNone(parse(value))

    def test_unannotated_and_legacy_parenthesized_formats_are_unchanged(self):
        for value in ('20 May 2026', 'May 20, 2026', 'May 20 2026', '5/20/2026',
                      '20/5/2026', '2026-05-20', '2026/05/20', 'Wednesday, May 20, 2026',
                      'Wed, May 20, 2026 (Weekday)', 'May 20, 2026 (Wednesday)',
                      '2026-05-20 00:00:00'):
            with self.subTest(value=value):
                self.assertEqual(parse(value), date(2026, 5, 20))
        for value in (None, '', 'not a date', 20260520):
            self.assertIsNone(parse(value))

    def test_date_objects_timestamps_and_yearless_policy_are_unchanged(self):
        for value in (date(2027, 3, 19), datetime(2027, 3, 19, 13, 45), pd.Timestamp('2027-03-19')):
            self.assertEqual(parse(value), date(2027, 3, 19))
        self.assertEqual(parse('May 20'), date(date.today().year, 5, 20))

    def test_real_day_grouping_preserves_counts_and_derives_weekend_from_date(self):
        values, types = [], []
        for value in SAVED_DATES:
            values.extend([value, '', '', '', ''])
            types.extend(['activity', 'food', 'activity', 'food', 'activity'])
        grouped = count_rows(pd.DataFrame({'Date': values}), 'Date', types)
        self.assertEqual(len(grouped), len(SAVED_DATES))
        for group, expected in zip(grouped, SAVED_DATES.values()):
            self.assertEqual(group['activity_count'], 3)
            self.assertEqual(group['food_count'], 2)
            self.assertEqual(group['parsed_date'], expected)
            self.assertEqual(group['is_weekend'], expected.weekday() >= 5)
        # Parsing a date does not award credit for claims in its annotation:
        # existing day-type logic still uses the actual calendar day.
        group = count_rows(pd.DataFrame({'Date': ['19 Mar 2027 • Fri • Weekend']}), 'Date', ['food'])[0]
        self.assertFalse(group['is_weekend'])

    def test_fix_does_not_correct_wrong_row_counts_or_promote_notes_to_dates(self):
        label = 'Tue 11 May 2027 · Weekday'
        note = 'TRIP NOTES • Washington, DC • Family of 4 • Read before booking'
        groups = count_rows(pd.DataFrame({'Date': [label] * 5 + [note]}), 'Date', ['activity'] * 6)
        self.assertEqual(groups[0]['parsed_date'], date(2027, 5, 11))
        self.assertEqual((groups[0]['activity_count'], groups[0]['food_count']), (5, 0))
        self.assertEqual(groups[1]['date'], note)
        self.assertIsNone(groups[1]['parsed_date'])


if __name__ == '__main__':
    unittest.main()
