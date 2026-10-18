"""Offline regressions for omitted zero chart-anchor rows; no API or judge."""
import ast
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import re
import sys
from typing import List, Dict
import unittest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'codex_benchmark').is_dir())
sys.path.insert(0,str(ROOT/'src'))
FAMILY=Path('src/browsergym/knows/eval/tasks/sheets_7_running_analysis')
STAGED=Path(os.environ.get('KNOWS_RUNNING_CHART_TEST_SOURCE',ROOT))
spec=importlib.util.spec_from_file_location('anchor_context_tested',STAGED/FAMILY/'chart_context.py')
ctx=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctx)
ns={'List':List,'Dict':Dict,'re':re}
p=ROOT/'src/browsergym/knows/eval/eval_utils/google_sheets_utils.py'
node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='find_urls_in_sheet')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
find_urls=ns['find_urls_in_sheet']
def fixture(row_missing=True):
 rows=[{'values':[]}for _ in range(220)]
 for idx,url in [(23,'https://example.org/adult'),(27,'https://example.org/athlete')]:rows[idx]={'values':[{}]*6+[{'hyperlink':url}]}
 raw={'sheets':[{'properties':{'sheetId':0,'title':'Raw','gridProperties':{'rowCount':1000,'columnCount':35}},'data':[{'rowData':[]}]},{'properties':{'sheetId':7,'title':'Strength','gridProperties':{'rowCount':1000,'columnCount':26}},'data':[{'rowData':rows}]}]}
 a={'sheetId':7,'columnIndex':6}
 if not row_missing:a['rowIndex']=1
 c={'chart_id':1,'sheet_id':7,'sheet_name':'Strength','position':{'type':'overlay','anchor_cell':{'sheet_id':7,'row':None if row_missing else 1,'col':6},'height':371,'width':600},'raw_chart':{'position':{'overlayPosition':{'anchorCell':a}},'spec':{'basicChart':{'series':[]}}}}
 return c,raw
class RunningChartAnchorTests(unittest.TestCase):
 def test_omitted_row_is_zero_and_native_source_urls_are_found(self):
  c,raw=fixture();old=deepcopy((c,raw));bound=ctx.bind_chart_context([c],raw,0)[0]
  self.assertEqual(bound['position']['anchor_cell']['row'],0)
  start=bound['position']['anchor_cell']['row']+bound['position']['height']//20+1
  self.assertEqual(start,19)
  self.assertEqual(set(find_urls(ctx.chart_host_rows(bound),start,60)),{'https://example.org/adult','https://example.org/athlete'})
  self.assertEqual(find_urls(ctx.chart_host_rows(bound),283,60),[])
  self.assertEqual((c,raw),old)
 def test_explicit_anchor_remains_exact(self):
  c,raw=fixture(False);before=deepcopy(c['position']);bound=ctx.bind_chart_context([c],raw,0)[0]
  self.assertEqual(bound['position'],before)
 def test_zero_anchor_restores_actual_same_tab_overlap_check(self):
  c,raw=fixture();other=deepcopy(c);other['chart_id']=2
  other['position']['anchor_cell'].update(row=6,col=1)
  other['raw_chart']['position']['overlayPosition']['anchorCell'].update(rowIndex=6,columnIndex=1)
  charts=ctx.bind_chart_context([c,other],raw,0)
  self.assertTrue(ctx.check_chart_overlap(charts[0],0,293,charts,0,35)[0])
 def test_explicit_null_bool_negative_and_out_of_grid_fail_closed(self):
  for key,value in [('rowIndex',None),('rowIndex',True),('rowIndex',-1),('rowIndex',1000),('columnIndex',None),('columnIndex',False),('columnIndex',-1),('columnIndex',26),('sheetId',None),('sheetId',True),('sheetId',0),('sheetId',99)]:
   with self.subTest(key=key,value=value):
    c,raw=fixture();c['raw_chart']['position']['overlayPosition']['anchorCell'][key]=value
    with self.assertRaises(ctx.ChartContextError):ctx.bind_chart_context([c],raw,0)
 def test_missing_raw_anchor_column_or_parsed_identity_is_not_invented(self):
  for case in ('raw','parsed_row','parsed_col'):
   c,raw=fixture()
   if case=='raw':c['raw_chart'].pop('position')
   elif case=='parsed_row':c['position']['anchor_cell']['row']=3
   else:c['position']['anchor_cell']['col']=2
   with self.subTest(case=case),self.assertRaises(ctx.ChartContextError):ctx.bind_chart_context([c],raw,0)
 def test_omitted_column_is_preserved_without_extra_normalization(self):
  c,raw=fixture();c['raw_chart']['position']['overlayPosition']['anchorCell'].pop('columnIndex');c['position']['anchor_cell']['col']=None
  bound=ctx.bind_chart_context([c],raw,0)[0]
  self.assertIsNone(bound['position']['anchor_cell']['col']);self.assertEqual(bound['position']['anchor_cell']['row'],0)
 def test_chart_selection_metadata_and_non_overlay_positions_unchanged(self):
  c,raw=fixture();c.update(title='',chart_type='LINE',series=[],data_range={})
  bound=ctx.bind_chart_context([c],raw,0)[0]
  for k in ('title','chart_type','series','data_range','raw_chart'):self.assertEqual(bound[k],c[k])
  c['position']={'type':'new_sheet'};c['raw_chart']['position']={'newSheet':True}
  self.assertEqual(ctx.bind_chart_context([c],raw,0)[0]['position'],c['position'])
if __name__=='__main__':unittest.main()
