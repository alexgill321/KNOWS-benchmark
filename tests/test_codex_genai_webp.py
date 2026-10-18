"""Offline WebP transport regression; no real model, download, or credentials."""
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import Dict, Optional
import unittest
from unittest.mock import Mock
from urllib.parse import urlparse

from google.genai import types
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'src/browsergym/knows/eval/eval_utils/models.py'
WEB = ROOT / 'src/browsergym/knows/eval/eval_utils/web_utils.py'


def definition(path, name, namespace):
    # Full models.py can authenticate Hugging Face at import time. Load only
    # the unchanged public factory with an explicitly fake GenAI client.
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def pixels(fmt, mode='RGB'):
    data = io.BytesIO()
    Image.new(mode, (7, 5), (40, 90, 140, 170) if mode == 'RGBA' else (40, 90, 140)).save(
        data, format=fmt, **({'lossless': True} if fmt == 'WEBP' else {}))
    return data.getvalue()


class GenAIWebPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.client = Mock()
        self.client.models.generate_content.return_value = SimpleNamespace(text='No')
        self.factory_client = Mock(return_value=self.client)
        self.factory = definition(MODELS, 'gemma_google_ai', dict(
            os=SimpleNamespace(environ={'GOOGLE_AI_API_KEY': 'offline-fixture'}),
            genai=SimpleNamespace(Client=self.factory_client), types=types, Image=Image, io=io))
        self.query = self.factory(model_id='gemini-3.8-flash')

    def query_file(self, path):
        self.messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': 'Original judge instruction.'}]},
            {'role': 'user', 'content': [{'type': 'image', 'image': str(path)},
                                        {'type': 'text', 'text': 'Original decoration and colour question.'}]},
        ]
        with redirect_stdout(io.StringIO()):
            return self.query(self.messages)

    def assert_forwarded(self, path, payload, mime):
        self.assertEqual(self.query_file(path), 'No')
        call = self.client.models.generate_content.call_args
        self.assertEqual(set(call.kwargs), {'model', 'contents'})
        self.assertEqual(call.kwargs['model'], 'gemini-3.8-flash')
        content = call.kwargs['contents']
        self.assertEqual(content[0], 'Original judge instruction.')
        self.assertEqual(content[2], 'Original decoration and colour question.')
        self.assertEqual(content[1].inline_data.mime_type, mime)
        self.assertEqual(content[1].inline_data.data, payload)
        self.assertEqual(path.read_bytes(), payload)
        self.assertEqual(self.messages[1]['content'][0]['image'], str(path))

    def test_valid_rgb_and_alpha_webp_forward_original_bytes_and_judge_settings(self):
        for mode in ('RGB', 'RGBA'):
            with self.subTest(mode=mode):
                payload = pixels('WEBP', mode)
                path = self.root / f'{mode}.webp'
                path.write_bytes(payload)
                with Image.open(path) as image:
                    image.load()
                    self.assertEqual(image.format, 'WEBP')
                    self.assertEqual(image.size, (7, 5))
                self.assert_forwarded(path, payload, 'image/webp')

    def test_mislabelled_and_corrupt_webp_fail_before_genai_request(self):
        for payload in (pixels('PNG'), pixels('JPEG'), b'<html>Sign in</html>',
                        b'', pixels('WEBP')[:20], b'RIFF\x10\x00\x00\x00WEBPgarbage'):
            with self.subTest(payload=payload[:16]):
                self.client.reset_mock()
                path = self.root / 'wrong.webp'
                path.write_bytes(payload)
                self.assertEqual(self.query_file(path), 'Error: Google GenAI API unavailable')
                self.client.models.generate_content.assert_not_called()
                self.assertEqual(path.read_bytes(), payload)

    def test_existing_png_jpg_jpeg_transport_is_unchanged(self):
        for suffix, fmt, mime in (('png', 'PNG', 'image/png'), ('jpg', 'JPEG', 'image/jpeg'),
                                  ('jpeg', 'JPEG', 'image/jpeg')):
            with self.subTest(suffix=suffix):
                path = self.root / f'image.{suffix}'
                payload = pixels(fmt)
                path.write_bytes(payload)
                self.assert_forwarded(path, payload, mime)

    def test_other_formats_are_not_silently_added(self):
        path = self.root / 'image.gif'
        path.write_bytes(pixels('GIF'))
        self.assertEqual(self.query_file(path), 'Error: Google GenAI API unavailable')
        self.client.models.generate_content.assert_not_called()

    def test_real_downloader_content_type_to_webp_client_pipeline_preserves_bytes(self):
        payload = pixels('WEBP', 'RGBA')
        request = Mock(return_value=SimpleNamespace(
            status_code=200, headers={'Content-Type': 'image/webp'}, content=payload))
        download = definition(WEB, 'download_image_from_url', dict(
            os=os, re=re, urlparse=urlparse, Optional=Optional, Dict=Dict,
            requests=SimpleNamespace(get=request)))
        # An image URL can end in .jpg while content negotiation returns WebP.
        url = 'https://images.example.invalid/decoration.jpg'
        path = Path(download(url, str(self.root)))
        self.assertEqual(path.suffix, '.webp')
        self.assertEqual(path.read_bytes(), payload)
        self.assertEqual(request.call_args.args, (url,))
        self.assertFalse('wayback' in str(request.call_args))
        self.assert_forwarded(path, payload, 'image/webp')

    def test_real_downloader_mislabelled_html_is_not_submitted_to_judge(self):
        payload = b'<html>Image service unavailable</html>'
        request = Mock(return_value=SimpleNamespace(
            status_code=200, headers={'Content-Type': 'image/webp'}, content=payload))
        download = definition(WEB, 'download_image_from_url', dict(
            os=os, re=re, urlparse=urlparse, Optional=Optional, Dict=Dict,
            requests=SimpleNamespace(get=request)))
        path = Path(download('https://images.example.invalid/decoration.jpg', str(self.root)))
        self.assertEqual(self.query_file(path), 'Error: Google GenAI API unavailable')
        self.client.models.generate_content.assert_not_called()
        self.assertEqual(path.read_bytes(), payload)


if __name__ == '__main__':
    unittest.main()
