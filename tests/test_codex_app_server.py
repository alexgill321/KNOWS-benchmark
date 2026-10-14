"""Protocol behavior tests without paid model calls or a real Codex login."""

import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from codex_benchmark.app_server import AppServer, AppServerError, AppServerRPCError, _redact


FAKE_SERVER = r'''#!/usr/bin/env python3
import json, os, sys

assert all(name not in os.environ for name in ('GOOGLE_APPLICATION_CREDENTIALS','GOOGLE_REFRESH_TOKEN','OPENAI_API_KEY','EVALUATOR_PASSWORD'))

def emit(value):
    print(json.dumps(value), flush=True)

def response(message, result):
    emit({'id':message['id'],'result':result})

pending = None
prompt = ''
for line in sys.stdin:
    message = json.loads(line)
    method = message.get('method')
    params = message.get('params', {})
    if method == 'initialize':
        response(message, {'userAgent':'fake-codex/0.159.0'})
    elif method == 'account/read':
        response(message, {'account':{'type':'chatgpt','planType':'pro','email':'private@example.test'}})
    elif method == 'config/read':
        response(message, {'config':{'mcp_servers':{'private-server':{'command':'secret-command','env':{'SECRET':'must-not-log'}}}}})
    elif method == 'model/list':
        response(message, {'data':[{'id':'gpt-6-luna','supportedReasoningEfforts':[{'reasoningEffort':'max'}]}], 'nextCursor':None})
    elif method == 'thread/start':
        assert params['environments'] == []
        assert params['ephemeral'] is True
        assert params['allowProviderModelFallback'] is False
        assert params['config']['mcp_servers.private-server.enabled'] is False
        assert params['config']['features.shell_tool'] is False
        assert params['config']['features.code_mode_host'] is True
        assert params['config']['features.code_mode'] is False
        response(message, {'thread':{'id':'thread-1'},'model':params['model'],
                           'reasoningEffort':params['config']['model_reasoning_effort'], 'instructionSources':[]})
    elif method == 'turn/start':
        prompt = params['input'][0]['text']
        # Deliberately interleave a server request before the RPC acknowledgement.
        if prompt == 'unexpected':
            emit({'id':'server-1','method':'item/commandExecution/requestApproval','params':{'threadId':'thread-1'}})
        elif prompt == 'forbidden':
            emit({'method':'item/started','params':{'threadId':'thread-1','turnId':'turn-1','item':{'type':'fileChange'}}})
        else:
            emit({'id':'server-1','method':'item/tool/call','params':{'threadId':'thread-1','turnId':'turn-1','callId':'call-1','tool':'browser','arguments':{'action':'observe'}}})
        response(message, {'turn':{'id':'turn-1','status':'inProgress'}})
    elif message.get('id') == 'server-1' and 'result' in message:
        assert message['result']['success'] is True
        if prompt == 'repeat':
            emit({'id':'server-2','method':'item/tool/call','params':{'threadId':'thread-1','turnId':'turn-1','callId':'call-2','tool':'browser','arguments':{}}})
            continue
        emit({'method':'thread/tokenUsage/updated','params':{'threadId':'thread-1','turnId':'turn-1','tokenUsage':{'total':{'totalTokens':123}}}})
        emit({'method':'item/completed','params':{'threadId':'thread-1','turnId':'turn-1','item':{'id':'a1','type':'agentMessage','phase':'final_answer','text':'Done.'}}})
        emit({'method':'turn/completed','params':{'threadId':'thread-1','turn':{'id':'turn-1','status':'completed','error':None}}})
    elif method == 'probe/first':
        pending = message
    elif method == 'probe/second':
        response(message, {'value':2})
        response(pending, {'value':1})
    elif method == 'probe/error':
        emit({'id':message['id'],'error':{'code':-32000,'message':'intentional test error'}})
    elif method in ('turn/interrupt', 'thread/unsubscribe'):
        response(message, {})
'''


TOOLS = [{"type": "function", "name": "browser", "description": "Browser action",
          "inputSchema": {"type": "object", "properties": {}}}]


class AppServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        executable = Path(self.directory.name) / "codex"
        executable.write_text(FAKE_SERVER)
        executable.chmod(0o755)
        self.patch = patch("codex_benchmark.app_server.shutil.which", return_value=str(executable))
        self.patch.start()
        self.log_path = Path(self.directory.name) / "events.jsonl"

    async def asyncTearDown(self):
        self.patch.stop()
        self.directory.cleanup()

    async def test_interleaved_rpc_results_and_server_tool_calls(self):
        calls = []

        async def handler(name, arguments):
            calls.append((name, arguments))
            return {"success": True, "contentItems": [{"type": "inputImage", "imageUrl": "data:image/png;base64,abc"}]}

        async with AppServer(log_path=self.log_path) as server:
            first, second = await asyncio.gather(server.request("probe/first"), server.request("probe/second"))
            self.assertEqual((first["value"], second["value"]), (1, 2))
            result = await server.run_turn("gpt-6-luna", "max", "complete task", TOOLS, handler)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["final_text"], "Done.")
            self.assertEqual(result["tool_calls"], 1)
            self.assertEqual(result["token_usage"]["total"]["totalTokens"], 123)
            self.assertEqual(calls, [("browser", {"action": "observe"})])
            self.assertGreater(result["duration_seconds"], 0)
        log = self.log_path.read_text()
        self.assertNotIn("private@example.test", log)
        self.assertNotIn("secret-command", log)
        self.assertNotIn("must-not-log", log)
        self.assertNotIn("data:image/png;base64,abc", log)

    async def test_whole_turn_deadline_includes_slow_browser_tool(self):
        async def handler(*args):
            await asyncio.sleep(10)

        async with AppServer(log_path=self.log_path) as server:
            started = time.monotonic()
            result = await server.run_turn("gpt-6-luna", "max", "complete task", TOOLS, handler, timeout_seconds=1)
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(result["tool_calls"], 1)
            self.assertLess(time.monotonic() - started, 4)
        self.assertIn('"method": "turn/interrupt"', self.log_path.read_text())

    async def test_action_budget_returns_partial_run_for_grading(self):
        async def handler(*args):
            return {"success": True, "contentItems": []}

        async with AppServer(log_path=self.log_path) as server:
            result = await server.run_turn("gpt-6-luna", "max", "repeat", TOOLS, handler, max_tool_calls=1)
            self.assertEqual(result["status"], "action_limit")
            self.assertEqual(result["tool_calls"], 1)
        self.assertIn('"method": "turn/interrupt"', self.log_path.read_text())

    async def test_unexpected_server_requests_fail_closed(self):
        async with AppServer(log_path=self.log_path) as server:
            with self.assertRaisesRegex(AppServerError, "Unexpected server request"):
                await server.run_turn("gpt-6-luna", "max", "unexpected", TOOLS, None)
        self.assertIn('"code": -32601', self.log_path.read_text())

    async def test_forbidden_builtin_tool_fails_run(self):
        async with AppServer() as server:
            with self.assertRaisesRegex(AppServerError, "Forbidden built-in tool"):
                await server.run_turn("gpt-6-luna", "max", "forbidden", TOOLS, None)

    async def test_model_and_effort_must_match_account_catalog(self):
        async with AppServer() as server:
            with self.assertRaisesRegex(AppServerError, "substitution is disabled"):
                await server.run_turn("not-available", "max", "test", TOOLS, None)
            with self.assertRaisesRegex(AppServerError, "does not support effort"):
                await server.run_turn("gpt-6-luna", "low", "test", TOOLS, None)
            with self.assertRaisesRegex(AppServerError, "Ultra includes automatic delegation"):
                await server.run_turn("gpt-6-luna", "ultra", "test", TOOLS, None)
            with self.assertRaisesRegex(AppServerRPCError, "intentional test error"):
                await server.request("probe/error")

    async def test_evaluator_credentials_are_not_inherited(self):
        secrets = {name: 'sensitive-test-value' for name in (
            'GOOGLE_APPLICATION_CREDENTIALS', 'GOOGLE_REFRESH_TOKEN', 'OPENAI_API_KEY', 'EVALUATOR_PASSWORD')}
        with patch.dict(os.environ, secrets):
            async with AppServer() as server:
                self.assertEqual(server.account['type'], 'chatgpt')

    async def test_api_key_authentication_is_rejected(self):
        executable = Path(self.directory.name) / "codex"
        executable.write_text(FAKE_SERVER.replace("'type':'chatgpt'", "'type':'apiKey'"))
        with self.assertRaisesRegex(AppServerError, 'ChatGPT subscription login is required'):
            async with AppServer():
                self.fail('API-key session should not start')

    async def test_private_rpc_errors_remain_diagnosable_and_redacted(self):
        executable = Path(self.directory.name) / "codex"
        private_response = "response(message, {'account':{'type':'chatgpt','planType':'pro','email':'private@example.test'}})"
        error_response = "emit({'id':message['id'],'error':{'code':-32000,'message':'Account temporarily unavailable','data':{'email':'private@example.test','access_token':'private-token'}}})"
        executable.write_text(FAKE_SERVER.replace(private_response, error_response))
        with self.assertRaisesRegex(AppServerRPCError, 'Account temporarily unavailable'):
            async with AppServer(log_path=self.log_path):
                self.fail('RPC failure should abort startup')
        log = self.log_path.read_text()
        self.assertIn('Account temporarily unavailable', log)
        self.assertIn('"code": -32000', log)
        self.assertNotIn('private@example.test', log)
        self.assertNotIn('private-token', log)

    def test_redaction_keeps_usage_but_removes_credentials(self):
        self.assertEqual(_redact({"access_token": "secret", "email": "secret", "tokenUsage": {"inputTokens": 12}}),
                         {"access_token": "[redacted]", "email": "[redacted]", "tokenUsage": {"inputTokens": 12}})


if __name__ == "__main__":
    unittest.main()
