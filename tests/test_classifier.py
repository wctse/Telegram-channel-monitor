import asyncio
import json
import unittest

import aiohttp
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from src.classifier import LLMClassifier


class TestParseTriage(unittest.TestCase):
    def test_valid_signal(self):
        content = json.dumps({"is_signal": True, "confidence": 0.85, "category": "trade_idea"})
        result = LLMClassifier._parse_triage(content)
        self.assertEqual(result, {"is_signal": True, "confidence": 0.85, "category": "trade_idea"})

    def test_valid_noise(self):
        content = json.dumps({"is_signal": False, "confidence": 0.2, "category": "noise"})
        result = LLMClassifier._parse_triage(content)
        self.assertFalse(result["is_signal"])

    def test_missing_fields_use_defaults(self):
        content = json.dumps({})
        result = LLMClassifier._parse_triage(content)
        self.assertFalse(result["is_signal"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["category"], "noise")

    def test_invalid_json_returns_none(self):
        self.assertIsNone(LLMClassifier._parse_triage("not json"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(LLMClassifier._parse_triage(""))


class TestParseEnrich(unittest.TestCase):
    def test_valid_with_tickers(self):
        content = json.dumps({
            "is_signal": True, "confidence": 0.9,
            "thesis": "BTC bull run",
            "tickers": [{"symbol": "BTC", "bias": "bullish"}],
            "category": "trade_idea",
        })
        result = LLMClassifier._parse_enrich(content)
        self.assertTrue(result["is_signal"])
        self.assertEqual(len(result["tickers"]), 1)
        self.assertEqual(result["tickers"][0]["symbol"], "BTC")
        self.assertEqual(result["tickers"][0]["bias"], "bullish")

    def test_tickers_as_strings(self):
        content = json.dumps({
            "is_signal": True, "confidence": 0.8,
            "thesis": "test", "tickers": ["BTC", "ETH"],
            "category": "macro",
        })
        result = LLMClassifier._parse_enrich(content)
        self.assertEqual(result["tickers"][0], {"symbol": "BTC", "bias": "neutral"})
        self.assertEqual(result["tickers"][1], {"symbol": "ETH", "bias": "neutral"})

    def test_missing_tickers(self):
        content = json.dumps({"is_signal": False, "confidence": 0.1})
        result = LLMClassifier._parse_enrich(content)
        self.assertEqual(result["tickers"], [])

    def test_invalid_json_returns_none(self):
        self.assertIsNone(LLMClassifier._parse_enrich("{broken"))

    def test_bias_normalized_to_lowercase(self):
        content = json.dumps({
            "is_signal": True, "confidence": 0.9, "thesis": "t",
            "tickers": [{"symbol": "BTC", "bias": "BULLISH"}],
            "category": "trade_idea",
        })
        result = LLMClassifier._parse_enrich(content)
        self.assertEqual(result["tickers"][0]["bias"], "bullish")


class TestSessionLifecycle(unittest.TestCase):
    def test_get_session_before_open_raises(self):
        c = LLMClassifier("ollama", "http://localhost", "m", "p1", "p2")
        with self.assertRaises(RuntimeError):
            c._get_session()

    def test_open_and_close(self):
        async def _run():
            c = LLMClassifier("ollama", "http://localhost", "m", "p1", "p2")
            await c.open()
            session = c._get_session()
            self.assertIsInstance(session, aiohttp.ClientSession)
            self.assertFalse(session.closed)
            await c.close()
            self.assertIsNone(c._session)

        asyncio.run(_run())

    def test_double_close_safe(self):
        async def _run():
            c = LLMClassifier("ollama", "http://localhost", "m", "p1", "p2")
            await c.open()
            await c.close()
            await c.close()  # should not raise

        asyncio.run(_run())

    def test_triage_empty_text(self):
        async def _run():
            c = LLMClassifier("ollama", "http://localhost", "m", "p1", "p2")
            result = await c.triage("")
            self.assertIsNone(result)
            result = await c.triage("   ")
            self.assertIsNone(result)

        asyncio.run(_run())


class TestOllamaIntegration(unittest.TestCase):
    """Tests against a fake Ollama HTTP server."""

    def test_ollama_success(self):
        async def _run():
            app = web.Application()
            response_body = {"message": {"content": json.dumps(
                {"is_signal": True, "confidence": 0.9, "category": "macro"}
            )}}

            async def handler(request):
                return web.json_response(response_body)

            app.router.add_post("/api/chat", handler)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]

            c = LLMClassifier("ollama", f"http://127.0.0.1:{port}", "m", "sys", "sys")
            await c.open()
            try:
                result = await c.triage("test message")
                self.assertIsNotNone(result)
                self.assertTrue(result["is_signal"])
                self.assertAlmostEqual(result["confidence"], 0.9)
            finally:
                await c.close()
                await runner.cleanup()

        asyncio.run(_run())

    def test_ollama_error_status(self):
        async def _run():
            app = web.Application()

            async def handler(request):
                return web.Response(status=500, text="internal error")

            app.router.add_post("/api/chat", handler)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]

            c = LLMClassifier("ollama", f"http://127.0.0.1:{port}", "m", "sys", "sys")
            await c.open()
            try:
                result = await c.triage("test")
                self.assertIsNone(result)
            finally:
                await c.close()
                await runner.cleanup()

        asyncio.run(_run())


class TestOpenAIIntegration(unittest.TestCase):
    """Tests against a fake OpenAI-compatible HTTP server."""

    def test_openai_success(self):
        async def _run():
            app = web.Application()
            response_body = {"choices": [{"message": {"content": json.dumps(
                {"is_signal": True, "confidence": 0.85, "category": "trade_idea"}
            )}}]}

            async def handler(request):
                data = await request.json()
                self.assertEqual(data["model"], "gpt-test")
                return web.json_response(response_body)

            app.router.add_post("/chat/completions", handler)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]

            c = LLMClassifier("openai", f"http://127.0.0.1:{port}", "gpt-test", "sys", "sys", api_key="sk-test")
            await c.open()
            try:
                result = await c.triage("test message")
                self.assertIsNotNone(result)
                self.assertTrue(result["is_signal"])
            finally:
                await c.close()
                await runner.cleanup()

        asyncio.run(_run())

    def test_openai_malformed_response(self):
        async def _run():
            app = web.Application()

            async def handler(request):
                return web.json_response({"bad": "response"})

            app.router.add_post("/chat/completions", handler)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]

            c = LLMClassifier("openai", f"http://127.0.0.1:{port}", "m", "sys", "sys")
            await c.open()
            try:
                result = await c.triage("test")
                self.assertIsNone(result)
            finally:
                await c.close()
                await runner.cleanup()

        asyncio.run(_run())

    def test_openai_empty_choices(self):
        async def _run():
            app = web.Application()

            async def handler(request):
                return web.json_response({"choices": []})

            app.router.add_post("/chat/completions", handler)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]

            c = LLMClassifier("openai", f"http://127.0.0.1:{port}", "m", "sys", "sys")
            await c.open()
            try:
                result = await c.triage("test")
                self.assertIsNone(result)
            finally:
                await c.close()
                await runner.cleanup()

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
