import asyncio
import sys
import unittest
from unittest import mock

# Mock telegram dependencies so tests run without python-telegram-bot installed
if "telegram" not in sys.modules:
    _tg = mock.MagicMock()
    sys.modules["telegram"] = _tg
    sys.modules["telegram.ext"] = _tg.ext

from src.forwarder import _truncate


class TestTruncate(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(_truncate("hello", 10), "hello")

    def test_exact_length_unchanged(self):
        self.assertEqual(_truncate("hello", 5), "hello")

    def test_long_text_truncated(self):
        result = _truncate("hello world", 8)
        self.assertEqual(result, "hello...")
        self.assertEqual(len(result), 8)

    def test_empty_string(self):
        self.assertEqual(_truncate("", 10), "")


class TestForwardSignalMessage(unittest.TestCase):
    """Test that forward_signal builds correct HTML-escaped messages."""

    def _make_forwarder(self):
        from src.forwarder import SignalForwarder
        fwd = SignalForwarder.__new__(SignalForwarder)
        fwd.bot_token = "fake"
        fwd.app = mock.AsyncMock()
        fwd.app.bot.send_message = mock.AsyncMock()
        return fwd

    def test_html_escaping_in_original_text(self):
        """Messages containing < > & must be HTML-escaped to avoid Telegram API errors."""
        async def _run():
            fwd = self._make_forwarder()
            with mock.patch("src.forwarder.get_bot_users", return_value=[123]):
                await fwd.forward_signal(
                    classification={
                        "tickers": [{"symbol": "BTC", "bias": "bullish"}],
                        "confidence": 0.9,
                        "category": "trade_idea",
                        "thesis": "P&L > 100% when <conditions> met",
                    },
                    original_text="BTC > 100k & ETH < 5k",
                    channel_name="Test <Channel>",
                )
            call_args = fwd.app.bot.send_message.call_args
            text = call_args.kwargs["text"]
            # Raw < > & must not appear unescaped
            self.assertNotIn("&c", text.split("&amp;")[0] if "&amp;" in text else "")
            self.assertIn("&amp;", text)
            self.assertIn("&gt;", text)
            self.assertIn("&lt;", text)

        asyncio.run(_run())

    def test_no_users_skips_send(self):
        async def _run():
            fwd = self._make_forwarder()
            with mock.patch("src.forwarder.get_bot_users", return_value=[]):
                await fwd.forward_signal(
                    classification={"tickers": [], "confidence": 0.5, "category": "noise", "thesis": ""},
                    original_text="test",
                    channel_name="ch",
                )
            fwd.app.bot.send_message.assert_not_called()

        asyncio.run(_run())

    def test_send_failure_does_not_crash(self):
        """A failed send to one user should not prevent sending to others."""
        async def _run():
            fwd = self._make_forwarder()
            call_count = 0

            async def side_effect(**kwargs):
                nonlocal call_count
                call_count += 1
                if kwargs["chat_id"] == 111:
                    raise Exception("blocked by user")

            fwd.app.bot.send_message = mock.AsyncMock(side_effect=side_effect)

            with mock.patch("src.forwarder.get_bot_users", return_value=[111, 222]):
                await fwd.forward_signal(
                    classification={"tickers": [], "confidence": 0.8, "category": "macro", "thesis": "test"},
                    original_text="msg",
                    channel_name="ch",
                )
            self.assertEqual(call_count, 2)

        asyncio.run(_run())

    def test_ticker_bias_icons(self):
        async def _run():
            fwd = self._make_forwarder()
            with mock.patch("src.forwarder.get_bot_users", return_value=[123]):
                await fwd.forward_signal(
                    classification={
                        "tickers": [
                            {"symbol": "BTC", "bias": "bullish"},
                            {"symbol": "ETH", "bias": "bearish"},
                            {"symbol": "SOL", "bias": "neutral"},
                        ],
                        "confidence": 0.9,
                        "category": "trade_idea",
                        "thesis": "test",
                    },
                    original_text="test",
                    channel_name="ch",
                )
            text = fwd.app.bot.send_message.call_args.kwargs["text"]
            self.assertIn("🟢", text)
            self.assertIn("🔴", text)
            self.assertIn("⚪", text)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
