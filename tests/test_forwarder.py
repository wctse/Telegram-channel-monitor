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


def _make_update(chat_id, username="testuser", first_name="Test", text=None):
    update = mock.MagicMock()
    update.effective_user.username = username
    update.effective_user.first_name = first_name
    update.effective_chat.id = chat_id
    update.message.reply_text = mock.AsyncMock()
    update.message.text = text
    return update


class TestGroupRegistrationGate(unittest.IsolatedAsyncioTestCase):
    def _make_forwarder(self, allowed_groups=None):
        from src.forwarder import SignalForwarder
        fwd = SignalForwarder.__new__(SignalForwarder)
        fwd.bot_token = "fake"
        fwd.app = mock.MagicMock()
        fwd._allowed_chat_ids = set()
        fwd._allowed_groups = {g.strip().lower() for g in (allowed_groups or [])}
        fwd._pending_chats = set()
        return fwd

    async def test_correct_group_answer_registers_user(self):
        """User who answers with the correct group name gets registered."""
        fwd = self._make_forwarder(allowed_groups=["alpha-group"])

        start_update = _make_update(chat_id=999)
        with mock.patch("src.forwarder.get_bot_users", return_value=[]):
            await fwd._handle_start(start_update, mock.MagicMock())

        self.assertIn(999, fwd._pending_chats)

        text_update = _make_update(chat_id=999, text="alpha-group")
        with mock.patch("src.forwarder.save_bot_user") as mock_save:
            await fwd._handle_text(text_update, mock.MagicMock())

        mock_save.assert_called_once_with(999, "testuser", "Test")
        self.assertNotIn(999, fwd._pending_chats)

    async def test_wrong_group_answer_rejects_user(self):
        """User answering with an unrecognized group stays pending and is not registered."""
        fwd = self._make_forwarder(allowed_groups=["alpha-group"])
        fwd._pending_chats.add(999)

        text_update = _make_update(chat_id=999, text="wrong-group")
        with mock.patch("src.forwarder.save_bot_user") as mock_save:
            await fwd._handle_text(text_update, mock.MagicMock())

        mock_save.assert_not_called()
        self.assertIn(999, fwd._pending_chats)

    async def test_empty_allowed_groups_allows_open_registration(self):
        """When allowed_groups is empty, /start registers anyone without asking for a group."""
        fwd = self._make_forwarder(allowed_groups=[])

        start_update = _make_update(chat_id=999)
        with mock.patch("src.forwarder.get_bot_users", return_value=[]), \
             mock.patch("src.forwarder.save_bot_user") as mock_save:
            await fwd._handle_start(start_update, mock.MagicMock())

        mock_save.assert_called_once()
        self.assertNotIn(999, fwd._pending_chats)


if __name__ == "__main__":
    unittest.main()
