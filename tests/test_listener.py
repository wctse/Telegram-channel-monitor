import asyncio
import json
import sys
import unittest
from datetime import datetime, timezone
from unittest import mock

# Stub telethon so tests run without the package installed
_tel = mock.MagicMock()
sys.modules.setdefault("telethon", _tel)
sys.modules.setdefault("telethon.events", _tel.events)

from src.listener import ChannelListener


def _make_listener():
    with mock.patch("src.listener.TelegramClient"):
        return ChannelListener(
            api_id=12345,
            api_hash="fake",
            classifier=mock.MagicMock(),
            forwarder=mock.MagicMock(),
            channels=[],
        )


def _entry(message_id, text="msg", sender_id=42, sender_name="alice"):
    return {
        "message_id": message_id,
        "text": text,
        "sender_id": sender_id,
        "sender_name": sender_name,
        "timestamp": datetime.now(timezone.utc),
    }


class TestMessageBatching(unittest.IsolatedAsyncioTestCase):
    async def test_two_messages_from_same_sender_merge(self):
        """Two messages arriving within the batch window produce a single _process_entries call with both."""
        listener = _make_listener()
        process_mock = mock.AsyncMock()
        listener._process_entries = process_mock

        await listener._enqueue_batch(100, {}, "ch", _entry(1, "first"), 0.02)
        await listener._enqueue_batch(100, {}, "ch", _entry(2, "second"), 0.02)

        await asyncio.sleep(0.06)

        process_mock.assert_called_once()
        entries = process_mock.call_args.args[3]
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["text"], "first")
        self.assertEqual(entries[1]["text"], "second")

    async def test_different_senders_get_separate_calls(self):
        """Messages from different senders are not merged."""
        listener = _make_listener()
        process_mock = mock.AsyncMock()
        listener._process_entries = process_mock

        await listener._enqueue_batch(100, {}, "ch", _entry(1, sender_id=1), 0.02)
        await listener._enqueue_batch(100, {}, "ch", _entry(2, sender_id=2), 0.02)

        await asyncio.sleep(0.06)

        self.assertEqual(process_mock.call_count, 2)
        for call in process_mock.call_args_list:
            self.assertEqual(len(call.args[3]), 1)

    async def test_new_message_resets_timer(self):
        """A second message from same sender resets the flush timer, not trigger two flushes."""
        listener = _make_listener()
        process_mock = mock.AsyncMock()
        listener._process_entries = process_mock

        await listener._enqueue_batch(100, {}, "ch", _entry(1, "a"), 0.05)
        await asyncio.sleep(0.02)
        await listener._enqueue_batch(100, {}, "ch", _entry(2, "b"), 0.05)
        await asyncio.sleep(0.1)

        process_mock.assert_called_once()
        entries = process_mock.call_args.args[3]
        self.assertEqual(len(entries), 2)

    async def test_flush_directly_calls_process_entries(self):
        """_flush_batch_after with delay=0 processes all accumulated entries immediately."""
        listener = _make_listener()
        process_mock = mock.AsyncMock()
        listener._process_entries = process_mock

        key = (100, 42)
        listener._batches[key] = {
            "entries": [_entry(1, "x"), _entry(2, "y")],
            "channel_id": 100,
            "ch_conf": {},
            "channel_name": "ch",
            "task": None,
        }

        await listener._flush_batch_after(key, 0)

        process_mock.assert_called_once()
        self.assertEqual(len(process_mock.call_args.args[3]), 2)
        self.assertNotIn(key, listener._batches)


class TestSignalThrottling(unittest.IsolatedAsyncioTestCase):
    def _make_listener_with_signal_mocks(self, tickers=None, thesis="BTC going up"):
        listener = _make_listener()
        listener.classifier.triage = mock.AsyncMock(return_value={
            "is_signal": True, "confidence": 0.9, "category": "trade_idea",
        })
        listener.classifier.enrich = mock.AsyncMock(return_value={
            "is_signal": True,
            "confidence": 0.9,
            "tickers": tickers or [{"symbol": "BTC", "bias": "bullish"}],
            "thesis": thesis,
            "category": "trade_idea",
        })
        listener.forwarder.forward_signal = mock.AsyncMock(return_value={123: 456})
        return listener

    def _existing_signal(self, tickers, thesis="BTC going up"):
        return {
            "id": 1,
            "tickers": json.dumps(tickers),
            "thesis": thesis,
            "original_texts": json.dumps([]),
            "bot_message_ids": json.dumps({}),
            "channel_name": "ch",
        }

    async def _run_process_entries(self, listener, existing_signals, mock_save_signal):
        entries = [_entry(1, "BTC 200k incoming")]
        with mock.patch("src.listener.save_message"), \
             mock.patch("src.listener.get_recent_messages", return_value=[]), \
             mock.patch("src.listener.get_sender_messages_today", return_value=[]), \
             mock.patch("src.listener.get_forwarded_signals_today", return_value=existing_signals), \
             mock_save_signal:
            await listener._process_entries(100, {}, "ch", entries)

    async def test_new_signal_is_forwarded(self):
        """No prior signals today → forward and save."""
        listener = self._make_listener_with_signal_mocks()
        mock_save = mock.patch("src.listener.save_forwarded_signal")

        with mock.patch("src.listener.save_message"), \
             mock.patch("src.listener.get_recent_messages", return_value=[]), \
             mock.patch("src.listener.get_sender_messages_today", return_value=[]), \
             mock.patch("src.listener.get_forwarded_signals_today", return_value=[]), \
             mock.patch("src.listener.save_forwarded_signal") as mock_save_fn:
            await listener._process_entries(100, {}, "ch", [_entry(1, "BTC 200k")])

        listener.forwarder.forward_signal.assert_called_once()
        mock_save_fn.assert_called_once()

    async def test_duplicate_signal_is_throttled(self):
        """Same sender+ticker+bias already forwarded today → not forwarded again."""
        listener = self._make_listener_with_signal_mocks()
        existing = [self._existing_signal([{"symbol": "BTC", "bias": "bullish"}])]

        with mock.patch("src.listener.save_message"), \
             mock.patch("src.listener.get_recent_messages", return_value=[]), \
             mock.patch("src.listener.get_sender_messages_today", return_value=[]), \
             mock.patch("src.listener.get_forwarded_signals_today", return_value=existing), \
             mock.patch("src.listener.save_forwarded_signal") as mock_save_fn:
            await listener._process_entries(100, {}, "ch", [_entry(2, "still bullish BTC")])

        listener.forwarder.forward_signal.assert_not_called()
        mock_save_fn.assert_not_called()

    async def test_different_ticker_not_throttled(self):
        """Different ticker from same sender → forwarded normally."""
        listener = self._make_listener_with_signal_mocks(
            tickers=[{"symbol": "ETH", "bias": "bullish"}],
            thesis="ETH going up",
        )
        existing = [self._existing_signal([{"symbol": "BTC", "bias": "bullish"}], thesis="BTC going up")]

        with mock.patch("src.listener.save_message"), \
             mock.patch("src.listener.get_recent_messages", return_value=[]), \
             mock.patch("src.listener.get_sender_messages_today", return_value=[]), \
             mock.patch("src.listener.get_forwarded_signals_today", return_value=existing), \
             mock.patch("src.listener.save_forwarded_signal") as mock_save_fn:
            await listener._process_entries(100, {}, "ch", [_entry(3, "ETH to 10k")])

        listener.forwarder.forward_signal.assert_called_once()
        mock_save_fn.assert_called_once()

    async def test_thesis_update_when_throttled(self):
        """Throttled signal with updated thesis triggers an edit on the existing bot message."""
        listener = self._make_listener_with_signal_mocks(
            tickers=[{"symbol": "BTC", "bias": "bullish"}],
            thesis="Updated thesis with new catalyst",
        )
        listener.forwarder.update_signal_message = mock.AsyncMock()
        existing = [self._existing_signal(
            [{"symbol": "BTC", "bias": "bullish"}],
            thesis="Old thesis",
        )]
        # bot_message_ids non-empty so update_signal_message is called
        existing[0]["bot_message_ids"] = json.dumps({"123": 456})

        with mock.patch("src.listener.save_message"), \
             mock.patch("src.listener.get_recent_messages", return_value=[]), \
             mock.patch("src.listener.get_sender_messages_today", return_value=[]), \
             mock.patch("src.listener.get_forwarded_signals_today", return_value=existing), \
             mock.patch("src.listener.save_forwarded_signal"):
            await listener._process_entries(100, {}, "ch", [_entry(4, "BTC update")])

        listener.forwarder.forward_signal.assert_not_called()
        listener.forwarder.update_signal_message.assert_called_once()


if __name__ == "__main__":
    unittest.main()
