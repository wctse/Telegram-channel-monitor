import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from unittest import mock

from src.db import init_db, save_message, save_bot_user, get_recent_messages, get_bot_users, get_connection, DB_PATH


class DBTestCase(unittest.TestCase):
    """Base class that redirects DB_PATH to a temporary file for each test."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._patcher = mock.patch("src.db.DB_PATH", self._tmp.name)
        self._patcher.start()
        init_db()

    def tearDown(self):
        self._patcher.stop()
        os.unlink(self._tmp.name)


class TestInitDB(DBTestCase):
    def test_tables_exist(self):
        conn = get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r["name"] for r in tables}
        self.assertIn("messages", names)
        self.assertIn("bot_users", names)
        conn.close()

    def test_idempotent(self):
        init_db()  # second call should not raise
        conn = get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        self.assertTrue(len(tables) >= 2)
        conn.close()


class TestSaveMessage(DBTestCase):
    def _make_ts(self, **kwargs):
        return datetime(2025, 4, 5, 12, 0, 0, tzinfo=timezone.utc) + timedelta(**kwargs)

    def test_insert_without_classification(self):
        save_message(
            channel_id=100, channel_name="test", message_id=1,
            sender_id=None, sender_name=None, text="hello",
            timestamp=self._make_ts(),
        )
        conn = get_connection()
        row = conn.execute("SELECT * FROM messages WHERE message_id=1").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["text"], "hello")
        self.assertIsNone(row["is_signal"])
        conn.close()

    def test_insert_with_classification(self):
        cls = {"is_signal": True, "confidence": 0.9, "category": "trade_idea",
               "thesis": "BTC going up", "tickers": [{"symbol": "BTC", "bias": "bullish"}]}
        save_message(
            channel_id=100, channel_name="test", message_id=2,
            sender_id=42, sender_name="alice", text="signal msg",
            timestamp=self._make_ts(), classification=cls,
        )
        conn = get_connection()
        row = conn.execute("SELECT * FROM messages WHERE message_id=2").fetchone()
        self.assertTrue(row["is_signal"])
        self.assertAlmostEqual(row["confidence"], 0.9)
        self.assertEqual(row["thesis"], "BTC going up")
        conn.close()

    def test_upsert_preserves_id(self):
        """ON CONFLICT DO UPDATE should keep the original row id and created_at."""
        ts = self._make_ts()
        save_message(
            channel_id=100, channel_name="test", message_id=3,
            sender_id=None, sender_name=None, text="v1",
            timestamp=ts,
        )
        conn = get_connection()
        row1 = conn.execute("SELECT id, created_at FROM messages WHERE message_id=3").fetchone()
        conn.close()

        # Update with enrichment
        cls = {"is_signal": True, "confidence": 0.95, "category": "macro",
               "thesis": "enriched", "tickers": []}
        save_message(
            channel_id=100, channel_name="test", message_id=3,
            sender_id=None, sender_name=None, text="v1",
            timestamp=ts, classification=cls,
        )
        conn = get_connection()
        row2 = conn.execute("SELECT id, created_at, thesis FROM messages WHERE message_id=3").fetchone()
        conn.close()

        self.assertEqual(row1["id"], row2["id"])
        self.assertEqual(row1["created_at"], row2["created_at"])
        self.assertEqual(row2["thesis"], "enriched")

    def test_timestamp_format_compatible_with_sqlite(self):
        """Stored timestamps must use SQLite-compatible format (space separator, no TZ)."""
        ts = datetime(2025, 4, 5, 10, 30, 0, tzinfo=timezone.utc)
        save_message(
            channel_id=100, channel_name="test", message_id=4,
            sender_id=None, sender_name=None, text="ts test",
            timestamp=ts,
        )
        conn = get_connection()
        row = conn.execute("SELECT timestamp FROM messages WHERE message_id=4").fetchone()
        stored = row["timestamp"]
        self.assertEqual(stored, "2025-04-05 10:30:00")
        self.assertNotIn("T", stored)
        self.assertNotIn("+", stored)
        conn.close()


class TestGetRecentMessages(DBTestCase):
    def test_returns_recent_only(self):
        now = datetime.now(timezone.utc)
        # Old message (2 hours ago)
        save_message(
            channel_id=100, channel_name="test", message_id=1,
            sender_id=None, sender_name=None, text="old",
            timestamp=now - timedelta(hours=2),
        )
        # Recent message (5 minutes ago)
        save_message(
            channel_id=100, channel_name="test", message_id=2,
            sender_id=None, sender_name=None, text="recent",
            timestamp=now - timedelta(minutes=5),
        )
        results = get_recent_messages(100, lookback_minutes=30)
        self.assertIn("recent", results)
        self.assertNotIn("old", results)

    def test_filters_by_channel(self):
        now = datetime.now(timezone.utc)
        save_message(
            channel_id=100, channel_name="ch1", message_id=1,
            sender_id=None, sender_name=None, text="ch1 msg",
            timestamp=now - timedelta(minutes=5),
        )
        save_message(
            channel_id=200, channel_name="ch2", message_id=2,
            sender_id=None, sender_name=None, text="ch2 msg",
            timestamp=now - timedelta(minutes=5),
        )
        results = get_recent_messages(100, lookback_minutes=30)
        self.assertEqual(results, ["ch1 msg"])

    def test_empty_text_excluded(self):
        now = datetime.now(timezone.utc)
        save_message(
            channel_id=100, channel_name="test", message_id=1,
            sender_id=None, sender_name=None, text="",
            timestamp=now - timedelta(minutes=5),
        )
        save_message(
            channel_id=100, channel_name="test", message_id=2,
            sender_id=None, sender_name=None, text="real",
            timestamp=now - timedelta(minutes=5),
        )
        results = get_recent_messages(100, lookback_minutes=30)
        self.assertEqual(results, ["real"])

    def test_respects_limit(self):
        now = datetime.now(timezone.utc)
        for i in range(10):
            save_message(
                channel_id=100, channel_name="test", message_id=i,
                sender_id=None, sender_name=None, text=f"msg{i}",
                timestamp=now - timedelta(minutes=1),
            )
        results = get_recent_messages(100, lookback_minutes=30, limit=3)
        self.assertEqual(len(results), 3)


class TestBotUsers(DBTestCase):
    def test_save_and_get(self):
        save_bot_user(111, "alice", "Alice")
        save_bot_user(222, "bob", "Bob")
        users = get_bot_users()
        self.assertEqual(sorted(users), [111, 222])

    def test_upsert(self):
        save_bot_user(111, "alice", "Alice")
        save_bot_user(111, "alice_v2", "Alice V2")
        users = get_bot_users()
        self.assertEqual(users, [111])

    def test_empty(self):
        self.assertEqual(get_bot_users(), [])


if __name__ == "__main__":
    unittest.main()
