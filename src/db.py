import sqlite3
import os
import json
from datetime import datetime


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "messages.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id INTEGER NOT NULL,
            channel_name TEXT,
            message_id INTEGER NOT NULL,
            sender_id INTEGER,
            sender_name TEXT,
            text TEXT,
            timestamp DATETIME NOT NULL,
            is_signal BOOLEAN,
            confidence REAL,
            thesis TEXT,
            tickers TEXT,
            category TEXT,
            raw_classification TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(channel_id, message_id)
        );

        CREATE TABLE IF NOT EXISTS bot_users (
            chat_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            registered_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel_id);
        CREATE INDEX IF NOT EXISTS idx_messages_signal ON messages(is_signal);
        CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);

        CREATE TABLE IF NOT EXISTS forwarded_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_key TEXT NOT NULL,
            channel_id INTEGER NOT NULL,
            channel_name TEXT,
            signal_date TEXT NOT NULL,
            tickers TEXT NOT NULL,
            thesis TEXT,
            original_texts TEXT,
            bot_message_ids TEXT,
            confidence REAL,
            category TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_forwarded_signals_lookup
            ON forwarded_signals(sender_key, signal_date);
    """)
    conn.commit()
    conn.close()


def save_message(
    channel_id: int,
    channel_name: str,
    message_id: int,
    sender_id: int | None,
    sender_name: str | None,
    text: str,
    timestamp: datetime,
    classification: dict | None = None,
):
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO messages
            (channel_id, channel_name, message_id, sender_id, sender_name, text, timestamp,
             is_signal, confidence, thesis, tickers, category, raw_classification)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(channel_id, message_id) DO UPDATE SET
                channel_name = excluded.channel_name,
                sender_id = excluded.sender_id,
                sender_name = excluded.sender_name,
                text = excluded.text,
                timestamp = excluded.timestamp,
                is_signal = excluded.is_signal,
                confidence = excluded.confidence,
                thesis = excluded.thesis,
                tickers = excluded.tickers,
                category = excluded.category,
                raw_classification = excluded.raw_classification
            """,
            (
                channel_id,
                channel_name,
                message_id,
                sender_id,
                sender_name,
                text,
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                classification.get("is_signal") if classification else None,
                classification.get("confidence") if classification else None,
                classification.get("thesis") if classification else None,
                json.dumps(classification.get("tickers", [])) if classification else None,
                classification.get("category") if classification else None,
                json.dumps(classification) if classification else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def save_bot_user(chat_id: int, username: str | None, first_name: str | None):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO bot_users (chat_id, username, first_name) VALUES (?, ?, ?)",
            (chat_id, username, first_name),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_messages(channel_id: int, lookback_minutes: int = 30, limit: int = 50) -> list[str]:
    """Retrieve recent message texts from a channel within the lookback window."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT text FROM messages
            WHERE channel_id = ?
              AND timestamp >= datetime('now', ? || ' minutes')
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (channel_id, f"-{lookback_minutes}", limit),
        ).fetchall()
        return [row["text"] for row in rows if row["text"]]
    finally:
        conn.close()


def get_forwarded_signals_today(sender_key: str, signal_date: str) -> list[dict]:
    """Get all forwarded signals from this sender on this date."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM forwarded_signals WHERE sender_key = ? AND signal_date = ?",
            (sender_key, signal_date),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def save_forwarded_signal(
    sender_key: str,
    channel_id: int,
    channel_name: str,
    signal_date: str,
    tickers: list[dict],
    thesis: str,
    original_texts: list[str],
    bot_message_ids: dict,
    confidence: float,
    category: str,
) -> int:
    conn = get_connection()
    try:
        cursor = conn.execute(
            """INSERT INTO forwarded_signals
               (sender_key, channel_id, channel_name, signal_date, tickers, thesis,
                original_texts, bot_message_ids, confidence, category)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sender_key, channel_id, channel_name, signal_date,
                json.dumps(tickers), thesis, json.dumps(original_texts),
                json.dumps(bot_message_ids), confidence, category,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def update_forwarded_signal(signal_id: int, thesis: str, original_texts: list[str]):
    """Update thesis and original texts for an existing forwarded signal."""
    conn = get_connection()
    try:
        conn.execute(
            """UPDATE forwarded_signals
               SET thesis = ?, original_texts = ?, updated_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (thesis, json.dumps(original_texts), signal_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_sender_messages_today(sender_id: int | None, channel_id: int) -> list[str]:
    """Get messages from the same sender in this channel today (UTC)."""
    conn = get_connection()
    try:
        if sender_id is not None:
            rows = conn.execute(
                """SELECT text FROM messages
                   WHERE channel_id = ? AND sender_id = ?
                     AND date(timestamp) = date('now')
                   ORDER BY timestamp ASC""",
                (channel_id, sender_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT text FROM messages
                   WHERE channel_id = ?
                     AND date(timestamp) = date('now')
                   ORDER BY timestamp ASC""",
                (channel_id,),
            ).fetchall()
        return [row["text"] for row in rows if row["text"]]
    finally:
        conn.close()


def get_bot_users() -> list[int]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT chat_id FROM bot_users").fetchall()
        return [row["chat_id"] for row in rows]
    finally:
        conn.close()
