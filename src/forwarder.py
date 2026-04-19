import logging
import os
from html import escape as html_escape
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from src.db import save_bot_user, get_bot_users, update_forwarded_signal


def _load_allowed_chat_ids() -> set[int]:
    raw = os.getenv("ALLOWED_CHAT_IDS", "")
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                ids.add(int(part))
            except ValueError:
                logging.getLogger(__name__).warning("Invalid chat ID in ALLOWED_CHAT_IDS: %s", part)
    return ids

logger = logging.getLogger(__name__)


class SignalForwarder:
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.app: Application | None = None
        self._allowed_chat_ids = _load_allowed_chat_ids()

    async def start_bot(self):
        self.app = Application.builder().token(self.bot_token).build()
        self.app.add_handler(CommandHandler("start", self._handle_start))
        self.app.add_handler(CommandHandler("status", self._handle_status))

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        logger.info("Bot started and polling for /start commands")

    async def stop_bot(self):
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        chat_id = update.effective_chat.id
        if self._allowed_chat_ids and chat_id not in self._allowed_chat_ids:
            logger.warning("Unauthorized /start from chat_id=%d (%s)", chat_id, user.username)
            await update.message.reply_text("⛔ You are not authorized to use this bot.")
            return
        save_bot_user(chat_id, user.username, user.first_name)
        logger.info("Registered user: %s (chat_id=%d)", user.username, chat_id)
        await update.message.reply_text(
            f"✅ Registered! Your chat ID: {chat_id}\n\n"
            "You'll receive high-signal messages from monitored channels here."
        )

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if self._allowed_chat_ids and chat_id not in self._allowed_chat_ids:
            await update.message.reply_text("⛔ You are not authorized to use this bot.")
            return
        users = get_bot_users()
        await update.message.reply_text(
            f"📊 Bot status:\n"
            f"- Registered users: {len(users)}\n"
            f"- Your chat ID: {chat_id}"
        )

    @staticmethod
    def _bias_icon(bias: str) -> str:
        return "🟢" if bias == "bullish" else "🔴" if bias == "bearish" else "⚪"

    @staticmethod
    def _category_icon(category: str) -> str:
        return {
            "trade_idea": "💡",
            "macro": "🌐",
            "fundamental": "📊",
            "risk_warning": "⚠️",
        }.get(category, "🔔")

    @staticmethod
    def _timeframe_icon(timeframe: str) -> str:
        return {
            "minutes": "⚡",
            "hours": "⏱",
            "days": "📅",
            "weeks": "🗓",
        }.get(timeframe, "❔")

    @staticmethod
    def _render_signal_message(classification: dict, original_texts: list[str], channel_name: str) -> str:
        raw_tickers = classification.get("tickers", [])
        confidence = classification.get("confidence", 0.0)
        category = classification.get("category", "unknown")
        timeframe = classification.get("timeframe", "") or "unspecified"
        thesis = classification.get("thesis", "")

        normalized = []
        for t in raw_tickers:
            if isinstance(t, dict):
                normalized.append((str(t.get("symbol", "?")), t.get("bias", "neutral").lower()))
            else:
                normalized.append((str(t), "neutral"))

        # Header ticker summary: up to 3 inline, rest as +N
        summary_parts = [f"{SignalForwarder._bias_icon(b)} {html_escape(s)}" for s, b in normalized[:3]]
        if len(normalized) > 3:
            summary_parts.append(f"+{len(normalized) - 3}")
        ticker_summary = " ".join(summary_parts) if summary_parts else "—"

        # Body ticker list (no redundant BULLISH/BEARISH label — dot conveys it)
        ticker_lines = [
            f"  {SignalForwarder._bias_icon(b)} <b>{html_escape(s)}</b>"
            for s, b in normalized
        ]
        tickers_block = "\n".join(ticker_lines) if ticker_lines else "  None"

        if len(original_texts) == 1:
            originals_block = html_escape(_truncate(original_texts[0], 800))
        else:
            parts = []
            per_msg_limit = max(200, 800 // len(original_texts))
            for i, t in enumerate(original_texts, 1):
                parts.append(f"[{i}] {html_escape(_truncate(t, per_msg_limit))}")
            originals_block = "\n\n".join(parts)

        cat_icon = SignalForwarder._category_icon(category)
        tf_icon = SignalForwarder._timeframe_icon(timeframe)

        msg_label = "Original messages" if len(original_texts) > 1 else "Original message"
        return (
            f"{cat_icon} {ticker_summary} · {tf_icon} {html_escape(timeframe)} — {html_escape(channel_name)}\n\n"
            f"<b>Category:</b> {html_escape(category)}\n"
            f"<b>Timeframe:</b> {tf_icon} {html_escape(timeframe)}\n"
            f"<b>Confidence:</b> {confidence:.0%}\n\n"
            f"<b>Tickers:</b>\n{tickers_block}\n\n"
            f"<b>Why:</b>\n{html_escape(thesis)}\n\n"
            f"<b>{msg_label}:</b>\n<blockquote>{originals_block}</blockquote>"
        )

    async def forward_signal(
        self,
        classification: dict,
        original_text: str | list[str],
        channel_name: str,
    ) -> dict[int, int]:
        """Send signal to all bot users. Returns {chat_id: message_id} for sent messages.

        `original_text` may be a single string or a list of strings (batched messages).
        """
        users = get_bot_users()
        if not users:
            logger.warning("No registered bot users to forward to")
            return {}

        original_texts = original_text if isinstance(original_text, list) else [original_text]
        message = self._render_signal_message(classification, original_texts, channel_name)
        sent = {}
        for chat_id in users:
            try:
                msg = await self.app.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode="HTML",
                )
                sent[chat_id] = msg.message_id
            except Exception as e:
                logger.error("Failed to send to chat_id=%d: %s", chat_id, e)
        return sent

    async def update_signal_message(
        self,
        signal_id: int,
        bot_message_ids: dict,
        classification: dict,
        original_texts: list[str],
        channel_name: str,
    ):
        """Edit existing bot messages with updated thesis/reasons."""
        message = self._render_signal_message(classification, original_texts, channel_name)
        for chat_id_str, msg_id in bot_message_ids.items():
            try:
                await self.app.bot.edit_message_text(
                    chat_id=int(chat_id_str),
                    message_id=msg_id,
                    text=message,
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.error("Failed to edit message for chat_id=%s: %s", chat_id_str, e)
        update_forwarded_signal(signal_id, classification.get("thesis", ""), original_texts)


    _ERROR_ALERT_CHAT_ID = 115436546

    async def send_error_alert(self, error_message: str):
        """Send an error alert to the admin chat."""
        if not self.app or not self.app.bot:
            return

        text = (
            f"⚠️ <b>Error Alert</b>\n\n"
            f"<pre>{html_escape(_truncate(error_message, 3000))}</pre>"
        )

        try:
            await self.app.bot.send_message(
                chat_id=self._ERROR_ALERT_CHAT_ID,
                text=text,
                parse_mode="HTML",
            )
        except Exception:
            pass  # Avoid recursion — don't log errors from error alerting


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
