import logging
import os
from html import escape as html_escape
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from src.db import save_bot_user, get_bot_users


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

    async def forward_signal(self, classification: dict, original_text: str, channel_name: str):
        users = get_bot_users()
        if not users:
            logger.warning("No registered bot users to forward to")
            return

        raw_tickers = classification.get("tickers", [])
        confidence = classification.get("confidence", 0.0)
        category = classification.get("category", "unknown")
        thesis = classification.get("thesis", "")

        ticker_lines = []
        for t in raw_tickers:
            if isinstance(t, dict):
                symbol = html_escape(str(t.get("symbol", "?")))
                bias = t.get("bias", "neutral").lower()
                icon = "🟢" if bias == "bullish" else "🔴" if bias == "bearish" else "⚪"
                ticker_lines.append(f"  {icon} <b>{symbol}</b> — {bias.upper()}")
            else:
                ticker_lines.append(f"  ⚪ <b>{html_escape(str(t))}</b> — NEUTRAL")
        tickers_block = "\n".join(ticker_lines) if ticker_lines else "  None"

        message = (
            f"🔔 <b>Signal Detected</b> — {html_escape(channel_name)}\n\n"
            f"<b>Category:</b> {html_escape(category)}\n"
            f"<b>Confidence:</b> {confidence:.0%}\n\n"
            f"<b>Tickers:</b>\n{tickers_block}\n\n"
            f"<b>Why:</b>\n{html_escape(thesis)}\n\n"
            f"<b>Original message:</b>\n<blockquote>{html_escape(_truncate(original_text, 800))}</blockquote>"
        )

        for chat_id in users:
            try:
                await self.app.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.error("Failed to send to chat_id=%d: %s", chat_id, e)


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
