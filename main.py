import asyncio
import logging
import os
import re
import sys
import time
from collections import deque

import yaml
from dotenv import load_dotenv

from src.db import init_db
from src.classifier import LLMClassifier
from src.forwarder import SignalForwarder
from src.listener import ChannelListener


class TelegramErrorHandler(logging.Handler):
    """Logging handler that sends ERROR+ messages to Telegram via the forwarder.

    - Rate-limited to MAX_PER_MINUTE messages per 60-second sliding window.
    - Identical messages (same logger + message template) are silenced after
      the first occurrence; a count is appended when a *new* error arrives.
    - Sensitive patterns (API keys, tokens) are redacted before sending.
    """

    MAX_PER_MINUTE = 10
    WINDOW_SECONDS = 60
    _REDACT_RE = re.compile(
        r'(Bearer\s+)\S+|'            # Authorization headers
        r'(api[_-]?key["\s:=]+)\S+|'   # api_key=... / api-key: ...
        r'(token["\s:=]+)\S+',         # token=...
        re.IGNORECASE,
    )

    def __init__(self, forwarder: "SignalForwarder"):
        super().__init__(level=logging.ERROR)
        self.forwarder = forwarder
        self._send_times: deque[float] = deque()
        self._seen: dict[str, int] = {}  # msg_key -> suppressed count
        self._suppressed_total: int = 0

    _POLLING_SUPPRESS = "Exception happened while polling"

    def emit(self, record: logging.LogRecord):
        # Suppress transient polling network errors (auto-retried by the library)
        if record.name.startswith("telegram.ext") and self._POLLING_SUPPRESS in record.getMessage():
            return

        now = time.monotonic()

        # --- dedup: key on logger name + unformatted message template ---
        msg_key = f"{record.name}:{record.getMessage()}"
        if msg_key in self._seen:
            self._seen[msg_key] += 1
            self._suppressed_total += 1
            return
        self._seen[msg_key] = 0

        # --- sliding-window rate limit ---
        while self._send_times and self._send_times[0] <= now - self.WINDOW_SECONDS:
            self._send_times.popleft()
        if len(self._send_times) >= self.MAX_PER_MINUTE:
            self._suppressed_total += 1
            return

        try:
            self._send_times.append(now)
            message = self._redact(self.format(record))

            # Append suppressed summary if any
            suppressed = self._suppressed_total
            seen_dupes = {k: v for k, v in self._seen.items() if v > 0}
            self._suppressed_total = 0
            self._seen.clear()
            self._seen[msg_key] = 0  # keep current key registered

            footer_parts = []
            if suppressed:
                footer_parts.append(f"+{suppressed} suppressed error(s) since last alert")
            if seen_dupes:
                footer_parts.append(
                    "silenced repeats: "
                    + ", ".join(f"{k.split(':', 1)[-1][:80]}... x{v}" for k, v in seen_dupes.items())
                )
            if footer_parts:
                message += "\n\n(" + " | ".join(footer_parts) + ")"

            loop = asyncio.get_running_loop()
            loop.create_task(self.forwarder.send_error_alert(message))
        except RuntimeError:
            pass  # no running loop yet

    @classmethod
    def _redact(cls, text: str) -> str:
        return cls._REDACT_RE.sub(lambda m: m.group(0).split()[0] + " [REDACTED]" if m.group(0).split() else "[REDACTED]", text)

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("monitor.log"),
    ],
)

# Silence noisy libraries at root level
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("telethon").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


async def main():
    # Validate env vars
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not api_id or not api_hash:
        logger.error("TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env")
        sys.exit(1)
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN must be set in .env")
        sys.exit(1)

    config = load_config()

    # Init database
    init_db()

    # Init classifier
    llm_conf = config.get("llm", {})
    classifier_conf = config.get("classifier", {})
    provider = llm_conf.get("provider", "ollama")
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OLLAMA_HOST") or llm_conf.get("base_url", "http://localhost:11434")
    classifier = LLMClassifier(
        provider=provider,
        base_url=base_url,
        model=llm_conf.get("model", "qwen3:4b"),
        triage_prompt=classifier_conf.get("triage_prompt", "Classify as signal or noise."),
        enrich_prompt=classifier_conf.get("enrich_prompt", "Analyze the signal in detail."),
        api_key=api_key,
        timeout=llm_conf.get("timeout", 120),
        fallback_model=llm_conf.get("fallback_model"),
    )

    # Init forwarder bot
    bot_conf = config.get("bot", {})
    forwarder = SignalForwarder(
        bot_token,
        allowed_groups=bot_conf.get("allowed_groups", []),
    )

    # Init channel listener
    channels = config.get("channels", [])
    listener_conf = config.get("listener", {})
    listener = ChannelListener(
        api_id=int(api_id),
        api_hash=api_hash,
        classifier=classifier,
        forwarder=forwarder,
        channels=channels,
        max_reconnect_attempts=listener_conf.get("max_reconnect_attempts", 5),
        reconnect_delay_seconds=listener_conf.get("reconnect_delay_seconds", 10),
        max_disconnects_in_window=listener_conf.get("max_disconnects_in_window", 5),
        disconnect_window_seconds=listener_conf.get("disconnect_window_seconds", 600),
    )

    # Start bot and listener concurrently
    try:
        await classifier.open()
        await forwarder.start_bot()

        # Attach Telegram error alerts if enabled
        if config.get("error_alerts", {}).get("enabled", True):
            error_handler = TelegramErrorHandler(forwarder)
            error_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logging.getLogger().addHandler(error_handler)
            logger.info("Telegram error alerts enabled")

        logger.info("Bot is running. Send /start to your bot to register for alerts.")
        await listener.start()
    finally:
        logger.info("Shutting down...")
        await forwarder.stop_bot()
        await listener.stop()
        await classifier.close()


if __name__ == "__main__":
    asyncio.run(main())
