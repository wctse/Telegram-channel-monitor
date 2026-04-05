import asyncio
import logging
import os
import sys

import yaml
from dotenv import load_dotenv

from src.db import init_db
from src.classifier import LLMClassifier
from src.forwarder import SignalForwarder
from src.listener import ChannelListener

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
    )

    # Init forwarder bot
    forwarder = SignalForwarder(bot_token)

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
        logger.info("Bot is running. Send /start to your bot to register for alerts.")
        await listener.start()
    finally:
        logger.info("Shutting down...")
        await forwarder.stop_bot()
        await listener.stop()
        await classifier.close()


if __name__ == "__main__":
    asyncio.run(main())
