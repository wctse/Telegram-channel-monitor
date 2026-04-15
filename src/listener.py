import asyncio
import json
import time
import logging
import sqlite3
from collections import deque
from datetime import datetime, timezone

from telethon import TelegramClient, events

from src.classifier import LLMClassifier
from src.forwarder import SignalForwarder
from src.db import save_message, get_recent_messages, get_sender_messages_today, get_forwarded_signals_today, save_forwarded_signal, update_forwarded_signal

logger = logging.getLogger(__name__)


class ChannelListener:
    def __init__(
        self,
        api_id: int,
        api_hash: str,
        classifier: LLMClassifier,
        forwarder: SignalForwarder,
        channels: list[dict],
        max_reconnect_attempts: int = 5,
        reconnect_delay_seconds: int = 10,
        max_disconnects_in_window: int = 5,
        disconnect_window_seconds: int = 600,
    ):
        self.client = TelegramClient(
            "session",
            api_id,
            api_hash,
            auto_reconnect=False,
            connection_retries=1,
        )
        self.classifier = classifier
        self.forwarder = forwarder
        self.channels = channels
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.max_disconnects_in_window = max_disconnects_in_window
        self.disconnect_window_seconds = disconnect_window_seconds
        self._channel_map: dict[int, dict] = {}
        self._handlers_registered = False
        self._stopping = False
        self._disconnect_timestamps: deque[float] = deque()
        self._raw_update_counts: dict[str, int] = {}
        self._last_heartbeat: float = 0.0
        self._heartbeat_interval: float = 3600.0

    async def start(self):
        self._stopping = False
        self._clear_update_state()
        reconnect_attempt = 0

        while not self._stopping:
            try:
                await self.client.start()
                logger.info("Telethon client connected")
                reconnect_attempt = 0

                if not self._channel_map:
                    await self._build_channel_map()

                if not self._channel_map:
                    logger.error("No channels to monitor. Check config.yaml and ensure channel IDs are set.")
                    return

                if not self._handlers_registered:
                    self._register_handlers()

                logger.info("Listening for new messages in %d channel(s)", len(self._channel_map))
                logger.info("Monitored norm_ids: %s", {self._normalize_id(cid) for cid in self._channel_map})
                await self.client.run_until_disconnected()

                if self._stopping:
                    break

                disconnect_count = self._record_disconnect()
                if self.max_disconnects_in_window > 0 and disconnect_count >= self.max_disconnects_in_window:
                    raise RuntimeError(
                        "Telethon disconnected %d time(s) within %d second(s)"
                        % (disconnect_count, self.disconnect_window_seconds)
                    )

                raise ConnectionError("Telethon client disconnected unexpectedly")

            except RuntimeError as e:
                logger.error("%s. Stopping listener.", e)
                raise

            except Exception as e:
                if self._stopping:
                    break

                reconnect_attempt += 1
                if reconnect_attempt > self.max_reconnect_attempts:
                    logger.error(
                        "Telethon reconnect failed after %d consecutive attempts. Stopping listener.",
                        self.max_reconnect_attempts,
                    )
                    raise RuntimeError("Telethon reconnect limit reached") from e

                delay_seconds = self.reconnect_delay_seconds * reconnect_attempt
                logger.warning(
                    "Connection lost (%s). Reconnect attempt %d/%d in %d second(s).",
                    e,
                    reconnect_attempt,
                    self.max_reconnect_attempts,
                    delay_seconds,
                )
                await asyncio.sleep(delay_seconds)

    def _record_disconnect(self) -> int:
        now = time.monotonic()
        self._disconnect_timestamps.append(now)
        window_start = now - self.disconnect_window_seconds

        while self._disconnect_timestamps and self._disconnect_timestamps[0] < window_start:
            self._disconnect_timestamps.popleft()

        return len(self._disconnect_timestamps)

    async def _build_channel_map(self):
        self._channel_map.clear()

        for ch_conf in self.channels:
            if not ch_conf.get("enabled", True):
                continue

            ch_id = ch_conf.get("id")
            ch_name = ch_conf.get("name", "Unknown")

            if ch_id is None:
                resolve_target = ch_conf.get("username") or ch_name
                resolved = await self._resolve_channel(resolve_target)
                if resolved is None:
                    logger.warning("Could not resolve channel: %s — skipping", resolve_target)
                    continue
                ch_id = resolved

            self._channel_map[ch_id] = ch_conf
            logger.info("Monitoring channel: %s (raw_id=%d, norm_id=%d)", ch_name, ch_id, self._normalize_id(ch_id))

    def _register_handlers(self):
        monitored_norm_ids = {self._normalize_id(cid) for cid in self._channel_map}

        @self.client.on(events.Raw())
        async def on_raw_update(update):
            update_type = type(update).__name__
            chat_id = None
            if hasattr(update, 'message') and hasattr(update.message, 'peer_id'):
                peer = update.message.peer_id
                if hasattr(peer, 'channel_id'):
                    chat_id = peer.channel_id
                elif hasattr(peer, 'chat_id'):
                    chat_id = peer.chat_id
                elif hasattr(peer, 'user_id'):
                    chat_id = peer.user_id
            elif hasattr(update, 'channel_id'):
                chat_id = update.channel_id

            if chat_id and self._normalize_id(chat_id) in monitored_norm_ids:
                self._raw_update_counts[update_type] = self._raw_update_counts.get(update_type, 0) + 1
                self._maybe_log_heartbeat()

        @self.client.on(events.NewMessage())
        async def on_new_message(event):
            await self._handle_message(event)

        self._handlers_registered = True

    def _maybe_log_heartbeat(self):
        now = time.monotonic()
        if now - self._last_heartbeat < self._heartbeat_interval:
            return
        self._last_heartbeat = now
        total = sum(self._raw_update_counts.values())
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(self._raw_update_counts.items()))
        logger.info("💓 Heartbeat: %d raw updates in last hour (%s)", total, breakdown)
        self._raw_update_counts.clear()

    @staticmethod
    def _clear_update_state():
        """Clear saved update state so Telethon doesn't spend ages catching up
        on hundreds of channels via getChannelDifference on reconnect."""
        try:
            conn = sqlite3.connect("session.session")
            conn.execute("DELETE FROM update_state")
            conn.commit()
            conn.close()
            logger.info("Cleared session update_state to skip catch-up")
        except Exception as e:
            logger.warning("Could not clear update_state: %s", e)

    async def stop(self):
        self._stopping = True
        await self.client.disconnect()

    async def _resolve_channel(self, identifier) -> int | None:
        if isinstance(identifier, str):
            # If it's a username or t.me link, try getting the entity directly
            if identifier.startswith("@") or "t.me/" in identifier or identifier.isalnum():
                try:
                    entity = await self.client.get_entity(identifier)
                    logger.info("Resolved '%s' -> id=%d via get_entity", identifier, entity.id)
                    return entity.id
                except Exception as e:
                    logger.debug("Could not get entity for '%s': %s", identifier, e)

        # Fallback: search through dialogs by name
        name = str(identifier)
        try:
            async for dialog in self.client.iter_dialogs():
                if dialog.name and dialog.name.strip().lower() == name.strip().lower():
                    logger.info("Resolved '%s' -> id=%d via dialog search", name, dialog.id)
                    return dialog.id
        except Exception as e:
            logger.error("Error resolving channel '%s' via dialogs: %s", name, e)
            
        return None

    # Normalizes IDs (e.g. removes -100 prefix if present) so we can match them reliably
    def _normalize_id(self, chat_id: int) -> int:
        s_id = str(chat_id)
        if s_id.startswith("-100"):
            return int(s_id[4:])
        if s_id.startswith("-"):
            return int(s_id[1:])
        return chat_id

    async def _handle_message(self, event):
        msg = event.message
        text = msg.text or msg.raw_text or ""

        if not text.strip():
            return

        raw_id = event.chat_id
        if raw_id is None:
            return
            
        # Try to find a match in our config using both raw and normalized IDs
        norm_id = self._normalize_id(raw_id)
        
        ch_conf = None
        matched_id = None
        
        for cfg_id, conf in self._channel_map.items():
            if self._normalize_id(cfg_id) == norm_id:
                ch_conf = conf
                matched_id = cfg_id
                break

        if not ch_conf:
            return  # Message is from an unmonitored channel

        channel_name = ch_conf.get("name", str(matched_id))
        threshold = ch_conf.get("confidence_threshold", 0.7)
        lookback = ch_conf.get("context_lookback_minutes", 30)

        sender_id = None
        sender_name = None
        if msg.sender:
            sender_id = msg.sender_id
            sender_name = getattr(msg.sender, "username", None) or getattr(
                msg.sender, "first_name", None
            )
        timestamp = msg.date or datetime.now(timezone.utc)

        # Stage 1: Triage — quick signal/noise check
        logger.info("📥 Triaging message from '%s': %.80s...", channel_name, text.replace('\n', ' '))
        triage_result = await self.classifier.triage(text)

        # Save to DB with triage result
        save_message(
            channel_id=matched_id,
            channel_name=channel_name,
            message_id=msg.id,
            sender_id=sender_id,
            sender_name=sender_name,
            text=text,
            timestamp=timestamp,
            classification=triage_result,
        )

        if not triage_result or not triage_result.get("is_signal") or triage_result.get("confidence", 0) < threshold:
            conf = triage_result.get("confidence", 0) if triage_result else 0
            cat = triage_result.get("category", "error") if triage_result else "error"
            logger.info("⬜ NOISE (%.0f%% confidence) in %s | Category: %s", conf * 100, channel_name, cat)
            return

        # Stage 2: Enrich — pull context and sender history for full synthesis
        logger.info("🔍 Signal detected in '%s', enriching with %d-min context...", channel_name, lookback)
        context_texts = get_recent_messages(matched_id, lookback_minutes=lookback)
        sender_history = get_sender_messages_today(sender_id, matched_id)
        enriched = await self.classifier.enrich(text, context_texts, sender_history_texts=sender_history)

        if enriched and enriched.get("is_signal") and enriched.get("confidence", 0) >= threshold:
            # Update DB record with enriched classification
            save_message(
                channel_id=matched_id,
                channel_name=channel_name,
                message_id=msg.id,
                sender_id=sender_id,
                sender_name=sender_name,
                text=text,
                timestamp=timestamp,
                classification=enriched,
            )

            # Throttle check: same sender + same day + same ticker + same side
            sender_key = str(sender_id) if sender_id else f"ch_{matched_id}"
            signal_date = timestamp.strftime("%Y-%m-%d")
            enriched_tickers = enriched.get("tickers", [])

            existing_signals = get_forwarded_signals_today(sender_key, signal_date)
            already_forwarded = set()
            for sig in existing_signals:
                try:
                    for t in json.loads(sig["tickers"]):
                        already_forwarded.add((t["symbol"].upper(), t["bias"].lower()))
                except (json.JSONDecodeError, KeyError):
                    pass

            new_combos = [
                t for t in enriched_tickers
                if (t.get("symbol", "").upper(), t.get("bias", "neutral").lower()) not in already_forwarded
            ]

            if new_combos:
                # At least one new ticker+bias — forward as new signal
                logger.info(
                    "🔔 SIGNAL (%.0f%% confidence) in %s: %s",
                    enriched["confidence"] * 100,
                    channel_name,
                    enriched.get("thesis", "")[:150],
                )
                bot_msg_ids = await self.forwarder.forward_signal(enriched, text, channel_name)
                save_forwarded_signal(
                    sender_key=sender_key,
                    channel_id=matched_id,
                    channel_name=channel_name,
                    signal_date=signal_date,
                    tickers=enriched_tickers,
                    thesis=enriched.get("thesis", ""),
                    original_texts=[text],
                    bot_message_ids={str(k): v for k, v in bot_msg_ids.items()},
                    confidence=enriched.get("confidence", 0),
                    category=enriched.get("category", ""),
                )
            else:
                # All ticker+bias combos already forwarded today — throttle
                logger.info(
                    "🔇 Throttled signal from sender=%s in %s (all ticker+bias already forwarded today)",
                    sender_key, channel_name,
                )
                # Find the best matching existing signal to update
                new_thesis = enriched.get("thesis", "")
                best_match = self._find_best_matching_signal(existing_signals, enriched_tickers)
                if best_match and new_thesis and new_thesis != best_match.get("thesis", ""):
                    old_texts = json.loads(best_match.get("original_texts", "[]"))
                    old_texts.append(text)
                    bot_msg_ids = json.loads(best_match.get("bot_message_ids", "{}"))
                    updated_classification = {
                        "tickers": json.loads(best_match["tickers"]),
                        "confidence": enriched.get("confidence", 0),
                        "category": best_match.get("category", ""),
                        "thesis": new_thesis,
                    }
                    if bot_msg_ids:
                        await self.forwarder.update_signal_message(
                            best_match["id"], bot_msg_ids, updated_classification,
                            old_texts, best_match.get("channel_name", channel_name),
                        )
                        logger.info("📝 Updated throttled signal (id=%d) with new thesis", best_match["id"])
                    else:
                        update_forwarded_signal(best_match["id"], new_thesis, old_texts)
                        logger.info("📝 Updated thesis for throttled signal (id=%d), no bot messages to edit", best_match["id"])
        else:
            conf = enriched.get("confidence", 0) if enriched else 0
            logger.info("⬜ Enrichment downgraded signal in %s (%.0f%% confidence)", channel_name, conf * 100)

    @staticmethod
    def _find_best_matching_signal(existing_signals: list[dict], tickers: list[dict]) -> dict | None:
        """Find the existing forwarded signal with the most ticker overlap."""
        target = {(t.get("symbol", "").upper(), t.get("bias", "neutral").lower()) for t in tickers}
        best = None
        best_overlap = 0
        for sig in existing_signals:
            try:
                sig_tickers = {(t["symbol"].upper(), t["bias"].lower()) for t in json.loads(sig["tickers"])}
            except (json.JSONDecodeError, KeyError):
                continue
            overlap = len(target & sig_tickers)
            if overlap > best_overlap:
                best_overlap = overlap
                best = sig
        return best
