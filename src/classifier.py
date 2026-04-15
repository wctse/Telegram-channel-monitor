import asyncio
import json
import logging
import aiohttp

logger = logging.getLogger(__name__)


class LLMClassifier:
    """Supports two providers: 'ollama' (local) and 'api' (OpenRouter, Groq, etc.)."""

    def __init__(
        self,
        provider: str,
        base_url: str,
        model: str,
        triage_prompt: str,
        enrich_prompt: str,
        api_key: str | None = None,
        timeout: int = 120,
        fallback_model: str | None = None,
    ):
        self.provider = provider  # "ollama" or "api"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback_model = fallback_model
        self.triage_prompt = triage_prompt
        self.enrich_prompt = enrich_prompt
        self.api_key = api_key
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None

    async def open(self):
        """Create the shared HTTP session. Call once at startup."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(connect=10, sock_read=self.timeout),
            )

    async def close(self):
        """Close the shared HTTP session. Call once at shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            raise RuntimeError("LLMClassifier session not open. Call await classifier.open() first.")
        return self._session

    async def triage(self, message_text: str) -> dict | None:
        """Stage 1: Quick per-message classification — is this signal or noise?"""
        if not message_text or not message_text.strip():
            return None
        raw = await self._llm_call(self.triage_prompt, message_text)
        if raw is None:
            return None
        return self._parse_triage(raw)

    async def enrich(self, signal_text: str, context_texts: list[str], sender_history_texts: list[str] | None = None) -> dict | None:
        """Stage 2: Given a signal message and surrounding context, produce a full synthesis."""
        parts = []
        if sender_history_texts:
            parts.append("PREVIOUS MESSAGES FROM THIS SENDER TODAY (oldest first):")
            for i, t in enumerate(sender_history_texts, 1):
                parts.append(f"--- sender message {i} ---\n{t}")
            parts.append("")
        if context_texts:
            parts.append("RECENT CONTEXT FROM THIS CHANNEL (oldest first):")
            for i, t in enumerate(context_texts, 1):
                parts.append(f"--- message {i} ---\n{t}")
            parts.append("")
        parts.append("SIGNAL MESSAGE (classify this):")
        parts.append(signal_text)
        user_content = "\n".join(parts)

        raw = await self._llm_call(self.enrich_prompt, user_content)
        if raw is None:
            return None
        return self._parse_enrich(raw)

    async def _llm_call(self, system_prompt: str, user_content: str) -> str | None:
        """Route to the correct provider and return raw LLM content string."""
        if self.provider == "api":
            return await self._call_api(system_prompt, user_content)
        else:
            return await self._call_ollama(system_prompt, user_content)

    async def _call_ollama(self, system_prompt: str, user_content: str) -> str | None:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
            },
        }

        for attempt in range(2):
            try:
                session = self._get_session()
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        logger.error("Ollama API error: %s", await resp.text())
                        return None

                    data = await resp.json()
                    return data.get("message", {}).get("content", "")

            except asyncio.TimeoutError:
                if attempt == 0:
                    logger.warning("Ollama request timed out (attempt %d), retrying...", attempt + 1)
                else:
                    logger.error("Ollama request timed out after %d attempts", attempt + 1)
                    return None
            except aiohttp.ClientError as e:
                logger.error("HTTP error calling Ollama: %s", e)
                return None
            except Exception as e:
                logger.error("Unexpected error in classifier: %r", e, exc_info=True)
                return None
        return None

    async def _call_api(self, system_prompt: str, user_content: str) -> str | None:
        result = await self._call_api_with_model(self.model, system_prompt, user_content)
        if result is not None:
            return result
        if self.fallback_model:
            logger.warning("Primary model %s failed, falling back to %s", self.model, self.fallback_model)
            return await self._call_api_with_model(self.fallback_model, system_prompt, user_content)
        return None

    async def _call_api_with_model(self, model: str, system_prompt: str, user_content: str) -> str | None:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(2):
            try:
                session = self._get_session()
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as resp:
                    if resp.status != 200:
                        logger.error("API error (model=%s): %s", model, await resp.text())
                        return None

                    data = await resp.json()
                    if not isinstance(data, dict):
                        logger.warning("API returned non-dict response (model=%s, attempt %d): %s", model, attempt + 1, data)
                        continue
                    choices = data.get("choices")
                    if not choices or not isinstance(choices, list):
                        logger.warning("Malformed API response (model=%s, attempt %d): missing 'choices'. Body: %s", model, attempt + 1, data)
                        continue
                    return choices[0].get("message", {}).get("content", "")

            except asyncio.TimeoutError:
                if attempt == 0:
                    logger.warning("API request timed out (model=%s, attempt %d), retrying...", model, attempt + 1)
                else:
                    logger.error("API request timed out (model=%s) after %d attempts", model, attempt + 1)
                    return None
            except aiohttp.ClientError as e:
                logger.warning("HTTP error calling API (model=%s, attempt %d): %s", model, attempt + 1, e)
            except Exception as e:
                logger.error("Unexpected error in classifier (model=%s): %r", model, e, exc_info=True)
                return None

        logger.error("API call failed (model=%s) after %d attempts", model, attempt + 1)
        return None

    @staticmethod
    def _parse_triage(content: str) -> dict | None:
        try:
            result = json.loads(content)
            return {
                "is_signal": bool(result.get("is_signal", False)),
                "confidence": float(result.get("confidence", 0.0)),
                "category": str(result.get("category", "noise")),
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse triage response: %s", e)
            return None

    @staticmethod
    def _parse_enrich(content: str) -> dict | None:
        try:
            result = json.loads(content)
            raw_tickers = result.get("tickers", [])
            tickers = []
            for t in raw_tickers:
                if isinstance(t, dict):
                    tickers.append({
                        "symbol": str(t.get("symbol", "")),
                        "bias": str(t.get("bias", "neutral")).lower(),
                    })
                else:
                    tickers.append({"symbol": str(t), "bias": "neutral"})
            return {
                "is_signal": bool(result.get("is_signal", False)),
                "confidence": float(result.get("confidence", 0.0)),
                "thesis": str(result.get("thesis", "")),
                "tickers": tickers,
                "category": str(result.get("category", "noise")),
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse enrich response: %s", e)
            return None
