# Telegram Channel Monitor

Monitors private Telegram channels, classifies messages as signal vs noise using an LLM (local or cloud), and forwards high-signal messages to your Telegram bot.

## Quick Start

```bash
git clone <repo-url> && cd telegram-channel-monitor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in credentials
cp config.yaml.example config.yaml  # configure channels + LLM
python main.py                # first run asks for Telegram phone + code
```

Then send `/start` to your bot on Telegram.

## Configuration

### `.env`
| Variable | Required | Description |
|---|---|---|
| `TELEGRAM_API_ID` | Yes | From https://my.telegram.org |
| `TELEGRAM_API_HASH` | Yes | From https://my.telegram.org |
| `TELEGRAM_BOT_TOKEN` | Yes | From @BotFather |
| `ALLOWED_CHAT_IDS` | Recommended | Comma-separated Telegram chat IDs allowed to use the bot. If empty, all users are rejected. |
| `LLM_API_KEY` | API provider only | e.g. OpenRouter, Groq |

### `config.yaml` — LLM provider

**OpenRouter:**
```yaml
llm:
  provider: "api"
  base_url: "https://openrouter.ai/api/v1"
  model: "qwen/qwen3.5-9b"
```

**Local ollama:**
```yaml
llm:
  provider: "ollama"
  base_url: "http://localhost:11434"
  model: "qwen3:4b"
```

## Architecture

```
Telegram Channel → Telethon Listener → LLM Classifier → Bot Forwarder → You
                                              ↓
                                        SQLite Logger
```

## Files

- `main.py` — entrypoint
- `src/listener.py` — Telethon channel listener
- `src/classifier.py` — LLM classifier (ollama or OpenAI-compatible API)
- `src/forwarder.py` — Telegram bot forwarder with /start registration
- `src/db.py` — SQLite persistence
- `config.yaml.example` — channels, model, prompt config template

## VPS Deployment

1. Get a **$5/mo VPS** (Hetzner CX22, DigitalOcean, etc.)
2. Clone, install deps, configure `.env` and `config.yaml` as above
3. First run interactively (`python main.py`) to complete Telegram login
4. Then set up as a systemd service for auto-restart:

```bash
sudo tee /etc/systemd/system/telegram-monitor.service << 'EOF'
[Unit]
Description=Telegram Channel Monitor
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/home/your-user/telegram-channel-monitor
ExecStart=/home/your-user/telegram-channel-monitor/venv/bin/python main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now telegram-monitor
sudo journalctl -u telegram-monitor -f  # view logs
```
