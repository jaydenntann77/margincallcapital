# margincallcapital — Roostoo Trading Bot

> SG vs. HK Quant Trading Hackathon

An automated crypto trading bot that runs on the **Roostoo** mock trading platform using an **EMA crossover strategy** (9 / 21 periods). Designed for continuous operation on an **AWS EC2** instance.

---

## Project Layout

```
margincallcapital/
├── bot.py             # ← Main entry point. Run this to start the bot.
├── api_client.py      # High-level Roostoo wrapper (retry, precision, helpers)
├── strategy.py        # EMA crossover signal generator (BUY / SELL / HOLD)
├── risk_manager.py    # Position sizing, balance checks, cooldown, stale orders
├── logger.py          # Structured console + file logging
├── config.py          # All tuneable parameters (loads credentials from .env)
├── client.py          # Low-level Roostoo REST client (HMAC-SHA256 signing)
├── .env.example       # Environment variable template
├── requirements.txt   # Python dependencies
│
├── signals/           # Research: vectorised backtesting engine & momentum signals
├── research/          # Jupyter notebooks
└── tests/             # Live API integration tests
```

---

## Strategy

| Parameter | Default | Description |
|---|---|---|
| Fast EMA | 9 bars | Short-term trend |
| Slow EMA | 21 bars | Long-term trend |
| Signal | Crossover | BUY when fast crosses above slow; SELL when it crosses below |
| Trade size | 20% of portfolio | Per-signal allocation |
| Sell size | 50% of holdings | Fraction of position closed on SELL signal |
| Max position | 30% of portfolio | Per-asset cap |
| Order type | Limit (market fallback) | Maker fee 0.05%; market order 0.1% used if limit fails |
| Poll interval | 60 s | One price fetch per pair per minute |
| Warmup | 21 bars | No signals until both EMAs have converged |

---

## Local Setup

### 1. Prerequisites

- Python 3.10 or newer
- `git`

### 2. Clone & install

```bash
git clone https://github.com/jaydenntann77/margincallcapital.git
cd margincallcapital

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your Roostoo API credentials:

```dotenv
ROOSTOO_API_KEY=your_api_key_here
ROOSTOO_API_SECRET=your_api_secret_here
ROOSTOO_BASE_URL=https://mock.roostoo.com
LOG_LEVEL=INFO
```

### 4. (Optional) Tune parameters

Open `config.py` to adjust trading pairs, EMA periods, position sizing, poll interval, etc. No other files need to change.

---

## Running the Bot

### Live mode

```bash
python bot.py
```

### Dry-run mode — signals logged, no orders placed

```bash
python bot.py --dry-run
```

Use `--dry-run` first to verify connectivity and watch signals fire before going live.

### Run API integration tests

```bash
python tests/test_roostoo.py
```

---

## Deploying to AWS EC2

### 1. Launch an EC2 instance

| Setting | Recommended |
|---|---|
| AMI | Ubuntu 22.04 LTS |
| Instance type | t3.micro (free tier) |
| Storage | 8 GB gp3 |
| Security group | Allow SSH (port 22) from your IP only |

### 2. Connect and set up the environment

```bash
ssh -i your-key.pem ubuntu@<ec2-public-ip>

# Install system dependencies
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git

# Clone the repo
git clone https://github.com/jaydenntann77/margincallcapital.git
cd margincallcapital

# Create virtual environment and install packages
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
cp .env.example .env
nano .env    # fill in ROOSTOO_API_KEY, ROOSTOO_API_SECRET, ROOSTOO_BASE_URL
```

### 4a. Run with `nohup` (quick start)

```bash
source .venv/bin/activate
nohup python bot.py > /dev/null 2>&1 &
echo $! > bot.pid
echo "Bot started with PID $(cat bot.pid)"
```

All output goes to `bot.log` via the file handler. Monitor it with:

```bash
tail -f bot.log
```

Stop the bot:

```bash
kill $(cat bot.pid)
```

### 4b. Run with `systemd` (recommended for production)

Create the service unit:

```bash
sudo nano /etc/systemd/system/roostoo-bot.service
```

Paste the following (adjust paths if your username differs from `ubuntu`):

```ini
[Unit]
Description=Roostoo EMA Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/margincallcapital
EnvironmentFile=/home/ubuntu/margincallcapital/.env
ExecStart=/home/ubuntu/margincallcapital/.venv/bin/python bot.py
Restart=on-failure
RestartSec=30
StandardOutput=append:/home/ubuntu/margincallcapital/bot.log
StandardError=append:/home/ubuntu/margincallcapital/bot.log

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable roostoo-bot
sudo systemctl start roostoo-bot
```

Check status and logs:

```bash
sudo systemctl status roostoo-bot
journalctl -u roostoo-bot -f
tail -f bot.log
```

---

## Stopping and Restarting Safely

The bot listens for `SIGTERM` and `SIGINT` and finishes the current polling cycle before exiting — no in-flight logic is abandoned mid-cycle.

```bash
# Graceful stop (waits for current cycle to complete)
sudo systemctl stop roostoo-bot

# Restart after a config change
sudo systemctl restart roostoo-bot

# One-liner tail
tail -100 bot.log
```

When the bot restarts, it fetches all open orders from the exchange on startup and re-registers them for stale-order tracking automatically.

---

## Configuration Reference

All parameters live in `config.py`:

| Variable | Default | Description |
|---|---|---|
| `TRADING_PAIRS` | 10 liquid pairs | Pairs monitored and traded |
| `FAST_EMA_PERIOD` | `9` | Short EMA period |
| `SLOW_EMA_PERIOD` | `21` | Long EMA period |
| `TRADE_FRACTION` | `0.20` | Fraction of total portfolio per BUY |
| `MAX_POSITION_FRAC` | `0.30` | Max per-asset allocation |
| `SELL_FRACTION` | `0.50` | Fraction of holdings liquidated per SELL |
| `MIN_ORDER_VALUE_USD` | `15.0` | Minimum order value in USD |
| `LIMIT_ORDER_OFFSET` | `0.001` | 0.1% price nudge for limit fills |
| `LIMIT_ORDER_TIMEOUT` | `120` | Seconds before stale limit is cancelled |
| `SIGNAL_COOLDOWN_SECONDS` | `300` | Min gap between same-direction signals |
| `POLL_INTERVAL_SECONDS` | `60` | Polling frequency |
| `MAX_RETRIES` | `3` | API retry attempts on network errors |
| `TAKER_FEE` | `0.001` | Market order commission |
| `MAKER_FEE` | `0.0005` | Limit order commission |

---

## Log Format

`bot.log` is written to the working directory in this format:

```
2025-03-20 14:32:01 | INFO     | Bot                  | ─── Cycle 1 ───
2025-03-20 14:32:02 | INFO     | Bot                  | [BTC/USD] Warming up — 5/21 bars collected, holding.
2025-03-20 14:32:22 | INFO     | Bot                  | [SIGNAL] BTC/USD     BUY  | qty=0.002500 | last_price=84132.10000 | limit_price=84216.23210
2025-03-20 14:32:22 | INFO     | ApiClient            | Placing LIMIT BUY  BTC/USD  qty=0.00250  price=84216.23
2025-03-20 14:32:23 | INFO     | Bot                  | [ORDER]  LIMIT BUY  BTC/USD       | id=ORD-12345      | qty=0.002500 | price=84216.23210
```

---

## License

For hackathon use only.
