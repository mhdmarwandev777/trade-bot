"""
Claude AI Trading Bot — Alpaca Markets
Works natively on Mac. Paper trading by default.

Install:
    pip install alpaca-py anthropic pandas numpy python-dotenv

.env file:
    ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
    ALPACA_API_KEY=PKxxxxxxxxxxxxxxxx
    ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ALPACA_BASE_URL=https://paper-api.alpaca.markets
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
import anthropic
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    GetOrdersRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

load_dotenv()

# ─────────────────────────────────────────────────────────
#  CONFIG — edit these
# ─────────────────────────────────────────────────────────

# Choose your market:
#   Stocks:  "AAPL", "SPY", "QQQ", "TSLA"
#   Crypto:  "BTC/USD", "ETH/USD"  (24/7 market — best for bots)
SYMBOL      = "BTC/USD"       # change to "SPY" for stocks
MARKET_TYPE = "crypto"        # "crypto" or "stock"

SLEEP_SECS     = 60 * 15      # analyze every 15 minutes
MIN_CONFIDENCE = 65            # skip trades below this %
BARS           = 100           # candles to fetch (≥50 required for MA50)

# Position sizing (crypto)
MAX_NOTIONAL_PCT = 0.10        # max 10% of available cash per trade
MAX_NOTIONAL     = 500         # hard cap in USD per trade

# For stocks: how many shares per trade
QTY = 1

DAILY_LOSS_LIMIT_PCT = 0.03   # stop trading if down 3% in a day

# ─────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
#  CLIENTS
# ─────────────────────────────────────────────────────────

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
claude_client  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

if MARKET_TYPE == "crypto":
    data_client = CryptoHistoricalDataClient()
else:
    data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# ─────────────────────────────────────────────────────────
#  ACCOUNT INFO
# ─────────────────────────────────────────────────────────

def print_account():
    account = trading_client.get_account()
    log.info(f"Account: {account.id}")
    log.info(f"Cash:    ${float(account.cash):,.2f}")
    log.info(f"Portfolio value: ${float(account.portfolio_value):,.2f}")
    log.info(f"Paper trading: {account.status}")

def get_notional() -> float:
    """Dynamic position sizing: 10% of cash, capped at MAX_NOTIONAL."""
    account = trading_client.get_account()
    cash = float(account.cash)
    return min(cash * MAX_NOTIONAL_PCT, MAX_NOTIONAL)


# Daily loss limit
def is_daily_loss_limit_hit() -> bool:
    account  = trading_client.get_account()
    equity   = float(account.equity)
    last_eq  = float(account.last_equity)   # equity at start of day
    daily_pl = (equity - last_eq) / last_eq
    if daily_pl < -DAILY_LOSS_LIMIT_PCT:
        log.warning(f"Daily loss limit hit: {daily_pl:.2%} — stopping for today")
        return True
    return False

# ─────────────────────────────────────────────────────────
#  MARKET DATA — fetch OHLCV candles
# ─────────────────────────────────────────────────────────

def get_candles() -> pd.DataFrame:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(hours=BARS)  # generous lookback buffer

    if MARKET_TYPE == "crypto":
        req = CryptoBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=start,
            end=end,
            limit=BARS,
        )
        bars = data_client.get_crypto_bars(req)
    else:
        req = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=start,
            end=end,
            limit=BARS,
        )
        bars = data_client.get_stock_bars(req)

    symbol_key = SYMBOL.replace("/", "")
    df = bars.df

    # Flatten multi-index if present
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol_key, level=0) if symbol_key in df.index.get_level_values(0) else df.droplevel(0)

    df = df[["open", "high", "low", "close", "volume"]].dropna()

    if len(df) < 50:
        raise ValueError(f"Not enough data: got {len(df)} bars, need at least 50 for MA50")

    log.info(f"  Fetched {len(df)} candles for {SYMBOL}")
    return df

# ─────────────────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────────────────

def calc_indicators(df: pd.DataFrame) -> dict:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # RSI (14) — Wilder's smoothing (EWM, not simple rolling mean)
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi   = round(float(100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1])), 1)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = round(float((ema12 - ema26).iloc[-1]), 4)
    sig   = round(float((ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1]), 4)
    hist  = round(macd - sig, 4)

    # Moving averages
    ma20  = round(float(close.rolling(20).mean().iloc[-1]), 4)
    ma50  = round(float(close.rolling(50).mean().iloc[-1]), 4)

    # Bollinger Bands
    std      = float(close.rolling(20).std().iloc[-1])
    bb_upper = round(ma20 + 2 * std, 4)
    bb_lower = round(ma20 - 2 * std, 4)

    # ATR (14)
    tr  = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = round(float(tr.rolling(14).mean().iloc[-1]), 4)

    # Volume ratio: current bar vs 20-bar average
    vol_avg   = float(vol.rolling(20).mean().iloc[-1])
    vol_ratio = round(float(vol.iloc[-1]) / vol_avg, 2) if vol_avg > 0 else 1.0

    last   = round(float(close.iloc[-1]), 4)
    prev10 = round(float(close.iloc[-10]), 4)

    return {
        "last":      last,
        "high":      round(float(high.max()), 4),
        "low":       round(float(low.min()), 4),
        "rsi":       rsi,
        "macd":      macd,
        "macd_sig":  sig,
        "macd_hist": hist,
        "ma20":      ma20,
        "ma50":      ma50,
        "bb_upper":  bb_upper,
        "bb_lower":  bb_lower,
        "atr":       atr,
        "vol_ratio": vol_ratio,
        "trend":     "up" if last > prev10 else "down",
        "ma_cross":  "golden" if ma20 > ma50 else "death",
    }

# ─────────────────────────────────────────────────────────
#  CLAUDE AI ANALYSIS
# ─────────────────────────────────────────────────────────

def ask_claude(ind: dict, df: pd.DataFrame) -> dict:
    # Last 20 candles as a table for pattern context
    recent_candles = df.tail(20)[["open", "high", "low", "close", "volume"]].round(2).to_string()

    prompt = f"""You are a professional trading analyst. Analyze this live {SYMBOL} market data and return a precise trade signal.

SYMBOL: {SYMBOL}
TIMEFRAME: 15-minute candles
CURRENT PRICE: {ind['last']}
SESSION: High={ind['high']}  Low={ind['low']}
TREND (10-bar): {ind['trend']}

TECHNICAL INDICATORS:
- RSI(14):       {ind['rsi']}  {'→ OVERBOUGHT' if ind['rsi'] > 70 else '→ OVERSOLD' if ind['rsi'] < 30 else '→ neutral'}
- MACD:          {ind['macd']}  Signal: {ind['macd_sig']}  Hist: {ind['macd_hist']}  {'→ bullish' if ind['macd'] > ind['macd_sig'] else '→ bearish'}
- MA20 / MA50:   {ind['ma20']} / {ind['ma50']}  → {ind['ma_cross']} cross
- Bollinger:     Upper={ind['bb_upper']}  Lower={ind['bb_lower']}  {'→ above upper (overbought)' if ind['last'] > ind['bb_upper'] else '→ below lower (oversold)' if ind['last'] < ind['bb_lower'] else '→ inside bands'}
- ATR(14):       {ind['atr']}  (current volatility)
- Volume ratio:  {ind['vol_ratio']}x  {'→ HIGH volume (confirms move)' if ind['vol_ratio'] > 1.5 else '→ low volume (weak signal)' if ind['vol_ratio'] < 0.7 else '→ normal volume'}

RECENT CANDLES (oldest → newest, last 20 bars):
{recent_candles}

DECISION RULES:
- Signal BUY or SELL only when at least 3 indicators agree
- If mixed signals or uncertain → HOLD
- stop_loss must be at least 1x ATR away from entry
- take_profit must give minimum 1:2 risk/reward ratio
- confidence below 65 → always return HOLD
- Low volume (ratio < 0.7) reduces confidence by 10 points

Respond ONLY with valid JSON, absolutely no extra text or markdown:
{{"signal":"BUY","entry":{ind['last']},"stop_loss":0.0,"take_profit":0.0,"confidence":75,"reason":"brief one sentence explanation"}}"""

    last_err = None
    for attempt in range(3):
        try:
            response = claude_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)

            # Validate SL/TP are non-zero for actionable signals
            if result.get("signal") in ("BUY", "SELL"):
                if result.get("stop_loss", 0) == 0 or result.get("take_profit", 0) == 0:
                    log.warning(f"  Claude returned zero SL/TP on attempt {attempt + 1}, retrying...")
                    last_err = ValueError("Zero SL/TP in response")
                    time.sleep(1)
                    continue

            return result

        except (json.JSONDecodeError, KeyError) as e:
            last_err = e
            log.warning(f"  Claude parse error (attempt {attempt + 1}): {e}")
            time.sleep(1)

    raise RuntimeError(f"Claude failed after 3 attempts: {last_err}")

# ─────────────────────────────────────────────────────────
#  POSITION MANAGEMENT
# ─────────────────────────────────────────────────────────

def get_position() -> float:
    """Returns current position qty or 0 if none."""
    try:
        symbol_key = SYMBOL.replace("/", "")
        pos = trading_client.get_open_position(symbol_key)
        return float(pos.qty)
    except Exception:
        return 0.0

def close_position():
    """Close existing position before reversing."""
    try:
        symbol_key = SYMBOL.replace("/", "")
        trading_client.close_position(symbol_key)
        log.info(f"  Closed existing {SYMBOL} position")
        time.sleep(2)
    except Exception as e:
        log.warning(f"  Could not close position: {e}")

# ─────────────────────────────────────────────────────────
#  ORDER EXECUTION
# ─────────────────────────────────────────────────────────

def place_order(signal: dict):
    action = signal["signal"]

    if action == "HOLD":
        log.info(f"  → HOLD | {signal['reason']}")
        return

    if signal["confidence"] < MIN_CONFIDENCE:
        log.info(f"  → Skipped — confidence {signal['confidence']}% < {MIN_CONFIDENCE}%")
        return

    current_qty = get_position()

    # Crypto: Alpaca does not support short selling — SELL = close long only
    if MARKET_TYPE == "crypto" and action == "SELL":
        if current_qty > 0:
            close_position()
            log.info("  → Closed LONG (SELL signal on crypto — no shorting available)")
        else:
            log.info("  → SELL skipped — no long position to close (crypto can't short)")
        return

    # If already in same direction, skip
    if action == "BUY"  and current_qty > 0:
        log.info(f"  → Already LONG {SYMBOL}, skipping")
        return
    if action == "SELL" and current_qty < 0:
        log.info(f"  → Already SHORT {SYMBOL}, skipping")
        return

    # Close opposite position first
    if current_qty != 0:
        close_position()

    side     = OrderSide.BUY if action == "BUY" else OrderSide.SELL
    notional = get_notional()
    sl_price = round(signal["stop_loss"],   2)
    tp_price = round(signal["take_profit"], 2)

    def _build_order(bracket: bool):
        if MARKET_TYPE == "crypto":
            kwargs = dict(
                symbol=SYMBOL.replace("/", ""),
                notional=notional,
                side=side,
                time_in_force=TimeInForce.GTC,
            )
        else:
            kwargs = dict(
                symbol=SYMBOL,
                qty=QTY,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        if bracket:
            kwargs["order_class"]  = OrderClass.BRACKET
            kwargs["take_profit"]  = TakeProfitRequest(limit_price=tp_price)
            kwargs["stop_loss"]    = StopLossRequest(stop_price=sl_price)
        return MarketOrderRequest(**kwargs)

    # Try bracket order first; fall back to plain market order if not supported
    order = None
    try:
        order = trading_client.submit_order(_build_order(bracket=True))
        log.info("  → Bracket order placed (SL/TP attached)")
    except Exception as e:
        log.warning(f"  Bracket order rejected ({e}), retrying as plain market order...")
        try:
            order = trading_client.submit_order(_build_order(bracket=False))
            log.warning("  → Plain market order placed — monitor SL/TP manually!")
        except Exception as e2:
            log.error(f"  → Order FAILED: {e2}")
            return

    log.info(f"  → {action} order filled!")
    log.info(f"     Order ID:    {order.id}")
    log.info(f"     Symbol:      {SYMBOL}")
    log.info(f"     Side:        {action}")
    log.info(f"     Notional:    ${notional:.2f}")
    log.info(f"     Confidence:  {signal['confidence']}%")
    log.info(f"     Entry:       ~{signal['entry']}")
    log.info(f"     Stop loss:   {sl_price}")
    log.info(f"     Take profit: {tp_price}")
    log.info(f"     Reason:      {signal['reason']}")

# ─────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────

def main():
    log.info("=" * 55)
    log.info("  Claude AI Trading Bot — Alpaca Markets")
    log.info("=" * 55)
    print_account()
    log.info(f"  Symbol:         {SYMBOL}")
    log.info(f"  Timeframe:      M15")
    log.info(f"  Min confidence: {MIN_CONFIDENCE}%")
    log.info("=" * 55)

    cycle = 0
    while True:
        cycle += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.info(f"\n[Cycle #{cycle}] {now}")

        if is_daily_loss_limit_hit():
            log.info("Bot paused — daily loss limit reached. Resuming tomorrow.")
            time.sleep(60 * 60)   # sleep 1 hour then recheck
            continue

        try:
            # 1. Fetch market data
            df  = get_candles()
            ind = calc_indicators(df)

            log.info(
                f"  Price: {ind['last']} | RSI={ind['rsi']} | MACD={ind['macd']} | "
                f"Trend={ind['trend']} | MA cross={ind['ma_cross']} | Vol={ind['vol_ratio']}x"
            )

            # 2. Pre-filter: skip Claude if setup has no clear directional bias
            rsi_neutral = 42 < ind["rsi"] < 58
            macd_weak   = abs(ind["macd_hist"]) < ind["last"] * 0.0001
            if rsi_neutral and macd_weak:
                log.info(
                    f"  Skipping Claude — weak setup "
                    f"(RSI={ind['rsi']}, MACD hist={ind['macd_hist']})"
                )
            else:
                # 3. Ask Claude for signal
                signal = ask_claude(ind, df)
                log.info(f"  Claude → {signal['signal']} | {signal['confidence']}% confidence")
                log.info(f"  Reason: {signal['reason']}")

                # 4. Execute if valid
                place_order(signal)

        except Exception as e:
            log.error(f"  Error in cycle: {e}", exc_info=True)

        log.info(f"  Sleeping {SLEEP_SECS // 60} min until next cycle...\n")
        time.sleep(SLEEP_SECS)


if __name__ == "__main__":
    main()
