"""
Claude AI Trading Bot — Alpaca Markets  (Enhanced v2)
======================================================
Improvements over v1:
  1. Multi-timeframe confirmation  (M15 + H1 must agree)
  2. Trailing stop + drawdown tracking
  3. Portfolio-aware Claude prompt  (position, P&L, regime)
  4. Risk-per-trade position sizing  (risk 1% of equity per trade)

Install:
    pip install alpaca-py anthropic pandas numpy python-dotenv

.env file:
    ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
    ALPACA_API_KEY=PKxxxxxxxxxxxxxxxx
    ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ALPACA_BASE_URL=https://paper-api.alpaca.markets
    TELEGRAM_BOT_TOKEN=123456789:ABCxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # optional
    TELEGRAM_CHAT_ID=123456789                                          # optional
"""

import os
import json
import time
import logging
import urllib.request
import urllib.parse
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
#  CONFIG
# ─────────────────────────────────────────────────────────

SYMBOL      = "BTC/USD"   # "AAPL", "SPY", "ETH/USD", etc.
MARKET_TYPE = "crypto"    # "crypto" or "stock"

SLEEP_SECS     = 60 * 15  # analyze every 15 minutes
MIN_CONFIDENCE = 65        # skip trades below this %
BARS_M15       = 100       # candles for M15 (primary)
BARS_H1        = 60        # candles for H1  (confirmation)

# Risk-per-trade: lose at most this % of equity if SL is hit
RISK_PER_TRADE_PCT = 0.01   # 1% of equity per trade
MAX_NOTIONAL       = 1000   # hard cap in USD regardless of sizing

# Drawdown guard
DAILY_LOSS_LIMIT_PCT = 0.03   # pause if down 3% in a day
MAX_DRAWDOWN_PCT     = 0.08   # pause if down 8% from equity peak (session)

# Trailing stop: once price moves this many × ATR in our favour,
# move stop to break-even; keep trailing by TRAIL_STEP_ATR × ATR after that.
TRAIL_ATR_TRIGGER = 1.0   # activate trailing after 1× ATR profit
TRAIL_STEP_ATR    = 0.5   # trail increments of 0.5× ATR

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
#  PERSISTENT STATE  (survives restarts)
# ─────────────────────────────────────────────────────────

STATE_FILE = "bot_state.json"

def load_state() -> dict:
    defaults = {
        "cycle":            0,
        "peak_equity":      0.0,
        "session_trades":   [],        # list of {side, entry, exit, pnl}
        "trailing_stop":    None,      # active trailing stop price (float or None)
        "trade_entry_price": None,     # price at which current position was entered
        "trade_side":       None,      # "BUY" or "SELL"
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                saved = json.load(f)
            defaults.update(saved)
        except Exception:
            pass
    return defaults

def save_state(state: dict):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log.warning(f"Could not save state: {e}")

# ─────────────────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────────────────

def send_telegram(msg: str) -> None:
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        url  = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id":    TG_CHAT,
            "text":       msg,
            "parse_mode": "HTML",
        }).encode()
        urllib.request.urlopen(url, data=data, timeout=10)
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


def build_status_msg(ind_m15: dict, ind_h1: dict, signal: dict | None,
                     mtf_ok: bool, state: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def rsi_label(rsi):
        return "🔴 OB" if rsi > 70 else "🟢 OS" if rsi < 30 else "⚪️ Neutral"

    def macd_label(m, s):
        return "📈 Bull" if m > s else "📉 Bear"

    trend_icon = "🚀" if ind_m15["trend"] == "up" else "🔻"
    ma_label   = "✨ Golden" if ind_m15["ma_cross"] == "golden" else "💀 Death"
    mtf_label  = "✅ ALIGNED" if mtf_ok else "❌ DIVERGED"

    ts = state.get("trailing_stop")
    ts_str = f"${ts:,.2f}" if ts else "—"

    msg = (
        f"📊 <b>{SYMBOL} · M15 Scan</b>\n"
        f"🕐 {now}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Price:  <b>${ind_m15['last']:,.2f}</b>  {trend_icon}  {ma_label}\n"
        f"🔁 MTF H1: {mtf_label}\n"
        f"─────────────────────\n"
        f"  <b>M15</b> RSI={ind_m15['rsi']} {rsi_label(ind_m15['rsi'])} | "
        f"MACD {macd_label(ind_m15['macd'], ind_m15['macd_sig'])}\n"
        f"  <b>H1 </b> RSI={ind_h1['rsi']} {rsi_label(ind_h1['rsi'])} | "
        f"MACD {macd_label(ind_h1['macd'], ind_h1['macd_sig'])}\n"
        f"  Vol: {ind_m15['vol_ratio']}x  |  ATR: {ind_m15['atr']}\n"
        f"  Trailing stop: {ts_str}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
    )

    if signal is None:
        msg += "⏭️ <i>Skipped (weak setup or MTF divergence)</i>"
    else:
        icon = "🟢" if signal["signal"] == "BUY" else "🔴" if signal["signal"] == "SELL" else "🟡"
        msg += (
            f"🤖 Claude: {icon} <b>{signal['signal']}</b> ({signal['confidence']}%)\n"
            f"📐 Regime: {signal.get('regime','?')}\n"
            f"💬 {signal['reason']}"
        )
    return msg

# ─────────────────────────────────────────────────────────
#  CLIENTS
# ─────────────────────────────────────────────────────────

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
TG_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT    = os.getenv("TELEGRAM_CHAT_ID", "")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
claude_client  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

if MARKET_TYPE == "crypto":
    data_client = CryptoHistoricalDataClient()
else:
    data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# ─────────────────────────────────────────────────────────
#  ACCOUNT / RISK HELPERS
# ─────────────────────────────────────────────────────────

def print_account():
    account = trading_client.get_account()
    log.info(f"Account: {account.id}")
    log.info(f"Cash:    ${float(account.cash):,.2f}")
    log.info(f"Portfolio value: ${float(account.portfolio_value):,.2f}")
    log.info(f"Paper trading: {account.status}")


def get_equity() -> float:
    return float(trading_client.get_account().equity)


def is_daily_loss_limit_hit() -> bool:
    account  = trading_client.get_account()
    equity   = float(account.equity)
    last_eq  = float(account.last_equity)
    daily_pl = (equity - last_eq) / last_eq
    if daily_pl < -DAILY_LOSS_LIMIT_PCT:
        log.warning(f"Daily loss limit hit: {daily_pl:.2%}")
        send_telegram(
            f"⛔️ <b>DAILY LOSS LIMIT HIT</b>\n"
            f"📉 Daily P&amp;L: <b>{daily_pl:.2%}</b>\n"
            f"⏸ Bot paused for 1 hour"
        )
        return True
    return False


def is_max_drawdown_hit(state: dict) -> bool:
    equity = get_equity()
    if equity > state["peak_equity"]:
        state["peak_equity"] = equity          # update high-water mark
        save_state(state)
    if state["peak_equity"] > 0:
        dd = (state["peak_equity"] - equity) / state["peak_equity"]
        if dd > MAX_DRAWDOWN_PCT:
            log.warning(f"Max drawdown hit: {dd:.2%} from peak ${state['peak_equity']:,.2f}")
            send_telegram(
                f"⛔️ <b>MAX DRAWDOWN HIT</b>\n"
                f"📉 Drawdown: <b>{dd:.2%}</b> from peak\n"
                f"⏸ Bot paused for 1 hour"
            )
            return True
    return False


# ── IMPROVEMENT 4: Risk-per-trade position sizing ────────
def calc_notional(entry: float, stop_loss: float) -> float:
    """
    Size so that if SL is hit, we lose exactly RISK_PER_TRADE_PCT of equity.
    notional = (equity × risk%) / (|entry - sl| / entry)
    Capped at MAX_NOTIONAL.
    """
    equity = get_equity()
    risk_dollars = equity * RISK_PER_TRADE_PCT
    sl_distance_pct = abs(entry - stop_loss) / entry
    if sl_distance_pct == 0:
        return min(equity * 0.05, MAX_NOTIONAL)   # fallback: 5% of equity
    notional = risk_dollars / sl_distance_pct
    notional = min(notional, MAX_NOTIONAL)
    log.info(f"  Position sizing: equity=${equity:,.2f} risk=${risk_dollars:.2f} "
             f"SL-dist={sl_distance_pct:.3%} → notional=${notional:.2f}")
    return round(notional, 2)

# ─────────────────────────────────────────────────────────
#  MARKET DATA
# ─────────────────────────────────────────────────────────

def _fetch_bars(timeframe: TimeFrame, n_bars: int) -> pd.DataFrame:
    """Generic bar fetcher for any timeframe."""
    end   = datetime.now(timezone.utc)
    # generous lookback so we always get n_bars after exchange gaps
    hours = n_bars * (1 if timeframe == TimeFrame.Hour else 0.25) * 2
    start = end - timedelta(hours=max(hours, 48))

    if MARKET_TYPE == "crypto":
        req  = CryptoBarsRequest(symbol_or_symbols=SYMBOL, timeframe=timeframe,
                                 start=start, end=end, limit=n_bars)
        bars = data_client.get_crypto_bars(req)
    else:
        req  = StockBarsRequest(symbol_or_symbols=SYMBOL, timeframe=timeframe,
                                start=start, end=end, limit=n_bars)
        bars = data_client.get_stock_bars(req)

    symbol_key = SYMBOL.replace("/", "")
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol_key, level=0) if symbol_key in df.index.get_level_values(0) \
             else df.droplevel(0)

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df


def get_candles_m15() -> pd.DataFrame:
    df = _fetch_bars(TimeFrame(15, TimeFrameUnit.Minute), BARS_M15)
    if len(df) < 50:
        raise ValueError(f"M15: only {len(df)} bars (need ≥50)")
    log.info(f"  Fetched {len(df)} M15 candles")
    return df


def get_candles_h1() -> pd.DataFrame:
    df = _fetch_bars(TimeFrame.Hour, BARS_H1)
    if len(df) < 26:
        raise ValueError(f"H1: only {len(df)} bars (need ≥26 for MACD)")
    log.info(f"  Fetched {len(df)} H1 candles")
    return df

# ─────────────────────────────────────────────────────────
#  INDICATORS  (shared for both timeframes)
# ─────────────────────────────────────────────────────────

def calc_indicators(df: pd.DataFrame) -> dict:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # RSI (Wilder's smoothing)
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi   = round(float(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-9))), 1)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = round(float((ema12 - ema26).iloc[-1]), 4)
    sig   = round(float((ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1]), 4)
    hist  = round(macd - sig, 4)

    # MAs (only reliable if enough bars)
    ma20 = round(float(close.rolling(20).mean().iloc[-1]), 4) if len(df) >= 20 else None
    ma50 = round(float(close.rolling(50).mean().iloc[-1]), 4) if len(df) >= 50 else None

    # Bollinger
    std      = float(close.rolling(20).std().iloc[-1]) if len(df) >= 20 else 0
    bb_upper = round((ma20 or 0) + 2 * std, 4)
    bb_lower = round((ma20 or 0) - 2 * std, 4)

    # ATR
    tr  = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = round(float(tr.rolling(14).mean().iloc[-1]), 4)

    # Volume ratio
    vol_avg   = float(vol.rolling(20).mean().iloc[-1]) if len(df) >= 20 else float(vol.mean())
    vol_ratio = round(float(vol.iloc[-1]) / vol_avg, 2) if vol_avg > 0 else 1.0

    last   = round(float(close.iloc[-1]), 4)
    prev10 = round(float(close.iloc[max(-10, -len(df))]), 4)

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
        "ma_cross":  "golden" if (ma20 and ma50 and ma20 > ma50) else "death",
    }

# ─────────────────────────────────────────────────────────
#  IMPROVEMENT 1: Multi-timeframe confirmation
# ─────────────────────────────────────────────────────────

def mtf_agrees(ind_m15: dict, ind_h1: dict) -> tuple[bool, str]:
    """
    Return (aligned, reason).
    A BUY signal on M15 is only valid if H1 is also bullish (MACD > signal,
    RSI not overbought, trend up). Vice-versa for SELL.
    """
    m15_bullish = (ind_m15["macd"] > ind_m15["macd_sig"] and
                   ind_m15["trend"] == "up")
    h1_bullish  = (ind_h1["macd"]  > ind_h1["macd_sig"]  and
                   ind_h1["rsi"] < 70)

    m15_bearish = (ind_m15["macd"] < ind_m15["macd_sig"] and
                   ind_m15["trend"] == "down")
    h1_bearish  = (ind_h1["macd"]  < ind_h1["macd_sig"]  and
                   ind_h1["rsi"] > 30)

    if m15_bullish and h1_bullish:
        return True, "Both timeframes bullish"
    if m15_bearish and h1_bearish:
        return True, "Both timeframes bearish"
    return False, (f"M15 {'bull' if m15_bullish else 'bear'} vs "
                   f"H1 {'bull' if h1_bullish else 'bear'} — diverged")

# ─────────────────────────────────────────────────────────
#  IMPROVEMENT 2: Trailing stop management
# ─────────────────────────────────────────────────────────

def update_trailing_stop(state: dict, current_price: float, atr: float) -> bool:
    """
    Check and update trailing stop. Returns True if stop was hit (position must close).
    Modifies state in place.
    """
    qty   = get_position()
    side  = state.get("trade_side")
    entry = state.get("trade_entry_price")
    ts    = state.get("trailing_stop")

    if qty == 0 or not side or not entry:
        # No open position — clear trailing state
        state["trailing_stop"]     = None
        state["trade_entry_price"] = None
        state["trade_side"]        = None
        return False

    trigger = TRAIL_ATR_TRIGGER * atr
    step    = TRAIL_STEP_ATR    * atr

    if side == "BUY":
        profit = current_price - entry
        if profit >= trigger:
            new_stop = current_price - step
            if ts is None or new_stop > ts:
                state["trailing_stop"] = round(new_stop, 2)
                log.info(f"  Trailing stop updated → ${state['trailing_stop']:,.2f}")
        if ts and current_price <= ts:
            log.info(f"  Trailing stop HIT at ${current_price:,.2f} (stop was ${ts:,.2f})")
            return True

    elif side == "SELL":   # short (stocks only — crypto can't short)
        profit = entry - current_price
        if profit >= trigger:
            new_stop = current_price + step
            if ts is None or new_stop < ts:
                state["trailing_stop"] = round(new_stop, 2)
                log.info(f"  Trailing stop updated → ${state['trailing_stop']:,.2f}")
        if ts and current_price >= ts:
            log.info(f"  Trailing stop HIT at ${current_price:,.2f} (stop was ${ts:,.2f})")
            return True

    return False

# ─────────────────────────────────────────────────────────
#  IMPROVEMENT 3: Portfolio-aware Claude prompt
# ─────────────────────────────────────────────────────────

def ask_claude(ind_m15: dict, ind_h1: dict, df_m15: pd.DataFrame, state: dict) -> dict:
    """Ask Claude with full portfolio context + chain-of-thought JSON response."""

    account        = trading_client.get_account()
    equity         = float(account.equity)
    cash           = float(account.cash)
    daily_pl       = equity - float(account.last_equity)
    daily_pl_pct   = daily_pl / float(account.last_equity) * 100
    current_qty    = get_position()
    position_side  = state.get("trade_side") or "NONE"
    entry_price    = state.get("trade_entry_price") or 0.0
    trailing_stop  = state.get("trailing_stop") or 0.0

    open_pnl = 0.0
    if current_qty != 0 and entry_price:
        if position_side == "BUY":
            open_pnl = (ind_m15["last"] - entry_price) * current_qty
        else:
            open_pnl = (entry_price - ind_m15["last"]) * abs(current_qty)

    session_trades = state.get("session_trades", [])
    wins  = sum(1 for t in session_trades if t.get("pnl", 0) > 0)
    total = len(session_trades)
    win_rate_str = f"{wins}/{total} ({wins/total*100:.0f}%)" if total else "N/A"

    recent_candles = df_m15.tail(20)[["open","high","low","close","volume"]].round(2).to_string()

    prompt = f"""You are a professional quantitative trading analyst. Analyze the market data and portfolio state below, then return a precise JSON trade signal.

═══════════════ PORTFOLIO STATE ═══════════════
Account equity:       ${equity:,.2f}
Available cash:       ${cash:,.2f}
Today's P&L:          ${daily_pl:+.2f}  ({daily_pl_pct:+.2f}%)
Open position:        {position_side}  qty={current_qty}  entry=${entry_price:,.2f}
Open trade P&L:       ${open_pnl:+.2f}
Trailing stop:        ${trailing_stop:,.2f} (0 = inactive)
Session win rate:     {win_rate_str}

═══════════════ M15 INDICATORS (PRIMARY) ═══════════════
Price:       {ind_m15['last']}   High: {ind_m15['high']}  Low: {ind_m15['low']}
RSI(14):     {ind_m15['rsi']}   {'→ OVERBOUGHT' if ind_m15['rsi']>70 else '→ OVERSOLD' if ind_m15['rsi']<30 else '→ neutral'}
MACD:        {ind_m15['macd']}  Signal: {ind_m15['macd_sig']}  Hist: {ind_m15['macd_hist']}
MA20/MA50:   {ind_m15['ma20']} / {ind_m15['ma50']}  → {ind_m15['ma_cross']} cross
Bollinger:   Upper={ind_m15['bb_upper']}  Lower={ind_m15['bb_lower']}
ATR(14):     {ind_m15['atr']}
Volume:      {ind_m15['vol_ratio']}x avg  {'→ HIGH (confirms move)' if ind_m15['vol_ratio']>1.5 else '→ LOW (weak)' if ind_m15['vol_ratio']<0.7 else '→ normal'}
Trend:       {ind_m15['trend']}

═══════════════ H1 INDICATORS (CONFIRMATION) ═══════════════
Price:       {ind_h1['last']}
RSI(14):     {ind_h1['rsi']}   {'→ OVERBOUGHT' if ind_h1['rsi']>70 else '→ OVERSOLD' if ind_h1['rsi']<30 else '→ neutral'}
MACD:        {ind_h1['macd']}  Signal: {ind_h1['macd_sig']}  Hist: {ind_h1['macd_hist']}
MA cross:    {ind_h1['ma_cross']}
Trend:       {ind_h1['trend']}

═══════════════ RECENT M15 CANDLES (last 20) ═══════════════
{recent_candles}

═══════════════ DECISION RULES ═══════════════
1. Classify market regime: "trending_up", "trending_down", "ranging", or "volatile"
2. BUY/SELL only when ≥3 M15 indicators AND H1 trend agree
3. Do NOT signal BUY if already LONG (position_side=BUY); do NOT signal SELL if already SHORT
4. If already in a position and seeing reversal signals → signal CLOSE instead
5. stop_loss must be at least 1× ATR from entry
6. take_profit must be at least 2× the SL distance (minimum 1:2 R:R)
7. Low volume (ratio <0.7) reduces confidence by 10 points
8. If today's P&L is already negative and confidence <75 → return HOLD
9. In "ranging" regime → prefer HOLD unless price is at a band extreme
10. confidence <{MIN_CONFIDENCE} → always return HOLD

Think step-by-step in "reasoning_steps" before giving your final signal.

Respond ONLY with valid JSON, no markdown, no extra text:
{{
  "reasoning_steps": "1. Regime is X because... 2. M15 says... 3. H1 confirms/denies... 4. Portfolio context means...",
  "regime": "trending_up",
  "signal": "BUY",
  "entry": {ind_m15['last']},
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "confidence": 75,
  "reason": "one concise sentence for the Telegram notification"
}}"""

    last_err = None
    for attempt in range(3):
        try:
            response = claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)

            log.info(f"  Claude reasoning: {result.get('reasoning_steps','')[:200]}")

            if result.get("signal") in ("BUY", "SELL"):
                if result.get("stop_loss", 0) == 0 or result.get("take_profit", 0) == 0:
                    log.warning(f"  Zero SL/TP on attempt {attempt+1}, retrying...")
                    last_err = ValueError("Zero SL/TP")
                    time.sleep(1)
                    continue

            return result

        except (json.JSONDecodeError, KeyError) as e:
            last_err = e
            log.warning(f"  Claude parse error (attempt {attempt+1}): {e}")
            time.sleep(1)

    raise RuntimeError(f"Claude failed after 3 attempts: {last_err}")

# ─────────────────────────────────────────────────────────
#  POSITION MANAGEMENT
# ─────────────────────────────────────────────────────────

def get_position() -> float:
    try:
        symbol_key = SYMBOL.replace("/", "")
        pos = trading_client.get_open_position(symbol_key)
        return float(pos.qty)
    except Exception:
        return 0.0


def close_position(state: dict, reason: str = "signal"):
    """Close position and record trade result."""
    try:
        symbol_key  = SYMBOL.replace("/", "")
        current_qty = get_position()
        entry       = state.get("trade_entry_price", 0.0)
        side        = state.get("trade_side")

        trading_client.close_position(symbol_key)
        log.info(f"  Closed {SYMBOL} position — {reason}")
        time.sleep(2)

        # Record trade
        account     = trading_client.get_account()
        exit_price  = float(trading_client.get_account().equity)  # approximate
        if entry and side and current_qty:
            current_price = calc_indicators(get_candles_m15())["last"]
            pnl = (current_price - entry) * current_qty if side == "BUY" \
                  else (entry - current_price) * abs(current_qty)
            state["session_trades"].append({
                "side":  side,
                "entry": entry,
                "exit":  current_price,
                "pnl":   round(pnl, 2),
                "reason": reason,
            })

        # Clear trailing state
        state["trailing_stop"]     = None
        state["trade_entry_price"] = None
        state["trade_side"]        = None
        save_state(state)

    except Exception as e:
        log.warning(f"  Could not close position: {e}")


def wait_for_fill(order_id: str, timeout: int = 30) -> bool:
    """Poll until order is filled or timeout. Returns True if filled."""
    for _ in range(timeout):
        try:
            order = trading_client.get_order_by_id(order_id)
            if order.status.value in ("filled", "partially_filled"):
                return True
            if order.status.value in ("canceled", "expired", "rejected"):
                log.warning(f"  Order {order_id} ended with status: {order.status}")
                return False
        except Exception:
            pass
        time.sleep(1)
    log.warning(f"  Order {order_id} not confirmed filled after {timeout}s")
    return False

# ─────────────────────────────────────────────────────────
#  ORDER EXECUTION
# ─────────────────────────────────────────────────────────

def place_order(signal: dict, state: dict):
    action = signal["signal"]

    if action == "HOLD":
        log.info(f"  → HOLD | {signal['reason']}")
        return

    if signal["confidence"] < MIN_CONFIDENCE:
        log.info(f"  → Skipped — confidence {signal['confidence']}% < {MIN_CONFIDENCE}%")
        return

    current_qty = get_position()

    # Handle CLOSE signal (reversal detected by Claude)
    if action == "CLOSE":
        if current_qty != 0:
            close_position(state, reason="Claude CLOSE signal")
            send_telegram(f"🔄 <b>POSITION CLOSED</b>\n💬 {signal['reason']}")
        return

    # Crypto: no shorting
    if MARKET_TYPE == "crypto" and action == "SELL":
        if current_qty > 0:
            close_position(state, reason="SELL signal (crypto close long)")
            send_telegram(
                f"🔴 <b>LONG CLOSED — {SYMBOL}</b>\n"
                f"💬 {signal['reason']}\n"
                f"🎯 Confidence: {signal['confidence']}%"
            )
        else:
            log.info("  → SELL skipped — no long to close (crypto can't short)")
        return

    # Skip if already in same direction
    if action == "BUY"  and current_qty > 0:
        log.info("  → Already LONG, skipping")
        return
    if action == "SELL" and current_qty < 0:
        log.info("  → Already SHORT, skipping")
        return

    # Slippage guard: skip if price moved >0.5% from Claude's entry
    entry_price  = signal["entry"]
    current_price = signal["entry"]   # will be overridden below if we can
    try:
        current_price = calc_indicators(get_candles_m15())["last"]
        slippage_pct  = abs(current_price - entry_price) / entry_price
        if slippage_pct > 0.005:
            log.warning(f"  → Skipped — slippage {slippage_pct:.3%} > 0.5% "
                        f"(entry={entry_price}, now={current_price})")
            return
    except Exception:
        pass  # proceed if we can't check

    # Close opposite position first
    if current_qty != 0:
        close_position(state, reason="reversing position")

    side     = OrderSide.BUY if action == "BUY" else OrderSide.SELL
    sl_price = round(signal["stop_loss"],   2)
    tp_price = round(signal["take_profit"], 2)

    # ── IMPROVEMENT 4: risk-based notional ──────────────
    notional = calc_notional(entry_price, sl_price)

    def _build_order(bracket: bool):
        if MARKET_TYPE == "crypto":
            kwargs = dict(symbol=SYMBOL.replace("/", ""), notional=notional,
                          side=side, time_in_force=TimeInForce.GTC)
        else:
            qty_shares = max(1, int(notional / entry_price))
            kwargs     = dict(symbol=SYMBOL, qty=qty_shares,
                              side=side, time_in_force=TimeInForce.DAY)
        if bracket:
            kwargs["order_class"] = OrderClass.BRACKET
            kwargs["take_profit"] = TakeProfitRequest(limit_price=tp_price)
            kwargs["stop_loss"]   = StopLossRequest(stop_price=sl_price)
        return MarketOrderRequest(**kwargs)

    order = None
    try:
        order = trading_client.submit_order(_build_order(bracket=True))
        log.info("  → Bracket order placed (SL/TP attached)")
    except Exception as e:
        log.warning(f"  Bracket rejected ({e}), trying plain market order...")
        try:
            order = trading_client.submit_order(_build_order(bracket=False))
            log.warning("  → Plain market order placed — monitor SL/TP manually!")
        except Exception as e2:
            log.error(f"  → Order FAILED: {e2}")
            return

    # Confirm fill
    filled = wait_for_fill(str(order.id))
    if not filled:
        log.warning("  → Order not confirmed filled — check manually")

    # Update state for trailing stop tracking
    state["trade_side"]        = action
    state["trade_entry_price"] = entry_price
    state["trailing_stop"]     = None
    save_state(state)

    log.info(f"  → {action} | ${notional:.2f} notional | conf={signal['confidence']}% | "
             f"SL={sl_price} TP={tp_price}")

    icon = "🟢" if action == "BUY" else "🔴"
    send_telegram(
        f"{icon} <b>{action} ORDER PLACED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💱 Symbol:      <b>{SYMBOL}</b>\n"
        f"💰 Notional:    <b>${notional:,.2f}</b>\n"
        f"🎯 Confidence:  {signal['confidence']}%\n"
        f"📐 Regime:      {signal.get('regime','?')}\n"
        f"📍 Entry:       ~{entry_price:,}\n"
        f"🛑 Stop Loss:   {sl_price:,}\n"
        f"✅ Take Profit: {tp_price:,}\n"
        f"🆔 Order ID:    <code>{order.id}</code>\n"
        f"💡 {signal['reason']}"
    )

# ─────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  Claude AI Trading Bot v2 — Alpaca Markets")
    log.info("=" * 60)
    print_account()
    log.info(f"  Symbol:          {SYMBOL}")
    log.info(f"  Timeframes:      M15 (primary) + H1 (confirmation)")
    log.info(f"  Risk per trade:  {RISK_PER_TRADE_PCT:.0%} of equity")
    log.info(f"  Min confidence:  {MIN_CONFIDENCE}%")
    log.info(f"  Max drawdown:    {MAX_DRAWDOWN_PCT:.0%} from peak")
    log.info("=" * 60)

    state   = load_state()
    account = trading_client.get_account()
    equity  = float(account.equity)

    if state["peak_equity"] == 0:
        state["peak_equity"] = equity
        save_state(state)

    send_telegram(
        f"🤖 <b>Claude Trading Bot v2 Started</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💱 Symbol:      <b>{SYMBOL}</b>\n"
        f"⏱ Timeframes:  M15 + H1 confirmation\n"
        f"💵 Equity:      <b>${equity:,.2f}</b>\n"
        f"📐 Risk/trade:  {RISK_PER_TRADE_PCT:.0%} of equity\n"
        f"🎯 Min conf:    {MIN_CONFIDENCE}%\n"
        f"🛡 Loss limit:  {DAILY_LOSS_LIMIT_PCT:.0%}/day | {MAX_DRAWDOWN_PCT:.0%} max DD"
    )

    while True:
        state["cycle"] = state.get("cycle", 0) + 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.info(f"\n[Cycle #{state['cycle']}] {now}")

        # ── Risk gates ──────────────────────────────────
        if is_daily_loss_limit_hit() or is_max_drawdown_hit(state):
            time.sleep(60 * 60)
            continue

        signal   = None
        mtf_ok   = False
        ind_m15  = {}
        ind_h1   = {}

        try:
            # 1. Fetch both timeframes
            df_m15  = get_candles_m15()
            df_h1   = get_candles_h1()
            ind_m15 = calc_indicators(df_m15)
            ind_h1  = calc_indicators(df_h1)

            log.info(
                f"  M15 → Price={ind_m15['last']} RSI={ind_m15['rsi']} "
                f"MACD={ind_m15['macd']} Trend={ind_m15['trend']}"
            )
            log.info(
                f"  H1  → Price={ind_h1['last']}  RSI={ind_h1['rsi']} "
                f"MACD={ind_h1['macd']} Trend={ind_h1['trend']}"
            )

            # ── IMPROVEMENT 2: Check trailing stop first ──
            trail_hit = update_trailing_stop(state, ind_m15["last"], ind_m15["atr"])
            if trail_hit:
                log.info("  Trailing stop triggered — closing position")
                close_position(state, reason="trailing stop")
                send_telegram(
                    f"🔔 <b>TRAILING STOP HIT — {SYMBOL}</b>\n"
                    f"📍 Price: ${ind_m15['last']:,.2f}\n"
                    f"🛑 Stop was: ${state.get('trailing_stop') or 0:,.2f}"
                )
                save_state(state)
                time.sleep(SLEEP_SECS)
                continue

            # ── IMPROVEMENT 1: Multi-timeframe gate ──────
            mtf_ok, mtf_reason = mtf_agrees(ind_m15, ind_h1)
            log.info(f"  MTF check: {'✅' if mtf_ok else '❌'} {mtf_reason}")

            # Pre-filter: skip weak setups (no Claude API call wasted)
            rsi_neutral = 42 < ind_m15["rsi"] < 58
            macd_weak   = abs(ind_m15["macd_hist"]) < ind_m15["last"] * 0.0001

            if not mtf_ok:
                log.info(f"  Skipping Claude — MTF diverged: {mtf_reason}")
            elif rsi_neutral and macd_weak:
                log.info(f"  Skipping Claude — weak M15 setup (RSI={ind_m15['rsi']} MACD hist={ind_m15['macd_hist']})")
            else:
                # ── IMPROVEMENT 3: Portfolio-aware Claude ─
                signal = ask_claude(ind_m15, ind_h1, df_m15, state)
                log.info(f"  Claude → {signal['signal']} | {signal['confidence']}% | regime={signal.get('regime')}")
                log.info(f"  Reason: {signal['reason']}")

                place_order(signal, state)

            # Status update to Telegram every cycle
            send_telegram(build_status_msg(ind_m15, ind_h1, signal, mtf_ok, state))

        except Exception as e:
            log.error(f"  Cycle error: {e}", exc_info=True)
            send_telegram(f"⚠️ <b>Bot Error</b>\n<code>{e}</code>")

        save_state(state)
        log.info(f"  Sleeping {SLEEP_SECS // 60} min...\n")
        time.sleep(SLEEP_SECS)


if __name__ == "__main__":
    main()
