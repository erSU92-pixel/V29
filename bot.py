#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD Paper Bot — GitHub Actions Edition
Tasks: tick | retrain | health | report
Data: Yahoo Finance (no signup)
"""
import argparse, json, logging, math, os, sqlite3, sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import joblib, numpy as np, pandas as pd, requests
from pytz import timezone as ptz
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ============================================================
# CONFIG
# ============================================================
def _s(k, d=""): return (os.getenv(k) or "").strip() or d
def _f(k, d):
    try: return float(_s(k) or d)
    except Exception: return d


@dataclass(frozen=True)
class CFG:
    db_path:    str = _s("DB_PATH", "state/state.db")
    model_dir:  str = _s("MODEL_DIR", "state/models")
    tg_token:   str = _s("TELEGRAM_TOKEN")
    tg_chat:    str = _s("TELEGRAM_CHAT_ID")   # optional — auto-detect if empty

    symbol: str = "XAU/USD"
    contract_size: float = 100.0
    atr_mult_sl: float = 1.5
    rr: float = 2.0
    max_bars: int = 96
    min_oos_auc: float = 0.55
    min_oos_precision: float = 0.42
    min_prob: float = 0.42

    spread_usd: float = 0.35
    slippage_usd: float = 0.10
    slippage_bps: float = 2.0
    max_spread_usd: float = 1.20

    cooldown_min: int = 120
    daily_max_signals: int = 3
    daily_max_loss_pct: float = 4.0
    weekly_max_loss_pct: float = 8.0
    risk_pct: float = 1.0
    kelly_fraction: float = 0.25
    min_lot: float = 0.01
    max_lot: float = 1.0
    initial_equity: float = _f("ACCOUNT_EQUITY", 1000.0)

    cb_loss_streak: int = 5
    cb_loss_streak_cooldown_h: int = 24


CFG = CFG()
LON = ptz("Europe/London")
NY  = ptz("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("bot")


# ============================================================
# UTILS + TELEGRAM (auto-detect chat)
# ============================================================
_tg_chat: Optional[str] = None


def now_utc() -> datetime: return datetime.now(timezone.utc)
def iso(dt: datetime) -> str: return dt.astimezone(timezone.utc).isoformat()


def resolve_chat(store) -> Optional[str]:
    """Find or auto-detect Telegram chat_id (via getUpdates)."""
    global _tg_chat
    if _tg_chat: return _tg_chat
    if CFG.tg_chat:
        _tg_chat = CFG.tg_chat; return _tg_chat
    saved = store.get_meta("tg_chat_id")
    if saved:
        _tg_chat = saved; return _tg_chat
    if not CFG.tg_token:
        log.warning("no TELEGRAM_TOKEN — notifications disabled")
        return None
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{CFG.tg_token}/getUpdates",
            params={"limit": 20, "timeout": 0}, timeout=10)
        d = r.json()
        if not d.get("ok"):
            log.warning("getUpdates: %s", d.get("description"))
            return None
        for u in reversed(d.get("result", [])):
            m = u.get("message") or u.get("channel_post")
            if m and m.get("chat", {}).get("id"):
                cid = str(m["chat"]["id"])
                store.set_meta("tg_chat_id", cid)
                _tg_chat = cid
                log.info("chat_id auto-detected: %s", cid)
                return cid
        log.info("no message from user yet — kirim /start ke bot")
    except Exception as e:
        log.warning("getUpdates: %s", e)
    return None


def send_telegram(msg: str, silent: bool = False) -> bool:
    chat = _tg_chat or CFG.tg_chat
    if not CFG.tg_token or not chat: return False
    for attempt in range(3):
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{CFG.tg_token}/sendMessage",
                json={"chat_id": chat, "text": msg,
                      "parse_mode": "HTML", "disable_notification": silent},
                timeout=15)
            if r.status_code == 200: return True
            if r.status_code == 429:
                import time as t
                t.sleep(int(r.json().get("parameters", {}).get("retry_after", 5)) + 1)
                continue
            if 500 <= r.status_code < 600:
                import time as t; t.sleep(2*(attempt+1)); continue
            return False
        except Exception:
            import time as t; t.sleep(2*(attempt+1))
    return False


# ============================================================
# STATE STORE
# ============================================================
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
    symbol TEXT NOT NULL, side TEXT NOT NULL, entry REAL NOT NULL,
    sl REAL NOT NULL, tp REAL NOT NULL, lot REAL NOT NULL,
    risk_usd REAL NOT NULL, prob REAL NOT NULL, source TEXT,
    status TEXT NOT NULL, broker_position_id TEXT, closed_at TEXT,
    exit_price REAL, pnl REAL, outcome TEXT
);
CREATE INDEX IF NOT EXISTS ix_signals_status ON signals(status);
CREATE INDEX IF NOT EXISTS ix_signals_created ON signals(created_at);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL REFERENCES signals(id),
    symbol TEXT NOT NULL, side TEXT NOT NULL, entry REAL NOT NULL,
    current_sl REAL NOT NULL, current_tp REAL NOT NULL, lot REAL NOT NULL,
    open_at TEXT NOT NULL, broker_position_id TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
    equity REAL NOT NULL, daily_pnl REAL NOT NULL, mode TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cb_state (
    name TEXT PRIMARY KEY, tripped INTEGER NOT NULL DEFAULT 0,
    tripped_at TEXT, reason TEXT, cooldown_until TEXT
);
CREATE TABLE IF NOT EXISTS model_registry (
    version TEXT PRIMARY KEY, created_at TEXT NOT NULL,
    auc REAL, precision REAL, is_active INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
    event TEXT NOT NULL, payload TEXT NOT NULL
);
"""


class StateStore:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, timeout=30)
        self._conn.executescript(SCHEMA); self._conn.commit()

    def get_meta(self, k, d=None):
        r = self._conn.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r[0] if r else d

    def set_meta(self, k, v):
        self._conn.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (k, str(v)))
        self._conn.commit()

    def daily_reset_if_needed(self):
        today = now_utc().strftime("%Y-%m-%d")
        if self.get_meta("last_reset_date") == today: return
        eq = self.current_equity()
        self.set_meta("last_reset_date", today)
        self.set_meta("daily_signals", "0")
        self.set_meta("daily_start_equity", str(eq))

    def daily_signals(self): return int(self.get_meta("daily_signals", "0"))
    def inc_daily_signals(self):
        self.set_meta("daily_signals", str(self.daily_signals() + 1))

    def daily_pnl(self) -> float:
        r = self._conn.execute(
            "SELECT COALESCE(SUM(pnl),0) FROM signals WHERE closed_at >= ?",
            (now_utc().strftime("%Y-%m-%dT00:00:00+00:00"),)).fetchone()
        return float(r[0] or 0.0)

    def weekly_pnl(self) -> float:
        r = self._conn.execute(
            "SELECT COALESCE(SUM(pnl),0) FROM signals WHERE closed_at >= ?",
            ((now_utc() - timedelta(days=7)).isoformat(),)).fetchone()
        return float(r[0] or 0.0)

    def current_equity(self) -> float:
        return float(self.get_meta("current_equity", str(CFG.initial_equity)))

    def update_equity(self, delta):
        self.set_meta("current_equity", str(self.current_equity() + delta))

    def last_signal_time(self) -> Optional[datetime]:
        t = self.get_meta("last_signal_time")
        return datetime.fromisoformat(t) if t else None

    def set_last_signal_time(self, dt): self.set_meta("last_signal_time", dt.isoformat())

    def insert_signal(self, sig, status="FILLED", pos_id=None) -> int:
        cur = self._conn.execute(
            """INSERT INTO signals(created_at,symbol,side,entry,sl,tp,lot,risk_usd,
                                   prob,source,status,broker_position_id)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sig["created_at"], sig["symbol"], sig["side"], sig["entry"],
             sig["sl"], sig["tp"], sig["lot"], sig["risk_usd"], sig["prob"],
             sig.get("source"), status, pos_id))
        self._conn.commit(); return cur.lastrowid

    def update_signal(self, sid, **kw):
        cols = ",".join(f"{k}=?" for k in kw)
        self._conn.execute(f"UPDATE signals SET {cols} WHERE id=?", (*kw.values(), sid))
        self._conn.commit()

    def open_positions(self) -> List[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute("SELECT * FROM positions").fetchall()
        self._conn.row_factory = None; return rows

    def insert_position(self, pos) -> int:
        cur = self._conn.execute(
            """INSERT INTO positions(signal_id,symbol,side,entry,current_sl,
                                    current_tp,lot,open_at,broker_position_id)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (pos["signal_id"], pos["symbol"], pos["side"], pos["entry"],
             pos["current_sl"], pos["current_tp"], pos["lot"],
             pos["open_at"], pos.get("broker_position_id")))
        self._conn.commit(); return cur.lastrowid

    def delete_position(self, pid):
        self._conn.execute("DELETE FROM positions WHERE id=?", (pid,))
        self._conn.commit()

    def recent_closed(self, n=50):
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute(
            "SELECT * FROM signals WHERE status='CLOSED' "
            "AND outcome IN ('WIN','LOSS') ORDER BY closed_at DESC LIMIT ?", (n,)
        ).fetchall()
        self._conn.row_factory = None; return rows

    def cb_tripped(self, name) -> bool:
        r = self._conn.execute(
            "SELECT tripped,cooldown_until FROM cb_state WHERE name=?", (name,)).fetchone()
        if not r or not r[0]: return False
        if r[1] and datetime.fromisoformat(r[1]) < now_utc():
            self._conn.execute("UPDATE cb_state SET tripped=0 WHERE name=?", (name,))
            self._conn.commit(); return False
        return True

    def cb_trip(self, name, reason, cooldown_min=None):
        until = ((now_utc() + timedelta(minutes=cooldown_min)).isoformat()
                 if cooldown_min else None)
        self._conn.execute(
            "INSERT OR REPLACE INTO cb_state(name,tripped,tripped_at,reason,cooldown_until)"
            " VALUES(?,1,?,?,?)",
            (name, now_utc().isoformat(), reason, until))
        self._conn.commit(); log.error("CB trip: %s | %s", name, reason)

    def cb_reset(self, name):
        self._conn.execute("UPDATE cb_state SET tripped=0 WHERE name=?", (name,))
        self._conn.commit()

    def audit(self, event, payload):
        self._conn.execute("INSERT INTO audit(ts,event,payload) VALUES(?,?,?)",
                           (now_utc().isoformat(), event, json.dumps(payload, default=str)))
        self._conn.commit()

    def snapshot_equity(self):
        self._conn.execute(
            "INSERT INTO equity(ts,equity,daily_pnl,mode) VALUES(?,?,?,?)",
            (now_utc().isoformat(), self.current_equity(), self.daily_pnl(), "paper"))
        self._conn.commit()

    def heartbeat(self): self.set_meta("last_heartbeat", iso(now_utc()))

    def wal_checkpoint(self):
        try: self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception: pass

    def close(self):
        try: self._conn.close()
        except Exception: pass


# ============================================================
# CIRCUIT BREAKERS
# ============================================================
def check_cb(store) -> Tuple[bool, Optional[str]]:
    for name in ("global_kill", "loss_streak", "daily_dd", "weekly_dd", "model_degraded"):
        if store.cb_tripped(name): return False, name
    eq = store.current_equity()
    start = float(store.get_meta("daily_start_equity", str(eq)))
    if store.daily_pnl() <= -(CFG.daily_max_loss_pct * start / 100):
        store.cb_trip("daily_dd", f"daily={store.daily_pnl():.2f}", 24*60)
        return False, "daily_dd"
    if store.weekly_pnl() <= -(CFG.weekly_max_loss_pct * eq / 100):
        store.cb_trip("weekly_dd", f"weekly={store.weekly_pnl():.2f}", 7*24*60)
        return False, "weekly_dd"
    recent = store.recent_closed(CFG.cb_loss_streak)
    if len(recent) >= CFG.cb_loss_streak and \
       all(r["outcome"] == "LOSS" for r in recent[:CFG.cb_loss_streak]):
        store.cb_trip("loss_streak", f"{CFG.cb_loss_streak} streak",
                      CFG.cb_loss_streak_cooldown_h * 60)
        return False, "loss_streak"
    return True, None


# ============================================================
# DATA (Yahoo only — no signup)
# ============================================================
def fetch_candles() -> pd.DataFrame:
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
           "?interval=15m&range=60d")
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            r.raise_for_status()
            res = r.json().get("chart", {}).get("result")
            if not res: raise ValueError("empty result")
            item = res[0]
            ts = item.get("timestamp") or []
            q = (item.get("indicators", {}).get("quote") or [{}])[0]
            if not ts or not q.get("close"): raise ValueError("missing")
            df = pd.DataFrame({
                "datetime": pd.to_datetime(ts, unit="s", utc=True),
                "open": q["open"], "high": q["high"],
                "low": q["low"], "close": q["close"],
            }).dropna()
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce") - 25.0  # GC=F → spot proxy
            return df.dropna().sort_values("datetime").reset_index(drop=True)
        except Exception as e:
            last = e
            import time as t; t.sleep(2*(attempt+1))
    raise RuntimeError(f"yahoo: {last}")


def sanitize(df) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 500: return None
    df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    last_ts = df["datetime"].iloc[-1]
    now = now_utc().replace(second=0, microsecond=0)
    expected = now - timedelta(minutes=now.minute % 15)
    if last_ts >= expected:
        df = df.iloc[:-1].reset_index(drop=True)
    return df if len(df) >= 400 else None


# ============================================================
# FEATURES
# ============================================================
FEATURES = [
    "rsi","adx","di_difference","ema_distance","ema_slope","trend_strength",
    "atr_pct","atr_regime","realized_vol","vol_zscore","bb_position","keltner_position",
    "body_ratio","upper_wick_ratio","lower_wick_ratio","bos_score","liquidity_sweep_score",
    "return_1","return_3","return_5","hour_sin","hour_cos","dayofweek",
    "session_london","session_newyork","session_overlap",
]


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["close"].diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    rs = g / l.replace(0, np.nan)
    df["rsi"] = (100 - 100/(1 + rs)).fillna(50.0)

    pc = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"], (df["high"]-pc).abs(),
                    (df["low"]-pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    df["atr_pct"] = df["atr"] / (df["close"] + 1e-12)
    df["atr_regime"] = df["atr"] / (df["atr"].rolling(100).mean() + 1e-12)

    ef = df["close"].ewm(span=9, adjust=False).mean()
    es = df["close"].ewm(span=21, adjust=False).mean()
    et = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_distance"] = (ef-es)/(df["atr"]+1e-12)
    df["ema_slope"] = (ef-ef.shift(3))/(df["atr"]+1e-12)
    df["trend_strength"] = (df["close"]-et)/(df["atr"]+1e-12)

    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = np.where((up>dn)&(up>0), up, 0.0)
    ndm = np.where((dn>up)&(dn>0), dn, 0.0)
    pdi = 100*pd.Series(pdm, index=df.index).ewm(alpha=1/14, adjust=False).mean()/(df["atr"]+1e-12)
    ndi = 100*pd.Series(ndm, index=df.index).ewm(alpha=1/14, adjust=False).mean()/(df["atr"]+1e-12)
    df["di_difference"] = pdi - ndi
    df["adx"] = (100*(pdi-ndi).abs()/(pdi+ndi+1e-12)).ewm(alpha=1/14, adjust=False).mean()

    m, s = df["close"].rolling(20).mean(), df["close"].rolling(20).std()
    df["bb_position"] = (df["close"]-m)/(2*s+1e-12)
    km = df["close"].ewm(span=20, adjust=False).mean()
    kb = df["atr"].ewm(span=20, adjust=False).mean()*2
    df["keltner_position"] = (df["close"]-km)/(kb+1e-12)

    ret = df["close"].pct_change()
    df["realized_vol"] = ret.rolling(20).std()*np.sqrt(96)
    rv_std = df["realized_vol"].rolling(100).std()
    df["vol_zscore"] = ((df["realized_vol"] - df["realized_vol"].rolling(100).mean())
                       / (rv_std + 1e-12)).clip(-10, 10)

    rng = (df["high"]-df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"]-df["open"]).abs()/(rng+1e-12)
    df["upper_wick_ratio"] = (df["high"]-df[["open","close"]].max(axis=1))/(rng+1e-12)
    df["lower_wick_ratio"] = (df[["open","close"]].min(axis=1)-df["low"])/(rng+1e-12)

    ph = df["high"].rolling(5).max().shift(1)
    pl = df["low"].rolling(5).min().shift(1)
    df["bos_score"] = np.where((df["high"]>ph)&(df["close"]>ph), 1.0,
                       np.where((df["low"]<pl)&(df["close"]<pl), -1.0, 0.0))
    df["liquidity_sweep_score"] = np.where((df["low"]<pl)&(df["close"]>pl), 1.0,
                                   np.where((df["high"]>ph)&(df["close"]<ph), -1.0, 0.0))

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    df["hour_sin"] = np.sin(2*np.pi*df["datetime"].dt.hour/24)
    df["hour_cos"] = np.cos(2*np.pi*df["datetime"].dt.hour/24)
    df["dayofweek"] = df["datetime"].dt.dayofweek
    lon_hr = df["datetime"].dt.tz_convert(LON).dt.hour
    ny_min = df["datetime"].dt.tz_convert(NY).dt.hour*60 + \
             df["datetime"].dt.tz_convert(NY).dt.minute
    df["session_london"] = lon_hr.between(8, 16).astype(int)
    df["session_newyork"] = ((ny_min >= 9*60+30) & (ny_min < 17*60)).astype(int)
    df["session_overlap"] = (df["session_london"] & df["session_newyork"]).astype(int)

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


# ============================================================
# LABELING
# ============================================================
if HAS_NUMBA:
    @njit(cache=True)
    def _label_k(c, h, l, a, tp_mult, sl_mult, max_bars):
        n = len(c)
        bw = np.zeros(n, dtype=np.int8); sw = np.zeros(n, dtype=np.int8)
        v = np.zeros(n, dtype=np.bool_)
        for i in range(n - max_bars):
            ai = a[i]
            if not np.isfinite(ai) or ai <= 0: continue
            v[i] = True
            tp_u = c[i] + ai*tp_mult; sl_u = c[i] - ai*sl_mult
            tp_d = c[i] - ai*tp_mult; sl_d = c[i] + ai*sl_mult
            for j in range(i+1, i+max_bars+1):
                if l[j] <= sl_u: break
                if h[j] >= tp_u: bw[i] = 1; break
            for j in range(i+1, i+max_bars+1):
                if h[j] >= sl_d: break
                if l[j] <= tp_d: sw[i] = 1; break
        return bw, sw, v


def _label_py(c, h, l, a, tp_mult, sl_mult, max_bars):
    n = len(c)
    bw = np.zeros(n, dtype=np.int8); sw = np.zeros(n, dtype=np.int8)
    v = np.zeros(n, dtype=np.bool_)
    for i in range(n - max_bars):
        ai = a[i]
        if not np.isfinite(ai) or ai <= 0: continue
        v[i] = True
        tp_u = c[i] + ai*tp_mult; sl_u = c[i] - ai*sl_mult
        tp_d = c[i] - ai*tp_mult; sl_d = c[i] + ai*sl_mult
        for j in range(i+1, i+max_bars+1):
            if l[j] <= sl_u: break
            if h[j] >= tp_u: bw[i] = 1; break
        for j in range(i+1, i+max_bars+1):
            if h[j] >= sl_d: break
            if l[j] <= tp_d: sw[i] = 1; break
    return bw, sw, v


def label_direction(df, tp_mult, sl_mult, max_bars):
    df = df.copy()
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    a = df["atr"].values.astype(np.float64)
    if HAS_NUMBA: bw, sw, v = _label_k(c, h, l, a, tp_mult, sl_mult, max_bars)
    else: bw, sw, v = _label_py(c, h, l, a, tp_mult, sl_mult, max_bars)
    df["buy_win"] = bw; df["sell_win"] = sw; df["label_valid"] = v
    return df


# ============================================================
# MODEL
# ============================================================
def build_model():
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                                max_features="sqrt", random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.03,
                                    subsample=0.8, random_state=42)
    ens = VotingClassifier([("rf", rf), ("gb", gb)], voting="soft", n_jobs=-1)
    return CalibratedClassifierCV(ens, method="sigmoid", cv=3)


def walk_forward_gate(X, y, fwd_bars, n_folds=4) -> Tuple[bool, dict]:
    n = len(X)
    fold = n // (n_folds + 1)
    if fold < 200: return False, {"reason": "too few samples"}
    aucs, precs = [], []
    for k in range(n_folds):
        tr_end = fold * (k+1)
        va_end = min(tr_end + fold, n)
        tr_idx = np.arange(0, max(0, tr_end - fwd_bars))
        va_idx = np.arange(tr_end, va_end)
        if len(tr_idx) < 300 or len(va_idx) < 100: continue
        if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[va_idx])) < 2: continue
        w = compute_sample_weight("balanced", y[tr_idx]) * \
            np.exp(-np.arange(len(tr_idx))[::-1] / 500.0)
        try:
            m = build_model(); m.fit(X[tr_idx], y[tr_idx], sample_weight=w)
            p = m.predict_proba(X[va_idx])[:, 1]
            aucs.append(roc_auc_score(y[va_idx], p))
            precs.append(precision_score(y[va_idx], m.predict(X[va_idx]), zero_division=0))
        except Exception as e:
            log.warning("fold %d: %s", k, e)
    if not aucs: return False, {"reason": "all folds failed"}
    ma, mp = float(np.mean(aucs)), float(np.mean(precs))
    ok = (ma >= CFG.min_oos_auc) and (mp >= CFG.min_oos_precision)
    return ok, {"folds": len(aucs), "mean_auc": ma, "mean_prec": mp}


def _train_core(store, df) -> bool:
    df_feat = calculate_features(df)
    if len(df_feat) < 400: return False
    df_lab = label_direction(df_feat, CFG.atr_mult_sl * CFG.rr,
                              CFG.atr_mult_sl, CFG.max_bars)
    train = df_lab[df_lab["label_valid"]]
    if len(train) < 400: return False
    X = train[FEATURES].values
    result = {"buy": None, "sell": None, "metrics": {}}
    for name, ycol in (("buy", "buy_win"), ("sell", "sell_win")):
        y = train[ycol].values
        ok, m = walk_forward_gate(X, y, CFG.max_bars)
        result["metrics"][name] = {**m, "gate_ok": ok}
        if not ok:
            log.warning("[%s] gate FAIL: %s", name, m); continue
        w = compute_sample_weight("balanced", y) * \
            np.exp(-np.arange(len(y))[::-1] / 500.0)
        mdl = build_model(); mdl.fit(X, y, sample_weight=w)
        result[name] = mdl
    if result["buy"] is None and result["sell"] is None:
        return False
    Path(CFG.model_dir).mkdir(parents=True, exist_ok=True)
    version = now_utc().strftime("%Y%m%d_%H%M%S")
    joblib.dump({"buy": result["buy"], "sell": result["sell"],
                 "metrics": result["metrics"], "features": FEATURES},
                Path(CFG.model_dir) / f"{version}.joblib")
    store._conn.execute("UPDATE model_registry SET is_active=0")
    store._conn.execute(
        "INSERT OR REPLACE INTO model_registry(version,created_at,auc,precision,is_active)"
        " VALUES(?,?,?,?,1)",
        (version, now_utc().isoformat(),
         result["metrics"].get("buy", {}).get("mean_auc", 0),
         result["metrics"].get("buy", {}).get("mean_prec", 0)))
    store._conn.commit()
    return True, version, result["metrics"]


def load_active_models(store) -> Optional[dict]:
    r = store._conn.execute(
        "SELECT version FROM model_registry WHERE is_active=1 LIMIT 1").fetchone()
    if not r: return None
    p = Path(CFG.model_dir) / f"{r[0]}.joblib"
    if not p.exists(): return None
    try: return joblib.load(p)
    except Exception as e:
        log.error("load model: %s", e); return None


# ============================================================
# SESSION / NEWS / REGIME
# ============================================================
def is_valid_session() -> bool:
    t = now_utc()
    if t.weekday() >= 5: return False
    lon, ny = t.astimezone(LON), t.astimezone(NY)
    london = 8 <= lon.hour < 17
    ny_active = (ny.hour > 9 or (ny.hour == 9 and ny.minute >= 30)) and ny.hour < 17
    return london or ny_active


def is_news_window() -> bool:
    t = now_utc().astimezone(NY)
    if t.weekday() >= 5: return False
    if t.hour == 8 and 25 <= t.minute <= 45: return True
    if t.hour == 14 and t.minute <= 30: return True
    return False


def regime_ok(row) -> Tuple[bool, str]:
    regime = row.get("atr_regime", 1.0)
    adx = row.get("adx", 20.0)
    vol_z = row.get("vol_zscore", 0.0)
    if regime > 2.5: return False, f"vol_spike ({regime:.2f})"
    if regime < 0.5: return False, f"dead ({regime:.2f})"
    if adx < 15 and abs(vol_z) < 1.0: return False, f"choppy (adx={adx:.0f})"
    return True, "ok"


# ============================================================
# RISK
# ============================================================
def compute_risk_pct(store) -> float:
    recent = store.recent_closed(30)
    if len(recent) < 15: return min(CFG.risk_pct, 0.25)
    wr = sum(1 for r in recent if r["outcome"] == "WIN") / len(recent)
    p, q, b = wr, 1 - wr, CFG.rr
    kelly = max((p*b - q) / b, 0.0) * CFG.kelly_fraction
    return 0.0 if kelly <= 0 else min(kelly * 100, CFG.risk_pct)


def compute_lot(store, sl_dist: float) -> Optional[float]:
    if sl_dist < 1e-6: return None
    rp = compute_risk_pct(store)
    if rp <= 0: return None
    risk_usd = store.current_equity() * rp / 100
    raw = risk_usd / (sl_dist * CFG.contract_size)
    if not np.isfinite(raw) or raw < CFG.min_lot: return None
    return min(math.floor(raw * 100) / 100, CFG.max_lot)


# ============================================================
# PAPER EXECUTION
# ============================================================
def paper_check_exits(store, df):
    if df is None or df.empty: return
    for row in store.open_positions():
        try:
            open_at = datetime.fromisoformat(row["open_at"])
            if open_at.tzinfo is None: open_at = open_at.replace(tzinfo=timezone.utc)
        except Exception: continue
        bars = df[df["datetime"] > open_at]
        if bars.empty: continue
        horizon = open_at + timedelta(minutes=CFG.max_bars * 15)
        outcome, exit_price = None, None
        for _, bar in bars.iterrows():
            if row["side"] == "BUY":
                if bar["low"] <= row["current_sl"]:
                    outcome, exit_price = "LOSS", row["current_sl"]; break
                if bar["high"] >= row["current_tp"]:
                    outcome, exit_price = "WIN", row["current_tp"]; break
            else:
                if bar["high"] >= row["current_sl"]:
                    outcome, exit_price = "LOSS", row["current_sl"]; break
                if bar["low"] <= row["current_tp"]:
                    outcome, exit_price = "WIN", row["current_tp"]; break
            if bar["datetime"] >= horizon:
                outcome, exit_price = "TIMEOUT", float(bar["close"]); break
        if outcome is None: continue
        slip = CFG.slippage_usd + abs(exit_price) * CFG.slippage_bps / 10000
        eff = exit_price - slip if row["side"] == "BUY" else exit_price + slip
        pnl = ((eff - row["entry"]) if row["side"] == "BUY"
               else (row["entry"] - eff)) * row["lot"] * CFG.contract_size
        store.update_signal(row["signal_id"], status="CLOSED", outcome=outcome,
                            pnl=pnl, exit_price=eff, closed_at=iso(now_utc()))
        store.delete_position(row["id"])
        store.update_equity(pnl)
        store.audit("closed", {"signal_id": row["signal_id"], "outcome": outcome,
                                "pnl": pnl, "exit": eff})
        log.info("closed %s → %s | $%.2f", row["side"], outcome, pnl)


def paper_execute(store, sig, current_price):
    slip = CFG.slippage_usd + abs(current_price) * CFG.slippage_bps / 10000
    half = CFG.spread_usd / 2
    fill = (current_price + half + slip if sig["side"] == "BUY"
            else current_price - half - slip)
    pid = f"PAPER-{now_utc().strftime('%Y%m%d%H%M%S%f')}"
    sid = store.insert_signal(sig, "FILLED", pid)
    store.insert_position({
        "signal_id": sid, "symbol": sig["symbol"], "side": sig["side"],
        "entry": fill, "current_sl": sig["sl"], "current_tp": sig["tp"],
        "lot": sig["lot"], "open_at": iso(now_utc()), "broker_position_id": pid,
    })
    return sid, fill


# ============================================================
# SIGNAL GENERATION
# ============================================================
def generate_signal(store, df_feat, models) -> Optional[dict]:
    last = df_feat.iloc[-1]
    X = last[FEATURES].values.reshape(1, -1).astype(np.float64)
    buy_p = sell_p = 0.0
    if models.get("buy"):
        try:
            p = models["buy"].predict_proba(X)[0]
            buy_p = float(p[1]) if 1 in models["buy"].classes_ else 0.0
        except Exception: pass
    if models.get("sell"):
        try:
            p = models["sell"].predict_proba(X)[0]
            sell_p = float(p[1]) if 1 in models["sell"].classes_ else 0.0
        except Exception: pass
    log.info("BUY p=%.3f | SELL p=%.3f (min=%.2f)", buy_p, sell_p, CFG.min_prob)

    side, prob = None, 0.0
    if buy_p >= CFG.min_prob and buy_p >= sell_p: side, prob = "BUY", buy_p
    elif sell_p >= CFG.min_prob: side, prob = "SELL", sell_p
    if not side: return None

    mid = float(last["close"]); atr = float(last["atr"])
    if not np.isfinite(atr) or atr <= 0: return None
    if CFG.spread_usd > CFG.max_spread_usd: return None

    entry = mid + CFG.spread_usd/2 if side == "BUY" else mid - CFG.spread_usd/2
    if side == "BUY":
        sl = entry - atr * CFG.atr_mult_sl
        tp = entry + atr * CFG.atr_mult_sl * CFG.rr
    else:
        sl = entry + atr * CFG.atr_mult_sl
        tp = entry - atr * CFG.atr_mult_sl * CFG.rr
    sl_dist = abs(entry - sl)
    lot = compute_lot(store, sl_dist)
    if lot is None: return None

    return {"created_at": iso(now_utc()), "symbol": CFG.symbol, "side": side,
            "entry": entry, "sl": sl, "tp": tp, "lot": lot,
            "risk_usd": lot * sl_dist * CFG.contract_size,
            "prob": prob, "source": "paper"}


# ============================================================
# TASKS
# ============================================================
def task_tick(store):
    log.info("=== tick ===")
    store.daily_reset_if_needed()

    # data
    try:
        raw = fetch_candles()
        df = sanitize(raw)
        if df is None: log.error("bad data"); return
        age = (now_utc() - df["datetime"].iloc[-1]).total_seconds() / 60
        log.info("data: %d bars, last %.1fm ago", len(df), age)
        if age > 45:
            store.cb_trip("data_stale", f"age={age:.0f}m", 30); return
    except Exception as e:
        log.error("fetch: %s", e); return

    # exits first
    try: paper_check_exits(store, df)
    except Exception as e: log.error("exits: %s", e)

    # gates
    ok, why = check_cb(store)
    if not ok: log.info("CB: %s", why); store.heartbeat(); return
    if not is_valid_session(): log.info("off-session"); store.heartbeat(); return
    if is_news_window(): log.info("news"); store.heartbeat(); return
    lt = store.last_signal_time()
    if lt and (now_utc() - lt).total_seconds()/60 < CFG.cooldown_min:
        log.info("cooldown"); store.heartbeat(); return
    if store.daily_signals() >= CFG.daily_max_signals:
        log.info("daily limit"); store.heartbeat(); return

    # auto-train if no model
    models = load_active_models(store)
    if not models:
        log.info("no model — auto-training")
        try:
            ok = _train_core(store, df)
            if not ok:
                log.error("auto-train failed"); store.heartbeat(); return
        except Exception as e:
            log.error("train: %s", e); return
        models = load_active_models(store)
        if not models: return

    # features
    try: df_feat = calculate_features(df)
    except Exception as e: log.error("feat: %s", e); return
    if len(df_feat) < 400: return

    rok, rwhy = regime_ok(df_feat.iloc[-1])
    if not rok: log.info("regime: %s", rwhy); store.heartbeat(); return

    try: sig = generate_signal(store, df_feat, models)
    except Exception as e: log.error("sig: %s", e); return
    if not sig:
        log.info("no signal"); store.heartbeat(); store.snapshot_equity(); return

    try:
        sid, fill = paper_execute(store, sig, float(df_feat.iloc[-1]["close"]))
        store.inc_daily_signals()
        store.set_last_signal_time(now_utc())
        store.audit("submitted", {"signal_id": sid, "side": sig["side"],
                                   "fill": fill, "lot": sig["lot"]})
        send_telegram(
            f"{'🟢' if sig['side']=='BUY' else '🔴'} <b>XAUUSD {sig['side']}</b>\n"
            f"entry={fill:.2f} sl={sig['sl']:.2f} tp={sig['tp']:.2f}\n"
            f"lot={sig['lot']:.2f} risk=${sig['risk_usd']:.2f} "
            f"prob={sig['prob']*100:.1f}%")
        log.info("SUBMITTED %s lot=%.2f @ %.2f", sig["side"], sig["lot"], fill)
    except Exception as e:
        log.error("exec: %s", e)

    store.heartbeat(); store.snapshot_equity()


def task_retrain(store):
    log.info("=== retrain ===")
    try:
        raw = fetch_candles(); df = sanitize(raw)
        if df is None: raise RuntimeError("bad data")
    except Exception as e:
        log.error("fetch: %s", e); send_telegram(f"❌ Retrain: data ({e})"); return
    try:
        r = _train_core(store, df)
    except Exception as e:
        log.error("train: %s", e); send_telegram(f"❌ Retrain: {e}"); return
    if r is False or r is None:
        log.error("gate FAILED both")
        store.cb_trip("model_degraded", "walk-forward fail", 12*60)
        send_telegram("⚠️ Retrain: <b>GATE FAILED</b>. Live diblokir 12 jam.")
        return
    _, version, m = r
    store.cb_reset("model_degraded")
    log.info("model %s", version)
    send_telegram(
        f"✅ Retrain v{version}\n"
        f"buy:  auc={m.get('buy',{}).get('mean_auc',0):.3f} "
        f"prec={m.get('buy',{}).get('mean_prec',0):.3f} "
        f"gate={m.get('buy',{}).get('gate_ok')}\n"
        f"sell: auc={m.get('sell',{}).get('mean_auc',0):.3f} "
        f"prec={m.get('sell',{}).get('mean_prec',0):.3f} "
        f"gate={m.get('sell',{}).get('gate_ok')}")


def task_health(store):
    log.info("=== health ===")
    alerts = []
    hb = store.get_meta("last_heartbeat")
    if hb:
        age = (now_utc() - datetime.fromisoformat(hb)).total_seconds() / 60
        if age > 90: alerts.append(f"💀 heartbeat {age:.0f}m stale")
    else:
        alerts.append("⚠️ no heartbeat yet")
    for row in store._conn.execute(
            "SELECT name,reason FROM cb_state WHERE tripped=1").fetchall():
        alerts.append(f"🔴 CB: {row[0]} — {row[1]}")
    pnl = store.daily_pnl()
    if pnl < -50: alerts.append(f"📉 daily ${pnl:.2f}")
    n_pos = len(store.open_positions())
    if alerts:
        send_telegram("🔍 <b>Health</b>\n" + "\n".join(alerts))
    else:
        send_telegram(f"💓 OK | eq=${store.current_equity():.2f} "
                      f"pos={n_pos} pnl=${pnl:+.2f}", silent=True)


def task_report(store):
    log.info("=== report ===")
    df = pd.read_sql(
        "SELECT * FROM signals WHERE status='CLOSED' "
        "AND closed_at >= ?",
        store._conn, params=((now_utc() - timedelta(days=7)).isoformat(),))
    if df.empty:
        send_telegram("📊 <b>Weekly</b>\nNo closed trades this week")
        return
    n = len(df)
    wins = (df["outcome"] == "WIN").sum()
    wr = wins / n if n else 0
    pnl = df["pnl"].sum()
    gross_loss = abs(df.loc[df["outcome"] == "LOSS", "pnl"].sum())
    pf = df.loc[df["outcome"] == "WIN", "pnl"].sum() / gross_loss if gross_loss else 0
    avg_win = df.loc[df["outcome"] == "WIN", "pnl"].mean() if wins else 0
    avg_loss = df.loc[df["outcome"] == "LOSS", "pnl"].mean() if (n-wins) else 0
    eq = store.current_equity()
    verdict = "✅ PASS" if (wr >= 0.4 and pf >= 1.5 and pnl >= 0) else \
              "⚠️ MARGINAL" if (wr >= 0.33 and pf >= 1.1) else "❌ FAIL"
    send_telegram(
        f"📊 <b>Weekly Report</b>\n"
        f"{verdict}\n\n"
        f"trades={n}  wr={wr*100:.1f}%\n"
        f"pnl=${pnl:+.2f}  pf={pf:.2f}\n"
        f"avg win=${avg_win:+.2f}  avg loss=${avg_loss:+.2f}\n"
        f"equity=${eq:.2f}")


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="tick",
                    choices=["tick", "retrain", "health", "report"])
    args = ap.parse_args()

    store = StateStore(CFG.db_path)
    try:
        resolve_chat(store)
        if args.task == "tick": task_tick(store)
        elif args.task == "retrain": task_retrain(store)
        elif args.task == "health": task_health(store)
        elif args.task == "report": task_report(store)
    finally:
        store.wal_checkpoint(); store.close()


if __name__ == "__main__":
    try: main()
    except Exception as e:
        log.critical("FATAL: %s", e, exc_info=True)
        send_telegram(f"🛑 CRASH: {str(e)[:200]}")
        sys.exit(1)
