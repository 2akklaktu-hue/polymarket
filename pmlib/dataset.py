"""Загрузка выгрузки рынков в DataFrame с нормальными типами и признаками."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# feeType вида "politics_fees" — самый надёжный признак категории,
# он проставлен самой площадкой, а не угадан по тексту вопроса.
_FEE_CAT = re.compile(r"^(.*?)_fees$")

# Повторяющиеся высокочастотные рынки: 5-минутные и часовые «вверх/вниз»,
# внутридневные ценовые коридоры. Для калибровки это шум: тысячи почти
# одинаковых исходов из одного распределения ломают независимость наблюдений.
_HIGHFREQ = re.compile(
    r"(?:up or down)"
    r"|(?:\b\d{1,2}:\d{2}\s*(?:am|pm)\b)"
    r"|(?:\b\d{1,2}(?:am|pm)\s*-\s*\d{1,2}(?:am|pm)\b)"
    r"|(?:\bhourly\b)"
    r"|(?:\bevery \d+ minutes?\b)",
    re.I)

# Запасная классификация по тексту вопроса: feeType появился у Polymarket
# недавно, у рынков до 2026 года его просто нет. Грубо, но для нарезки хватает.
_KW_RAW = [
    ("crypto", r"\b(?:bitcoin|btc|ethereum|eth|solana|xrp|dogecoin|crypto|"
               r"coinbase|binance|altcoin|stablecoin)\b"),
    ("sports", r"\b(?:nba|nfl|mlb|nhl|ufc|premier league|la liga|"
               r"champions league|world cup|super bowl|olympics|f1|grand prix|"
               r"win the series|beat the)\b"),
    ("politics", r"\b(?:president|election|senate|congress|governor|primary|"
                 r"nominee|impeach|cabinet|parliament|prime minister|vote|"
                 r"ballot|democrat|republican)\b"),
    ("geopolitics", r"\b(?:ceasefire|invasion|war|nato|sanctions|peace deal|"
                    r"nuclear|missile|hostage|treaty|troops)\b"),
    ("economics", r"\b(?:fed|interest rate|inflation|cpi|gdp|recession|"
                  r"unemployment|jobs report|fomc|tariff)\b"),
    ("finance", r"\b(?:s&p|nasdaq|dow|stock|ipo|earnings|merger|acquisition|"
                r"bankrupt)\b"),
    ("tech", r"\b(?:openai|gpt|llm|ai model|apple|google|tesla|spacex|launch|"
             r"nvidia|chatgpt|gemini|claude)\b"),
    ("culture", r"\b(?:oscar|grammy|emmy|box office|album|movie|netflix|"
                r"celebrity|engaged|married)\b"),
    ("weather", r"\b(?:hurricane|temperature|snow|rainfall|storm|tornado)\b"),
]
_KW = [(cat, re.compile(pat, re.I)) for cat, pat in _KW_RAW]


def guess_category(q: str) -> str:
    for cat, rx in _KW:
        if rx.search(q):
            return cat
    return "other"


def _f(x, default=np.nan) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _jsl(x, default=None):
    """Поля outcomes/outcomePrices/clobTokenIds приходят строкой с JSON внутри."""
    if isinstance(x, list):
        return x
    if not x:
        return default or []
    try:
        v = json.loads(x)
        return v if isinstance(v, list) else (default or [])
    except (TypeError, ValueError, json.JSONDecodeError):
        return default or []


def load(path: str | Path) -> pd.DataFrame:
    """Читает jsonl. Последняя строка может быть обрезана, если файл ещё
    пишется параллельно — такие строки пропускаются."""
    path = Path(path)
    rows, bad = [], 0
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"  (пропущено битых строк: {bad})")
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in ("startDate", "endDate", "closedTime", "createdAt"):
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True, format="mixed")

    for c in ("volumeNum", "liquidity", "spread", "bestBid", "bestAsk",
              "lastTradePrice", "orderMinSize", "orderPriceMinTickSize",
              "oneDayPriceChange", "oneWeekPriceChange", "umaBond"):
        if c in df:
            df[c] = df[c].map(_f)

    df["outcomes_l"] = df["outcomes"].map(_jsl)
    df["prices_l"] = df["outcomePrices"].map(_jsl)
    df["tokens_l"] = df["clobTokenIds"].map(_jsl)

    # исход: цена первого ("Yes") исхода после резолва — строго 0 или 1
    def _res(pl):
        if len(pl) < 2:
            return np.nan
        v = _f(pl[0])
        return v if v in (0.0, 1.0) else np.nan

    df["resolved_yes"] = df["prices_l"].map(_res)

    # категория: официальный feeType, где он есть; иначе — эвристика по тексту
    df["category_fee"] = df["feeType"].fillna("").map(
        lambda s: (m.group(1) if (m := _FEE_CAT.match(str(s))) else None))
    df["category_is_official"] = df["category_fee"].notna()
    df["category"] = np.where(
        df["category_is_official"],
        df["category_fee"],
        df["question"].fillna("").map(guess_category))

    df["fee_rate"] = df["feeSchedule"].map(
        lambda d: _f(d.get("rate")) if isinstance(d, dict) else np.nan)

    df["q"] = df["question"].fillna("")
    df["is_highfreq"] = df["q"].str.contains(_HIGHFREQ)
    df["duration_days"] = (df["endDate"] - df["startDate"]).dt.total_seconds() / 86400
    df["n_outcomes"] = df["outcomes_l"].map(len)

    return df


def clean_for_calibration(df: pd.DataFrame,
                          min_volume: float = 1000.0,
                          min_duration_days: float = 1.0) -> pd.DataFrame:
    """Отсев мусора перед изучением калибровки.

    Что выбрасываем и почему:
      * нерезолвленные / кривые исходы — нет «правды», сравнивать не с чем;
      * высокочастотные повторяющиеся рынки — нарушают независимость;
      * рынки без объёма — цена там не отражает ничьё мнение;
      * сверхкороткие рынки — там цена это исполнение, а не прогноз;
      * не бинарные — отдельная математика.
    """
    m = (
        df["resolved_yes"].notna()
        & (~df["is_highfreq"])
        & (df["volumeNum"] >= min_volume)
        & (df["duration_days"] >= min_duration_days)
        & (df["n_outcomes"] == 2)
        & df["endDate"].notna()
    )
    return df.loc[m].copy()
