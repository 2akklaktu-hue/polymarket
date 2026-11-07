"""Где спрятана наценка: структура цен во взаимоисключающих событиях.

Идея теста. Во взаимоисключающем событии («кто победит») ровно один исход
выигрывает, значит истинные вероятности обязаны давать в сумме 1. Если сложить
лучшие ask по всем исходам, получится больше 1 — это наценка (overround),
плата за вход. Вопрос: наценка размазана ровно или сидит в дешёвых исходах?

Оценка «справедливой» вероятности берётся нормировкой середин стакана:
    fair_i = mid_i / sum(mid)
Затем для каждого исхода считается, во сколько раз ask выше справедливой цены.
Если у дешёвых исходов это отношение систематически больше — покупка лонгшотов
по рынку заведомо убыточна ещё до всякого прогноза.

Тест самодостаточный: не требует истории и известных исходов, опирается только
на арифметику взаимоисключающих событий.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmlib.api import Api, ApiError, GAMMA  # noqa: E402
from pmlib.econ import effective_price, FEE_RATES, DEFAULT_FEE_RATE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data/out"
MIN_LIQ = 5000.0


def fetch_events(api: Api, min_liq: float = MIN_LIQ) -> list[dict]:
    evs, off = [], 0
    while True:
        try:
            b = api.get(f"{GAMMA}/events",
                        {"limit": 100, "offset": off, "closed": "false",
                         "liquidity_min": min_liq, "order": "liquidity",
                         "ascending": "false"})
        except ApiError:
            break                      # offset упёрся в потолок пагинации
        if not b:
            break
        evs += b
        if len(b) < 100:
            break
        off += 100
    return evs


def build_rows(evs: list[dict]) -> pd.DataFrame:
    rows = []
    for e in evs:
        if not (e.get("negRisk") or e.get("enableNegRisk")):
            continue
        mk = [m for m in (e.get("markets") or [])
              if not m.get("closed") and m.get("acceptingOrders")
              and m.get("bestAsk") is not None and m.get("bestBid") is not None]
        if len(mk) < 3:
            continue
        try:
            ask = np.array([float(m["bestAsk"]) for m in mk])
            bid = np.array([float(m["bestBid"]) for m in mk])
        except (TypeError, ValueError):
            continue
        mid = (ask + bid) / 2.0
        s_mid = mid.sum()
        # Набор исходов должен быть полным: если сумма середин далека от 1,
        # это либо неполный список вариантов, либо мёртвые котировки.
        # Такие события для теста непригодны — на них сумма ask < 1 не арбитраж.
        if not (0.90 <= s_mid <= 1.15):
            continue
        fair = mid / s_mid
        tag = (e.get("tags") or [{}])
        cat = (tag[1].get("label") if len(tag) > 1 else tag[0].get("label")) or "?"
        for m, a, b_, f in zip(mk, ask, bid, fair):
            rate = FEE_RATES.get(str(m.get("feeType") or "").replace("_fees", ""),
                                 DEFAULT_FEE_RATE)
            rows.append({
                "event": e.get("title", "")[:70],
                "outcome": m.get("groupItemTitle") or m.get("question", "")[:40],
                "cat": cat,
                "n_out": len(mk),
                "ask": a, "bid": b_, "fair": f,
                "ask_eff": effective_price(min(max(a, 1e-6), 1 - 1e-6), rate),
                "sum_ask": float(ask.sum()), "sum_mid": float(s_mid),
                "liq": float(e.get("liquidity") or 0),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # во сколько раз реальная цена входа выше справедливой оценки
    df["markup"] = df["ask_eff"] / df["fair"].clip(lower=1e-6)
    # и та же величина в единицах вероятности
    df["markup_abs"] = df["ask_eff"] - df["fair"]
    return df


BINS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0]


def main() -> int:
    api = Api(min_interval=0.05)
    print("загружаю активные события...", flush=True)
    evs = fetch_events(api)
    print(f"событий: {len(evs)} ({api.n_calls} запросов)")

    df = build_rows(evs)
    if df.empty:
        print("нет пригодных событий")
        return 1
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTDIR / "overround.parquet", index=False)

    n_ev = df["event"].nunique()
    print(f"пригодных событий (сумма середин близка к 1): {n_ev}, исходов: {len(df)}")
    print(f"медианная сумма ask по событию: {df.groupby('event')['sum_ask'].first().median():.4f}\n")

    df["bin"] = pd.cut(df["fair"], BINS, include_lowest=True)
    g = df.groupby("bin", observed=True).agg(
        n=("ask", "size"),
        справедливая=("fair", "mean"),
        цена_входа=("ask_eff", "mean"),
        наценка_разы=("markup", "median"),
        наценка_пп=("markup_abs", "median"),
    ).round(4)
    # сколько процентов ожидаемой доходности съедает наценка
    g["потеря_ROI"] = (1.0 - 1.0 / g["наценка_разы"]).round(3)
    print("=== НАЦЕНКА ПО УРОВНЮ СПРАВЕДЛИВОЙ ВЕРОЯТНОСТИ ===")
    print("(справедливая = нормированная середина стакана; цена входа = ask + комиссия)")
    print(g.to_string())
    g.to_csv(OUTDIR / "overround_by_price.csv", encoding="utf-8-sig")

    print("\n=== ТО ЖЕ ПО КАТЕГОРИЯМ (медиана наценки, разы) ===")
    pv = df.pivot_table(index="cat", columns="bin", values="markup",
                        aggfunc="median", observed=True)
    keep = [c for c in pv.columns if df[df["bin"] == c].shape[0] > 50]
    cnt = df.groupby("cat", observed=True).size().sort_values(ascending=False)
    print(pv.loc[cnt.head(8).index, keep].round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
