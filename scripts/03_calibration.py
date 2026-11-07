"""Калибровка Polymarket: насколько цена соответствует реальной частоте исхода.

Главный вопрос: если рынок оценивает событие в p центов, как часто оно
на самом деле происходит? Если систематически реже — лонгшоты переоценены
(классический favorite-longshot bias), и покупать их в среднем убыточно.
Если чаще — есть эксплуатируемый эдж.

Два методических момента, без которых цифры врут:

1. ЯКОРЬ ПО ВРЕМЕНИ. Цена берётся за N дней до ПОСЛЕДНЕЙ СДЕЛКИ, а не до
   плановой даты окончания. Многие рынки решаются досрочно, и цена за день
   до endDate у них — уже посчитанный результат (0 или 1), что даёт фиктивно
   идеальную калибровку.

2. КЛАСТЕРИЗАЦИЯ. Рынки внутри одного события ("кто станет номинантом" —
   20 кандидатов) не независимы: ровно один из них выстрелит. Наивный
   доверительный интервал на таких данных слишком узкий, поэтому основной
   интервал считается бутстрэпом по СОБЫТИЯМ, а не по рынкам.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmlib.dataset import load, clean_for_calibration  # noqa: E402
from pmlib.econ import FEE_RATES, DEFAULT_FEE_RATE, effective_price, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data/out"

HORIZONS = [1, 3, 7, 30]          # дни до последней сделки
# Границы корзин: мелко на дешёвом конце — именно он нас интересует.
BINS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30,
        0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]
N_BOOT = 2000
RNG = np.random.default_rng(12345)


def load_series(path: Path) -> dict[str, dict]:
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        out[r["conditionId"]] = r
    return out


def price_at(series: list[list], target_ts: float) -> float:
    """Последняя известная цена на момент target_ts (ряд отсортирован по времени)."""
    lo, hi = 0, len(series) - 1
    if not series or series[0][0] > target_ts:
        return np.nan
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if series[mid][0] <= target_ts:
            lo = mid
        else:
            hi = mid - 1
    return float(series[lo][1])


def build_panel(df: pd.DataFrame, ser: dict[str, dict]) -> pd.DataFrame:
    """Одна строка = рынок; колонки p_1d, p_3d, ... + исход."""
    recs = []
    for r in df.itertuples():
        s = ser.get(r.conditionId)
        if not s or len(s["series"]) < 3:
            continue
        t_last, t_first = s["t_last"], s["t_first"]
        row = {
            "conditionId": r.conditionId,
            "eventId": getattr(r, "eventId", None),
            "question": r.question,
            "category": r.category,
            "volume": r.volumeNum,
            "y": r.resolved_yes,
            "endDate": r.endDate,
            "n_trades_seen": s["n_trades_seen"],
            "span_days": (t_last - t_first) / 86400.0,
            "p_final": float(s["series"][-1][1]),
        }
        for h in HORIZONS:
            tgt = t_last - h * 86400
            # горизонт доступен, только если ряд реально покрывает этот момент
            row[f"p_{h}d"] = price_at(s["series"], tgt) if t_first <= tgt else np.nan
        recs.append(row)
    return pd.DataFrame(recs)


def boot_ci_by_event(y: np.ndarray, ev: np.ndarray, n_boot: int = N_BOOT,
                     alpha: float = 0.05) -> tuple[float, float]:
    """Бутстрэп доли с ресемплингом по событиям (кластерам)."""
    uniq = pd.unique(ev)
    if len(uniq) < 3:
        return (np.nan, np.nan)
    groups = {u: y[ev == u] for u in uniq}
    keys = list(groups.keys())
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = RNG.choice(len(keys), size=len(keys), replace=True)
        vals = np.concatenate([groups[keys[i]] for i in pick])
        means[b] = vals.mean()
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def calibration_table(panel: pd.DataFrame, col: str, fee_rate: float | None = None
                      ) -> pd.DataFrame:
    d = panel[panel[col].notna() & panel["y"].notna()].copy()
    d["bin"] = pd.cut(d[col], BINS, include_lowest=True)
    rows = []
    for b, g in d.groupby("bin", observed=True):
        n = len(g)
        if n == 0:
            continue
        k = int(g["y"].sum())
        freq = k / n
        p_mean = float(g[col].mean())
        rate = fee_rate if fee_rate is not None else DEFAULT_FEE_RATE
        p_eff = effective_price(min(max(p_mean, 1e-6), 1 - 1e-6), rate)
        lo_w, hi_w = wilson(k, n)
        ev = g["eventId"].fillna(g["conditionId"]).to_numpy()
        lo_b, hi_b = boot_ci_by_event(g["y"].to_numpy(float), ev)
        rows.append({
            "корзина": str(b),
            "n": n,
            "цена_средн": round(p_mean, 4),
            "факт_частота": round(freq, 4),
            "разница": round(freq - p_mean, 4),
            "ci_lo_наивн": round(lo_w, 4),
            "ci_hi_наивн": round(hi_w, 4),
            "ci_lo_событ": round(lo_b, 4) if lo_b == lo_b else None,
            "ci_hi_событ": round(hi_b, 4) if hi_b == hi_b else None,
            "безубыток": round(p_eff, 4),
            # доходность стратегии «покупать YES во всех рынках этой корзины»
            "ROI_покуп_YES": round(freq / p_eff - 1.0, 4),
            # и зеркальная: покупать NO по цене 1-p
            "ROI_покуп_NO": round((1 - freq) / effective_price(
                min(max(1 - p_mean, 1e-6), 1 - 1e-6), rate) - 1.0, 4),
        })
    return pd.DataFrame(rows)


def main() -> int:
    mk = ROOT / "data/raw/markets_closed.jsonl"
    sp = ROOT / "data/raw/price_series.jsonl"
    if not sp.exists():
        print("нет data/raw/price_series.jsonl — сначала запусти 02_fetch_trade_prices.py")
        return 1

    min_vol = float(sys.argv[1]) if len(sys.argv) > 1 else 5000.0
    df = clean_for_calibration(load(mk), min_volume=min_vol)
    ser = load_series(sp)
    print(f"рынков: {len(df)} | рядов цен: {len(ser)}")

    panel = build_panel(df, ser)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTDIR / "panel.parquet", index=False)
    print(f"панель: {len(panel)} рынков, {panel['eventId'].nunique()} событий\n")

    for h in HORIZONS:
        col = f"p_{h}d"
        t = calibration_table(panel, col)
        if t.empty:
            continue
        t.to_csv(OUTDIR / f"calibration_{h}d.csv", index=False, encoding="utf-8-sig")
        n = int(t["n"].sum())
        print(f"=== ЦЕНА ЗА {h} ДН. ДО ПОСЛЕДНЕЙ СДЕЛКИ (n={n}) ===")
        print(t.to_string(index=False))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
