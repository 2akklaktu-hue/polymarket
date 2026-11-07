"""Пересчёт калибровки по годам и обновление models/calibration.json.

Зачем отдельный скрипт: главный вывод исследования — коэффициент перекоса
НЕ постоянен. По рынкам 2024 года b = 1.205, по 2025 — уже 1.042. Подогнать
модель один раз и пользоваться ей нельзя, её надо пересчитывать, и делать
это должно быть просто.

Якорь по времени — плановая дата окончания рынка, а не последняя сделка.
Причина в 03_calibration.py: у рынков, решившихся досрочно, цена за день до
последней сделки — это уже посчитанный результат, и калибровка выходит
фиктивно идеальной.

Файл рынков большой (сотни тысяч строк), поэтому он читается построчно
с немедленным отсевом, а не грузится в DataFrame целиком.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pmlib.dataset import load, clean_for_calibration  # noqa: E402
from importlib import import_module  # noqa: E402

cal = import_module("03_calibration")

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data/out"
HORIZON_DAYS = 7
MIN_ROWS_PER_YEAR = 400
N_BOOT = 300


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit(x: np.ndarray, y: np.ndarray, start=(0.0, 1.0)) -> tuple[float, float]:
    def nll(th):
        z = th[0] + th[1] * x
        return -np.sum(y * np.log(sig(z) + 1e-12) + (1 - y) * np.log(1 - sig(z) + 1e-12))
    return tuple(minimize(nll, list(start), method="Nelder-Mead").x)


def fit_with_ci(d: pd.DataFrame, col: str, rng) -> dict:
    x = logit(d[col].to_numpy())
    y = d["y"].to_numpy(float)
    ev = d["eventId"].fillna(d["conditionId"]).to_numpy()
    a, b = fit(x, y)

    # бутстрэп по СОБЫТИЯМ: рынки внутри события не независимы
    keys = pd.unique(ev)
    idx = {k: np.where(ev == k)[0] for k in keys}
    bs = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[keys[i]] for i in rng.choice(len(keys), len(keys), True)])
        bs.append(fit(x[pick], y[pick], (a, b))[1])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return {"a": float(a), "b": float(b), "ci": [float(lo), float(hi)],
            "n": int(len(d)), "events": int(d["eventId"].nunique())}


def main() -> int:
    mk = ROOT / "data/raw/markets_closed.jsonl"
    sp = ROOT / "data/raw/price_series.jsonl"
    min_vol = float(sys.argv[1]) if len(sys.argv) > 1 else 5000.0

    print("читаю ряды цен...", flush=True)
    ser = cal.load_series(sp)
    print(f"  рядов: {len(ser)}", flush=True)

    print("читаю рынки...", flush=True)
    df = clean_for_calibration(load(mk), min_volume=min_vol)
    df = df[df["conditionId"].isin(ser.keys())]
    print(f"  рынков с историей и прошедших отбор: {len(df)}", flush=True)

    recs = []
    for r in df.itertuples():
        s = ser.get(r.conditionId)
        if not s or len(s["series"]) < 3 or pd.isna(r.endDate):
            continue
        tgt = r.endDate.timestamp() - HORIZON_DAYS * 86400
        if s["t_first"] > tgt:          # ряд не покрывает момент наблюдения
            continue
        p = cal.price_at(s["series"], tgt)
        if not np.isfinite(p) or not (0.0 < p < 1.0):
            continue
        recs.append({"conditionId": r.conditionId,
                     "eventId": getattr(r, "eventId", None),
                     "year": r.endDate.year, "p": p, "y": r.resolved_yes})
    q = pd.DataFrame(recs)
    print(f"наблюдений на горизонте {HORIZON_DAYS} дн.: {len(q)}\n", flush=True)
    if q.empty:
        return 1

    OUTDIR.mkdir(parents=True, exist_ok=True)
    q.to_parquet(OUTDIR / "panel_refit.parquet", index=False)

    rng = np.random.default_rng(17)
    print(f"{'год':>6} {'n':>7} {'событий':>8} {'a':>8} {'b':>8} {'95% интервал b':>20}")
    by_year = {}
    for y, g in q.groupby("year"):
        if len(g) < MIN_ROWS_PER_YEAR:
            continue
        f = fit_with_ci(g, "p", rng)
        by_year[int(y)] = f
        flag = "" if (f["ci"][0] > 1.0 or f["ci"][1] < 1.0) else "   <- единица в интервале"
        print(f"{int(y):>6} {f['n']:>7} {f['events']:>8} {f['a']:>8.4f} {f['b']:>8.4f} "
              f"[{f['ci'][0]:>7.4f},{f['ci'][1]:>7.4f}]{flag}")

    if not by_year:
        print("\nни одного года с достаточной выборкой")
        return 1

    newest = max(by_year)
    cur = by_year[newest]
    print(f"\nсамый свежий год с достаточной выборкой: {newest}")
    print(f"  a = {cur['a']:.4f}, b = {cur['b']:.4f}, интервал {cur['ci']}")

    path = OUTDIR.parent.parent / "models/calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    import datetime as dt
    stable = cur["ci"][0] > 1.0 or cur["ci"][1] < 1.0
    path.write_text(json.dumps({
        "model": "logit(p_true) = a + b * logit(p_market)",
        "a": round(cur["a"], 4), "b": round(cur["b"], 4),
        "horizon_days": HORIZON_DAYS,
        "fitted_on": dt.date.today().isoformat(),
        "fitted_window": f"рынки, закрывшиеся в {newest} году",
        "n_markets": cur["n"], "n_events": cur["events"],
        "b_ci95": [round(v, 4) for v in cur["ci"]],
        "significant": bool(stable),
        "note": ("Коэффициент нестабилен во времени, поэтому берётся оценка по "
                 "самому свежему году, а не объединённая по всем. "
                 + ("Интервал единицу НЕ включает: перекос значим."
                    if stable else
                    "Интервал ВКЛЮЧАЕТ единицу: перекос статистически "
                    "неотличим от нуля, калибровочная поправка сигнала "
                    "практически не даёт.")),
        "by_year": {str(k): {"a": round(v["a"], 4), "b": round(v["b"], 4),
                             "ci95": [round(c, 4) for c in v["ci"]],
                             "n": v["n"], "events": v["events"]}
                    for k, v in sorted(by_year.items())},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nмодель записана: {path}")

    print(f"\nчто это даёт на практике (год {newest}):")
    print(f"{'цена':>7} {'скорр. вероятн.':>16} {'разница':>10}")
    for p in (0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97):
        qq = sig(cur["a"] + cur["b"] * logit(p))
        print(f"{p:>7.2f} {qq:>16.4f} {qq - p:>+10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
