"""Скачки цены на слухах: сработал ли памп или откатился.

Гипотеза пользователя: цена стоит 5c, появляются слухи и объём, цена уезжает
к 50c, а по факту событие не наступает и выигрывает NO.

Что считается. По каждому закрытому рынку ищется ПЕРВЫЙ скачок (цена была
<= T_lo, затем в пределах окна W часов дошла до >= T_hi) и берётся фактический
исход. Главная величина — ЭДЖ: насколько фактическая частота побед NO
отличается от того, что обещает сама цена входа. Эффективный рынок даёт ноль.
Дополнительно считается контрольная группа — те же уровни цены БЕЗ скачка,
иначе нельзя отличить «памп переоценён» от «так ведут себя все рынки на этом
уровне цены».

ВАЖНО, про подглядывание в будущее. Буфер до конца считается от ПЛАНОВОЙ даты
окончания (endDate), а не от фактической резолюции (closedTime). Момент
фактической резолюции зависит от исхода: когда слух подтверждается, рынок
закрывают почти сразу после движения. Фильтр по closedTime выбрасывал ровно
эти случаи — 44% выборки, в которых YES побеждал в 69% — и создавал
фальшивый эдж в сотни процентов. Плановая дата видна в момент входа, поэтому
отбор по ней честен и воспроизводим в реальном времени.

Ограничение данных: в сжатых рядах лежит только VWAP цены по корзинам времени,
объёма в них нет, поэтому условие «подскочил объём» здесь не проверяется —
только движение цены. Шаг между точками в среднем 4.7 часа.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmlib.econ import DEFAULT_FEE_RATE, taker_fee_per_share, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "data/raw/price_series.jsonl"
OUTCOMES = ROOT / "data/out/outcomes.parquet"
OUT = ROOT / "data/out/shocks.parquet"

BUFFER_H = 24.0          # часов до ПЛАНОВОГО конца должно остаться
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
RNG = np.random.default_rng(7)

# Наценка входа по уровню цены, из scripts/04_overround.py
# (data/out/overround_by_price.csv). Вход всегда хуже середины стакана,
# а на дешёвых исходах сильно хуже.
_OVERROUND = [(0.005, 1.54), (0.01, 1.71), (0.02, 1.33), (0.03, 1.22),
              (0.05, 1.18), (0.10, 1.17), (0.20, 1.13), (0.35, 1.07),
              (0.50, 1.05), (0.70, 1.04), (1.01, 1.02)]


def entry_price(mid: float) -> float:
    """Реальная цена покупки доли при середине стакана mid."""
    for hi, mult in _OVERROUND:
        if mid <= hi:
            return min(mid * mult, 0.995)
    return min(mid * 1.02, 0.995)


def roi_buy(price_mid: float, won: bool,
            fee_rate: float = DEFAULT_FEE_RATE) -> float:
    """Доходность покупки доли по рынку, на 1 вложенный доллар."""
    c = entry_price(price_mid)
    cost = c + taker_fee_per_share(c, fee_rate)
    return ((1.0 if won else 0.0) - cost) / cost


def find_shock(s: list, t_lo: float, t_hi: float, w_h: float,
               plan_ts: float) -> dict | None:
    """Первый скачок: цена была <= t_lo, за w_h часов дошла до >= t_hi.

    plan_ts — плановая дата окончания рынка. Если на момент скачка до неё
    осталось меньше BUFFER_H часов, сигнала нет: в реальном времени такую
    ставку не делают, слишком мало времени до резолюции.
    """
    w = w_h * 3600.0
    for j in range(1, len(s)):
        tj, pj = s[j]
        if pj < t_hi:
            continue
        lo_p, lo_t = 1.0, None
        for i in range(j - 1, -1, -1):
            ti, pi = s[i]
            if tj - ti > w:
                break
            if pi < lo_p:
                lo_p, lo_t = pi, ti
        if lo_t is None or lo_p > t_lo:
            continue
        if (plan_ts - tj) < BUFFER_H * 3600.0:
            return None
        return {"t": tj, "p_entry": pj, "p_before": lo_p,
                "rise_h": (tj - lo_t) / 3600.0,
                "days_left": (plan_ts - tj) / 86400.0}
    return None


def find_calm(s: list, t_hi: float, w_h: float, plan_ts: float,
              max_rise: float = 0.05) -> dict | None:
    """Контроль: цена того же уровня, но БЕЗ скачка за те же w_h часов."""
    w = w_h * 3600.0
    band_hi = t_hi + 0.15
    for j in range(1, len(s)):
        tj, pj = s[j]
        if not (t_hi <= pj <= band_hi):
            continue
        lo_p = 1.0
        seen = False
        for i in range(j - 1, -1, -1):
            ti, pi = s[i]
            if tj - ti > w:
                break
            seen = True
            lo_p = min(lo_p, pi)
        if not seen or (pj - lo_p) >= max_rise:
            continue
        if (plan_ts - tj) < BUFFER_H * 3600.0:
            continue
        return {"t": tj, "p_entry": pj, "days_left": (plan_ts - tj) / 86400.0}
    return None


def boot_ci(vals: np.ndarray, groups: np.ndarray, n: int = 600) -> tuple:
    """Бутстрэп среднего с ресемплом по СОБЫТИЯМ, а не по рынкам.

    Рынки одного события (negRisk-набор «кого назначат») решаются одной
    новостью, поэтому считать их независимыми наблюдениями нельзя.
    """
    uniq = pd.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    out = np.empty(n)
    for k in range(n):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        out[k] = vals[sel].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def stats(d: pd.DataFrame) -> dict:
    """Эдж и доходность по набору событий."""
    no_won = (d["resolved_yes"] == 0).to_numpy()
    n = len(d)
    k = int(no_won.sum())
    wl, wh = wilson(k, n)
    promised = (1.0 - d["p_entry"]).to_numpy()
    edge = no_won.astype(float) - promised            # факт минус обещание цены
    g = d["eventId"].to_numpy()
    e_lo, e_hi = boot_ci(edge, g)
    roi_no = np.array([roi_buy(1.0 - p, bool(w))
                       for p, w in zip(d["p_entry"], no_won)])
    r_lo, r_hi = boot_ci(roi_no, g)
    return {"n": n, "events": d["eventId"].nunique(),
            "entry": float(d["p_entry"].mean()),
            "no_fact": k / n, "wl": wl, "wh": wh,
            "no_promised": float(promised.mean()),
            "edge": float(edge.mean()), "e_lo": e_lo, "e_hi": e_hi,
            "roi": float(roi_no.mean()), "r_lo": r_lo, "r_hi": r_hi}


def line(tag: str, st: dict) -> str:
    return (f"{tag:>18} {st['n']:>5} {st['events']:>5} {st['entry']:>6.3f} "
            f"{st['no_fact']:>7.1%} {st['no_promised']:>8.1%} "
            f"{st['edge'] * 100:>+6.1f} [{st['e_lo'] * 100:+5.1f};{st['e_hi'] * 100:+5.1f}] "
            f"{st['roi']:>+7.1%} [{st['r_lo']:+.0%};{st['r_hi']:+.0%}]")


def load_rows() -> list:
    oc = pd.read_parquet(OUTCOMES, columns=[
        "conditionId", "resolved_yes", "question", "slug", "endDate",
        "closedTime", "volumeNum", "feeType", "eventId"])
    oc = oc[oc["endDate"].notna()].copy()
    # Внимание: parquet отдаёт datetime64[us], а не [ns], поэтому
    # astype("int64") / 1e9 дал бы время в 1000 раз меньше и отправил все
    # даты в 1970 год. Считаем секунды от эпохи явно.
    oc["plan_ts"] = (oc["endDate"] - EPOCH).dt.total_seconds()
    info = oc.set_index("conditionId").to_dict("index")
    print(f"исходов загружено: {len(info)}", flush=True)

    rows = []
    for ln in SERIES.open(encoding="utf-8"):
        if not ln.strip():
            continue
        r = json.loads(ln)
        m = info.get(r["conditionId"])
        if not m:
            continue
        s = [(float(t), float(p)) for t, p in r["series"]]
        if len(s) < 4:
            continue
        rows.append((r["conditionId"], s, m))
    print(f"рядов с известным исходом: {len(rows)}", flush=True)
    return rows


def main() -> int:
    rows = load_rows()

    grid = [(lo, hi, w) for w in (24.0, 72.0)
            for lo in (0.05, 0.10, 0.15) for hi in (0.30, 0.40, 0.50)
            if hi > lo + 0.1]

    print()
    print("ЭДЖ СТАВКИ НА NO ПОСЛЕ СКАЧКА ЦЕНЫ ВВЕРХ")
    print("эдж, п.п. = фактическая частота побед NO минус та, что обещает цена входа.")
    print("Эффективный рынок даёт ноль. Интервалы 95%, бутстрэп по событиям.")
    print("ROI — после наценки входа и тейкерской комиссии, ставки равного размера.")
    print("-" * 118)
    print(f"{'T_lo->T_hi за W':>18} {'n':>5} {'соб':>5} {'вход':>6} "
          f"{'NO факт':>7} {'NO обещ':>8} {'эдж, п.п.':>21} {'ROI NO':>22}")

    keep = None
    for t_lo, t_hi, w_h in grid:
        ev = []
        for cid, s, m in rows:
            sh = find_shock(s, t_lo, t_hi, w_h, m["plan_ts"])
            if sh:
                ev.append({**sh, "cid": cid, "resolved_yes": m["resolved_yes"],
                           "eventId": m["eventId"] or cid,
                           "question": m["question"], "slug": m["slug"],
                           "volume": m["volumeNum"], "feeType": m["feeType"]})
        if len(ev) < 40:
            continue
        d = pd.DataFrame(ev)
        st = stats(d)
        print(line(f"{t_lo:.2f}->{t_hi:.2f} {w_h:.0f}ч", st))
        if t_lo == 0.15 and t_hi == 0.40 and w_h == 72.0:
            keep = d

    print()
    print("КОНТРОЛЬ: та же цена, но БЕЗ скачка (рост меньше 5 п.п. за W часов)")
    print(f"{'уровень за W':>18} {'n':>5} {'соб':>5} {'вход':>6} "
          f"{'NO факт':>7} {'NO обещ':>8} {'эдж, п.п.':>21} {'ROI NO':>22}")
    for w_h in (24.0, 72.0):
        for t_hi in (0.30, 0.40, 0.50):
            ev = []
            for cid, s, m in rows:
                cm = find_calm(s, t_hi, w_h, m["plan_ts"])
                if cm:
                    ev.append({**cm, "resolved_yes": m["resolved_yes"],
                               "eventId": m["eventId"] or cid})
            if len(ev) < 40:
                continue
            print(line(f"{t_hi:.2f}+ {w_h:.0f}ч", stats(pd.DataFrame(ev))))

    if keep is not None:
        keep.to_parquet(OUT, index=False)
        print()
        print(f"события 0.15->0.40 за 72ч сохранены: {OUT} ({len(keep)} шт.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
