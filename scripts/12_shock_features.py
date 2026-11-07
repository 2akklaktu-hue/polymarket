"""Таблица скачков цены с признаками — основа для поиска усиливающих условий.

Берётся мягкий детектор (цена была <= P_LO, за W часов дошла до >= P_HI) и по
каждому найденному скачку сохраняются признаки, известные В МОМЕНТ входа:
форма скачка, запас времени, тип рынка, активность. Исход сохраняется рядом,
но в признаки не входит.

Почему первый скачок на рынок, а не все: иначе один затяжной памп даст
несколько почти одинаковых наблюдений и выборка раздуется искусственно.

Буфер до конца считается от endDate (плановая дата), а не от closedTime —
момент фактической резолюции зависит от исхода и даёт фальшивый эдж.
Подробно в README, раздел 6.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmlib.dataset import guess_category  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "data/raw/price_series.jsonl"
OUTCOMES = ROOT / "data/out/outcomes.parquet"
OUT = ROOT / "data/out/shock_events.parquet"

P_LO = 0.15              # мягкие пороги: нужен запас событий для разбивок
P_HI = 0.30
W_H = 72.0
BUFFER_H = 24.0
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

# Вопросы про высказывания — тот тип рынка, с которого начался весь разговор:
# «скажет ли он такое слово», «упомянет ли», «твитнет ли».
_VERBAL = re.compile(
    r"\b(?:say|says|said|saying|mention|mentions|mentioned|tweet|tweets|"
    r"tweeted|post|posts|posted|speak|speaks|utter|word|words|phrase|"
    r"announce|announces|announced|call|calls|called|declare|claim|claims|"
    r"statement|speech|interview|press conference)\b", re.I)


def main() -> int:
    oc = pd.read_parquet(OUTCOMES, columns=[
        "conditionId", "resolved_yes", "question", "slug", "endDate",
        "createdAt", "volumeNum", "feeType", "eventId"])
    oc = oc[oc["endDate"].notna()].copy()
    # parquet отдаёт datetime64[us]: считаем секунды от эпохи явно,
    # иначе astype("int64")/1e9 уводит даты в 1970 год.
    oc["plan_ts"] = (oc["endDate"] - EPOCH).dt.total_seconds()
    info = oc.set_index("conditionId").to_dict("index")
    print(f"исходов загружено: {len(info)}", flush=True)

    rows = []
    n_series = 0
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
        n_series += 1

        plan_ts = m["plan_ts"]
        w = W_H * 3600.0
        for j in range(1, len(s)):
            tj, pj = s[j]
            if pj < P_HI:
                continue
            lo_p, lo_t = 1.0, None
            for i in range(j - 1, -1, -1):
                ti, pi = s[i]
                if tj - ti > w:
                    break
                if pi < lo_p:
                    lo_p, lo_t = pi, ti
            if lo_t is None or lo_p > P_LO:
                continue
            if (plan_ts - tj) < BUFFER_H * 3600.0:
                break                    # сигнала нет: слишком близко к концу
            rise_h = max((tj - lo_t) / 3600.0, 0.01)
            jump = pj - lo_p
            q = m["question"] or ""
            rows.append({
                "cid": r["conditionId"],
                "t": tj,
                "p_before": lo_p,
                "p_entry": pj,
                "jump": jump,
                "rise_h": rise_h,
                "speed_pp_h": 100.0 * jump / rise_h,
                "days_left": (plan_ts - tj) / 86400.0,
                "days_lived": (tj - s[0][0]) / 86400.0,
                "volume": m["volumeNum"],
                "n_trades": r.get("n_trades_seen"),
                "capped": bool(r.get("capped")),
                # NB: pd.NA/NaN в Python истинно, поэтому проверка именно
                # через pd.isna — иначе категория становится строкой "nan".
                "category": (guess_category(q) if pd.isna(m["feeType"])
                             else str(m["feeType"]).replace("_fees", "")),
                "verbal": bool(_VERBAL.search(q)),
                "resolved_yes": m["resolved_yes"],
                "eventId": m["eventId"] or r["conditionId"],
                "question": q,
                "slug": m["slug"],
            })
            break

    d = pd.DataFrame(rows)
    d["year"] = pd.to_datetime(d["t"], unit="s", utc=True).dt.year
    d["no_won"] = (d["resolved_yes"] == 0)
    d["promised_no"] = 1.0 - d["p_entry"]
    d["edge"] = d["no_won"].astype(float) - d["promised_no"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT, index=False)

    print(f"рядов просмотрено: {n_series}")
    print(f"скачков {P_LO}->{P_HI} за {W_H:.0f}ч с запасом {BUFFER_H:.0f}ч: {len(d)}")
    print(f"событий (eventId): {d['eventId'].nunique()}")
    print(f"средний эдж: {100 * d['edge'].mean():+.1f} п.п.")
    print()
    print("по годам:")
    print(d.groupby("year").agg(n=("no_won", "size"),
                                эдж_пп=("edge", lambda x: round(100 * x.mean(), 1))
                                ).to_string())
    print()
    print("по категориям (топ-12 по размеру):")
    g = d.groupby("category").agg(n=("no_won", "size"),
                                  эдж_пп=("edge", lambda x: round(100 * x.mean(), 1)))
    print(g.sort_values("n", ascending=False).head(12).to_string())
    print()
    print(f"вопросов про высказывания (verbal): {int(d['verbal'].sum())}, "
          f"эдж {100 * d.loc[d['verbal'], 'edge'].mean():+.1f} п.п.")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
