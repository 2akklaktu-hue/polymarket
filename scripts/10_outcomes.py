"""Потоковый сбор исходов закрытых рынков в компактную таблицу.

Зачем отдельно от pmlib.dataset.load: тот читает весь markets_closed.jsonl
(820 МБ) в память одним read_text и требует цену исхода РОВНО 0.0 или 1.0,
а у рынков до 2021 года там записано 0.0000096 — они молча теряются.
Здесь поток построчно и допуск по порогу.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/markets_closed.jsonl"
OUT = ROOT / "data/out/outcomes.parquet"

EPS = 0.01          # допуск: p0 > 1-EPS это YES, p0 < EPS это NO


def _jsl(x):
    if isinstance(x, list):
        return x
    try:
        v = json.loads(x) if x else []
        return v if isinstance(v, list) else []
    except (TypeError, ValueError):
        return []


def main() -> int:
    if not SRC.exists():
        print(f"нет {SRC}")
        return 1

    rows = []
    n = bad = n_multi = n_unres = n_mid = 0
    with SRC.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            n += 1

            outs = _jsl(r.get("outcomes"))
            pl = _jsl(r.get("outcomePrices"))
            if len(outs) != 2 or len(pl) != 2:
                n_multi += 1
                continue
            try:
                p0 = float(pl[0])
            except (TypeError, ValueError):
                n_unres += 1
                continue
            if p0 > 1 - EPS:
                res = 1
            elif p0 < EPS:
                res = 0
            else:
                n_mid += 1          # не доведён до резолюции
                continue

            rows.append({
                "conditionId": r.get("conditionId"),
                "resolved_yes": res,
                "question": r.get("question") or "",
                "slug": r.get("slug") or "",
                "endDate": r.get("endDate"),
                "closedTime": r.get("closedTime"),
                "createdAt": r.get("createdAt"),
                "volumeNum": r.get("volumeNum"),
                "feeType": r.get("feeType"),
                "negRisk": bool(r.get("negRisk")),
                "eventId": r.get("eventId"),
                "eventTitle": r.get("eventTitle") or "",
            })
            if n % 50000 == 0:
                print(f"  прочитано {n}, годных {len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df = df[df["conditionId"].notna()].drop_duplicates("conditionId")
    for c in ("endDate", "closedTime", "createdAt"):
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True, format="mixed")
    df["volumeNum"] = pd.to_numeric(df["volumeNum"], errors="coerce")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print()
    print(f"строк в файле:        {n}   битых: {bad}")
    print(f"не бинарных:          {n_multi}")
    print(f"цена исхода в середине (нет резолюции): {n_mid}")
    print(f"ГОДНЫХ рынков:        {len(df)}  ->  {OUT}")
    print(f"доля YES:             {df['resolved_yes'].mean():.3f}")
    print(f"период endDate:       {df['endDate'].min()} .. {df['endDate'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
