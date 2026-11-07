"""Выгрузка метаданных рынков Polymarket.

Почему так сложно: keyset-курсор gamma-api сломан (не продвигается), а offset
жёстко ограничен 5000. Поэтому пространство рынков режется окнами по endDate;
если окно упирается в лимит 5000, оно рекурсивно дробится пополам.

Результат: data/raw/markets_closed.jsonl, data/raw/markets_open.jsonl
Прогресс пишется по окнам в data/raw/_chunks/, запуск можно возобновлять.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmlib.api import Api, ApiError, GAMMA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data/raw/_chunks"
PAGE = 100
# gamma отклоняет offset+limit > 5000, поэтому останавливаемся заранее
OFFSET_CAP = 4800

KEEP = [
    "id", "question", "slug", "conditionId", "clobTokenIds",
    "outcomes", "outcomePrices", "closed", "active", "archived",
    "startDate", "endDate", "closedTime", "createdAt",
    "volumeNum", "liquidity", "spread", "bestBid", "bestAsk", "lastTradePrice",
    "negRisk", "negRiskMarketID", "groupItemTitle",
    "feesEnabled", "feeSchedule", "feeType", "takerBaseFee", "makerBaseFee",
    "orderMinSize", "orderPriceMinTickSize",
    "umaResolutionStatus", "umaBond", "resolvedBy",
    "oneDayPriceChange", "oneWeekPriceChange", "enableOrderBook",
]


def slim(m: dict) -> dict:
    row = {k: m.get(k) for k in KEEP}
    evs = m.get("events") or []
    if evs:
        e = evs[0]
        row["eventId"] = e.get("id")
        row["eventSlug"] = e.get("slug")
        row["eventTitle"] = e.get("title")
        ser = e.get("series") or []
        row["seriesSlug"] = ser[0].get("slug") if ser else None
    return row


def iso(d: dt.date) -> str:
    return d.strftime("%Y-%m-%dT00:00:00Z")


def fetch_window(api: Api, closed: bool, lo: dt.date, hi: dt.date,
                 depth: int = 0) -> list[dict]:
    """Все рынки с endDate в [lo, hi). При переполнении окно делится пополам."""
    rows, offset, overflow = [], 0, False
    while offset < OFFSET_CAP:
        p = {"limit": PAGE, "offset": offset, "closed": str(closed).lower(),
             "end_date_min": iso(lo), "end_date_max": iso(hi)}
        try:
            batch = api.get(f"{GAMMA}/markets", p)
        except ApiError as e:
            if "offset too large" not in str(e):
                raise
            overflow = True
            break
        if not batch:
            return rows
        rows.extend(batch)
        if len(batch) < PAGE:
            return rows
        offset += PAGE
    else:
        overflow = True

    if not overflow:
        return rows

    # упёрлись в потолок — окно слишком широкое, дробим
    span = (hi - lo).days
    if span <= 1:
        print(f"    ! окно {lo} шириной 1 день переполнено, часть данных потеряна",
              flush=True)
        return rows
    mid = lo + dt.timedelta(days=span // 2)
    return fetch_window(api, closed, lo, mid, depth + 1) + \
           fetch_window(api, closed, mid, hi, depth + 1)


def months(start: dt.date, end: dt.date):
    cur = dt.date(start.year, start.month, 1)
    while cur < end:
        nxt = dt.date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        yield cur, nxt
        cur = nxt


def run(closed: bool, out: Path, start: dt.date, end: dt.date) -> None:
    api = Api(min_interval=0.04)
    CHUNKS.mkdir(parents=True, exist_ok=True)
    tag = "closed" if closed else "open"
    seen, total = set(), 0

    with out.open("w", encoding="utf-8") as fout:
        for lo, hi in months(start, end):
            cf = CHUNKS / f"{tag}_{lo:%Y%m}.jsonl"
            if cf.exists():                       # возобновление
                raw = [json.loads(x) for x in cf.read_text(encoding="utf-8").splitlines() if x.strip()]
            else:
                raw = [slim(m) for m in fetch_window(api, closed, lo, hi)]
                cf.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in raw),
                              encoding="utf-8")
            new = 0
            for r in raw:
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                new += 1
            total += new
            if raw:
                print(f"  {lo:%Y-%m}: {len(raw):>5} рынков ({new} новых) | всего {total:>7} "
                      f"| {api.n_calls} запросов", flush=True)
    print(f"ГОТОВО {tag}: {total} рынков -> {out}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "closed"
    START, END = dt.date(2020, 1, 1), dt.date(2027, 6, 1)
    if which in ("closed", "both"):
        print("=== ЗАКРЫТЫЕ рынки ===", flush=True)
        run(True, ROOT / "data/raw/markets_closed.jsonl", START, END)
    if which in ("open", "both"):
        print("=== ОТКРЫТЫЕ рынки ===", flush=True)
        run(False, ROOT / "data/raw/markets_open.jsonl", START, dt.date(2031, 1, 1))
