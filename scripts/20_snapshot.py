"""Снимок живого рынка: цены, спред и накопленный оборот всех торгуемых рынков.

Запускать по cron каждые 5-10 минут. Один прогон — около 25 секунд и 25
запросов, потому что gamma отдаёт bestBid/bestAsk/volumeNum сразу в списке
рынков, и обходить стаканы по одному не нужно.

Рынки берутся в порядке убывания оборота. Глубже offset 2100 gamma отвечает
422, но это и не нужно: за этой границей остаются рынки, на которых всё равно
нельзя торговать.

Зачем копить снимки — см. pmlib/snap.py и README, раздел 6.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmlib.api import Api, ApiError, GAMMA  # noqa: E402
from pmlib import snap  # noqa: E402

# Фильтры «рынок настоящий». Те же по смыслу, что в 05_screen.py: у мёртвого
# рынка bid=0.001 и ask=0.999, его середина равна 0.5 и не значит ничего.
MIN_BID = 0.002
MAX_ASK = 0.998
MAX_SPREAD = 0.08         # чуть шире, чем у скринера: тут не торгуем, а пишем
MIN_VOL = 1000.0          # $ накопленного оборота рынка
MIN_HOURS_LEFT = 12.0     # рынки, закрывающиеся в ближайшие часы, не нужны

# Высокочастотные повторяющиеся рынки: их тысячи, они забьют снимки шумом.
_HIGHFREQ = re.compile(
    r"(?:up or down)|(?:\b\d{1,2}:\d{2}\s*(?:am|pm)\b)"
    r"|(?:\b\d{1,2}(?:am|pm)\s*-\s*\d{1,2}(?:am|pm)\b)"
    r"|(?:\bhourly\b)|(?:\bevery \d+ minutes?\b)", re.I)

PAGE = 100
MAX_OFFSET = 5000


def _f(x, default=float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _end_ts(s) -> float:
    """endDate вида 2027-01-01T04:59:00Z в epoch-секунды UTC."""
    if not s:
        return float("nan")
    try:
        return float(time.mktime(time.strptime(str(s)[:19], "%Y-%m-%dT%H:%M:%S"))
                     - time.timezone)
    except ValueError:
        return float("nan")


def fetch_live(api: Api) -> list[dict]:
    """Торгуемые рынки в порядке убывания оборота."""
    out: list[dict] = []
    off = 0
    while off < MAX_OFFSET:
        try:
            batch = api.get(f"{GAMMA}/markets",
                            {"limit": PAGE, "offset": off, "closed": "false",
                             "active": "true", "order": "volumeNum",
                             "ascending": "false"})
        except ApiError:
            break              # за offset ~2100 gamma отвечает 422, это норма
        if not batch:
            break
        out.extend(batch)
        off += PAGE
        if len(batch) < PAGE:
            break
    return out


def main() -> int:
    now = time.time()
    api = Api(min_interval=0.05)
    raw = fetch_live(api)
    print(f"получено рынков: {len(raw)} ({api.n_calls} запросов)", flush=True)
    if not raw:
        print("! gamma не отдала ни одного рынка")
        return 1

    reg, next_id = snap.registry_load()
    prev = snap.state_load()
    prev_rows = prev.get("rows") or {}
    full = (now - float(prev.get("ts") or 0)) > snap.FULL_EVERY_H * 3600.0

    new_markets: list[dict] = []
    items: list[tuple] = []
    state: dict[str, list] = {}
    n_skip = 0

    for r in raw:
        cid = r.get("conditionId")
        if not cid:
            continue
        q = r.get("question") or ""
        bid, ask = _f(r.get("bestBid")), _f(r.get("bestAsk"))
        last, vol = _f(r.get("lastTradePrice")), _f(r.get("volumeNum"), 0.0)
        if not (bid > MIN_BID and ask < MAX_ASK and (ask - bid) <= MAX_SPREAD):
            n_skip += 1
            continue
        if vol < MIN_VOL or _HIGHFREQ.search(q) or not r.get("enableOrderBook"):
            n_skip += 1
            continue
        left_h = (_end_ts(r.get("endDate")) - now) / 3600.0
        if left_h == left_h and left_h < MIN_HOURS_LEFT:   # NaN пропускаем
            n_skip += 1
            continue

        mid_id = reg.get(cid)
        if mid_id is None:
            mid_id = next_id
            next_id += 1
            reg[cid] = mid_id
            new_markets.append({
                "id": mid_id, "cid": cid, "slug": r.get("slug") or "",
                "question": q, "end": r.get("endDate") or "",
                "neg": int(bool(r.get("negRisk"))),
            })
        key = str(mid_id)
        state[key] = [snap._round(bid), snap._round(ask),
                      snap._round(last), snap._round(vol, 2)]
        if full or snap.changed(prev_rows, key, bid, ask, last, vol):
            items.append((mid_id, bid, ask, last, vol))

    snap.registry_append(new_markets)
    path = snap.snapshot_write(now, items) if items else None
    snap.state_save(state, now)

    kb = (path.stat().st_size / 1024.0) if path else 0.0
    print(f"отфильтровано: {n_skip} | в снимке: {len(state)} рынков")
    print(f"записано строк: {len(items)}"
          f"{' (ПОЛНЫЙ снимок)' if full else ' (только изменения)'}")
    print(f"новых рынков в реестре: {len(new_markets)}")
    if path:
        print(f"файл: {path} ({kb:.0f} КБ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
