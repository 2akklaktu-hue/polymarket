"""Хранение снимков живого рынка — накопление истории нужного разрешения.

Зачем это вообще. Исторические данные Polymarket дают около 4.7 часа на точку,
и этого принципиально мало, чтобы проверить правило, живущее на горизонте
минут: разница между входом на пике и входом через несколько часов составляет
всю величину эффекта (README, раздел 6). Единственный способ получить нужное
разрешение — писать свои снимки и ждать.

Формат. Два файла плюс служебный:
    data/live/markets.csv              реестр: короткий id -> conditionId, slug
    data/live/snap-YYYY-MM-DD.csv.gz   снимки: ts,id,bid,ask,last,vol
    data/live/last.json                последнее состояние, для дедупликации

Почему короткий id, а не conditionId: он занимает 66 символов, и при 1500
рынках каждые 10 минут это десятки мегабайт в сутки на одни идентификаторы.

В снимок попадают только рынки, у которых с прошлого раза изменилась хотя бы
одна цифра. Раз в FULL_EVERY_H часов пишется полный снимок, чтобы история не
зависела от сохранности last.json (в GitHub Actions он живёт в кэше и может
пропасть).

vol — НАКОПЛЕННЫЙ оборот рынка. Объём за период считается как разность двух
снимков, поэтому отдельные запросы сделок не нужны.
"""
from __future__ import annotations

import csv
import gzip
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data/live"
REGISTRY = LIVE / "markets.csv"
STATE = LIVE / "last.json"

FULL_EVERY_H = 6.0
COLS = ["ts", "id", "bid", "ask", "last", "vol"]
REG_COLS = ["id", "cid", "slug", "question", "end", "neg"]


def _round(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)


def registry_load() -> tuple[dict[str, int], int]:
    """Возвращает (cid -> id, следующий свободный id)."""
    if not REGISTRY.exists():
        return {}, 1
    m: dict[str, int] = {}
    mx = 0
    with REGISTRY.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                i = int(row["id"])
            except (TypeError, ValueError):
                continue
            m[row["cid"]] = i
            mx = max(mx, i)
    return m, mx + 1


def registry_append(rows: list[dict]) -> None:
    """Дописывает в реестр новые рынки (ожидает поля REG_COLS)."""
    if not rows:
        return
    LIVE.mkdir(parents=True, exist_ok=True)
    new = not REGISTRY.exists()
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REG_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def registry_frame() -> pd.DataFrame:
    if not REGISTRY.exists():
        return pd.DataFrame(columns=REG_COLS)
    return pd.read_csv(REGISTRY, dtype={"cid": str, "slug": str})


def state_load() -> dict:
    if not STATE.exists():
        return {}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def state_save(rows: dict[str, list], ts: float) -> None:
    LIVE.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"ts": ts, "rows": rows}, separators=(",", ":")),
                     encoding="utf-8")


def changed(prev: dict, mid: str, bid, ask, last, vol) -> bool:
    """Изменилось ли состояние рынка с прошлого снимка."""
    old = prev.get(mid)
    if not old:
        return True
    return [_round(bid), _round(ask), _round(last), _round(vol, 2)] != old


def snapshot_write(ts: float, items: list[tuple]) -> Path:
    """Дописывает снимок. items: (id, bid, ask, last, vol)."""
    LIVE.mkdir(parents=True, exist_ok=True)
    day = time.strftime("%Y-%m-%d", time.gmtime(ts))
    path = LIVE / f"snap-{day}.csv.gz"
    # заголовка нет намеренно: файл дописывается многократно, порядок колонок
    # фиксирован в COLS, при чтении имена задаются явно
    with gzip.open(path, "at", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for mid, bid, ask, last, vol in items:
            w.writerow([int(ts), mid, _round(bid), _round(ask),
                        _round(last), _round(vol, 2)])
    return path


def snapshot_load(hours: float = 48.0, now: float | None = None) -> pd.DataFrame:
    """Снимки за последние `hours` часов из суточных файлов."""
    now = now or time.time()
    since = now - hours * 3600.0
    frames = []
    for p in sorted(LIVE.glob("snap-*.csv.gz")):
        try:
            frames.append(pd.read_csv(p, names=COLS, dtype={"id": "Int64"}))
        except (OSError, EOFError, pd.errors.ParserError):
            continue           # битый хвост файла не должен ронять разбор
    if not frames:
        return pd.DataFrame(columns=COLS)
    d = pd.concat(frames, ignore_index=True)
    d = d[d["ts"] >= since]
    return d.sort_values("ts").reset_index(drop=True)
