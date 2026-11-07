"""Восстановление истории цен закрытых рынков из ленты сделок.

Зачем так: CLOB-эндпоинт /prices-history отдаёт данные только по рынкам,
закрытым считанные недели назад. Для обучающей выборки нужна история за годы,
а она доступна только через data-api /trades.

Особенности data-api /trades:
  * сортировка от НОВЫХ к старым;
  * offset ограничен 10 000 — глубже истории не достать;
  * фильтров по времени нет;
  * в ленте вперемешку сделки по YES и NO токенам, приводим к YES-эквиваленту
    (для outcomeIndex=1 цена YES = 1 - price).

На выходе на каждый рынок пишется сжатый ряд: VWAP по корзинам времени,
до N_BUCKETS точек. Этого достаточно, чтобы потом посчитать цену на любом
горизонте, и на порядки меньше, чем хранить все сделки.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmlib.api import Api, ApiError  # noqa: E402
from pmlib.dataset import load, clean_for_calibration  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/price_series.jsonl"

PAGES = 3           # сколько страниц по 1000 сделок тянуть с конца
N_BUCKETS = 48      # во сколько точек сжимать ряд
WORKERS = 24


def series_for(api: Api, cid: str) -> dict | None:
    """Сжатый ряд YES-цены по последним PAGES*1000 сделкам рынка."""
    try:
        rows = api.trades_pages(cid, pages=PAGES, limit=1000)
    except ApiError:
        return None
    if len(rows) < 5:
        return None

    ts = np.array([int(r["timestamp"]) for r in rows], dtype=np.int64)
    px = np.array([float(r["price"]) for r in rows], dtype=float)
    # в v2 поле называется outcome_index (в v1 было outcomeIndex)
    idx = np.array([int(r.get("outcome_index") or 0) for r in rows], dtype=int)
    sz = np.array([float(r.get("size") or 0.0) for r in rows], dtype=float)

    yes = np.where(idx == 1, 1.0 - px, px)          # приведение к YES
    ok = np.isfinite(yes) & (yes > 0) & (yes < 1) & (sz > 0)
    ts, yes, sz = ts[ok], yes[ok], sz[ok]
    if ts.size < 5:
        return None

    order = np.argsort(ts)                          # по возрастанию времени
    ts, yes, sz = ts[order], yes[order], sz[order]

    # сжатие в N_BUCKETS корзин равной длительности, VWAP внутри корзины
    t0, t1 = int(ts[0]), int(ts[-1])
    if t1 <= t0:
        return None
    edges = np.linspace(t0, t1 + 1, N_BUCKETS + 1)
    b = np.clip(np.digitize(ts, edges) - 1, 0, N_BUCKETS - 1)
    pts = []
    for k in range(N_BUCKETS):
        m = b == k
        if not m.any():
            continue
        w = sz[m]
        pts.append([int(ts[m][-1]), round(float((yes[m] * w).sum() / w.sum()), 5)])

    return {
        "conditionId": cid,
        "n_trades_seen": int(ts.size),
        "capped": len(rows) >= PAGES * 1000,
        "t_first": t0, "t_last": t1,
        "series": pts,
    }


def main() -> int:
    src = ROOT / "data/raw/markets_closed.jsonl"
    if not src.exists():
        print("нет data/raw/markets_closed.jsonl — сначала запусти 01_fetch_markets.py")
        return 1

    df = clean_for_calibration(load(src), min_volume=float(sys.argv[1]) if len(sys.argv) > 1 else 5000.0)
    df = df.sort_values("endDate", ascending=False)
    print(f"рынков после отсева: {len(df)}", flush=True)

    done = set()
    if OUT.exists():
        for ln in OUT.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                done.add(json.loads(ln)["conditionId"])
        print(f"уже собрано ранее: {len(done)}", flush=True)

    todo = [c for c in df["conditionId"].dropna().unique() if c not in done]
    print(f"к загрузке: {len(todo)}", flush=True)
    if not todo:
        return 0

    api = Api(min_interval=0.012)
    lock = threading.Lock()
    t0 = time.time()
    n_ok = n_empty = 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f, ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(series_for, api, c): c for c in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            cid = futs[fu]
            try:
                res = fu.result()
            except Exception as e:                       # noqa: BLE001
                res = None
                if i < 5:
                    print("  ошибка:", repr(e)[:120], flush=True)
            with lock:
                if res:
                    f.write(json.dumps(res) + "\n")
                    n_ok += 1
                else:
                    n_empty += 1
                if i % 500 == 0:
                    el = time.time() - t0
                    rate = i / el
                    print(f"  {i}/{len(todo)} | ок {n_ok} пусто {n_empty} | "
                          f"{rate:.1f} рынков/с | осталось ~{(len(todo)-i)/rate/60:.0f} мин",
                          flush=True)
                    f.flush()
    print(f"ГОТОВО: {n_ok} рядов, {n_empty} без данных -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
