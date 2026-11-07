"""Объём вокруг каждого скачка цены — то, чего нет в сжатой истории.

Зачем. Исходная гипотеза пользователя звучала как «растёт объём И процент»,
но в data/raw/price_series.jsonl сохранён только VWAP цены по корзинам, объём
выброшен. Поэтому половину сигнала на истории проверить было нельзя. Здесь
объём дотягивается точечно: только по рынкам, где скачок уже найден.

Как. Сделки v2 отдаются от новых к старым, фильтров по времени для выборки по
рынку нет, поэтому идём курсором с конца, пропуская всё, что новее момента
скачка, и собираем окно [t - BASE_D суток, t]. Скачки найдены внутри последних
~3000 сделок рынка (ряды цен строились из них), так что MAX_PAGES хватает.

Что считается: объём в долларах за 72 часа и за 24 часа до входа, базовый
дневной объём до этого окна, их отношение, число сделок и уникальных
кошельков, крупнейшая сделка и доля покупок ДА в объёме.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmlib.api import Api, ApiError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data/out/shock_events.parquet"
OUT_JSONL = ROOT / "data/out/shock_volume.jsonl"
OUT = ROOT / "data/out/shock_volume.parquet"

WIN_H = 72.0             # окно скачка
BASE_D = 10.0            # сколько суток назад брать базовый объём
MAX_PAGES = 8            # страниц по 1000 сделок на рынок
WORKERS = 12


def collect(api: Api, cid: str, t: float) -> dict:
    """Объём вокруг момента t по рынку cid."""
    t_win = t - WIN_H * 3600.0
    t_base = t - BASE_D * 86400.0
    win: list[dict] = []
    base_usd = 0.0
    cur: str | None = None
    pages = 0
    covered = False

    for _ in range(MAX_PAGES):
        try:
            batch, cur = api.trades(cid, limit=1000, cursor=cur)
        except ApiError:
            break
        pages += 1
        if not batch:
            covered = True
            break
        oldest = float(batch[-1].get("timestamp") or 0)
        for r in batch:
            ts = float(r.get("timestamp") or 0)
            if ts > t or ts < t_base:
                continue
            usd = float(r.get("size") or 0.0) * float(r.get("price") or 0.0)
            if ts >= t_win:
                win.append({"ts": ts, "usd": usd,
                            "w": r.get("proxy_wallet") or "",
                            "side": r.get("side") or "",
                            "oi": int(r.get("outcome_index") or 0)})
            else:
                base_usd += usd
        if oldest < t_base or not cur:
            covered = True
            break

    vol72 = sum(x["usd"] for x in win)
    vol24 = sum(x["usd"] for x in win if x["ts"] >= t - 24 * 3600.0)
    # покупка ДА это BUY по токену YES либо SELL по токену NO
    buy_yes = sum(x["usd"] for x in win
                  if (x["side"] == "BUY") == (x["oi"] == 0))
    base_day = base_usd / max(BASE_D - WIN_H / 24.0, 0.1)
    return {
        "cid": cid,
        "vol_usd_72h": round(vol72, 2),
        "vol_usd_24h": round(vol24, 2),
        "base_usd_per_day": round(base_day, 2),
        "vol_ratio": round((vol72 / 3.0) / base_day, 3) if base_day > 1e-6 else None,
        "n_trades_72h": len(win),
        "n_wallets_72h": len({x["w"] for x in win if x["w"]}),
        "max_trade_usd": round(max((x["usd"] for x in win), default=0.0), 2),
        "buy_yes_share": round(buy_yes / vol72, 3) if vol72 > 1e-6 else None,
        "pages": pages,
        "covered": covered,
    }


def main() -> int:
    d = pd.read_parquet(EVENTS, columns=["cid", "t"])
    done: set[str] = set()
    if OUT_JSONL.exists():
        for ln in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                done.add(json.loads(ln)["cid"])
        print(f"уже собрано ранее: {len(done)}", flush=True)
    todo = [(str(c), float(t)) for c, t in zip(d["cid"], d["t"])
            if str(c) not in done]
    print(f"к загрузке: {len(todo)} из {len(d)}", flush=True)

    if todo:
        api = Api(min_interval=0.02)
        lock = threading.Lock()
        t0 = time.time()
        OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with OUT_JSONL.open("a", encoding="utf-8") as f, \
                ThreadPoolExecutor(WORKERS) as ex:
            futs = {ex.submit(collect, api, c, t): c for c, t in todo}
            for i, fu in enumerate(as_completed(futs), 1):
                try:
                    res = fu.result()
                except Exception as e:                      # noqa: BLE001
                    print("  ошибка:", repr(e)[:110], flush=True)
                    continue
                with lock:
                    f.write(json.dumps(res) + "\n")
                    if i % 200 == 0:
                        el = time.time() - t0
                        rate = i / el
                        print(f"  {i}/{len(todo)} | {rate:.1f}/с | "
                              f"осталось ~{(len(todo) - i) / rate / 60:.1f} мин",
                              flush=True)
                        f.flush()

    rows = [json.loads(ln) for ln in
            OUT_JSONL.read_text(encoding="utf-8").splitlines() if ln.strip()]
    v = pd.DataFrame(rows).drop_duplicates("cid")
    v.to_parquet(OUT, index=False)
    print()
    print(f"собрано рынков: {len(v)}  -> {OUT}")
    print(f"окно покрыто полностью: {int(v['covered'].sum())}")
    print(f"медианный объём за 72ч: ${v['vol_usd_72h'].median():,.0f}")
    print(f"медианное отношение к базовому: {v['vol_ratio'].median():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
