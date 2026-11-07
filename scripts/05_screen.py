"""Живой скринер: поиск рынков, где расхождение перекрывает стоимость входа.

Почему устроено именно так. Историческая калибровка (см. 03/04) показала:
покупка по рыночной цене убыточна на ЛЮБОМ уровне цены — статистический
перекос либо отрицательный, либо меньше наценки на входе. Значит искать надо
не «среднюю закономерность», а ОТДЕЛЬНЫЕ рынки с большим расхождением.

Две оценки справедливой вероятности, обе не требуют внешних данных:

  1. КАЛИБРОВОЧНАЯ. logit(q) = a + b*logit(mid), коэффициенты из 03_calibration.
     Поправка на то, что рынок систематически сжимает вероятности к середине.

  2. СОСЕДСКАЯ (только для взаимоисключающих событий с полным набором исходов).
     Истинные вероятности обязаны давать в сумме 1, поэтому fair_i = mid_i/sum(mid).
     Расхождение ask и fair для одного исхода на фоне остальных — сигнал.

Два режима входа:

  ТЕЙКЕР — купить по ask прямо сейчас. Платится комиссия и весь спред.
  МЕЙКЕР — выставить лимитную заявку внутри спреда. Комиссия НУЛЕВАЯ
     (feeSchedule.takerOnly = true), спред работает на тебя, но заявка
     исполнится только если кто-то захочет продать по твоей цене.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmlib.api import Api, ApiError, GAMMA  # noqa: E402
from pmlib import book as B  # noqa: E402
from pmlib.econ import FEE_RATES, DEFAULT_FEE_RATE, effective_price, evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data/out"

MIN_LIQ_EVENT = 3000.0     # $ ликвидности события, ниже — не смотрим
MIN_DEPTH_USD = 30.0       # $ на лучшем уровне: должно хватать на нашу ставку
STAKE_USD = 15.0           # типичный размер ставки, от него считаем исполнение
MIN_EDGE_TAKER = 0.03      # порог: 3 п.п. чистого эджа после всех издержек
MIN_EDGE_MAKER = 0.05      # мейкеру порог выше: заявка может не исполниться
MIN_DAYS_LEFT = 1.0        # рынки, закрывающиеся сегодня, пропускаем

# --- фильтры «рынок настоящий» -------------------------------------------
# Без них скринер ломается предсказуемым образом: у мёртвого рынка bid=0.001,
# ask=0.999, середина равна 0.5, и калибровка объявляет её справедливой
# вероятностью. Получаются «эджи» под 50 п.п. на рынках, где никто не торгует.
# Середина стакана несёт информацию только когда с обеих сторон стоят живые
# котировки и спред узкий.
MAX_SPREAD_ABS = 0.06      # шире 6 центов — котировки не информативны
MAX_SPREAD_REL = 0.50      # и не шире половины самой цены
MIN_BID = 0.002            # односторонний стакан не считается рынком
MIN_MARKET_VOL = 2000.0    # $ оборота на самом рынке, а не только на событии
# Полнота набора исходов: сумма середин должна быть близка к 1, иначе
# «дешевизна» объясняется просто отсутствующими вариантами.
COMPLETE_LO, COMPLETE_HI = 0.92, 1.12


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_fit() -> tuple[float, float]:
    """Коэффициенты калибровки. Лежат в models/ (в репозитории), а не в data/,
    чтобы скринер работал на чистой машине без пересчёта истории."""
    f = ROOT / "models/calibration.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        return float(d["a"]), float(d["b"])
    legacy = OUTDIR / "calib_fit.npy"
    if legacy.exists():
        a, b = np.load(legacy)
        return float(a), float(b)
    print("! нет models/calibration.json — поправка отключена (a=0, b=1)")
    return 0.0, 1.0


@dataclass
class Signal:
    event: str
    outcome: str
    slug: str
    kind: str            # 'тейкер' или 'мейкер'
    basis: str           # на какой оценке справедливой цены основан сигнал
    price: float         # цена входа (ask для тейкера, лимит для мейкера)
    fair: float          # оценка истинной вероятности
    edge: float          # fair - цена_с_комиссией, в п.п. вероятности
    roi: float
    kelly: float
    stake: float         # рекомендуемый размер, $
    depth_usd: float
    spread: float
    days: float | None
    url: str

    def line(self) -> str:
        d = f"{self.days:.0f}д" if self.days else "?"
        return (f"[{self.kind}] {self.event} → {self.outcome}\n"
                f"   вход {self.price:.3f} | справедливо {self.fair:.3f} | "
                f"эдж {self.edge*100:+.1f}пп | ROI {self.roi*100:+.0f}% | "
                f"до конца {d}\n"
                f"   основание: {self.basis} | глубина ${self.depth_usd:,.0f} | "
                f"спред {self.spread:.3f} | размер ${self.stake:.2f}\n"
                f"   {self.url}")


def fetch_events(api: Api, min_liq: float) -> list[dict]:
    evs, off = [], 0
    while True:
        try:
            b = api.get(f"{GAMMA}/events",
                        {"limit": 100, "offset": off, "closed": "false",
                         "liquidity_min": min_liq, "order": "liquidity",
                         "ascending": "false"})
        except ApiError:
            break
        if not b:
            break
        evs += b
        if len(b) < 100:
            break
        off += 100
    return evs


def days_left(m: dict) -> float | None:
    import datetime as dt
    s = m.get("endDate")
    if not s:
        return None
    try:
        end = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return max((end - dt.datetime.now(dt.UTC)).total_seconds() / 86400.0, 0.0)


def scan(api: Api, a: float, b: float, stake: float = STAKE_USD,
         min_taker: float = MIN_EDGE_TAKER,
         min_maker: float = MIN_EDGE_MAKER) -> list[Signal]:
    evs = fetch_events(api, MIN_LIQ_EVENT)
    print(f"событий: {len(evs)}", flush=True)

    # 1) собираем кандидатов и их токены
    cands = []                       # (event, market, fair_peer|None)
    for e in evs:
        mk = [m for m in (e.get("markets") or [])
              if not m.get("closed") and m.get("acceptingOrders")
              and m.get("clobTokenIds")]
        if not mk:
            continue
        fair_peer = {}
        # Оценка «сумма исходов = 1» верна ТОЛЬКО если набор исходов полон.
        # Считать надо все непогашенные исходы события, включая неторгуемые:
        # в событиях часто есть исход "Other" без стакана, и если его
        # выбросить, сумма нормируется не на ту величину, а редкий исход
        # превращается в фиктивный эдж на фаворите.
        all_open = [m for m in (e.get("markets") or []) if not m.get("closed")]
        if (e.get("negRisk") or e.get("enableNegRisk")) and len(all_open) >= 3:
            try:
                mid = np.array([(float(m["bestAsk"]) + float(m["bestBid"])) / 2
                                for m in all_open])
                ok = True
            except (TypeError, ValueError, KeyError):
                ok = False
            # исход без котировок делает набор непроверяемым
            if ok and all(m.get("acceptingOrders") for m in all_open) \
                    and COMPLETE_LO <= mid.sum() <= COMPLETE_HI:
                for m, f in zip(all_open, mid / mid.sum()):
                    fair_peer[m["conditionId"]] = float(f)
        for m in mk:
            cands.append((e, m, fair_peer.get(m["conditionId"])))

    print(f"рынков к проверке: {len(cands)}", flush=True)

    # 2) стаканы батчами
    tok_of = {}
    for _, m, _ in cands:
        try:
            tok_of[m["conditionId"]] = json.loads(m["clobTokenIds"])[0]
        except (ValueError, IndexError, TypeError):
            pass
    toks = list(tok_of.values())
    books: dict[str, B.Book] = {}
    for i in range(0, len(toks), 100):
        chunk = toks[i:i + 100]
        try:
            for raw in api.books(chunk):
                bk = B.parse(raw)
                books[bk.asset_id] = bk
        except Exception as ex:                       # noqa: BLE001
            print(f"  стаканы {i}: {repr(ex)[:80]}", flush=True)
    print(f"стаканов получено: {len(books)}", flush=True)

    # 3) оценка
    out: list[Signal] = []
    for e, m, fair_peer in cands:
        tok = tok_of.get(m["conditionId"])
        bk = books.get(tok) if tok else None
        if not bk or not bk.asks or not bk.bids:
            continue
        mid, ask, bid = bk.mid, bk.best_ask, bk.best_bid
        if mid is None or not (0.001 < mid < 0.999):
            continue

        # рынок должен быть настоящим, иначе середина стакана — фикция
        spread = bk.spread or 1.0
        if bid < MIN_BID:
            continue
        if spread > MAX_SPREAD_ABS or spread / mid > MAX_SPREAD_REL:
            continue
        try:
            if float(m.get("volumeNum") or 0) < MIN_MARKET_VOL:
                continue
        except (TypeError, ValueError):
            continue

        depth = bk.ask_depth_usd(max_price=min(1.0, ask * 1.5))
        if depth < MIN_DEPTH_USD:
            continue

        rate = FEE_RATES.get(str(m.get("feeType") or "").replace("_fees", ""),
                             DEFAULT_FEE_RATE)
        dleft = days_left(m)
        # у рынка, закрывающегося через часы, цена уже отражает почти всё,
        # а аннуализация доходности теряет смысл
        if dleft is not None and dleft < MIN_DAYS_LEFT:
            continue
        name = (m.get("groupItemTitle") or m.get("question") or "")[:60]
        url = f"https://polymarket.com/event/{e.get('slug','')}"

        # --- оценки справедливой вероятности ---
        options = [("калибровка", float(sig(a + b * logit(mid))))]
        if fair_peer is not None:
            options.append(("сумма исходов = 1", fair_peer))

        for basis, fair in options:
            # ТЕЙКЕР: реальное исполнение нашего размера по стакану
            fill = B.buy_for_budget(bk, stake, rate)
            if fill.shares > 0 and np.isfinite(fill.eff_price):
                edge = fair - fill.eff_price
                if edge >= min_taker and fill.eff_price < 0.999:
                    ev = evaluate(fill.avg_price, fair, rate, dleft)
                    out.append(Signal(
                        event=e.get("title", "")[:60], outcome=name,
                        slug=e.get("slug", ""), kind="тейкер", basis=basis,
                        price=round(fill.eff_price, 4), fair=round(fair, 4),
                        edge=round(edge, 4), roi=round(ev.roi, 4),
                        kelly=round(ev.kelly, 4),
                        stake=round(min(stake, depth * 0.2), 2),
                        depth_usd=round(depth), spread=round(bk.spread or 0, 4),
                        days=dleft, url=url))

            # МЕЙКЕР: лимит на один тик выше лучшего бида, комиссии нет
            limit = round(bid + bk.tick, 4)
            if limit < ask and 0.001 <= limit <= 0.999:
                edge_m = fair - limit          # мейкер комиссию не платит
                if edge_m >= min_maker:
                    ev = evaluate(limit, fair, 0.0, dleft)
                    out.append(Signal(
                        event=e.get("title", "")[:60], outcome=name,
                        slug=e.get("slug", ""), kind="мейкер", basis=basis,
                        price=limit, fair=round(fair, 4), edge=round(edge_m, 4),
                        roi=round(ev.roi, 4), kelly=round(ev.kelly, 4),
                        stake=round(stake, 2), depth_usd=round(depth),
                        spread=round(bk.spread or 0, 4), days=dleft, url=url))

    # дедупликация: на один исход оставляем сильнейший сигнал
    best: dict[tuple, Signal] = {}
    for s in out:
        k = (s.slug, s.outcome, s.kind)
        if k not in best or s.edge > best[k].edge:
            best[k] = s
    res = sorted(best.values(), key=lambda s: -s.edge)
    return res


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="скринер рынков Polymarket")
    ap.add_argument("--stake", type=float, default=STAKE_USD,
                    help="размер ставки в долларах, от него считается исполнение")
    ap.add_argument("--min-edge", type=float, default=MIN_EDGE_TAKER,
                    help="минимальный чистый эдж тейкера в долях (0.03 = 3 п.п.)")
    ap.add_argument("--near", action="store_true",
                    help="показать лучшие кандидаты, даже не прошедшие порог")
    args = ap.parse_args()

    a, b = load_fit()
    print(f"калибровка: a={a:.4f} b={b:.4f}")
    api = Api(min_interval=0.04)
    lo_t = 0.0 if args.near else args.min_edge
    lo_m = 0.0 if args.near else max(args.min_edge, MIN_EDGE_MAKER)
    sigs = scan(api, a, b, stake=args.stake, min_taker=lo_t, min_maker=lo_m)
    if args.near:
        print("\nРЕЖИМ ДИАГНОСТИКИ: пороги сняты, ниже просто лучшее, что есть.")
        print("Положительный эдж здесь НЕ означает выгодную сделку.\n")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "signals.json").write_text(
        json.dumps([asdict(s) for s in sigs], ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\nнайдено сигналов: {len(sigs)}  ({api.n_calls} запросов)\n")
    for s in sigs[:20]:
        print(s.line())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
