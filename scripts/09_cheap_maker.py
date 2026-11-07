"""Мейкер на дешёвых исходах: лимитные заявки в диапазоне 2-15 центов.

Почему именно эта связка. Из исследования (см. README):
  * тейкером убыточно на любом уровне цены — спред и комиссия съедают всё;
  * мейкер комиссию не платит вовсе (feeSchedule.takerOnly = true);
  * калибровка 2026 года даёт на дешёвом крае положительную точечную оценку
    эджа (хотя и незначимую), а на фаворитах — значимо отрицательную;
  * заявка на 20 долей по 5 центов стоит доллар, то есть доступна при любом банке.

ГЛАВНАЯ ЛОВУШКА, ради которой написан этот скрипт.

Наивный расчёт: середина M = 0.075, бид 0.05, калибровка даёт справедливое
значение fair(M) = 0.080. Ставим лимит на 0.051 и радуемся эджу +2.9 п.п.

Так считать нельзя. Заявку по 0.05 исполнят, когда кто-то захочет продать
по 0.05 — а захотят обычно потому, что цена туда и идёт. К моменту исполнения
справедливая цена уже около самой заявки, и «захваченный спред» исчезает.
Это неблагоприятный отбор, и он не поправка на полях, а основная статья.

Честная модель — смесь двух видов встречного потока:

    EV = pi * (fair(M) - L)  +  (1 - pi) * (fair(L) - L)

  pi     — доля продавцов, выходящих не по информации (нетерпение, ребаланс,
           нужны деньги). Для них цена не изменилась, и спред достаётся нам.
  1 - pi — доля информированных. Для них рынок уже переехал на нашу цену,
           и всё, что остаётся, — калибровочный эдж в самой точке L.

pi снаружи не наблюдается. Поэтому скрипт не подставляет догадку, а считает
обе границы и ПОРОГ БЕЗУБЫТОЧНОСТИ по pi: какую долю неинформированного
потока нужно иметь, чтобы заявка вышла в ноль. Дальше решать человеку.

Отдельно: если рынок участвует в программе вознаграждений и односторонней
заявки достаточно (середина в [0.10, 0.90]), доход от неё добавляется.
На середине ниже 0.10 односторонняя заявка не получает ничего — таких
рынков большинство в нашем диапазоне, поэтому вознаграждения тут редкость.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pmlib.api import Api, ApiError, CLOB, GAMMA  # noqa: E402
from pmlib import book as B  # noqa: E402
from importlib import import_module  # noqa: E402

maker = import_module("07_maker")

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data/out"

PRICE_LO, PRICE_HI = 0.015, 0.16    # рабочий диапазон «дешёвых исходов»
MIN_DAYS_LEFT = 3.0
MIN_MARKET_VOL = 3000.0             # $ оборота: рынок должен быть живым
MIN_BID = 0.002                     # односторонний стакан — не рынок
# Спред должен быть узким ОТНОСИТЕЛЬНО цены. При bid 0.011 / ask 0.043
# «середина 0.027» — арифметическая фикция: реальных котировок нет, и любой
# расчёт эджа от такой середины выдаёт фантастические числа.
MAX_SPREAD_REL = 0.30
MIN_ASK_DEPTH = 50.0                # $ по ту сторону: есть с кем торговать
MIN_BID_DEPTH = 30.0                # $ на нашей стороне: очередь существует
# Насколько далеко от середины разрешено стоять. Чем дальше заявка, тем
# меньше шанс, что её исполнит случайный продавец, а не информированный:
# до дальней цены рынок доходит именно на новостях. Модель смеси этого
# не знает, поэтому ограничиваем расстояние явно.
MAX_DIST_FROM_MID_REL = 0.18


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_model() -> tuple[float, float, dict]:
    f = ROOT / "models/calibration.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    return float(d["a"]), float(d["b"]), d


class EdgeCurve:
    """Эдж калибровки с доверительными границами, интерполяция по сетке цен.

    Пессимистичный сценарий обязан опираться на НИЖНЮЮ границу, а не на
    точечную оценку. Точечная на всём дешёвом крае положительна, поэтому
    любой инструмент, построенный на ней, объявит все заявки безусловно
    выгодными — чего данные не показывают: интервал везде накрывает ноль.
    """

    def __init__(self, meta: dict):
        c = (meta.get("edge_curve") or {}).get("по_ценам") or {}
        if not c:
            raise RuntimeError("в models/calibration.json нет edge_curve — "
                               "пересчитай доверительные границы эджа")
        pts = sorted((float(k), v) for k, v in c.items())
        self.x = np.array([p for p, _ in pts])
        self.mean = np.array([v["edge"] for _, v in pts])
        self.lo = np.array([v["lo"] for _, v in pts])

    def edge(self, p: float) -> float:
        return float(np.interp(p, self.x, self.mean))

    def edge_lo(self, p: float) -> float:
        return float(np.interp(p, self.x, self.lo))


@dataclass
class Idea:
    event: str
    outcome: str
    mid: float
    bid: float
    ask: float
    limit: float              # цена нашей лимитной заявки
    shares: float
    capital: float
    fair_mid: float           # справедливо при нынешней середине
    fair_limit: float         # справедливо, если рынок переедет на нашу цену
    ev_best: float            # $ на сделку, если поток неинформированный
    ev_worst: float           # $ на сделку, если поток информированный
    pi_breakeven: float | None  # нужная доля неинформированного потока
    reward_day: float         # $/день из программы, 0 если не участвует
    days: float
    url: str

    def line(self) -> str:
        pb = ("в плюсе при любом потоке" if self.pi_breakeven is None
              else f"нужно {self.pi_breakeven*100:.0f}% неинформир. потока")
        rew = f" | вознаграждения ${self.reward_day:.2f}/день" if self.reward_day else ""
        return (f"{self.event} → {self.outcome}\n"
                f"   стакан {self.bid:.3f} / {self.ask:.3f} (середина {self.mid:.3f}), "
                f"до конца {self.days:.0f} дн.{rew}\n"
                f"   ЗАЯВКА: купить {self.shares:.0f} долей по {self.limit:.3f} "
                f"= ${self.capital:.2f}\n"
                f"   если поток случайный: {self.ev_best:+.3f}$ | "
                f"если информированный: {self.ev_worst:+.3f}$ | {pb}\n"
                f"   {self.url}")


def evaluate(e, m, bk: B.Book, curve: "EdgeCurve", budget: float,
             cfg: dict | None) -> Idea | None:
    mid, bid, ask = bk.mid, bk.best_bid, bk.best_ask
    if mid is None or bid is None or ask is None:
        return None
    if not (PRICE_LO <= mid <= PRICE_HI) or bid < MIN_BID:
        return None
    spread = bk.spread or 0.0
    if spread <= 0 or spread / mid > MAX_SPREAD_REL:
        return None
    if bk.ask_depth_usd() < MIN_ASK_DEPTH:
        return None
    if sum(pr * sz for pr, sz in bk.bids) < MIN_BID_DEPTH:
        return None
    try:
        if float(m.get("volumeNum") or 0) < MIN_MARKET_VOL:
            return None
    except (TypeError, ValueError):
        return None
    dl = maker.days_left(m)
    if dl is None or dl < MIN_DAYS_LEFT:
        return None

    tick = bk.tick or 0.001
    limit = round(bid + tick, 4)          # встаём перед очередью на лучшем биде
    if limit >= ask or limit <= 0:
        return None
    if (mid - limit) / mid > MAX_DIST_FROM_MID_REL:
        return None

    # Оптимистично: продавец вышел не по информации, справедливая цена
    # осталась на середине — точечная оценка калибровки там.
    fair_mid = mid + curve.edge(mid)
    ev_best = fair_mid - limit
    # Пессимистично: рынок переехал на нашу цену, и всё, что остаётся, —
    # калибровочный эдж в самой точке L, взятый по НИЖНЕЙ границе.
    fair_lim = limit + curve.edge_lo(limit)
    ev_worst = fair_lim - limit
    if ev_best <= 0:
        return None                       # даже в лучшем случае минус

    # порог по доле неинформированного потока: pi*ev_best + (1-pi)*ev_worst = 0
    if ev_worst >= 0:
        pi_be = None                      # в плюсе при любом составе потока
    else:
        denom = ev_best - ev_worst
        pi_be = (-ev_worst) / denom if denom > 0 else None
        if pi_be is None or pi_be >= 1.0:
            return None

    # размер: заполняем бюджет целиком, но не меньше минимального ордера
    min_sh = float(m.get("orderMinSize") or 5.0)
    shares = max(min_sh, budget / limit)
    if shares * limit > budget * 1.01:
        shares = budget / limit
    if shares < min_sh:
        return None
    capital = shares * limit

    reward = 0.0
    if cfg and cfg.get("rate", 0) > 0 and mid >= 0.10:
        # односторонняя заявка получает треть баллов и только если середина
        # не ниже 0.10; ниже — программа требует обе стороны
        v, ms = cfg["v"], cfg["min_size"]
        if shares >= ms:
            s_c = abs(mid - limit) * 100.0
            my = maker.score(v, s_c, shares) / maker.ONE_SIDED_DIVISOR
            qb = maker.book_side_score(bk.bids, mid, v, ms)
            qa = maker.book_side_score(bk.asks, mid, v, ms)
            bq = maker.q_min(qb, qa, mid)
            if my > 0:
                reward = cfg["rate"] * my / (bq + my)

    return Idea(
        event=e.get("title", "")[:56],
        outcome=(m.get("groupItemTitle") or m.get("question") or "")[:46],
        mid=round(mid, 4), bid=round(bid, 4), ask=round(ask, 4),
        limit=limit, shares=round(shares, 1), capital=round(capital, 2),
        fair_mid=round(fair_mid, 4), fair_limit=round(fair_lim, 4),
        ev_best=round(ev_best * shares, 4), ev_worst=round(ev_worst * shares, 4),
        pi_breakeven=pi_be, reward_day=round(reward, 3), days=round(dl, 1),
        url=f"https://polymarket.com/event/{e.get('slug','')}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="мейкер на дешёвых исходах")
    ap.add_argument("--budget", type=float, default=15.0, help="потолок на одну заявку, $")
    ap.add_argument("--max-pi", type=float, default=0.60,
                    help="не показывать, если нужная доля неинформированного потока выше")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    a, b, meta = load_model()
    curve = EdgeCurve(meta)
    print(f"модель: {meta.get('fitted_window')}, a={a} b={b}")
    print("пессимистичный сценарий считается по НИЖНЕЙ доверительной границе эджа")
    print(f"диапазон цен: {PRICE_LO}-{PRICE_HI}, бюджет на заявку ${args.budget}\n")

    api = Api(min_interval=0.03)
    try:
        cfgs = maker.fetch_reward_configs(api)
    except ApiError:
        cfgs = {}
    evs = maker.fetch_events(api)
    print(f"событий: {len(evs)}", flush=True)

    tok_of = {}
    for e in evs:
        for m in (e.get("markets") or []):
            if m.get("closed") or not m.get("acceptingOrders"):
                continue
            try:
                tok_of[json.loads(m["clobTokenIds"])[0]] = (e, m)
            except (KeyError, ValueError, IndexError, TypeError):
                pass

    toks = list(tok_of)
    books = {}
    for i in range(0, len(toks), 100):
        try:
            for raw in api.books(toks[i:i + 100]):
                bk = B.parse(raw)
                books[bk.asset_id] = bk
        except Exception as ex:                       # noqa: BLE001
            print(f"  стаканы {i}: {repr(ex)[:70]}", flush=True)
    print(f"стаканов: {len(books)}", flush=True)

    ideas = []
    for tok, (e, m) in tok_of.items():
        bk = books.get(tok)
        if not bk:
            continue
        o = evaluate(e, m, bk, curve, args.budget, cfgs.get(m.get("conditionId")))
        if o and (o.pi_breakeven is None or o.pi_breakeven <= args.max_pi):
            ideas.append(o)

    ideas.sort(key=lambda o: (o.pi_breakeven if o.pi_breakeven is not None else -1.0))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "cheap_maker.json").write_text(
        json.dumps([asdict(o) for o in ideas], ensure_ascii=False, indent=1),
        encoding="utf-8")

    print(f"\nнайдено заявок: {len(ideas)}")
    if not ideas:
        print("ничего не проходит фильтры — это нормальный исход")
        return 0
    safe = [o for o in ideas if o.pi_breakeven is None]
    print(f"из них выгодных при ЛЮБОМ составе встречного потока: {len(safe)}\n")
    for o in ideas[:args.top]:
        print(o.line())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
