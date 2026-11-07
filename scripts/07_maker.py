"""Мейкерское направление: вознаграждения за ликвидность против риска.

ДОХОД. Polymarket платит маркет-мейкерам из ежедневного пула. Баллы резидентной
заявки начисляются по официальной формуле:

    S(v, s) = ((v - s) / v)^2 * размер

  v — окно программы (rewards_max_spread, центы от середины)
  s — расстояние заявки от середины, центы; дальше v баллов нет вовсе

Баллы сторон считаются отдельно, затем:

    середина в [0.10, 0.90]:  Q = max(min(Q_bid, Q_ask), max(Q_bid, Q_ask)/3)
    середина вне этого:       Q = min(Q_bid, Q_ask)      # только двусторонние

Замер раз в минуту, пул делится пропорционально баллам. Конкуренцию не надо
угадывать: все резидентные заявки видны в стакане, знаменатель считается прямо.

РАСХОД. Без него расчёт бессмыслен. Ранняя версия этого скрипта показывала
«396% в день» — ровно потому, что считала только доход. Заявка в 4¢ от середины
на рынке со спредом 8¢ предлагает цену на 4¢ лучше рынка; её заберёт первый же
информированный участник. Пулы лежат неразобранными не по недосмотру, а потому
что их стережёт неблагоприятный отбор.

Модель расхода. Котируем обе стороны на расстоянии d от середины. За день
середина смещается на случайную величину со стандартным отклонением sigma.
Исполняют нас с той стороны, куда ушла цена, и убыток равен превышению хода
над d. Для нормального хода:

    E[(|X| - d)+] = 2 * ( sigma*phi(d/sigma) - d*(1 - Phi(d/sigma)) )

на одну долю в день. Sigma берётся из фактической истории цен рынка.

ЧЕГО МОДЕЛЬ ВСЁ РАВНО НЕ УЧИТЫВАЕТ:
  * Разложение стакана по участникам: формула берёт min по сторонам для каждого
    мейкера отдельно, а видны только суммарные уровни. Знаменатель приблизителен.
  * Реакцию конкурентов: встанешь близко к середине — подвинутся и они.
  * Скачки на новостях: нормальное приближение недооценивает редкие большие ходы,
    а именно они и разоряют мейкера.
Поэтому чистая доходность здесь — ВЕРХНЯЯ оценка, а не ожидание.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from math import erf, exp, pi, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmlib.api import Api, ApiError, CLOB, GAMMA  # noqa: E402
from pmlib import book as B  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data/out"

ONE_SIDED_DIVISOR = 3.0
EXTREME_LO, EXTREME_HI = 0.10, 0.90

MIN_DAYS_LEFT = 2.0        # рынку надо прожить достаточно, чтобы пул начислился
MIN_MARKET_LIQ = 500.0     # иначе снимок стакана не показателен
HISTORY_FOR_TOP = 200      # по скольким кандидатам тянуть историю цен ради sigma
WORKERS = 12


# ------------------------------------------------------------- математика

def _phi(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2 * pi)


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def adverse_per_share(sigma: float, d: float) -> float:
    """Ожидаемый убыток от неблагоприятного отбора на долю в день, в долларах.

    sigma и d — в единицах вероятности (0.01 = один цент).
    """
    if sigma <= 0:
        return 0.0
    if d <= 0:
        return sigma * sqrt(2.0 / pi)          # предел при d -> 0
    z = d / sigma
    return 2.0 * (sigma * _phi(z) - d * (1.0 - _Phi(z)))


def score(v_cents: float, s_cents: float, size: float) -> float:
    if v_cents <= 0 or s_cents < 0 or s_cents > v_cents:
        return 0.0
    return ((v_cents - s_cents) / v_cents) ** 2 * size


def book_side_score(levels, mid: float, v_cents: float, min_size: float) -> float:
    """Баллы одной стороны стакана. Заявки меньше минимального размера
    в программе не участвуют и в знаменатель не идут."""
    tot = 0.0
    for price, size in levels:
        if size < min_size:
            continue
        tot += score(v_cents, abs(mid - price) * 100.0, size)
    return tot


def q_min(q_bid: float, q_ask: float, mid: float) -> float:
    if EXTREME_LO <= mid <= EXTREME_HI:
        return max(min(q_bid, q_ask), max(q_bid, q_ask) / ONE_SIDED_DIVISOR)
    return min(q_bid, q_ask)


def days_left(m: dict) -> float | None:
    import datetime as dt
    s = m.get("endDate")
    if not s:
        return None
    try:
        end = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (end - dt.datetime.now(dt.UTC)).total_seconds() / 86400.0


# ------------------------------------------------------------- данные

def fetch_reward_configs(api: Api) -> dict[str, dict]:
    """Официальная конфигурация программы по всем рынкам (CLOB, с пагинацией)."""
    out, cursor = {}, ""
    while True:
        p = {"next_cursor": cursor} if cursor else {}
        j = api.get(f"{CLOB}/rewards/markets/current", p)
        rows = j.get("data") or []
        for r in rows:
            cid = r.get("condition_id")
            rate = float(r.get("total_daily_rate") or 0)
            if cid and rate > 0:
                out[cid] = {
                    "rate": rate,
                    "v": float(r.get("rewards_max_spread") or 0),
                    "min_size": float(r.get("rewards_min_size") or 0),
                }
        cursor = j.get("next_cursor") or ""
        if not rows or not cursor or cursor == "LTE=":
            break
    return out


def fetch_events(api: Api, min_liq: float = 1000.0) -> list[dict]:
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


def sigma_daily(api: Api, token: str) -> float | None:
    """Дневное стандартное отклонение цены рынка по фактической истории."""
    try:
        h = api.price_history(token, interval="1m", fidelity=60)
    except ApiError:
        return None
    if len(h) < 24:
        return None
    p = np.array([float(x["p"]) for x in h], dtype=float)
    t = np.array([int(x["t"]) for x in h], dtype=float)
    d = np.diff(p)
    dt_h = np.diff(t) / 3600.0
    ok = (dt_h > 0.5) & (dt_h < 3.0)          # только соседние часовые точки
    if ok.sum() < 12:
        return None
    per_hour = float(np.std(d[ok]))
    return per_hour * sqrt(24.0)


# ------------------------------------------------------------- оценка

@dataclass
class MakerOpp:
    event: str
    outcome: str
    mid: float
    spread: float
    daily_pool: float
    max_spread_c: float
    min_size: float
    two_sided: bool
    quote_at_c: float          # на каком расстоянии от середины стоим, центы
    capital: float
    share: float
    reward_day: float          # $ вознаграждения в день
    sigma_day: float | None    # дневная волатильность цены
    adverse_day: float | None  # $ ожидаемых потерь в день
    net_day: float | None
    net_yield: float | None    # чистая доходность на капитал, в день
    days_left: float
    url: str

    def line(self) -> str:
        req = "обе стороны" if self.two_sided else "одна сторона"
        s = (f"{self.event} → {self.outcome}\n"
             f"   середина {self.mid:.3f} | спред {self.spread:.3f} | пул "
             f"${self.daily_pool:.0f}/день | окно {self.max_spread_c:.1f}¢ | "
             f"мин. {self.min_size:.0f} долей ({req}) | до конца {self.days_left:.0f} дн.\n"
             f"   капитал ${self.capital:,.0f}, котировка в {self.quote_at_c:.1f}¢ от середины\n"
             f"   доход: доля {self.share*100:.1f}% = ${self.reward_day:.2f}/день")
        if self.sigma_day is None:
            return s + "\n   расход: волатильность неизвестна, оценка невозможна\n   " + self.url
        return (s + f"\n   расход: волатильность {self.sigma_day*100:.1f}¢/день → "
                f"отбор ${self.adverse_day:.2f}/день\n"
                f"   ИТОГО ${self.net_day:+.2f}/день = {self.net_yield*100:+.2f}% "
                f"на капитал в день\n   {self.url}")


def evaluate(e: dict, m: dict, bk: B.Book, cfg: dict,
             quote_at: float | None = None) -> MakerOpp | None:
    dl = days_left(m)
    if dl is None or dl < MIN_DAYS_LEFT:
        return None
    try:
        if float(m.get("liquidity") or 0) < MIN_MARKET_LIQ:
            return None
    except (TypeError, ValueError):
        return None

    v, min_size, rate = cfg["v"], cfg["min_size"], cfg["rate"]
    if v <= 0 or min_size <= 0:
        return None
    mid, bid, ask = bk.mid, bk.best_bid, bk.best_ask
    if mid is None or bid is None or ask is None or not (0.01 < mid < 0.99):
        return None

    two_sided = not (EXTREME_LO <= mid <= EXTREME_HI)
    half = (bk.spread or 0) / 2 * 100.0
    # по умолчанию встаём на лучшей цене рынка; ближе — больше баллов и риска
    d_c = quote_at if quote_at is not None else min(half, v)
    if d_c > v:
        return None

    qb = book_side_score(bk.bids, mid, v, min_size)
    qa = book_side_score(bk.asks, mid, v, min_size)
    book_q = q_min(qb, qa, mid)

    my_side = score(v, d_c, min_size)
    my_q = my_side if two_sided else my_side / ONE_SIDED_DIVISOR
    if my_q <= 0:
        return None

    share = my_q / (book_q + my_q)
    reward = rate * share

    d = d_c / 100.0
    p_buy = max(mid - d, 0.001)
    if two_sided:
        capital = min_size * (p_buy + max(1.0 - (mid + d), 0.001))
    else:
        capital = min_size * p_buy
    if capital <= 0:
        return None

    return MakerOpp(
        event=e.get("title", "")[:56],
        outcome=(m.get("groupItemTitle") or m.get("question") or "")[:46],
        mid=round(mid, 4), spread=round(bk.spread or 0, 4), daily_pool=rate,
        max_spread_c=v, min_size=min_size, two_sided=two_sided,
        quote_at_c=round(d_c, 2), capital=round(capital, 2), share=share,
        reward_day=round(reward, 4), sigma_day=None, adverse_day=None,
        net_day=None, net_yield=None, days_left=round(dl, 1),
        url=f"https://polymarket.com/event/{e.get('slug','')}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="анализ мейкерских вознаграждений")
    ap.add_argument("--budget", type=float, default=0.0, help="потолок капитала, $")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--quote-at", type=float, default=None,
                    help="расстояние котировки от середины в центах (по умолчанию — лучшая цена)")
    args = ap.parse_args()

    api = Api(min_interval=0.03)
    cfgs = fetch_reward_configs(api)
    print(f"рынков в программе вознаграждений: {len(cfgs)}", flush=True)
    print(f"суммарный дневной пул: ${sum(c['rate'] for c in cfgs.values()):,.0f}", flush=True)

    evs = fetch_events(api)
    print(f"активных событий: {len(evs)}", flush=True)

    tok_of = {}
    for e in evs:
        for m in (e.get("markets") or []):
            cid = m.get("conditionId")
            if m.get("closed") or not m.get("acceptingOrders") or cid not in cfgs:
                continue
            try:
                tok_of[json.loads(m["clobTokenIds"])[0]] = (e, m, cfgs[cid])
            except (KeyError, ValueError, IndexError, TypeError):
                pass
    print(f"рынков к оценке: {len(tok_of)}", flush=True)

    toks = list(tok_of)
    books = {}
    for i in range(0, len(toks), 100):
        try:
            for raw in api.books(toks[i:i + 100]):
                b = B.parse(raw)
                books[b.asset_id] = b
        except Exception as ex:                        # noqa: BLE001
            print(f"  стаканы {i}: {repr(ex)[:70]}", flush=True)

    opps = []
    for tok, (e, m, cfg) in tok_of.items():
        bk = books.get(tok)
        if bk:
            o = evaluate(e, m, bk, cfg, args.quote_at)
            if o:
                opps.append((tok, o))
    print(f"прошло фильтры: {len(opps)}", flush=True)

    if args.budget > 0:
        opps = [(t, o) for t, o in opps if o.capital <= args.budget]
        print(f"укладывается в ${args.budget:,.0f}: {len(opps)}", flush=True)

    # волатильность считаем только для самых доходных кандидатов: по запросу на рынок
    opps.sort(key=lambda p: -(p[1].reward_day / max(p[1].capital, 1e-9)))
    head = opps[:HISTORY_FOR_TOP]
    print(f"тяну историю цен по {len(head)} кандидатам для оценки риска...", flush=True)

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(sigma_daily, api, t): o for t, o in head}
        for fu in as_completed(futs):
            o = futs[fu]
            try:
                sg = fu.result()
            except Exception:                          # noqa: BLE001
                sg = None
            if sg is None:
                continue
            o.sigma_day = round(sg, 5)
            o.adverse_day = round(o.min_size * adverse_per_share(sg, o.quote_at_c / 100.0), 4)
            o.net_day = round(o.reward_day - o.adverse_day, 4)
            o.net_yield = o.net_day / o.capital

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "maker_opps.json").write_text(
        json.dumps([asdict(o) for _, o in opps], ensure_ascii=False, indent=1),
        encoding="utf-8")

    scored = [o for _, o in head if o.net_yield is not None]
    print(f"\nс посчитанным риском: {len(scored)}")
    if not scored:
        return 0
    pos = [o for o in scored if o.net_day > 0]
    print(f"из них с положительным итогом: {len(pos)}")

    # Ранжировать по проценту доходности бессмысленно: 100% крошечного пула
    # на крошечном капитале даёт заоблачный процент и три доллара в кармане.
    # Сортируем по абсолютным деньгам в день.
    pos.sort(key=lambda o: -o.net_day)
    print(f"суммарно по ним: ${sum(o.net_day for o in pos):.2f}/день "
          f"при капитале ${sum(o.capital for o in pos):,.0f}\n")
    print("ВНИМАНИЕ: это ВЕРХНЯЯ оценка. Доля пула взята из мгновенного снимка")
    print("стакана; как только встанешь в пустое окно, туда придут другие.\n")
    for o in pos[:args.top]:
        print(o.line())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
