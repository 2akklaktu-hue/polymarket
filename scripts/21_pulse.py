"""Монитор резких движений: ищет скачки цены и присылает разбор в Telegram.

Что присылает: сам скачок (откуда куда и за сколько), объём, реальную цену
входа на НЕТ, крупные сделки, свежие комментарии из обсуждения и — главное —
честную справку о том, что об этом говорит история. Решение остаётся за человеком.

Почему справка честная, а не «бери НЕТ». Проверка на 2589 скачках 2023-2026
(README, раздел 6) не подтвердила эдж ни в одну сторону: цена входа в истории
известна только как средняя за часы скачка, а при входе по доступной цене эдж
исчезает. Зато объём оказался информативным в обратную сторону: чем он больше,
тем чаще движение ПРОДОЛЖАЕТСЯ. Это и сообщается.

Два источника скачка:
  1. oneDayPriceChange из gamma — работает с первого запуска, окно ровно сутки;
  2. собственные снимки (scripts/20_snapshot.py) — окно любое, точность до
     интервала cron. Используются, когда накопились.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmlib.api import Api, ApiError, GAMMA, DATA  # noqa: E402
from pmlib import snap  # noqa: E402
from pmlib.econ import DEFAULT_FEE_RATE, taker_fee_per_share  # noqa: E402
from pmlib.notify import Telegram, esc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SENT = ROOT / "data/out/pulse_sent.json"

# --- что считать скачком -------------------------------------------------
T_LO = 0.15            # цена была не выше
T_HI = 0.30            # и дошла не ниже
WINDOW_H = 24.0        # окно по снимкам (для oneDayPriceChange оно всегда 24ч)
MIN_VOL24 = 3000.0     # $ оборота за сутки: ниже — просто неликвид
MIN_HOURS_LEFT = 24.0  # должно остаться время до планового конца
MAX_SPREAD = 0.06      # шире — войти по разумной цене нельзя
MAX_ASK = 0.90         # дороже 90c ставка на НЕТ становится лотереей

# --- повторные отправки --------------------------------------------------
REPEAT_AFTER_H = 12.0   # не присылать тот же рынок чаще
REPEAT_IF_MOVED = 0.08  # ...если цена не сдвинулась ещё на столько

BIG_TRADE_USD = 500.0
PAGE = 100
MAX_OFFSET = 5000

_HIGHFREQ = re.compile(
    r"(?:up or down)|(?:\b\d{1,2}:\d{2}\s*(?:am|pm)\b)"
    r"|(?:\b\d{1,2}(?:am|pm)\s*-\s*\d{1,2}(?:am|pm)\b)"
    r"|(?:\bhourly\b)|(?:\bevery \d+ minutes?\b)", re.I)


def _f(x, default=float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _end_ts(s) -> float:
    if not s:
        return float("nan")
    try:
        return float(time.mktime(time.strptime(str(s)[:19], "%Y-%m-%dT%H:%M:%S"))
                     - time.timezone)
    except ValueError:
        return float("nan")


def fetch_markets(api: Api) -> list[dict]:
    """Живые рынки в порядке убывания СУТОЧНОГО оборота.

    Сортировка по volume24hr, а не по общему объёму: нас интересует то, где
    торгуют сейчас. Глубже offset ~2100 gamma отвечает 422 — это нормально.
    """
    out: list[dict] = []
    off = 0
    while off < MAX_OFFSET:
        try:
            batch = api.get(f"{GAMMA}/markets",
                            {"limit": PAGE, "offset": off, "closed": "false",
                             "active": "true", "order": "volume24hr",
                             "ascending": "false"})
        except ApiError:
            break
        if not batch:
            break
        out.extend(batch)
        off += PAGE
        if len(batch) < PAGE:
            break
    return out


def snapshot_moves(hours: float) -> dict[str, tuple[float, float]]:
    """По снимкам: cid -> (минимальная середина за окно, часов назад).

    Пусто, если снимков ещё нет — тогда работает только суточный источник.
    """
    d = snap.snapshot_load(hours=hours)
    if d.empty:
        return {}
    reg = snap.registry_frame()
    if reg.empty:
        return {}
    d = d.dropna(subset=["bid", "ask"])
    if d.empty:
        return {}
    d["mid"] = (d["bid"] + d["ask"]) / 2.0
    idx = d.groupby("id")["mid"].idxmin()
    lo = d.loc[idx, ["id", "mid", "ts"]]
    now = time.time()
    by_id = {int(r.id): (float(r.mid), (now - float(r.ts)) / 3600.0)
             for r in lo.itertuples() if r.id == r.id}
    out: dict[str, tuple[float, float]] = {}
    for r in reg.itertuples():
        v = by_id.get(int(r.id))
        if v:
            out[str(r.cid)] = v
    return out


def big_trades(api: Api, cid: str, hours: float = 24.0,
               n: int = 3) -> list[dict]:
    """Крупные сделки ЗА ВРЕМЯ ДВИЖЕНИЯ — чем оно объясняется в деньгах.

    Фильтр по времени обязателен: лента отдаётся от новых к старым за всю
    жизнь рынка, и без отсечки в сообщение попадают сделки по ценам, которых
    на рынке давно нет.

    Сторона приводится к ставке: продажа НЕТ по 0.93 — это ставка на ДА по
    7 центам, и писать её как «SELL No 0.93» значит путать читателя.
    """
    try:
        j = api.get(f"{DATA}/v2/trades",
                    {"condition": cid, "limit": 200, "filter_type": "CASH",
                     "filter_amount": BIG_TRADE_USD})
    except ApiError:
        return []
    since = time.time() - hours * 3600.0
    out = []
    for r in (j.get("data") or []):
        ts = _f(r.get("timestamp"), 0.0)
        if ts < since:
            continue
        px = _f(r.get("price"), 0.0)
        is_yes_token = str(r.get("outcome") or "").strip().lower() == "yes"
        buying = str(r.get("side") or "").upper() == "BUY"
        # ставка на ДА: купил токен ДА либо продал токен НЕТ
        on_yes = (buying == is_yes_token)
        out.append({
            "usd": _f(r.get("size"), 0.0) * px,
            "side_ru": "на ДА" if on_yes else "на НЕТ",
            "price": px if is_yes_token == on_yes else 1.0 - px,
            "who": str(r.get("pseudonym") or r.get("name") or "кто-то"),
            "ts": ts,
        })
    out.sort(key=lambda r: r["usd"], reverse=True)
    return out[:n]


# Боты-трекеры засоряют обсуждение отчётами о чужих позициях. Для вопроса
# «на каком слухе выросло» они бесполезны.
_BOT_NOISE = re.compile(r"0x[0-9a-fA-F]{12,}|unrealized profit|holds [\d,.]+ *usd",
                        re.I)


def comments(api: Api, event_id: str | None, hours: float = 48.0,
             n: int = 3) -> list[dict]:
    """Свежие комментарии события — там причину называют словами."""
    if not event_id:
        return []
    try:
        j = api.get(f"{GAMMA}/comments",
                    {"parent_entity_type": "Event",
                     "parent_entity_id": int(event_id), "limit": 40,
                     "order": "createdAt", "ascending": "false"})
    except (ApiError, TypeError, ValueError):
        return []
    since = time.time() - hours * 3600.0
    out = []
    for c in (j if isinstance(j, list) else []):
        body = re.sub(r"\s+", " ", str(c.get("body") or "")).strip()
        if len(body) < 15 or _BOT_NOISE.search(body):
            continue
        ts = _end_ts(c.get("createdAt"))
        if ts == ts and ts < since:
            continue
        out.append({"body": body, "ts": ts})
        if len(out) >= n:
            break
    return out


def verdict(vol24: float) -> str:
    """Что говорит история при таком объёме. Без обещаний — их нет.

    Пороги пересчитаны: в бэктесте объём мерился за 72 часа, здесь приходит
    суточный, поэтому границы $50k и $2k за трое суток дают примерно $16.7k
    и $0.7k в сутки.
    """
    if vol24 >= 16700:
        return ("объём большой. На истории при таком объёме цена продолжала "
                "движение в 63% случаев, а ставка против него теряла 45%. "
                "Против такого движения исторически играть было убыточно.")
    if vol24 < 700:
        return ("объём мизерный. На истории такие движения чаще откатывались "
                "(НЕТ побеждал в 72%), но вход там дорогой из-за спреда, и "
                "заметного плюса это не давало.")
    return ("объём средний. Ровно в этой зоне история не даёт перевеса "
            "ни одной стороне.")


def build_message(m: dict, p_before: float, mid: float, win_h: float,
                  src: str, trades: list[dict], cmts: list[dict]) -> str:
    q = str(m.get("question") or "")
    bid, ask = _f(m.get("bestBid")), _f(m.get("bestAsk"))
    vol24 = _f(m.get("volume24hr"), 0.0)
    vol_all = _f(m.get("volumeNum"), 0.0)
    left_d = (_end_ts(m.get("endDate")) - time.time()) / 86400.0
    # купить НЕТ по рынку значит продать ДА: цена НЕТ = 1 - лучший бид по ДА
    no_px = 1.0 - bid
    no_fee = taker_fee_per_share(no_px, DEFAULT_FEE_RATE)
    ev = (m.get("events") or [{}])[0]
    ev_slug = ev.get("slug") or m.get("slug")
    url = f"https://polymarket.com/event/{ev_slug}"

    L = [f"⚡ <b>{esc(q)}</b>", ""]
    L.append(f"ДА: <b>{p_before * 100:.0f}¢ → {mid * 100:.0f}¢</b> "
             f"({(mid - p_before) * 100:+.0f} п.п. за {win_h:.0f}ч, {esc(src)})")
    L.append(f"Объём: <b>${vol24:,.0f}</b> за сутки (всего ${vol_all:,.0f})")
    L.append(f"Стакан: бид {bid:.3f} / аск {ask:.3f}, "
             f"спред {(ask - bid) * 100:.1f}¢")
    L.append(f"Вход на НЕТ: <b>{no_px:.3f}</b> + комиссия {no_fee:.3f} "
             f"= <b>{no_px + no_fee:.3f}</b>")
    if left_d == left_d:
        L.append(f"До планового конца: {left_d:.0f} дн.")

    if trades:
        L.append("")
        L.append("<b>Крупные сделки за сутки:</b>")
        for t in trades:
            L.append(f"• ставка <b>{t['side_ru']}</b> ${t['usd']:,.0f} "
                     f"по {t['price']:.3f} — {esc(t['who'])}")
    if cmts:
        L.append("")
        L.append("<b>Из обсуждения:</b>")
        for c in cmts:
            L.append(f"• {esc(c['body'][:160])}")
    L.append("")
    L.append("<b>Что говорят данные:</b>")
    L.append(esc("Проверка на 2589 скачках 2023-2026 не подтвердила перевес "
                 "ни одной из сторон. Сейчас " + verdict(vol24)))
    L.append("")
    L.append(f'<a href="{url}">Открыть рынок</a>')
    L.append("<i>Это не рекомендация. Решение за тобой.</i>")
    return "\n".join(L)


def load_sent() -> dict:
    if not SENT.exists():
        return {}
    try:
        return json.loads(SENT.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_sent(d: dict) -> None:
    SENT.parent.mkdir(parents=True, exist_ok=True)
    SENT.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="ничего не отправлять, только показать в консоли")
    ap.add_argument("--top", type=int, default=5, help="сколько событий максимум")
    ap.add_argument("--min-jump", type=float, default=T_HI - T_LO,
                    help="минимальный скачок в долях вероятности")
    a = ap.parse_args()

    now = time.time()
    api = Api(min_interval=0.05)
    raw = fetch_markets(api)
    print(f"живых рынков: {len(raw)} ({api.n_calls} запросов)", flush=True)
    moves = snapshot_moves(WINDOW_H)
    print(f"рынков со своей историей снимков: {len(moves)}", flush=True)

    found = []
    for m in raw:
        cid = m.get("conditionId")
        q = str(m.get("question") or "")
        if not cid or _HIGHFREQ.search(q) or not m.get("enableOrderBook"):
            continue
        bid, ask = _f(m.get("bestBid")), _f(m.get("bestAsk"))
        if not (bid > 0.002 and ask < MAX_ASK and (ask - bid) <= MAX_SPREAD):
            continue
        vol24 = _f(m.get("volume24hr"), 0.0)
        if vol24 < MIN_VOL24:
            continue
        left_h = (_end_ts(m.get("endDate")) - now) / 3600.0
        if left_h == left_h and left_h < MIN_HOURS_LEFT:
            continue

        mid = (bid + ask) / 2.0
        if mid < T_HI:
            continue

        cand = []
        snp = moves.get(str(cid))                     # источник 1: свои снимки
        if snp and snp[0] <= T_LO:
            cand.append((snp[0], max(snp[1], 0.1), "по моим снимкам"))
        chg = _f(m.get("oneDayPriceChange"), float("nan"))   # источник 2: сутки
        last = _f(m.get("lastTradePrice"), mid)
        if chg == chg:
            p24 = last - chg
            if p24 <= T_LO:
                cand.append((p24, 24.0, "за сутки по данным площадки"))
        if not cand:
            continue
        p_before, win_h, src = min(cand, key=lambda c: c[0])
        if (mid - p_before) < a.min_jump:
            continue
        found.append((mid - p_before, m, p_before, mid, win_h, src, vol24))

    found.sort(key=lambda x: -x[0])
    print(f"скачков найдено: {len(found)}", flush=True)

    sent = load_sent()
    tg = None if a.dry else Telegram()
    n_out = 0
    for jump, m, p_before, mid, win_h, src, vol24 in found:
        if n_out >= a.top:
            break
        cid = str(m.get("conditionId"))
        prev = sent.get(cid) or {}
        age_h = (now - float(prev.get("ts") or 0)) / 3600.0
        if prev and age_h < REPEAT_AFTER_H and \
                abs(mid - float(prev.get("mid") or 0)) < REPEAT_IF_MOVED:
            continue
        trades = big_trades(api, cid)
        cmts = comments(api, (m.get("events") or [{}])[0].get("id"))
        text = build_message(m, p_before, mid, win_h, src, trades, cmts)
        if a.dry:
            print("\n" + "-" * 70)
            print(re.sub(r"<[^>]+>", "", text))
        else:
            tg.send(text)
        sent[cid] = {"ts": now, "mid": round(mid, 4)}
        n_out += 1

    if not a.dry:
        save_sent(sent)
    print(f"\nотправлено: {n_out}"
          f"{' (пробный прогон, ничего не послано)' if a.dry else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
