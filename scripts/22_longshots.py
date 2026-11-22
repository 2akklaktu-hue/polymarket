"""Дешёвые исходы, у которых есть причина стоить дороже.

Что ищет. Исходы по 3-15 центов, про которые что-то ВНЕ цены Polymarket
говорит, что шанс выше:

  1. новости   — свежие заголовки по теме, оценённые ИИ (pmlib/judge.py);
  2. площадки  — тот же исход на Kalshi или Manifold стоит заметно дороже;
  3. игроки    — дешёвую сторону покупают кошельки с прибыльной историей.

Что присылает.
  * Сигнал — сразу, в том же прогоне: сильный (согласны два источника из
    трёх) и средний (один источник).
  * Поправку к нему — тоже сразу: средний стал сильным, либо сигнал снят
    (подтверждение пропало, цена ушла из диапазона, рынок закрылся).
  * Сводку раз в сутки — до трёх действующих сигналов и итог по прошлым:
    сколько закрылось и сколько выиграло.

Чего НЕ обещает. Исход за 5 центов не может иметь «высокий шанс»: на 23 тысячах
закрытых рынков такие исходы сбывались в 4-5% случаев (README, раздел 2).
Лучшее, что здесь можно найти, — исход, у которого шанс 10-15% вместо 5%.
Восемь-девять таких ставок из десяти всё равно проиграют. Поэтому в каждом
сообщении это сказано прямо, а журнал data/live/signals.csv копит факты,
по которым видно, работает ли отбор вообще.

Сколько работы за прогон. Кандидатов около двухсот, а запросы и ИИ не
бесплатны, поэтому отбор идёт воронкой:
  все дешёвые -> ~16 самых «оживших» (рост цены, всплеск оборота)
              -> по ним собираются новости, площадки, игроки
              -> лучшие уходят на оценку ИИ: до 3 за прогон и не больше
                 PM_AI_PER_DAY в сутки, равномерно по часам.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import sys
import time
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmlib import judge, sources  # noqa: E402
from pmlib.api import Api, ApiError, GAMMA  # noqa: E402
from pmlib.econ import DEFAULT_FEE_RATE, evaluate, taker_fee_per_share  # noqa: E402
from pmlib.notify import Telegram, esc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
pulse = import_module("21_pulse")      # общий код: список рынков, разбор дат
_f, _end_ts = pulse._f, pulse._end_ts

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data/out/longshots_state.json"
LOG = ROOT / "data/live/signals.csv"
LOG_COLS = ["ts", "kind", "strength", "cid", "side", "cost", "mkt", "p_ai",
            "p_est", "sources", "question", "result"]

# --- что считать дешёвым исходом ----------------------------------------
P_LO, P_HI = 0.03, 0.15   # цена входа; ниже 3c переплата на входе 20-70%
MAX_SPREAD = 0.02         # абсолютный спред
MAX_SPREAD_REL = 0.35     # и относительно цены: 2c спреда при цене 4c — много
MIN_VOL24 = 1000.0        # $ оборота за сутки
MIN_DAYS, MAX_DAYS = 1.0, 60.0
# Здесь слухи цену не двигают: спорт считают букмекеры, курс и погоду — модели.
SKIP_CATS = ("sports", "crypto", "weather")
_PRICE_LEVEL = re.compile(
    r"(?:\b(?:close|closes|finish|finishes|settle|settles)\s+(?:above|below|at)\b)"
    # «(LOW)»/«(HIGH)» между глаголом и суммой: «Gold (GC) dip to (LOW) $3,000»
    r"|(?:\b(?:dip|drop|fall|rise|reach|hit)s?\s+(?:to\s+)?(?:\([^)]*\)\s*)?\$)"
    r"|(?:\b(?:above|below|over|under)\s+\$[\d,.]+)"
    r"|(?:\bFDV\b)|(?:\bmarket cap\b)|(?:\ball[- ]time high\b)"
    # счётчики и уровни: число твитов, доходность облигаций — та же история,
    # что с курсами: исход решает цифра, а не новость
    r"|(?:\btweets?\b)|(?:\byield\b)"
    # число постов: «post 20-39 Truth Social posts», «200+ posts». Нужен
    # диапазон или «+» перед словом — просто «Will Trump post about…» проходит
    r"|(?:\b\d[\d,]*\s*(?:[-–]\s*\d[\d,]*|\+)\s+(?:[\w']+\s+){0,3}posts?\b)"
    # чужие шансы: «Ossoff's nomination odds dip to 10%» — исход решает цена
    # другого рынка, а не новость
    r"|(?:\b(?:odds|chances?)\b[^?]{0,40}?\b(?:dip|drop|fall|rise|reach|hit|"
    r"climb|exceed|surge)\w*\s+(?:to\s+|above\s+|below\s+)?\d[\d.]*\s*%)", re.I)

# --- сколько работы за прогон --------------------------------------------
N_EVIDENCE = 12           # самых «оживших» — на сбор подтверждений
N_ROTATE = 4              # плюс столько давно не проверявшихся: за пару суток
                          # через сбор проходит весь список, а не одна верхушка
EV_TTL_H = 3.0            # подтверждения свежее этого не пересобираются
EV_KEEP_H = 36.0          # старше — в сводку не идут
NEWS_FRESH_H = 72.0       # заголовки старше в сообщение не попадают
AI_TTL_H = 24.0           # оценка ИИ живёт сутки...
AI_MOVED = 0.25           # ...если цена не ушла на четверть
AI_STALE_H = 72.0         # через трое суток пересчитывается в любом случае


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


AI_PER_RUN = _env_int("PM_AI_PER_RUN", 3)
# потолок зависит от провайдера: у Claude его держит цена (6 в сутки),
# у бесплатного OpenRouter — 50 запросов в сутки (16 оценок с запасом на повторы)
AI_PER_DAY = _env_int("PM_AI_PER_DAY", judge.default_per_day())
AI_TIME_S = 360.0         # дольше на ИИ за прогон не тратим: бесплатные модели
                          # бывают медленными, а у задания в Actions 15 минут
DIGEST_HOUR_UTC = _env_int("PM_DIGEST_HOUR_UTC", 9)
DIGEST_TOP = 3

# --- что слать сразу, не дожидаясь сводки --------------------------------
ALERT_LEVEL = 1           # с какой силы: 1 — средние и сильные, 2 — только сильные
MISS_RUNS = 2             # сигнала нет столько прогонов подряд — сообщаем «снят».
                          # Один прогон без подтверждения бывает случайным:
                          # площадка не ответила, цена качнулась через порог
COOLDOWN_H = 12.0         # снятый сигнал не объявляется заново раньше этого,
                          # иначе на границе порога бот дёргался бы туда-сюда
WATCH_MOVED = 0.15        # цена присланного сигнала ушла на 15% — ИИ пересчитывает

# --- когда источник считается «за» ---------------------------------------
AI_RATIO = 1.5            # оценка ИИ не ниже полутора цен входа
EST_RATIO = 1.2           # и после поправки к рынку остаётся запас 20%
AI_OUTLIER_RATIO = 4.0    # ИИ выше рынка вчетверо...
AI_OUTLIER_GAP = 0.20     # ...и на 20 п.п. — в одиночку такое не засчитывается
VENUE_RATIO = 1.3         # другая площадка даёт на 30% больше цены входа
VENUE_LOWER = 0.75        # а если на четверть меньше здешней — это предупреждение
VENUE_SIM = 0.6           # без ИИ: рынок «тот же», если слова совпали на 60%
VENUE_DAYS = 3.0          # ...и срок отличается не больше чем на три дня
VENUE_MAX_RATIO = 3.0     # ...и цена там не выше здешней втрое
VENUE_MAX_GAP = 0.10      # (или на 10 п.п. — что больше): иначе вопрос не тот
# Подтверждением считается только площадка с настоящими деньгами.
# На Manifold деньги игровые: его цена показывается, но в счёт не идёт.
REAL_MONEY = ("Kalshi",)
# Насколько верить ИИ против рынка. Рынок в среднем точнее, поэтому оценка
# модели не подменяет цену, а сдвигает её на долю расстояния.
SHRINK = {"low": 0.25, "medium": 0.5, "high": 0.7}

STRENGTH_RU = {2: "сильный", 1: "средний", 0: "слабый"}


# ------------------------------------------------------------------ отбор

def category(m: dict) -> str:
    """Категория по feeType площадки: sports_fees_v3 -> sports."""
    mt = re.match(r"[a-z]+", str(m.get("feeType") or ""))
    return mt.group(0) if mt else "other"


def _fee_rate(m: dict) -> float:
    fs = m.get("feeSchedule")
    if m.get("feesEnabled") is False or not isinstance(fs, dict):
        return 0.0
    return _f(fs.get("rate"), DEFAULT_FEE_RATE)


def _is_yes_no(m: dict) -> bool:
    try:
        o = json.loads(m.get("outcomes") or "[]")
    except (TypeError, ValueError):
        return False
    return [str(x).strip().lower() for x in o] == ["yes", "no"]


def select(raw: list[dict], now: float) -> list[dict]:
    """Дешёвые стороны живых рынков. У рынка в диапазон попадает одна сторона."""
    # Остальные исходы того же события — подсказка ИИ, как рынок понимает
    # вопрос. 22.11 модель дала 80% исходу «ровно одно повышение ставки»,
    # хотя по соседним исходам видно, что рынок ждёт два.
    by_event: dict[str, list[tuple[str, str, float]]] = {}
    for m in raw:
        eid = str(((m.get("events") or [{}])[0]).get("id") or "")
        b, s = _f(m.get("bestBid")), _f(m.get("bestAsk"))
        # у мёртвого стакана бид 0.001 и аск 0.999: середина 0.5 ничего не значит
        mid = (b + s) / 2.0 if b > 0.002 and s < 0.998 else _f(m.get("lastTradePrice"))
        if eid and mid == mid:
            by_event.setdefault(eid, []).append(
                (str(m.get("conditionId")),
                 str(m.get("groupItemTitle") or m.get("question") or ""), mid))
    out = []
    for m in raw:
        cid, q = m.get("conditionId"), str(m.get("question") or "")
        if not cid or not m.get("enableOrderBook") or m.get("acceptingOrders") is False:
            continue
        if pulse._HIGHFREQ.search(q) or _PRICE_LEVEL.search(q):
            continue
        cat = category(m)
        if cat in SKIP_CATS or m.get("sportsMarketType") or not _is_yes_no(m):
            continue
        bid, ask = _f(m.get("bestBid")), _f(m.get("bestAsk"))
        if not (bid > 0.002 and ask < 0.998):
            continue
        vol24 = _f(m.get("volume24hr"), 0.0)
        end = str(m.get("endDate") or "")
        days = (_end_ts(end) - now) / 86400.0
        if vol24 < MIN_VOL24 or not (MIN_DAYS <= days <= MAX_DAYS):
            continue
        spread, mid = ask - bid, (bid + ask) / 2.0
        for yes in (True, False):
            # купить НЕТ значит продать ДА: цена НЕТ = 1 - лучший бид по ДА
            px = ask if yes else 1.0 - bid
            if not (P_LO <= px <= P_HI):
                continue
            if spread > MAX_SPREAD or spread / px > MAX_SPREAD_REL:
                continue
            rate = _fee_rate(m)
            chg1 = _f(m.get("oneDayPriceChange"), 0.0)
            chg7 = _f(m.get("oneWeekPriceChange"), 0.0)
            ev = (m.get("events") or [{}])[0]
            slug, ev_slug = m.get("slug") or "", ev.get("slug") or ""
            c = {
                "cid": str(cid), "q": q, "yes": yes,
                "side": "ДА" if yes else "НЕТ",
                "key": f"{cid}|{'YES' if yes else 'NO'}",
                "px": px, "rate": rate,
                "cost": px + taker_fee_per_share(px, rate),
                "mkt": mid if yes else 1.0 - mid,
                "d1": chg1 if yes else -chg1,
                "d7": chg7 if yes else -chg7,
                "vol24": vol24, "vol1wk": _f(m.get("volume1wk"), 0.0),
                "days": days, "end": end, "cat": cat,
                "desc": str(m.get("description") or ""),
                "event_title": str(ev.get("title") or ""),
                "siblings": [(lbl, p) for k, lbl, p in sorted(
                    by_event.get(str(ev.get("id") or ""), []), key=lambda x: -x[2])
                    if k != str(cid)][:12],
                "url": (f"https://polymarket.com/event/{ev_slug}/{slug}"
                        if ev_slug and slug else
                        f"https://polymarket.com/event/{ev_slug or slug}"),
            }
            c["score1"] = score1(c)
            out.append(c)
    return out


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def score1(c: dict) -> float:
    """Насколько исход «ожил» — по данным, которые уже есть в списке рынков.

    Рост цены дешёвой стороны и всплеск оборота — то, как на рынке выглядит
    слух. Это не сигнал, а только повод потратить запросы на проверку.
    """
    rel1 = _clip(c["d1"] / max(c["mkt"] - c["d1"], 0.005), -0.5, 2.0)
    rel7 = _clip(c["d7"] / max(c["mkt"] - c["d7"], 0.005), -0.5, 2.0)
    surge = c["vol24"] / max(c["vol1wk"] / 7.0, 1.0)
    return (rel1 + 0.3 * rel7 + 0.5 * _clip(math.log2(max(surge, 1e-6)), -1.0, 3.0)
            + 0.15 * math.log10(max(c["vol24"], 1.0)))


# ---------------------------------------------------------- подтверждения

def gather(api: Api, c: dict, wallets: dict) -> dict:
    query = sources.search_query(c["q"])
    return {
        "ts": time.time(),
        "news": sources.news(query),
        "venues": sources.venues(c["q"], query),
        "whales": sources.whales(api, c["cid"], c["yes"], wallets),
    }


def _smart_usd(w: dict, key: str) -> float:
    return sum(s["usd"] for s in (w.get(key) or []))


def venue_plausible(c: dict, pv: float) -> bool:
    """Может ли чужой рынок с такой ценой быть тем же вопросом.

    Разрыв в разы между площадками не живёт: это почти наверняка другой или
    перевёрнутый вопрос («уйдёт ли» против «останется ли»). Так «Altman out
    as CEO?» за 4c совпал по словам с рынком на 93%. Проверка действует и
    тогда, когда совпадение подтвердил ИИ: слабая модель ошибается так же.
    """
    return pv <= max(VENUE_MAX_RATIO * c["mkt"], c["mkt"] + VENUE_MAX_GAP)


def venue_bid(c: dict, v: dict, pv: float) -> float:
    """По какой цене нашу сторону готовы КУПИТЬ на той площадке.

    Середина между покупкой и продажей при широком разбросе ничего не значит:
    бид 1c и аск 11c дают «6%», хотя платить столько никто не готов. Настоящее
    расхождение — когда тамошние покупатели дают не меньше здешней цены входа.
    """
    bid, ask = v.get("bid", v["prob"]), v.get("ask", v["prob"])
    # pv ближе к цене ДА того рынка — значит, наша сторона там тоже ДА
    same_side = abs(pv - v["prob"]) <= abs(pv - (1.0 - v["prob"]))
    return bid if same_side else 1.0 - ask


def venue_guess(c: dict, ev: dict) -> tuple[dict, float] | None:
    """Без ИИ: похожий рынок другой площадки и его оценка нашей стороны.

    Грубо и намеренно строго: слова совпали и срок почти тот же. «Тот же
    вопрос до конца года» вместо «до 30 ноября» сюда не пройдёт.
    """
    end = _end_ts(c["end"])
    for v in ev.get("venues") or []:
        if v["sim"] < VENUE_SIM or not v.get("close"):
            continue
        close = _end_ts(v["close"] + "T00:00:00")
        if not (close == close) or abs(close - end) > VENUE_DAYS * 86400.0:
            continue
        pv = v["prob"] if c["yes"] else 1.0 - v["prob"]
        if venue_plausible(c, pv):
            return v, pv
    return None


def score2(c: dict, ev: dict, now: float) -> float:
    """Очередь на оценку ИИ: кто «ожил» и уже имеет бесплатные подтверждения."""
    w = ev.get("whales") or {}
    s = c["score1"] + min(w.get("for_usd", 0.0) / 3000.0, 1.0)
    sf, sa = _smart_usd(w, "smart_for"), _smart_usd(w, "smart_against")
    if sf > sa:
        s += 1.5
    elif sa > 0:
        s -= 1.0
    g = venue_guess(c, ev)
    if g and g[1] >= VENUE_RATIO * c["cost"]:
        s += 1.5
    fresh = sum(1 for x in ev.get("news") or []
                if x.get("ts") and now - x["ts"] < 48 * 3600.0)
    return s + min(fresh / 4.0, 1.0)


def assess(c: dict, ev: dict | None, ai: dict | None) -> dict:
    """Сводит три источника в одну оценку и считает, сколько из них «за»."""
    ev = ev or {}
    w = ev.get("whales") or {}
    cost = c["cost"]
    a: dict = {"p_ai": None, "p_est": None, "venue": None, "pv": None,
               "venue_exact": False}

    if ai:
        a["p_ai"] = ai["p"]
        # trust < 1 у бесплатных моделей: их оценка сдвигает цену слабее
        k = SHRINK[ai["confidence"]] * float(ai.get("trust", 1.0))
        a["p_est"] = c["mkt"] + k * (ai["p"] - c["mkt"])
        if ai.get("venue") and venue_plausible(c, ai["venue_side_prob"]):
            # ИИ подтвердил, что условия совпадают, и цена этому не противоречит
            a["venue"], a["pv"], a["venue_exact"] = ai["venue"], ai["venue_side_prob"], True
    else:
        g = venue_guess(c, ev)    # ИИ не смотрел — остаётся грубая проверка
        if g:
            a["venue"], a["pv"] = g
    a["ai_ok"] = bool(ai and ai["confidence"] != "low" and ai["news_relevant"]
                      and ai["p"] >= AI_RATIO * cost
                      and a["p_est"] >= EST_RATIO * cost)
    a["venue_ok"] = (a["pv"] is not None and a["pv"] >= VENUE_RATIO * cost
                     and a["venue"]["venue"] in REAL_MONEY
                     and venue_bid(c, a["venue"], a["pv"]) >= cost)

    a["smart_for"] = _smart_usd(w, "smart_for")
    a["smart_against"] = _smart_usd(w, "smart_against")
    a["n_smart"] = len(w.get("smart_for") or [])
    a["whales_ok"] = a["smart_for"] > 0 and a["smart_for"] > a["smart_against"]

    # Расхождение с рынком в разы — чаще неверно понятый вопрос, чем находка.
    # 22.11 бесплатная модель дала 80% исходу «ровно одно повышение ставки»,
    # не заметив, что одно повышение в этом году уже было; рынок давал 7%.
    # Такую оценку засчитываем, только если её поддержал другой источник.
    a["ai_outlier"] = bool(ai and ai["p"] > max(AI_OUTLIER_RATIO * c["mkt"],
                                                c["mkt"] + AI_OUTLIER_GAP))
    if a["ai_ok"] and a["ai_outlier"] and not (a["venue_ok"] or a["whales_ok"]):
        a["ai_ok"] = False
    # ИИ уверенно посмотрел и запаса не увидел — тогда сигнала нет вовсе.
    # 22.11 ушли два «средних» сигнала, где покупку кошелька перевесила бы
    # оценка ИИ, равная цене: сообщение противоречило само себе. Неуверенная
    # оценка (low) не решает ни за, ни против.
    a["ai_against"] = bool(ai and ai["confidence"] != "low"
                           and a["p_est"] < EST_RATIO * cost)

    n_ok = int(a["ai_ok"]) + int(a["venue_ok"]) + int(a["whales_ok"])
    if a["ai_against"]:
        n_ok = 0
    a["level"] = min(n_ok, 2)
    a["sources"] = "+".join(k for k, ok in (("news", a["ai_ok"]),
                                            ("venue", a["venue_ok"]),
                                            ("whales", a["whales_ok"])) if ok)
    # Ориентир для сообщения — только из источника, который сам «за»:
    # оценка ИИ, если он подтвердил, иначе цена на другой площадке.
    # Прежде здесь стояла любая оценка ИИ, и при подтверждении с Kalshi
    # сообщение показывало неуверенную оценку ниже цены входа.
    a["guide"] = (a["p_est"] if a["ai_ok"]
                  else a["pv"] if a["venue_ok"] else None)
    a["ratio"] = (a["guide"] if a["guide"] is not None else c["mkt"]) / cost
    a["note"] = ("ИИ против" if a["ai_against"] else
                 "ИИ: расхождение подозрительно велико" if a["ai_outlier"]
                 and not a["ai_ok"] else "")
    return a


# --------------------------------------------------------------- сообщения

def pct(p: float) -> str:
    v = p * 100.0
    if v < 1.0:
        return "меньше 1%"
    return f"{v:.0f}%" if v >= 10 else f"{v:.1f}%".replace(".0%", "%")


def cents(p: float) -> str:
    return f"{p * 100:.1f}".rstrip("0").rstrip(".") + "¢"


def why_lines(c: dict, ai: dict | None, a: dict) -> list[str]:
    """Причины простыми словами — по строке на источник, которому есть что сказать."""
    L = []
    if ai and ai.get("reason"):
        if a["ai_ok"]:
            L.append(f"• <b>Новости:</b> {esc(ai['reason'])}")
        else:
            # мнение ИИ полезно прочитать, но подтверждением оно не считается
            L.append(f"• <b>Новости</b> (ИИ даёт {pct(ai['p'])}, в счёт не идёт): "
                     f"{esc(ai['reason'])}")
    if a["venue"] is not None and a["pv"] is not None:
        v = a["venue"]
        word = "тот же исход" if a["venue_exact"] else "похожий рынок"
        name = (f'<a href="{esc(v["url"])}">{esc(v["venue"])}</a>'
                if v.get("url") else esc(v["venue"]))
        if v["venue"] not in REAL_MONEY:
            name += " (игровые деньги)"
        if a["pv"] >= VENUE_RATIO * c["cost"]:
            tail = "" if a["venue_ok"] else " — в счёт не идёт"
            L.append(f"• <b>{name}:</b> {word} там оценивают в {pct(a['pv'])} "
                     f"(здесь {pct(c['mkt'])}){tail}")
        elif a["pv"] < VENUE_LOWER * c["mkt"]:
            # площадка, где исход стоит дешевле, — довод против, а не за
            L.append(f"• ⚠ <b>{name}:</b> {word} там оценивают ниже — "
                     f"в {pct(a['pv'])} (здесь {pct(c['mkt'])})")
    if a["smart_for"] > 0:
        n = a["n_smart"]
        who = "кошелёк" if n == 1 else "кошелька" if n < 5 else "кошельков"
        verb = "купил" if n == 1 else "купили"
        L.append(f"• <b>Игроки:</b> {n} {who} с прибыльной историей {verb} эту "
                 f"сторону на ${a['smart_for']:,.0f} за двое суток")
    return L


def news_lines(c: dict, ev: dict | None, now: float, n: int = 2) -> list[str]:
    """Свежие заголовки по теме — когда ИИ их не оценивал, судить читателю.

    Поиск по словам приносит и постороннее («Tesla» в новости про Lucid),
    поэтому заголовок берётся, только если в нём или в его анонсе есть
    минимум два значимых слова вопроса.
    """
    q = sources.stems(sources.search_query(c["q"]))
    need = min(2, len(q))
    out = []
    for x in (ev or {}).get("news") or []:
        if not x.get("ts") or now - x["ts"] > NEWS_FRESH_H * 3600.0:
            continue
        text = x["title"] + " " + (x.get("snippet") or "")
        if not need or len(q & sources.stems(text)) < need:
            continue
        src = f" — {esc(x['source'])}" if x.get("source") else ""
        out.append(f"   «{esc(x['title'][:120])}»{src}")
        if len(out) >= n:
            break
    return ["• <b>Свежие новости</b> (бот их не оценивал):"] + out if out else []


def signal_message(c: dict, ev: dict | None, ai: dict | None, a: dict,
                   now: float, upgraded: bool = False) -> str:
    """Сообщение о сигнале, сильном или среднем. Уходит сразу, как найден."""
    p = a["guide"]
    if a["ai_ok"]:
        guide = f", оценка бота — около <b>{pct(p)}</b>"
    elif p is not None:
        # без ИИ своей оценки нет: ориентир — цена того же исхода у соседей
        guide = f", на {esc(a['venue']['venue'])} этот исход стоит <b>{pct(p)}</b>"
    else:
        guide = "; своей оценки у бота нет"      # подтверждают только игроки
    if a["level"] >= 2:
        head = "🎯 <b>Сильный сигнал</b>" + (
            " — был средним, подтвердился вторым источником" if upgraded else "")
    else:
        head = "🟡 <b>Средний сигнал</b> — подтверждает один источник"
    L = [head, f"<b>{esc(c['q'])}</b>", "",
         f"Ставка: <b>{c['side']} по {cents(c['cost'])}</b> — выплата "
         f"{1.0 / c['cost']:.0f}×",
         f"Рынок даёт {pct(c['mkt'])}{guide}", "",
         "<b>Почему рынок может ошибаться:</b>"]
    why = why_lines(c, ai, a)
    if not why and a["p_ai"] is not None:
        why = [f"• <b>Новости:</b> ИИ оценивает шанс в {pct(a['p_ai'])}"]
    L += why
    if not ai:
        L += news_lines(c, ev, now)
    risks = []
    if ai and ai.get("risk"):
        risks.append(esc(ai["risk"]))
    if a["smart_against"] > 0:
        risks.append(f"Прибыльные кошельки покупали и обратную сторону — "
                     f"на ${a['smart_against']:,.0f}.")
    if risks:
        L += ["", "<b>Что может пойти не так:</b> " + " ".join(risks)]
    lose = _clip(round((1.0 - (p if p is not None else c["mkt"])) * 10), 1, 9)
    L += ["", f"Честно о шансах: даже при верной оценке такая ставка "
              f"проигрывает примерно в {lose:.0f} случаях из 10."]
    k = evaluate(c["px"], p, c["rate"]).kelly * 0.25 if p is not None else 0.0
    if k > 0:
        L.append(f"Размер по четверти Келли: {k * 100:.1f}% банка.")
    L += [f"До конца: {c['days']:.0f} дн. · оборот за сутки ${c['vol24']:,.0f}",
          "", f'<a href="{esc(c["url"])}">Открыть рынок</a>',
          "<i>Это оценка, а не рекомендация. Решение за тобой. Если сигнал "
          "пропадёт, бот напишет об этом отдельно.</i>"]
    return "\n".join(L)


def why_gone(prev: dict, m: dict | None, c: dict | None,
             asm: dict | None) -> tuple[str, float | None]:
    """Почему присланного сигнала больше нет и сколько исход стоит сейчас.

    m — рынок из свежего списка (None: его там нет), c — он же как кандидат
    (None: не прошёл отбор), asm — его оценка в этом прогоне (None: не оценён).
    """
    if m is None:
        return "рынок закрылся или ушёл из числа активных", None
    bid, ask = _f(m.get("bestBid")), _f(m.get("bestAsk"))
    px = (ask if prev.get("yes") else 1.0 - bid) if bid == bid and ask == ask else None
    if c is None:
        if px is not None and px > P_HI:
            return (f"цена выросла выше {cents(P_HI)} — дешёвым этот исход "
                    f"бот больше не считает"), px
        if px is not None and px < P_LO:
            return f"цена упала ниже {cents(P_LO)}", px
        return ("рынок перестал проходить отбор: мал оборот, широк спред "
                "или до конца осталось меньше суток"), px
    if asm is None:
        return "по рынку не удалось собрать свежие данные", c["cost"]
    if asm["ai_outlier"] and not asm["ai_ok"]:
        return (f"ИИ дал {pct(asm['p_ai'])} — в разы выше рынка ({pct(c['mkt'])}), "
                f"похоже, он неверно понял вопрос, а другие источники его "
                f"не поддержали"), c["cost"]
    if asm["ai_against"]:
        return (f"ИИ не видит запаса: его оценка {pct(asm['p_est'])}, "
                f"рынок — {pct(c['mkt'])}"), c["cost"]
    parts = []
    if asm["p_est"] is not None:
        parts.append(f"оценка бота теперь {pct(asm['p_est'])}")
    if asm["venue"] is not None and asm["pv"] is not None:
        parts.append(f"{asm['venue']['venue']} даёт {pct(asm['pv'])}")
    parts.append(f"рынок — {pct(c['mkt'])}")
    return "подтверждение пропало: " + ", ".join(parts), c["cost"]


def gone_message(items: list[tuple[dict, str, float | None]]) -> str:
    """«Сигнал снят» — по одному или списком, если снято несколько сразу.

    Только факты: что было, что стало, почему. Несколько снятий за прогон
    уходят одним сообщением, чтобы не засыпать чат.
    """
    def body(prev: dict, reason: str, px: float | None) -> list[str]:
        now_s = f", сейчас {cents(px)}" if px is not None else ""
        return [f"<b>{esc(prev.get('q') or '')}</b>",
                f"{prev.get('side') or 'Ставка'}: при сигнале было "
                f"{cents(float(prev.get('cost') or 0.0))}{now_s}.",
                f"Причина: {esc(reason)}.",
                f'<a href="{esc(prev.get("url") or "")}">Открыть рынок</a>']

    if len(items) == 1:
        return "\n".join(["⚪ <b>Сигнал снят</b>", ""] + body(*items[0]))
    L = [f"⚪ <b>Сняты сигналы: {len(items)}</b>"]
    for i, it in enumerate(items, 1):
        b = body(*it)
        L += ["", f"{i}. {b[0]}"] + b[1:]
    return "\n".join(L)


def update_signals(rated: list[tuple], sent: dict, now: float) -> list[tuple]:
    """Что изменилось с прошлого прогона. Меняет `sent`, возвращает события.

    ("new", t)    сигнал появился — t это (кандидат, данные, ИИ, оценка);
    ("up", t)     средний стал сильным;
    ("gone", key) сигнала нет MISS_RUNS прогонов подряд.

    Раньше в Telegram сразу уходили только сильные, а средние ждали суточной
    сводки. За сутки цена и новости успевают смениться, поэтому теперь сразу
    уходит всё от ALERT_LEVEL и выше, а об исчезновении сигнала сообщается
    отдельно — иначе старое сообщение висело бы в чате как действующее.
    """
    events: list[tuple] = []
    live = {t[0]["key"]: t for t in rated if t[3]["level"] >= ALERT_LEVEL}
    for key, t in live.items():
        c, asm = t[0], t[3]
        prev = sent.get(key)
        if prev and prev.get("active"):
            prev["miss"], prev["ts"] = 0, now
            if asm["level"] > int(prev.get("level") or 0):
                prev["level"] = asm["level"]
                events.append(("up", t))
            continue
        if prev and now - float(prev.get("gone_ts") or 0) < COOLDOWN_H * 3600.0 \
                and asm["level"] <= int(prev.get("level") or 0):
            continue                      # недавно снят и сильнее не стал
        sent[key] = {"ts": now, "active": True, "miss": 0, "level": asm["level"],
                     "cost": round(c["cost"], 4), "cid": c["cid"], "yes": c["yes"],
                     "side": c["side"], "q": c["q"], "url": c["url"]}
        events.append(("new", t))
    for key, prev in sent.items():
        if not prev.get("active") or key in live:
            continue
        prev["miss"] = int(prev.get("miss") or 0) + 1
        if prev["miss"] >= MISS_RUNS:
            prev["active"], prev["gone_ts"], prev["ts"] = False, now, now
            events.append(("gone", key))
    return events


def digest_message(rows: list[tuple], n_cands: int, n_strong: int,
                   ai_on: bool, ai_free: bool, card: str, now: float) -> str:
    """Сводка: только средние и сильные — у слабых нет причины вне цены.

    Слабые раньше занимали свободные места и размывали сводку (16.11 таких
    было два из трёх). Если показать нечего — одна строка: видно, что бот
    жив, а читать нечего.
    """
    if not rows:
        return ("📋 <b>Дешёвые исходы за сутки:</b> сегодня ничего стоящего. "
                f"Просмотрено {n_cands} исходов по {cents(P_LO)}–{cents(P_HI)}, "
                "ни у одного нет подтверждения вне цены.")
    L = ["📋 <b>Дешёвые исходы за сутки</b>",
         f"Просмотрено {n_cands} исходов по {cents(P_LO)}–{cents(P_HI)}. "
         f"Сильных сигналов: {n_strong}."]
    if not ai_on:
        L.append("<i>Источников два: другие площадки и крупные игроки. Новости "
                 "бот показывает, но не оценивает — ИИ выключен.</i>")
    elif ai_free:
        L.append("<i>Новости оценивает бесплатная модель ИИ. Её оценке бот "
                 "доверяет меньше, чем цене рынка.</i>")
    for i, (c, ev, ai, a) in enumerate(rows, 1):
        if a["ai_ok"]:
            est = f", оценка {pct(a['guide'])}"
        else:                   # при выключенном ИИ повторять это в каждой строке незачем
            est = ", ИИ не оценивал" if ai_on and not ai else ""
        L += ["", f"{i}. <b>{esc(c['q'])}</b>",
              f"{c['side']} по {cents(c['cost'])} (выплата {1.0 / c['cost']:.0f}×) · "
              f"рынок {pct(c['mkt'])}{est} · <b>{STRENGTH_RU[a['level']]}</b>"]
        why = why_lines(c, ai, a)
        if not why and a["p_ai"] is not None:
            # подтверждение дал ИИ, но без пояснения — показать хотя бы цифру
            why = [f"• <b>Новости:</b> ИИ оценивает шанс в {pct(a['p_ai'])}"]
        L += why
        if not ai:
            L += news_lines(c, ev, now)
        L.append(f'<a href="{esc(c["url"])}">Открыть рынок</a> · '
                 f'до конца {c["days"]:.0f} дн.')
    n_src = "трёх" if ai_on else "двух"
    L += ["", card, "",
          f"<i>«Сильный» — согласны два источника из {n_src}, «средний» — один. "
          "Исходы без подтверждений в сводку не попадают. Это оценки, "
          "а не рекомендации.</i>"]
    return "\n".join(L)


# --------------------------------------------------- журнал и итог сигналов

def log_load() -> list[dict]:
    if not LOG.exists():
        return []
    with LOG.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def log_save(rows: list[dict]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def log_add(rows: list[dict], kind: str, c: dict, a: dict) -> bool:
    """Одна запись на исход и вид сообщения: повтор в сводке не пишется."""
    side = "YES" if c["yes"] else "NO"
    if any(r["cid"] == c["cid"] and r["side"] == side and r["kind"] == kind
           for r in rows):
        return False
    rows.append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": kind, "strength": STRENGTH_RU[a["level"]],
        "cid": c["cid"], "side": side,
        "cost": f"{c['cost']:.4f}", "mkt": f"{c['mkt']:.4f}",
        "p_ai": "" if a["p_ai"] is None else f"{a['p_ai']:.4f}",
        "p_est": "" if a["p_est"] is None else f"{a['p_est']:.4f}",
        "sources": a["sources"], "question": c["q"], "result": "",
    })
    return True


def log_resolve(api: Api, rows: list[dict]) -> int:
    """Проставляет win/lose закрывшимся рынкам. Возвращает число новых исходов."""
    open_cids = sorted({r["cid"] for r in rows if not r.get("result")})
    n = 0
    for i in range(0, len(open_cids), 20):
        try:
            got = api.get(f"{GAMMA}/markets",
                          [("condition_ids", x) for x in open_cids[i:i + 20]]
                          + [("limit", 50)])
        except ApiError:
            continue
        for m in got or []:
            if not m.get("closed"):
                continue
            try:
                pr = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
            except (TypeError, ValueError):
                continue
            if sorted(pr) != [0.0, 1.0]:
                continue             # отменён или поделён пополам — не исход
            yes_won = pr[0] == 1.0
            for r in rows:
                if r["cid"] == m.get("conditionId") and not r.get("result"):
                    r["result"] = "win" if (r["side"] == "YES") == yes_won else "lose"
                    n += 1
    return n


def scorecard(rows: list[dict]) -> str:
    """Итог по закрывшимся сигналам: сколько выиграло и что дала бы ставка $1."""
    first: dict[tuple, dict] = {}
    for r in rows:                   # исход считается один раз, по первой записи
        first.setdefault((r["cid"], r["side"]), r)
    L = []
    for name in ("сильный", "средний", "слабый"):
        done = [r for r in first.values()
                if r["strength"] == name and r.get("result")]
        if not done:
            continue
        wins = [r for r in done if r["result"] == "win"]
        pnl = sum(1.0 / float(r["cost"]) - 1.0 for r in wins) - (len(done) - len(wins))
        L.append(f"• {name}: закрылось {len(done)}, выиграло {len(wins)}, "
                 f"итог {'+' if pnl >= 0 else '−'}${abs(pnl):.2f}")
    if L:
        return "\n".join(["<b>Как сыграли прошлые сигналы</b> "
                          "(если ставить $1 на каждый):"] + L)
    if first:
        return (f"<b>Как сыграли прошлые сигналы:</b> пока ни один из "
                f"{len(first)} присланных исходов не закрылся.")
    return ("<b>Как сыграли прошлые сигналы:</b> это первая сводка, итог "
            "появится, когда рынки начнут закрываться.")


# ----------------------------------------------------------------- состояние

def state_load() -> dict:
    try:
        st = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st = {}
    for k in ("ev", "ai", "sent", "wallets"):
        if not isinstance(st.get(k), dict):
            st[k] = {}
    return st


def state_save(st: dict, now: float) -> None:
    """Сохраняет кэш, выбрасывая устаревшее, чтобы файл не рос вечно."""
    def fresh(d: dict, hours: float) -> dict:
        return {k: v for k, v in d.items()
                if now - float(v.get("ts") or 0) < hours * 3600.0}

    st["ev"] = fresh(st["ev"], 72.0)
    st["ai"] = fresh(st["ai"], 14 * 24.0)
    st["sent"] = fresh(st["sent"], 14 * 24.0)
    st["wallets"] = fresh(st["wallets"], sources.WALLET_TTL_H)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, separators=(",", ":")),
                     encoding="utf-8")


def ai_needed(c: dict, old: dict | None, now: float, watched: bool = False) -> bool:
    """Нужна ли новая оценка ИИ. watched — по исходу уже прислан сигнал.

    За присланным сигналом следим строже: как только цена ушла на
    WATCH_MOVED, оценка пересчитывается, не дожидаясь суток.
    """
    if not old:
        return True
    age_h = (now - float(old.get("ts") or 0)) / 3600.0
    if age_h >= AI_STALE_H:
        return True
    moved = abs(c["cost"] - float(old.get("cost") or c["cost"])) / c["cost"]
    if watched:
        return moved >= WATCH_MOVED
    return age_h >= AI_TTL_H and moved >= AI_MOVED


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="ничего не отправлять, показать сообщения в консоли")
    ap.add_argument("--digest", action="store_true",
                    help="собрать сводку сейчас, не дожидаясь её часа")
    ap.add_argument("--no-ai", action="store_true",
                    help="не обращаться к ИИ, даже если ключ задан")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):       # консоль Windows не знает «¢»
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    now = time.time()
    today = time.strftime("%Y-%m-%d", time.gmtime(now))
    api = Api(min_interval=0.06)
    st = state_load()

    raw = pulse.fetch_markets(api)
    if not raw:
        print("! gamma не отдала ни одного рынка")
        return 1
    cands = select(raw, now)
    cands.sort(key=lambda c: -c["score1"])
    print(f"живых рынков: {len(raw)} | дешёвых исходов: {len(cands)}", flush=True)

    # --- 1. подтверждения: верхушка по активности + давно не проверявшиеся
    def ev_age(c: dict) -> float:
        return now - float((st["ev"].get(c["cid"]) or {}).get("ts") or 0)

    # Исходы, по которым сигнал уже прислан, проверяются в каждом прогоне и
    # первыми: именно про них надо вовремя узнать, что ситуация изменилась.
    watching = {k for k, v in st["sent"].items() if v.get("active")}
    watch = [c for c in cands if c["key"] in watching]
    rest = sorted(cands[N_EVIDENCE:], key=lambda c: -ev_age(c))
    seen: set[str] = set()
    todo = []
    for c in watch + cands[:N_EVIDENCE] + rest[:N_ROTATE]:
        if c["cid"] not in seen and ev_age(c) > EV_TTL_H * 3600.0:
            seen.add(c["cid"])
            todo.append(c)
    for c in todo:
        try:
            st["ev"][c["cid"]] = gather(api, c, st["wallets"])
        except Exception as e:                      # noqa: BLE001
            # один сломанный рынок не должен отменять остальные
            print(f"  ! сбор по «{c['q'][:50]}»: {type(e).__name__}: {e}", flush=True)
    print(f"подтверждения собраны по {len(todo)} исходам", flush=True)

    def ev_of(c: dict) -> dict | None:
        ev = st["ev"].get(c["cid"])
        return ev if ev and now - float(ev.get("ts") or 0) < EV_KEEP_H * 3600.0 else None

    known = [c for c in cands if ev_of(c)]
    for c in known:
        c["score2"] = score2(c, ev_of(c), now)
    # в очереди на ИИ первыми идут исходы с уже присланным сигналом
    known.sort(key=lambda c: (0 if c["key"] in watching else 1, -c["score2"]))

    # --- 2. оценка ИИ: лучшие по бесплатным признакам, в пределах бюджета
    ai_on = judge.available() and not a.no_ai
    ai_free = ai_on and judge.is_free()
    day = st.get("ai_day") or {}
    if day.get("day") != today:
        day = {"day": today, "n": 0, "usd": 0.0}
    if ai_on:
        # Суточный потолок раздаётся по часам: иначе он кончался бы в первые
        # два прогона после полуночи, и на новость в середине дня не оставалось.
        hour_now = time.gmtime(now).tm_hour
        paced = math.ceil(AI_PER_DAY * (hour_now + 1) / 24.0)
        left = min(AI_PER_RUN, paced - int(day["n"]))
        t_ai = time.time()
        for c in known:
            if left <= 0 or time.time() - t_ai > AI_TIME_S:
                break
            if not ai_needed(c, st["ai"].get(c["key"]), now,
                             watched=c["key"] in watching):
                continue
            ev = ev_of(c)
            try:
                r = judge.estimate(c, ev)
            except judge.JudgeOff as e:
                print(f"! ИИ выключен до конца прогона: {e}", flush=True)
                break
            left -= 1
            day["n"] = int(day["n"]) + 1
            if r is None:
                continue
            day["usd"] = float(day["usd"]) + r["usd"]
            # сам рынок площадки, а не его номер: список к следующему прогону
            # пересоберётся, и номер укажет на другой рынок
            r["venue"] = ev["venues"][r["venue_index"]] if r["venue_index"] >= 0 else None
            st["ai"][c["key"]] = dict(r, ts=now, cost=c["cost"])
            print(f"  ИИ: {r['p'] * 100:.1f}% ({r['confidence']}) при цене "
                  f"{c['cost'] * 100:.1f}¢ — {c['q'][:60]} [{r['model']}]", flush=True)
        spent = "бесплатно" if ai_free else f"~${float(day['usd']):.3f}"
        print(f"ИИ ({judge.model()}): за сутки {day['n']} оценок из "
              f"{AI_PER_DAY}, {spent}", flush=True)
    else:
        print("ИИ не используется: " + ("флаг --no-ai" if a.no_ai else
                                        "не задан ключ OPENROUTER_API_KEY "
                                        "или ANTHROPIC_API_KEY"), flush=True)
    st["ai_day"] = day

    # --- 3. сводим источники
    rated = []
    for c in known:
        ai = st["ai"].get(c["key"])
        if ai and now - float(ai.get("ts") or 0) >= AI_STALE_H * 3600.0:
            ai = None                      # протухшая оценка хуже, чем никакой
        rated.append((c, ev_of(c), ai, assess(c, ev_of(c), ai)))
    # При равной силе вперёд идут исходы, которые ИИ уже оценил: у них есть
    # объяснение. Иначе сводку занимали неоценённые — у них «запас» считается
    # от голой цены и почти одинаков, так что порядок выходил случайным.
    rated.sort(key=lambda t: (-t[3]["level"], 0 if t[2] else 1,
                              -t[3]["ratio"], -t[0]["score2"]))
    strong = [t for t in rated if t[3]["level"] == 2]
    print(f"оценено: {len(rated)} | сильных: {len(strong)} | средних: "
          f"{sum(1 for t in rated if t[3]['level'] == 1)} | с оценкой ИИ: "
          f"{sum(1 for t in rated if t[2])}", flush=True)
    # Верх списка — в отчёт прогона: по нему видно, что бот думает о рынках,
    # даже когда в Telegram ничего не ушло.
    for c, _ev, ai, asm in rated[:6]:
        told = (f"ИИ {ai['p'] * 100:.1f}% ({ai['confidence']}) -> оценка "
                f"{asm['p_est'] * 100:.1f}% [{ai.get('model', '?')}]" if ai
                else "ИИ не оценивал")
        note = f" ({asm['note']})" if asm["note"] else ""
        print(f"  {STRENGTH_RU[asm['level']]:8s} {c['side']:3s} {c['cost'] * 100:4.1f}¢ "
              f"рынок {c['mkt'] * 100:4.1f}% | {told} | "
              f"{asm['sources'] or 'без подтверждений'}{note} | {c['q'][:70]}",
              flush=True)

    tg = None if a.dry else Telegram()
    log = log_load()
    n_closed = log_resolve(api, log)
    if n_closed:
        print(f"закрылось прошлых сигналов: {n_closed}", flush=True)

    def out(text: str) -> None:
        if a.dry:
            print("\n" + "-" * 70 + "\n" + re.sub(r"<[^>]+>", "", text))
        else:
            tg.send(text)

    # --- 4. сигналы — сразу: новые, усилившиеся и снятые
    # в пробном прогоне память об отправленном не трогаем: работаем с копией
    sent = st["sent"] if not a.dry else copy.deepcopy(st["sent"])
    by_cid = {str(m.get("conditionId")): m for m in raw}
    cand_by_key = {c["key"]: c for c in cands}
    rated_by_key = {t[0]["key"]: t for t in rated}
    n_sent = 0
    gone: list[tuple] = []
    for kind, x in update_signals(rated, sent, now):
        if kind == "gone":
            prev, t = sent[x], rated_by_key.get(x)
            reason, px = why_gone(prev, by_cid.get(str(prev.get("cid"))),
                                  cand_by_key.get(x), t[3] if t else None)
            # если рынок закрылся, журнал уже знает исход — он важнее причины
            side = "YES" if prev.get("yes") else "NO"
            res = next((r["result"] for r in log if r["cid"] == prev.get("cid")
                        and r["side"] == side and r.get("result")), "")
            if res:
                reason = "рынок закрылся, ставка " + (
                    "выиграла" if res == "win" else "проиграла")
            gone.append((prev, reason, px))
            continue
        c, ev, ai, asm = x
        out(signal_message(c, ev, ai, asm, now, upgraded=kind == "up"))
        n_sent += 1
        if not a.dry:
            log_add(log, "strong" if asm["level"] >= 2 else "alert", c, asm)
    if gone:
        out(gone_message(gone))
    n_gone = len(gone)

    # --- 5. сводка — раз в сутки
    hour = time.gmtime(now).tm_hour
    if a.digest or (st.get("digest_day") != today and hour >= DIGEST_HOUR_UTC):
        # только средние и сильные; rated уже отсортирован по силе
        top = [t for t in rated if t[3]["level"] >= 1][:DIGEST_TOP]
        out(digest_message(top, len(cands), len(strong), ai_on, ai_free,
                           scorecard(log), now))
        if not a.dry:
            st["digest_day"] = today
            for c, ev, ai, asm in top:
                log_add(log, "digest", c, asm)

    if not a.dry:
        log_save(log)
    state_save(st, now)   # кэш сохраняется и в пробном прогоне: он стоит запросов
    print(f"\nсигналов: новых и усилившихся {n_sent}, снятых {n_gone}"
          f"{' (пробный прогон, ничего не послано)' if a.dry else ' — отправлено'}"
          f" | под наблюдением: {sum(1 for v in sent.values() if v.get('active'))}"
          f" | запросов к Polymarket: {api.n_calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
