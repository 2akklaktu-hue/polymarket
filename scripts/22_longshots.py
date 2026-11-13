"""Дешёвые исходы, у которых есть причина стоить дороже.

Что ищет. Исходы по 3-15 центов, про которые что-то ВНЕ цены Polymarket
говорит, что шанс выше:

  1. новости   — свежие заголовки по теме, оценённые ИИ (pmlib/judge.py);
  2. площадки  — тот же исход на Kalshi или Manifold стоит заметно дороже;
  3. игроки    — дешёвую сторону покупают кошельки с прибыльной историей.

Что присылает.
  * Сильный сигнал — сразу: согласны минимум два источника из трёх.
  * Сводка раз в сутки — три лучших кандидата с честной пометкой силы
    и итогом по прошлым сигналам: сколько закрылось и сколько выиграло.

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
    r"|(?:\b(?:dip|drop|fall|rise|reach|hit)s?\s+(?:to\s+)?\$)"
    r"|(?:\b(?:above|below|over|under)\s+\$[\d,.]+)"
    r"|(?:\bFDV\b)|(?:\bmarket cap\b)|(?:\ball[- ]time high\b)"
    # счётчики и уровни: число твитов, доходность облигаций — та же история,
    # что с курсами: исход решает цифра, а не новость
    r"|(?:\btweets?\b)|(?:\byield\b)", re.I)

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
RESEND_AFTER_H = 48.0

# --- когда источник считается «за» ---------------------------------------
AI_RATIO = 1.5            # оценка ИИ не ниже полутора цен входа
EST_RATIO = 1.2           # и после поправки к рынку остаётся запас 20%
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
                     and a["venue"]["venue"] in REAL_MONEY)

    a["smart_for"] = _smart_usd(w, "smart_for")
    a["smart_against"] = _smart_usd(w, "smart_against")
    a["n_smart"] = len(w.get("smart_for") or [])
    a["whales_ok"] = a["smart_for"] > 0 and a["smart_for"] > a["smart_against"]

    n_ok = int(a["ai_ok"]) + int(a["venue_ok"]) + int(a["whales_ok"])
    # ИИ посмотрел и запаса не увидел — два других источника его не перевесят
    if a["p_est"] is not None and a["p_est"] < EST_RATIO * cost:
        n_ok = min(n_ok, 1)
    a["level"] = min(n_ok, 2)
    a["sources"] = "+".join(k for k, ok in (("news", a["ai_ok"]),
                                            ("venue", a["venue_ok"]),
                                            ("whales", a["whales_ok"])) if ok)
    best = a["p_est"] if a["p_est"] is not None else (a["pv"] or c["mkt"])
    a["ratio"] = best / cost
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
        L.append(f"• <b>Новости:</b> {esc(ai['reason'])}")
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


def strong_message(c: dict, ev: dict | None, ai: dict | None, a: dict,
                   now: float) -> str:
    p = a["p_est"] if a["p_est"] is not None else a["pv"]
    # без ИИ своей оценки у бота нет: ориентир — цена того же исхода у соседей
    guide = (f"оценка бота — около <b>{pct(p)}</b>" if a["p_est"] is not None
             else f"на {esc(a['venue']['venue'])} этот исход стоит <b>{pct(p)}</b>")
    L = ["🎯 <b>Сильный сигнал</b>", f"<b>{esc(c['q'])}</b>", "",
         f"Ставка: <b>{c['side']} по {cents(c['cost'])}</b> — выплата "
         f"{1.0 / c['cost']:.0f}×",
         f"Рынок даёт {pct(c['mkt'])}, {guide}", "",
         "<b>Почему рынок может ошибаться:</b>"]
    L += why_lines(c, ai, a)
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
    lose = _clip(round((1.0 - p) * 10), 1, 9)
    L += ["", f"Честно о шансах: даже при верной оценке такая ставка "
              f"проигрывает примерно в {lose:.0f} случаях из 10."]
    k = evaluate(c["px"], p, c["rate"]).kelly * 0.25
    if k > 0:
        L.append(f"Размер по четверти Келли: {k * 100:.1f}% банка.")
    L += [f"До конца: {c['days']:.0f} дн. · оборот за сутки ${c['vol24']:,.0f}",
          "", f'<a href="{esc(c["url"])}">Открыть рынок</a>',
          "<i>Это оценка, а не рекомендация. Решение за тобой.</i>"]
    return "\n".join(L)


def digest_message(rows: list[tuple], n_cands: int, n_strong: int,
                   ai_on: bool, ai_free: bool, card: str, now: float) -> str:
    L = ["📋 <b>Дешёвые исходы за сутки</b>",
         f"Просмотрено {n_cands} исходов по {cents(P_LO)}–{cents(P_HI)}. "
         f"Сильных сигналов: {n_strong}."]
    if not ai_on:
        L.append("<i>Источников два: другие площадки и крупные игроки. Новости "
                 "бот показывает, но не оценивает — ИИ выключен.</i>")
    elif ai_free:
        L.append("<i>Новости оценивает бесплатная модель ИИ. Её оценке бот "
                 "доверяет меньше, чем цене рынка.</i>")
    if not rows:
        L += ["", "Сегодня показать нечего: подходящих исходов не нашлось. "
                  "Так бывает, и это не поломка."]
    elif all(a["level"] == 0 for _c, _ev, _ai, a in rows):
        # главное — первой строкой, чтобы не приходилось вычитывать её из пометок
        L += ["", "<b>Сегодня сигналов нет:</b> ни у одного исхода нет "
                  "подтверждения. Ниже три самых заметных — только для наблюдения."]
    for i, (c, ev, ai, a) in enumerate(rows, 1):
        if a["p_est"] is not None:
            est = f", оценка {pct(a['p_est'])}"
        else:                   # при выключенном ИИ повторять это в каждой строке незачем
            est = ", ИИ не оценивал" if ai_on else ""
        L += ["", f"{i}. <b>{esc(c['q'])}</b>",
              f"{c['side']} по {cents(c['cost'])} (выплата {1.0 / c['cost']:.0f}×) · "
              f"рынок {pct(c['mkt'])}{est} · <b>{STRENGTH_RU[a['level']]}</b>"]
        # без подтверждений исход попадает сюда только по активности торгов
        L += why_lines(c, ai, a) or ["• Подтверждений нет: показан как один из "
                                     "самых активных за сутки."]
        if not ai:
            L += news_lines(c, ev, now)
        L.append(f'<a href="{esc(c["url"])}">Открыть рынок</a> · '
                 f'до конца {c["days"]:.0f} дн.')
    n_src = "трёх" if ai_on else "двух"
    L += ["", card, "",
          f"<i>«Сильный» — согласны два источника из {n_src}, «средний» — один, "
          "«слабый» — ни одного, показан только для наблюдения. Это оценки, "
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


def ai_needed(c: dict, old: dict | None, now: float) -> bool:
    if not old:
        return True
    age_h = (now - float(old.get("ts") or 0)) / 3600.0
    if age_h >= AI_STALE_H:
        return True
    moved = abs(c["cost"] - float(old.get("cost") or c["cost"])) / c["cost"]
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

    rest = sorted(cands[N_EVIDENCE:], key=lambda c: -ev_age(c))
    todo = [c for c in cands[:N_EVIDENCE] + rest[:N_ROTATE]
            if ev_age(c) > EV_TTL_H * 3600.0]
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
    known.sort(key=lambda c: -c["score2"])

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
            if not ai_needed(c, st["ai"].get(c["key"]), now):
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
        print(f"  {STRENGTH_RU[asm['level']]:8s} {c['side']:3s} {c['cost'] * 100:4.1f}¢ "
              f"рынок {c['mkt'] * 100:4.1f}% | {told} | "
              f"{asm['sources'] or 'без подтверждений'} | {c['q'][:70]}", flush=True)

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

    # --- 4. сильные — сразу
    n_sent = 0
    for c, ev, ai, asm in strong:
        prev = st["sent"].get(c["key"])
        if prev and now - float(prev.get("ts") or 0) < RESEND_AFTER_H * 3600.0:
            continue
        out(strong_message(c, ev, ai, asm, now))
        n_sent += 1
        if not a.dry:
            st["sent"][c["key"]] = {"ts": now, "cost": round(c["cost"], 4)}
            log_add(log, "strong", c, asm)

    # --- 5. сводка — раз в сутки
    hour = time.gmtime(now).tm_hour
    if a.digest or (st.get("digest_day") != today and hour >= DIGEST_HOUR_UTC):
        top = rated[:DIGEST_TOP]       # слабые попадают, только если мест хватает
        out(digest_message(top, len(cands), len(strong), ai_on, ai_free,
                           scorecard(log), now))
        if not a.dry:
            st["digest_day"] = today
            for c, ev, ai, asm in top:
                log_add(log, "digest", c, asm)

    if not a.dry:
        log_save(log)
    state_save(st, now)   # кэш сохраняется и в пробном прогоне: он стоит запросов
    print(f"\nсильных сигналов: {n_sent}"
          f"{' (пробный прогон, ничего не послано)' if a.dry else ' отправлено'}"
          f" | запросов к Polymarket: {api.n_calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
