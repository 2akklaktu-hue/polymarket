"""Источники, не зависящие от цены Polymarket: новости, другие площадки, игроки.

Зачем. Цена рынка сама про себя ничего нового не скажет: «ДА стоит 60¢,
значит ДА вероятнее» — тавтология. Чтобы утверждать, что исход за 5¢
недооценён, нужна причина извне. Здесь три таких причины:

  news()     свежие заголовки по теме рынка (Bing News RSS, запасной — Google);
  venues()   цена того же исхода на Kalshi и Manifold;
  whales()   покупки дешёвой стороны кошельками с прибыльной историей.

Все функции устойчивы к отказу источника: при любой ошибке возвращают пустой
результат. Отвалившийся Bing не должен ронять прогон, он просто не даст
подтверждения.

Metaculus не подключён: его API отвечает 403 без личного токена.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import requests

from .api import Api, ApiError, DATA

_UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/127.0 Safari/537.36"),
    "Accept": "*/*",
}

_STOP = set("""
will the a an be by before after in on of to for at is are was were does do did
has have had and or than this that these those from with as its it his her
their who what when which how many much any new next end between over under
more less least most not no yes if into out up down per vs during until
through above below there then again ever still become announce announces
announced win wins officially happen
""".split())
_MONTHS = set("""
january february march april may june july august september october november
december jan feb mar apr jun jul aug sep sept oct nov dec
""".split())
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'&.]*|\d[\d,.]*")


def _is_date_number(t: str) -> bool:
    """Число месяца или год: для поиска это шум, а не тема."""
    if not t.isdigit():
        return False
    n = int(t)
    return n <= 31 or 2020 <= n <= 2040


def tokens(text: str) -> list[str]:
    """Значимые слова вопроса: без служебных, без дат."""
    out = []
    for t in _TOKEN.findall(text.replace("-", " ").replace("/", " ")):
        t = t.strip(".',").lower()
        if not t or (len(t) < 2 and not t.isdigit()):
            continue
        if t in _STOP or t in _MONTHS or _is_date_number(t):
            continue
        out.append(t)
    return out


def stems(text: str) -> set[str]:
    """Значимые слова, грубо приведённые к основе.

    «rates» и «rate», «Iran's» и «Iran» должны считаться одним словом, иначе
    заголовок по теме не опознаётся как относящийся к вопросу.
    """
    out = set()
    for t in tokens(text):
        if t.endswith("'s"):
            t = t[:-2]
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        if t:
            out.add(t)
    return out


def search_query(question: str, n: int = 6) -> str:
    """Поисковая строка из вопроса рынка.

    «Will the Fed increase interest rates by 50+ bps after the December 2026
    meeting?» -> «fed increase interest rates 50 bps». Даты выбрасываются
    намеренно: новостной поиск по «November 30» находит календарь, а не тему.
    """
    seen: list[str] = []
    for t in tokens(question):
        if t not in seen:
            seen.append(t)
    return " ".join(seen[:n])


def similarity(a: str, b: str) -> float:
    """Доля общих значимых слов (коэффициент Дайса), 0..1."""
    A, B = set(tokens(a)), set(tokens(b))
    if not A or not B:
        return 0.0
    return 2.0 * len(A & B) / (len(A) + len(B))


def _get(url: str, params: dict | None = None, timeout: float = 15.0):
    try:
        r = requests.get(url, params=params, headers=_UA, timeout=timeout)
    except requests.RequestException:
        return None
    return r if r.status_code == 200 else None


# ------------------------------------------------------------------ новости

def _rss_items(content: bytes) -> list[dict]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        src = ""
        for ch in it:
            # у Bing источник лежит в <News:Source>, у Google — в <source>;
            # пространство имён у Bing меняется от запроса к запросу
            if ch.tag.lower().endswith("source") and (ch.text or "").strip():
                src = ch.text.strip()
                break
        try:
            ts = parsedate_to_datetime(it.findtext("pubDate") or "").timestamp()
        except (TypeError, ValueError):
            ts = float("nan")
        desc = re.sub(r"<[^>]+>", " ", it.findtext("description") or "")
        out.append({"title": title, "source": src, "ts": ts,
                    "snippet": re.sub(r"\s+", " ", desc).strip()[:220]})
    return out


def news(query: str, days: float = 7.0, n: int = 8) -> list[dict]:
    """Свежие заголовки по запросу, от новых к старым.

    Основной источник — Bing News: отдаёт RSS без ключа и доступен отовсюду.
    Google News подключается, только если Bing ничего не вернул (из части
    стран он недоступен, поэтому таймаут короткий).
    """
    if not query:
        return []
    items: list[dict] = []
    r = _get("https://www.bing.com/news/search",
             {"q": query, "format": "rss", "qft": 'interval="8"'})  # за неделю
    if r is not None:
        items = _rss_items(r.content)
    if not items:
        r = _get("https://news.google.com/rss/search",
                 {"q": f"{query} when:{int(days)}d", "hl": "en-US",
                  "gl": "US", "ceid": "US:en"}, timeout=8.0)
        if r is not None:
            items = _rss_items(r.content)
    since = time.time() - days * 86400.0
    fresh = [x for x in items if not (x["ts"] == x["ts"]) or x["ts"] >= since]
    fresh.sort(key=lambda x: -(x["ts"] if x["ts"] == x["ts"] else 0.0))
    seen: set[str] = set()
    out = []
    for x in fresh:                      # перепечатки одного заголовка не нужны
        key = " ".join(tokens(x["title"])[:6])
        if key in seen:
            continue
        seen.add(key)
        # None вместо NaN: результат уходит в JSON-кэш, а NaN в нём невалиден
        out.append(dict(x, ts=x["ts"] if x["ts"] == x["ts"] else None))
        if len(out) >= n:
            break
    return out


# ----------------------------------------------------------- другие площадки

def _fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def kalshi(question: str, query: str, n: int = 4) -> list[dict]:
    """Похожие рынки Kalshi. Вероятность — середина между бидом и аском ДА."""
    r = _get("https://api.elections.kalshi.com/v1/search/series",
             {"query": query, "order_by": "querymatch", "page_size": 6})
    if r is None:
        return []
    try:
        page = r.json().get("current_page") or []
    except ValueError:
        return []
    out = []
    for ev in page:
        title = str(ev.get("event_title") or ev.get("series_title") or "")
        sub = str(ev.get("event_subtitle") or "")
        series = str(ev.get("series_ticker") or "")
        for mk in ev.get("markets") or []:
            bid = _fnum(mk.get("yes_bid_dollars"))
            ask = _fnum(mk.get("yes_ask_dollars"))
            if not (bid == bid and ask == ask) or ask <= 0 or (ask - bid) > 0.10:
                continue                 # пустой или слишком широкий стакан
            name = " — ".join(x for x in (title, str(mk.get("yes_subtitle") or ""),
                                           sub) if x)
            out.append({
                "venue": "Kalshi", "title": name, "prob": (bid + ask) / 2.0,
                "close": str(mk.get("close_ts") or "")[:10],
                "url": f"https://kalshi.com/markets/{series.lower()}",
                "sim": similarity(question, name),
            })
    out.sort(key=lambda x: -x["sim"])
    return out[:n]


def manifold(question: str, query: str, n: int = 2) -> list[dict]:
    """Похожие рынки Manifold. Деньги там игровые, поэтому вес у них меньше."""
    r = _get("https://api.manifold.markets/v0/search-markets",
             {"term": query, "filter": "open", "contractType": "BINARY",
              "limit": 6, "sort": "score"})
    if r is None:
        return []
    try:
        rows = r.json()
    except ValueError:
        return []
    out = []
    for x in rows if isinstance(rows, list) else []:
        p = _fnum(x.get("probability"))
        if not (p == p) or int(x.get("uniqueBettorCount") or 0) < 8:
            continue                     # рынок на трёх участников — не мнение
        ct = _fnum(x.get("closeTime"))
        close = (time.strftime("%Y-%m-%d", time.gmtime(ct / 1000.0))
                 if ct == ct and 0 < ct < 4e12 else "")
        out.append({
            "venue": "Manifold", "title": str(x.get("question") or ""),
            "prob": p, "close": close, "url": str(x.get("url") or ""),
            "sim": similarity(question, str(x.get("question") or "")),
        })
    out.sort(key=lambda x: -x["sim"])
    return out[:n]


def venues(question: str, query: str) -> list[dict]:
    """Кандидаты на «тот же исход» с других площадок, от похожих к непохожим.

    Совпадение слов ещё не значит совпадения условий: срок и порог решают
    всё («до 30 ноября» и «до конца года» — разные ставки). Окончательно
    это проверяет ИИ; без него используется только грубый порог похожести.
    """
    rows = kalshi(question, query) + manifold(question, query)
    rows = [r for r in rows if r["sim"] >= 0.25]
    rows.sort(key=lambda r: -r["sim"])
    return rows[:6]


# ------------------------------------------------------------ крупные игроки

SMART_PNL = 5000.0        # $ прибыли за всё время
SMART_MARGIN = 0.02       # и не меньше 2% от оборота: отсекает маркет-мейкеров
WALLET_TTL_H = 24.0 * 7


def wallet_stats(api: Api, wallet: str, cache: dict) -> dict:
    """Прибыль и оборот кошелька за всё время (data-api v2 leaderboard)."""
    now = time.time()
    hit = cache.get(wallet)
    if hit and (now - float(hit.get("ts") or 0)) < WALLET_TTL_H * 3600.0:
        return hit
    try:
        d = (api.get(f"{DATA}/v2/leaderboard", {"user": wallet}) or {}).get("data") or {}
    except ApiError:
        # сбой не кэшируем: иначе кошелёк на неделю остался бы «неизвестным»
        return {"ts": now, "pnl": None, "vol": None}
    pnl, vol = _fnum(d.get("pnl")), _fnum(d.get("volume"))
    # None вместо NaN: кэш уходит в JSON, а NaN в нём невалиден
    st = {"ts": now, "pnl": pnl if pnl == pnl else None,
          "vol": vol if vol == vol else None}
    cache[wallet] = st
    return st


def is_smart(st: dict) -> bool:
    pnl, vol = st.get("pnl"), st.get("vol")
    if pnl is None or vol is None:
        return False
    return pnl >= SMART_PNL and vol > 0 and pnl / vol >= SMART_MARGIN


def whales(api: Api, cid: str, cheap_is_yes: bool, cache: dict,
           hours: float = 48.0, min_usd: float = 300.0) -> dict:
    """Кто за последние `hours` часов покупал дешёвую сторону и кто — дорогую.

    Считаются только ПОКУПКИ. Продажа двусмысленна: это может быть и ставка
    против, и выход из старой позиции с прибылью, а по ленте их не различить.

    Лента по умолчанию отдаёт сделки тейкеров, то есть тех, кто сам пришёл
    и забрал цену из стакана, — это и есть «кто-то решил, что надо брать».
    """
    out = {"for_usd": 0.0, "against_usd": 0.0, "n_for": 0, "n_against": 0,
           "smart_for": [], "smart_against": []}
    try:
        j = api.get(f"{DATA}/v2/trades",
                    {"condition": cid, "limit": 200, "filter_type": "CASH",
                     "filter_amount": min_usd})
    except ApiError:
        return out
    since = time.time() - hours * 3600.0
    by_wallet: dict[str, dict] = {}
    for r in (j.get("data") or []):
        ts = _fnum(r.get("timestamp"))
        if not (ts == ts) or ts < since:
            continue
        if str(r.get("side") or "").upper() != "BUY":
            continue
        is_yes = str(r.get("outcome_index")) == "0"
        usd = _fnum(r.get("size")) * _fnum(r.get("price"))
        w = str(r.get("proxy_wallet") or "")
        if not (usd == usd) or not w:
            continue
        rec = by_wallet.setdefault(w, {
            "who": str(r.get("pseudonym") or r.get("name") or "кто-то"),
            "for": 0.0, "against": 0.0})
        rec["for" if is_yes == cheap_is_yes else "against"] += usd

    for rec in by_wallet.values():
        out["for_usd"] += rec["for"]
        out["against_usd"] += rec["against"]
        out["n_for"] += int(rec["for"] > 0)
        out["n_against"] += int(rec["against"] > 0)
    # статистику запрашиваем только у самых крупных: по запросу на кошелёк
    top = sorted(by_wallet.items(),
                 key=lambda kv: -max(kv[1]["for"], kv[1]["against"]))[:6]
    for w, rec in top:
        st = wallet_stats(api, w, cache)
        if not is_smart(st):
            continue
        side = "smart_for" if rec["for"] >= rec["against"] else "smart_against"
        out[side].append({"who": rec["who"], "pnl": st["pnl"],
                          "usd": max(rec["for"], rec["against"])})
    return out
