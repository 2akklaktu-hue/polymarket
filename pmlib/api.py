"""Клиент публичного API Polymarket.

Три хоста:
  gamma-api  — метаданные рынков и событий (вопрос, даты, исход, объём)
  clob       — стаканы, текущие цены, история цен
  data-api   — сделки, держатели

Ключи не нужны: всё, что используется здесь, — чтение публичных данных.
Обязателен «браузерный» User-Agent, иначе Cloudflare отдаёт 403.
"""
from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0 Safari/537.36"
)


class Api:
    """HTTP-обёртка с ретраями, бэкоффом и мягким троттлингом."""

    def __init__(self, min_interval: float = 0.06, timeout: int = 30, retries: int = 5):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": _UA, "Accept": "application/json"})
        self.min_interval = min_interval
        self.timeout = timeout
        self.retries = retries
        self._last = 0.0
        self.n_calls = 0
        self._lock = threading.Lock()   # клиент шарится между потоками

    def get(self, url: str, params: dict | None = None) -> Any:
        for attempt in range(self.retries):
            with self._lock:           # общий троттлинг на все потоки
                gap = self.min_interval - (time.monotonic() - self._last)
                if gap > 0:
                    time.sleep(gap)
                self._last = time.monotonic()
                self.n_calls += 1
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
            except requests.RequestException:
                time.sleep((2 ** attempt) * 0.5 + random.random())
                continue
            if r.status_code == 200:
                return r.json()
            # 429/5xx — ждём и пробуем снова; 4xx — смысла нет
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep((2 ** attempt) * 1.0 + random.random())
                continue
            raise ApiError(f"{r.status_code} {url} {r.text[:200]}")
        raise ApiError(f"не удалось за {self.retries} попыток: {url}")

    # ---------- gamma ----------

    def markets_keyset(self, closed: bool | None = None, limit: int = 500,
                       **extra) -> Iterator[dict]:
        """Все рынки через keyset-пагинацию (offset ограничен 5000)."""
        cursor = None
        while True:
            p = {"limit": limit, **extra}
            if closed is not None:
                p["closed"] = str(closed).lower()
            if cursor:
                p["next_cursor"] = cursor
            j = self.get(f"{GAMMA}/markets/keyset", p)
            batch = j.get("markets") or []
            if not batch:
                return
            yield from batch
            cursor = j.get("next_cursor")
            if not cursor:
                return

    def markets(self, **params) -> list[dict]:
        return self.get(f"{GAMMA}/markets", params)

    # ---------- clob ----------

    def price_history(self, token_id: str, interval: str = "max",
                      fidelity: int = 60) -> list[dict]:
        """Свечи [{t: unix_sec, p: price}]. Для давно закрытых рынков часто пусто."""
        j = self.get(f"{CLOB}/prices-history",
                     {"market": token_id, "interval": interval, "fidelity": fidelity})
        return j.get("history") or []

    def book(self, token_id: str) -> dict:
        return self.get(f"{CLOB}/book", {"token_id": token_id})

    def books(self, token_ids: list[str]) -> list[dict]:
        """Батч стаканов — один запрос вместо N."""
        j = self.s.post(f"{CLOB}/books",
                        json=[{"token_id": t} for t in token_ids],
                        timeout=self.timeout)
        j.raise_for_status()
        return j.json()

    # ---------- data-api v2 ----------
    # v1 отключён: отдаёт 410 «Data API v1 has been retired». В v2 пагинация
    # курсорная, ограничения глубины (прежние 10 000 по offset) больше нет,
    # limit до 1000. Фильтры start/end есть, но работают только при выборке
    # по кошельку — для рынка их нет, идти можно только курсором с конца.
    # Поле outcomeIndex переименовано в outcome_index.

    def trades(self, condition_id: str, limit: int = 1000,
               cursor: str | None = None) -> tuple[list[dict], str | None]:
        """Страница сделок рынка, от НОВЫХ к старым.

        Возвращает (сделки, курсор следующей страницы); курсор None — конец.
        По умолчанию taker_only, то есть каждая сделка приходит один раз.
        """
        p: dict[str, Any] = {"condition": condition_id, "limit": limit}
        if cursor:
            p["cursor"] = cursor
        j = self.get(f"{DATA}/v2/trades", p)
        pg = j.get("pagination") or {}
        nxt = pg.get("next_cursor") if pg.get("has_more") else None
        return (j.get("data") or []), nxt

    def trades_pages(self, condition_id: str, pages: int = 3,
                     limit: int = 1000) -> list[dict]:
        """До `pages` страниц сделок рынка, начиная с самых свежих."""
        rows: list[dict] = []
        cur: str | None = None
        for _ in range(pages):
            batch, cur = self.trades(condition_id, limit=limit, cursor=cur)
            rows.extend(batch)
            if not cur or not batch:
                break
        return rows


class ApiError(RuntimeError):
    pass


def jsonl_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jsonl_read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]
