"""Работа со стаканом CLOB: реальная цена исполнения, а не «цена на экране».

Зачем отдельный модуль: на Polymarket видимая цена — это середина или последняя
сделка, а купить по ней нельзя. На дешёвых рынках лучший ask может быть вдвое
выше mid, и весь расчётный эдж съедается на входе. Все оценки в проекте должны
строиться от стоимости реального исполнения нужного объёма.

Особенность формата API: bids приходят по ВОЗРАСТАНИЮ цены, asks — по УБЫВАНИЮ,
то есть лучшие уровни в конце списков.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .econ import DEFAULT_FEE_RATE, taker_fee_per_share


@dataclass
class Book:
    asset_id: str = ""
    bids: list[tuple[float, float]] = field(default_factory=list)  # (цена, размер), лучший первым
    asks: list[tuple[float, float]] = field(default_factory=list)
    tick: float = 0.001
    min_size: float = 5.0

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    @property
    def spread(self) -> float | None:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return None

    @property
    def mid(self) -> float | None:
        if self.bids and self.asks:
            return (self.asks[0][0] + self.bids[0][0]) / 2.0
        return None

    def ask_depth_usd(self, max_price: float = 1.0) -> float:
        """Сколько долларов можно потратить, не задрав цену выше max_price."""
        return sum(p * s for p, s in self.asks if p <= max_price)


def parse(raw: dict) -> Book:
    """Разбор ответа /book или элемента /books в нормализованный вид."""
    def lv(rows, reverse):
        out = [(float(r["price"]), float(r["size"])) for r in (rows or [])
               if float(r.get("size", 0)) > 0]
        out.sort(key=lambda x: x[0], reverse=reverse)
        return out

    return Book(
        asset_id=str(raw.get("asset_id") or ""),
        bids=lv(raw.get("bids"), reverse=True),    # лучший bid — максимальная цена
        asks=lv(raw.get("asks"), reverse=False),   # лучший ask — минимальная цена
        tick=float(raw.get("tick_size") or 0.001),
        min_size=float(raw.get("min_order_size") or 5.0),
    )


@dataclass
class Fill:
    shares: float          # сколько долей реально куплено
    gross: float           # стоимость долей без комиссии, $
    fee: float             # комиссия тейкера, $
    total: float           # полная сумма списания, $
    avg_price: float       # средняя цена доли без комиссии
    eff_price: float       # средняя цена доли с комиссией = точка безубыточности
    complete: bool         # хватило ли глубины стакана


def buy_shares(book: Book, shares: float,
               fee_rate: float = DEFAULT_FEE_RATE) -> Fill:
    """Проход по стакану вверх: сколько стоит купить `shares` долей прямо сейчас."""
    need, gross, fee, got = shares, 0.0, 0.0, 0.0
    for price, size in book.asks:
        if need <= 0:
            break
        take = min(need, size)
        gross += take * price
        fee += take * taker_fee_per_share(price, fee_rate)
        got += take
        need -= take
    if got <= 0:
        return Fill(0, 0, 0, 0, float("nan"), float("nan"), False)
    return Fill(shares=got, gross=gross, fee=fee, total=gross + fee,
                avg_price=gross / got, eff_price=(gross + fee) / got,
                complete=need <= 1e-9)


def buy_for_budget(book: Book, budget: float,
                   fee_rate: float = DEFAULT_FEE_RATE) -> Fill:
    """Сколько долей даст бюджет в долларах с учётом проскальзывания и комиссии."""
    left, gross, fee, got = budget, 0.0, 0.0, 0.0
    for price, size in book.asks:
        if left <= 1e-9:
            break
        unit = price + taker_fee_per_share(price, fee_rate)   # стоимость одной доли
        if unit <= 0:
            continue
        take = min(size, left / unit)
        gross += take * price
        fee += take * taker_fee_per_share(price, fee_rate)
        got += take
        left -= take * unit
    if got <= 0:
        return Fill(0, 0, 0, 0, float("nan"), float("nan"), False)
    return Fill(shares=got, gross=gross, fee=fee, total=gross + fee,
                avg_price=gross / got, eff_price=(gross + fee) / got,
                complete=left <= 1e-6)


def sell_shares(book: Book, shares: float,
                fee_rate: float = DEFAULT_FEE_RATE) -> Fill:
    """Проход по бидам вниз: сколько выручим за продажу `shares` долей."""
    need, gross, fee, got = shares, 0.0, 0.0, 0.0
    for price, size in book.bids:
        if need <= 0:
            break
        take = min(need, size)
        gross += take * price
        fee += take * taker_fee_per_share(price, fee_rate)
        got += take
        need -= take
    if got <= 0:
        return Fill(0, 0, 0, 0, float("nan"), float("nan"), False)
    return Fill(shares=got, gross=gross, fee=fee, total=gross - fee,
                avg_price=gross / got, eff_price=(gross - fee) / got,
                complete=need <= 1e-9)
