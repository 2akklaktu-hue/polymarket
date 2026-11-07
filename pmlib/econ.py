"""Экономика ставки на Polymarket: комиссии, эдж, размер, требуемая выборка.

Все функции работают с ценой одной доли (share) в долларах, 0 < p < 1.
Доля, оказавшаяся верной, погашается по $1.00.

Комиссия (официальная формула Polymarket, только для тейкеров):

    fee = C * feeRate * p * (1 - p)

где C — число долей, p — цена исполнения. Мейкеры комиссию не платят.
Максимум p(1-p) = 0.25 при p = 0.5, поэтому «потолок» комиссии равен
rate/4 на долю — это и есть публикуемые $1.00 / $1.75 на 100 долей.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

# Ставки тейкера по категориям (docs.polymarket.com, help-центр).
FEE_RATES = {
    "crypto": 0.07,
    "sports": 0.05,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "other": 0.05,
    "finance": 0.04,
    "politics": 0.04,
    "mentions": 0.04,
    "tech": 0.04,
    "geopolitics": 0.0,
}
DEFAULT_FEE_RATE = 0.05


def taker_fee_per_share(p: float, fee_rate: float = DEFAULT_FEE_RATE) -> float:
    """Комиссия тейкера в долларах на одну долю, купленную по цене p."""
    return fee_rate * p * (1.0 - p)


def effective_price(p: float, fee_rate: float = DEFAULT_FEE_RATE) -> float:
    """Полная стоимость доли с комиссией = точка безубыточности по вероятности.

    Покупка выгодна тогда и только тогда, когда истинная вероятность
    исхода выше этого числа.
    """
    return p + taker_fee_per_share(p, fee_rate)


def fee_drag(p: float, fee_rate: float = DEFAULT_FEE_RATE) -> float:
    """Комиссия как доля вложенного капитала: feeRate * (1 - p).

    Для лонгшотов стремится к feeRate (максимум), для p=0.5 — вдвое меньше.
    Дешёвые исходы наказываются комиссией сильнее всего.
    """
    return fee_rate * (1.0 - p)


@dataclass
class Edge:
    """Результат оценки одной ставки."""
    price: float            # цена входа (ask), $
    p_true: float           # оценка истинной вероятности
    fee_rate: float
    p_eff: float            # цена с комиссией = безубыточная вероятность
    edge_prob: float        # p_true - p_eff, в единицах вероятности
    roi: float              # ожидаемая доходность на вложенный доллар
    days: float | None      # до резолва
    roi_annual: float | None
    kelly: float            # доля банка по полному Келли
    win_prob: float         # = p_true, вероятность, что ставка вообще сыграет

    def __str__(self) -> str:
        a = f"{self.roi_annual*100:>8.1f}%/год" if self.roi_annual is not None else "     n/a"
        return (f"цена {self.price:.3f} | безубыток {self.p_eff:.3f} | "
                f"p_true {self.p_true:.3f} | эдж {self.edge_prob:+.3f} | "
                f"ROI {self.roi*100:+7.1f}% | {a} | Келли {self.kelly*100:5.2f}%")


def evaluate(price: float, p_true: float,
             fee_rate: float = DEFAULT_FEE_RATE,
             days: float | None = None) -> Edge:
    """Полная оценка покупки YES по цене `price` при истинной вероятности `p_true`."""
    if not (0.0 < price < 1.0):
        raise ValueError(f"цена вне (0,1): {price}")
    p_eff = effective_price(price, fee_rate)
    roi = p_true / p_eff - 1.0

    roi_annual = None
    if days is not None and days > 0 and roi > -1.0:
        # Геометрическое приведение к году: сколько раз можно прокрутить капитал.
        # На рынках, которым остались часы, показатель улетает в бесконечность
        # и смысла не имеет — поэтому горизонт снизу ограничен сутками,
        # а результат обрезан. Аннуализация тут для сравнения, а не для обещаний.
        d = max(days, 1.0)
        try:
            roi_annual = min((1.0 + roi) ** (365.0 / d) - 1.0, 1e6)
        except OverflowError:
            roi_annual = 1e6

    # Келли: ставим p_eff, получаем $1. Чистые шансы b = (1 - p_eff)/p_eff.
    b = (1.0 - p_eff) / p_eff
    kelly = (p_true * b - (1.0 - p_true)) / b if b > 0 else 0.0

    return Edge(price=price, p_true=p_true, fee_rate=fee_rate, p_eff=p_eff,
                edge_prob=p_true - p_eff, roi=roi, days=days,
                roi_annual=roi_annual, kelly=max(kelly, 0.0), win_prob=p_true)


# ---------------------------------------------------------------- статистика

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Доверительный интервал Уилсона для доли k/n.

    Для калибровки обязателен: нормальный интервал на редких событиях
    (лонгшоты!) даёт бессмыслицу вроде отрицательных границ.
    """
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    d = 1.0 + z * z / n
    c = ph + z * z / (2 * n)
    r = z * sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def n_bets_to_detect(p_market: float, p_true: float,
                     alpha: float = 0.05, power: float = 0.80) -> float:
    """Сколько ставок нужно, чтобы доказать, что эдж реален, а не везение.

    Двухвыборочная задача сводится к проверке доли: наблюдаем k побед из n,
    нулевая гипотеза «частота = p_market», альтернатива «= p_true».
    Односторонний тест.
    """
    if p_true == p_market:
        return float("inf")
    z_a, z_b = 1.6449, 0.8416          # z(0.95) и z(0.80)
    if abs(alpha - 0.05) > 1e-9:
        z_a = 1.9600 if alpha <= 0.025 else 1.2816
    num = z_a * sqrt(p_market * (1 - p_market)) + z_b * sqrt(p_true * (1 - p_true))
    return (num / (p_true - p_market)) ** 2


def prob_no_wins(p_true: float, n: int) -> float:
    """Вероятность не выиграть ни разу за n ставок. Реальность лонгшотов."""
    return (1.0 - p_true) ** n
