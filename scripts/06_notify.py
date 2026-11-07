"""Прогон скринера и отправка новых сигналов в Telegram.

Запускается по расписанию. Главное требование — НЕ спамить: один и тот же
сигнал висит на рынке часами, а проверка идёт каждые 5-10 минут. Поэтому
ведётся файл состояния с уже отправленными сигналами; повторно сигнал уходит
только если эдж заметно вырос или прошло достаточно времени.

Тишина — нормальный режим работы. Если сигналов нет, ничего не отправляется.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmlib.api import Api  # noqa: E402
from pmlib.notify import Telegram, esc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

screen = import_module("05_screen")

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data/out/notified.json"

REPEAT_AFTER_H = 12.0     # через сколько часов можно напомнить о том же сигнале
EDGE_JUMP = 0.02          # либо если эдж вырос на 2 п.п. — это новая ситуация
MAX_PER_RUN = 6           # больше шести сообщений за прогон — это спам


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    # чистим записи старше недели, чтобы файл не рос вечно
    cutoff = time.time() - 7 * 86400
    st = {k: v for k, v in st.items() if v.get("t", 0) > cutoff}
    STATE.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")


def is_new(st: dict, key: str, edge: float) -> bool:
    prev = st.get(key)
    if prev is None:
        return True
    if (time.time() - prev.get("t", 0)) > REPEAT_AFTER_H * 3600:
        return True
    return edge - prev.get("edge", 0.0) >= EDGE_JUMP


def fmt(s) -> str:
    d = f"{s.days:.0f} дн." if s.days else "срок неизвестен"
    kelly_stake = ""
    if s.kelly > 0:
        kelly_stake = f"\n<b>Келли:</b> {s.kelly*100:.2f}% банка"
    return (
        f"<b>{esc(s.event)}</b>\n"
        f"Исход: <b>{esc(s.outcome)}</b>\n\n"
        f"<b>Режим:</b> {esc(s.kind)}"
        + (" (лимитная заявка, комиссии нет)" if s.kind == "мейкер"
           else " (покупка по рынку)")
        + f"\n<b>Цена входа:</b> {s.price:.3f}\n"
        f"<b>Оценка вероятности:</b> {s.fair:.3f}\n"
        f"<b>Эдж:</b> {s.edge*100:+.1f} п.п.  |  <b>ROI:</b> {s.roi*100:+.0f}%\n"
        f"<b>До конца:</b> {d}\n"
        f"<b>Основание:</b> {esc(s.basis)}\n"
        f"<b>Глубина стакана:</b> ${s.depth_usd:,.0f} | спред {s.spread:.3f}\n"
        f"<b>Размер:</b> ${s.stake:.2f}"
        + kelly_stake
        + f"\n\n{esc(s.url)}"
    )


def digest(sigs, n: int = 5) -> str:
    """Ежедневная сводка «лучшее из имеющегося».

    Нужна ровно затем, чтобы не приходилось опускать порог сигналов. Порог
    отделяет сделки с положительным ожиданием от всех остальных; если его
    снизить ради регулярных уведомлений, бот начнёт присылать предложения,
    на которых теряют деньги. Поэтому сводка идёт ОТДЕЛЬНЫМ потоком и прямо
    помечена как не-рекомендация.
    """
    head = ("<b>Сводка за сутки</b>\n"
            "Ниже — лучшее, что нашлось на рынке. Порог не пройден ни одним "
            "из них, это <b>не предложения о покупке</b>, а картина дня.\n")
    if not sigs:
        return head + "\nСегодня не нашлось вообще ничего. Так тоже бывает."
    rows = []
    for s in sigs[:n]:
        d = f"{s.days:.0f} дн." if s.days else "?"
        rows.append(f"\n<b>{esc(s.outcome)}</b> — {esc(s.event)}\n"
                    f"вход {s.price:.3f}, оценка {s.fair:.3f}, "
                    f"эдж {s.edge*100:+.1f} п.п. ({esc(s.kind)}), до конца {d}\n"
                    f"{esc(s.url)}")
    return head + "".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stake", type=float, default=screen.STAKE_USD)
    ap.add_argument("--min-edge", type=float, default=screen.MIN_EDGE_TAKER)
    ap.add_argument("--dry-run", action="store_true",
                    help="посчитать и напечатать, но ничего не отправлять")
    ap.add_argument("--digest", action="store_true",
                    help="прислать сводку лучшего за сутки, даже если порог не пройден")
    args = ap.parse_args()

    if args.digest:
        a, b = screen.load_fit()
        api = Api(min_interval=0.04)
        sigs = screen.scan(api, a, b, stake=args.stake,
                           min_taker=0.0, min_maker=0.0)
        text = digest(sigs)
        if args.dry_run:
            print(text)
            return 0
        Telegram().send(text)
        print(f"сводка отправлена ({len(sigs)} кандидатов рассмотрено)")
        return 0

    a, b = screen.load_fit()
    api = Api(min_interval=0.04)
    sigs = screen.scan(api, a, b, stake=args.stake,
                       min_taker=args.min_edge,
                       min_maker=max(args.min_edge, screen.MIN_EDGE_MAKER))
    print(f"сигналов найдено: {len(sigs)}")

    st = load_state()
    fresh = [s for s in sigs if is_new(st, f"{s.slug}|{s.outcome}|{s.kind}", s.edge)]
    fresh = fresh[:MAX_PER_RUN]
    print(f"из них новых: {len(fresh)}")

    if not fresh:
        print("отправлять нечего — тишина это нормально")
        return 0

    if args.dry_run:
        for s in fresh:
            print("\n---\n" + fmt(s))
        return 0

    tg = Telegram()
    for s in fresh:
        tg.send(fmt(s))
        st[f"{s.slug}|{s.outcome}|{s.kind}"] = {"t": time.time(), "edge": s.edge}
        print(f"отправлено: {s.outcome[:40]} эдж {s.edge*100:+.1f}пп")
    save_state(st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
