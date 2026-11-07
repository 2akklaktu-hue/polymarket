"""Одноразовая настройка Telegram: проверяет токен и узнаёт твой chat_id.

Порядок действий:
  1. В Telegram напиши @BotFather -> /newbot -> придумай имя.
  2. Скопируй выданный токен в файл .env:  TELEGRAM_BOT_TOKEN=...
  3. Найди своего бота в Telegram и нажми Start (или напиши ему любое слово).
  4. Запусти этот скрипт — он допишет TELEGRAM_CHAT_ID в .env и пришлёт тест.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmlib.notify import Telegram, load_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    load_env()
    try:
        tg = Telegram()
    except RuntimeError as e:
        print("!", e)
        return 1

    me = tg.me()
    print(f"бот на связи: @{me.get('username')} ({me.get('first_name')})")

    ups = tg.updates()
    chats = {}
    for u in ups:
        msg = u.get("message") or u.get("channel_post") or {}
        ch = msg.get("chat") or {}
        if ch.get("id"):
            chats[str(ch["id"])] = ch.get("username") or ch.get("first_name") or ch.get("title")

    if not chats:
        # Telegram отдаёт обновления только 24 часа и только те, что ещё
        # не считаны другим процессом. Если chat_id уже известен из .env,
        # опрос не нужен — пробуем отправить напрямую.
        import os
        known = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not known:
            print("! боту никто не писал, и chat_id в .env пуст.")
            print("  Открой бота в Telegram, напиши ему любое слово "
                  "(не просто Start) и запусти скрипт снова.")
            return 1
        print(f"обновлений нет, пробую известный chat_id из .env: {known}")
        try:
            tg.send("<b>Polymarket-скринер подключён.</b>\n"
                    "Сюда будут падать найденные расхождения цены и оценки "
                    "вероятности.", chat_id=known)
        except RuntimeError as e:
            print(f"! не вышло: {e}")
            print("  Скорее всего chat_id неверный. Напиши боту любое слово "
                  "и запусти скрипт снова — он определит адрес сам.")
            return 1
        print("тестовое сообщение отправлено — проверь Telegram")
        return 0

    cid, who = list(chats.items())[-1]
    print(f"нашёл чат: {cid} ({who})")

    env = ROOT / ".env"
    text = env.read_text(encoding="utf-8") if env.exists() else ""
    if "TELEGRAM_CHAT_ID" not in text:
        text = text.rstrip("\n") + f"\nTELEGRAM_CHAT_ID={cid}\n"
        env.write_text(text, encoding="utf-8")
        print(f"записал TELEGRAM_CHAT_ID в {env}")

    tg.send("<b>Polymarket-скринер подключён.</b>\n"
            "Сюда будут падать найденные расхождения цены и оценки вероятности.",
            chat_id=cid)
    print("тестовое сообщение отправлено — проверь Telegram")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
