"""Отправка уведомлений в Telegram.

Токен бота и chat_id читаются из переменных окружения или из файла .env
рядом с проектом. В коде и в репозитории их нет и быть не должно.

    TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
    TELEGRAM_CHAT_ID=987654321
"""
from __future__ import annotations

import html
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.telegram.org/bot{token}/{method}"


def load_env(path: Path | None = None) -> None:
    """Простейший .env-лоадер: KEY=VALUE, строки с # игнорируются."""
    p = path or (ROOT / ".env")
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Telegram:
    def __init__(self, token: str | None = None, chat_id: str | None = None):
        load_env()
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        if not self.token:
            raise RuntimeError(
                "нет TELEGRAM_BOT_TOKEN. Создай бота у @BotFather и положи токен "
                "в файл .env (см. .env.example)")

    def _call(self, method: str, **params):
        r = requests.post(API.format(token=self.token, method=method),
                          json=params, timeout=30)
        j = r.json()
        if not j.get("ok"):
            raise RuntimeError(f"telegram {method}: {j.get('description')}")
        return j["result"]

    def me(self) -> dict:
        return self._call("getMe")

    def updates(self, offset: int | None = None) -> list[dict]:
        p = {"timeout": 0}
        if offset is not None:
            p["offset"] = offset
        return self._call("getUpdates", **p)

    def send(self, text: str, chat_id: str | None = None,
             preview: bool = False) -> dict:
        cid = chat_id or self.chat_id
        if not cid:
            raise RuntimeError("нет TELEGRAM_CHAT_ID — запусти scripts/setup_telegram.py")
        return self._call("sendMessage", chat_id=cid, text=text,
                          parse_mode="HTML",
                          link_preview_options={"is_disabled": not preview})


def esc(s: str) -> str:
    return html.escape(str(s), quote=False)
