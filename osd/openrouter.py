"""Виклики OpenRouter Chat Completions: один запит з повторами + N паралельних ранів.

- `chat_one(messages, ...)` — один запит. Тимчасові помилки (429 / 5xx / таймаут / мережа)
  повторюються з короткою паузою (джитер), доки не вдасться або не мине `max_wait_s`.
  Одночасних запитів до однієї моделі — не більше `MAX_INFLIGHT_PER_MODEL` на весь процес.
  Чому так: `qwen/qwen3.8-flash` має одного провайдера (Alibaba) і часто віддає 429
  `upstream_provider_shared_pool` — вичерпаний спільний пул OpenRouter. Кожна спроба —
  незалежна «лотерея», тож довгий експоненційний backoff не допомагає. Постійне рішення —
  власний ключ провайдера (BYOK) в OpenRouter → Settings → Integrations.
- `chat_parallel(messages, n_parallel, first_only, validate, ...)` — N однакових паралельних
  запитів (стохастика VLM). Відповідь кожного рану перевіряється `validate(content)`
  (парсер; виняток = невдала спроба → повтор лише цього рану до `max_attempts`).
    * `first_only=False` — чекати всі N ранів і повернути всі (у порядку індексів);
    * `first_only=True`  — повернути ПЕРШИЙ успішний ран і не чекати решту. Уже відправлені
      HTTP-запити неможливо відкликати: вони завершаться у фоні (і можуть бути протарифіковані),
      але нові спроби/повтори не стартують. Якщо успішних немає — повертаються всі (невдалі) рани.
- `summarize(runs)` — середні час і ціна по ранах.

API-ключ: аргумент `api_key` → змінна оточення `OPENROUTER_API_KEY` → файл `final_solution/.env`.
"""

from __future__ import annotations

import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable

import requests
from dotenv import load_dotenv

from .constants import (API_KEY_ENV, ENV_FILE, MAX_ATTEMPTS, MAX_INFLIGHT_PER_MODEL, MAX_TOKENS,
                        OPENROUTER_URL, RATE_LIMIT_DELAY_MAX_S, RATE_LIMIT_DELAY_S,
                        RATE_LIMIT_MAX_WAIT_S, TEMPERATURE, TIMEOUT)

__all__ = ["chat_one", "chat_parallel", "summarize", "set_api_key", "get_api_key",
           "build_messages", "TRANSIENT_STATUS"]

TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
RUN_BACKOFF_S = 3.0          # пауза перед повтором рану з непарсабельною відповіддю (×2 далі)
RUN_BACKOFF_MAX_S = 30.0

Messages = list[dict[str, Any]]
Log = Callable[[str], None]


# ---------------------------------------------------------------------------
# API-ключ / messages
# ---------------------------------------------------------------------------

def set_api_key(api_key: str | None) -> None:
    """Задати ключ для процесу (порожній/None — нічого не змінювати, ключ візьметься з .env)."""
    key = (api_key or "").strip()
    if key:
        os.environ[API_KEY_ENV] = key


def get_api_key(api_key: str | None = None) -> str:
    key = (api_key or "").strip()
    if not key:
        if ENV_FILE.exists():
            load_dotenv(ENV_FILE, override=False)
        key = os.environ.get(API_KEY_ENV, "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError(f"не знайдено {API_KEY_ENV}: передай api_key, задай змінну оточення "
                           f"або додай рядок {API_KEY_ENV}=sk-or-v1-... у {ENV_FILE}")
    return key


def build_messages(system_prompt: str | None, user_prompt: str | None,
                   content: list[dict] | None = None) -> Messages:
    """system + user (текст інструкції, потім content: text/image_url блоки)."""
    messages: Messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    user: list[dict] = []
    if user_prompt:
        user.append({"type": "text", "text": user_prompt})
    user.extend(content or [])
    messages.append({"role": "user", "content": user})
    return messages


# ---------------------------------------------------------------------------
# Обмеження одночасних запитів до моделі
# ---------------------------------------------------------------------------

class _Gate:
    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.inflight = 0

    def acquire(self, stop: threading.Event | None) -> bool:
        with self.cond:
            while self.inflight >= max(1, MAX_INFLIGHT_PER_MODEL):
                if stop is not None and stop.is_set():
                    return False
                self.cond.wait(0.5)
            self.inflight += 1
            return True

    def release(self) -> None:
        with self.cond:
            self.inflight -= 1
            self.cond.notify()


_gates: dict[str, _Gate] = {}
_gates_lock = threading.Lock()


def _gate(model: str) -> _Gate:
    with _gates_lock:
        return _gates.setdefault(model, _Gate())


# ---------------------------------------------------------------------------
# Один запит
# ---------------------------------------------------------------------------

def _empty_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "reasoning_tokens": 0, "cached_tokens": 0, "cost": 0.0}


def _extract_usage(data: dict) -> dict:
    u = data.get("usage") or {}
    return {
        "prompt_tokens": u.get("prompt_tokens", 0) or 0,
        "completion_tokens": u.get("completion_tokens", 0) or 0,
        "total_tokens": u.get("total_tokens", 0) or 0,
        "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
        "cost": float(u.get("cost", 0.0) or 0.0),
    }


def _result(ok: bool, model: str, latency: float, *, content=None, reasoning=None, data=None,
            error: str | None = None, transient: bool = False) -> dict:
    data = data or {}
    choice = (data.get("choices") or [{}])[0] if isinstance(data, dict) else {}
    return {
        "ok": ok,
        "content": content,
        "reasoning": reasoning,
        "model": data.get("model", model) if isinstance(data, dict) else model,
        "id": data.get("id") if isinstance(data, dict) else None,
        "finish_reason": choice.get("finish_reason"),
        "usage": _extract_usage(data) if isinstance(data, dict) and data else _empty_usage(),
        "latency_s": round(latency, 3),
        "error": error,
        "transient": transient,
    }


def _post_once(payload: dict, headers: dict, timeout: int) -> dict:
    """Один HTTP-запит → результат (`transient=True` — помилку варто повторити)."""
    model = payload["model"]
    t0 = time.perf_counter()
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    except (requests.Timeout, requests.ConnectionError) as e:
        return _result(False, model, time.perf_counter() - t0,
                       error=f"{type(e).__name__}: {e}", transient=True)
    except requests.RequestException as e:
        return _result(False, model, time.perf_counter() - t0, error=f"{type(e).__name__}: {e}")
    latency = time.perf_counter() - t0
    if resp.status_code != 200:
        return _result(False, model, latency, error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                       transient=resp.status_code in TRANSIENT_STATUS)
    try:
        data = resp.json()
    except ValueError:
        return _result(False, model, latency, error=f"не-JSON відповідь: {resp.text[:300]}",
                       transient=True)
    # OpenRouter інколи віддає HTTP 200 з {"error": {...}} (помилка провайдера)
    choice = (data.get("choices") or [{}])[0]
    err = data.get("error") or choice.get("error")
    if err:
        code = err.get("code") if isinstance(err, dict) else None
        msg = (f"{code or ''} {err.get('message', '')} {err.get('metadata', '') or ''}".strip()
               if isinstance(err, dict) else str(err))
        try:
            transient = int(code) in TRANSIENT_STATUS
        except (TypeError, ValueError):
            transient = True  # невідома помилка провайдера — варто повторити
        return _result(False, model, latency, data=data, error=f"provider: {msg}", transient=transient)
    msg = choice.get("message") or {}
    return _result(True, model, latency, content=msg.get("content"),
                   reasoning=msg.get("reasoning"), data=data)


def chat_one(messages: Messages, model: str, reasoning_effort: str | None = None,
             temperature: float | None = TEMPERATURE, max_tokens: int | None = MAX_TOKENS,
             timeout: int = TIMEOUT, api_key: str | None = None,
             extra_body: dict | None = None, max_wait_s: float = RATE_LIMIT_MAX_WAIT_S,
             stop: threading.Event | None = None) -> dict:
    """Один запит з повторами тимчасових помилок.

    Повертає {ok, content, reasoning, model, id, finish_reason, usage{..., cost},
    latency_s (останньої спроби), error, transient_retries, wall_s, cost_total
    (сума ціни всіх спроб)}.
    """
    payload: dict[str, Any] = {"model": model, "messages": messages, "usage": {"include": True}}
    payload["reasoning"] = ({"effort": reasoning_effort, "exclude": False}
                            if reasoning_effort else {"exclude": False})
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if extra_body:
        payload.update(extra_body)
    headers = {"Authorization": f"Bearer {get_api_key(api_key)}",
               "Content-Type": "application/json"}

    gate = _gate(model)
    t0 = time.monotonic()
    if not gate.acquire(stop):
        r = _result(False, model, 0.0, error="cancelled")
        r.update(transient_retries=0, wall_s=0.0, cost_total=0.0)
        return r
    n_retries, delay, cost_total = 0, RATE_LIMIT_DELAY_S, 0.0
    try:
        while True:
            r = _post_once(payload, headers, timeout)
            cost_total += r["usage"]["cost"]
            if r["ok"] or not r["transient"]:
                break
            pause = delay * random.uniform(0.7, 1.3)
            if time.monotonic() - t0 + pause > max_wait_s or (stop is not None and stop.is_set()):
                break
            n_retries += 1
            if stop is not None:
                if stop.wait(pause):
                    break
            else:
                time.sleep(pause)
            delay = min(RATE_LIMIT_DELAY_MAX_S, delay * 1.5)
    finally:
        gate.release()
    r.update(transient_retries=n_retries, wall_s=round(time.monotonic() - t0, 2),
             cost_total=round(cost_total, 8))
    return r


# ---------------------------------------------------------------------------
# N паралельних ранів
# ---------------------------------------------------------------------------

def _run_slot(i: int, messages: Messages, model: str, validate: Callable[[str], Any] | None,
              max_attempts: int, stop: threading.Event, log: Log | None, kw: dict) -> dict:
    """Один ран: запит → validate; невдача → повтор (до `max_attempts`)."""
    t0 = time.monotonic()
    attempts: list[dict] = []
    r: dict = {}
    parsed, err = None, "not started"
    for attempt in range(1, max(1, max_attempts) + 1):
        if stop.is_set():
            err = err if attempts else "cancelled"
            break
        r = chat_one(messages, model=model, stop=stop, **kw)
        if not r["ok"]:
            err = f"API: {r['error']} [finish={r['finish_reason']}]"
        else:
            try:
                parsed = validate(r["content"] or "") if validate else r["content"]
                err = None
            except Exception as e:  # noqa: BLE001 — будь-яка помилка парсера = невдала спроба
                err = (f"parse: {e} [finish={r['finish_reason']}, "
                       f"content={(r['content'] or '')[:200]!r}]")
        attempts.append({"attempt": attempt, "ok": err is None, "error": err,
                         "latency_s": r["latency_s"], "wall_s": r["wall_s"],
                         "transient_retries": r["transient_retries"], "cost": r["cost_total"],
                         "usage": r["usage"], "finish_reason": r["finish_reason"]})
        if err is None:
            break
        if attempt < max_attempts and not stop.is_set():
            if log:
                log(f"⚠️ ран #{i}: {err[:160]} → спроба {attempt + 1}/{max_attempts}")
            stop.wait(min(RUN_BACKOFF_MAX_S, RUN_BACKOFF_S * 2 ** (attempt - 1)))
    usage = [a["usage"] for a in attempts]
    run = {
        "run": i,
        "ok": err is None,
        "parsed": parsed,
        "error": err,
        "content": r.get("content"),
        "reasoning": r.get("reasoning"),
        "model": r.get("model", model),
        "finish_reason": r.get("finish_reason"),
        "time_s": round(time.monotonic() - t0, 2),          # повний час рану (з усіма повторами)
        "latency_s": r.get("latency_s"),                    # час останнього HTTP-запиту
        "cost_usd": round(sum(a["cost"] for a in attempts), 8),
        "prompt_tokens": sum(u["prompt_tokens"] for u in usage),
        "completion_tokens": sum(u["completion_tokens"] for u in usage),
        "reasoning_tokens": sum(u["reasoning_tokens"] for u in usage),
        "n_attempts": len(attempts),
        "transient_retries": sum(a["transient_retries"] for a in attempts),
        "attempts": attempts,
    }
    if log and not stop.is_set():  # покинуті у режимі first_only рани не логуються
        status = "✅" if run["ok"] else "❌"
        log(f"{status} ран #{i}: {run['time_s']:.1f} c, ${run['cost_usd']:.5f}, "
            f"спроб {run['n_attempts']}, повторів 429/5xx {run['transient_retries']}"
            + ("" if run["ok"] else f" · {err[:160]}"))
    return run


def chat_parallel(messages: Messages, model: str, n_parallel: int = 1, first_only: bool = False,
                  validate: Callable[[str], Any] | None = None,
                  max_attempts: int = MAX_ATTEMPTS, reasoning_effort: str | None = None,
                  temperature: float | None = TEMPERATURE, max_tokens: int | None = MAX_TOKENS,
                  timeout: int = TIMEOUT, api_key: str | None = None,
                  extra_body: dict | None = None, log: Log | None = None) -> list[dict]:
    """`n_parallel` однакових паралельних ранів одного запиту.

    Кожен ран: {run, ok, parsed (результат validate), error, content, reasoning, time_s,
    latency_s, cost_usd, *_tokens, n_attempts, transient_retries, attempts}.
    `first_only=False` → усі рани (за індексом); `first_only=True` → [перший успішний]
    (або всі рани, якщо жоден не вдався).
    """
    n = max(1, int(n_parallel))
    kw = dict(reasoning_effort=reasoning_effort, temperature=temperature, max_tokens=max_tokens,
              timeout=timeout, api_key=get_api_key(api_key), extra_body=extra_body)
    stop = threading.Event()
    ex = ThreadPoolExecutor(max_workers=n, thread_name_prefix="openrouter")
    futures = [ex.submit(_run_slot, i, messages, model, validate, max_attempts, stop, log, kw)
               for i in range(n)]
    if not first_only:
        try:
            return [f.result() for f in futures]
        finally:
            ex.shutdown(wait=True)

    pending, done_runs = set(futures), []
    try:
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                run = f.result()
                if run["ok"]:
                    stop.set()  # решта ранів не стартує нових спроб; in-flight дорахуються у фоні
                    if log and pending:
                        log(f"🏁 перший успішний — ран #{run['run']}, решту ({len(pending)}) не чекаємо")
                    return [run]
                done_runs.append(run)
        return sorted(done_runs, key=lambda r: r["run"])
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def summarize(runs: list[dict]) -> dict:
    """Середні час / ціна по ранах (+ сума ціни, кількість успішних)."""
    n = len(runs)
    ok = [r for r in runs if r.get("ok")]
    return {
        "n_runs": n,
        "n_ok": len(ok),
        "mean_time_s": round(sum(r["time_s"] for r in runs) / n, 2) if n else 0.0,
        "mean_cost_usd": round(sum(r["cost_usd"] for r in runs) / n, 6) if n else 0.0,
        "total_cost_usd": round(sum(r["cost_usd"] for r in runs), 6),
        "max_time_s": round(max((r["time_s"] for r in runs), default=0.0), 2),
    }
