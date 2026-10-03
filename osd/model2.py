"""Модель 2 — зчитування значень OSD-телеметрії з кадрів із bbox моделі 1.

`prepare_model2_input`: на ЧИСТІ кадри наносяться bbox вибраного рану моделі 1 з
непрозорістю `BBOX_OPACITY` (крізь рамки видно OSD) → запит `prompt_model2.build_messages`
(три кадри + каталог назв bbox; координати моделі НЕ передаються).
`run_model2`: N паралельних ранів; відповідь кожного строго парситься (усі ключі рівно раз,
bbox_name лише з каталогу), далі bbox фільтруються до тих, з яких зчитано хоч одне значення.
"""

from __future__ import annotations

from .bbox_draw import draw_bboxes_on_frames
from .constants import (BBOX_OPACITY, FIRST_ONLY, MAX_ATTEMPTS, MAX_TOKENS, MODEL_2,
                        N_PARALLEL_M2, REASONING_2, TEMPERATURE, TIMEOUT)
from .openrouter import Log, chat_parallel, summarize
from .prompt_model2 import KEY_NAMES, build_messages, parse_readings_response, readings_to_csv

__all__ = ["prepare_model2_input", "run_model2", "readings_to_dict", "filter_used_bboxes",
           "empty_readings"]


def prepare_model2_input(frames: list[dict], bboxes: list[dict],
                         opacity: float = BBOX_OPACITY) -> dict:
    """Чисті кадри + bbox моделі 1 → {annotated_frames, request, bboxes}."""
    annotated = draw_bboxes_on_frames(frames, bboxes, opacity=opacity)
    names = [b["label"] for b in bboxes if b.get("label")]
    return {"annotated_frames": annotated, "request": build_messages(annotated, names),
            "bboxes": list(bboxes), "opacity": opacity}


def empty_readings(n_frames: int = 3) -> list[dict]:
    return [{"key": k, "values": [None] * n_frames, "bbox_name": None} for k in KEY_NAMES]


def readings_to_dict(readings: list[dict]) -> dict:
    """[{key, values, bbox_name}] → {key: {"values": [v0, v1, v2], "bbox_name": ...}} (усі ключі)."""
    return {r["key"]: {"values": list(r["values"]), "bbox_name": r["bbox_name"]} for r in readings}


def filter_used_bboxes(bboxes: list[dict], readings: list[dict]) -> tuple[list[dict], list[str]]:
    """Лише bbox, з яких зчитано хоч одне не-null значення → (filtered_bboxes, used_names)."""
    used = {r["bbox_name"] for r in readings
            if r.get("bbox_name") and any(v is not None for v in r["values"])}
    return [b for b in bboxes if b.get("label") in used], sorted(used)


def _finish_run(run: dict, bboxes: list[dict]) -> dict:
    readings = run["parsed"] if run["ok"] else []
    run["readings"] = readings
    run["dict"] = readings_to_dict(readings)
    run["csv"] = readings_to_csv(readings) if readings else ""
    run["filtered_bboxes"], run["used_bbox_names"] = filter_used_bboxes(bboxes, readings)
    return run


def _no_bbox_run() -> dict:
    """Модель 1 не знайшла OSD → запит не робиться, усі ключі null."""
    readings = empty_readings()
    return {"run": 0, "ok": True, "parsed": readings, "error": None, "content": None,
            "reasoning": None, "model": None, "finish_reason": None, "time_s": 0.0,
            "latency_s": 0.0, "cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
            "reasoning_tokens": 0, "n_attempts": 0, "transient_retries": 0, "attempts": [],
            "skipped": "модель 1 не знайшла жодного bbox — запит не робився"}


def run_model2(prepared: dict, n_parallel: int = N_PARALLEL_M2, first_only: bool = FIRST_ONLY,
               model: str = MODEL_2, reasoning_effort: str | None = REASONING_2,
               max_attempts: int = MAX_ATTEMPTS, temperature: float | None = TEMPERATURE,
               max_tokens: int | None = MAX_TOKENS, timeout: int = TIMEOUT,
               api_key: str | None = None, log: Log | None = None) -> dict:
    """N паралельних ранів моделі 2 на виході `prepare_model2_input`.

    Кожен ран — результат `chat_parallel` + `readings` ([{key, values, bbox_name}] по всіх
    ключах), `dict` ({key: {values, bbox_name}}), `csv`, `filtered_bboxes`, `used_bbox_names`.
    """
    bboxes = prepared["bboxes"]
    names = prepared["request"]["bbox_names"]
    if not bboxes:
        runs = [_no_bbox_run()]
    else:
        runs = chat_parallel(
            prepared["request"]["messages"], model=model, n_parallel=n_parallel,
            first_only=first_only,
            validate=lambda text: parse_readings_response(text, bbox_names=names),
            max_attempts=max_attempts, reasoning_effort=reasoning_effort,
            temperature=temperature, max_tokens=max_tokens, timeout=timeout, api_key=api_key,
            log=log)
    runs = [_finish_run(r, bboxes) for r in runs]
    return {"model": model, "reasoning_effort": reasoning_effort, "first_only": first_only,
            "runs": runs, "summary": summarize(runs)}
