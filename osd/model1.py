"""Модель 1 — детекція OSD bbox на все відео.

Вхід — готовий запит `prompt_model1.build_messages(...)` (system + user з few-shot і
кадрами із сіткою). `run_model1` робить `n_parallel` однакових паралельних запитів
(стохастика VLM) через `openrouter.chat_parallel`; відповідь кожного рану строго
парситься `parse_bboxes_response` (непарсабельна → повтор рану).

`prepare_model1_input` — повна підготовка входу з відео (кадри → сітка → content → промпт).
"""

from __future__ import annotations

from pathlib import Path

from .constants import (COORD_RANGE, FIRST_ONLY, MAX_ATTEMPTS, MAX_SIDE, MAX_TOKENS, MODEL_1,
                        N_FRAMES, N_PARALLEL_M1, REASONING_1, TEMPERATURE, TIMEOUT, VIDEOS_DIR)
from .extract_frames import extract_frames
from .frames_to_content import estimate_content_mb, frames_to_content
from .grid import overlay_grid_on_frames
from .openrouter import Log, chat_parallel, summarize
from .prompt_model1 import build_messages, parse_bboxes_response

__all__ = ["prepare_model1_input", "run_model1", "select_run"]


def prepare_model1_input(video: str | Path, videos_dir: str | Path = VIDEOS_DIR,
                         n_frames: int = N_FRAMES, max_side: int | None = MAX_SIDE) -> dict:
    """Відео → {frames (чисті), grid_frames, content (кадри із сіткою), request (повний запит)}."""
    frames = extract_frames(video, n_frames=n_frames, max_side=max_side, videos_dir=videos_dir)
    grid_frames = overlay_grid_on_frames(frames)
    content = frames_to_content(grid_frames)
    request = build_messages(content, n_frames=len(frames))
    return {"frames": frames, "grid_frames": grid_frames, "content": content, "request": request,
            "content_mb": estimate_content_mb(request["content"])}


def run_model1(request: dict, n_parallel: int = N_PARALLEL_M1, first_only: bool = FIRST_ONLY,
               model: str = MODEL_1, reasoning_effort: str | None = REASONING_1,
               max_attempts: int = MAX_ATTEMPTS, temperature: float | None = TEMPERATURE,
               max_tokens: int | None = MAX_TOKENS, timeout: int = TIMEOUT,
               coord_range: int = COORD_RANGE, api_key: str | None = None,
               log: Log | None = None) -> dict:
    """N паралельних ранів моделі 1.

    Повертає {model, reasoning_effort, runs, summary}; кожен ран — результат
    `chat_parallel` + `bboxes` ([{x_min, y_min, x_max, y_max, label}] у шкалі 0..1000,
    [] для невдалого рану або відео без OSD).
    """
    runs = chat_parallel(
        request["messages"], model=model, n_parallel=n_parallel, first_only=first_only,
        validate=lambda text: parse_bboxes_response(text, coord_range=coord_range),
        max_attempts=max_attempts, reasoning_effort=reasoning_effort, temperature=temperature,
        max_tokens=max_tokens, timeout=timeout, api_key=api_key, log=log)
    for r in runs:
        r["bboxes"] = list(r["parsed"]) if r["ok"] else []
    return {"model": model, "reasoning_effort": reasoning_effort, "first_only": first_only,
            "runs": runs, "summary": summarize(runs)}


def select_run(result: dict, index: int | None = None) -> dict:
    """Ран за позицією в `result["runs"]` (None = перший успішний; помилка, якщо їх немає)."""
    runs = result["runs"]
    if index is not None:
        if not 0 <= index < len(runs):
            raise IndexError(f"індекс рану {index} поза 0..{len(runs) - 1}")
        run = runs[index]
        if not run["ok"]:
            raise RuntimeError(f"ран #{run['run']} невдалий: {run['error']}")
        return run
    for run in runs:
        if run["ok"]:
            return run
    raise RuntimeError("жоден ран не вдався: " + "; ".join(
        f"#{r['run']}: {(r['error'] or '')[:200]}" for r in runs))
