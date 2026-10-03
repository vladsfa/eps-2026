"""Повний інференс без візуалізації: відео → bbox моделі 1, bbox моделі 2, словник моделі 2.

    from osd.pipeline import infer_video
    res = infer_video("IMG_3812.MP4")
    res["m1_bboxes"], res["m2_bboxes"], res["readings"]

`readings` — {key: {"values": [v_кадр0, v_кадр1, v_кадр2], "bbox_name": str|None}} по ВСІХ
REQUIRED_KEYS (null → None). Результат також містить кадри, усі рани обох моделей і
середні час/ціну — цього достатньо для `viz_results.show_results`.

CLI:
    python -m osd.pipeline IMG_3812.MP4
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

from .constants import (BATCH_DIR, FIRST_ONLY, MAX_ATTEMPTS, MAX_SIDE, MODEL_1, MODEL_2,
                        N_FRAMES, N_PARALLEL_M1, N_PARALLEL_M2, REASONING_1, REASONING_2,
                        VIDEOS_DIR)
from .model1 import prepare_model1_input, run_model1, select_run
from .model2 import prepare_model2_input, run_model2
from .openrouter import Log

__all__ = ["infer_video", "assemble_result", "save_result", "load_result", "result_dir",
           "is_cached"]

RESULT_PKL = "result.pkl"
RESULT_JSON = "result.json"


def _run_index(result: dict, run: dict) -> int:
    return next(i for i, r in enumerate(result["runs"]) if r is run)


def assemble_result(video: str, frames: list[dict], m1: dict, m2: dict,
                    selected_run_m1: int, selected_run_m2: int,
                    elapsed_s: float | None = None) -> dict:
    """Зібрати підсумок пайплайну (той самий формат і для ноутбука, і для `infer_video`).

    `selected_run_*` — позиції ранів у `m1["runs"]` / `m2["runs"]`.
    """
    r1, r2 = m1["runs"][selected_run_m1], m2["runs"][selected_run_m2]
    return {
        "video": Path(video).name,
        "frames": [{k: f[k] for k in ("index", "time_sec", "timestamp", "width", "height",
                                      "jpeg_bytes", "image")} for f in frames],
        "m1_bboxes": r1["bboxes"],
        "m2_bboxes": r2["filtered_bboxes"],
        "readings": r2["dict"],
        "readings_csv": r2["csv"],
        "selected_run_m1": selected_run_m1,
        "selected_run_m2": selected_run_m2,
        "m1": m1,
        "m2": m2,
        "elapsed_s": elapsed_s,
    }


def infer_video(video: str | Path, videos_dir: str | Path = VIDEOS_DIR,
                n_parallel_m1: int = N_PARALLEL_M1, n_parallel_m2: int = N_PARALLEL_M2,
                first_only: bool = FIRST_ONLY, model_1: str = MODEL_1,
                reasoning_1: str | None = REASONING_1, model_2: str = MODEL_2,
                reasoning_2: str | None = REASONING_2, max_attempts: int = MAX_ATTEMPTS,
                n_frames: int = N_FRAMES, max_side: int | None = MAX_SIDE,
                api_key: str | None = None, log: Log | None = None) -> dict:
    """Відео → модель 1 (перший успішний ран) → модель 2 (перший успішний ран) → підсумок.

    Кидає RuntimeError, якщо жоден ран моделі 1 або моделі 2 не вдався.
    """
    t0 = time.perf_counter()
    inp = prepare_model1_input(video, videos_dir=videos_dir, n_frames=n_frames, max_side=max_side)
    m1 = run_model1(inp["request"], n_parallel=n_parallel_m1, first_only=first_only,
                    model=model_1, reasoning_effort=reasoning_1, max_attempts=max_attempts,
                    api_key=api_key, log=log)
    run1 = select_run(m1)
    prepared = prepare_model2_input(inp["frames"], run1["bboxes"])
    m2 = run_model2(prepared, n_parallel=n_parallel_m2, first_only=first_only, model=model_2,
                    reasoning_effort=reasoning_2, max_attempts=max_attempts, api_key=api_key,
                    log=log)
    run2 = select_run(m2)
    return assemble_result(str(video), inp["frames"], m1, m2, _run_index(m1, run1),
                           _run_index(m2, run2), elapsed_s=round(time.perf_counter() - t0, 2))


# ---------------------------------------------------------------------------
# Кеш результатів (batch-ноутбук)
# ---------------------------------------------------------------------------

def result_dir(video: str | Path, base: str | Path = BATCH_DIR) -> Path:
    return Path(base) / Path(video).stem


def is_cached(video: str | Path, base: str | Path = BATCH_DIR) -> bool:
    return (result_dir(video, base) / RESULT_PKL).exists()


def _json_view(res: dict) -> dict:
    """Читабельна частина результату (без кадрів і сирих відповідей)."""
    def runs(m):
        return [{k: r.get(k) for k in ("run", "ok", "error", "time_s", "latency_s", "cost_usd",
                                       "n_attempts", "transient_retries")} for r in m["runs"]]
    return {
        "video": res["video"],
        "m1_bboxes": res["m1_bboxes"],
        "m2_bboxes": res["m2_bboxes"],
        "readings": res["readings"],
        "selected_run_m1": res["selected_run_m1"],
        "selected_run_m2": res["selected_run_m2"],
        "elapsed_s": res["elapsed_s"],
        "m1": {"model": res["m1"]["model"], "summary": res["m1"]["summary"], "runs": runs(res["m1"])},
        "m2": {"model": res["m2"]["model"], "summary": res["m2"]["summary"], "runs": runs(res["m2"])},
        "frames": [{k: f[k] for k in ("index", "time_sec", "timestamp", "width", "height")}
                   for f in res["frames"]],
    }


def save_result(res: dict, base: str | Path = BATCH_DIR) -> Path:
    """Зберегти результат у `<base>/<video>/` (result.pkl — повний, result.json — читабельний)."""
    out = result_dir(res["video"], base)
    out.mkdir(parents=True, exist_ok=True)
    (out / RESULT_JSON).write_text(json.dumps(_json_view(res), ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    tmp = out / (RESULT_PKL + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(res, fh)
    tmp.replace(out / RESULT_PKL)  # pkl — останнім: маркер завершеного результату
    return out


def load_result(video: str | Path, base: str | Path = BATCH_DIR) -> dict:
    with open(result_dir(video, base) / RESULT_PKL, "rb") as fh:
        return pickle.load(fh)


def _main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="OSD-пайплайн: відео → bbox м1, bbox м2, словник м2")
    ap.add_argument("video")
    ap.add_argument("--videos-dir", default=str(VIDEOS_DIR))
    a = ap.parse_args(argv)
    res = infer_video(a.video, videos_dir=a.videos_dir, log=print)
    print(json.dumps({k: v for k, v in _json_view(res).items() if k != "frames"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
