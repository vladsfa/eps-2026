"""Візуалізація ранів моделі 1: час і ціна кожного рану + середнє, кадри кожного рану з bbox."""

from __future__ import annotations

import pandas as pd

from .bbox_draw import draw_bboxes_on_frames
from .constants import VIZ_OPACITY
from .image_utils import show_images

__all__ = ["runs_table", "show_model1_runs"]


def _display(obj) -> None:
    try:
        from IPython.display import display
        display(obj)
    except ImportError:
        print(obj)


def runs_table(result: dict, count_col: str = "bbox", count_fn=None) -> pd.DataFrame:
    """Таблиця ранів (час, ціна, токени, спроби) + рядок «середнє»."""
    rows = []
    for pos, r in enumerate(result["runs"]):
        rows.append({
            "ран": f"#{pos}" + (f" (слот {r['run']})" if r["run"] != pos else ""),
            "ok": "✅" if r["ok"] else "❌",
            count_col: count_fn(r) if count_fn else None,
            "час, с": r["time_s"],
            "останній запит, с": r["latency_s"],
            "ціна, $": r["cost_usd"],
            "prompt tok": r["prompt_tokens"],
            "completion tok": r["completion_tokens"],
            "reasoning tok": r["reasoning_tokens"],
            "спроб": r["n_attempts"],
            "повторів 429/5xx": r["transient_retries"],
            "помилка": (r["error"] or "")[:120],
        })
    df = pd.DataFrame(rows)
    if len(df):
        mean = {c: df[c].mean() for c in ("час, с", "останній запит, с", "ціна, $", "prompt tok",
                                           "completion tok", "reasoning tok", "спроб",
                                           "повторів 429/5xx")}
        df = pd.concat([df, pd.DataFrame([{"ран": "середнє", "ok": "", count_col: None, **mean,
                                           "помилка": ""}])], ignore_index=True)
    return df.set_index("ран")


def _print_summary(result: dict, name: str) -> None:
    s = result["summary"]
    mode = "перший успішний (first_only)" if result.get("first_only") else "усі рани"
    print(f"📊 {name} · {result['model']} (reasoning={result['reasoning_effort']}) · {mode}: "
          f"успішних {s['n_ok']}/{s['n_runs']} · середній час {s['mean_time_s']:.1f} с · "
          f"середня ціна ${s['mean_cost_usd']:.5f} · сума ${s['total_cost_usd']:.5f}")
    if result.get("first_only"):
        print("   (у first_only решта паралельних запитів дораховується у фоні — їхня ціна тут не врахована)")


def show_model1_runs(frames: list[dict], m1: dict, opacity: float = VIZ_OPACITY,
                     video_name: str = "") -> None:
    """Час/ціна по ранах + середнє; для кожного рану — кадри з його bbox."""
    _print_summary(m1, "Модель 1 (bbox)")
    _display(runs_table(m1, "bbox", lambda r: len(r["bboxes"])))
    for pos, run in enumerate(m1["runs"]):
        print(f"\n===== ран моделі 1 #{pos} · "
              + (f"{len(run['bboxes'])} bbox" if run["ok"] else f"ПОМИЛКА: {run['error']}")
              + f" · {run['time_s']:.1f} с · ${run['cost_usd']:.5f} =====")
        if not run["ok"]:
            continue
        images = draw_bboxes_on_frames(frames, run["bboxes"], opacity=opacity)
        titles = [f"ран #{pos} · кадр {f['index']} · t={f['timestamp']}" for f in frames]
        show_images(images, titles, ncols=len(images),
                    suptitle=f"{video_name} — модель 1, ран #{pos}: {len(run['bboxes'])} bbox")
