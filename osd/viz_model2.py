"""Візуалізація ранів моделі 2: час і ціна + середнє, кадри з відфільтрованими bbox
(лише ті, з яких зчитано значення) і таблиця словника відповіді без ключів, де всі значення null."""

from __future__ import annotations

import pandas as pd

from .bbox_draw import colors_for, draw_bboxes_on_frames
from .constants import VIZ_OPACITY
from .image_utils import show_images
from .viz_model1 import _display, _print_summary, runs_table

__all__ = ["readings_table", "show_model2_runs"]


def readings_table(readings: dict, drop_all_null: bool = False, null: str = "null") -> pd.DataFrame:
    """{key: {values, bbox_name}} → таблиця key | value_0 | value_1 | value_2 | bbox_name."""
    rows = []
    for key, r in readings.items():
        vals = list(r["values"])
        if drop_all_null and all(v is None for v in vals):
            continue
        row = {"key": key}
        row.update({f"value_{i}": (v if v is not None else null) for i, v in enumerate(vals)})
        row["bbox_name"] = r["bbox_name"] if r["bbox_name"] is not None else null
        rows.append(row)
    cols = ["key", *[f"value_{i}" for i in range(3)], "bbox_name"]
    return pd.DataFrame(rows, columns=cols if not rows else None).set_index("key")


def show_model2_runs(frames: list[dict], m2: dict, m1_bboxes: list[dict],
                     opacity: float = VIZ_OPACITY, video_name: str = "") -> None:
    """Час/ціна по ранах + середнє; для кожного рану — кадри з bbox, що лишились, і словник без all-null ключів."""
    _print_summary(m2, "Модель 2 (значення)")
    _display(runs_table(m2, "ключів з даними",
                        lambda r: sum(any(v is not None for v in x["values"]) for x in r["readings"])))
    names = [b["label"] for b in m1_bboxes]
    for pos, run in enumerate(m2["runs"]):
        print(f"\n===== ран моделі 2 #{pos} · "
              + ("OK" if run["ok"] else f"ПОМИЛКА: {run['error']}")
              + f" · {run['time_s']:.1f} с · ${run['cost_usd']:.5f} =====")
        if run.get("skipped"):
            print(f"  ℹ️ {run['skipped']}")
        if not run["ok"]:
            continue
        print(f"  bbox: {len(names)} від моделі 1 → {len(run['filtered_bboxes'])} зі значеннями")
        fb = run["filtered_bboxes"]
        images = draw_bboxes_on_frames(frames, fb, opacity=opacity, colors=colors_for(fb, m1_bboxes))
        titles = [f"ран м2 #{pos} · кадр {f['index']} · t={f['timestamp']}" for f in frames]
        show_images(images, titles, ncols=len(images),
                    suptitle=f"{video_name} — модель 2, ран #{pos}: {len(fb)} bbox зі значеннями")
        table = readings_table(run["dict"], drop_all_null=True)
        print(f"📋 словник рану #{pos} без all-null ключів: {len(table)} з {len(run['dict'])} ключів")
        _display(table if len(table) else "(усі ключі null)")
