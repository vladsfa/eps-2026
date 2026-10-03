"""Візуалізація повних результатів пайплайну (вихід `pipeline.infer_video` / `assemble_result`).

1. Середні ціна і час за виклики моделі 1 і моделі 2 (+ разом).
2. По рядку на кадр: оригінал | bbox моделі 1 | bbox моделі 2 (лише bbox зі значеннями).
3. Одна таблиця внизу — словник моделі 2 без ключів, де всі значення null.
"""

from __future__ import annotations

import pandas as pd

from .bbox_draw import colors_for, draw_bboxes
from .constants import VIZ_OPACITY
from .image_utils import show_images
from .viz_model1 import _display
from .viz_model2 import readings_table

__all__ = ["cost_time_table", "show_results"]


def cost_time_table(res: dict) -> pd.DataFrame:
    rows = []
    for name, key in (("модель 1 (bbox)", "m1"), ("модель 2 (значення)", "m2")):
        m = res[key]
        s = m["summary"]
        rows.append({"виклик": name, "модель": f"{m['model']} ({m['reasoning_effort']})",
                     "ранів": s["n_runs"], "успішних": s["n_ok"],
                     "середній час, с": s["mean_time_s"], "середня ціна, $": s["mean_cost_usd"]})
    rows.append({"виклик": "разом (м1 + м2)", "модель": "",
                 "ранів": rows[0]["ранів"] + rows[1]["ранів"],
                 "успішних": rows[0]["успішних"] + rows[1]["успішних"],
                 "середній час, с": round(rows[0]["середній час, с"] + rows[1]["середній час, с"], 2),
                 "середня ціна, $": round(rows[0]["середня ціна, $"] + rows[1]["середня ціна, $"], 6)})
    return pd.DataFrame(rows).set_index("виклик")


def show_results(res: dict, opacity: float = VIZ_OPACITY, width: float = 20.0) -> pd.DataFrame:
    """Показати повні результати; повертає очищену таблицю словника моделі 2."""
    m1b, m2b = res["m1_bboxes"], res["m2_bboxes"]
    print(f"🎬 {res['video']} · bbox: модель 1 = {len(m1b)}, модель 2 = {len(m2b)} · "
          f"ран м1 #{res['selected_run_m1']}, ран м2 #{res['selected_run_m2']}"
          + (f" · повний час пайплайну {res['elapsed_s']:.1f} с" if res.get("elapsed_s") else ""))
    _display(cost_time_table(res))

    m2_colors = colors_for(m2b, m1b)
    images, titles = [], []
    for f in res["frames"]:
        tag = f"кадр {f['index']} · t={f['timestamp']}"
        images += [f["image"], draw_bboxes(f["image"], m1b, opacity=opacity),
                   draw_bboxes(f["image"], m2b, opacity=opacity, colors=m2_colors)]
        titles += [f"{tag} · оригінал", f"{tag} · модель 1: {len(m1b)} bbox",
                   f"{tag} · модель 2: {len(m2b)} bbox"]
    show_images(images, titles, ncols=3, width=width, suptitle=res["video"])

    table = readings_table(res["readings"], drop_all_null=True)
    print(f"📋 словник моделі 2 без all-null ключів: {len(table)} з {len(res['readings'])} ключів")
    _display(table if len(table) else "(усі ключі null)")
    return table
