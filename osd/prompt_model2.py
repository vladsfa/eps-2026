"""Промпт моделі 2 — зчитування OSD-телеметрії з кадрів ІЗ bbox моделі 1 + його побудова.

Промпт (REQUIRED_KEYS / SYSTEM_PROMPT / USER_PROMPT_TEMPLATE) — дослівно з робочого пайплайну.

Вхід (ОДИН запит на всі кадри):
- РІВНО ТРИ кадри, на які ВЖЕ нанесені bbox вибраного рану моделі 1 (кольорові рамки +
  назви, напівпрозорі `BBOX_OPACITY`, щоб крізь них було видно OSD);
- текстовий каталог назв bbox моделі 1 (координати НЕ передаються).
Повідомлення: system = SYSTEM_PROMPT; user = [USER_PROMPT (з каталогом),
"Read the OSD values ...", <frame id="0"> image </frame>, ...].

Відповідь — ОДИН CSV-блок `key,value_0,value_1,value_2,bbox_name` (усі REQUIRED_KEYS рівно по
одному разу; value_i — значення з кадру i дослівно з одиницями; відсутнє/невпевнене → null;
bbox_name — одна назва з каталогу на всі кадри, null якщо всі значення null).
"""

from __future__ import annotations

import csv
import io

from PIL import Image

from .constants import IMAGE_DETAIL
from .image_utils import image_block, pil_to_jpeg_bytes
from .openrouter import build_messages as _build_messages

__all__ = ["REQUIRED_KEYS", "KEY_NAMES", "SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE", "CSV_HEADER",
           "build_user_prompt", "frames_to_content", "build_messages",
           "parse_readings_response", "readings_to_csv"]

# (key, опис). Порядок фіксований — він же задає порядок рядків у відповіді.
REQUIRED_KEYS = [
    ("latitude", "current aircraft position latitude"),
    ("longitude", "current aircraft position longitude"),
    ("gps_status", "positioning state: NO GPS, 2D FIX, 3D FIX, etc."),
    ("gps_satellites_count", "number of satellites"),
    ("altitude", "altitude with unit of measurement"),
    ("altitude_reference", "altitude reference: relative, msl, agl; unknown -> null"),
    ("heading", "aircraft heading in degrees (digits, compass tape, or compass/attitude dial angle, e.g. '180°')"),
    ("ground_speed", "speed relative to ground"),
    ("airspeed", "airspeed"),
    ("speed", "speed whose type cannot be determined"),
    ("vertical_speed", "climb or descent rate"),
    ("distance_to_home", "distance to the home point"),
    ("distance_travelled", "accumulated path length"),
    ("osd_date", "date shown by the OSD itself"),
    ("osd_clock_time", "time of day, not flight duration"),
    ("flight_time", "timer explicitly designated as flight time"),
    ("osd_timer", "shown timer with unknown purpose"),
    ("flight_mode", "flight mode: e.g. AIR, FBWA, N"),
    ("armed", "explicitly shown ARMED / DISARMED state"),
    ("battery_voltage", "total battery voltage"),
    ("cell_voltage", "battery cell voltage"),
    ("battery_remaining_pct", "remaining charge in percent"),
    ("battery_current", "current draw"),
    ("throttle_pct", "throttle level in percent"),
    ("warnings", "problem messages: LAND NOW, low voltage, etc."),
]

KEY_SET = frozenset(k for k, _ in REQUIRED_KEYS)

_KEY_DOC = "\n".join(f"- {k}: {d}" for k, d in REQUIRED_KEYS)

SYSTEM_PROMPT = f"""You read drone-video OSD (on-screen display) telemetry values.
You receive EXACTLY THREE frames (frame 0, 1, 2 in order). Each already has COLORED bounding boxes with NAMES drawn on it (e.g. top_left_altitude_value).
The user message also lists the EXACT box-name catalog from the previous detection stage — the set of keys and box NAMES is the SAME on all three frames, only values may differ across frames.
No coordinates are given to you — but bbox_name in your answer MUST match one name from the catalog EXACTLY (character-for-character, as written on the images). Never invent, rename, or rephrase a box name.
YOUR ONLY JOB: look at EVERY drawn box on ALL THREE frames and read the fixed keys below.
Fixed keys (output EVERY key exactly once, in this order):
{_KEY_DOC}
Rules:
1. Review every drawn box across the three frames, one by one. A key's value may sit inside any box; the holding box is the same on all frames. If it is in none of the boxes on any frame, the key is null on all frames.
2. value_0/value_1/value_2: VERBATIM as seen on frame 0/1/2, WITH unit and original spacing (e.g. "100 km/h", "22.3 V"). Copy digits exactly, never round, never convert units. For `heading`: if shown on a compass dial, rose, or tape without explicit digits, extract or estimate the angle in degrees (e.g. "180°", "45°") from the dial or pointer, per frame.
3. NEVER guess unknown values. If a value on a frame is missing, illegible, cut off, or you are not sure it is exactly the requested key (e.g. speed of unknown type -> only `speed`, never `ground_speed`; timer of unknown purpose -> only `osd_timer`, never `flight_time`; altitude with unknown reference -> `altitude_reference` is null), output null for that frame's cell (extracting heading angle from a visible compass is expected).
4. bbox_name: ONE shared NAME of the drawn box that holds the key's value, copied EXACTLY from the catalog (identical for all three frames). If the key is null on ALL three frames, bbox_name is null too.
5. CONSISTENCY ACROSS FRAMES: all three values of one key MUST come from the SAME box — read value_0/value_1/value_2 from the same drawn box on frame 0/1/2. Values must be mutually consistent (same indicator, same unit/format; only the digits may change). Never take value_0 from one box and value_1 from another.
6. `warnings`: copy each visible problem message verbatim per frame; several messages in one cell joined with "; ". No problems visible on a frame -> null in that frame's cell.
7. Output ONLY the CSV block described in the user message. No explanations, no markdown fences if possible, no extra text.
"""

USER_PROMPT_TEMPLATE = """Look at these EXACTLY THREE frames (frame 0, 1, 2 in order). Each already shows drawn bounding boxes with names.
Box-name catalog from the detection stage (bbox_name MUST be exactly one of these, or null):
{bbox_catalog}
Read EVERY fixed key across all three frames ({n_keys} keys = {n_keys} data rows).
Missing / illegible / unsure on a frame -> null in that frame's cell. Values verbatim with units, e.g. "100 km/h", "22.3 V" (for compass, extract heading angle in degrees e.g. "180°").
One shared bbox_name per key for all frames (exactly as in the catalog above); null on all three values -> bbox_name null.
All three values of one key MUST come from the SAME box — read value_0/value_1/value_2 from the same drawn box on frame 0/1/2 (same indicator, same unit/format; only digits may change).

Respond with exactly one CSV block with header (one row per key):

key,value_0,value_1,value_2,bbox_name
<one row per key; value with a comma or space wrapped in double quotes>

Example (2 keys, boxes named as drawn):
key,value_0,value_1,value_2,bbox_name
ground_speed,"100 km/h","101 km/h",null,top_bar_speed_alt
latitude,null,null,null,null
No JSON, no markdown, no extra text.
"""

CSV_HEADER = "key,value_0,value_1,value_2,bbox_name"


def build_user_prompt(bbox_names: list[str] | None = None) -> str:
    """Build the user prompt for THREE frames (каталог назв bbox з моделі 1).

    Args:
        bbox_names: назви боксів вибраного рану моделі 1
            (`det["all_plain_bboxes"][SELECTED_RUN]` → `[b["label"] ...]`).
            Підставляються в каталог дослівно; координати НЕ передаються.
    """
    n = len(REQUIRED_KEYS)
    catalog = "\n".join(f"- {b}" for b in (bbox_names or [])) or "(none)"
    return USER_PROMPT_TEMPLATE.format(n_keys=n, bbox_catalog=catalog)


def _norm_cell(v: str | None) -> str | None:
    """'null'/порожнє → None, інакше дослівний рядок (без зовнішніх лапок)."""
    if v is None:
        return None
    s = v.strip().strip('"').strip("'").strip()
    if s == "" or s.lower() == "null" or s.lower() == "none":
        return None
    return v.strip()  # значення — дослівно, лише trim по краях


def parse_readings_response(text: str,
                            required_keys: list[str] | None = None,
                            bbox_names: list[str] | None = None) -> list[dict]:
    """Строго розпарсити об'єднаний CSV ТРЬОХ кадрів у список readings.

    Очікує блок `key,value_0,value_1,value_2,bbox_name` (заголовок обов'язковий,
    fences толеруються). Повертає [{key, values:[v0|None, v1|None, v2|None],
    bbox_name|None}] (bbox_name спільний для всіх кадрів; всі null → bbox null).
    Якщо задано `bbox_names` (каталог з моделі 1) — `bbox_name` мусить дослівно
    збігатися з одним ім'ям із каталогу, інакше ValueError.
    Будь-яке відхилення (немає заголовка, брак/дублі рядків, невідомий ключ,
    не 5 полів, bbox поза каталогом) → ValueError.
    """
    keys = list(required_keys) if required_keys else [k for k, _ in REQUIRED_KEYS]
    key_set = set(keys)
    catalog = set(bbox_names) if bbox_names else None
    t = (text or "").strip()
    if t.startswith("```"):  # модель додала fences всупереч інструкції
        t = "\n".join(ln for ln in t.splitlines() if not ln.strip().startswith("```")).strip()
    if not t:
        raise ValueError("порожня відповідь")
    lines = [ln for ln in t.splitlines() if ln.strip()]
    # заголовок — перший непорожній рядок, що містить 'key'
    hdr_idx = next((i for i, ln in enumerate(lines) if "key" in ln.lower()), None)
    if hdr_idx is None:
        raise ValueError("немає CSV-заголовка 'key,value_0,value_1,value_2,bbox_name'")
    hdr = [c.strip().lower() for c in lines[hdr_idx].split(",")]
    if hdr != ["key", "value_0", "value_1", "value_2", "bbox_name"]:
        raise ValueError(f"невірний заголовок: {lines[hdr_idx]!r}")
    body = "\n".join(lines[hdr_idx + 1:])
    rows = [r for r in csv.reader(io.StringIO(body), skipinitialspace=True)
            if any((c or "").strip() for c in r)]
    readings: list[dict] = []
    seen: set[str] = set()
    for i, p in enumerate(rows, 1):
        if len(p) != 5:
            raise ValueError(f"рядок {i}: треба 5 полів, отримано {len(p)}: {p!r}")
        key = p[0].strip()
        if key not in key_set:
            raise ValueError(f"рядок {i}: невідомий ключ {key!r}")
        if key in seen:
            raise ValueError(f"рядок {i}: дубль ключа {key!r}")
        seen.add(key)
        values = [_norm_cell(p[1]), _norm_cell(p[2]), _norm_cell(p[3])]
        bbox_name = _norm_cell(p[4])
        if all(v is None for v in values):
            bbox_name = None  # ключ ніде не знайдено → обидва null
        elif bbox_name is None:
            raise ValueError(f"рядок {i}: є значення, але bbox_name null")
        elif catalog is not None and bbox_name not in catalog:
            raise ValueError(f"рядок {i}: bbox_name {bbox_name!r} немає в каталозі моделі 1")
        readings.append({"key": key,
                         "values": values, "bbox_name": bbox_name})
    # повнота: кожен ключ рівно раз
    missing = [k for k in keys if k not in seen]
    if missing:
        raise ValueError(f"бракує {len(missing)} рядків, напр.: {missing[:5]}")
    readings.sort(key=lambda r: keys.index(r["key"]))
    return readings


def readings_to_csv(readings: list[dict]) -> str:
    """Readings ТРЬОХ кадрів → CSV-текст `key,value_0,value_1,value_2,bbox_name` (None → null)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["key", "value_0", "value_1", "value_2", "bbox_name"])
    for r in readings:
        vals = r.get("values", [None, None, None])
        w.writerow([r["key"],
                    vals[0] if vals[0] is not None else "null",
                    vals[1] if vals[1] is not None else "null",
                    vals[2] if vals[2] is not None else "null",
                    r["bbox_name"] if r["bbox_name"] is not None else "null"])
    return buf.getvalue()


KEY_NAMES = [k for k, _ in REQUIRED_KEYS]


def frames_to_content(annotated_frames: list[Image.Image], detail: str = IMAGE_DETAIL) -> list[dict]:
    """Кадри з bbox → content (вступний текст + `<frame id="i">` блоки, i з 0)."""
    n = len(annotated_frames)
    content: list[dict] = [
        {"type": "text",
         "text": f"Read the OSD values from the drawn boxes on these {n} frames (in order)."},
    ]
    for i, img in enumerate(annotated_frames):
        content.append({"type": "text", "text": f"<frame id=\"{i}\">"})
        content.append(image_block(pil_to_jpeg_bytes(img), detail=detail))
        content.append({"type": "text", "text": "</frame>"})
    return content


def build_messages(annotated_frames: list[Image.Image], bbox_names: list[str]) -> dict:
    """Повний запит моделі 2: {messages, system_prompt, user_prompt, content, bbox_names}."""
    user_prompt = build_user_prompt(bbox_names=bbox_names)
    content = frames_to_content(annotated_frames)
    return {
        "messages": _build_messages(SYSTEM_PROMPT, user_prompt, content),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "content": content,
        "bbox_names": list(bbox_names),
    }
