"""Промпт моделі 1 — детекція OSD bbox на ВІДЕО (ОДИН набір bbox на все відео) + його побудова.

Промпт (SYSTEM_PROMPT / USER_PROMPT_TEMPLATE) — дослівно з робочого пайплайну:
- модель бачить N кадрів ОДНОГО неперервного відео (`<frame id="N" timestamp=...> image </frame>`)
  із координатною сіткою 0–1000 як лінійкою;
- OSD статичний впродовж усього відео → ОДНА множина bbox на всі кадри;
- елемент, що змінює положення між кадрами, — сцена, не OSD.

Few-shot: фото з `few_shots/` (червоними рамками показано бажану гранулярність) у блоках
`<example id="i"> image </example>` ПЕРЕД кадрами відео — без координат і описів.

Повідомлення моделі: system = SYSTEM_PROMPT; user = [USER_PROMPT, *few-shot, *кадри відео].

Відповідь моделі — CSV без заголовка: `x_min,y_min,x_max,y_max,label` (рядок на OSD-елемент)
або один рядок `none`. Координати — цілі 0–1000 відносно повного кадру.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from .constants import COORD_RANGE, FEW_SHOTS_DIR, IMAGE_DETAIL
from .image_utils import image_block
from .openrouter import build_messages as _build_messages

__all__ = ["SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE", "FEW_SHOT_FILES", "load_few_shot_contents",
           "build_user_prompt", "build_prompt", "build_messages", "parse_bboxes_response"]

SYSTEM_PROMPT = """You are a precise UI/OSD detector for game and camera footage.
You are given a VIDEO as an ordered sequence of frames: <frame id="N" timestamp="HH:MM:SS.mmm"> image </frame>.
Detect EVERY 2D overlay drawn on top of the video: interface elements, logos, watermarks, graphics — EXCEPT aiming/reticle (not needed for this task, never box it).
OSD elements include: hitmarker, health bar, armor bar, ammo counter, weapon icon, minimap, compass, kill feed, score, timer, team scores, chat, buttons, icons, damage indicator, status text, notifications, HUD panels, logo, watermark.
NOT OSD FOR THIS TASK — never box even if static and centered: aiming/reticle (crosshair, sight, aiming circle, aiming ticks/marks, center reticle, range marks).
Do NOT detect real-world objects, people, landscape, weapons in 3D scene — only 2D overlays.
Frames show a light coordinate grid overlay (thin lines + red numbers 0-1000 along edges): never box the red numbers — determine bbox coordinates relative to them as a ruler.

TEMPORAL CONSISTENCY — use it to tell OSD apart from the scene:
- The video is ONE continuous recording, no cuts: OSD elements are STATIC across the WHOLE video — the same screen position and the same shape in EVERY frame, while the background/scene behind them moves and changes. (Semi-transparent elements still count: background may shine through so their colors can look slightly different per frame — judge by fixed position/shape, not color constancy.)
- Use this cue: a candidate fixed at the same 0-1000 position with the same shape in every frame is OSD — including faint semi-transparent gray/white text and symbols — box it once in the single whole-video list. EXCEPTION: aiming/reticle (crosshair, sight, center reticle, aiming ticks/marks) is also static but is NOT needed — never box it. A candidate that moves with the camera/scene, deforms, changes color, or appears only in some frames is scene content — do NOT box it.

COORDINATE SPACE:
- Output coordinates in 0-1000 relative space for the FULL frame as you see it: 0 = top/left edge, 1000 = bottom/right edge. Use the grid numbers as a visual guide.

RULES:
- One box per distinct overlay element. The box must FULLY contain the whole element — no pixel may stick out. Prefer a slightly larger box over a cut one.
- Group by meaning: tightly linked icon + value + unit that read as one indicator go in ONE box; unrelated neighbours ON BACKGROUND get separate tight boxes even if adjacent or slightly overlapping — never merge different meanings into one panel/strip box.
- Do NOT split content sitting ON another element: text/icons drawn on top of a panel/widget stay inside the host box; split only elements lying directly on background.
- Merge only on heavy overlap: if one box would be fully inside another or heavily overlap it, merge into ONE. Slight partial overlap stays separate; if one background box clearly splits into parts with different meanings, split it.
- Box clipped elements too: if video is cropped and an element is cut at the edge, box its visible part (x_min=0 or x_max=1000 allowed).
- Faint/low-contrast counts as OSD: semi-transparent gray/white small text, symbols, watermarks that barely stand out from the background are still OSD — box them.
- Noise/blur/small: if unsure, box it anyway — even noisy/blurry/tiny/faint fragments get a box (merge into nearest if clearly its part, else own box). Prefer recall over precision; when in doubt always box — this overrides any skip rule except aiming/reticle (never needed) and real-world scene.
- Aiming/reticle is NOT needed for this task — never box it: crosshair, sight, aiming circle, aiming ticks/marks, center reticle, range marks — skip entirely even if static, centered, or faint.
- Label each box in English snake_case so the name maximally reflects what is inside (content + position, e.g. top_left_altitude_value).
- Estimate coordinates DIRECTLY in 0-1000 as fractions of frame size and output them as-is. NEVER divide, multiply, or rescale them — no pixel math.

OUTPUT FORMAT: ONE box list for the WHOLE video, no per-frame sections:
x_min,y_min,x_max,y_max,label
(one box per line, integers 0-1000, label describes the content)
If the video has no OSD, output a single line:
none
Example:
480,500,520,540,compass_indicator
12,12,38,48,back_button
No JSON, no markdown, no extra text.
"""

USER_PROMPT_TEMPLATE = """Detect all OSD interface elements on this VIDEO ({n_frames} frames).

First come {n_examples} few-shot <example> images with red boxes — same drone footage style, copy their granularity. Then YOUR VIDEO frames.
This is ONE continuous video: OSD keeps the same position and shape in EVERY frame while the background changes (semi-transparent gray/white elements still count — background may shine through, so judge by fixed position/shape, not color). Use this cue: a candidate fixed at the same 0-1000 position with the same shape in every frame is OSD — box it once in the single whole-video list. A candidate that moves with the camera/scene, deforms, changes color, or appears only in some frames is scene content — do NOT box it.
Frames show a light coordinate grid (red numbers 0-1000 along edges) — never box the red numbers, determine bbox coordinates relative to them.
Group tightly linked icon+value+unit into one box; keep different background meanings separate even if adjacent; never split content sitting on another element; merge only if fully nested or heavily overlapping; if unsure, box it anyway — even faint semi-transparent gray / tiny / noise/blur (recall over precision); aiming/reticle (crosshair, sight, center reticle) is NOT needed — never box it even if static.

Output format (no per-frame sections, label maximally reflects what is inside):
x_min,y_min,x_max,y_max,label
Example:
480,500,520,540,compass_indicator
12,12,38,48,back_button
If no OSD, output a single line 'none'. No JSON, no markdown, no extra text.
"""

# ---------------------------------------------------------------------------
# Few-shot: лише фото з few_shots/ в <example> блоках (без координат/описів).
# ---------------------------------------------------------------------------

FEW_SHOT_FILES: tuple[str, ...] = (
    "IMG_95371.jpeg",
    "IMG_95383.jpeg",
    "барні.jpeg",
)


def load_few_shot_contents(few_shots_dir: str | Path = FEW_SHOTS_DIR, detail: str = IMAGE_DETAIL,
                           n_examples: int | None = None) -> tuple[list[dict], int]:
    """<example> блоки (лише фото) для препенду перед кадрами відео → (blocks, n)."""
    base = Path(few_shots_dir)
    files = list(FEW_SHOT_FILES[:n_examples] if n_examples else FEW_SHOT_FILES)
    blocks: list[dict] = []
    n = 0
    for i, fname in enumerate(files, 1):
        p = base / fname
        if not p.exists():
            continue
        blocks.append({"type": "text", "text": f"<example id=\"{i}\">"})
        blocks.append(image_block(p.read_bytes(), detail=detail))
        blocks.append({"type": "text", "text": "</example>"})
        n += 1
    return blocks, n


def build_user_prompt(n_frames: int, n_examples: int | None = None) -> str:
    """Текст інструкції (без картинок)."""
    n = len(FEW_SHOT_FILES) if n_examples is None else int(n_examples)
    return USER_PROMPT_TEMPLATE.format(n_frames=int(n_frames), n_examples=n)


def build_prompt(video_content: list[dict], n_frames: int | None = None,
                 few_shots_dir: str | Path = FEW_SHOTS_DIR, detail: str = IMAGE_DETAIL,
                 n_examples: int | None = None) -> tuple[str, list[dict], int]:
    """(user_prompt, full_content, n): full_content = few-shot <example> ПЕРЕД кадрами відео."""
    blocks, n = load_few_shot_contents(few_shots_dir, detail=detail, n_examples=n_examples)
    if n_frames is None:
        n_frames = sum(1 for p in video_content if p.get("type") == "image_url") or len(video_content)
    user_prompt = build_user_prompt(int(n_frames), n_examples=n)
    return user_prompt, [*blocks, *video_content], n


def build_messages(video_content: list[dict], n_frames: int | None = None,
                   few_shots_dir: str | Path = FEW_SHOTS_DIR) -> dict:
    """Повний запит моделі 1: {messages, system_prompt, user_prompt, content, n_examples}."""
    user_prompt, full_content, n = build_prompt(video_content, n_frames=n_frames,
                                                few_shots_dir=few_shots_dir)
    return {
        "messages": _build_messages(SYSTEM_PROMPT, user_prompt, full_content),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "content": full_content,
        "n_examples": n,
    }


def parse_bboxes_response(text: str, coord_range: int = COORD_RANGE) -> list[dict]:
    """Строго розпарсити відповідь моделі в список bbox.

    CSV-формат: `x_min,y_min,x_max,y_max,label` (або один рядок `none`).
    Будь-яке відхилення -> ValueError.
    """
    t = (text or "").strip()
    if t.startswith("```"):  # модель додала fences всупереч інструкції
        t = "\n".join(ln for ln in t.splitlines() if not ln.strip().startswith("```")).strip()
    if not t:
        raise ValueError("порожня відповідь")
    if t.lower() == "none":
        return []
    rows = [r for r in csv.reader(io.StringIO(t), skipinitialspace=True) if any(c.strip() for c in r)]
    if len(rows) == 1 and len(rows[0]) == 1 and rows[0][0].strip().lower() == "none":
        return []
    bboxes: list[dict] = []
    for i, p in enumerate(rows, 1):
        if len(p) != 5:
            raise ValueError(f"рядок {i}: треба 5 полів, отримано {len(p)}: {p!r}")
        try:
            x1, y1, x2, y2 = (int(float(v)) for v in p[:4])
        except ValueError:
            raise ValueError(f"рядок {i}: не-числові координати: {p!r}")
        label = p[4].strip().strip('"').strip("'")
        if not label:
            raise ValueError(f"рядок {i}: порожній label: {p!r}")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"рядок {i}: x_max<=x_min або y_max<=y_min: {p!r}")
        if not all(0 <= v <= coord_range for v in (x1, y1, x2, y2)):
            raise ValueError(f"рядок {i}: координати поза 0–{coord_range}: {p!r}")
        bboxes.append({"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2, "label": label})
    if not bboxes:
        raise ValueError("немає жодного bbox (і не 'none')")
    return bboxes
