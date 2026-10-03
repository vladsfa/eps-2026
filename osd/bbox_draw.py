"""Малювання bbox (шкала 0..1000) на кадрах: рамки + назви, з непрозорістю.

bbox: {x_min, y_min, x_max, y_max, label}, origin — верхній лівий кут кадру.
Розмір назв і товщина рамок — динамічні від найбільшої сторони кадру,
назви без фону (текст + тонкий чорний stroke). Колір bbox — за його індексом
у списку (`PALETTE`), тож той самий bbox має той самий колір на всіх кадрах.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from .constants import BBOX_LABEL_SCALE, BBOX_LINE_SCALE, COORD_RANGE, VIZ_OPACITY
from .image_utils import load_font

__all__ = ["draw_bboxes", "draw_bboxes_on_frames", "bbox_overlay", "colors_for", "PALETTE"]

REF_SIDE = 1000.0
BASE_FONT_SIZE = 12.0
BASE_LINE_WIDTH = 2.0

PALETTE = [
    (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (255, 165, 0), (50, 255, 50), (255, 255, 255),
]


def _label_size(max_side: int, label_scale: float) -> int:
    return max(8, int(round(BASE_FONT_SIZE * (float(max_side) / REF_SIDE) * float(label_scale))))


def _line_width(max_side: int, line_scale: float) -> int:
    return max(1, int(round(BASE_LINE_WIDTH * (float(max_side) / REF_SIDE) * float(line_scale))))


def bbox_overlay(size: tuple[int, int], bboxes: list[dict], coord_range: int = COORD_RANGE,
                 label_scale: float = BBOX_LABEL_SCALE, line_scale: float = BBOX_LINE_SCALE,
                 opacity: float = VIZ_OPACITY, colors: list[tuple[int, int, int]] | None = None,
                 ) -> Image.Image:
    """Прозорий RGBA-шар розміру `size` з рамками й назвами (для накладання на кадр/відео).

    `colors` — явні кольори на кожен bbox (інакше `PALETTE` за індексом).
    """
    w, h = size
    max_side = max(w, h)
    font_size = _label_size(max_side, label_scale)
    line_width = _line_width(max_side, line_scale)
    font = load_font(font_size)
    stroke_w = max(1, font_size // 12)
    alpha = int(round(255 * min(1.0, max(0.0, float(opacity)))))

    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for i, b in enumerate(bboxes):
        x1 = max(0, int(round(float(b["x_min"]) / coord_range * w)))
        y1 = max(0, int(round(float(b["y_min"]) / coord_range * h)))
        x2 = min(w - 1, int(round(float(b["x_max"]) / coord_range * w)))
        y2 = min(h - 1, int(round(float(b["y_max"]) / coord_range * h)))
        if x2 <= x1 or y2 <= y1:
            continue
        rgb = colors[i] if colors else PALETTE[i % len(PALETTE)]
        color = (*rgb, alpha)
        label = str(b.get("label", f"box_{i}"))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        tb = draw.textbbox((0, 0), label, font=font, stroke_width=stroke_w)
        th = tb[3] - tb[1]
        tx = min(max(x1 + 2, 2), w - 2)
        ty = y1 - th - 4
        if ty < 2:  # bbox біля верхнього краю — назва всередину
            ty = y1 + 2
        draw.text((tx, ty), label, fill=color, font=font,
                  stroke_width=stroke_w, stroke_fill=(0, 0, 0, alpha))
    return layer


def draw_bboxes(image: Image.Image, bboxes: list[dict], coord_range: int = COORD_RANGE,
                label_scale: float = BBOX_LABEL_SCALE, line_scale: float = BBOX_LINE_SCALE,
                opacity: float = VIZ_OPACITY, colors: list[tuple[int, int, int]] | None = None,
                ) -> Image.Image:
    """Новий PIL RGB-кадр з bbox (оригінал не змінюється)."""
    img = image.convert("RGBA")
    layer = bbox_overlay(img.size, bboxes, coord_range=coord_range, label_scale=label_scale,
                         line_scale=line_scale, opacity=opacity, colors=colors)
    return Image.alpha_composite(img, layer).convert("RGB")


def draw_bboxes_on_frames(frames: list, bboxes: list[dict], opacity: float = VIZ_OPACITY,
                          colors: list[tuple[int, int, int]] | None = None, **kwargs) -> list:
    """Ті самі bbox на кожному кадрі (записи кадрів з полем `image` або PIL)."""
    return [draw_bboxes(f["image"] if isinstance(f, dict) else f, bboxes,
                        opacity=opacity, colors=colors, **kwargs) for f in frames]


def colors_for(bboxes: list[dict], reference: list[dict]) -> list[tuple[int, int, int]]:
    """Кольори `bboxes` такі ж, як у тих самих bbox (за `label`) у `reference`.

    Потрібно, щоб відфільтровані bbox моделі 2 мали ті ж кольори, що й у моделі 1.
    """
    idx = {b.get("label"): i for i, b in enumerate(reference)}
    return [PALETTE[idx.get(b.get("label"), i) % len(PALETTE)] for i, b in enumerate(bboxes)]
