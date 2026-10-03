"""Координатна сітка 0..1000 БЕЗ падінгів (inner): лінії й цифри прямо на кадрі.

- Розмір кадру не змінюється: 0 = верхній/лівий край, 1000 = нижній/правий.
  x_px = x_norm / 1000 * W, y_px = y_norm / 1000 * H (та сама формула, що в `bbox_draw`).
- Лінії сітки — напівпрозорі (з тінню), цифри — лише вздовж верхнього (X) і лівого (Y)
  країв, без фону; засічка завжди в істинній позиції поділки.
- Розмір цифр і товщина ліній — динамічні від найбільшої сторони кадру.
"""

from __future__ import annotations

from typing import Iterable

from PIL import Image, ImageDraw

from .constants import (COORD_RANGE, GRID_ALPHA, GRID_SCALE, LABEL_COLOR, LABEL_SCALE,
                        NUM_STEPS)
from .image_utils import load_font, pil_to_jpeg_bytes

__all__ = ["overlay_grid", "overlay_grid_on_frames", "parse_color"]

REF_SIDE = 1000.0
BASE_FONT_SIZE = 14.0
BASE_GRID_WIDTH = 1.0

NAMED_COLORS = {
    "red": (255, 0, 0), "yellow": (255, 255, 0), "green": (0, 255, 0),
    "cyan": (0, 255, 255), "magenta": (255, 0, 255), "white": (255, 255, 255),
    "black": (0, 0, 0), "dark": (30, 30, 30), "orange": (255, 165, 0), "lime": (50, 255, 50),
}


def parse_color(value: str | tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    """Колір: назва ('red'), hex ('#ff0000') або RGB-кортеж."""
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1]), int(value[2])
    s = str(value).strip().lower()
    if s in NAMED_COLORS:
        return NAMED_COLORS[s]
    h = s.lstrip("#")
    if len(h) == 6:
        try:
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        except ValueError:
            pass
    raise ValueError(f"невідомий колір: {value!r} (приклади: 'red', '#ff0000', (255, 0, 0))")


def dynamic_font_size(max_side: int, label_scale: float = 1.0) -> int:
    """Розмір цифр пропорційно найбільшій стороні (мінімум 12 px — щоб VLM бачила лінійку)."""
    return max(12, int(round(BASE_FONT_SIZE * (float(max_side) / REF_SIDE) * float(label_scale))))


def dynamic_grid_width(max_side: int, grid_scale: float = 1.0) -> int:
    """Товщина ліній сітки пропорційно найбільшій стороні (мінімум 1 px)."""
    return max(1, int(round(BASE_GRID_WIDTH * (float(max_side) / REF_SIDE) * float(grid_scale))))


def overlay_grid(image: Image.Image, num_steps: int = NUM_STEPS, coord_range: int = COORD_RANGE,
                 grid_alpha: float = GRID_ALPHA, label_scale: float = LABEL_SCALE,
                 grid_scale: float = GRID_SCALE,
                 label_color: str | tuple[int, int, int] | list[int] = LABEL_COLOR,
                 grid_color: tuple[int, int, int] = (255, 255, 255),
                 shadow_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Намалювати сітку на копії кадру (той самий розмір)."""
    img_w, img_h = image.size
    max_side = max(img_w, img_h)
    label_rgb = parse_color(label_color)
    font_size = dynamic_font_size(max_side, label_scale)
    grid_width = dynamic_grid_width(max_side, grid_scale)

    # 1. напівпрозорі лінії на окремому RGBA-шарі (не затирають OSD)
    overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    grid_rgba = (*grid_color, int(255 * grid_alpha))
    shadow_rgba = (*shadow_color, int(255 * grid_alpha))
    for i in range(1, num_steps):
        x = int(round(i * img_w / num_steps))
        odraw.line([(x + grid_width, 0), (x + grid_width, img_h)], fill=shadow_rgba, width=grid_width)
        odraw.line([(x, 0), (x, img_h)], fill=grid_rgba, width=grid_width)
        y = int(round(i * img_h / num_steps))
        odraw.line([(0, y + grid_width), (img_w, y + grid_width)], fill=shadow_rgba, width=grid_width)
        odraw.line([(0, y), (img_w, y)], fill=grid_rgba, width=grid_width)
    canvas = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    # 2. засічки + цифри вздовж верхнього (X) і лівого (Y) країв, всередині кадру
    draw = ImageDraw.Draw(canvas)
    font = load_font(font_size)
    tick_len = font_size + 4
    tick_w = max(2, grid_width + 1)
    for i in range(num_steps + 1):
        norm_val = int(i * (coord_range / num_steps))
        label = str(norm_val)
        x_true = min(int(round(norm_val / coord_range * img_w)), img_w - 1)
        y_true = min(int(round(norm_val / coord_range * img_h)), img_h - 1)
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]

        draw.line([(x_true, 0), (x_true, tick_len)], fill=label_rgb, width=tick_w)
        cx = min(max(x_true - tw // 2, 2), img_w - tw - 2)
        draw.text((cx, tick_len + 2), label, fill=label_rgb, font=font)

        draw.line([(0, y_true), (tick_len, y_true)], fill=label_rgb, width=tick_w)
        cy = min(max(y_true - th // 2, 2), img_h - th - 2)
        draw.text((tick_len + 2, cy), label, fill=label_rgb, font=font)
    return canvas


def overlay_grid_on_frames(frames: Iterable[dict], num_steps: int = NUM_STEPS,
                           coord_range: int = COORD_RANGE, grid_alpha: float = GRID_ALPHA,
                           label_scale: float = LABEL_SCALE, grid_scale: float = GRID_SCALE,
                           label_color: str | tuple[int, int, int] | list[int] = LABEL_COLOR,
                           ) -> list[dict]:
    """Сітка на батч кадрів → НОВІ записи + `grid_image` / `grid_jpeg_bytes` / `grid_size_kb`.

    Оригінальні `image` / `jpeg_bytes` лишаються без змін.
    """
    out = []
    for f in frames:
        canvas = overlay_grid(f["image"], num_steps=num_steps, coord_range=coord_range,
                              grid_alpha=grid_alpha, label_scale=label_scale,
                              grid_scale=grid_scale, label_color=label_color)
        jpeg = pil_to_jpeg_bytes(canvas)
        out.append({**f, "grid_image": canvas, "grid_jpeg_bytes": jpeg,
                    "grid_size_kb": round(len(jpeg) / 1024, 1)})
    return out
