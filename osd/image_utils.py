"""Спільні хелпери для зображень: шрифт, PIL ↔ JPEG / base64 data-URL."""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageFont

from .constants import IMAGE_DETAIL, JPEG_QUALITY

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Helvetica.ttc",                 # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",     # linux
    "C:/Windows/Fonts/arial.ttf",                          # windows
)


@lru_cache(maxsize=64)
def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Системний шрифт заданого розміру, fallback — вбудований PIL."""
    for p in _FONT_CANDIDATES:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # старий Pillow без size
        return ImageFont.load_default()


def pil_to_jpeg_bytes(img: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def jpeg_bytes_to_pil(jpeg: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg)).convert("RGB")


def jpeg_to_data_url(jpeg: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode('utf-8')}"


def image_block(jpeg: bytes, detail: str = IMAGE_DETAIL) -> dict:
    """Блок `image_url` OpenAI/OpenRouter Chat Completions з JPEG-байтів."""
    return {"type": "image_url", "image_url": {"url": jpeg_to_data_url(jpeg), "detail": detail}}


def show_images(images: list[Image.Image], titles: list[str] | None = None, ncols: int = 3,
                width: float = 20.0, suptitle: str | None = None, title_size: int = 10) -> None:
    """Сітка зображень у matplotlib (висота рядка — за пропорціями найвищого кадру)."""
    import sys

    import matplotlib.pyplot as plt

    if not images:
        return
    ncols = max(1, min(ncols, len(images)))
    nrows = (len(images) + ncols - 1) // ncols
    if nrows == 1:  # один ряд: ширина колонок ∝ пропорціям → однакова висота без порожнечі
        ratios = [im.size[0] / im.size[1] for im in images]
        fig, axes = plt.subplots(1, ncols, figsize=(width, width / sum(ratios) + 0.6), squeeze=False,
                                 gridspec_kw={"width_ratios": ratios})
    else:
        aspect = max(im.size[1] / im.size[0] for im in images)
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, width / ncols * aspect * nrows + 0.6),
                                 squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k < len(images):
            ax.imshow(images[k])
            if titles:
                ax.set_title(titles[k], fontsize=title_size)
    if suptitle:
        fig.suptitle(suptitle, fontsize=title_size + 3)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_size_inches()[1]))
    else:
        fig.tight_layout()
    sys.stdout.flush()  # текст до картинки — у правильному порядку в Jupyter
    plt.show()


def data_url_to_pil(url: str) -> Image.Image:
    """`data:image/...;base64,...` → PIL RGB (для візуалізації промптів)."""
    b64 = url.split(",", 1)[1] if url.startswith("data:") else url
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
