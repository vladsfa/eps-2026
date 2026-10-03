"""Кадри із сіткою → content (вхід моделі 1) у форматі OpenRouter Chat Completions.

Кожен кадр — три блоки (без текстової інструкції — її додає промпт):
    {"type": "text", "text": '<frame id="1" timestamp="00:00:14.048">'},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "high"}},
    {"type": "text", "text": "</frame>"},
"""

from __future__ import annotations

from typing import Iterable

from .constants import IMAGE_DETAIL
from .image_utils import image_block

__all__ = ["frames_to_content", "estimate_content_mb"]


def frames_to_content(frames: Iterable[dict], jpeg_key: str = "grid_jpeg_bytes",
                      detail: str = IMAGE_DETAIL) -> list[dict]:
    """Записи кадрів → content; JPEG береться з поля `jpeg_key` (за замовчуванням — кадр із сіткою)."""
    content: list[dict] = []
    for f in frames:
        jpeg = f.get(jpeg_key)
        if not jpeg:
            raise KeyError(f"кадр #{f.get('index')} не має поля {jpeg_key!r} "
                           f"(спершу виклич grid.overlay_grid_on_frames)")
        content.append({"type": "text", "text": f'<frame id="{f["index"]}" timestamp="{f["timestamp"]}">'})
        content.append(image_block(jpeg, detail=detail))
        content.append({"type": "text", "text": "</frame>"})
    return content


def estimate_content_mb(content: list[dict]) -> float:
    """Приблизний розмір content у МБ (текст + base64-картинки)."""
    total = sum(len(str(p.get("text", ""))) for p in content if p.get("type") == "text")
    total += sum(len(p.get("image_url", {}).get("url", "")) for p in content if p.get("type") == "image_url")
    return round(total / 1024 / 1024, 2)
