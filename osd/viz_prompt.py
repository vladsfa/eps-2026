"""Візуалізація повного промпту (system + user) з картинками замість base64 — прямо в тексті.

Блоки content виводяться по черзі, як їх бачить модель: текст друкується дослівно, а кожен
`image_url` (base64) декодується і показується (matplotlib) рівно в тому місці, де він стоїть —
між своїм відкривальним (`<example id="1">`, `<frame id="0">`) і закривальним тегом.
"""

from __future__ import annotations

import sys

from PIL import Image

from .image_utils import data_url_to_pil

__all__ = ["show_prompt"]

_RULE = "=" * 110


def _show_inline(img: Image.Image, max_width: float = 9.0, max_height: float = 5.5) -> None:
    """Одне зображення в потоці тексту (вписане в max_width × max_height дюймів)."""
    import matplotlib.pyplot as plt

    w, h = img.size
    fw, fh = max_width, max_width * h / w
    if fh > max_height:
        fh, fw = max_height, max_height * w / h
    sys.stdout.flush()  # текст до картинки — у правильному порядку в Jupyter
    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.imshow(img)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    plt.show()


def show_prompt(request: dict | list, title: str = "", max_width: float = 9.0,
                max_height: float = 5.5) -> None:
    """Показати запит: `request` — вихід `prompt_model*.build_messages` або список messages."""
    messages = request["messages"] if isinstance(request, dict) else request
    n_img = sum(1 for m in messages if isinstance(m["content"], list)
                for b in m["content"] if b.get("type") == "image_url")
    n_chars = sum(len(m["content"]) if isinstance(m["content"], str) else
                  sum(len(b.get("text", "")) for b in m["content"]) for m in messages)
    print(f"{_RULE}\n🧾 {title or 'ПРОМПТ'}: {len(messages)} повідомлення, "
          f"{n_chars} символів тексту, {n_img} зображень\n{_RULE}")

    for m in messages:
        print(f"\n┏━━━━━━━━━━ [{m['role'].upper()}] ━━━━━━━━━━")
        content = m["content"]
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        for block in blocks:
            if block.get("type") == "image_url":
                img = data_url_to_pil(block["image_url"]["url"])
                print(f"[зображення {img.size[0]}×{img.size[1]}]")
                _show_inline(img, max_width=max_width, max_height=max_height)
            else:
                print(block.get("text", ""))
        print("┗" + "━" * 40)
    sys.stdout.flush()
