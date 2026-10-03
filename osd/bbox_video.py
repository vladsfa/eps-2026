"""Нанесення bbox на ВСЕ відео + GIF-прев'ю для ноутбуків.

OSD статичний, тому bbox однакові для всіх кадрів: рамки й назви малюються ОДИН раз на
прозорому PNG-шарі розміру відео (шкала 0..1000 → пікселі повного кадру) і накладаються
ffmpeg-фільтром `overlay` на кожен кадр → mp4 (H.264, yuv420p, без звуку).

GitHub не відтворює відео всередині .ipynb, тому в ноутбуках показується анімований GIF
(зменшений; довгі відео прискорюються до `GIF_MAX_FRAMES` кадрів).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .bbox_draw import bbox_overlay, colors_for
from .constants import (GIF_COLORS, GIF_FPS, GIF_MAX_FRAMES, GIF_MAX_SIDE, OUTPUTS_DIR, VIDEO_CRF,
                        VIDEOS_DIR, VIZ_OPACITY)
from .extract_frames import get_video_meta, resolve_video_path
from .image_utils import jpeg_bytes_to_pil

__all__ = ["video_frame_size", "render_bbox_video", "make_gif", "show_gif", "show_video_gif"]


def _run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg помилка: {(res.stderr or b'')[-800:].decode('utf-8', 'ignore')}")


def video_frame_size(video: str | Path, videos_dir: str | Path = VIDEOS_DIR) -> tuple[int, int]:
    """Розмір кадру після декодування (з урахуванням повороту, як його бачить ffmpeg)."""
    path = resolve_video_path(video, videos_dir)
    res = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"], capture_output=True)
    if res.returncode != 0 or not res.stdout:
        raise RuntimeError(f"не вдалося декодувати перший кадр {path}")
    return jpeg_bytes_to_pil(res.stdout).size


def render_bbox_video(video: str | Path, bboxes: list[dict], out_path: str | Path | None = None,
                      opacity: float = VIZ_OPACITY, reference_bboxes: list[dict] | None = None,
                      videos_dir: str | Path = VIDEOS_DIR, suffix: str = "bbox") -> Path:
    """Відео з нанесеними bbox → mp4 (H.264). Повертає шлях.

    `reference_bboxes` — повний список bbox моделі 1, щоб відфільтровані bbox моделі 2
    мали ті ж кольори, що й у моделі 1.
    """
    path = resolve_video_path(video, videos_dir)
    out = Path(out_path) if out_path else OUTPUTS_DIR / "videos" / f"{path.stem}_{suffix}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    size = video_frame_size(path)
    colors = colors_for(bboxes, reference_bboxes) if reference_bboxes else None
    layer = bbox_overlay(size, bboxes, opacity=opacity, colors=colors)
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "overlay.png"
        layer.save(png)
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-i", str(png),
              "-filter_complex",
              "[0:v][1:v]overlay=0:0:format=auto,scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", str(VIDEO_CRF),
              "-movflags", "+faststart", "-an", str(out)])
    return out


def make_gif(video: str | Path, gif_path: str | Path | None = None, max_side: int = GIF_MAX_SIDE,
             fps: float = GIF_FPS, max_frames: int = GIF_MAX_FRAMES, colors: int = GIF_COLORS,
             videos_dir: str | Path = VIDEOS_DIR) -> Path:
    """mp4 → анімований GIF (зменшений; якщо кадрів > max_frames — відео прискорюється)."""
    path = resolve_video_path(video, videos_dir)
    gif = Path(gif_path) if gif_path else OUTPUTS_DIR / "gifs" / f"{path.stem}.gif"
    gif.parent.mkdir(parents=True, exist_ok=True)
    duration = get_video_meta(path)["duration_sec"] or 1.0
    speed = max(1.0, duration * fps / max_frames)
    vf = (f"setpts=PTS/{speed:.4f},fps={fps},"
          f"scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})':flags=lanczos,"
          f"split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
          f"[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vf", vf, "-loop", "0", str(gif)])
    return gif


def show_gif(gif_path: str | Path, caption: str | None = None) -> None:
    """Показати GIF у ноутбуці (вивід image/gif — його відображає й GitHub)."""
    import sys

    from IPython.display import Image, display

    gif_path = Path(gif_path)
    if caption:
        print(caption)
    print(f"🎞️ {gif_path.name}: {gif_path.stat().st_size / 1024:.0f} KB")
    sys.stdout.flush()
    display(Image(data=gif_path.read_bytes(), format="gif"))


def show_video_gif(video: str | Path, videos_dir: str | Path = VIDEOS_DIR,
                   caption: str | None = None, **gif_kwargs) -> Path:
    """GIF-прев'ю відео (вхідного або з bbox) + показ. Повертає шлях до GIF."""
    path = resolve_video_path(video, videos_dir)
    meta = get_video_meta(path)
    speed = max(1.0, meta["duration_sec"] * gif_kwargs.get("fps", GIF_FPS)
                / gif_kwargs.get("max_frames", GIF_MAX_FRAMES))
    gif = make_gif(path, **gif_kwargs)
    info = (f"{path.name}: {meta['width']}×{meta['height']}, {meta['duration_sec']:.1f} с, "
            f"{meta['fps']} fps" + (f" · GIF прискорено ×{speed:.1f}" if speed > 1 else ""))
    show_gif(gif, caption=f"{caption + ' · ' if caption else ''}{info}")
    return gif
