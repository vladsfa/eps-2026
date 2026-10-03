"""Екстракція N рівномірних кадрів з відео (ffmpeg/ffprobe, без запису на диск).

Рівномірність: t_i = (i + 1) * D / (N + 1), i = 0..N-1 — проміжки між кадрами
дорівнюють відступам від початку й кінця відео (кадри не торкаються країв).

Запис кадру: {index, time_sec, timestamp, width, height, size_kb, jpeg_bytes, image}.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .constants import JPEG_QUALITY_FFMPEG, MAX_SIDE, N_FRAMES, VIDEO_EXTS, VIDEOS_DIR
from .image_utils import jpeg_bytes_to_pil

__all__ = ["extract_frames", "get_video_meta", "uniform_timestamps", "format_ts",
           "list_videos", "resolve_video_path"]


def format_ts(total_sec: float) -> str:
    """Секунди → 'HH:MM:SS.mmm'."""
    total_sec = max(0.0, float(total_sec))
    mm, ss = divmod(total_sec, 60)
    hh, mm = divmod(mm, 60)
    return f"{int(hh):02d}:{int(mm):02d}:{ss:06.3f}"


def list_videos(videos_dir: str | Path = VIDEOS_DIR) -> list[str]:
    """Назви відеофайлів у папці (відсортовані)."""
    return sorted(p.name for p in Path(videos_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def resolve_video_path(video: str | Path, videos_dir: str | Path = VIDEOS_DIR) -> Path:
    """Назва відео (у `videos/`) або шлях → наявний шлях до файлу."""
    p = Path(video)
    if not p.exists():
        p = Path(videos_dir) / video
    if not p.exists():
        raise FileNotFoundError(f"відео не знайдено: {video} (папка {videos_dir})")
    return p


def get_video_meta(path: str | Path) -> dict:
    """Метадані відео через ffprobe: розмір, fps, тривалість, к-сть кадрів."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,avg_frame_rate,duration,nb_frames",
           "-show_entries", "format=duration,size", "-of", "json", str(path)]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format", {})
    try:
        num, den = map(int, stream.get("avg_frame_rate", "0/1").split("/"))
        fps = num / den if den else 0.0
    except ValueError:
        fps = 0.0
    nb = stream.get("nb_frames")
    return {
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": round(fps, 3),
        "duration_sec": round(float(fmt.get("duration") or stream.get("duration") or 0.0), 3),
        "nb_frames": int(nb) if str(nb or "").isdigit() else None,
        "size_mb": round(float(fmt.get("size", 0)) / (1024 * 1024), 2),
    }


def uniform_timestamps(duration_sec: float, n_frames: int) -> list[float]:
    """N таймкодів з однаковими проміжками включно з краями."""
    return [(i + 1) * duration_sec / (n_frames + 1) for i in range(n_frames)]


def _scale_filter(max_side: int | None) -> str | None:
    if max_side is None:
        return None
    return f"scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})'"


def extract_frames(video: str | Path, n_frames: int = N_FRAMES,
                   max_side: int | None = MAX_SIDE,
                   videos_dir: str | Path = VIDEOS_DIR, verbose: bool = False) -> list[dict]:
    """Витягнути `n_frames` рівномірних кадрів у пам'ять (ffmpeg → pipe → JPEG)."""
    path = resolve_video_path(video, videos_dir)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg не встановлено (потрібні ffmpeg і ffprobe у PATH)")
    if int(n_frames) < 1:
        raise ValueError(f"n_frames має бути >= 1, отримано {n_frames!r}")
    duration = get_video_meta(path)["duration_sec"]
    if duration <= 0:
        raise ValueError(f"не вдалося визначити тривалість відео: {path}")

    vf = _scale_filter(max_side)
    frames: list[dict] = []
    for idx, t in enumerate(uniform_timestamps(duration, int(n_frames))):
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path)]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-frames:v", "1", "-q:v", str(JPEG_QUALITY_FFMPEG),
                "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0 or not res.stdout:
            raise RuntimeError(f"ffmpeg не зміг витягнути кадр t={t:.3f} c: "
                               f"{(res.stderr or b'')[-500:].decode('utf-8', 'ignore')}")
        img = jpeg_bytes_to_pil(res.stdout)
        frames.append({
            "index": idx + 1,
            "time_sec": round(t, 3),
            "timestamp": format_ts(t),
            "width": img.size[0],
            "height": img.size[1],
            "size_kb": round(len(res.stdout) / 1024, 1),
            "jpeg_bytes": res.stdout,
            "image": img,
        })
    if verbose:
        print(f"🎞️ {path.name}: D={duration:.3f} c → {len(frames)} кадрів "
              f"(крок {duration / (int(n_frames) + 1):.3f} c), {frames[0]['width']}×{frames[0]['height']}")
    return frames
