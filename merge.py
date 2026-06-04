"""
클립 병합 + timeline.json 생성
- merged/{cam}_{day}.mp4       : 병합 영상
- merged/{cam}_{day}_timeline.json : 프레임 ↔ 절대시각 매핑
실행: python merge.py
"""
import os
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

BASE_DIR   = Path(__file__).parent
OUT_DIR    = BASE_DIR / "merged"
OUT_DIR.mkdir(exist_ok=True)

CLIP_PATTERN = r"XVR_(\w+)_\w+_(\d{14})_(\d{14})\.mp4"


def parse_clip(path: Path):
    import re
    m = re.match(CLIP_PATTERN, path.name)
    if not m:
        return None
    cam   = m.group(1)
    start = datetime.strptime(m.group(2), "%Y%m%d%H%M%S")
    end   = datetime.strptime(m.group(3), "%Y%m%d%H%M%S")
    return cam, start, end


def get_fps(path: Path):
    import subprocess, re
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True
    )
    frac = result.stdout.strip()
    if "/" in frac:
        a, b = frac.split("/")
        return float(a) / float(b)
    return float(frac) if frac else 25.0


def get_groups():
    groups = {}
    for ch_dir in sorted(BASE_DIR.glob("ch*")):
        if not ch_dir.is_dir():
            continue
        for day_dir in sorted(ch_dir.glob("[0-9]*")):
            clips = sorted(day_dir.glob("*.mp4"))
            if clips:
                key = f"{ch_dir.name}_{day_dir.name}"
                groups[key] = clips
    return groups


def build_timeline(key, clips, fps):
    """클립 목록 → timeline dict (ffmpeg 없이 파일명만으로 생성)"""
    timeline_clips = []
    frame_cursor   = 0
    for clip in clips:
        info = parse_clip(clip)
        if info is None:
            continue
        _, t_start, t_end = info
        duration_sec = (t_end - t_start).total_seconds()
        frame_count  = max(1, round(duration_sec * fps))
        timeline_clips.append({
            "clip":         clip.name,
            "frame_start":  frame_cursor,
            "frame_end":    frame_cursor + frame_count - 1,
            "time_start":   t_start.strftime("%Y-%m-%d %H:%M:%S"),
            "time_end":     t_end.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": duration_sec,
        })
        frame_cursor += frame_count
    return {
        "camera":       key.rsplit("_", 1)[0],
        "day":          key.rsplit("_", 1)[1],
        "fps":          fps,
        "total_frames": frame_cursor,
        "clips":        timeline_clips,
    }


def merge_group(key, clips):
    out_video    = OUT_DIR / f"{key}.mp4"
    out_timeline = OUT_DIR / f"{key}_timeline.json"

    if out_video.exists() and out_timeline.exists():
        print(f"  [SKIP] {key}")
        return

    fps      = get_fps(clips[0])
    timeline = build_timeline(key, clips, fps)

    # ── 타임라인만 생성 (영상이 이미 있을 때) ──
    if out_video.exists():
        print(f"  [TIMELINE] {key}: 영상 존재 → 타임라인만 생성")
    else:
        # ── ffmpeg 병합 ──
        print(f"  [MERGE] {key}: {len(clips)}개 클립")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for clip in clips:
                f.write(f"file '{clip.resolve()}'\n")
            list_file = f.name
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", list_file, "-c", "copy", str(out_video)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                print(f"  [ERROR] {result.stderr.decode()[-300:]}")
                return
        finally:
            os.unlink(list_file)

    with open(out_timeline, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)
    print(f"  [OK] {out_timeline.name}")


def main():
    groups = get_groups()
    print(f"총 {len(groups)}개 그룹 병합 시작\n")
    for key, clips in groups.items():
        merge_group(key, clips)
    print("\n완료")


if __name__ == "__main__":
    main()
