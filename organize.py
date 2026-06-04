import os
import shutil
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
OUT_DIR = Path(__file__).parent

# XVR_ch1_main_20260404000206_20260404000221.mp4
PATTERN = re.compile(r"XVR_(ch\d+)_\w+_(\d{4})(\d{2})(\d{2})\d{6}_\d+\.mp4")

def organize():
    files = list(RAW_DIR.glob("*.mp4"))
    if not files:
        print("raw 폴더에 mp4 파일이 없습니다.")
        return

    moved = 0
    skipped = 0

    for file in files:
        m = PATTERN.match(file.name)
        if not m:
            print(f"[SKIP] 패턴 불일치: {file.name}")
            skipped += 1
            continue

        channel = m.group(1)   # ch1
        day     = m.group(4)   # DD

        dest_dir = OUT_DIR / channel / day
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest = dest_dir / file.name
        shutil.move(str(file), str(dest))
        print(f"[MOVE] {file.name} -> {channel}/{day}/")
        moved += 1

    print(f"\n완료: {moved}개 이동, {skipped}개 건너뜀")

if __name__ == "__main__":
    organize()
