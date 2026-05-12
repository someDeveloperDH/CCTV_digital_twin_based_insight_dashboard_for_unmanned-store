"""
calibrate_perspective.py - 원근 보정용 4점 캘리브레이션 도구

사용법:
    python calibrate_perspective.py --source data/test1.mp4

동작:
    1. 영상 첫 프레임이 뜸
    2. 매장 바닥의 사각형 4점을 순서대로 클릭 (좌상→우상→우하→좌하)
       ※ 실제로는 같은 크기의 직사각형이지만, 카메라 원근 때문에 사다리꼴로 보이는 영역
    3. 4점 찍으면 자동으로 calibration.json 저장
    4. test_mivolo_v2.py에서 --calibration calibration.json 으로 사용

팁:
    - 바닥 타일 패턴이 있으면 정사각형 타일 4개 꼭짓점을 찍으면 정확
    - 없으면 매장 바닥에서 직사각형으로 보이는 영역 (진열대 사이 통로 등) 활용
    - 상단(멀리)이 좁고 하단(가까이)이 넓은 사다리꼴 형태가 일반적
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np


class PointSelector:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.display = frame.copy()
        self.points = []
        self.labels = ["① 좌상 (멀리 왼쪽)", "② 우상 (멀리 오른쪽)",
                       "③ 우하 (가까이 오른쪽)", "④ 좌하 (가까이 왼쪽)"]
        self.colors = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0)]
        self.done = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            self._redraw()

            if len(self.points) == 4:
                self.done = True

        elif event == cv2.EVENT_RBUTTONDOWN and len(self.points) > 0:
            # 우클릭으로 마지막 점 삭제
            self.points.pop()
            self._redraw()

    def _redraw(self):
        self.display = self.frame.copy()

        # 안내 텍스트
        h, w = self.display.shape[:2]
        overlay = self.display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.display, 0.3, 0, self.display)

        if len(self.points) < 4:
            guide = f"[{self.labels[len(self.points)]}] 을 클릭하세요 (우클릭=되돌리기)"
        else:
            guide = "4점 완료! 아무 키나 누르면 저장됩니다."

        cv2.putText(self.display, guide, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 점 그리기
        for i, (px, py) in enumerate(self.points):
            color = self.colors[i]
            cv2.circle(self.display, (px, py), 8, color, -1)
            cv2.circle(self.display, (px, py), 10, (255, 255, 255), 2)
            cv2.putText(self.display, f"{i+1}", (px + 12, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 선 연결
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(self.display, self.points[i], self.points[i + 1],
                         (255, 255, 0), 2)
            if len(self.points) == 4:
                cv2.line(self.display, self.points[3], self.points[0],
                         (255, 255, 0), 2)

                # 변환 결과 미리보기
                self._preview_transform()

    def _preview_transform(self):
        """변환 결과 미리보기를 우상단에 표시"""
        src = np.float32(self.points)

        # 목표: 정사각형 (200x200)
        dst_size = 200
        dst = np.float32([[0, 0], [dst_size, 0],
                          [dst_size, dst_size], [0, dst_size]])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(self.frame, M, (dst_size, dst_size))

        # 미리보기를 우상단에 배치
        h, w = self.display.shape[:2]
        x_off = w - dst_size - 10
        y_off = 70
        self.display[y_off:y_off + dst_size, x_off:x_off + dst_size] = warped
        cv2.rectangle(self.display, (x_off - 2, y_off - 2),
                      (x_off + dst_size + 2, y_off + dst_size + 2),
                      (255, 255, 0), 2)
        cv2.putText(self.display, "Preview", (x_off, y_off - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def main():
    parser = argparse.ArgumentParser(description="원근 보정 캘리브레이션 도구")
    parser.add_argument("--source", type=str, default="data/test1.mp4",
                        help="입력 영상 경로")
    parser.add_argument("--output", type=str, default="calibration.json",
                        help="캘리브레이션 저장 경로")
    parser.add_argument("--frame", type=int, default=30,
                        help="캘리브레이션에 사용할 프레임 번호")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"[오류] 파일 없음: {args.source}")
        sys.exit(1)

    # 프레임 읽기
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[오류] 프레임을 읽을 수 없습니다.")
        sys.exit(1)

    h, w = frame.shape[:2]
    print("=" * 60)
    print("원근 보정 캘리브레이션")
    print("=" * 60)
    print(f"영상: {args.source} ({w}x{h})")
    print()
    print("매장 바닥에서 실제로는 직사각형인 영역의 4 꼭짓점을 클릭하세요.")
    print("순서: ① 좌상(멀리) → ② 우상(멀리) → ③ 우하(가까이) → ④ 좌하(가까이)")
    print("우클릭으로 마지막 점을 되돌릴 수 있습니다.")
    print("=" * 60)

    selector = PointSelector(frame)
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", selector.mouse_callback)

    while True:
        cv2.imshow("Calibration", selector.display)
        key = cv2.waitKey(30)

        if selector.done:
            cv2.imshow("Calibration", selector.display)
            cv2.waitKey(0)
            break

        if key == 27:  # ESC
            print("[취소] 캘리브레이션이 취소되었습니다.")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    # 캘리브레이션 데이터 저장
    src_points = selector.points
    calibration_data = {
        "source": args.source,
        "frame_size": [w, h],
        "src_points": src_points,
        "description": "4점: 좌상(멀리)→우상(멀리)→우하(가까이)→좌하(가까이)"
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(calibration_data, f, indent=2, ensure_ascii=False)

    print(f"\n[완료] 캘리브레이션 저장: {args.output}")
    print(f"  4점 좌표: {src_points}")
    print(f"\n사용법:")
    print(f"  python test_mivolo_v2.py --source {args.source} --save-video --show --calibration {args.output}")


if __name__ == "__main__":
    main()