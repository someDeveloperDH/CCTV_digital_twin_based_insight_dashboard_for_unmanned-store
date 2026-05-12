# 매장 CCTV 분석 파이프라인

CCTV 영상에서 사람을 추적하고, 남녀노소(4클래스)로 분류하여 클래스별 히트맵을 생성합니다.

## 기술 스택
- **Detection**: YOLOv8n (COCO pretrained, person class)
- **Tracking**: ByteTrack (IoU 기반, CPU 최적화)
- **Classification**: DeepFace (OpenCV backend, 성별/연령 추정)
- **Heatmap**: OpenCV Gaussian Blur + Matplotlib

## 프로젝트 구조
```
├── main.py                  # 메인 파이프라인
├── tracker.py               # YOLOv8 + ByteTrack 추적
├── classifier.py            # DeepFace 성별/연령 분류
├── heatmap_generator.py     # 히트맵 생성
├── utils.py                 # 유틸리티 함수
├── requirements.txt         # 의존성
├── data/
│   └── test1.mp4            # 입력 영상 (직접 준비)
└── output/                  # 결과물 저장 디렉토리
    ├── tracked_output.mp4   # 추적 결과 영상
    ├── heatmap_young_male.png
    ├── heatmap_young_female.png
    ├── heatmap_old_male.png
    ├── heatmap_old_female.png
    ├── heatmap_combined.png
    ├── heatmap_comparison.png
    └── tracking_coordinates.csv
```

## 설치 및 실행

### 1. 환경 설정
```bash
# Python 3.9 이상 권장
pip install -r requirements.txt
```

### 2. 영상 준비
```bash
mkdir -p data
# data/test1.mp4 에 테스트 영상 배치
```

### 3. 실행
```bash
# 기본 실행 (히트맵만 생성)
python main.py

# 추적 결과 영상도 저장
python main.py --save-video

# GUI 환경에서 실시간 확인
python main.py --save-video --show

# 옵션 전체 보기
python main.py --help
```

## 실행 옵션
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--source` | `data/test1.mp4` | 입력 영상 경로 |
| `--output-dir` | `output` | 결과 저장 디렉토리 |
| `--save-video` | off | 추적 결과 영상 저장 |
| `--show` | off | 실시간 화면 표시 |
| `--classify-interval` | 5 | 분류 시도 간격 (프레임) |
| `--classify-max-attempts` | 10 | ID당 최대 분류 시도 |
| `--heatmap-resolution` | 1.0 | 히트맵 해상도 배율 |

## 4클래스 분류 기준
| 클래스 | 조건 |
|--------|------|
| 젊은 남성 (young_male) | 나이 < 40 & 남성 |
| 젊은 여성 (young_female) | 나이 < 40 & 여성 |
| 중장년 남성 (old_male) | 나이 >= 40 & 남성 |
| 중장년 여성 (old_female) | 나이 >= 40 & 여성 |

## CPU 환경 최적화 포인트
1. **YOLOv8n**: 가장 가벼운 nano 모델 사용
2. **ByteTrack**: Re-ID 모델 없이 IoU 기반으로 동작 → 추가 연산 없음
3. **분류 캐싱**: ID당 한 번만 DeepFace 호출, 이후는 캐시 사용
4. **OpenCV backend**: DeepFace의 가장 빠른 얼굴 검출기 사용

## 참고사항
- CPU 환경에서 1분 영상 기준 수 분 소요될 수 있음
- 첫 실행 시 YOLOv8n, DeepFace 모델 자동 다운로드
- 얼굴이 잘 안 보이는 객체는 기본값(young_male)으로 분류됨
- OpenCV는 한글 렌더링이 안 되므로 영상 내 텍스트는 영문으로 표시
