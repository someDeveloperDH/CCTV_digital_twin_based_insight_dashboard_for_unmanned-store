# CCTV 분석 파이프라인 변경 이력

> 변경 기록 포맷: 6하원칙 (누가·언제·어디서·무엇을·왜·어떻게) + 대안 비교 + 해결 여부

---

## [001] YOLOv8-Face 모델 다운로드 404 오류

- **언제**: 파이프라인 초기 구축 단계
- **어디서**: `face_mosaic.py` — 모델 로딩 부분
- **무엇을**: YOLOv8-Face 가중치 파일 다운로드 실패
- **왜**: 원래 URL(`github.com/akanametov/yolo-face`)이 404 반환

### 문제 상황
face_mosaic.py 실행 시 모델 파일을 다운로드하는 URL이 존재하지 않아 프로그램이 즉시 종료됨.

### 해결 대안들
| 대안 | 내용 | 단점 |
|------|------|------|
| A | GitHub 다른 레포 탐색 | 유지보수 불명확 |
| B | HuggingFace Hub 공식 배포본 사용 | — |
| C | OpenCV DNN 기반 얼굴 감지로 대체 | 정확도 낮음 |

### 선택: B — HuggingFace URL로 변경
`arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt`
공식 배포본으로 안정성이 높고 버전 관리가 명확함.

### 해결 여부: ✅ 해결

---

## [002] OpenCV GUI 동작 불가

- **언제**: zone_config.py 첫 실행 시
- **어디서**: WSL2 환경, opencv 패키지
- **무엇을**: `cv2.imshow()` 호출 시 화면 미표시
- **왜**: `opencv-python-headless`와 `opencv-python`이 동시 설치되어 충돌

### 문제 상황
zone_config.py 실행 시 창이 뜨지 않거나 즉시 닫힘.

### 해결 대안들
| 대안 | 내용 | 단점 |
|------|------|------|
| A | headless 제거 후 opencv-python 재설치 | — |
| B | matplotlib으로 시각화 대체 | 인터랙티브 폴리곤 그리기 불가 |
| C | WSLg 설정 점검 | 근본 원인이 패키지 충돌이라 의미 없음 |

### 선택: A
```bash
pip uninstall opencv-python-headless
pip install opencv-python --force-reinstall
```

### 해결 여부: ✅ 해결

---

## [003] analyze.py가 원본 영상을 입력으로 사용

- **언제**: analyze.py 초기 작성 직후
- **어디서**: `analyze.py` — `BASE_DIR` 경로 설정
- **무엇을**: 모자이크가 적용되지 않은 원본 영상으로 분석 수행
- **왜**: 입력 경로를 `mosaic/` 대신 기본 경로로 설정

### 문제 상황
프라이버시 보호를 위해 모자이크 처리를 먼저 하기로 했으나, 분석 단계에서 원본 영상을 읽어 들임. 파이프라인 설계 의도 위반.

### 해결 대안들
| 대안 | 내용 |
|------|------|
| A | `MOSAIC_DIR`를 `mosaic/`로 변경 |
| B | 모자이크 단계를 analyze.py 내부에 통합 | 처리 속도 저하, 관심사 분리 원칙 위반 |

### 선택: A — 입력 경로를 `mosaic/`로 수정

### 해결 여부: ✅ 해결

---

## [004] 성별/나이 분류를 얼굴 기반(InsightFace)으로 설계

- **언제**: analyze.py 성별 분류기 첫 설계 시
- **어디서**: `analyze.py` — 분류기 클래스
- **무엇을**: InsightFace(얼굴 특징 기반) 분류 사용
- **왜**: 얼굴 모자이크 처리 후 얼굴 정보가 소실되므로 분류 불가

### 문제 상황
모자이크된 영상에서 얼굴 좌표를 뽑더라도 해당 영역은 픽셀이 뭉개져 있어 얼굴 특징 추출 불가.

### 해결 대안들
| 대안 | 내용 | 단점 |
|------|------|------|
| A | InsightFace 유지 (얼굴 영역 제외한 원본 사용) | 프라이버시 원칙 위반 |
| B | MobileNetV3 바디 기반 분류 (하체 75% 영역) | 파인튜닝 없으면 정확도 낮음 |
| C | 체형 비율 규칙 기반 (키 추정값 임계값) | 단순하지만 카메라 캘리브레이션 필요 |

### 선택: B — MobileNetV3 바디 기반으로 교체
얼굴 정보 없이도 동작 가능하고, 추후 파인튜닝으로 정확도 개선 가능.

### 해결 여부: ⚠️ 부분 해결
구조는 변경됐으나 **파인튜닝 없이 pretrained 가중치만 사용 중**이라 분류 헤드가 랜덤 초기화 상태. 실제 분류 정확도 미보장.

---

## [005] body_proportion() 함수 미사용 파라미터

- **언제**: analyze.py 코드 리뷰 시
- **어디서**: `analyze.py:162` — `body_proportion(y1, y2, x1, x2, fh)`
- **무엇을**: `x1`, `x2` 파라미터가 함수 내부에서 사용되지 않음
- **왜**: 원근 보정 계산에 수평 좌표는 불필요 (높이 방향만 필요)

### 해결 대안들
| 대안 | 내용 |
|------|------|
| A | 파라미터 제거 → `body_proportion(y1, y2, fh)` |
| B | 수평 좌표도 활용하는 로직 추가 | 불필요한 복잡성 증가 |

### 선택: A — 미사용 파라미터 제거 및 호출부 수정

### 해결 여부: ✅ 해결

---

## [006] match_ch1(), build_visit_data() 시그니처 불일치

- **언제**: analyze.py 함수 리팩토링 후
- **어디서**: `analyze.py:608~615` — main() 내 호출부
- **무엇을**: `match_ch1`에 불필요한 `cam_names` 인자, `build_visit_data`에 불필요한 `zones` 인자 전달
- **왜**: 함수 시그니처에서 해당 파라미터를 제거했으나 호출부를 업데이트하지 않음

### 문제 상황
```python
# 잘못된 호출 (오류 발생)
match_ch1(ch1_td, store_tds, gid_maps, store_cams, next_counter)  # cam_names 불필요
build_visit_data(..., ch1_td, ch1_gid_map, zones)                  # zones 불필요
```

### 해결 대안들
| 대안 | 내용 |
|------|------|
| A | 호출부에서 잉여 인자 제거 |
| B | 함수 시그니처에 파라미터를 다시 추가 | 불필요한 파라미터 유지, 코드 품질 저하 |

### 선택: A — 호출부 수정

### 해결 여부: ✅ 해결

---

## [007] 파이프라인의 목적 부합성 평가

- **언제**: 2026-05-06
- **어디서**: 전체 파이프라인
- **무엇을**: 현재 구현이 실제 목표(성별·나이·구매 감지)를 달성하는지 점검
- **왜**: 코드는 존재하지만 실제 동작 및 정확도가 검증되지 않음

### 문제 상황 요약

| 목표 | 상태 | 비고 |
|------|------|------|
| 성별 분류 | ❌ 미동작 | 분류 헤드 랜덤 초기화, 파인튜닝 없음 |
| 성인/미성년 분류 | ⚠️ 불확실 | 원근 보정 로직은 있으나 미검증 |
| 매대 체류 시간 | ⚠️ 조건부 | polygon 설정 완료 시 동작 |
| 구매 여부 감지 | ⚠️ 단순 | 계산대 앞 체류 시간 기준 (POS 미연동) |
| 구매 물품 특정 | ❌ 불가 | 방문 매대의 모든 상품으로 일괄 가정 |
| Cross-camera Re-ID | ⚠️ 미검증 | 설계는 합리적, 실 데이터 미검증 |
| end-to-end 실행 | ❌ 미실행 | 한 번도 실제 영상으로 실행 안 됨 |

### 미결 과제 (우선순위 순)
1. **성별/나이 분류기 교체 또는 파인튜닝** — 파이프라인 핵심 기능
2. **analyze.py end-to-end 실행 및 디버깅**
3. **구매 감지 로직 정교화** (선택적)

### 해결 여부: 🔴 진행 중

---

## [008] 파이프라인 전면 재설계 — 목적 부합성 + 속도 개선

- **언제**: 2026-05-06
- **어디서**: analyze.py, face_mosaic.py, validate.py, run_pipeline.py(신규)
- **무엇을**: 비동작 성별 분류기 교체 + 단계별 게이팅 + 처리 속도 개선
- **왜**: 평가 결과 핵심 기능(성별/나이)이 미동작이고 end-to-end 실행 경로가 없었음

### 문제 상황
| 항목 | 문제 |
|------|------|
| 성별 분류 | MobileNetV3 분류 헤드 랜덤 초기화 → 실질적 미동작 |
| 파이프라인 진입점 | 단계별 실행 방법이 없음 |
| 처리 속도 | 프레임 스킵 없음, FP16 미적용, BATCH_SIZE=4 |
| 디버깅 수단 | 단계별 수치 출력 없음 |

### 해결 대안들

**성별 분류:**
| 대안 | 내용 | 단점 |
|------|------|------|
| A | 파인튜닝 | 레이블 데이터 필요 |
| B | HuggingFace 얼굴 기반 모델 | 모자이크 이후라 불가 |
| C | **COCO 키포인트 어깨/골반 비율** | 정확도 ~65%, 별도 모델 불필요 |

**속도 개선:**
| 항목 | 이전 | 이후 |
|------|------|------|
| YOLO 모델 | yolov8n.pt | yolov8n-pose.pt (키포인트 포함) |
| 프레임 처리율 | 100% | 33% (FRAME_SKIP=2, 3× 속도) |
| 정밀도 | FP32 | FP16 (CUDA, ~1.5× 속도) |
| 모자이크 배치 | 4 | 8 |

### 선택: C + FRAME_SKIP + FP16 + run_pipeline.py 신설

**성별**: 키포인트 5(좌어깨), 6(우어깨), 11(좌골반), 12(우골반)의 너비 비율
- ratio ≥ 1.08 → Male (신뢰도 0.58~0.82)
- ratio ≤ 0.98 → Female (신뢰도 0.58~0.82)
- 0.98~1.08 구간 → None (불확실)

**게이팅 기준치:**
| 단계 | 지표 | 기준 |
|------|------|------|
| 데이터 확인 | 카메라당 merged 파일 | ≥1 |
| 모자이크 | 처리 속도 | ≥3.0fps |
| 구역 설정 | 설정률 | ≥67% |
| 분석 | 카메라당 추적 수 | ≥3명 |
| 분석 | 처리 속도 | ≥1.0fps |
| CSV 출력 | 방문자 수 | ≥1명 |
| 검증 | 종합 점수 | ≥0.30 |

### 변경된 파일
- `analyze.py`: 전면 재작성 (YOLO pose, 키포인트 성별, FRAME_SKIP, FP16, metrics 반환)
- `face_mosaic.py`: BATCH_SIZE 4→8, FP16, metrics 반환
- `validate.py`: `run_validation()` 진입점 추가
- `run_pipeline.py`: 신규 — 6단계 오케스트레이터, PASS/FAIL 게이팅, JSON 로그

### 해결 여부: ✅ 구조 완성 / ⏳ 실 영상 검증 필요

---

## [009] 파이프라인 단계별 소요 시간 출력 추가

- **언제**: 2026-05-07
- **어디서**: `run_pipeline.py` — `run_stage()` 함수 및 최종 요약 블록
- **무엇을**: 각 단계가 끝날 때 소요 시간을 알 수 없었음
- **왜**: 전체 시간만 출력되고 어느 단계에서 병목이 발생했는지 파악 불가
- **어떻게**: `run_stage()` 내부에 타이머 추가, 종료 시 단계별 요약표 출력

### 문제 상황
파이프라인 종료 후 총 소요 시간만 표시되어, 영상 분석/모자이크 등 어느 단계가 느린지 알 수 없었음.

### 해결 대안들
| 대안 | 내용 | 단점 |
|------|------|------|
| A | 각 stage 함수 내부에서 반환값에 elapsed 포함 | 모든 함수 시그니처 변경 필요 |
| B | **`run_stage()`에서 fn() 호출 전후 타이머 측정** | 변경 최소화 |
| C | 외부 `time.time()` 직접 호출 위치 마다 삽입 | 중복 코드 |

### 선택: B — `run_stage()` 한 곳만 수정
```python
t0_stage = time.time()
ok, meta = fn(*a)
elapsed_stage = round(time.time() - t0_stage, 1)
run_log["stages"].append({..., "elapsed_sec": elapsed_stage})
```

### 출력 형태
```
  단계                        소요시간  결과      비율
  데이터 확인                     0.3초  ✅ PASS   ░ 0.1%
  얼굴 모자이크                  61.2초  ✅ PASS   ████████████ 20.5%
  영상 분석                     231.4초  ✅ PASS   ████████████████████████████████████ 77.6%
  합계                          298.6초  (4.98분)
```
- 5%당 █ 1칸으로 병목 단계를 시각적으로 표현
- `logs/run_*.json`에도 `elapsed_sec` 필드로 저장

### 해결 여부: ✅ 해결

---

## [010] ch1 기준 타임라인 + Re-ID 전면 재설계

- **언제**: 2026-05-07
- **어디서**: `merge.py`, `run_pipeline.py`, `analyze.py`
- **무엇을**: 타임라인 자동 생성 + ch1 anchor 기반 Re-ID 재설계
- **왜**: 두 가지 잘못된 가정 발견
  1. ch3/ch6 영업 전 구간(ch6는 자정부터 녹화)도 분석에 포함됨
  2. ch3→ch6 단방향 가정으로 ch3만 방문하거나 왔다갔다하는 손님 ID 누락

### 문제 상황
| 카메라 | 녹화 시작 | 녹화 종료 |
|--------|----------|----------|
| ch1 | 08:41 | 23:49 (영업시간) |
| ch3 | 06:16 | 23:47 (영업 전 2.5시간 포함) |
| ch6 | 00:17 | 23:48 (자정부터 시작) |

### 해결 대안들
| 대안 | 내용 | 단점 |
|------|------|------|
| A | ch1 클립 시간을 전체 분석 창으로 고정 | 개인별 창 반영 안 됨 |
| B | **ch1 개인별 [entry, exit] 창으로 store 매칭** | 복잡도 증가 |
| C | 전체 시간 포함 후 후처리 필터링 | 불필요한 연산 |

### 선택: B + 영업 창 스킵 최적화

| 변경 항목 | 내용 |
|-----------|------|
| `assign_global_ids()` | 제거 (ch3↔ch6 직접 매칭 폐지) |
| `match_ch1()` | 제거 |
| `match_by_ch1_window()` | 신설: ch1 개인별 창, 왔다갔다 허용, 중복 제거 |
| `process_camera()` | `biz_window` 파라미터 추가 |
| `merge.py` | 영상 있으면 타임라인만 생성 |
| Stage 1 | 타임라인 없으면 자동 생성 |
| Re-ID 가중치 | 외모(0.35)+색상(0.30)+체형(0.20)+시간(0.15), 동선 제거 |

### 해결 여부: ✅ 구조 완성 / ⏳ 실 영상 검증 필요

---

## [011] zone_config.py 카메라별 구역 설정 재설계

- **언제**: 2026-05-07
- **어디서**: `zone_config.py`, `zones.json`, `analyze.py`, `run_pipeline.py`
- **무엇을**: 구역 설정 창이 안 뜨고, 카메라별로 다른 구역이 보이는 문제
- **왜**: 세 가지 문제 동시 발생
  1. mosaic 경로 오류로 창이 조용히 종료됨
  2. 카메라 하나에만 폴리곤 설정 — ch1/ch3/ch6 각각 화각이 다름
  3. `"polygon": []` 단일 구조로 카메라별 구분 불가

### 해결 대안들
| 대안 | 내용 | 단점 |
|------|------|------|
| A | 카메라마다 별도 zones_ch3.json | 파일 관리 복잡 |
| B | **zones.json에 `polygons: {cam: [...]}` 구조** | 마이그레이션 필요 |
| C | 공통 좌표계로 통합 | 캘리브레이션 필요 |

### 선택: B
- `zones.json`: `polygon` → `polygons: {"ch1":[], "ch3":[], "ch6":[]}`
- `zone_config.py`: `--cameras ch3,ch6,ch1` 순회, mosaic→merged 폴백, S=skip 키
- `analyze.py`: `load_zones(cam)` — 카메라별 폴리곤만 로드
- `run_pipeline.py` Stage 3: 카메라별 현황 출력

### 실행 방법
```bash
python zone_config.py --cameras ch3,ch6,ch1 --day 10
```

### 해결 여부: ✅ 해결

---

## [012] run_pipeline.py Stage 3에 구역 설정 GUI 통합

- **언제**: 2026-05-07
- **어디서**: `run_pipeline.py` — `stage_zones()`
- **무엇을**: 구역 미설정 시 파이프라인 중단 → GUI 자동 실행 후 재검증
- **왜**: end-to-end 단일 실행 요구. zone_config.py 별도 실행은 파이프라인 분리

### 문제 상황
구역 미설정 시 Stage 3 FAIL로 파이프라인 중단. 수동으로 `python zone_config.py` 실행 필요.

### 해결 대안들
| 대안 | 내용 | 단점 |
|------|------|------|
| A | FAIL 후 안내 메시지만 출력 | 사용자가 별도 실행 필요 |
| B | **미설정 감지 시 zone_config GUI 자동 실행** | — |
| C | 구역 설정 없이 전체 프레임 분석 | 매대 체류 시간 측정 불가 |

### 선택: B
```
Stage 3 실행 → 미설정 카메라 감지 → GUI 자동 실행 → 폴리곤 그리기 → 재검증 → PASS/FAIL
```

### 해결 여부: ✅ 해결

---

## [013] 구역 설정 3가지 문제 + FP16 dtype 충돌 수정

- **언제**: 2026-05-07
- **어디서**: `zones.json`, `zone_config.py`, `analyze.py`
- **무엇을**: 4가지 문제 동시 발생
- **왜**: 설계 단계에서 누락된 구역, UI 미흡, FP16 이중 변환 버그

### 문제 상황

| # | 문제 | 원인 |
|---|------|------|
| 1 | 계산대·출입문 구역 없음 | zones.json에 미포함 |
| 2 | 구역 지정 전 안내 없음 | UI 없음 |
| 3 | 카메라별 구역 선택 불가 | 모든 구역 강제 순회 |
| 4 | `RuntimeError: c10::Half != float` | `model.model.half()` + `half=True` 이중 변환 |

### 해결

**문제 1:** zones.json에 계산대(checkout), 출입문(entrance) 추가 → 총 11개

**문제 2, 3:** `select_visible_zones()` 함수 신설
```
[ch3] 이 카메라에서 보이는 구역 선택
  힌트: ch3은 매장 앞쪽을 바라봅니다
   1. 아이스크림1   ⬜ 미설정
   ...
  보이는 구역 번호 입력 (예: 1,2,4): 1,2,3,7
```
- 선택된 구역만 그리기 진입
- 미선택 구역은 자동 skip 처리

**문제 4:** `model.model.half()` 제거
- YOLO 내부 fuse_conv_and_bn이 FP16 상태에서 실행되면 충돌
- `model.track(..., half=True)`만 사용 → YOLO가 fusion 후 FP16 변환

### 해결 여부: ✅ 해결

---

## [014] 계산대 상품 감지 모듈 추가 (checkout_detector.py)

- **언제**: 2026-05-07
- **어디서**: 신규 `checkout_detector.py`, `analyze.py`
- **무엇을**: CCTV만으로 구매 상품 종류 감지
- **왜**: 기존 "계산대 체류 시간" 방식은 구매 여부만 판단, 무엇을 샀는지 불가

### 문제 상황
- CCTV만 사용 (소리, POS 연동 없음)
- ch1이 계산대를 내려다보고 있음 → 계산대 위 상품 감지 가능

### 선택: 배경 차분 + CLIP 제로샷 분류

**흐름:**
```
계산대 ROI 배경 차분 (MOG2)
  → N프레임 지속 변화 = 상품 올려놓음
  → CLIP 제로샷 분류 (막대형 아이스크림 / 콘 / 음료 / 과자)
  → (zone_ids, confidence, label) 반환
  → visit_data에 purchased_zone_ids 기록
```

**핵심 파라미터:**
- MIN_CHANGE_AREA=600: 노이즈와 실제 상품 구분
- PERSIST_FRAMES=8: 손이 지나가는 것 vs 놓여진 상품 구분
- DEBOUNCE_FRAMES=45: 중복 감지 방지
- CLIP_CONF_MIN=0.30: 낮은 신뢰도 결과 제외

**CLIP 레이블 (영문, 다국어 성능 우수):**
- ice cream bar popsicle → ice1, ice5, ice6
- ice cream cone → ice2, ice3
- ice cream tub container → ice4
- beverage can bottle → beverage
- snack bag chips → snack1, snack2

### 해결 여부: ✅ 구조 완성 / ⏳ 실 영상 정확도 검증 필요

---

## [015] 스캐너 구역 + 상품 DB 매칭 방식으로 구매 감지 재설계

- **언제**: 2026-05-07
- **어디서**: `checkout_detector.py` (재작성), `build_product_db.py` (신규), `analyze.py`, `zones.json`
- **무엇을**: CLIP 제로샷 → 사용자 직접 촬영 DB + 코사인 유사도 매칭으로 교체
- **왜**: 스캐너 구역이 카메라에서 명확히 보임. 직접 DB 구성이 더 정확하고 현실적인 MVP

### 설계
```
사용자 작업:
  product_db/<상품명>.jpg 촬영 저장
  python build_product_db.py → embeddings.npz 생성
  zone_config에서 "스캐너" 구역 폴리곤 설정

분석 시:
  스캐너 구역에 물건 올려짐 (배경 차분 + 6프레임 지속)
    → MobileNetV3 임베딩 추출
    → embeddings.npz 코사인 유사도 1-NN 매칭
    → zones.json["products"]에서 zone_name / price 조회
    → visit_data.json purchased_items에 기록
```

### 변경 사항
| 항목 | 이전 | 이후 |
|------|------|------|
| 분류 방법 | CLIP 제로샷 (범용) | MobileNetV3 + 상품 DB (특화) |
| DB 구성 | 없음 | product_db/*.jpg → embeddings.npz |
| 구역 이름 | 계산대 | 스캐너 (zones.json에 추가) |
| 선명도 선택 | 없음 | Laplacian 분산으로 최선 크롭 선택 |
| 결과 구조 | zone_ids + label | product + zone_name + price + similarity |

### 사용 방법
```bash
# 1. 상품 촬영 후 저장
cp 더위사냥_사진.jpg product_db/더위사냥.jpg

# 2. DB 빌드
python build_product_db.py

# 3. zone_config에서 스캐너 구역 설정
python run_pipeline.py --day 10
```

### 해결 여부: ✅ 구조 완성 / ⏳ 실 상품 이미지 DB 구성 필요

---

## [016] 구역 ID/이름 불일치 수정 (문제 3)

- **언제**: 2026-05-07
- **어디서**: `run_pipeline.py` — `stage_export()` CSV 후처리
- **무엇을**: 구역ID 컬럼이 한글 이름으로 저장되어 GT(영문 ID)와 Jaccard = 0 발생
- **왜**: analyze.py가 zone name(한글)으로 체류시간 저장 → export가 그대로 구역ID에 사용

### 문제 상황
| 컬럼 | 기존 PR | GT | 결과 |
|------|---------|-----|------|
| 구역ID | 아이스크림1 | ice1 | 불일치 → Jaccard=0 |
| 구역명 | 아이스크림1 | 음료 | 동일 컬럼 중복 |

### 해결
run_pipeline.py stage_export에서 CSV 생성 직후 후처리:
- zones.json에서 `name→id` 매핑 생성
- `ice_batch_purchases.csv` 구역ID: 한글 → 영문 ID
- `ice_zone_kpi.csv` 구역: 한글 → 영문 ID
- 구역명은 한글 그대로 유지

### 수정 후
| 컬럼 | PR | GT |
|------|----|----|
| 구역ID | ice1 | ice1 ✅ |
| 구역명 | 아이스크림1 | - |

### 해결 여부: ✅ 해결

---

## [017] BoT-SORT + MiVOLO v2 통합 (run_pipeline.py)

- **언제**: 2026-05-08
- **어디서**: `run_pipeline.py` — `_setup_botsort()`, `_run_mivolo_pass()`, `stage_analyze()`
- **무엇을**: 객체 추적기 교체(BoT-SORT) + 성별/나이 분류기 교체(MiVOLO v2)
- **왜**: 오감지 162명(추적 단절) 문제와 성별 편향(adult_female→minor_male) 문제 해결

### 해결 대안들
| 항목 | 기존 | 변경 |
|------|------|------|
| 추적기 | ByteTrack (IoU만) | BoT-SORT (IoU + 외모 Re-ID) |
| 성별/나이 | MobileNetV3 + 키포인트 비율 | MiVOLO v2 (바디 전용 모드) |

### 구현 방식 (run_pipeline.py만 수정)

**BoT-SORT**: `_setup_botsort()`
- `ultralytics.YOLO.track` 전역 monkey-patch
- `tracker="botsort.yaml"` 주입 → analyze.py 수정 없이 적용

**MiVOLO v2**: `_run_mivolo_pass(day)`
- `run_analysis()` 완료 후 후처리 패스
- 카메라별 20프레임마다 샘플링 → MiVOLO 바디 분류
- 시간 창 매칭으로 방문자별 다수결 → visit_data.json gender/is_adult 업데이트
- 가중치 없으면 자동 건너뜀 (INFO 표시)

### 필요 파일
```
weights/mivolo_d1.pth.tar          ← MiVOLO GitHub Releases
weights/yolov8x_person_face.pt     ← 옵션
```

### 해결 여부
- BoT-SORT: ✅ 적용 완료
- MiVOLO: ✅ 구조 완성 / ⏳ 가중치 다운로드 필요
