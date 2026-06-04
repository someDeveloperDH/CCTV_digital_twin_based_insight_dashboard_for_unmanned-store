# :camera: CCTV 기반 무인매장 디지털 트윈 인사이트 대시보드

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/>&nbsp;&nbsp;<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/>&nbsp;&nbsp;<img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=flat&logo=YOLO&logoColor=black"/>&nbsp;&nbsp;<img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=FastAPI&logoColor=white"/>&nbsp;&nbsp;<img src="https://img.shields.io/badge/React-61DAFB?style=flat&logo=React&logoColor=black"/>&nbsp;&nbsp;<img src="https://img.shields.io/badge/Vite-646CFF?style=flat&logo=Vite&logoColor=white"/>&nbsp;&nbsp;<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=OpenCV&logoColor=white"/>&nbsp;&nbsp;<img src="https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=flat&logo=TailwindCSS&logoColor=white"/>
</p>

<br>

## 📌 프로젝트 개요

### 소개

무인매장은 직원이 없어 고객 행동 데이터를 수집하기 어렵습니다.  
하루 매출이 얼마인지, 어느 매대가 잘 팔리는지, 언제 손님이 몰리는지 — 이 모든 것을 **감**으로만 운영해야 합니다.

본 프로젝트는 **매장 내 CCTV 영상만으로** 이 문제를 해결합니다.  
7일치 실측 데이터로 시뮬레이터를 보정(Calibrate)하고, 4개월치 디지털 트윈 데이터를 생성한 뒤,  
딥러닝 모델이 **내일의 KPI를 예측**하여 대시보드에 띄워줍니다.

- **기간**: 2026.04 ~ 2026.06
- **대상**: 아이스크림 무인매장 (채널 3대: ch1·ch3·ch6, 구역 9개)

<br>

## 👥 팀원 및 역할

### 팀원

안은정|이준구|박동혁
:-:|:-:|:-:
<a href="https://github.com/2293095-bli" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/2jungoo" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/someDeveloperDH" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>

### 역할

| 이름 | 역할 및 담당 업무 |
|------|--------------------------------------------------------------|
| **안은정** <br> (PM) | -KPI 지표 연구 및 선정 <br> -UI/UX 설계 <br> - 데이터 컨텍 및 수집 <<br> - 데이터셋 전처리 (`PurchaseGridDataset`) <br> - 문서화작업 (판넬, PPT, 보고서 ) <br> - 경영적 서비스 구상 |
| **이준구** | - Transformer 예측 모델 설계·학습 (`model/model.py`, `train.py`) <br> - SpaceTimeBlock(공간+시간 이중 어텐션) 구현 <br> - FastAPI 예측 서버 구현 (`server/`) <br> - 모델 성능 평가 및 비교 분석  React 시뮬레이션 대시보드 개발 (`simul/`) <br> - Web Worker 기반 4개월 배치 시뮬레이션 (`simulationWorker.js`) <br> - 실측 KPI 기반 시뮬레이터 보정(Calibration)  <br> - AI 예측 패널 및 히트맵 시각화 |
| **박동혁** <br> (CCTV분석 - 개발)  | - CCTV 영상 분석 파이프라인 설계 및 구현 (`run_pipeline.py`) <br> - YOLOv8-pose 기반 사람 감지·추적(BoT-SORT, Re-ID) <br> - 키포인트 기반 성별·나이 판별 (생체역학 모델) <br> - MiVOLO v2 후처리 분류 연동 <br> - 구역 체류·구매 판정 로직 및 KPI CSV 내보내기 <br> - SpaceTimeBlock(공간+시간 이중 어텐션) 구현 <br> - 데이터셋 전처리 (`PurchaseGridDataset`) <br> - Crossformer 다중 KPI 예측 모델 구현 (`kpi_crossformer.py`)<br> - 데이터셋 전처리 (`PurchaseGridDataset`)|


<br>

## 🏗️ 서비스 아키텍처

```
[실제 매장 CCTV]
      │
      ▼
┌─────────────────────────────────────────────────┐
│            CCTV 분석 파이프라인 (Python)           │
│                                                  │
│  organize.py → merge.py → face_mosaic.py        │
│       → zone_config.py → analyze.py → export.py │
│                                                  │
│  · YOLOv8n-pose : 사람 감지 + 키포인트            │
│  · BoT-SORT     : 다중 객체 추적 (Re-ID)          │
│  · MiVOLO v2    : 성별·나이 분류                  │
│  · zones.json   : 구역별 폴리곤                   │
└──────────────────┬──────────────────────────────┘
                   │  KPI CSV 3종
                   │  (ice_zone_kpi / ice_class_kpi / ice_batch_purchases)
                   ▼
┌─────────────────────────────────────────────────┐
│         디지털 트윈 파이프라인 (React + FastAPI)    │
│                                                  │
│  Calibration  →  4개월 시뮬레이션  →  KPI 분석    │
│  (9구역 전환율     (84,406건, 120일)   BatchAnalytics│
│   100% 일치)                                     │
│                                                  │
│  [딥러닝 학습]          [대시보드]                  │
│  Transformer  ──────▶  실시간 시뮬 뷰              │
│  Crossformer  ──────▶  배치 분석 + 인사이트 5종    │
│  LSTM (비교)           AI 예측 패널               │
│                                                  │
│  FastAPI (POST /predict) ◀──▶ AIPredictionPanel  │
└─────────────────────────────────────────────────┘
```

<br>

## 📁 파일 구조

```
📦 cctv1/
├── 📂 cctv 분석 파이프라인
│   ├── run_pipeline.py          # 전체 파이프라인 오케스트레이터 (PASS/FAIL 게이팅)
│   ├── organize.py              # 원본 영상 채널·날짜별 정리
│   ├── merge.py                 # 클립 병합 + 타임라인 JSON 생성
│   ├── face_mosaic.py           # YOLOv8n-face 기반 얼굴 모자이크
│   ├── zone_config.py           # 구역 폴리곤 설정 GUI
│   ├── analyze.py               # 사람 감지·추적·분류·체류 분석
│   ├── export.py                # KPI CSV 내보내기
│   └── validate.py              # 결과 검증
│
├── 📂 simul/                    # React + Vite 대시보드
│   ├── src/
│   │   ├── simulationWorker.js  # Web Worker 배치 시뮬레이션
│   │   ├── BatchAnalytics.jsx   # KPI 분석 엔진 (computeKPIs)
│   │   ├── predictionModels.js  # 브라우저 경량 ML 모델
│   │   └── components/
│   │       └── AIPredictionPanel.jsx  # AI 예측 패널
│   └── server/                  # FastAPI 백엔드
│       ├── main.py              # REST API 서버
│       ├── predict.py           # Transformer + Crossformer 추론
│       └── converter.py         # 구매 이벤트 → 격자 텐서 변환
│
├── 📂 model/                    # 딥러닝 모델
│   ├── model.py                 # PurchaseGridTransformer (SpaceTimeBlock)
│   ├── train.py                 # 학습 스크립트
│   ├── dataset.py               # PurchaseGridDataset
│   ├── evaluate.py              # 성능 평가
│   └── config.yaml              # 하이퍼파라미터
│
├── 📂 retail/                   # YOLO 기반 행동 분석
│   ├── main.py                  # 포즈 추정 + 히트맵
│   └── heatmap_generator.py
│
├── 📂 weights/                  # 모델 가중치 저장 경로
├── 📂 zone_images/              # 구역 설정 캡처 이미지
├── zones.json                   # 구역별 카메라 폴리곤 좌표
├── kpi_crossformer.py           # Crossformer 학습 실행
├── rnn(lstm)_times.py           # LSTM 학습 실행
├── ice_batch_purchases.csv      # 실측 7일 구매 로그 (65건)
├── ice_zone_kpi.csv             # 실측 구역별 KPI (9구역)
└── ice_class_kpi.csv            # 실측 고객군별 KPI (4분류)
```

<br>

## ▶️ How to run

### 0. 환경 준비

```bash
# Python 의존성
pip install ultralytics torch torchvision opencv-python timm scipy
pip install fastapi uvicorn pydantic pandas pyyaml

# Node.js 의존성
cd simul && npm install
```

### 1. CCTV 분석 파이프라인

```bash
cd /home/killy/capston/cctv1

# 전체 파이프라인 (권장)
python run_pipeline.py --day 10

# 옵션
python run_pipeline.py --day 10 --skip-mosaic    # 모자이크 단계 건너뜀
python run_pipeline.py --day 10 --force          # FAIL 나도 강제 진행
python run_pipeline.py --day 10 --display        # 영상 실시간 표시

# 단계별 개별 실행
python organize.py
python merge.py
python face_mosaic.py
python zone_config.py --camera ch3    # 구역 설정 GUI
python analyze.py --day 10
python export.py
python validate.py
```


### 2. 딥러닝 모델 학습

```bash
# Transformer (메인 모델)
cd model && python train.py

# Crossformer
cd .. && python kpi_crossformer_local.py

# LSTM (비교 모델)
python "rnn(lstm)_times_local.py"
```

### 3. FastAPI 예측 서버

```bash
pip install -r server/requirements.txt

# 서버 실행
uvicorn server.main:app --host 0.0.0.0 --port 8000

# 개발 모드 (자동 재시작)
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 대시보드 실행

```bash
cd simul
npm install
npm run dev -- --host    # WSL 환경: Network IP로 접속
```

> WSL 환경에서는 터미널에 표시되는 **Network 주소(172.xx.xx.xx:5173)** 를 Windows 브라우저에서 열어주세요.

<br>

## 📊 평가

### 딥러닝 모델 성능 비교

3가지 모델을 동일한 시뮬레이션 데이터(84,406건, 120일)로 학습 후 비교했습니다.  
학습/검증/테스트는 시간 순서 기준으로 분할 (Train 1~84일 /  Val 85~104일  / Test 105~120일).

<br>

- **Transformer 선택 이유**: 구역 판매 순위를 완벽히 예측(ρ=1.000)하고, MAPE 4.0%로 오차가 가장 낮으며, 기준 모델 대비 31% 향상
- **LSTM 탈락 이유**: 점심·저녁 피크 패턴을 포착하지 못해 기준 모델보다 오히려 성능 열세
- **Crossformer 역할**: 체류 시간·효율 지수 등 다중 KPI 동시 예측 담당

### 대시보드 자동 인사이트

시뮬레이션 완료 후 자동으로 생성되는 인사이트 5종:

| # | 인사이트 | 조건 |
|---|----------|------|
| 1 | 🔴 **비효율 구역** | 방문 ≥5회 & 전환율 < 전체 평균 × 60% |
| 2 | 🛒 **교차구매 기회** | 같은 고객이 함께 방문한 구역 쌍 최다 (≥3건) |
| 3 | 👑 **VIP 고객군** | 가담가(객단가) 최고 + 매출 점유율 |
| 4 | ⏰ **피크 시간대** | 매출 최고/최저 시간대 비율 |
| 5 | 📈 **매출 추이** | 전반 7일 vs 후반 7일 일평균 성장률 |

<br>

## 📂 주요 데이터 파일

| 파일 | 내용 | 건수 |
|------|------|:----:|
| `ice_batch_purchases.csv` | 실측 매장 7일 구매 로그 | 495건 |
| `ice_zone_kpi.csv` | 실측 구역별 KPI | 9구역 |
| `ice_class_kpi.csv` | 실측 고객군별 KPI | 4분류 |
| `model/data/batch_purchases_v2.csv` | 4개월 시뮬레이션 결과 (학습 데이터) | 84,406건 |
| `hour_kpi.csv` | 시간대별 집계 (120일 × 12시간) | 1,440행 |
| `zones.json` | 구역별 카메라 폴리곤 좌표 | 9구역 × 3카메라 |
| `zone_images/` | 구역 설정 캡처 이미지 | 3장 (ch1·ch3·ch6) |

