# CCTV 기반 무인매장 디지털 트윈 인사이트 대시보드

CCTV 영상 분석을 통해 무인매장 고객 행동 데이터를 수집하고, 시뮬레이션 및 예측 모델로 매장 운영 인사이트를 제공하는 시스템입니다.

## 프로젝트 구조

```
├── simul/          # 시뮬레이션 + React 대시보드 (Vite)
│   ├── src/        # React 컴포넌트, 예측 모델, Web Worker
│   └── server/     # FastAPI 백엔드 서버
├── retail/         # YOLO 기반 고객 행동 분석 (포즈 추정, 히트맵)
├── model/          # Crossformer 시계열 예측 모델
├── kpi_crossformer.py      # Crossformer KPI 예측 실행
├── rnn(lstm)_times.py      # LSTM 시계열 예측 실행
└── *.csv           # KPI 데이터
```

## 실행 방법

### 대시보드 (simul)
```bash
cd simul
npm install
npm run dev
```

### 백엔드 서버
```bash
cd simul/server
pip install -r requirements.txt
python main.py
```

### YOLO 행동 분석 (retail)
```bash
cd retail
pip install -r requirements.txt
python main.py
```

### 예측 모델 (model)
```bash
pip install -r simul/server/requirements.txt
python train.py
```

## 팀원

- 안은정
- 이준구
- 박동혁
