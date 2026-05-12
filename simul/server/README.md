# KPI 예측 서버

Transformer + Crossformer 체크포인트를 로드해 시뮬레이터 데이터로부터
다음날 KPI 5종(#1 구역별 전환율, #3 체류시간, #6 시간대별 전환율, #7 전환효율, #8 비효율구역)을
예측하는 FastAPI 서버입니다.

## 사전 조건

- Python 3.10+
- `model/checkpoints/best.pt` (Transformer 체크포인트)
- `checkpoints_crossformer/best_kpi.pt` (Crossformer 체크포인트)

## 설치

```powershell
# 캡스톤 프로젝트 루트에서 실행
cd "C:\Users\dlwns\2026\1학기\캡스톤"
pip install -r simul/server/requirements.txt
```

## 실행

```powershell
# 프로젝트 루트에서 실행 (경로 중요)
cd "C:\Users\dlwns\2026\1학기\캡스톤"
uvicorn simul.server.main:app --port 8000
```

서버가 시작되면 터미널에 다음이 출력됩니다:
```
[KPI 서버] 모델 로딩 중...
  [Transformer] 7구역 × 12시간대  loaded
  [Crossformer] 7×12×4 tokens  loaded
[KPI 서버] 준비 완료 — http://localhost:8000
```

## API

- `GET  /health` — 서버 상태 확인
- `POST /api/predict-kpi` — KPI 예측 (request/response 형식은 main.py 참고)

## 프론트엔드 연동

simul 대시보드(Vite, :5173)는 `vite.config.js`의 proxy 설정으로
`/api` 요청을 이 서버(`:8000`)로 자동 포워딩합니다.
시뮬레이션 후 "AI 예측" 탭에서 결과를 확인하세요.
