# 예측 모델 기술 문서

**프로젝트**: 매장 CCTV 기반 실시간 고객 속성 분류 및 히트맵 기반 상품 배치 최적화  
**작성일**: 2026-04-22  
**대상 파일**: `src/predictionModels.js`, `src/models/lstmForecaster.js`, `src/models/dqnOptimizer.js`

---

## 목차

1. [공통 피처 벡터](#1-공통-피처-벡터-25차원)
2. [Logistic Regression — 구매 전환율 예측](#2-logistic-regression--구매-전환율-예측)
3. [Linear Regression — 매대별 예상 매출 예측](#3-linear-regression--매대별-예상-매출-예측)
4. [UCB1 Multi-Armed Bandit — 신상품 배치 구역 추천](#4-ucb1-multi-armed-bandit--신상품-배치-구역-추천)
5. [Hourly Demand — 시간대별 수요 예측](#5-hourly-demand--시간대별-수요-예측)
6. [고객 가치 분석 (CLV + 세그먼트)](#6-고객-가치-분석-clv--세그먼트)
7. [교차 구매 확률 분석 (Cross-Sell)](#7-교차-구매-확률-분석-cross-sell)
8. [LSTM — 시계열 매출 예측](#8-lstm--시계열-매출-예측)
9. [DQN — 매대 배치 순서 최적화](#9-dqn--매대-배치-순서-최적화)
10. [모델 검증 파이프라인](#10-모델-검증-파이프라인)

---

## 1. 공통 피처 벡터 (25차원)

모델 1~3은 동일한 25차원 피처를 공유한다.  
소스: `buildFeature(classType, zoneId, dwellTime, hour, maxDwell, simHours)`

| 인덱스 | 피처 | 설명 |
|--------|------|------|
| 0~3 | 고객 분류 one-hot | adult_male / adult_female / minor_male / minor_female |
| 4~11 | 구역 one-hot | snack / beverage / daily / cosmetic / stationery / frozen / premium / bakery |
| 12 | 체류시간 정규화 | `dwellTime / maxDwell`, 0~1 |
| 13 | 시간대 정규화 | `hour / (operatingHours - 1)`, 0~1 |
| 14 | 선호 구역 여부 | 해당 고객군의 선호 구역이면 1, 아니면 0 |
| 15 | 선호×체류 교호 | feat[14] × feat[12] — 선호 구역에서 오래 머물수록 구매 신호 증폭 |
| 16 | 성인 여부 | adult_* → 1 |
| 17 | 미성년 여부 | minor_* → 1 |
| 18 | 체류시간 log 변환 | `log(1 + dwell) / log(1 + maxDwell)` — 비선형 포착 |
| 19 | 체류시간 제곱 | feat[12]² — 고체류 구간 가중 |
| 20 | 구역 평균 가격 정규화 | `zoneAvgPrice / 26000` (premium 기준) |
| 21 | 음료 구역 플래그 | beverage는 전 클래스 선호 — 독립 신호 |
| 22 | 미성년×과자 교호 | minor + snack → 1 — 가장 강한 선호 쌍 |
| 23 | 성년여성×화장품 교호 | adult_female + cosmetic → 1 |
| 24 | 시간대×선호 교호 | feat[13] × feat[14] — 늦은 시간 선호 구역 방문 = 목적 구매 신호 |

**설계 의도**: feat[14]~[24]의 교호작용·비선형 항은 단순 선형 모델이 잡지 못하는 "선호 구역에서 오래 머문 성년 여성은 화장품 구매율이 급등한다"류의 패턴을 피처 엔지니어링으로 사전에 내재화한다.

---

## 2. Logistic Regression — 구매 전환율 예측

**파일**: `src/predictionModels.js` → `trainLogisticRegression()`  
**목적**: 고객 분류·체류시간·시간대 조건에서 특정 매대에서 구매할 확률 P(구매 | 조건) 예측  
**출력**: 매대별 구매 확률 0~1 (대시보드 바차트)

### 알고리즘

- **모델**: Sigmoid를 사용하는 이진 분류 로지스틱 회귀  
  `P = σ(w·x + b) = 1 / (1 + e^{-z})`
- **학습 방법**: Mini-batch Gradient Descent (배치 크기 64)  
  `w ← w - α × (∂L/∂w + λ·w)`
- **정규화**: L2 Regularization (과적합 방지)  
- **학습률 스케줄**: 50 epoch마다 15% 감소 (`lr × 0.85^(epoch/50)`)

### 훈련 데이터 구성

구매 데이터만으로는 양성 편중이 발생하므로, 비구매 방문 음성 샘플을 함께 생성한다.

- **양성 샘플**: 실제 구매 기록 (최대 3,000건)
- **음성 샘플**: 구역×클래스별 방문했지만 구매하지 않은 추정 케이스
- **양/음 비율**: 매장 유형별로 다르게 설정 (편의점 1:5, 슈퍼마켓 1:3)

### 매장 유형별 하이퍼파라미터

| 매장 유형 | epochs | 학습률 | L2 λ | 클래스 가중치 | 음성 비율 |
|-----------|--------|--------|------|-------------|---------|
| convenience | 500 | 0.08 | 0.0003 | 적용 (√neg/pos) | 1:5 |
| supermarket | 300 | 0.10 | 0.001  | 미적용 | 1:3 |
| mall | 300 | 0.10 | 0.001  | 미적용 | 1:3 |

편의점은 구매율이 높아 양성 편중이 심하므로, 클래스 가중치(`sqrt(neg/pos)`)와 더 많은 학습 반복으로 보정한다.

### 평가 지표

- **Accuracy**: 임계값 0.5 기준 정분류율
- **Baseline Accuracy**: 항상 다수 클래스로 예측했을 때의 정확도 (`max(pos, neg) / total`)
- **AUC-ROC**: Mann-Whitney U 방식. 샘플 쌍이 50,000 초과 시 무작위 샘플링으로 추정

---

## 3. Linear Regression — 매대별 예상 매출 예측

**파일**: `src/predictionModels.js` → `trainRevenuePredictor()`  
**목적**: 동일 조건(고객 분류·체류시간·시간대)에서 매대별 기대 매출액 예측  
**출력**: 매대별 예상 매출 (원 단위, 대시보드 바차트)

### 알고리즘

- **모델**: 선형 회귀 `ŷ = w·x + b`
- **학습 방법**: Mini-batch GD (배치 64, epochs 250, lr 0.05, λ 0.001)
- **정규화**: L2 Regularization

### 학습 타깃 설계

개별 아이템 가격은 노이즈가 크므로, 예측 대상을 **구역 평균 가격**으로 고정한다.

```
타깃 = ZONE_AVG_PRICES[zoneId] / 26000   (0~1 정규화)
출력 = ŷ × 26000   (원 단위로 역정규화)
```

구역 평균 가격 참고치: 스낵 1,730원 / 음료 1,720원 / 화장품 15,800원 / 프리미엄 26,000원

### 평가 지표

- **R²** (결정계수): 모델이 분산의 몇 %를 설명하는지 (0~1, 높을수록 좋음)
- `R² = 1 - SS_res / SS_tot`

---

## 4. UCB1 Multi-Armed Bandit — 신상품 배치 구역 추천

**파일**: `src/predictionModels.js` → `runUCB1Bandit()`  
**목적**: 전환율 탐색-활용 균형을 통해 신상품 진열에 최적인 매대 구역 추천  
**출력**: 추천 구역 + 신뢰도 + 수렴 과정

### 알고리즘: UCB1 (Upper Confidence Bound 1)

각 매대를 하나의 arm으로 보고, 500번의 가상 탐색으로 최적 arm을 찾는다.

```
UCB1 점수 = Q(arm) + √(2 × ln(총_탐색수) / arm_탐색수)
```

- `Q(arm)`: 해당 구역의 평균 보상 (실제 구매 전환율 기반 Bernoulli 샘플링)
- 우측 항: 탐색 보너스 — 적게 선택된 arm을 강제로 탐색시키는 역할

### 동작 과정

1. 각 arm(매대)을 1회씩 초기 탐색
2. 이후 500 trial 동안 UCB1 점수 최대 arm을 선택 → Bernoulli 보상 수집 → Q 업데이트
3. 최종 Q값 최고 arm을 신상품 배치 구역으로 추천

### 신뢰도 계산

```
신뢰도 = min(100, round((best.n / maxPulls) × (best.Q / bestTrueRate) × 100))
```

충분히 탐색되고 전환율이 높을수록 신뢰도가 높아진다.

### Logistic Regression과의 차이

| 구분 | Logistic Regression | UCB1 Bandit |
|------|---------------------|-------------|
| 질문 | "이 고객이 이 매대에서 살 확률?" | "신상품을 어느 매대에 놓아야 최적인가?" |
| 입력 | 고객 분류, 체류시간, 시간대 | 매대별 전환율 통계 |
| 출력 | 매대별 구매 확률 | 최적 배치 구역 1개 + 신뢰도 |
| 방법 | 지도학습 | 탐색-활용 균형 (강화학습 계열) |

---

## 5. Hourly Demand — 시간대별 수요 예측

**파일**: `src/predictionModels.js` → `predictHourlyDemand()`  
**목적**: 다음 영업일 시간대별 예상 방문자 수 및 매출 예측  
**출력**: 시간대별 예측값 + 트렌드 방향 + 신뢰도

### 알고리즘: 가중 이동평균 + 트렌드 보정

머신러닝 없이 통계적 방법만 사용하는 경량 모델이다.

**1단계: 전반/후반 트렌드 계산**
```
overallTrend = 후반 절반 평균 매출 / 전반 절반 평균 매출
trendFactor  = clamp(overallTrend, 0.7, 1.3)   (과도한 보정 방지)
```

**2단계: 최근 7일 이동평균**
```
recentAvg = mean(지난 7일 해당 시간대 방문자 수)
```

**3단계: 최종 예측**
```
predictedVisitors = round(recentAvg × trendFactor)
predictedRevenue  = round(recentAvgRevenue × trendFactor)
```

**트렌드 판정**: `overallTrend > 1.05` → 상승, `< 0.95` → 하락, 그 외 → 안정  
**신뢰도**: `min(100, round(데이터_수 / 전체_일수 × 100))`

---

## 6. 고객 가치 분석 (CLV + 세그먼트)

**파일**: `src/predictionModels.js` → `computeCustomerValueAnalysis()`  
**목적**: 고객 분류별 생애 가치(LTV) 추정 및 VIP/loyal/casual/at-risk 세그먼트 분류  
**출력**: 클래스별 LTV, 세그먼트, 선호 매대 Top 3, 피크 시간대

### LTV 산출 공식

```
avgBasket     = 총 매출 / 구매 고객 수   (평균 구매액)
convRate      = 구매 고객 수 / 방문 고객 수   (전환율)
avgZones      = 구매자 1인당 평균 구매 매대 수

LTV = avgBasket × convRate × max(avgZones, 1)
```

단일 방문 기반 LTV로, 재방문 기간은 시뮬레이션 범위 밖이다.

### 세그먼트 분류 (4분위 기반)

| 순위 | 세그먼트 | 기준 |
|------|----------|------|
| 상위 25% | VIP | LTV 최상위 |
| 25~50% | loyal | LTV 상위 |
| 50~75% | casual | LTV 하위 |
| 하위 25% | at-risk | LTV 최하위 |

---

## 7. 교차 구매 확률 분석 (Cross-Sell)

**파일**: `src/predictionModels.js` → `predictCrossSell()`  
**목적**: 매대 간 동시 구매 패턴 분석 — "A를 산 고객이 B도 살 확률"  
**출력**: P(B|A) 조건부 확률 매트릭스 + 상위 5쌍 + 자동 인사이트

### 알고리즘: 조건부 확률 직접 계산

```
P(B | A) = (A와 B를 동시에 구매한 고객 수) / (A를 구매한 고객 수)
```

머신러닝이 아닌 빈도 기반 확률 계산이다.

### 활용

- **매대 배치**: P(B|A)가 높은 쌍은 인접 배치 → 동선 상 자연스러운 교차 구매 유도
- **번들 프로모션**: 높은 쌍을 묶음 할인 대상으로 설정
- **신규 동선 설계**: 낮은 쌍에 연결 쿠폰 → 미발굴 교차 구매 유도

---

## 8. LSTM — 시계열 매출 예측

**파일**: `src/models/lstmForecaster.js` → `trainLSTMForecaster()`  
**목적**: 과거 일별 매출 패턴을 학습해 **미래 14일 매출 예측**  
**런타임**: TensorFlow.js (브라우저 WebGL 가속)  
**출력**: 14일 예측값 + 90% 신뢰구간 + 학습 곡선 + MAPE/RMSE/R²

### 입력 피처 (4차원)

| 피처 | 값 | 전처리 |
|------|-----|--------|
| revenue | 일별 총 매출 | Min-Max 정규화 |
| visitors | 일별 방문자 수 | Min-Max 정규화 |
| dayOfWeek | 요일 (0~6) | `/6` 정규화 |
| convRate | 일별 구매 전환율 | 그대로 사용 (0~1) |

### 모델 구조

```
Input: (7일 슬라이딩 윈도우, 4 피처)
  → LSTM(32 units, return_sequences=True)
  → LSTM(16 units)
  → Dense(8, relu)
  → Dense(1)           ← 다음 날 정규화된 매출 출력

손실 함수: Mean Squared Error
최적화기: Adam (lr=0.01)
학습: 50 epochs, batch=8, validation_split=0.15
```

### 예측 방식: 롤링(Rolling) 예측

14일을 한 번에 예측하는 것이 아니라, 1일씩 예측하고 그 결과를 다음 입력에 추가한다.

```
[D-7 ~ D-1] → 모델 → D+1 예측
[D-6 ~ D]   → 모델 → D+2 예측   (D는 D+1 예측값으로 대체)
...
```

### 불확실성 구간 (90% CI)

테스트셋 예측 잔차의 표준편차를 사용한다.

```
resStd  = std(actual - predicted)  on test set
lower   = max(0, revenue - 1.64 × resStd)
upper   =        revenue + 1.64 × resStd
```

1.64는 정규분포 90% 신뢰구간 z-값이다.

### Baseline 비교

Baseline = **선형 트렌드 연장** (마지막 7일 데이터로 선형회귀 → 미래 예측)

```python
slope = (n × Σxy - Σx × Σy) / (n × Σx² - (Σx)²)
ŷ(t) = intercept + slope × t
```

LSTM의 MAPE·RMSE가 Baseline보다 낮아야 유의미한 모델이다.

### 다른 예측 모델과의 차이

| 구분 | Hourly Demand | LSTM |
|------|---------------|------|
| 예측 대상 | 오늘~내일 시간대별 방문자 | 14일 뒤까지 일별 매출 |
| 시간 범위 | 하루 이내 | 2주 앞 |
| 계절성 학습 | 불가 (이동평균) | 가능 (시퀀스 메모리) |
| 복잡도 | 경량 통계 | 신경망 (TF.js 필요) |

---

## 9. DQN — 매대 배치 순서 최적화

**파일**: `src/models/dqnOptimizer.js` → `trainDQNOptimizer()`  
**목적**: 8개 매대를 8개 슬롯에 배치하는 **순열을 강화학습으로 최적화**  
**런타임**: TensorFlow.js  
**출력**: 추천 배치 순서 + 현재 대비 개선율 + 수렴 곡선

### 문제 정의

- **상태 (State)**: 현재 배치 순열의 one-hot 인코딩 (64차원 = 8 슬롯 × 8 구역)
- **액션 (Action)**: 두 슬롯을 서로 교환(swap), C(8,2) = **28가지**
- **보상 (Reward)**: 배치 완료 후 총 기대 매출 점수

```
Reward = Σ(slot 0~7) { accessWeight[slot] × expectedRevenue[zone at slot] }
```

`accessWeight`: 입구(400, 584)에서 각 슬롯까지의 역거리를 정규화한 값.  
입구에 가까운 슬롯일수록 가중치가 높으므로, 매출이 높은 구역을 접근성 좋은 위치에 배치하면 보상이 커진다.

### 네트워크 구조 (Q-Network)

```
Input: 64차원 one-hot 상태
  → Dense(64, relu)
  → Dense(64, relu)
  → Dense(28)     ← 28개 액션 각각의 Q값 출력

손실: Huber Loss (이상치에 강건)
최적화: Adam (lr=0.001)
```

### DQN 학습 설정

| 하이퍼파라미터 | 값 | 역할 |
|---------------|-----|------|
| episodes | 300 | 총 에피소드 수 |
| steps_per_ep | 20 | 에피소드당 swap 시도 수 |
| ε (epsilon) | 1.0 → 0.05 (선형 감소) | 탐색→활용 전환 비율 |
| γ (gamma) | 0.95 | 미래 보상 할인율 |
| Replay Buffer | 1,000 | 경험 재사용 버퍼 크기 |
| Batch | 32 | 학습 미니배치 크기 |
| Target Update | 매 5 episodes | Target Network 동기화 주기 |

### UCB1 Bandit과의 차이

| 구분 | UCB1 Bandit | DQN |
|------|-------------|-----|
| 결정 단위 | 구역 1개 선택 | 8개 구역 전체 순열 |
| 질문 | "신상품을 어디에 놓을까?" | "모든 매대를 어떻게 재배치할까?" |
| 입력 | 구역별 전환율 | 배치 상태 + 일별 매출 데이터 |
| 탐색 공간 | 8개 구역 | 8! = 40,320가지 순열 |
| 방법 | Bandit (1단계 결정) | 강화학습 (순차적 swap) |

---

## 10. 모델 검증 파이프라인

**파일**: `src/validationUtils.js`  
**목적**: 시뮬레이션 데이터를 Train/Test로 분할하여 각 모델의 실제 예측 성능을 정량 평가

### 데이터 분할

```
전체 데이터 (N일) → Train (앞 75%) | Test (뒤 25%)
예: 120일 → Train 90일 | Test 30일
```

### 각 모델의 검증 지표

| 모델 | 주요 지표 | 의미 |
|------|-----------|------|
| Logistic Regression | Accuracy, AUC, F1, Confusion Matrix | Train 데이터로 학습 → Test 데이터 예측 정확도 |
| Linear Regression | R², MAE, MAPE, RMSE | 예측 매출과 실제 구역 평균가격 간 오차 |
| UCB1 Bandit | 순위(Rank), Regret, 상관계수 | Train 추천 구역이 Test 기간 실제 전환율 1위인가? |
| Hourly Demand | MAPE (방문자, 매출) | Train 이동평균 예측 vs Test 실제 시간대 데이터 |
| 고객 가치 분석 | 세그먼트 안정성, LTV 순위 상관, 선호매대 일치율 | Train/Test 구간 간 분류 일관성 |
| 교차 구매 | 매트릭스 상관계수, Top-5 쌍 일치율 | Train/Test 구간 간 구매 패턴 안정성 |
| LSTM | MAPE, RMSE, R² vs Baseline | Test 기간 실제 매출 vs LSTM 예측 (별도 버튼으로 실행) |

### Baseline 정의 (모델별)

| 모델 | Baseline |
|------|----------|
| Logistic Regression | 항상 다수 클래스 예측 시 정확도 `max(pos, neg) / total` |
| LSTM | 직전 7일 선형 트렌드를 연장한 예측값 |

Baseline보다 높아야 해당 모델이 의미 있는 예측 능력을 가진다고 볼 수 있다.

---

## 모델 관계도

```
배치 시뮬레이션 데이터 (구매 기록, 구역 통계, 일별 통계)
        │
        ├─── [통계 기반] ──────────────────────────────────────────────────┐
        │     ├── Hourly Demand       → 내일 시간대별 방문자/매출 예측      │
        │     ├── CLV + 세그먼트      → 고객군별 가치 분류                  │
        │     └── Cross-Sell          → 매대 간 교차 구매 확률              │
        │                                                                  │
        ├─── [지도학습] ──────────────────────────────────────────────────┤
        │     ├── Logistic Regression → 특정 조건의 구매 전환 확률         │
        │     └── Linear Regression  → 특정 조건의 기대 매출액             │
        │                                                                  │
        ├─── [강화학습 계열] ─────────────────────────────────────────────┤
        │     ├── UCB1 Bandit         → 신상품 최적 배치 구역 (1개 추천)   │
        │     └── DQN                 → 8개 매대 전체 배치 순열 최적화      │
        │                                                                  │
        └─── [딥러닝] ────────────────────────────────────────────────────┘
              └── LSTM                → 미래 14일 일별 매출 시계열 예측
```
