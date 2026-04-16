// ── predictionModels.js ─────────────────────────────────────────────────────
// 순수 JS 기반 AI 예측 모델: Logistic Regression + UCB1 Multi-Armed Bandit
// React 의존성 없음

const CLASS_KEYS = ['adult_male', 'adult_female', 'minor_male', 'minor_female'];
const ZONE_KEYS  = ['snack', 'beverage', 'daily', 'cosmetic', 'stationery', 'frozen', 'premium', 'bakery'];

// ── 유틸리티 ────────────────────────────────────────────────────────────────

function sigmoid(z) {
  if (z > 500) return 1;
  if (z < -500) return 0;
  return 1 / (1 + Math.exp(-z));
}

// 시드 기반 간이 난수 생성기 (재현성)
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// Fisher-Yates 셔플
function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// Feature 벡터 생성 (14차원)
export function buildFeature(classType, zoneId, dwellTime, hour, maxDwell, simHours) {
  const feat = new Float64Array(14);
  // one-hot: 고객분류 (0~3)
  const clsIdx = CLASS_KEYS.indexOf(classType);
  if (clsIdx >= 0) feat[clsIdx] = 1;
  // one-hot: 구역 (4~11)
  const zIdx = ZONE_KEYS.indexOf(zoneId);
  if (zIdx >= 0) feat[4 + zIdx] = 1;
  // 정규화된 체류시간 (12)
  feat[12] = maxDwell > 0 ? Math.min(dwellTime / maxDwell, 1) : 0;
  // 정규화된 시간대 (13) — 동적 영업시간 기준
  const maxHour = Math.max(1, (simHours || 8) - 1);
  feat[13] = Math.min(hour / maxHour, 1);
  return feat;
}

// ── Logistic Regression ─────────────────────────────────────────────────────

/**
 * 훈련 데이터 생성: 양성(구매) + 음성(방문만) 샘플
 */
export function buildTrainingSamples(data) {
  const rng = mulberry32(42);
  const { purchases, zoneStats } = data;
  const simHours = data.operatingHours || 8;

  // 양성 샘플 (최대 3000개)
  let positives = purchases.map(p => ({
    classType: p.classType,
    zoneId: p.zoneId,
    dwellTime: p.dwellTime,
    hour: p.hour,
    label: 1,
  }));
  if (positives.length > 3000) {
    shuffle(positives, rng);
    positives = positives.slice(0, 3000);
  }

  // 음성 샘플 생성 — 구역×클래스별 비구매 방문 횟수 기반
  let negatives = [];
  const posCount = Math.max(positives.length, 1);
  for (const [zId, zs] of Object.entries(zoneStats)) {
    if (!ZONE_KEYS.includes(zId)) continue;
    for (const cls of CLASS_KEYS) {
      const visitCount = (zs.visitorsByClass && zs.visitorsByClass[cls]) || 0;
      const buyCount   = (zs.buyersByClass && zs.buyersByClass[cls]) || 0;
      // 비구매 방문 수, 구역×클래스당 최대 양성의 1배로 cap (과다 방지)
      let negCount = Math.max(0, visitCount - buyCount);
      negCount = Math.min(negCount, Math.ceil(posCount / (ZONE_KEYS.length * CLASS_KEYS.length) * 3));
      const isMinor = cls.startsWith('minor');
      for (let i = 0; i < negCount; i++) {
        negatives.push({
          classType: cls,
          zoneId: zId,
          dwellTime: isMinor
            ? 0.5 + rng() * 2.5
            : 1.0 + rng() * 5.0,
          hour: Math.floor(rng() * simHours),  // 동적 영업시간 범위
          label: 0,
        });
      }
    }
  }

  // 총 음성 샘플: 양성의 2~3배 유지 (불균형 방지)
  const maxNeg = posCount * 3;
  if (negatives.length > maxNeg) {
    shuffle(negatives, rng);
    negatives = negatives.slice(0, maxNeg);
  }

  return { samples: [...positives, ...negatives], simHours };
}

/**
 * Logistic Regression 학습 (Mini-batch GD + L2 정규화)
 */
export function trainLogisticRegression(data) {
  const { samples, simHours } = buildTrainingSamples(data);
  if (samples.length === 0) {
    return { weights: new Float64Array(14), bias: 0, maxDwell: 10, simHours: simHours || 8, accuracy: 0, auc: 0 };
  }

  const rng = mulberry32(123);

  // 최대 체류시간 산출
  let maxDwell = 0;
  for (const s of samples) if (s.dwellTime > maxDwell) maxDwell = s.dwellTime;
  if (maxDwell === 0) maxDwell = 10;

  // Feature 행렬 구축
  const N = samples.length;
  const D = 14;
  const X = new Float64Array(N * D);
  const y = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const feat = buildFeature(samples[i].classType, samples[i].zoneId, samples[i].dwellTime, samples[i].hour, maxDwell, simHours);
    X.set(feat, i * D);
    y[i] = samples[i].label;
  }

  // 가중치 초기화
  const w = new Float64Array(D);
  for (let i = 0; i < D; i++) w[i] = (rng() - 0.5) * 0.01;
  let b = 0;

  // 하이퍼파라미터
  const epochs = 200;
  const lr = 0.1;
  const batchSize = 64;
  const lambda = 0.001;

  // 인덱스 배열
  const indices = Array.from({ length: N }, (_, i) => i);

  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffle(indices, rng);

    for (let start = 0; start < N; start += batchSize) {
      const end = Math.min(start + batchSize, N);
      const m = end - start;

      // 그래디언트 계산
      const gw = new Float64Array(D);
      let gb = 0;

      for (let k = start; k < end; k++) {
        const idx = indices[k];
        // dot product
        let z = b;
        for (let j = 0; j < D; j++) z += w[j] * X[idx * D + j];
        const pred = sigmoid(z);
        const err = pred - y[idx];
        for (let j = 0; j < D; j++) gw[j] += err * X[idx * D + j];
        gb += err;
      }

      // 가중치 업데이트 (L2 정규화 포함)
      for (let j = 0; j < D; j++) {
        w[j] -= lr * (gw[j] / m + lambda * w[j]);
      }
      b -= lr * gb / m;
    }
  }

  // 정확도 및 AUC 계산
  const preds = new Float64Array(N);
  let correct = 0;
  for (let i = 0; i < N; i++) {
    let z = b;
    for (let j = 0; j < D; j++) z += w[j] * X[i * D + j];
    preds[i] = sigmoid(z);
    const predicted = preds[i] >= 0.5 ? 1 : 0;
    if (predicted === y[i]) correct++;
  }
  const accuracy = correct / N;

  // 간이 AUC (Mann-Whitney U 방식)
  const pos = [], neg = [];
  for (let i = 0; i < N; i++) {
    if (y[i] === 1) pos.push(preds[i]);
    else neg.push(preds[i]);
  }
  let auc = 0.5;
  if (pos.length > 0 && neg.length > 0) {
    let concordant = 0;
    // 전체 쌍 비교는 너무 비쌀 수 있으므로 샘플링
    const maxPairs = 50000;
    const totalPairs = pos.length * neg.length;
    if (totalPairs <= maxPairs) {
      for (let i = 0; i < pos.length; i++) {
        for (let j = 0; j < neg.length; j++) {
          if (pos[i] > neg[j]) concordant++;
          else if (pos[i] === neg[j]) concordant += 0.5;
        }
      }
      auc = concordant / totalPairs;
    } else {
      // 샘플링 기반 AUC 추정
      const sampleRng = mulberry32(777);
      for (let k = 0; k < maxPairs; k++) {
        const pi = pos[Math.floor(sampleRng() * pos.length)];
        const ni = neg[Math.floor(sampleRng() * neg.length)];
        if (pi > ni) concordant++;
        else if (pi === ni) concordant += 0.5;
      }
      auc = concordant / maxPairs;
    }
  }

  return { weights: w, bias: b, maxDwell, simHours, accuracy, auc };
}

/**
 * 단일 예측: P(구매 | classType, zoneId, dwellTime, hour)
 */
export function predictProba(model, classType, zoneId, dwellTime, hour) {
  const feat = buildFeature(classType, zoneId, dwellTime, hour, model.maxDwell, model.simHours);
  let z = model.bias;
  for (let j = 0; j < 14; j++) z += model.weights[j] * feat[j];
  return sigmoid(z);
}

/**
 * 전체 구역 예측: 활성 구역에 대한 구매 확률
 * @param {string[]} activeZoneIds - 현재 매장에서 사용 중인 구역 ID 목록 (없으면 전체)
 */
export function predictAllZones(model, classType, dwellTime, hour, activeZoneIds) {
  const ZONE_LABELS = {
    snack: '과자/스낵', beverage: '음료', daily: '생활용품',
    cosmetic: '화장품', stationery: '문구', frozen: '냉동식품',
    premium: '프리미엄', bakery: '베이커리',
  };
  const targetZones = activeZoneIds || ZONE_KEYS;
  return targetZones
    .filter(zId => ZONE_KEYS.includes(zId))
    .map(zoneId => ({
      zoneId,
      label: ZONE_LABELS[zoneId] || zoneId,
      probability: predictProba(model, classType, zoneId, dwellTime, hour),
    }))
    .sort((a, b) => b.probability - a.probability);
}

// ── UCB1 Multi-Armed Bandit ─────────────────────────────────────────────────

/**
 * UCB1 알고리즘으로 신상품 최적 배치 구역 추천
 */
export function runUCB1Bandit(zoneStats, trials = 500) {
  const rng = mulberry32(999);

  const ZONE_LABELS = {
    snack: '과자/스낵', beverage: '음료', daily: '생활용품',
    cosmetic: '화장품', stationery: '문구', frozen: '냉동식품',
    premium: '프리미엄', bakery: '베이커리',
  };

  // 각 arm의 실제 전환율 계산 — 활성 구역(방문자 > 0)만 포함
  const arms = ZONE_KEYS
    .filter(zoneId => zoneStats[zoneId] && zoneStats[zoneId].visitors > 0)
    .map(zoneId => {
    const zs = zoneStats[zoneId];
    const trueConvRate = zs.buyers / zs.visitors;
    return {
      zoneId,
      label: zs.label || ZONE_LABELS[zoneId] || zoneId,
      trueConvRate,
      Q: 0,        // 평균 보상
      n: 0,        // 선택 횟수
      totalReward: 0,
    };
  });

  // 최적 arm의 전환율 (regret 계산용)
  const bestTrueRate = Math.max(...arms.map(a => a.trueConvRate));

  const explorationHistory = [];
  const convergenceData = [];
  let cumulativeRegret = 0;

  // 초기: 각 arm을 1회씩 탐색
  const K = arms.length;
  for (let i = 0; i < K && i < trials; i++) {
    const arm = arms[i];
    const reward = rng() < arm.trueConvRate ? 1 : 0;
    arm.n = 1;
    arm.totalReward = reward;
    arm.Q = reward;
    cumulativeRegret += bestTrueRate - arm.trueConvRate;
    explorationHistory.push({
      trial: i + 1,
      chosenZoneId: arm.zoneId,
      reward,
      cumulativeRegret: +cumulativeRegret.toFixed(4),
    });
  }

  // UCB1 메인 루프
  for (let t = K; t < trials; t++) {
    const totalPulls = t; // t번째 시점의 총 pull 수

    // UCB1 점수 계산 및 최대 arm 선택
    let bestIdx = 0;
    let bestUCB = -Infinity;
    for (let i = 0; i < K; i++) {
      const arm = arms[i];
      const exploration = Math.sqrt(2 * Math.log(totalPulls) / arm.n);
      const ucb = arm.Q + exploration;
      if (ucb > bestUCB) {
        bestUCB = ucb;
        bestIdx = i;
      }
    }

    // 선택된 arm에서 보상 샘플링 (Bernoulli)
    const chosen = arms[bestIdx];
    const reward = rng() < chosen.trueConvRate ? 1 : 0;
    chosen.totalReward += reward;
    chosen.n += 1;
    chosen.Q = chosen.totalReward / chosen.n;
    cumulativeRegret += bestTrueRate - chosen.trueConvRate;

    explorationHistory.push({
      trial: t + 1,
      chosenZoneId: chosen.zoneId,
      reward,
      cumulativeRegret: +cumulativeRegret.toFixed(4),
    });

    // 수렴 데이터: 50 trial 간격
    if ((t + 1) % 50 === 0 || t === trials - 1) {
      const bestArmQ = Math.max(...arms.map(a => a.Q));
      convergenceData.push({ trial: t + 1, bestArmQ: +bestArmQ.toFixed(4) });
    }
  }

  // 최종 UCB 점수 계산
  const totalPulls = trials;
  for (const arm of arms) {
    arm.ucbBonus = arm.n > 0 ? Math.sqrt(2 * Math.log(totalPulls) / arm.n) : Infinity;
    arm.ucbScore = arm.Q + arm.ucbBonus;
  }

  // 추천: Q값이 가장 높은 arm
  const sorted = [...arms].sort((a, b) => b.Q - a.Q);
  const best = sorted[0];

  // 신뢰도: 최다 선택 비율 기반 (0~100)
  const maxPulls = Math.max(...arms.map(a => a.n));
  const confidence = Math.min(100, Math.round((best.n / maxPulls) * (best.Q / (bestTrueRate || 1)) * 100));

  return {
    arms: arms.map(a => ({
      zoneId: a.zoneId,
      label: a.label,
      Q: +a.Q.toFixed(4),
      n: a.n,
      ucbBonus: +a.ucbBonus.toFixed(4),
      ucbScore: +a.ucbScore.toFixed(4),
      trueConvRate: +a.trueConvRate.toFixed(4),
    })),
    recommendation: {
      zoneId: best.zoneId,
      label: best.label,
      confidence: Math.max(10, confidence), // 최소 10%
    },
    explorationHistory,
    convergenceData,
  };
}

// ── 매대별 예상 매출액 예측 (Linear Regression) ─────────────────────────────

/**
 * 매대별 예상 매출액을 예측하는 Linear Regression 모델
 * @param {object} data - 배치 시뮬레이션 결과
 * @returns {{ model, predictForZone: Function }}
 */
export function trainRevenuePredictor(data) {
  const ZONE_LABELS = {
    snack: '과자/스낵', beverage: '음료', daily: '생활용품',
    cosmetic: '화장품', stationery: '문구', frozen: '냉동식품',
    premium: '프리미엄', bakery: '베이커리',
  };
  const rng = mulberry32(55);
  const { purchases } = data;
  const simHours = data.operatingHours || 8;

  if (purchases.length === 0) {
    return {
      weights: new Float64Array(14), bias: 0, maxDwell: 10, maxPrice: 1,
      simHours, r2: 0,
      predictForZone: () => [],
    };
  }

  // 샘플 제한 (최대 3000건)
  let samples = purchases.slice();
  if (samples.length > 3000) {
    shuffle(samples, rng);
    samples = samples.slice(0, 3000);
  }

  let maxDwell = 0, maxPrice = 0;
  for (const s of samples) {
    if (s.dwellTime > maxDwell) maxDwell = s.dwellTime;
    if (s.price > maxPrice) maxPrice = s.price;
  }
  if (maxDwell === 0) maxDwell = 10;
  if (maxPrice === 0) maxPrice = 1;

  const N = samples.length;
  const D = 14;
  const X = new Float64Array(N * D);
  const y = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const feat = buildFeature(samples[i].classType, samples[i].zoneId, samples[i].dwellTime, samples[i].hour, maxDwell, simHours);
    X.set(feat, i * D);
    y[i] = samples[i].price / maxPrice; // 정규화
  }

  // Mini-batch GD
  const w = new Float64Array(D);
  for (let i = 0; i < D; i++) w[i] = (rng() - 0.5) * 0.01;
  let b = 0;
  const lr = 0.05, epochs = 150, batchSize = 64, lambda = 0.001;
  const indices = Array.from({ length: N }, (_, i) => i);

  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffle(indices, rng);
    for (let start = 0; start < N; start += batchSize) {
      const end = Math.min(start + batchSize, N);
      const m = end - start;
      const gw = new Float64Array(D);
      let gb = 0;
      for (let k = start; k < end; k++) {
        const idx = indices[k];
        let pred = b;
        for (let j = 0; j < D; j++) pred += w[j] * X[idx * D + j];
        const err = pred - y[idx];
        for (let j = 0; j < D; j++) gw[j] += err * X[idx * D + j];
        gb += err;
      }
      for (let j = 0; j < D; j++) w[j] -= lr * (gw[j] / m + lambda * w[j]);
      b -= lr * gb / m;
    }
  }

  // R² 계산
  let ssRes = 0, ssTot = 0;
  let yMean = 0;
  for (let i = 0; i < N; i++) yMean += y[i];
  yMean /= N;
  for (let i = 0; i < N; i++) {
    let pred = b;
    for (let j = 0; j < D; j++) pred += w[j] * X[i * D + j];
    ssRes += (y[i] - pred) ** 2;
    ssTot += (y[i] - yMean) ** 2;
  }
  const r2 = ssTot > 0 ? Math.max(0, 1 - ssRes / ssTot) : 0;

  function predictSingle(classType, zoneId, dwellTime, hour) {
    const feat = buildFeature(classType, zoneId, dwellTime, hour, maxDwell, simHours);
    let pred = b;
    for (let j = 0; j < D; j++) pred += w[j] * feat[j];
    return Math.max(0, pred * maxPrice);
  }

  function predictForZone(classType, dwellTime, hour, activeZoneIds) {
    const targets = activeZoneIds || ZONE_KEYS;
    return targets
      .filter(zId => ZONE_KEYS.includes(zId))
      .map(zoneId => ({
        zoneId,
        label: ZONE_LABELS[zoneId] || zoneId,
        expectedRevenue: Math.round(predictSingle(classType, zoneId, dwellTime, hour)),
      }))
      .sort((a, b) => b.expectedRevenue - a.expectedRevenue);
  }

  return { weights: w, bias: b, maxDwell, maxPrice, simHours, r2, predictForZone };
}

// ── 시간대별 방문자·매출 수요 예측 ──────────────────────────────────────────

/**
 * 시간대별 수요 예측 (이동평균 + 트렌드)
 * @param {object} data - 배치 시뮬레이션 결과
 * @returns {Array} 시간대별 예측 결과
 */
export function predictHourlyDemand(data) {
  const { hourlyStats, dailyStats } = data;
  const simHours = data.operatingHours || 8;
  const days = dailyStats.length;
  const startHour = Math.max(6, 22 - simHours);
  const HOUR_LABELS = Array.from({ length: simHours }, (_, i) => `${startHour + i}시`);

  // 시간대별 일별 데이터 집계
  const byHour = Array.from({ length: simHours }, () => ({ visitors: [], revenue: [], buyers: [] }));
  for (const hs of hourlyStats) {
    if (hs.hour >= 0 && hs.hour < simHours) {
      byHour[hs.hour].visitors.push(hs.visitors);
      byHour[hs.hour].revenue.push(hs.revenue);
      byHour[hs.hour].buyers.push(hs.buyers);
    }
  }

  // 트렌드 계산: 전반부 vs 후반부 비교
  const half = Math.floor(days / 2);
  const firstHalfRev = dailyStats.slice(0, half).reduce((s, d) => s + d.revenue, 0) / (half || 1);
  const secondHalfRev = dailyStats.slice(half).reduce((s, d) => s + d.revenue, 0) / ((days - half) || 1);
  const overallTrend = firstHalfRev > 0 ? secondHalfRev / firstHalfRev : 1;

  return Array.from({ length: simHours }, (_, h) => {
    const vArr = byHour[h].visitors;
    const rArr = byHour[h].revenue;
    const bArr = byHour[h].buyers;

    const avgV = vArr.length > 0 ? vArr.reduce((s, v) => s + v, 0) / vArr.length : 0;
    const avgR = rArr.length > 0 ? rArr.reduce((s, v) => s + v, 0) / rArr.length : 0;
    const avgB = bArr.length > 0 ? bArr.reduce((s, v) => s + v, 0) / bArr.length : 0;

    // 최근 7일 이동평균 (가중)
    const recent = vArr.slice(-7);
    const recentAvgV = recent.length > 0 ? recent.reduce((s, v) => s + v, 0) / recent.length : avgV;
    const recentR = rArr.slice(-7);
    const recentAvgR = recentR.length > 0 ? recentR.reduce((s, v) => s + v, 0) / recentR.length : avgR;

    // 예측값: 최근 평균 × 트렌드 보정
    const trendFactor = Math.min(1.3, Math.max(0.7, overallTrend));
    const predictedVisitors = Math.round(recentAvgV * trendFactor);
    const predictedRevenue = Math.round(recentAvgR * trendFactor);
    const predictedConvRate = predictedVisitors > 0 ? Math.min(1, avgB / avgV) : 0;

    // 트렌드 방향
    const trend = overallTrend > 1.05 ? 'up' : overallTrend < 0.95 ? 'down' : 'stable';

    // 신뢰도: 데이터 수 기반
    const confidence = Math.min(100, Math.round((vArr.length / days) * 100));

    return {
      hour: h,
      label: HOUR_LABELS[h],
      predictedVisitors,
      predictedRevenue,
      predictedConvRate: +predictedConvRate.toFixed(4),
      avgVisitors: Math.round(avgV),
      avgRevenue: Math.round(avgR),
      trend,
      confidence,
    };
  });
}

// ── 고객군별 가치 분석 (CLV + 세그먼트) ──────────────────────────────────────

const CUSTOMER_CLASSES_FULL = {
  adult_male:   { label: '성년 남성',   short: '성년남',   color: '#FF9600' },
  adult_female: { label: '성년 여성',   short: '성년여',   color: '#9314FF' },
  minor_male:   { label: '미성년 남성', short: '미성년남', color: '#00C800' },
  minor_female: { label: '미성년 여성', short: '미성년여', color: '#00A5FF' },
};

/**
 * 고객군별 가치 분석 및 세그먼트 분류
 * @param {object} data - 배치 시뮬레이션 결과
 * @returns {Array} 고객군별 분석 결과
 */
export function computeCustomerValueAnalysis(data) {
  const ZONE_LABELS = {
    snack: '과자/스낵', beverage: '음료', daily: '생활용품',
    cosmetic: '화장품', stationery: '문구', frozen: '냉동식품',
    premium: '프리미엄', bakery: '베이커리',
  };
  const { purchases, classStats, totalVisitors, totalBuyers } = data;
  const simHours = data.operatingHours || 8;
  const startHour = Math.max(6, 22 - simHours);
  const totalRevenue = purchases.reduce((s, p) => s + p.price, 0);

  // agentId별 구매 매대 수 집계
  const agentZones = {};
  purchases.forEach(p => {
    if (!agentZones[p.agentId]) agentZones[p.agentId] = { cls: p.classType, zones: new Set() };
    agentZones[p.agentId].zones.add(p.zoneId);
  });

  // 클래스별 평균 방문 매대 수
  const classZoneCounts = {};
  CLASS_KEYS.forEach(cls => { classZoneCounts[cls] = []; });
  Object.values(agentZones).forEach(a => {
    classZoneCounts[a.cls].push(a.zones.size);
  });

  // 클래스별 매대 구매 빈도
  const classZonePurchases = {};
  CLASS_KEYS.forEach(cls => { classZonePurchases[cls] = {}; });
  purchases.forEach(p => {
    classZonePurchases[p.classType][p.zoneId] = (classZonePurchases[p.classType][p.zoneId] || 0) + 1;
  });

  // 클래스별 시간대 매출
  const classHourRevenue = {};
  CLASS_KEYS.forEach(cls => { classHourRevenue[cls] = new Float64Array(simHours); });
  purchases.forEach(p => {
    if (p.hour >= 0 && p.hour < simHours) {
      classHourRevenue[p.classType][p.hour] += p.price;
    }
  });

  const results = CLASS_KEYS.map(cls => {
    const cs = classStats[cls];
    const info = CUSTOMER_CLASSES_FULL[cls];
    const avgBasket = cs.buyers > 0 ? Math.round(cs.revenue / cs.buyers) : 0;
    const convRate = cs.visitors > 0 ? cs.buyers / cs.visitors : 0;
    const zoneCnts = classZoneCounts[cls];
    const avgZonesVisited = zoneCnts.length > 0
      ? +(zoneCnts.reduce((s, v) => s + v, 0) / zoneCnts.length).toFixed(1)
      : 0;
    const revenueShare = totalRevenue > 0 ? cs.revenue / totalRevenue : 0;
    const lifetimeValue = Math.round(avgBasket * convRate * Math.max(avgZonesVisited, 1));

    // 선호 매대 Top 3
    const zp = classZonePurchases[cls];
    const preferredZones = Object.entries(zp)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([zoneId, count]) => ({ zoneId, label: ZONE_LABELS[zoneId] || zoneId, count }));

    // 피크 시간대
    const hr = classHourRevenue[cls];
    let peakH = 0, peakRev = 0;
    for (let h = 0; h < simHours; h++) {
      if (hr[h] > peakRev) { peakRev = hr[h]; peakH = h; }
    }

    return {
      classType: cls,
      label: info.label,
      short: info.short,
      color: info.color,
      avgBasket,
      convRate: +convRate.toFixed(4),
      avgZonesVisited,
      revenueShare: +revenueShare.toFixed(4),
      lifetimeValue,
      segment: '', // 아래에서 분류
      preferredZones,
      peakHour: { hour: peakH, label: `${startHour + peakH}시`, revenue: Math.round(peakRev) },
    };
  });

  // 세그먼트 분류 (LTV 기준 4분위)
  const sorted = [...results].sort((a, b) => b.lifetimeValue - a.lifetimeValue);
  sorted.forEach((r, i) => {
    const q = i / sorted.length;
    if (q < 0.25) r.segment = 'VIP';
    else if (q < 0.5) r.segment = 'loyal';
    else if (q < 0.75) r.segment = 'casual';
    else r.segment = 'at-risk';
    // 원본 results에도 반영
    const orig = results.find(o => o.classType === r.classType);
    if (orig) orig.segment = r.segment;
  });

  return results;
}

// ── 매대 간 교차 구매 확률 매트릭스 ─────────────────────────────────────────

/**
 * 매대 간 교차 구매 조건부 확률 P(B|A) 계산
 * @param {object} data - 배치 시뮬레이션 결과
 * @returns {{ matrix, topPairs, insights }}
 */
export function predictCrossSell(data) {
  const ZONE_LABELS = {
    snack: '과자/스낵', beverage: '음료', daily: '생활용품',
    cosmetic: '화장품', stationery: '문구', frozen: '냉동식품',
    premium: '프리미엄', bakery: '베이커리',
  };
  const { purchases, zoneStats } = data;

  // agentId별 구매 매대 집합
  const agentZones = {};
  purchases.forEach(p => {
    if (!agentZones[p.agentId]) agentZones[p.agentId] = new Set();
    agentZones[p.agentId].add(p.zoneId);
  });

  // 활성 구역
  const activeZones = ZONE_KEYS.filter(z => zoneStats[z] && zoneStats[z].buyers > 0);

  // 동시 구매 카운트 매트릭스
  const coCount = {};
  const buyerCount = {};
  activeZones.forEach(z => {
    buyerCount[z] = 0;
    coCount[z] = {};
    activeZones.forEach(z2 => { coCount[z][z2] = 0; });
  });

  Object.values(agentZones).forEach(zones => {
    const arr = [...zones].filter(z => activeZones.includes(z));
    arr.forEach(zA => {
      buyerCount[zA] = (buyerCount[zA] || 0) + 1;
      arr.forEach(zB => {
        if (zA !== zB) coCount[zA][zB]++;
      });
    });
  });

  // 조건부 확률 매트릭스 P(B|A)
  const matrix = {};
  activeZones.forEach(zA => {
    matrix[zA] = {};
    activeZones.forEach(zB => {
      if (zA === zB) { matrix[zA][zB] = 0; return; }
      matrix[zA][zB] = buyerCount[zA] > 0
        ? +(coCount[zA][zB] / buyerCount[zA]).toFixed(4)
        : 0;
    });
  });

  // Top 쌍 추출 (중복 제거: A→B와 B→A 중 높은 것만)
  const pairs = [];
  const seen = new Set();
  activeZones.forEach(zA => {
    activeZones.forEach(zB => {
      if (zA === zB) return;
      const key = [zA, zB].sort().join('|');
      if (seen.has(key)) return;
      seen.add(key);
      const probAB = matrix[zA]?.[zB] || 0;
      const probBA = matrix[zB]?.[zA] || 0;
      const maxProb = Math.max(probAB, probBA);
      const count = coCount[zA][zB] || 0;
      if (count > 0) {
        pairs.push({
          zoneA: zA, zoneB: zB,
          labelA: ZONE_LABELS[zA] || zA,
          labelB: ZONE_LABELS[zB] || zB,
          probability: +maxProb.toFixed(4),
          count,
        });
      }
    });
  });
  pairs.sort((a, b) => b.probability - a.probability);
  const topPairs = pairs.slice(0, 5);

  // 인사이트 자동 생성
  const insights = [];
  if (topPairs.length > 0) {
    const best = topPairs[0];
    insights.push(`"${best.labelA}"와 "${best.labelB}" 교차 구매율 ${(best.probability * 100).toFixed(1)}% — 두 매대 인접 배치 시 번들 상품 매출 증가 기대`);
  }
  if (topPairs.length >= 3) {
    const lowPair = pairs[pairs.length - 1];
    if (lowPair && lowPair.probability < 0.1) {
      insights.push(`"${lowPair.labelA}"와 "${lowPair.labelB}" 교차 구매율 ${(lowPair.probability * 100).toFixed(1)}%로 최저 — 동선 연결 또는 할인 쿠폰으로 시너지 유도 가능`);
    }
  }
  // 매대별 교차구매 잠재력
  const crossPotential = activeZones.map(z => {
    const avgProb = activeZones
      .filter(z2 => z2 !== z)
      .reduce((s, z2) => s + (matrix[z]?.[z2] || 0), 0) / Math.max(activeZones.length - 1, 1);
    return { zoneId: z, label: ZONE_LABELS[z] || z, avgCrossProb: avgProb };
  }).sort((a, b) => b.avgCrossProb - a.avgCrossProb);
  if (crossPotential.length > 0) {
    insights.push(`교차구매 유도 거점: "${crossPotential[0].label}" (평균 교차구매율 ${(crossPotential[0].avgCrossProb * 100).toFixed(1)}%) — 이 매대 주변에 신상품·프로모션 집중 권장`);
  }

  return { matrix, topPairs, insights, activeZones };
}
