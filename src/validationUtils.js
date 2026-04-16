// ═══════════════════════════════════════════════════════════════════════════
// 모델 검증 유틸리티 — 4개월 데이터를 3개월(train) + 1개월(test)로 분할하여
// 6개 ML 모델의 예측 성능을 정량적으로 평가
// ═══════════════════════════════════════════════════════════════════════════

import {
  trainLogisticRegression, predictProba,
  trainRevenuePredictor,
  runUCB1Bandit,
  predictHourlyDemand,
  computeCustomerValueAnalysis,
  predictCrossSell,
  buildFeature, buildTrainingSamples,
} from './predictionModels.js';

const CLASS_KEYS = ['adult_male', 'adult_female', 'minor_male', 'minor_female'];
const ZONE_KEYS  = ['snack', 'beverage', 'daily', 'cosmetic', 'stationery', 'frozen', 'premium', 'bakery'];

// ── 헬퍼 함수 ────────────────────────────────────────────────────────────

function sigmoid(z) {
  if (z > 500) return 1;
  if (z < -500) return 0;
  return 1 / (1 + Math.exp(-z));
}

function pearsonCorrelation(xs, ys) {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return 0;
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i]; sy += ys[i];
    sxx += xs[i] * xs[i]; syy += ys[i] * ys[i];
    sxy += xs[i] * ys[i];
  }
  const num = n * sxy - sx * sy;
  const den = Math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
  return den > 0 ? num / den : 0;
}

function spearmanCorrelation(xs, ys) {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return 0;
  const rank = arr => {
    const sorted = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
    const ranks = new Array(arr.length);
    sorted.forEach((s, r) => { ranks[s.i] = r + 1; });
    return ranks;
  };
  return pearsonCorrelation(rank(xs.slice(0, n)), rank(ys.slice(0, n)));
}

// 시드 기반 난수 (재현성)
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ── 데이터 분할 ──────────────────────────────────────────────────────────

/**
 * 120일 배치 데이터를 trainDays(기본 90) / testDays로 분할하고
 * 각 파티션에 대해 zoneStats, classStats 등을 재구축
 */
export function splitAndRebuildStats(fullData, trainDays = 90) {
  const totalDays = fullData.dailyStats.length;
  const simHours  = fullData.operatingHours || 8;

  // ── purchases 분할 ──
  const trainPurchases = fullData.purchases.filter(p => p.day < trainDays);
  const testPurchases  = fullData.purchases.filter(p => p.day >= trainDays);

  // ── dailyStats 분할 ──
  const trainDailyStats = fullData.dailyStats.slice(0, trainDays);
  const testDailyStats  = fullData.dailyStats.slice(trainDays);

  // ── hourlyStats 분할 ──
  const trainHourlyStats = fullData.hourlyStats.filter(h => h.day < trainDays);
  const testHourlyStats  = fullData.hourlyStats.filter(h => h.day >= trainDays);

  // ── zoneStats 재구축 ──
  const trainZoneStats = rebuildZoneStats(trainPurchases, fullData.zoneStats, trainDays, totalDays);
  const testZoneStats  = rebuildZoneStats(testPurchases, fullData.zoneStats, totalDays - trainDays, totalDays);

  // ── classStats 재구축 ──
  const trainClassStats = rebuildClassStats(trainPurchases, trainDailyStats);
  const testClassStats  = rebuildClassStats(testPurchases, testDailyStats);

  // ── 총계 ──
  const trainTotalVisitors = trainDailyStats.reduce((s, d) => s + d.visitors, 0);
  const testTotalVisitors  = testDailyStats.reduce((s, d) => s + d.visitors, 0);

  const trainBuyerIds = new Set(trainPurchases.map(p => p.agentId));
  const testBuyerIds  = new Set(testPurchases.map(p => p.agentId));

  const trainData = {
    purchases: trainPurchases,
    zoneStats: trainZoneStats,
    classStats: trainClassStats,
    dailyStats: trainDailyStats,
    hourlyStats: trainHourlyStats,
    operatingHours: simHours,
    totalVisitors: trainTotalVisitors,
    totalBuyers: trainBuyerIds.size,
  };

  const testData = {
    purchases: testPurchases,
    zoneStats: testZoneStats,
    classStats: testClassStats,
    dailyStats: testDailyStats,
    hourlyStats: testHourlyStats,
    operatingHours: simHours,
    totalVisitors: testTotalVisitors,
    totalBuyers: testBuyerIds.size,
  };

  return { trainData, testData, trainDays, testDays: totalDays - trainDays };
}

function rebuildZoneStats(purchases, fullZoneStats, periodDays, totalDays) {
  const ratio = periodDays / totalDays;
  const stats = {};

  ZONE_KEYS.forEach(zId => {
    const full = fullZoneStats[zId];
    if (!full) return;

    // 비율 추정: visitors, totalDwell
    const estimatedVisitors = Math.round(full.visitors * ratio);
    const estimatedDwell    = full.totalDwell * ratio;

    // purchases에서 정확 집계: buyers, revenue
    const zonePurchases = purchases.filter(p => p.zoneId === zId);
    const buyerSet = new Set(zonePurchases.map(p => p.agentId));
    const revenue  = zonePurchases.reduce((s, p) => s + p.price, 0);
    const totalDwellFromPurchases = zonePurchases.reduce((s, p) => s + p.dwellTime, 0);

    // 클래스별 집계
    const visitorsByClass = {};
    const buyersByClass   = {};
    CLASS_KEYS.forEach(cls => {
      visitorsByClass[cls] = Math.round((full.visitorsByClass?.[cls] || 0) * ratio);
      const clsBuyers = new Set(zonePurchases.filter(p => p.classType === cls).map(p => p.agentId));
      buyersByClass[cls] = clsBuyers.size;
    });

    stats[zId] = {
      label: full.label,
      visitors: estimatedVisitors,
      buyers: buyerSet.size,
      totalDwell: estimatedDwell,
      revenue,
      visitorsByClass,
      buyersByClass,
    };
  });

  return stats;
}

function rebuildClassStats(purchases, dailyStats) {
  const stats = {};
  CLASS_KEYS.forEach(cls => {
    const clsPurchases = purchases.filter(p => p.classType === cls);
    const buyerSet = new Set(clsPurchases.map(p => p.agentId));
    const revenue  = clsPurchases.reduce((s, p) => s + p.price, 0);

    // visitors: dailyStats에서 비율 추정 (전체 방문자 중 클래스 비율은 일정하다고 가정)
    const totalVisitors = dailyStats.reduce((s, d) => s + d.visitors, 0);
    // 클래스별 방문자 비율은 purchases 비율로 추정
    const allAgentIds = new Set(purchases.map(p => p.agentId));
    const clsAgentIds = new Set(clsPurchases.map(p => p.agentId));
    const clsRatio = allAgentIds.size > 0 ? clsAgentIds.size / allAgentIds.size : 0.25;

    stats[cls] = {
      visitors: Math.round(totalVisitors * clsRatio),
      buyers: buyerSet.size,
      revenue,
      purchaseCnt: clsPurchases.length,
    };
  });
  return stats;
}

// ── 1. Logistic Regression 검증 ──────────────────────────────────────────

export function validateLogisticRegression(trainData, testData) {
  // 학습
  const model = trainLogisticRegression(trainData);
  const trainAccuracy = model.accuracy;
  const trainAuc = model.auc;

  // 테스트 샘플 생성
  const { samples: testSamples } = buildTrainingSamples(testData);
  if (testSamples.length === 0) {
    return {
      train: { accuracy: trainAccuracy, auc: trainAuc },
      test: { accuracy: 0, auc: 0, precision: 0, recall: 0, f1: 0, confusion: { tp: 0, fp: 0, tn: 0, fn: 0 } },
      sampleCount: { train: 0, test: 0 },
    };
  }

  // 테스트 예측
  let tp = 0, fp = 0, tn = 0, fn = 0;
  const posPreds = [], negPreds = [];

  for (const s of testSamples) {
    const prob = predictProba(model, s.classType, s.zoneId, s.dwellTime, s.hour);
    const pred = prob >= 0.5 ? 1 : 0;

    if (s.label === 1 && pred === 1) tp++;
    else if (s.label === 0 && pred === 1) fp++;
    else if (s.label === 0 && pred === 0) tn++;
    else fn++;

    if (s.label === 1) posPreds.push(prob);
    else negPreds.push(prob);
  }

  const accuracy  = testSamples.length > 0 ? (tp + tn) / testSamples.length : 0;
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const recall    = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1        = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;

  // AUC (Mann-Whitney U)
  let auc = 0.5;
  if (posPreds.length > 0 && negPreds.length > 0) {
    let concordant = 0;
    const maxPairs = 50000;
    const totalPairs = posPreds.length * negPreds.length;
    if (totalPairs <= maxPairs) {
      for (let i = 0; i < posPreds.length; i++)
        for (let j = 0; j < negPreds.length; j++) {
          if (posPreds[i] > negPreds[j]) concordant++;
          else if (posPreds[i] === negPreds[j]) concordant += 0.5;
        }
      auc = concordant / totalPairs;
    } else {
      const rng = mulberry32(888);
      for (let k = 0; k < maxPairs; k++) {
        const pi = posPreds[Math.floor(rng() * posPreds.length)];
        const ni = negPreds[Math.floor(rng() * negPreds.length)];
        if (pi > ni) concordant++;
        else if (pi === ni) concordant += 0.5;
      }
      auc = concordant / maxPairs;
    }
  }

  return {
    train: { accuracy: trainAccuracy, auc: trainAuc },
    test: { accuracy, auc, precision, recall, f1, confusion: { tp, fp, tn, fn } },
    sampleCount: { train: trainData.purchases.length, test: testSamples.length },
  };
}

// ── 2. Linear Regression 검증 ────────────────────────────────────────────

export function validateLinearRegression(trainData, testData) {
  const model = trainRevenuePredictor(trainData);
  const trainR2 = model.r2;

  const testPurchases = testData.purchases;
  if (testPurchases.length === 0) {
    return {
      train: { r2: trainR2, mae: 0, mape: 0, rmse: 0 },
      test: { r2: 0, mae: 0, mape: 0, rmse: 0 },
      scatterData: [],
    };
  }

  // 테스트 예측
  const actuals = [];
  const preds   = [];
  const scatterData = [];

  // 샘플 제한 (산점도용)
  const rng = mulberry32(321);
  const sampleLimit = Math.min(testPurchases.length, 500);
  const indices = Array.from({ length: testPurchases.length }, (_, i) => i);
  shuffle(indices, rng);
  const sampleIndices = new Set(indices.slice(0, sampleLimit));

  for (let i = 0; i < testPurchases.length; i++) {
    const p = testPurchases[i];
    const feat = buildFeature(p.classType, p.zoneId, p.dwellTime, p.hour, model.maxDwell, model.simHours);
    let pred = model.bias;
    for (let j = 0; j < 14; j++) pred += model.weights[j] * feat[j];
    pred = Math.max(0, pred * model.maxPrice);

    actuals.push(p.price);
    preds.push(pred);

    if (sampleIndices.has(i)) {
      scatterData.push({ actual: p.price, predicted: Math.round(pred) });
    }
  }

  // R²
  const N = actuals.length;
  let yMean = 0;
  for (let i = 0; i < N; i++) yMean += actuals[i];
  yMean /= N;

  let ssRes = 0, ssTot = 0, sumAbsErr = 0, sumPctErr = 0, pctCount = 0;
  for (let i = 0; i < N; i++) {
    const err = actuals[i] - preds[i];
    ssRes += err * err;
    ssTot += (actuals[i] - yMean) * (actuals[i] - yMean);
    sumAbsErr += Math.abs(err);
    if (actuals[i] > 0) { sumPctErr += Math.abs(err) / actuals[i]; pctCount++; }
  }

  const r2   = ssTot > 0 ? Math.max(0, 1 - ssRes / ssTot) : 0;
  const mae  = sumAbsErr / N;
  const mape = pctCount > 0 ? (sumPctErr / pctCount) * 100 : 0;
  const rmse = Math.sqrt(ssRes / N);

  return {
    train: { r2: trainR2, mae: 0, mape: 0, rmse: 0 },  // train MAE는 모델에서 미제공
    test: { r2, mae: Math.round(mae), mape: +mape.toFixed(1), rmse: Math.round(rmse) },
    scatterData,
  };
}

// ── 3. UCB1 Bandit 검증 ──────────────────────────────────────────────────

export function validateUCB1(trainData, testData) {
  const trainResult = runUCB1Bandit(trainData.zoneStats, 500);
  const recZone = trainResult.recommendation.zoneId;

  // test 기간 실제 전환율
  const testConvRates = {};
  ZONE_KEYS.forEach(zId => {
    const zs = testData.zoneStats[zId];
    if (zs && zs.visitors > 0) {
      testConvRates[zId] = zs.buyers / zs.visitors;
    }
  });

  // 순위 계산
  const sorted = Object.entries(testConvRates)
    .sort(([, a], [, b]) => b - a);
  const rank = sorted.findIndex(([z]) => z === recZone) + 1;
  const bestTestRate = sorted.length > 0 ? sorted[0][1] : 0;
  const recTestRate  = testConvRates[recZone] || 0;
  const regret = bestTestRate - recTestRate;

  // Q값 vs 실제전환율 상관계수
  const qValues = [];
  const actualRates = [];
  const comparisonData = [];

  const ZONE_LABELS = {
    snack: '과자/스낵', beverage: '음료', daily: '생활용품',
    cosmetic: '화장품', stationery: '문구', frozen: '냉동식품',
    premium: '프리미엄', bakery: '베이커리',
  };

  for (const arm of trainResult.arms) {
    if (testConvRates[arm.zoneId] !== undefined) {
      qValues.push(arm.Q);
      actualRates.push(testConvRates[arm.zoneId]);
      comparisonData.push({
        zone: arm.label,
        zoneId: arm.zoneId,
        trainQ: +(arm.Q * 100).toFixed(1),
        testActual: +(testConvRates[arm.zoneId] * 100).toFixed(1),
        isRecommended: arm.zoneId === recZone,
      });
    }
  }

  const correlation = pearsonCorrelation(qValues, actualRates);
  const performanceRatio = bestTestRate > 0 ? recTestRate / bestTestRate : 0;

  return {
    recommendation: {
      zoneId: recZone,
      label: ZONE_LABELS[recZone] || recZone,
      confidence: trainResult.recommendation.confidence,
    },
    rank,
    totalZones: sorted.length,
    regret: +regret.toFixed(4),
    performanceRatio: +performanceRatio.toFixed(4),
    correlation: +correlation.toFixed(4),
    isHit: rank === 1,
    isTop3: rank <= 3,
    comparisonData,
    bestTestZone: sorted.length > 0 ? { zoneId: sorted[0][0], label: ZONE_LABELS[sorted[0][0]] || sorted[0][0], rate: +(sorted[0][1] * 100).toFixed(1) } : null,
  };
}

// ── 4. Hourly Demand 검증 ────────────────────────────────────────────────

export function validateHourlyDemand(trainData, testData) {
  const predictions = predictHourlyDemand(trainData);
  const simHours = trainData.operatingHours || 8;
  const testDays = testData.dailyStats.length;

  // test 기간 실제 시간대별 평균
  const testByHour = Array.from({ length: simHours }, () => ({ visitors: [], revenue: [] }));
  for (const hs of testData.hourlyStats) {
    if (hs.hour >= 0 && hs.hour < simHours) {
      testByHour[hs.hour].visitors.push(hs.visitors);
      testByHour[hs.hour].revenue.push(hs.revenue);
    }
  }

  const hourlyComparison = [];
  let totalVisitorErr = 0, totalRevenueErr = 0;
  let totalVisitorPct = 0, totalRevenuePct = 0;
  let pctCountV = 0, pctCountR = 0;

  for (let h = 0; h < simHours; h++) {
    const pred = predictions[h];
    const vArr = testByHour[h].visitors;
    const rArr = testByHour[h].revenue;
    const actualV = vArr.length > 0 ? Math.round(vArr.reduce((s, v) => s + v, 0) / vArr.length) : 0;
    const actualR = rArr.length > 0 ? Math.round(rArr.reduce((s, v) => s + v, 0) / rArr.length) : 0;

    const vErr = Math.abs(pred.predictedVisitors - actualV);
    const rErr = Math.abs(pred.predictedRevenue - actualR);

    totalVisitorErr += vErr;
    totalRevenueErr += rErr;
    if (actualV > 0) { totalVisitorPct += vErr / actualV; pctCountV++; }
    if (actualR > 0) { totalRevenuePct += rErr / actualR; pctCountR++; }

    hourlyComparison.push({
      hour: h,
      label: pred.label,
      predictedVisitors: pred.predictedVisitors,
      actualVisitors: actualV,
      visitorError: vErr,
      predictedRevenue: pred.predictedRevenue,
      actualRevenue: actualR,
      revenueError: rErr,
    });
  }

  const visitorMAE  = simHours > 0 ? Math.round(totalVisitorErr / simHours) : 0;
  const revenueMAE  = simHours > 0 ? Math.round(totalRevenueErr / simHours) : 0;
  const visitorMAPE = pctCountV > 0 ? +((totalVisitorPct / pctCountV) * 100).toFixed(1) : 0;
  const revenueMAPE = pctCountR > 0 ? +((totalRevenuePct / pctCountR) * 100).toFixed(1) : 0;

  return {
    hourlyComparison,
    visitorMAE, revenueMAE,
    visitorMAPE, revenueMAPE,
    overallMAPE: +((visitorMAPE + revenueMAPE) / 2).toFixed(1),
  };
}

// ── 5. Customer Value 검증 ───────────────────────────────────────────────

export function validateCustomerValue(trainData, testData) {
  const trainCV = computeCustomerValueAnalysis(trainData);
  const testCV  = computeCustomerValueAnalysis(testData);

  // 세그먼트 안정성
  let stableCount = 0;
  const comparison = CLASS_KEYS.map((cls, i) => {
    const tr = trainCV[i];
    const te = testCV[i];
    const segmentStable = tr.segment === te.segment;
    if (segmentStable) stableCount++;

    return {
      classType: cls,
      label: tr.label,
      short: tr.short,
      color: tr.color,
      trainSegment: tr.segment,
      testSegment: te.segment,
      segmentStable,
      trainLTV: tr.lifetimeValue,
      testLTV: te.lifetimeValue,
      ltvDelta: te.lifetimeValue - tr.lifetimeValue,
      trainRevenueShare: tr.revenueShare,
      testRevenueShare: te.revenueShare,
      revenueShareDelta: +(te.revenueShare - tr.revenueShare).toFixed(4),
      trainPreferredZones: tr.preferredZones.map(z => z.zoneId),
      testPreferredZones: te.preferredZones.map(z => z.zoneId),
    };
  });

  const segmentStability = stableCount / CLASS_KEYS.length;

  // LTV 순위 상관
  const trainLTVs = comparison.map(c => c.trainLTV);
  const testLTVs  = comparison.map(c => c.testLTV);
  const ltvRankCorr = spearmanCorrelation(trainLTVs, testLTVs);

  // 선호매대 일치율 (Top 3 중 겹치는 비율)
  let zoneOverlapTotal = 0;
  comparison.forEach(c => {
    const overlap = c.trainPreferredZones.filter(z => c.testPreferredZones.includes(z)).length;
    const maxLen  = Math.max(c.trainPreferredZones.length, 1);
    zoneOverlapTotal += overlap / maxLen;
  });
  const avgZoneOverlap = zoneOverlapTotal / CLASS_KEYS.length;

  return {
    comparison,
    segmentStability: +segmentStability.toFixed(4),
    ltvRankCorrelation: +ltvRankCorr.toFixed(4),
    avgZoneOverlap: +avgZoneOverlap.toFixed(4),
  };
}

// ── 6. Cross-Sell 검증 ───────────────────────────────────────────────────

export function validateCrossSell(trainData, testData) {
  const trainCS = predictCrossSell(trainData);
  const testCS  = predictCrossSell(testData);

  // 공통 구역
  const commonZones = trainCS.activeZones.filter(z => testCS.activeZones.includes(z));

  // 매트릭스 상관계수
  const trainVals = [];
  const testVals  = [];
  commonZones.forEach(zA => {
    commonZones.forEach(zB => {
      if (zA !== zB) {
        trainVals.push(trainCS.matrix[zA]?.[zB] || 0);
        testVals.push(testCS.matrix[zB] !== undefined ? (testCS.matrix[zA]?.[zB] || 0) : 0);
      }
    });
  });
  const matrixCorrelation = pearsonCorrelation(trainVals, testVals);

  // Top-5 쌍 일치율
  const trainTop5Ids = new Set(trainCS.topPairs.map(p => [p.zoneA, p.zoneB].sort().join('|')));
  const testTop5Ids  = new Set(testCS.topPairs.map(p => [p.zoneA, p.zoneB].sort().join('|')));
  let top5Overlap = 0;
  trainTop5Ids.forEach(id => { if (testTop5Ids.has(id)) top5Overlap++; });
  const top5OverlapRate = trainTop5Ids.size > 0 ? top5Overlap / trainTop5Ids.size : 0;

  // 쌍별 확률 편차
  let sumDelta = 0, maxDelta = 0, pairCount = 0;
  commonZones.forEach(zA => {
    commonZones.forEach(zB => {
      if (zA !== zB) {
        const tProb = trainCS.matrix[zA]?.[zB] || 0;
        const eProb = testCS.matrix[zA]?.[zB] || 0;
        const delta = Math.abs(tProb - eProb);
        sumDelta += delta;
        if (delta > maxDelta) maxDelta = delta;
        pairCount++;
      }
    });
  });
  const avgDelta = pairCount > 0 ? sumDelta / pairCount : 0;

  // 비교 데이터 (Top 5 쌍)
  const pairComparison = trainCS.topPairs.map(tp => {
    const key = [tp.zoneA, tp.zoneB].sort().join('|');
    const testMatch = testCS.topPairs.find(ep => [ep.zoneA, ep.zoneB].sort().join('|') === key);
    const testProb = testCS.matrix[tp.zoneA]?.[tp.zoneB] || testCS.matrix[tp.zoneB]?.[tp.zoneA] || 0;
    return {
      labelA: tp.labelA,
      labelB: tp.labelB,
      trainProb: tp.probability,
      testProb: +testProb.toFixed(4),
      inTestTop5: testTop5Ids.has(key),
    };
  });

  return {
    matrixCorrelation: +matrixCorrelation.toFixed(4),
    top5OverlapRate: +top5OverlapRate.toFixed(4),
    avgDelta: +avgDelta.toFixed(4),
    maxDelta: +maxDelta.toFixed(4),
    pairComparison,
  };
}

// ── 통합 검증 함수 ───────────────────────────────────────────────────────

/**
 * 전체 검증 파이프라인: 4개월 데이터 → 3/1 분할 → 6개 모델 검증
 * @param {object} fullData - 120일 배치 시뮬레이션 결과
 * @returns {object} 6개 모델 검증 결과
 */
export function runFullValidation(fullData) {
  const totalDays = fullData.dailyStats.length;
  const trainDays = Math.floor(totalDays * 0.75); // 90일 (4개월 중 3개월)

  const { trainData, testData, testDays } = splitAndRebuildStats(fullData, trainDays);

  const logistic      = validateLogisticRegression(trainData, testData);
  const linear        = validateLinearRegression(trainData, testData);
  const ucb1          = validateUCB1(trainData, testData);
  const hourly        = validateHourlyDemand(trainData, testData);
  const customerValue = validateCustomerValue(trainData, testData);
  const crossSell     = validateCrossSell(trainData, testData);

  return {
    trainDays,
    testDays,
    logistic,
    linear,
    ucb1,
    hourly,
    customerValue,
    crossSell,
  };
}
