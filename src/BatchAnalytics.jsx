import React, { useMemo, useRef, useEffect, useState, useCallback } from 'react';
import metricsIcecream    from './model-showcase/data/metrics_icecream.json';
import metricsConvenience from './model-showcase/data/metrics_convenience.json';
import metricsSupermarket from './model-showcase/data/metrics_supermarket.json';

const STORE_METRICS = {
  icecream:    metricsIcecream,
  convenience: metricsConvenience,
  supermarket: metricsSupermarket,
};
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, ReferenceLine, ReferenceArea,
  XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid, ScatterChart, Scatter, ZAxis,
} from 'recharts';
import { trainLogisticRegression, predictAllZones, runUCB1Bandit, trainRevenuePredictor, predictHourlyDemand, computeCustomerValueAnalysis, predictCrossSell } from './predictionModels.js';
import AIPredictionPanel from './components/AIPredictionPanel.jsx';
import { runFullValidation, runLSTMValidation } from './validationUtils.js';

// ── 소프트 퍼플·핑크 팔레트 ─────────────────────────────────────────────────
const SOFT_RAMP = ['#a78bfa','#9277ee','#8b7bf0','#b794f6','#c8a8ee','#dcb6e6','#f0abfc'];
const softAt = (i) => SOFT_RAMP[i % SOFT_RAMP.length];
const OURS    = '#8b7bf0';   // 우리/예측/Test
const BASE    = '#cbd5e1';   // 기준/실제/Train (회색)
const WEEKEND = '#f0abfc';   // 주말 강조 (연핑크)

const CLASS_PURPLE = {
  adult_male:   '#7c6df0',
  adult_female: '#a78bfa',
  minor_male:   '#c4a3ec',
  minor_female: '#f0abfc',
};

// ── 상수 ────────────────────────────────────────────────────────────────────

const CLASS_KEYS = ['adult_male', 'adult_female', 'minor_male', 'minor_female'];
const CUSTOMER_CLASSES = {
  adult_male:   { label: '성년 남성',   short: '성년남',   color: '#FF9600' },
  adult_female: { label: '성년 여성',   short: '성년여',   color: '#9314FF' },
  minor_male:   { label: '미성년 남성', short: '미성년남', color: '#00C800' },
  minor_female: { label: '미성년 여성', short: '미성년여', color: '#00A5FF' },
};
const DOW_LABELS = ['월', '화', '수', '목', '금', '토', '일'];
const VIRIDIS = [
  [68,1,84],[72,40,120],[62,73,137],[49,104,142],
  [38,130,142],[31,158,137],[53,183,121],[110,206,88],
  [181,222,43],[253,231,37],
];

function viridisColor(t) {
  const idx = Math.min(t * (VIRIDIS.length - 1), VIRIDIS.length - 1.001);
  const lo = Math.floor(idx), hi = lo + 1;
  const f  = idx - lo;
  const [r, g, b] = VIRIDIS[lo].map((c, i) => Math.round(c + f * (VIRIDIS[hi][i] - c)));
  return `rgb(${r},${g},${b})`;
}

const TOOLTIP_STYLE = {
  contentStyle: { background: '#fff', border: '1px solid #e5e7eb', color: '#1f2937', fontSize: '11px', borderRadius: '6px' },
  itemStyle:    { color: '#374151' },
  labelStyle:   { color: '#6b7280', fontWeight: 'bold', marginBottom: '2px' },
};
const TICK_STYLE = { fill: '#6b7280', fontSize: 10 };

// ── KPI 계산 ────────────────────────────────────────────────────────────────

function computeKPIs(data) {
  const { zoneStats, classStats, dailyStats, hourlyStats, totalVisitors, totalBuyers } = data;
  const days = dailyStats.length;
  const SIM_HOURS = data.operatingHours || 8;
  const startHour = Math.max(6, 22 - SIM_HOURS);
  const HOUR_LABELS = Array.from({ length: SIM_HOURS }, (_, i) => `${startHour + i}시`);

  // 전체 구매전환율
  const overallConvRate = totalVisitors > 0 ? totalBuyers / totalVisitors : 0;

  // 구역별 KPI
  const zoneKPIs = Object.entries(zoneStats).map(([id, s]) => {
    const convRate  = s.visitors > 0 ? s.buyers / s.visitors : 0;
    const avgDwell  = s.visitors > 0 ? s.totalDwell / s.visitors : 0;
    const buyerShare = totalBuyers > 0 ? s.buyers / totalBuyers : 0;
    const entryRate  = totalVisitors > 0 ? s.visitors / totalVisitors : 0;
    const efficiency = avgDwell > 0 ? convRate / avgDwell : 0; // 체류 대비 전환 효율
    return { id, label: s.label, convRate, avgDwell, buyerShare, entryRate, efficiency, revenue: s.revenue, visitors: s.visitors, buyers: s.buyers };
  });

  // 비효율 지수 (정규화)
  const maxDwell = Math.max(...zoneKPIs.map(z => z.avgDwell)) || 1;
  const maxConv  = Math.max(...zoneKPIs.map(z => z.convRate))  || 1;
  zoneKPIs.forEach(z => {
    z.inefficiency = (z.avgDwell / maxDwell) * (1 - z.convRate / maxConv);
  });

  // 신상품 배치 적합도 (구역 유입률 × 전환율 × 0.5 [고객군 적합도는 유입률 기반 추정])
  zoneKPIs.forEach(z => {
    z.newProductScore = z.entryRate * z.convRate * (1 + z.buyerShare);
  });

  // 고객군별 KPI
  const classKPIs = CLASS_KEYS.map(cls => {
    const s = classStats[cls];
    return {
      cls, label: CUSTOMER_CLASSES[cls].label, short: CUSTOMER_CLASSES[cls].short,
      color: CUSTOMER_CLASSES[cls].color,
      visitors: s.visitors, buyers: s.buyers, revenue: s.revenue,
      convRate: s.visitors > 0 ? s.buyers / s.visitors : 0,
      avgBasket: s.buyers > 0 ? s.revenue / s.buyers : 0,
    };
  });

  // 시간대별 KPI (hour 0~7 을 집계)
  const hourlyAgg = Array.from({ length: SIM_HOURS }, (_, h) => {
    const buckets = hourlyStats.filter(hs => hs.hour === h);
    const totalV  = buckets.reduce((s, x) => s + x.visitors, 0);
    const totalB  = buckets.reduce((s, x) => s + x.buyers,   0);
    const totalR  = buckets.reduce((s, x) => s + x.revenue,  0);
    return {
      hour: h, label: HOUR_LABELS[h],
      visitors: totalV, buyers: totalB, revenue: totalR,
      convRate: totalV > 0 ? totalB / totalV : 0,
    };
  });

  // 요일별 패턴
  const dowAgg = Array.from({ length: 7 }, (_, d) => {
    const relevant = dailyStats.filter(ds => ds.dayOfWeek === d);
    const avgRev  = relevant.length > 0 ? relevant.reduce((s, x) => s + x.revenue, 0) / relevant.length : 0;
    const avgVisit = relevant.length > 0 ? relevant.reduce((s, x) => s + x.visitors, 0) / relevant.length : 0;
    return { dow: DOW_LABELS[d], avgRevenue: Math.round(avgRev), avgVisitors: Math.round(avgVisit) };
  });

  // 일별 매출 추이
  const dailySalesChart = dailyStats.map(d => ({
    day: d.day + 1, revenue: d.revenue, visitors: d.visitors,
    convRate: +(d.convRate * 100).toFixed(1),
  }));

  // 다음달 예측: 후반 7일 트렌드 × 30일
  const last7  = dailyStats.slice(-7).map(d => d.revenue);
  const first7 = dailyStats.slice(0, 7).map(d => d.revenue);
  const avgLast7  = last7.reduce((s, v) => s + v, 0) / 7;
  const avgFirst7 = first7.reduce((s, v) => s + v, 0) / 7 || 1;
  const trend     = Math.min(avgLast7 / avgFirst7, 1.3); // 최대 30% 성장 캡
  const nextMonthForecast = Math.round(avgLast7 * 30 * trend);

  // 최적 인력 집중 시간대 (Top 3)
  const bestHours = [...hourlyAgg].sort((a, b) => b.revenue - a.revenue)
    .slice(0, 3).map(h => h.label);

  // 효율 최고/최저 구역
  const bestZone  = [...zoneKPIs].sort((a, b) => b.efficiency - a.efficiency)[0];
  const worstZone = [...zoneKPIs].sort((a, b) => b.inefficiency - a.inefficiency)[0];

  // ── 핵심 인사이트 5개 (무분별한 나열 방지, 가장 중요한 것만 선정) ──────────
  const insights = [];
  const totalRevenue = data.purchases.reduce((s, p) => s + p.price, 0);

  // [1] 비효율 구역: 방문은 많지만 전환율이 전체 평균의 60% 미만인 구역
  //     → 레이아웃·진열 개선이 가장 직접적인 매출 레버
  const sortedByIneff = [...zoneKPIs]
    .filter(z => z.visitors >= 5 && z.convRate < overallConvRate * 0.6)
    .sort((a, b) => b.visitors - a.visitors);
  if (sortedByIneff.length > 0) {
    const z = sortedByIneff[0];
    insights.push(`🔴 비효율 구역 "${z.label}": 방문 ${z.visitors}회 대비 전환율 ${(z.convRate * 100).toFixed(1)}% (전체 평균 ${(overallConvRate * 100).toFixed(1)}%의 절반 수준) — 진열 위치·가격 가시성·상품 구성 재검토 우선 권장`);
  } else if (worstZone && worstZone.avgDwell > 1.5) {
    insights.push(`🔴 비효율 구역 "${worstZone.label}": 평균 체류 ${worstZone.avgDwell.toFixed(1)}초 대비 전환율 ${(worstZone.convRate * 100).toFixed(1)}% — 가격 태그 가독성 또는 구역 위치 개선 검토`);
  }

  // [2] 교차구매 기회: 동일 고객이 함께 가장 많이 구매한 구역 쌍
  //     → 인접 배치 또는 번들 기획으로 즉시 객단가 향상 가능
  const agentZoneMap = {};
  data.purchases.forEach(p => {
    if (!agentZoneMap[p.agentId]) agentZoneMap[p.agentId] = [];
    if (!agentZoneMap[p.agentId].includes(p.zoneId)) agentZoneMap[p.agentId].push(p.zoneId);
  });
  const zonePairCount = {};
  Object.values(agentZoneMap).forEach(zones => {
    for (let i = 0; i < zones.length; i++)
      for (let j = i + 1; j < zones.length; j++) {
        const key = [zones[i], zones[j]].sort().join('|');
        zonePairCount[key] = (zonePairCount[key] || 0) + 1;
      }
  });
  const topCrossPair = Object.entries(zonePairCount).sort((a, b) => b[1] - a[1])[0];
  if (topCrossPair && topCrossPair[1] >= 3) {
    const [zA, zB] = topCrossPair[0].split('|');
    const la = zoneKPIs.find(z => z.id === zA)?.label || zA;
    const lb = zoneKPIs.find(z => z.id === zB)?.label || zB;
    const totalBuyersForPair = Object.keys(agentZoneMap).length;
    const crossPct = totalBuyersForPair > 0 ? ((topCrossPair[1] / totalBuyersForPair) * 100).toFixed(0) : 0;
    insights.push(`🔗 교차구매 기회: "${la}" + "${lb}" 동시 구매 ${topCrossPair[1]}건 (구매 고객의 ${crossPct}%) — 두 구역 인접 배치 또는 세트 묶음 할인으로 객단가 향상 효과 기대`);
  }

  // [3] VIP 고객군 전략: 객단가 최고 고객군의 선호 구역 우선 관리
  //     → 매출의 대부분을 차지하는 핵심 고객층 집중 대응
  const vipCls = [...classKPIs].filter(c => c.buyers > 0).sort((a, b) => b.avgBasket - a.avgBasket)[0];
  if (vipCls && totalRevenue > 0) {
    const vipRevShare = ((classKPIs.find(c => c.cls === vipCls.cls)?.revenue || 0) / totalRevenue * 100).toFixed(0);
    insights.push(`👑 VIP 고객군 "${vipCls.label}": 객단가 ₩${vipCls.avgBasket.toLocaleString()}, 전체 매출의 ${vipRevShare}% 점유 — 해당 고객 선호 구역 재고 우선 보충·프리미엄 상품 진열 강화 권장`);
  }

  // [4] 피크 시간대 재고 관리: 매출 최고 시간대 vs 최저 시간대 격차
  //     → 피크 전 재고 보충·모니터링 타이밍 설정 근거
  const hourRevSorted = [...hourlyAgg].sort((a, b) => b.revenue - a.revenue);
  if (hourRevSorted.length >= 2 && hourRevSorted[hourRevSorted.length - 1].revenue > 0) {
    const peak = hourRevSorted[0];
    const offPeak = hourRevSorted[hourRevSorted.length - 1];
    const ratio = (peak.revenue / offPeak.revenue).toFixed(1);
    insights.push(`⏰ 피크 시간대 "${peak.label}" 매출이 최저 "${offPeak.label}" 대비 ${ratio}배 — 피크 전 재고 보충 스케줄링 및 비피크 시간대 할인 행사로 매출 평탄화 가능`);
  } else if (bestHours.length > 0) {
    insights.push(`⏰ 매출 집중 시간대: ${bestHours.join(', ')} — 해당 시간대 재고 부족 방지 및 키오스크 정상 작동 점검 우선 권장`);
  }

  // [5] 매출 추세: 4개월 전반부 vs 후반부 성장률
  //     → 상품 기획 방향 결정 (확대 or 프로모션 필요)
  const half = Math.floor(days / 2);
  if (half >= 7 && totalRevenue > 0) {
    const avgFirst  = dailyStats.slice(0, half).reduce((s, d) => s + d.revenue, 0) / half;
    const avgSecond = dailyStats.slice(half).reduce((s, d) => s + d.revenue, 0) / (days - half);
    const growthPct = avgFirst > 0 ? ((avgSecond / avgFirst - 1) * 100).toFixed(1) : null;
    if (growthPct !== null) {
      const arrow = Number(growthPct) >= 5 ? '📈' : Number(growthPct) <= -5 ? '📉' : '➡️';
      insights.push(`${arrow} 매출 추세: 후반부 일평균이 전반부 대비 ${Number(growthPct) >= 0 ? '+' : ''}${growthPct}% — ${Number(growthPct) > 5 ? '성장 유지 위해 인기 구역 재고 확보 강화' : Number(growthPct) < -5 ? '하락 감지, 비효율 구역 정비 및 프로모션 시급' : '안정적 매출 유지 중, 신상품 도입으로 성장 모멘텀 확보 검토'}`);
    }
  } else {
    const revSortedZones = [...zoneKPIs].sort((a, b) => b.revenue - a.revenue);
    if (revSortedZones.length >= 2 && totalRevenue > 0) {
      const top2Rev   = revSortedZones[0].revenue + revSortedZones[1].revenue;
      const top2Share = ((top2Rev / totalRevenue) * 100).toFixed(0);
      insights.push(`📦 매출 집중도: 상위 2개 구역 ("${revSortedZones[0].label}", "${revSortedZones[1].label}")이 전체의 ${top2Share}% 점유 — 해당 구역 품절 리스크 집중 관리 필요`);
    }
  }

  return {
    overallConvRate, zoneKPIs, classKPIs, hourlyAgg, dowAgg,
    dailySalesChart, nextMonthForecast, bestHours, bestZone, worstZone, insights,
    SIM_HOURS, HOUR_LABELS,
  };
}

// ── 히트맵 미니 캔버스 ──────────────────────────────────────────────────────

function HeatmapMini({ heatGrids, classType, width = 200, height = 150 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !heatGrids) return;
    const ctx  = canvas.getContext('2d');
    const HEAT_W = 200, HEAT_H = 150;
    ctx.fillStyle = '#f3f4f6';
    ctx.fillRect(0, 0, HEAT_W, HEAT_H);

    let grid;
    if (classType === 'combined') {
      grid = new Float32Array(HEAT_W * HEAT_H);
      Object.values(heatGrids).forEach(g => { for (let i = 0; i < grid.length; i++) grid[i] += g[i]; });
    } else {
      grid = heatGrids[classType];
    }
    let maxVal = 0;
    for (let i = 0; i < grid.length; i++) if (grid[i] > maxVal) maxVal = grid[i];
    if (maxVal === 0) return;

    const id = ctx.createImageData(HEAT_W, HEAT_H);
    const d  = id.data;
    for (let i = 0; i < HEAT_W * HEAT_H; i++) {
      const v = grid[i] / maxVal;
      if (v > 0.005) {
        const t   = Math.min(v, 1);
        const idx = Math.min(t * 9, 8.999);
        const lo  = Math.floor(idx), hi = lo + 1;
        const f   = idx - lo;
        const C   = VIRIDIS;
        const r   = Math.round(C[lo][0] + f * (C[hi][0] - C[lo][0]));
        const g   = Math.round(C[lo][1] + f * (C[hi][1] - C[lo][1]));
        const b   = Math.round(C[lo][2] + f * (C[hi][2] - C[lo][2]));
        const a   = Math.max(40, Math.floor(Math.pow(v, 0.5) * 210));
        d[i*4]=r; d[i*4+1]=g; d[i*4+2]=b; d[i*4+3]=a;
      }
    }
    ctx.putImageData(id, 0, 0);
  }, [heatGrids, classType]);

  return (
    <canvas ref={canvasRef} width={200} height={150}
      style={{ width, height, borderRadius: '4px', imageRendering: 'pixelated' }} />
  );
}

// ── 서브 컴포넌트 ────────────────────────────────────────────────────────────

function Section({ title, children }) {
  return (
    <div className="card p-4">
      <p className="section-title">{title.replace(/^[^\s]*\s/, '')}</p>
      {children}
    </div>
  );
}

function KpiCard({ label, value, sub, color = '#6d5ce7' }) {
  return (
    <div className="bg-surface rounded-xl p-3 border border-border/60"
      style={{ boxShadow: '0 1px 2px rgba(16,24,40,0.04)' }}>
      <div className="text-xs text-text-muted mb-1">{label}</div>
      <div className="text-lg font-bold" style={{ color }}>{value}</div>
      {sub && <div className="text-xs text-text-muted mt-0.5">{sub}</div>}
    </div>
  );
}

function fmtWon(v) {
  if (v >= 10000000) return `₩${(v / 10000000).toFixed(1)}천만`;
  if (v >= 100000)   return `₩${Math.round(v / 10000)}만`;
  if (v >= 1000)     return `₩${Math.round(v / 1000)}K`;
  return `₩${v.toLocaleString()}`;
}

function dlCSV(filename, content) {
  const blob = new Blob(['\uFEFF' + content], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function ValidationCard({ label, metric, value, color, sub }) {
  return (
    <div className="bg-surface rounded-xl p-3 border border-border/60 text-center"
      style={{ boxShadow: '0 1px 2px rgba(16,24,40,0.04)' }}>
      <div className="text-xs text-text-muted mb-1">{label}</div>
      <div className="text-xl font-bold" style={{ color }}>{value}</div>
      <div className="text-xs text-text-muted mt-1">{metric}</div>
      {sub && <div className="text-xs text-text-muted mt-0.5">{sub}</div>}
    </div>
  );
}

function ConfCell({ value, label, color }) {
  return (
    <div style={{ background: color, borderRadius: '4px', padding: '8px', textAlign: 'center' }}>
      <div className="text-base font-bold text-white">{value.toLocaleString()}</div>
      <div className="text-xs text-white/70">{label}</div>
    </div>
  );
}

// ── 메인 컴포넌트 ────────────────────────────────────────────────────────────

export default function BatchAnalytics({ data, view = 'data' }) {
  const metrics = STORE_METRICS[data?.storeType] ?? metricsIcecream;
  const totalDays = data.dailyStats?.length || 120;

  const kpis = useMemo(() => computeKPIs(data), [data]);
  const {
    overallConvRate, zoneKPIs, classKPIs, hourlyAgg, dowAgg,
    dailySalesChart, nextMonthForecast, bestHours, bestZone, worstZone, insights,
  } = kpis;

  // ── 모델 검증 (4개월 → 3/1 분할) ──
  const [validationResults, setValidationResults] = useState(null);
  const [validationLoading, setValidationLoading] = useState(false);
  useEffect(() => {
    let cancelled = false;
    setValidationResults(null);
    if (view !== 'validation' || totalDays < 60) {
      setValidationLoading(false);
      return;
    }
    setValidationLoading(true);
    const timer = setTimeout(() => {
      try {
        const result = runFullValidation(data);
        if (!cancelled) setValidationResults(result);
      } finally {
        if (!cancelled) setValidationLoading(false);
      }
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [data, totalDays, view]);

  // ── AI 예측 모델 state ──
  // 활성 구역: zoneStats에 실제 데이터가 있는 구역만 (매장 유형에 따라 다름)
  const activeZoneIds = useMemo(() =>
    Object.entries(data.zoneStats)
      .filter(([, zs]) => zs.visitors > 0)
      .map(([id]) => id),
    [data.zoneStats]
  );

  const [lrClassType, setLrClassType] = useState('adult_female');
  const [lrDwellTime, setLrDwellTime] = useState(4.0);
  const [lrHour, setLrHour]           = useState(2);

  const lrModel = useMemo(() => trainLogisticRegression({ ...data, activeZones: activeZoneIds }, data.storeType), [data, activeZoneIds]);
  const lrPredictions = useMemo(
    () => predictAllZones(lrModel, lrClassType, lrDwellTime, lrHour, activeZoneIds),
    [lrModel, lrClassType, lrDwellTime, lrHour, activeZoneIds]
  );
  const banditResult = useMemo(() => runUCB1Bandit(data.zoneStats, 500), [data.zoneStats]);

  // ── LSTM / DQN 딥러닝 모델 state ──
  const [lstmResult,   setLstmResult]   = useState(null);
  const [dqnResult,    setDqnResult]    = useState(null);
  const [lstmLoading,  setLstmLoading]  = useState(false);
  const [dqnLoading,   setDqnLoading]   = useState(false);
  const [lstmProgress, setLstmProgress] = useState(0);
  const [dqnProgress,  setDqnProgress]  = useState(0);
  const [aiError,      setAiError]      = useState(null);

  // LSTM 검증 결과 (validation 탭)
  const [lstmValidation, setLstmValidation] = useState(null);
  const [lstmValLoading, setLstmValLoading] = useState(false);
  const [lstmValProgress, setLstmValProgress] = useState(0);

  // 데이터가 바뀌면 이전 결과 리셋 (재학습이 필요한 상태로)
  useEffect(() => {
    setLstmResult(null);
    setDqnResult(null);
    setLstmValidation(null);
    setAiError(null);
  }, [data]);

  // 학습 트리거 핸들러 — 사용자가 버튼을 눌러 명시적으로 시작
  const trainLSTM = useCallback(async () => {
    if (data.dailyStats?.length < 15) return;
    setLstmLoading(true);
    setLstmProgress(0);
    setAiError(null);
    try {
      const { trainLSTMForecaster } = await import('./models/lstmForecaster.js');
      const result = await trainLSTMForecaster(data.dailyStats, {
        horizon: 14,
        onProgress: p => setLstmProgress(p),
      });
      setLstmResult(result);
    } catch (e) {
      setAiError('LSTM 학습 실패: ' + (e.message || e));
    } finally {
      setLstmLoading(false);
    }
  }, [data]);

  const trainDQN = useCallback(async () => {
    if (data.dailyStats?.length < 15) return;
    setDqnLoading(true);
    setDqnProgress(0);
    setAiError(null);
    try {
      const { trainDQNOptimizer } = await import('./models/dqnOptimizer.js');
      const result = await trainDQNOptimizer(data, {
        episodes: 300,
        onProgress: p => setDqnProgress(p),
      });
      setDqnResult(result);
    } catch (e) {
      setAiError('DQN 학습 실패: ' + (e.message || e));
    } finally {
      setDqnLoading(false);
    }
  }, [data]);

  const trainBothAI = useCallback(async () => {
    await Promise.all([trainLSTM(), trainDQN()]);
  }, [trainLSTM, trainDQN]);

  const runLSTMValid = useCallback(async () => {
    if (data.dailyStats?.length < 15) return;
    setLstmValLoading(true);
    setLstmValProgress(0);
    try {
      const r = await runLSTMValidation(data, p => setLstmValProgress(p));
      setLstmValidation(r);
    } catch (e) {
      // swallow
    } finally {
      setLstmValLoading(false);
    }
  }, [data]);

  // ── 신규 예측 모델 ──
  const revenueModel = useMemo(() => trainRevenuePredictor(data), [data]);
  const revPredictions = useMemo(
    () => revenueModel.predictForZone(lrClassType, lrDwellTime, lrHour, activeZoneIds),
    [revenueModel, lrClassType, lrDwellTime, lrHour, activeZoneIds]
  );
  const hourlyDemand = useMemo(() => predictHourlyDemand(data), [data]);
  const customerValue = useMemo(() => computeCustomerValueAnalysis(data), [data]);
  const crossSell = useMemo(() => predictCrossSell(data), [data]);

  const heatGridsRestored = useMemo(() => {
    if (!data.heatGrids) return null;
    return Object.fromEntries(
      Object.entries(data.heatGrids).map(([cls, buf]) => [cls, new Float32Array(buf)])
    );
  }, [data.heatGrids]);

  // 구역별 구매전환율 BarChart 데이터
  const zoneConvData = [...zoneKPIs].sort((a, b) => b.convRate - a.convRate)
    .map(z => ({ name: z.label, 전환율: +(z.convRate * 100).toFixed(1), 체류: +z.avgDwell.toFixed(1) }));

  // 비효율 구역 (TOP 3)
  const ineffTop3 = [...zoneKPIs].sort((a, b) => b.inefficiency - a.inefficiency).slice(0, 3);
  const newProdTop3 = [...zoneKPIs].sort((a, b) => b.newProductScore - a.newProductScore).slice(0, 3);

  // CSV 다운로드 핸들러
  const downloadKpiCSV = () => {
    const header = '구역,방문객수,구매고객수,구매전환율(%),평균체류시간(초),총매출,구매고객비중(%),체류대비효율,비효율지수\n';
    const rows = zoneKPIs.map(z =>
      `${z.label},${z.visitors},${z.buyers},${(z.convRate*100).toFixed(1)},${z.avgDwell.toFixed(1)},${z.revenue},${(z.buyerShare*100).toFixed(1)},${z.efficiency.toFixed(4)},${(z.inefficiency*100).toFixed(1)}`
    ).join('\n');
    dlCSV('batch_zone_kpi.csv', header + rows);
  };

  const downloadClassCSV = () => {
    const header = '분류,분류명,방문객수,구매고객수,구매전환율(%),총매출,평균구매액\n';
    const rows = classKPIs.map(c =>
      `${c.cls},${c.label},${c.visitors},${c.buyers},${(c.convRate*100).toFixed(1)},${c.revenue},${Math.round(c.avgBasket)}`
    ).join('\n');
    dlCSV('batch_class_kpi.csv', header + rows);
  };

  const downloadPurchaseCSV = () => {
    const opHours = data.operatingHours || 8;
    const csvStartHour = Math.max(6, 22 - opHours);
    const header = '에이전트ID,고객분류,고객분류명,구역ID,구역명,상품명,가격,체류시간,일차,시간대\n';
    const rows = data.purchases.map(p =>
      `${p.agentId},${p.classType},${CUSTOMER_CLASSES[p.classType].label},${p.zoneId},${p.zoneLabel},${p.item},${p.price},${p.dwellTime.toFixed(1)},${p.day+1},${(p.hour + csvStartHour)}시`
    ).join('\n');
    dlCSV('batch_purchases.csv', header + rows);
  };

  return (
    <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3 bg-bg">

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* ── 데이터 분석 뷰 ── */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {view === 'data' && (<>

      {/* ── 1행: 요약 KPI 카드 ── */}
      <Section title={`📊 ${totalDays}일 핵심 KPI 요약`}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px' }}>
          <KpiCard label="총 방문객" value={`${data.totalVisitors.toLocaleString()}명`} color="#a78bfa" />
          <KpiCard label="전체 구매전환율"
            value={`${(overallConvRate * 100).toFixed(1)}%`}
            color={overallConvRate > 0.6 ? '#40cc80' : overallConvRate > 0.3 ? '#ffaa44' : '#ff6060'} />
          <KpiCard label="평균 체류시간"
            value={`${(zoneKPIs.reduce((s, z) => s + z.avgDwell, 0) / zoneKPIs.length).toFixed(1)}초`}
            color="#c080ff" />
          <KpiCard label="효율 최고 구역"
            value={bestZone?.label ?? '-'}
            sub={`전환효율 ${bestZone ? (bestZone.efficiency * 100).toFixed(2) : '-'}`}
            color="#40cc80" />
          <KpiCard label="개선 필요 구역"
            value={worstZone?.label ?? '-'}
            sub={`비효율 지수 ${worstZone ? (worstZone.inefficiency * 100).toFixed(0) : '-'}`}
            color="#ff6060" />
        </div>
      </Section>

      {/* ── 2행: 히트맵 ── */}
      <Section title={`🗺 고객 동선 히트맵 (${totalDays}일 누적)`}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px' }}>
          {['combined', ...Object.keys(CUSTOMER_CLASSES)].map(cls => (
            <div key={cls} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
              <div style={{ fontSize: '10px', color: cls === 'combined' ? '#374151' : CUSTOMER_CLASSES[cls]?.color, fontWeight: 'bold' }}>
                {cls === 'combined' ? '통합' : CUSTOMER_CLASSES[cls].short}
              </div>
              {heatGridsRestored && <HeatmapMini heatGrids={heatGridsRestored} classType={cls} width={140} height={105} />}
            </div>
          ))}
        </div>
      </Section>

      {/* ── 3행: 구역별 전환율 + 체류 시간 ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <Section title="🏪 구역별 구매전환율 (%)">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={zoneConvData} margin={{ top: 0, right: 8, left: -10, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={TICK_STYLE} angle={-30} textAnchor="end" interval={0} />
              <YAxis tick={TICK_STYLE} tickFormatter={v => `${v}%`} />
              <Tooltip {...TOOLTIP_STYLE} formatter={v => [`${v}%`, '구매전환율']} />
              <Bar dataKey="전환율" radius={[6, 6, 0, 0]}>
                {zoneConvData.map((_, i) => <Cell key={i} fill={softAt(i)} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Section>

        <Section title="⏱ 구역별 평균 체류시간 vs 전환율">
          <ResponsiveContainer width="100%" height={200}>
            <ScatterChart margin={{ top: 10, right: 20, left: -10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="체류" name="체류시간" type="number" tick={TICK_STYLE}
                label={{ value: '체류(초)', position: 'insideBottom', fill: '#6b7280', fontSize: 10, dy: 12 }} />
              <YAxis dataKey="전환율" name="전환율" type="number" tick={TICK_STYLE} tickFormatter={v => `${v}%`} />
              <ZAxis range={[120, 120]} />
              <Tooltip {...TOOLTIP_STYLE}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const p = payload[0]?.payload;
                  return (
                    <div style={{ ...TOOLTIP_STYLE.contentStyle, padding: '6px 10px' }}>
                      <div style={{ color: '#1f2937', fontWeight: 'bold' }}>{p?.name}</div>
                      <div>체류: {p?.체류}초</div>
                      <div>전환율: {p?.전환율}%</div>
                    </div>
                  );
                }}
              />
              <Scatter data={zoneConvData} fill={OURS} fillOpacity={0.9} stroke="#6d5ce7" strokeWidth={1} />
            </ScatterChart>
          </ResponsiveContainer>
        </Section>
      </div>

      {/* ── 4행: 시간대별 + 고객군별 전환율 ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <Section title="🕐 시간대별 구매전환율">
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={hourlyAgg} margin={{ top: 5, right: 15, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="label" tick={TICK_STYLE} />
              <YAxis tick={TICK_STYLE} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
              <Tooltip {...TOOLTIP_STYLE} formatter={v => [`${(v * 100).toFixed(1)}%`, '전환율']} />
              <Line type="monotone" dataKey="convRate" stroke="#6d5ce7" strokeWidth={2} dot={{ fill: '#6d5ce7', r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </Section>

        <Section title="👥 고객군별 구매전환율">
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={classKPIs.map(c => ({ name: c.short, 전환율: +(c.convRate * 100).toFixed(1), cls: c.cls }))}
              margin={{ top: 5, right: 15, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={TICK_STYLE} />
              <YAxis tick={TICK_STYLE} tickFormatter={v => `${v}%`} />
              <Tooltip {...TOOLTIP_STYLE} formatter={v => [`${v}%`, '전환율']} />
              <Bar dataKey="전환율" radius={[6, 6, 0, 0]}>
                {classKPIs.map((c, i) => <Cell key={i} fill={CLASS_PURPLE[c.cls]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Section>
      </div>

      {/* ── 5행: 일별 매출 추이 + 요일별 패턴 ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '12px' }}>
        <Section title="📈 일별 매출 추이 (30일)">
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={dailySalesChart} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="day" tick={TICK_STYLE} tickFormatter={v => `${v}일`} interval={4} />
              <YAxis tick={TICK_STYLE} tickFormatter={v => fmtWon(v)} />
              <Tooltip {...TOOLTIP_STYLE} formatter={(v, name) => [name === 'revenue' ? fmtWon(v) : `${v}%`, name === 'revenue' ? '매출' : '전환율']} />
              <Line type="monotone" dataKey="revenue" stroke="#6d5ce7" strokeWidth={2} dot={false} name="매출" />
            </LineChart>
          </ResponsiveContainer>
        </Section>

        <Section title="📅 요일별 평균 매출">
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={dowAgg} margin={{ top: 5, right: 8, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="dow" tick={TICK_STYLE} />
              <YAxis tick={TICK_STYLE} tickFormatter={v => fmtWon(v)} />
              <Tooltip {...TOOLTIP_STYLE} formatter={v => [fmtWon(v), '평균 매출']} />
              <Bar dataKey="avgRevenue" radius={[6, 6, 0, 0]}>
                {dowAgg.map((_, i) => <Cell key={i} fill={i >= 5 ? WEEKEND : OURS} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Section>
      </div>

      {/* ── 6행: 고객군별 매출 파이 + 구역별 ROI ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <Section title="💰 고객군별 매출 비중">
          <ResponsiveContainer width="100%" height={210}>
            <PieChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
              <Pie data={classKPIs.map(c => ({ name: c.label, value: c.revenue, cls: c.cls }))}
                dataKey="value" nameKey="name" cx="50%" cy="45%" outerRadius={56}
                label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
                labelLine={false}>
                {classKPIs.map((c, i) => <Cell key={i} fill={CLASS_PURPLE[c.cls]} />)}
              </Pie>
              <Tooltip {...TOOLTIP_STYLE} formatter={v => [fmtWon(v), '매출']} />
              <Legend verticalAlign="bottom" height={22} iconSize={8} wrapperStyle={{ fontSize: 10, color: '#9ca3af' }} />
            </PieChart>
          </ResponsiveContainer>
        </Section>

        <Section title="🏪 매대별 ROI (총 매출)">
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={[...zoneKPIs].sort((a, b) => b.revenue - a.revenue).map(z => ({ name: z.label, revenue: z.revenue }))}
              layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis type="number" tick={TICK_STYLE} tickFormatter={v => fmtWon(v)} />
              <YAxis type="category" dataKey="name" tick={{ fill: '#6b7280', fontSize: 9 }} width={52} />
              <Tooltip {...TOOLTIP_STYLE} formatter={v => [fmtWon(v), '총 매출']} />
              <Bar dataKey="revenue" fill={OURS} radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Section>
      </div>

      {/* ── 7행: 비효율 구역 TOP3 + 신상품 배치 적합도 TOP3 ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <Section title="⚠️ 비효율 구역 TOP3 (개선 우선순위)">
          {ineffTop3.map((z, i) => (
            <div key={z.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 8px', marginTop: '5px', background: '#fff1f2', borderRadius: '4px', borderLeft: `3px solid ${i === 0 ? '#ff4444' : i === 1 ? '#ff8844' : '#ffaa44'}` }}>
              <div>
                <div style={{ fontSize: '12px', color: '#1f2937', fontWeight: 'bold' }}>{i + 1}. {z.label}</div>
                <div style={{ fontSize: '10px', color: '#6b7280', marginTop: '2px' }}>
                  체류 {z.avgDwell.toFixed(1)}초 · 전환율 {(z.convRate * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{ fontSize: '11px', color: '#ff8888', fontWeight: 'bold' }}>비효율 {(z.inefficiency * 100).toFixed(0)}</div>
            </div>
          ))}
        </Section>

        <Section title="✨ 신상품 배치 적합 구역 TOP3">
          {newProdTop3.map((z, i) => (
            <div key={z.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 8px', marginTop: '5px', background: '#f0fdf4', borderRadius: '4px', borderLeft: `3px solid ${i === 0 ? '#40cc80' : i === 1 ? '#4aaa60' : '#4a8850'}` }}>
              <div>
                <div style={{ fontSize: '12px', color: '#1f2937', fontWeight: 'bold' }}>{i + 1}. {z.label}</div>
                <div style={{ fontSize: '10px', color: '#6b7280', marginTop: '2px' }}>
                  유입률 {(z.entryRate * 100).toFixed(1)}% · 전환율 {(z.convRate * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{ fontSize: '11px', color: '#40cc80', fontWeight: 'bold' }}>적합도 {(z.newProductScore * 100).toFixed(1)}</div>
            </div>
          ))}
        </Section>
      </div>

      {/* ── 8행: 인사이트 ── */}
      <Section title="💡 배치 기반 추천 인사이트">
        {insights.map((ins, i) => (
          <div key={i} style={{ padding: '8px 10px', marginTop: '6px', background: '#efedfd', borderRadius: '6px', borderLeft: '3px solid #6d5ce7', fontSize: '12px', lineHeight: '1.6', color: '#374151' }}>
            {ins}
          </div>
        ))}
      </Section>

      {/* ── 10행: 데이터 다운로드 ── */}
      <Section title="📥 배치 데이터 다운로드">
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {[
            { label: '📊 구역별 KPI CSV',   fn: downloadKpiCSV },
            { label: '👥 고객군별 KPI CSV', fn: downloadClassCSV },
            { label: '📋 전체 구매 내역 CSV', fn: downloadPurchaseCSV },
          ].map(({ label, fn }) => (
            <button key={label} onClick={fn} className="btn-primary text-xs py-1.5 px-3">
              {label}
            </button>
          ))}
        </div>
      </Section>

      </>)}

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* ── AI 예측 뷰 ── */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {view === 'ai' && <AIPredictionPanel data={data} kpis={kpis} />}

      {false && (<>

      {/* ── 11행: AI 예측 모델 ── */}
      <Section title="🤖 AI 예측 모델">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>

          {/* ── 좌: Logistic Regression ── */}
          <div style={{ background: '#f9fafb', borderRadius: '6px', padding: '12px', border: '1px solid #e5e7eb', color: '#374151' }}>
            <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#374151', marginBottom: '10px' }}>
              📈 구매 전환율 예측 (Logistic Regression)
            </div>

            {/* 컨트롤 영역 */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px', marginBottom: '12px' }}>
              {/* 고객분류 */}
              <div>
                <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '3px' }}>고객분류</div>
                <select value={lrClassType} onChange={e => setLrClassType(e.target.value)}
                  className="w-full bg-white border border-border rounded px-2 py-1 text-xs text-text">
                  {CLASS_KEYS.map(cls => (
                    <option key={cls} value={cls}>{CUSTOMER_CLASSES[cls].label}</option>
                  ))}
                </select>
              </div>
              {/* 체류시간 */}
              <div>
                <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '3px' }}>체류시간: {lrDwellTime.toFixed(1)}초</div>
                <input type="range" min="0.5" max="10" step="0.5" value={lrDwellTime}
                  onChange={e => setLrDwellTime(parseFloat(e.target.value))}
                  className="w-full accent-accent" />
              </div>
              {/* 시간대 */}
              <div>
                <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '3px' }}>시간대</div>
                <select value={lrHour} onChange={e => setLrHour(parseInt(e.target.value))}
                  className="w-full bg-white border border-border rounded px-2 py-1 text-xs text-text">
                  {kpis.HOUR_LABELS.map((label, i) => (
                    <option key={i} value={i}>{label}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* 예측 결과 BarChart (가로 방향) */}
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={lrPredictions.map(p => ({
                name: p.label,
                확률: +(p.probability * 100).toFixed(1),
              }))} layout="vertical" margin={{ top: 0, right: 30, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis type="number" domain={[0, 100]} tick={TICK_STYLE} tickFormatter={v => `${v}%`} />
                <YAxis type="category" dataKey="name" tick={TICK_STYLE} width={70} />
                <Tooltip {...TOOLTIP_STYLE} formatter={v => [`${v}%`, '구매 확률']} />
                <Bar dataKey="확률" radius={[0, 6, 6, 0]}>
                  {lrPredictions.map((p, i) => (
                    <Cell key={i} fill={`hsl(255,65%,${Math.max(42, 75 - p.probability * 35)}%)`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>

            {/* 모델 성능 지표 */}
            {(() => {
              const acc   = lrModel.accuracy * 100;
              const base  = (lrModel.baselineAcc || 0) * 100;
              const diff  = acc - base;
              const aucOk = lrModel.auc >= 0.55;
              const diffColor = diff >= 0 ? '#40cc80' : '#ff6060';
              return (
                <div style={{ marginTop: '8px', fontSize: '10px', color: '#6b7280', textAlign: 'center', lineHeight: '1.8' }}>
                  <div>
                    정확도: <span style={{ color: '#40cc80', fontWeight: 'bold' }}>{acc.toFixed(1)}%</span>
                    {base > 0 && (
                      <span style={{ color: '#9ca3af', marginLeft: '4px' }}>
                        (baseline {base.toFixed(1)}%
                        <span style={{ color: diffColor }}> {diff >= 0 ? '+' : ''}{diff.toFixed(1)}pp</span>)
                      </span>
                    )}
                  </div>
                  <div>
                    AUC: <span style={{ color: aucOk ? '#40cc80' : '#ffaa44', fontWeight: 'bold' }}>{lrModel.auc.toFixed(3)}</span>
                    {!aucOk && <span style={{ color: '#ffaa44', marginLeft: '4px' }}>⚠️ 신호 부족</span>}
                    {' | '}
                    샘플: <span style={{ color: '#a78bfa' }}>{data.purchases.length > 0 ? '충분' : '부족'}</span>
                  </div>
                </div>
              );
            })()}
          </div>

          {/* ── 우: UCB1 Bandit ── */}
          <div style={{ background: '#f9fafb', borderRadius: '6px', padding: '12px', border: '1px solid #e5e7eb', color: '#374151' }}>
            <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#374151', marginBottom: '10px' }}>
              🎰 신상품 최적 배치 (UCB1 Multi-Armed Bandit)
            </div>

            {/* 추천 카드 */}
            <div style={{ background: '#efedfd', borderRadius: '6px', padding: '10px 14px', marginBottom: '12px', border: '1px solid #e5e7eb', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '10px', color: '#6b7280' }}>🏆 신상품 최적 배치 구역</div>
                <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#40cc80', marginTop: '2px' }}>
                  {banditResult.recommendation.label}
                </div>
              </div>
              <span className={`badge text-xs font-bold ${banditResult.recommendation.confidence > 70 ? 'bg-green-100 text-ok' : banditResult.recommendation.confidence > 40 ? 'bg-amber-100 text-warn' : 'bg-red-100 text-danger'}`}>
                신뢰도 {banditResult.recommendation.confidence}%
              </span>
            </div>

            {/* 구역별 UCB Score BarChart */}
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={[...banditResult.arms].sort((a, b) => b.Q - a.Q).map(a => ({
                  name: a.label,
                  Q값: +(a.Q * 100).toFixed(1),
                  탐험횟수: a.n,
                  isRec: a.zoneId === banditResult.recommendation.zoneId,
                }))}
                layout="vertical" margin={{ top: 0, right: 40, left: 10, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis type="number" tick={TICK_STYLE} tickFormatter={v => `${v}%`} />
                <YAxis type="category" dataKey="name" tick={TICK_STYLE} width={70} />
                <Tooltip {...TOOLTIP_STYLE} formatter={(v, name) => {
                  if (name === 'Q값') return [`${v}%`, '평균 보상'];
                  return [v, name];
                }} />
                <Bar dataKey="Q값" radius={[0, 6, 6, 0]}>
                  {[...banditResult.arms].sort((a, b) => b.Q - a.Q).map((a, i) => (
                    <Cell key={i} fill={a.zoneId === banditResult.recommendation.zoneId ? '#16a34a' : '#d1d5db'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>

            {/* 수렴 차트 */}
            {banditResult.convergenceData.length > 0 && (
              <div style={{ marginTop: '8px' }}>
                <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '4px' }}>탐색 수렴 과정 (최적 arm Q값)</div>
                <ResponsiveContainer width="100%" height={100}>
                  <LineChart data={banditResult.convergenceData} margin={{ top: 4, right: 10, left: 0, bottom: 0 }}>
                    <XAxis dataKey="trial" tick={TICK_STYLE} />
                    <YAxis tick={TICK_STYLE} domain={[0, 'auto']} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip {...TOOLTIP_STYLE} formatter={v => [`${(v * 100).toFixed(1)}%`, 'Best Q']} />
                    <Line type="monotone" dataKey="bestArmQ" stroke="#40cc80" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* arm별 상세 정보 */}
            <div style={{ marginTop: '8px', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '4px' }}>
              {[...banditResult.arms].sort((a, b) => b.Q - a.Q).slice(0, 4).map(a => (
                <div key={a.zoneId} style={{ background: '#f3f4f6', borderRadius: '4px', padding: '6px', fontSize: '10px', textAlign: 'center',
                  border: a.zoneId === banditResult.recommendation.zoneId ? '1px solid #16a34a' : '1px solid #e5e7eb' }}>
                  <div style={{ color: '#6b7280', marginBottom: '2px' }}>{a.label}</div>
                  <div style={{ color: '#1f2937', fontWeight: 'bold' }}>Q: {(a.Q * 100).toFixed(1)}%</div>
                  <div style={{ color: '#9ca3af' }}>n={a.n}</div>
                </div>
              ))}
            </div>
          </div>

        </div>
      </Section>

      {/* ── 12행: 매대별 예상 매출 예측 ── */}
      <Section title="💰 매대별 예상 매출액 예측 (Linear Regression)">
        <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '8px' }}>
          위 구매전환율 예측과 동일한 조건 (고객분류·체류시간·시간대) 기반으로 매대별 기대 매출액을 예측합니다.
          {revenueModel.r2 > 0 && <span> | R² = <span style={{ color: '#40cc80', fontWeight: 'bold' }}>{revenueModel.r2.toFixed(3)}</span></span>}
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={revPredictions.map(p => ({ name: p.label, 예상매출: p.expectedRevenue }))}
            layout="vertical" margin={{ top: 0, right: 40, left: 10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis type="number" tick={TICK_STYLE} tickFormatter={v => fmtWon(v)} />
            <YAxis type="category" dataKey="name" tick={TICK_STYLE} width={70} />
            <Tooltip {...TOOLTIP_STYLE} formatter={v => [fmtWon(v), '예상 매출']} />
            <Bar dataKey="예상매출" radius={[0, 6, 6, 0]}>
              {revPredictions.map((_, i) => (
                <Cell key={i} fill={softAt(i)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Section>

      {/* ── 13행: 시간대별 수요 예측 ── */}
      <Section title="🕐 시간대별 수요 예측 (이동평균 + 트렌드)">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div>
            <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '6px' }}>예측 방문자 vs 실제 평균</div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={hourlyDemand} margin={{ top: 5, right: 15, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="label" tick={TICK_STYLE} />
                <YAxis tick={TICK_STYLE} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Bar dataKey="avgVisitors" fill={BASE} name="실제 평균" radius={[6, 6, 0, 0]} />
                <Bar dataKey="predictedVisitors" fill={OURS} name="예측(우리)" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '6px' }}>예측 매출 vs 실제 평균</div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={hourlyDemand} margin={{ top: 5, right: 15, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="label" tick={TICK_STYLE} />
                <YAxis tick={TICK_STYLE} tickFormatter={v => fmtWon(v)} />
                <Tooltip {...TOOLTIP_STYLE} formatter={(v, name) => [fmtWon(v), name]} />
                <Bar dataKey="avgRevenue" fill={BASE} name="실제 평균" radius={[6, 6, 0, 0]} />
                <Bar dataKey="predictedRevenue" fill={OURS} name="예측(우리)" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(90px, 1fr))', gap: '4px', marginTop: '8px' }}>
          {hourlyDemand.map(h => (
            <div key={h.hour} style={{ background: '#f3f4f6', borderRadius: '4px', padding: '5px', fontSize: '10px', textAlign: 'center',
              border: h.trend === 'up' ? '1px solid #bbf7d0' : h.trend === 'down' ? '1px solid #fecaca' : '1px solid #e5e7eb' }}>
              <div style={{ color: '#6b7280' }}>{h.label}</div>
              <div style={{ color: h.trend === 'up' ? '#16a34a' : h.trend === 'down' ? '#dc2626' : '#9ca3af', fontWeight: 'bold' }}>
                {h.trend === 'up' ? '▲' : h.trend === 'down' ? '▼' : '─'} {(h.predictedConvRate * 100).toFixed(0)}%
              </div>
              <div style={{ color: '#9ca3af' }}>신뢰 {h.confidence}%</div>
            </div>
          ))}
        </div>
      </Section>

      {/* ── 14행: 고객군별 가치 분석 ── */}
      <Section title="👑 고객 가치 분석 (CLV 세그먼트)">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
          {customerValue.map(cv => (
            <div key={cv.classType} style={{ background: '#f9fafb', borderRadius: '6px', padding: '12px', border: `1px solid ${cv.color}33`,
              borderTop: `3px solid ${cv.color}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <div style={{ fontSize: '12px', fontWeight: 'bold', color: cv.color }}>{cv.short}</div>
                <span className={`badge text-xs font-bold ${cv.segment === 'VIP' ? 'bg-green-100 text-ok' : cv.segment === 'loyal' ? 'bg-blue-100 text-accent' : cv.segment === 'casual' ? 'bg-amber-100 text-warn' : 'bg-red-100 text-danger'}`}>
                  {cv.segment.toUpperCase()}
                </span>
              </div>
              <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#1f2937', marginBottom: '6px' }}>
                LTV ₩{cv.lifetimeValue.toLocaleString()}
              </div>
              <div style={{ fontSize: '10px', color: '#6b7280', lineHeight: '1.8' }}>
                <div>객단가: ₩{cv.avgBasket.toLocaleString()}</div>
                <div>전환율: {(cv.convRate * 100).toFixed(1)}%</div>
                <div>평균 방문 매대: {cv.avgZonesVisited}개</div>
                <div>매출 기여: {(cv.revenueShare * 100).toFixed(1)}%</div>
                <div>피크: {cv.peakHour.label}</div>
              </div>
              {cv.preferredZones.length > 0 && (
                <div style={{ marginTop: '6px', fontSize: '10px' }}>
                  <div style={{ color: '#9ca3af', marginBottom: '3px' }}>선호 매대</div>
                  {cv.preferredZones.map((z, i) => (
                    <div key={z.zoneId} style={{ color: '#6b7280', display: 'flex', justifyContent: 'space-between' }}>
                      <span>{i + 1}. {z.label}</span>
                      <span style={{ color: '#6d5ce7' }}>{z.count}건</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </Section>

      {/* ── 15행: 교차 구매 분석 ── */}
      <Section title="🔗 교차 구매 확률 분석">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          {/* 좌: Top 교차 구매 쌍 */}
          <div>
            <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '6px' }}>교차 구매 확률 TOP 5 (P(B|A))</div>
            {crossSell.topPairs.map((pair, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 8px', marginTop: '4px',
                background: '#faf9fe', borderRadius: '6px', borderLeft: `3px solid hsl(255,${60 - i * 6}%,${62 - i * 5}%)` }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '11px', color: '#1f2937', fontWeight: 'bold' }}>
                    {pair.labelA} → {pair.labelB}
                  </div>
                  <div style={{ fontSize: '10px', color: '#6b7280' }}>{pair.count}건 동시 구매</div>
                </div>
                <div style={{ fontSize: '14px', fontWeight: 'bold', color: `hsl(255,65%,${Math.max(38, 72 - Math.round(pair.probability * 40))}%)` }}>
                  {(pair.probability * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>

          {/* 우: 교차구매 히트맵 (텍스트 매트릭스) */}
          <div>
            <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '6px' }}>매대 간 교차구매 확률 매트릭스 (%)</div>
            <div style={{ overflowX: 'auto' }}>
              {(() => {
                const SHORT = { ice1:'I-1', ice2:'I-2', ice3:'I-3', ice4:'I-4', ice5:'I-5', ice6:'I-6', beverage:'음료', snack1:'S-1', snack2:'S-2' };
                const shortLabel = z => SHORT[z] || z;
                return (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '10px' }}>
                <thead>
                  <tr style={{ background: 'linear-gradient(135deg, #6d5ce7, #8b7bf0)' }}>
                    <th style={{ padding: '4px 6px', color: 'rgba(255,255,255,0.8)', textAlign: 'left', fontWeight: 600 }}>A＼B</th>
                    {crossSell.activeZones.map(z => (
                      <th key={z} style={{ padding: '4px 6px', color: '#fff', textAlign: 'center', whiteSpace: 'nowrap', fontWeight: 600 }}>
                        {shortLabel(z)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {crossSell.activeZones.map((zA, rowIdx) => (
                    <tr key={zA} style={{ background: rowIdx % 2 === 1 ? '#faf9fe' : '#fff' }}>
                      <td style={{ padding: '3px 6px', color: '#6d5ce7', fontWeight: 'bold', whiteSpace: 'nowrap' }}>
                        {shortLabel(zA)}
                      </td>
                      {crossSell.activeZones.map(zB => {
                        const val = zA === zB ? null : (crossSell.matrix[zA]?.[zB] || 0);
                        const pct = val !== null ? (val * 100) : 0;
                        return (
                          <td key={zB} style={{
                            padding: '3px', textAlign: 'center',
                            background: val === null ? '#f3f4f6' : `rgba(109, 92, 231, ${Math.min(pct / 50, 0.65)})`,
                            color: val === null ? '#d1d5db' : pct > 20 ? '#fff' : '#4c3bb5',
                            fontWeight: pct > 30 ? 'bold' : 'normal',
                          }}>
                            {val === null ? '-' : pct.toFixed(0)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
                );
              })()}
            </div>
            {/* 교차 구매 인사이트 */}
            {crossSell.insights.length > 0 && (
              <div style={{ marginTop: '8px' }}>
                {crossSell.insights.map((ins, i) => (
                  <div key={i} style={{ padding: '5px 8px', marginTop: '3px', background: '#efedfd', borderRadius: '3px', fontSize: '10px', color: '#6b7280', borderLeft: '2px solid #6d5ce7' }}>
                    {ins}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </Section>

      {/* ── AI 딥러닝 KPI 예측 패널 ── */}
      <AIPredictionPanel data={data} kpis={kpis} />

      </>)}

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* ── 모델 검증 뷰 ── */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {view === 'validation' && (<>

      {/* ── AI 딥러닝 KPI 예측 모델 검증 (숨김 시작) ── */}
      {false && (<>

      {/* ── Row 1: 요약 카드 ── */}
      <Section title={`🔬 모델 검증 요약 (학습 ${validationResults.trainDays}일 → 검증 ${validationResults.testDays}일)`}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', marginBottom: '8px' }}>
          <ValidationCard label="Logistic Regression" metric="ACC"
            value={`${(validationResults.logistic.test.accuracy * 100).toFixed(1)}%`}
            sub={`AUC ${validationResults.logistic.test.auc.toFixed(3)} · 기준 ${(validationResults.logistic.test.baselineAcc * 100).toFixed(1)}%`}
            color={validationResults.logistic.test.accuracy > 0.80 ? '#40cc80' : validationResults.logistic.test.accuracy > 0.65 ? '#ffaa44' : '#ff6060'} />
          <ValidationCard label="Linear Regression" metric="R²" value={validationResults.linear.test.r2.toFixed(3)}
            color={validationResults.linear.test.r2 > 0.4 ? '#40cc80' : validationResults.linear.test.r2 > 0.2 ? '#ffaa44' : '#ff6060'} />
          <ValidationCard label="UCB1 Bandit" metric="추천 순위" value={`${validationResults.ucb1.rank}/${validationResults.ucb1.totalZones}위`}
            color={validationResults.ucb1.isHit ? '#40cc80' : validationResults.ucb1.isTop3 ? '#ffaa44' : '#ff6060'} />
          <ValidationCard label="시간대별 수요" metric="MAPE" value={`${validationResults.hourly.overallMAPE}%`}
            color={validationResults.hourly.overallMAPE < 20 ? '#40cc80' : validationResults.hourly.overallMAPE < 35 ? '#ffaa44' : '#ff6060'} />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
          <ValidationCard label="고객 가치 분석" metric="세그먼트 안정성" value={`${(validationResults.customerValue.segmentStability * 100).toFixed(0)}%`}
            color={validationResults.customerValue.segmentStability > 0.75 ? '#40cc80' : validationResults.customerValue.segmentStability > 0.5 ? '#ffaa44' : '#ff6060'} />
          <ValidationCard label="교차 구매" metric="상관계수" value={validationResults.crossSell.matrixCorrelation.toFixed(3)}
            color={validationResults.crossSell.matrixCorrelation > 0.7 ? '#40cc80' : validationResults.crossSell.matrixCorrelation > 0.4 ? '#ffaa44' : '#ff6060'} />
          <ValidationCard label="Transformer (KPI #1)" metric="구역 전환율 MAPE" value="4.0%"
            sub="ρ = 1.000 · 검증 20일" color="#40cc80" />
          <ValidationCard label="Crossformer (KPI #3)" metric="구역 체류시간 MAPE" value="8.6%"
            sub="ρ = 0.964 · 검증 20일" color="#40cc80" />
        </div>
      </Section>

      {/* ── Row 2: Logistic Regression 상세 ── */}
      <Section title="📈 Logistic Regression — 구매전환율 예측 검증">
        <div style={{ background: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: '6px', padding: '9px 11px', marginBottom: '10px', fontSize: '11px', color: '#6b7280', lineHeight: 1.55 }}>
          검증 임계값은 학습 구간에서 F1 기준으로 자동 보정했습니다
          <span style={{ color: '#1f2937', fontWeight: 'bold' }}> (threshold {validationResults.logistic.threshold.toFixed(2)})</span>.
          Test 기준 정확도는 단순 다수 클래스 기준
          <span style={{ color: '#1f2937', fontWeight: 'bold' }}> {(validationResults.logistic.test.baselineAcc * 100).toFixed(1)}%</span>
          대비 <span style={{ color: '#40cc80', fontWeight: 'bold' }}>{validationResults.logistic.test.lift.toFixed(2)}x</span> 입니다.
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
          {/* Confusion Matrix */}
          <div>
            <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '8px' }}>Confusion Matrix (Test Set)</div>
            <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr 1fr', gridTemplateRows: 'auto 1fr 1fr', gap: '2px' }}>
              <div />
              <div style={{ textAlign: 'center', fontSize: '10px', color: '#6b7280', padding: '4px' }}>예측: 구매</div>
              <div style={{ textAlign: 'center', fontSize: '10px', color: '#6b7280', padding: '4px' }}>예측: 미구매</div>
              <div style={{ fontSize: '10px', color: '#6b7280', padding: '8px', display: 'flex', alignItems: 'center' }}>실제: 구매</div>
              <ConfCell value={validationResults.logistic.test.confusion.tp} label="TP" color="#16a34a" />
              <ConfCell value={validationResults.logistic.test.confusion.fn} label="FN" color="#3a2a1a" />
              <div style={{ fontSize: '10px', color: '#6b7280', padding: '8px', display: 'flex', alignItems: 'center' }}>실제: 미구매</div>
              <ConfCell value={validationResults.logistic.test.confusion.fp} label="FP" color="#dc2626" />
              <ConfCell value={validationResults.logistic.test.confusion.tn} label="TN" color="#6d5ce7" />
            </div>
          </div>

          {/* Train vs Test 비교 */}
          <div>
            <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '8px' }}>Train vs Test 성능 비교</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={[
                { name: 'Accuracy', train: +(validationResults.logistic.train.accuracy * 100).toFixed(1), test: +(validationResults.logistic.test.accuracy * 100).toFixed(1) },
                { name: 'AUC', train: +(validationResults.logistic.train.auc * 100).toFixed(1), test: +(validationResults.logistic.test.auc * 100).toFixed(1) },
                { name: 'Precision', train: +(validationResults.logistic.train.precision * 100).toFixed(1), test: +(validationResults.logistic.test.precision * 100).toFixed(1) },
                { name: 'Recall', train: +(validationResults.logistic.train.recall * 100).toFixed(1), test: +(validationResults.logistic.test.recall * 100).toFixed(1) },
                { name: 'F1', train: +(validationResults.logistic.train.f1 * 100).toFixed(1), test: +(validationResults.logistic.test.f1 * 100).toFixed(1) },
              ]} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="name" tick={TICK_STYLE} />
                <YAxis tick={TICK_STYLE} tickFormatter={v => `${v}%`} domain={[0, 100]} />
                <Tooltip {...TOOLTIP_STYLE} formatter={v => [`${v}%`]} />
                <Bar dataKey="train" fill={BASE} name="Train(기준)" radius={[6, 6, 0, 0]} />
                <Bar dataKey="test" fill={OURS} name="Test(우리)" radius={[6, 6, 0, 0]} />
                <Legend wrapperStyle={{ fontSize: '10px', color: '#6b7280' }} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </Section>

      {/* ── Row 3: Linear Regression 상세 ── */}
      <Section title="💰 Linear Regression — 매출액 예측 검증">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
          {/* 예측 vs 실제 산점도 */}
          <div>
            <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '8px' }}>예측 vs 실제 매출 (Test Set, 샘플 {validationResults.linear.scatterData.length}건)</div>
            <ResponsiveContainer width="100%" height={220}>
              <ScatterChart margin={{ top: 10, right: 20, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="actual" name="실제" type="number" tick={TICK_STYLE}
                  label={{ value: '실제 가격', position: 'insideBottom', fill: '#6b7280', fontSize: 10, dy: 12 }} />
                <YAxis dataKey="predicted" name="예측" type="number" tick={TICK_STYLE}
                  label={{ value: '예측 가격', position: 'insideLeft', fill: '#6b7280', fontSize: 10, angle: -90 }} />
                <ZAxis range={[20, 20]} />
                <Tooltip {...TOOLTIP_STYLE} content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const p = payload[0]?.payload;
                  return (
                    <div style={{ ...TOOLTIP_STYLE.contentStyle, padding: '6px 10px' }}>
                      <div>실제: {fmtWon(p?.actual || 0)}</div>
                      <div>예측: {fmtWon(p?.predicted || 0)}</div>
                    </div>
                  );
                }} />
                <Scatter data={validationResults.linear.scatterData} fill={OURS} fillOpacity={0.9} stroke="#6d5ce7" strokeWidth={1} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* 지표 테이블 */}
          <div>
            <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '8px' }}>성능 지표 비교</div>
            <table className="data-table" style={{ fontSize: '11px' }}>
              <thead>
                <tr>
                  <th>지표</th>
                  <th className="col-base">Train(기준)</th>
                  <th className="col-ours">Test(우리)</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { name: 'R²', train: validationResults.linear.train.r2.toFixed(3), test: validationResults.linear.test.r2.toFixed(3) },
                  { name: 'MAE', train: '-', test: fmtWon(validationResults.linear.test.mae) },
                  { name: 'MAPE', train: '-', test: `${validationResults.linear.test.mape}%` },
                  { name: 'RMSE', train: '-', test: fmtWon(validationResults.linear.test.rmse) },
                ].map(row => (
                  <tr key={row.name}>
                    <td style={{ color: '#374151', borderBottom: '1px solid #ececf3' }}>{row.name}</td>
                    <td className="col-base" style={{ borderBottom: '1px solid #ececf3' }}>{row.train}</td>
                    <td className="col-ours" style={{ borderBottom: '1px solid #ececf3' }}>{row.test}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Section>

      {/* ── Row 4: UCB1 상세 ── */}
      <Section title="🎰 UCB1 Multi-Armed Bandit — 신상품 배치 검증">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
          {/* Train Q값 vs Test 실제전환율 */}
          <div>
            <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '8px' }}>구역별 Train Q값 vs Test 실제 전환율</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={validationResults.ucb1.comparisonData} margin={{ top: 5, right: 20, left: -10, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="zone" tick={TICK_STYLE} angle={-30} textAnchor="end" interval={0} />
                <YAxis tick={TICK_STYLE} tickFormatter={v => `${v}%`} />
                <Tooltip {...TOOLTIP_STYLE} formatter={v => [`${v}%`]} />
                <Bar dataKey="trainQ" fill={BASE} name="Train Q값" radius={[6, 6, 0, 0]} />
                <Bar dataKey="testActual" fill={OURS} name="Test 실제" radius={[6, 6, 0, 0]} />
                <Legend wrapperStyle={{ fontSize: '10px', color: '#6b7280' }} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* 텍스트 요약 */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{ background: '#efedfd', borderRadius: '6px', padding: '12px', border: '1px solid #e5e7eb' }}>
              <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '6px' }}>추천 결과</div>
              <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#40cc80' }}>
                추천: {validationResults.ucb1.recommendation.label}
              </div>
              <div style={{ fontSize: '11px', color: '#6b7280', marginTop: '4px' }}>
                Test 기간 실제 순위: <span style={{ color: validationResults.ucb1.isHit ? '#40cc80' : validationResults.ucb1.isTop3 ? '#ffaa44' : '#ff6060', fontWeight: 'bold' }}>
                  {validationResults.ucb1.rank}위 / {validationResults.ucb1.totalZones}개 구역
                </span>
              </div>
              {validationResults.ucb1.bestTestZone && (
                <div style={{ fontSize: '11px', color: '#6b7280', marginTop: '4px' }}>
                  Test 기간 실제 1위: <span style={{ color: '#1f2937' }}>{validationResults.ucb1.bestTestZone.label} ({validationResults.ucb1.bestTestZone.rate}%)</span>
                </div>
              )}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              <KpiCard label="Regret" value={`${(validationResults.ucb1.regret * 100).toFixed(2)}%p`}
                color={validationResults.ucb1.regret < 0.02 ? '#40cc80' : '#ffaa44'} />
              <KpiCard label="Q-실제 상관" value={validationResults.ucb1.correlation.toFixed(3)}
                color={validationResults.ucb1.correlation > 0.7 ? '#40cc80' : validationResults.ucb1.correlation > 0.4 ? '#ffaa44' : '#ff6060'} />
              <KpiCard label="성과비" value={`${(validationResults.ucb1.performanceRatio * 100).toFixed(1)}%`}
                color={validationResults.ucb1.performanceRatio > 0.9 ? '#40cc80' : '#ffaa44'} />
              <KpiCard label="적중" value={validationResults.ucb1.isHit ? 'HIT' : 'MISS'}
                color={validationResults.ucb1.isHit ? '#40cc80' : '#ff6060'} />
            </div>
          </div>
        </div>
      </Section>

      {/* ── Row 5: Hourly Demand 상세 ── */}
      <Section title="🕐 시간대별 수요 예측 검증">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
          <div>
            <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '8px' }}>
              시간대별 방문자: 예측 vs 실제 (MAE: {validationResults.hourly.visitorMAE}, MAPE: {validationResults.hourly.visitorMAPE}%)
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={validationResults.hourly.hourlyComparison} margin={{ top: 5, right: 15, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="label" tick={TICK_STYLE} />
                <YAxis tick={TICK_STYLE} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Bar dataKey="predictedVisitors" fill={OURS} name="예측" radius={[6, 6, 0, 0]} />
                <Bar dataKey="actualVisitors" fill={BASE} name="실제" radius={[6, 6, 0, 0]} />
                <Legend wrapperStyle={{ fontSize: '10px', color: '#6b7280' }} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '8px' }}>
              시간대별 매출: 예측 vs 실제 (MAE: {fmtWon(validationResults.hourly.revenueMAE)}, MAPE: {validationResults.hourly.revenueMAPE}%)
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={validationResults.hourly.hourlyComparison} margin={{ top: 5, right: 15, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="label" tick={TICK_STYLE} />
                <YAxis tick={TICK_STYLE} tickFormatter={v => fmtWon(v)} />
                <Tooltip {...TOOLTIP_STYLE} formatter={(v, name) => [fmtWon(v), name]} />
                <Bar dataKey="predictedRevenue" fill={OURS} name="예측" radius={[6, 6, 0, 0]} />
                <Bar dataKey="actualRevenue" fill={BASE} name="실제" radius={[6, 6, 0, 0]} />
                <Legend wrapperStyle={{ fontSize: '10px', color: '#6b7280' }} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </Section>

      {/* ── Row 6: Customer Value 상세 ── */}
      <Section title="👑 고객 가치 분석 검증 (세그먼트 안정성)">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
          {validationResults.customerValue.comparison.map(cv => (
            <div key={cv.classType} style={{ background: '#f9fafb', borderRadius: '6px', padding: '12px',
              border: `1px solid ${cv.color}33`, borderTop: `3px solid ${cv.color}` }}>
              <div style={{ fontSize: '12px', fontWeight: 'bold', color: cv.color, marginBottom: '8px' }}>{cv.short}</div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                <span style={{
                  fontSize: '10px', padding: '2px 6px', borderRadius: '8px', fontWeight: 'bold',
                  background: cv.trainSegment === 'VIP' ? '#dcfce7' : cv.trainSegment === 'loyal' ? '#dbeafe' : cv.trainSegment === 'casual' ? '#fef9c3' : '#fee2e2',
                  color: cv.trainSegment === 'VIP' ? '#16a34a' : cv.trainSegment === 'loyal' ? '#2563eb' : cv.trainSegment === 'casual' ? '#a16207' : '#dc2626',
                }}>{cv.trainSegment}</span>
                <span style={{ color: '#9ca3af', fontSize: '12px' }}>→</span>
                <span style={{
                  fontSize: '10px', padding: '2px 6px', borderRadius: '8px', fontWeight: 'bold',
                  background: cv.testSegment === 'VIP' ? '#dcfce7' : cv.testSegment === 'loyal' ? '#dbeafe' : cv.testSegment === 'casual' ? '#fef9c3' : '#fee2e2',
                  color: cv.testSegment === 'VIP' ? '#16a34a' : cv.testSegment === 'loyal' ? '#2563eb' : cv.testSegment === 'casual' ? '#a16207' : '#dc2626',
                }}>{cv.testSegment}</span>
                {cv.segmentStable
                  ? <span style={{ fontSize: '10px', color: '#40cc80' }}>✓ 일치</span>
                  : <span style={{ fontSize: '10px', color: '#ff6060' }}>✗ 변동</span>}
              </div>

              <div style={{ fontSize: '10px', color: '#6b7280', lineHeight: '1.8' }}>
                <div>Train LTV: ₩{cv.trainLTV.toLocaleString()}</div>
                <div>Test LTV: ₩{cv.testLTV.toLocaleString()} ({cv.ltvDelta >= 0 ? '+' : ''}{cv.ltvDelta.toLocaleString()})</div>
                <div>매출기여 변화: {(cv.revenueShareDelta * 100).toFixed(1)}%p</div>
              </div>
            </div>
          ))}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', marginTop: '10px' }}>
          <KpiCard label="세그먼트 안정성" value={`${(validationResults.customerValue.segmentStability * 100).toFixed(0)}%`}
            sub={`${CLASS_KEYS.length}개 중 ${Math.round(validationResults.customerValue.segmentStability * CLASS_KEYS.length)}개 일치`}
            color={validationResults.customerValue.segmentStability > 0.75 ? '#40cc80' : '#ffaa44'} />
          <KpiCard label="LTV 순위 상관" value={validationResults.customerValue.ltvRankCorrelation.toFixed(3)}
            color={validationResults.customerValue.ltvRankCorrelation > 0.7 ? '#40cc80' : '#ffaa44'} />
          <KpiCard label="선호매대 일치율" value={`${(validationResults.customerValue.avgZoneOverlap * 100).toFixed(0)}%`}
            color={validationResults.customerValue.avgZoneOverlap > 0.6 ? '#40cc80' : '#ffaa44'} />
        </div>
      </Section>

      {/* ── Row 7: Cross-Sell 상세 ── */}
      <Section title="🔗 교차 구매 예측 검증">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
          {/* Top 5 쌍 비교 */}
          <div>
            <div style={{ fontSize: '10px', color: '#6b7280', marginBottom: '8px' }}>Top-5 교차구매 쌍: Train vs Test 확률</div>
            {validationResults.crossSell.pairComparison.map((pair, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 8px', marginTop: '4px',
                background: '#f3f4f6', borderRadius: '4px', borderLeft: `3px solid ${pair.inTestTop5 ? '#16a34a' : '#d1d5db'}` }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '11px', color: '#1f2937' }}>{pair.labelA} ↔ {pair.labelB}</div>
                </div>
                <div style={{ fontSize: '11px', color: '#6b7280', textAlign: 'right' }}>
                  <span style={{ color: '#4c3bb5' }}>T:</span> {(pair.trainProb * 100).toFixed(1)}%
                  <span style={{ margin: '0 4px', color: '#3a3a5a' }}>→</span>
                  <span style={{ color: '#6d5ce7' }}>V:</span> {(pair.testProb * 100).toFixed(1)}%
                </div>
                {pair.inTestTop5 && <span style={{ fontSize: '10px', color: '#40cc80' }}>✓</span>}
              </div>
            ))}
          </div>

          {/* 지표 요약 */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              <KpiCard label="매트릭스 상관계수" value={validationResults.crossSell.matrixCorrelation.toFixed(3)}
                color={validationResults.crossSell.matrixCorrelation > 0.7 ? '#40cc80' : validationResults.crossSell.matrixCorrelation > 0.4 ? '#ffaa44' : '#ff6060'} />
              <KpiCard label="Top-5 쌍 일치율" value={`${(validationResults.crossSell.top5OverlapRate * 100).toFixed(0)}%`}
                color={validationResults.crossSell.top5OverlapRate > 0.6 ? '#40cc80' : '#ffaa44'} />
              <KpiCard label="평균 확률 편차" value={`${(validationResults.crossSell.avgDelta * 100).toFixed(2)}%p`}
                color={validationResults.crossSell.avgDelta < 0.05 ? '#40cc80' : '#ffaa44'} />
              <KpiCard label="최대 확률 편차" value={`${(validationResults.crossSell.maxDelta * 100).toFixed(2)}%p`}
                color={validationResults.crossSell.maxDelta < 0.1 ? '#40cc80' : '#ffaa44'} />
            </div>
            <div style={{ background: '#efedfd', borderRadius: '6px', padding: '10px', border: '1px solid #e5e7eb', fontSize: '11px', color: '#6b7280', lineHeight: '1.8' }}>
              <div style={{ color: '#6b7280', fontWeight: 'bold', marginBottom: '4px' }}>해석</div>
              {validationResults.crossSell.matrixCorrelation > 0.7
                ? <div>✓ Train과 Test 교차구매 패턴이 높은 일관성을 보임 → 매대 배치 전략의 신뢰도 높음</div>
                : validationResults.crossSell.matrixCorrelation > 0.4
                ? <div>△ 보통 수준의 일관성 → 추가 데이터 수집으로 패턴 안정화 필요</div>
                : <div>✗ 낮은 일관성 → 교차구매 패턴이 불안정하여 배치 전략 수립 시 주의 필요</div>}
            </div>
          </div>
        </div>
      </Section>

      </>)}

      {/* ── AI 딥러닝 KPI 예측 모델 검증 ── */}
      <Section title="AI 딥러닝 KPI 예측 모델 검증">
        <p className="text-xs text-text-muted mb-4">
          학습 1~84일 → 검증 85~104일 (20일) 분할.
          상세 검증 결과는 <a href="/model" target="_blank" className="text-accent hover:underline">모델 우수성 페이지 (/model)</a>에서 확인하세요.
        </p>

        <div className="grid grid-cols-2 gap-4 mb-4">
          {/* Transformer */}
          <div className="bg-gray-50 rounded-lg p-4 border border-border" style={{ borderTop: '3px solid #6d5ce7' }}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-bold text-accent">Transformer (KPI #1, #6)</span>
              <span className="badge bg-accent-light text-accent">epoch {metrics.transformer.training.earlyStoppingEpoch} · ★ Best</span>
            </div>
            <div className="grid grid-cols-3 gap-2 mb-3">
              {[
                { label: '구역 MAPE', value: `${metrics.transformer.metrics.zoneMAPE}%`, color: '#16a34a' },
                { label: '시간대 MAPE', value: `${metrics.transformer.metrics.hourMAPE}%`, color: '#16a34a' },
                { label: '구역 ρ', value: metrics.transformer.metrics.zoneSpearmanRho.toFixed(3), color: '#16a34a' },
              ].map(m => (
                <div key={m.label} className="bg-white rounded p-2 text-center border border-border">
                  <div className="text-xs text-text-muted mb-1">{m.label}</div>
                  <div className="text-base font-bold" style={{ color: m.color }}>{m.value}</div>
                </div>
              ))}
            </div>
            <div className="text-xs text-text-muted">
              Baseline 대비 <span className="text-ok font-semibold">+{metrics.transformer.metrics.baselineImprovement}%</span>
              {' · '}시간대 ρ {metrics.transformer.metrics.hourSpearmanRho.toFixed(3)}
            </div>
          </div>

          {/* Crossformer */}
          <div className="bg-gray-50 rounded-lg p-4 border border-border" style={{ borderTop: '3px solid #7c3aed' }}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-bold" style={{ color: '#7c3aed' }}>Crossformer (KPI #3, #7, #8)</span>
              <span className="badge bg-violet-50 text-violet-700">epoch {metrics.crossformer.training.earlyStoppingEpoch}</span>
            </div>
            <div className="grid grid-cols-3 gap-2 mb-3">
              {[
                { label: '체류 MAPE', value: `${metrics.crossformer.metrics.kpi3DwellMAPE}%`, color: '#16a34a' },
                { label: '효율 MAPE', value: `${metrics.crossformer.metrics.kpi7EfficiencyMAPE}%`, color: '#16a34a' },
                { label: '비효율 ρ', value: metrics.crossformer.metrics.kpi8InefficientRho.toFixed(3), color: '#d97706' },
              ].map(m => (
                <div key={m.label} className="bg-white rounded p-2 text-center border border-border">
                  <div className="text-xs text-text-muted mb-1">{m.label}</div>
                  <div className="text-base font-bold" style={{ color: m.color }}>{m.value}</div>
                </div>
              ))}
            </div>
            <div className="text-xs text-text-muted">
              고객군 순위 ρ {metrics.crossformer.metrics.kpi5CustomerRho.toFixed(3)}
              {' · '}체류 ρ {metrics.crossformer.metrics.kpi3DwellRho}
            </div>
          </div>
        </div>

        <div>
          <p className="text-xs text-text-muted mb-2">KPI별 MAPE 비교 — 검증셋 기준 (낮을수록 우수)</p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart
              data={[
                { kpi: 'KPI#1 구역전환율', mape: metrics.transformer.metrics.zoneMAPE, fill: '#6d5ce7' },
                { kpi: 'KPI#6 시간대전환율', mape: metrics.transformer.metrics.hourMAPE, fill: '#a78bfa' },
                { kpi: 'KPI#3 체류시간', mape: metrics.crossformer.metrics.kpi3DwellMAPE, fill: '#7c3aed' },
                { kpi: 'KPI#7 전환효율', mape: metrics.crossformer.metrics.kpi7EfficiencyMAPE, fill: '#9333ea' },
              ]}
              layout="vertical" margin={{ top: 0, right: 60, left: 10, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis type="number" tick={TICK_STYLE} tickFormatter={v => `${v}%`} domain={[0, 15]} />
              <YAxis type="category" dataKey="kpi" tick={TICK_STYLE} width={110} />
              <Tooltip {...TOOLTIP_STYLE} formatter={v => [`${v}%`, 'MAPE']} />
              <Bar dataKey="mape" radius={[0, 6, 6, 0]}>
                {[
                  { fill: '#6d5ce7' }, { fill: '#a78bfa' },
                  { fill: '#7c3aed' }, { fill: '#9333ea' },
                ].map((c, i) => <Cell key={i} fill={c.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      </>)}

    </div>
  );
}
