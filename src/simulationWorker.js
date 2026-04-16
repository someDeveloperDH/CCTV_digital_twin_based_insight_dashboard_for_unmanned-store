// ═══════════════════════════════════════════════════════════════════════════
// 배치 시뮬레이션 Web Worker  (렌더링 없이 30일 데이터 고속 생성)
// ═══════════════════════════════════════════════════════════════════════════

const HEAT_W = 200;
const HEAT_H = 150;

const CLASS_KEYS = ['adult_male', 'adult_female', 'minor_male', 'minor_female'];
const CUSTOMER_CLASSES = {
  adult_male:   { label: '성년 남성' },
  adult_female: { label: '성년 여성' },
  minor_male:   { label: '미성년 남성' },
  minor_female: { label: '미성년 여성' },
};

const ZONES = [
  { id: 'snack',      label: '과자/스낵',  x:  20, y:  50, w: 165, h: 70,
    preferred: ['minor_male', 'minor_female'],
    items: [
      { name: '새우깡', price: 1500 }, { name: '포카칩', price: 1800 },
      { name: '초코파이', price: 2500 }, { name: '오레오', price: 2200 },
      { name: '프링글스', price: 2800 }, { name: '허니버터칩', price: 2000 },
      { name: '꼬깔콘', price: 1200 }, { name: '빠다코코낫', price: 1500 },
      { name: '젤리', price: 1000 }, { name: '사탕', price: 800 },
    ],
  },
  { id: 'beverage',   label: '음료',       x: 210, y:  50, w: 165, h: 70,
    preferred: ['adult_male', 'adult_female', 'minor_male', 'minor_female'],
    items: [
      { name: '콜라', price: 1500 }, { name: '사이다', price: 1500 },
      { name: '커피', price: 2000 }, { name: '주스', price: 2500 },
      { name: '이온음료', price: 1800 }, { name: '녹차', price: 1200 },
      { name: '에너지드링크', price: 2200 }, { name: '우유', price: 1800 },
      { name: '두유', price: 1500 }, { name: '탄산수', price: 1200 },
    ],
  },
  { id: 'daily',      label: '생활용품',   x: 400, y:  50, w: 165, h: 70,
    preferred: ['adult_male', 'adult_female'],
    items: [
      { name: '샴푸', price: 8000 }, { name: '치약', price: 3500 },
      { name: '세제', price: 5000 }, { name: '비누', price: 2000 },
      { name: '칫솔', price: 3000 }, { name: '수건', price: 5000 },
      { name: '휴지', price: 4500 }, { name: '주방세제', price: 4000 },
      { name: '핸드크림', price: 6000 }, { name: '바디로션', price: 8000 },
    ],
  },
  { id: 'cosmetic',   label: '화장품',     x: 590, y:  50, w: 165, h: 70,
    preferred: ['adult_female'],
    items: [
      { name: '립스틱', price: 15000 }, { name: '파운데이션', price: 25000 },
      { name: '마스카라', price: 12000 }, { name: '아이섀도', price: 18000 },
      { name: '블러셔', price: 14000 }, { name: '컨실러', price: 16000 },
      { name: 'BB크림', price: 18000 }, { name: '선크림', price: 20000 },
      { name: '립밤', price: 8000 }, { name: '아이라이너', price: 12000 },
    ],
  },
  { id: 'stationery', label: '문구',       x:  20, y: 180, w: 165, h: 70,
    preferred: ['minor_male', 'minor_female'],
    items: [
      { name: '색연필', price: 3000 }, { name: '노트', price: 2000 },
      { name: '펜', price: 1000 }, { name: '스티커', price: 1500 },
      { name: '형광펜', price: 1500 }, { name: '지우개', price: 500 },
      { name: '자', price: 1000 }, { name: '풀', price: 1200 },
      { name: '테이프', price: 1500 }, { name: '스케치북', price: 4000 },
    ],
  },
  { id: 'frozen',     label: '냉동식품',   x: 210, y: 180, w: 165, h: 70,
    preferred: ['adult_male', 'adult_female'],
    items: [
      { name: '아이스크림', price: 2000 }, { name: '만두', price: 5000 },
      { name: '냉동피자', price: 8000 }, { name: '냉동치킨', price: 9000 },
      { name: '군만두', price: 6000 }, { name: '냉동볶음밥', price: 4500 },
      { name: '냉동새우', price: 12000 }, { name: '어묵', price: 3000 },
      { name: '핫도그', price: 2500 }, { name: '냉동라면', price: 3500 },
    ],
  },
  { id: 'premium',    label: '프리미엄',   x: 400, y: 180, w: 165, h: 70,
    preferred: ['adult_male', 'adult_female'],
    items: [
      { name: '와인', price: 25000 }, { name: '위스키', price: 45000 },
      { name: '프리미엄커피', price: 12000 }, { name: '수입과자', price: 8000 },
      { name: '샴페인', price: 35000 }, { name: '럼', price: 28000 },
      { name: '브랜디', price: 42000 }, { name: '수입초콜릿', price: 15000 },
      { name: '수입치즈', price: 18000 }, { name: '수입와인', price: 32000 },
    ],
  },
  { id: 'bakery',     label: '베이커리',   x: 590, y: 180, w: 165, h: 70,
    preferred: ['minor_female', 'adult_female'],
    items: [
      { name: '크로아상', price: 3000 }, { name: '케이크', price: 15000 },
      { name: '샌드위치', price: 4500 }, { name: '마카롱', price: 5000 },
      { name: '베이글', price: 2500 }, { name: '소금빵', price: 2000 },
      { name: '시나몬롤', price: 3500 }, { name: '머핀', price: 2500 },
      { name: '치즈케이크', price: 6000 }, { name: '에클레어', price: 4000 },
    ],
  },
];

// 히트맵 가우시안 커널 (구역 방문 위치 기반 – 실시간보다 넓은 반경)
const GRAV = 5, GSIG = 2.5;
const GAUSS_OFFSETS = [];
for (let dy = -GRAV; dy <= GRAV; dy++)
  for (let dx = -GRAV; dx <= GRAV; dx++)
    GAUSS_OFFSETS.push({ dx, dy, w: Math.exp(-(dx * dx + dy * dy) / (2 * GSIG * GSIG)) });

// ── 헬퍼 ────────────────────────────────────────────────────────────────────

function pickClass(ratios) {
  const total = Object.values(ratios).reduce((a, b) => a + b, 0);
  if (total === 0) return CLASS_KEYS[0];
  let r = Math.random() * total;
  for (const [cls, w] of Object.entries(ratios)) { r -= w; if (r <= 0) return cls; }
  return CLASS_KEYS[0];
}

function buildQueue(classType, activeZones) {
  const zones   = activeZones || ZONES;
  const pref    = zones.filter(z => z.preferred.includes(classType));
  const nonPref = zones.filter(z => !z.preferred.includes(classType));
  const shuffle = arr => [...arr].sort(() => Math.random() - 0.5);
  const nPref   = 1 + Math.floor(Math.random() * Math.min(pref.length, 3));
  const nNon    = Math.min(nonPref.length, 1 + Math.floor(Math.random() * 2));
  return [...shuffle(pref).slice(0, nPref), ...shuffle(nonPref).slice(0, nNon)];
}

function addHeatBlob(grid, canvasX, canvasY, intensity) {
  const hcx = Math.floor(canvasX / 4);
  const hcy = Math.floor(canvasY / 4);
  for (const { dx, dy, w } of GAUSS_OFFSETS) {
    const nx = hcx + dx, ny = hcy + dy;
    if (nx >= 0 && nx < HEAT_W && ny >= 0 && ny < HEAT_H)
      grid[ny * HEAT_W + nx] += intensity * w;
  }
}

// ── 배치 시뮬 분산 헬퍼 ─────────────────────────────────────────────────────

// 정규분포 난수 (Box-Muller)
function gaussRandom(mean, stddev) {
  let u, v, s;
  do { u = Math.random() * 2 - 1; v = Math.random() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
  return mean + stddev * u * Math.sqrt(-2 * Math.log(s) / s);
}

// 일별 방문자 변동 배율 (요일 패턴 + 랜덤 노이즈)
function dailyMultiplier(dayIdx, weekendBoost) {
  const dow = dayIdx % 7; // 0=월 ... 6=일
  const dowFactors = [0.90, 0.95, 1.00, 1.00, 1.10, weekendBoost, weekendBoost * 0.95];
  const noise = gaussRandom(1.0, 0.15);
  return Math.max(0.4, dowFactors[dow] * noise);
}

// 시간대별 도착 가중치 (점심+오후 피크)
function buildArrivalCurve(hours) {
  return Array.from({ length: hours }, (_, h) => {
    const t = hours > 1 ? h / (hours - 1) : 0.5;
    return 0.5 + 0.8 * Math.exp(-((t - 0.3) ** 2) / 0.04)
             + 0.6 * Math.exp(-((t - 0.7) ** 2) / 0.06);
  });
}

function pickHourWeighted(weights, totalWeight) {
  let r = Math.random() * totalWeight;
  for (let h = 0; h < weights.length; h++) { r -= weights[h]; if (r <= 0) return h; }
  return weights.length - 1;
}

// ── Worker 메시지 핸들러 ──────────────────────────────────────────────────────

self.onmessage = function ({ data }) {
  if (data.type === 'run')   runBatch(data.config, data.days || 30);
  if (data.type === 'abort') self.close();
};

// ── 배치 시뮬레이션 ───────────────────────────────────────────────────────────

function runBatch(config, days) {
  const SIM_HOURS_PER_DAY = config.operatingHours || 8;
  const baseDailyVisitors = config.dailyVisitors  || 500;
  const weekendBoost      = config.weekendBoost   || 1.15;
  const dwellScale        = config.dwellScale     || 1.0;
  const purchaseBonus     = config.purchaseBonus  ?? 0;

  // 매장 유형에 따른 활성 구역 필터링
  const availableZoneIds  = config.availableZones || ZONES.map(z => z.id);
  const activeZones       = ZONES.filter(z => availableZoneIds.includes(z.id));

  // 시간대별 도착 가중치
  const arrivalWeights = buildArrivalCurve(SIM_HOURS_PER_DAY);
  const totalArrivalW  = arrivalWeights.reduce((a, b) => a + b, 0);

  // 구역별 통계
  const zoneStats = {};
  ZONES.forEach(z => {
    zoneStats[z.id] = {
      label: z.label,
      visitors: 0, buyers: 0, totalDwell: 0, revenue: 0,
      visitorsByClass: Object.fromEntries(CLASS_KEYS.map(k => [k, 0])),
      buyersByClass:   Object.fromEntries(CLASS_KEYS.map(k => [k, 0])),
    };
  });

  // 고객군별 통계 (buyers = 고유 구매자 수)
  const classStats = {};
  const classBuyerSets = {};
  CLASS_KEYS.forEach(cls => {
    classStats[cls] = { visitors: 0, buyers: 0, revenue: 0, purchaseCnt: 0 };
    classBuyerSets[cls] = new Set();
  });

  // 일별 통계
  const dailyStats = Array.from({ length: days }, (_, i) => ({
    day: i, dayOfWeek: i % 7,
    visitors: 0, revenue: 0, purchaseCnt: 0,
    convRate: 0, avgBasket: 0,
    revenueByZone:  Object.fromEntries(ZONES.map(z => [z.id, 0])),
    revenueByClass: Object.fromEntries(CLASS_KEYS.map(k => [k, 0])),
    _buyerSet: new Set(),
  }));

  // 시간대별 통계 (days × hours)
  const BUCKETS = days * SIM_HOURS_PER_DAY;
  const hVisitors = new Int32Array(BUCKETS);
  const hBuyers   = new Int32Array(BUCKETS);
  const hRevenue  = new Float32Array(BUCKETS);

  // 히트맵
  const heatGrids = {};
  CLASS_KEYS.forEach(cls => { heatGrids[cls] = new Float32Array(HEAT_W * HEAT_H); });

  let uid = 0;
  const allPurchases = [];
  const totalBuyerSet = new Set();

  const startTime = Date.now();
  let lastPct = -1;

  // ── 일별 루프 (dailyVisitors 기반 + 일별 분산) ──────────────────────────
  for (let dayIdx = 0; dayIdx < days; dayIdx++) {
    const mult = dailyMultiplier(dayIdx, weekendBoost);
    const todayVisitors = Math.round(baseDailyVisitors * mult);

    for (let v = 0; v < todayVisitors; v++) {
      // 시간대별 도착 분포에 따라 시간 배정
      const hourInDay = pickHourWeighted(arrivalWeights, totalArrivalW);
      const bucket    = dayIdx * SIM_HOURS_PER_DAY + hourInDay;

      // 에이전트 생성
      const classType = pickClass(config.classRatios);
      const agentId   = ++uid;
      const minor     = classType.startsWith('minor');

      classStats[classType].visitors++;
      dailyStats[dayIdx].visitors++;
      hVisitors[bucket]++;

      // 구역 방문 처리 (통계적 즉시 처리 – 물리 시뮬 없음)
      // ★ per-zone 기저확률을 낮춰서 복합확률이 프리셋 목표 전환율에 근접하도록 보정
      //    avg 3.5구역 방문 기준:
      //    convenience(+0.18)→~86%, supermarket(+0.04)→~69%,
      //    mall(-0.18)→~36%, school(+0.12)→~80%
      const zones = buildQueue(classType, activeZones);
      let purchased = false;

      for (const z of zones) {
        const dwell = (minor ? 1 + Math.random() * 3 : 3 + Math.random() * 5)
                      * dwellScale;

        zoneStats[z.id].visitors++;
        zoneStats[z.id].totalDwell += dwell;
        zoneStats[z.id].visitorsByClass[classType]++;

        const isPref  = z.preferred.includes(classType);
        const buyProb = Math.min(0.85, Math.max(0.02,
          (isPref ? 0.35 : 0.08) + purchaseBonus));

        if (Math.random() < buyProb) {
          const item  = z.items[Math.floor(Math.random() * z.items.length)];
          const price = item.price;

          allPurchases.push({ agentId, classType, zoneId: z.id, zoneLabel: z.label,
                              item: item.name, price, dwellTime: dwell, day: dayIdx, hour: hourInDay });

          zoneStats[z.id].buyers++;
          zoneStats[z.id].buyersByClass[classType]++;
          zoneStats[z.id].revenue += price;
          classStats[classType].purchaseCnt++;
          classStats[classType].revenue += price;
          dailyStats[dayIdx].purchaseCnt++;
          dailyStats[dayIdx].revenue += price;
          dailyStats[dayIdx].revenueByZone[z.id]      += price;
          dailyStats[dayIdx].revenueByClass[classType] += price;
          dailyStats[dayIdx]._buyerSet.add(agentId);
          hRevenue[bucket] += price;
          purchased = true;
        }

        // 히트맵: 구역 아래쪽 방문 위치에 블롭
        const visitX = z.x + z.w / 2 + (Math.random() - 0.5) * z.w * 0.4;
        const visitY = z.y + z.h + 18 + Math.random() * 12;
        addHeatBlob(heatGrids[classType], visitX, visitY, dwell * 0.08);
      }

      if (purchased) {
        hBuyers[bucket]++;
        totalBuyerSet.add(agentId);
        classBuyerSets[classType].add(agentId);
      }
    }

    // 진행률 보고 (일 단위)
    const pct = Math.floor(((dayIdx + 1) / days) * 100);
    if (pct >= lastPct + 5) {
      lastPct = pct;
      self.postMessage({ type: 'progress', pct, elapsed: Date.now() - startTime });
    }
  }

  // ── 후처리 ────────────────────────────────────────────────────────────────

  // classStats.buyers를 고유 구매자 수로 확정
  CLASS_KEYS.forEach(cls => {
    classStats[cls].buyers = classBuyerSets[cls].size;
  });

  dailyStats.forEach(d => {
    const buyerCnt = d._buyerSet.size;
    d.convRate  = d.visitors > 0 ? buyerCnt / d.visitors : 0;
    d.avgBasket = buyerCnt > 0   ? d.revenue / buyerCnt  : 0;
    delete d._buyerSet;
  });

  const hourlyStats = Array.from({ length: BUCKETS }, (_, i) => ({
    day:      Math.floor(i / SIM_HOURS_PER_DAY),
    hour:     i % SIM_HOURS_PER_DAY,
    visitors: hVisitors[i],
    buyers:   hBuyers[i],
    revenue:  hRevenue[i],
    convRate: hVisitors[i] > 0 ? hBuyers[i] / hVisitors[i] : 0,
  }));

  // heatGrids Transferable
  const heatBuffers = {};
  const transfers   = [];
  CLASS_KEYS.forEach(cls => {
    heatBuffers[cls] = heatGrids[cls].buffer;
    transfers.push(heatGrids[cls].buffer);
  });

  self.postMessage({
    type: 'done',
    operatingHours: SIM_HOURS_PER_DAY,
    totalVisitors: uid,
    totalBuyers:   totalBuyerSet.size,
    purchases:     allPurchases,
    zoneStats, classStats, dailyStats, hourlyStats,
    heatGrids:     heatBuffers,
    elapsed:       Date.now() - startTime,
  }, transfers);
}
