import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, PieChart, Pie, CartesianGrid,
} from 'recharts';
import BatchAnalytics from './src/BatchAnalytics.jsx';

// ═══════════════════════════════════════════════════════════════════════════
// 상수
// ═══════════════════════════════════════════════════════════════════════════

const CANVAS_W = 800;
const CANVAS_H = 600;
const HEAT_W   = Math.floor(CANVAS_W / 4); // 200
const HEAT_H   = Math.floor(CANVAS_H / 4); // 150

const CUSTOMER_CLASSES = {
  adult_male:   { label: '성년 남성',   color: '#FF9600', icon: '👨', short: '성년남' },
  adult_female: { label: '성년 여성',   color: '#9314FF', icon: '👩', short: '성년여' },
  minor_male:   { label: '미성년 남성', color: '#00C800', icon: '👦', short: '미성년남' },
  minor_female: { label: '미성년 여성', color: '#00A5FF', icon: '👧', short: '미성년여' },
};
const CLASS_KEYS = Object.keys(CUSTOMER_CLASSES);

const ZONES = [
  {
    id: 'snack',      label: '과자/스낵', x:  20, y: 50, w: 165, h: 70,
    preferred: ['minor_male', 'minor_female'],
    items: [
      { name: '새우깡',       price: 1500 }, { name: '포카칩',     price: 1800 },
      { name: '초코파이',     price: 2500 }, { name: '오레오',     price: 2200 },
      { name: '프링글스',     price: 2800 }, { name: '허니버터칩', price: 2000 },
      { name: '꼬깔콘',       price: 1200 }, { name: '빠다코코낫', price: 1500 },
      { name: '젤리',         price: 1000 }, { name: '사탕',       price:  800 },
    ],
  },
  {
    id: 'beverage',   label: '음료', x: 210, y: 50, w: 165, h: 70,
    preferred: ['adult_male', 'adult_female', 'minor_male', 'minor_female'],
    items: [
      { name: '콜라',         price: 1500 }, { name: '사이다',       price: 1500 },
      { name: '커피',         price: 2000 }, { name: '주스',         price: 2500 },
      { name: '이온음료',     price: 1800 }, { name: '녹차',         price: 1200 },
      { name: '에너지드링크', price: 2200 }, { name: '우유',         price: 1800 },
      { name: '두유',         price: 1500 }, { name: '탄산수',       price: 1200 },
    ],
  },
  {
    id: 'daily',      label: '생활용품', x: 400, y: 50, w: 165, h: 70,
    preferred: ['adult_male', 'adult_female'],
    items: [
      { name: '샴푸',     price: 8000 }, { name: '치약',     price: 3500 },
      { name: '세제',     price: 5000 }, { name: '비누',     price: 2000 },
      { name: '칫솔',     price: 3000 }, { name: '수건',     price: 5000 },
      { name: '휴지',     price: 4500 }, { name: '주방세제', price: 4000 },
      { name: '핸드크림', price: 6000 }, { name: '바디로션', price: 8000 },
    ],
  },
  {
    id: 'cosmetic',   label: '화장품', x: 590, y: 50, w: 165, h: 70,
    preferred: ['adult_female'],
    items: [
      { name: '립스틱',     price: 15000 }, { name: '파운데이션', price: 25000 },
      { name: '마스카라',   price: 12000 }, { name: '아이섀도',   price: 18000 },
      { name: '블러셔',     price: 14000 }, { name: '컨실러',     price: 16000 },
      { name: 'BB크림',     price: 18000 }, { name: '선크림',     price: 20000 },
      { name: '립밤',       price:  8000 }, { name: '아이라이너', price: 12000 },
    ],
  },
  {
    id: 'stationery', label: '문구', x:  20, y: 180, w: 165, h: 70,
    preferred: ['minor_male', 'minor_female'],
    items: [
      { name: '색연필',   price: 3000 }, { name: '노트',     price: 2000 },
      { name: '펜',       price: 1000 }, { name: '스티커',   price: 1500 },
      { name: '형광펜',   price: 1500 }, { name: '지우개',   price:  500 },
      { name: '자',       price: 1000 }, { name: '풀',       price: 1200 },
      { name: '테이프',   price: 1500 }, { name: '스케치북', price: 4000 },
    ],
  },
  {
    id: 'frozen',     label: '냉동식품', x: 210, y: 180, w: 165, h: 70,
    preferred: ['adult_male', 'adult_female'],
    items: [
      { name: '아이스크림',   price:  2000 }, { name: '만두',       price: 5000 },
      { name: '냉동피자',     price:  8000 }, { name: '냉동치킨',   price: 9000 },
      { name: '군만두',       price:  6000 }, { name: '냉동볶음밥', price: 4500 },
      { name: '냉동새우',     price: 12000 }, { name: '어묵',       price: 3000 },
      { name: '핫도그',       price:  2500 }, { name: '냉동라면',   price: 3500 },
    ],
  },
  {
    id: 'premium',    label: '프리미엄', x: 400, y: 180, w: 165, h: 70,
    preferred: ['adult_male', 'adult_female'],
    items: [
      { name: '와인',       price: 25000 }, { name: '위스키',     price: 45000 },
      { name: '프리미엄커피', price: 12000 }, { name: '수입과자', price:  8000 },
      { name: '샴페인',     price: 35000 }, { name: '럼',         price: 28000 },
      { name: '브랜디',     price: 42000 }, { name: '수입초콜릿', price: 15000 },
      { name: '수입치즈',   price: 18000 }, { name: '수입와인',   price: 32000 },
    ],
  },
  {
    id: 'bakery',     label: '베이커리', x: 590, y: 180, w: 165, h: 70,
    preferred: ['minor_female', 'adult_female'],
    items: [
      { name: '크로아상',   price:  3000 }, { name: '케이크',     price: 15000 },
      { name: '샌드위치',   price:  4500 }, { name: '마카롱',     price:  5000 },
      { name: '베이글',     price:  2500 }, { name: '소금빵',     price:  2000 },
      { name: '시나몬롤',   price:  3500 }, { name: '머핀',       price:  2500 },
      { name: '치즈케이크', price:  6000 }, { name: '에클레어',   price:  4000 },
    ],
  },
];

const CHECKOUTS = [
  { id: 'co1', x:  20, y: 430, w: 85, h: 50 },
  { id: 'co2', x: 120, y: 430, w: 85, h: 50 },
  { id: 'co3', x: 220, y: 430, w: 85, h: 50 },
];

const ENT_X = CANVAS_W / 2;
const ENT_Y = CANVAS_H - 16;

// 커스텀 장애물 (마우스 클릭으로 추가/제거)
const BARRIER_SIZE = 24;
let _customBarriers = []; // { x, y, w, h }

// A* 경로탐색 그리드
const PF_CELL = 12;                             // 셀 크기 (px)
const PF_W    = Math.ceil(CANVAS_W / PF_CELL);  // ~67 cols
const PF_H    = Math.ceil(CANVAS_H / PF_CELL);  // 50 rows
let   _pfGrid  = null;
let   _pfDirty = true;  // true → 다음 getGrid() 시 재빌드

// KPI 프리셋별 시뮬레이션 파라미터 (구매 확률·체류 배율)
let _simConfig = { dwellScale: 1.0, purchaseBonus: 0.0 };

// ═══════════════════════════════════════════════════════════════════════════
// 성능 최적화 유틸리티
// ═══════════════════════════════════════════════════════════════════════════

// A* 용 MinHeap (O(log n) 삽입/추출 → Set O(n) 선형 탐색 대체)
class MinHeap {
  constructor() { this.h = []; }
  get size() { return this.h.length; }
  push(idx, p) { this.h.push({ idx, p }); this._up(this.h.length - 1); }
  pop() {
    const top = this.h[0].idx;
    const last = this.h.pop();
    if (this.h.length > 0) { this.h[0] = last; this._down(0); }
    return top;
  }
  _up(i) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.h[p].p <= this.h[i].p) break;
      [this.h[p], this.h[i]] = [this.h[i], this.h[p]]; i = p;
    }
  }
  _down(i) {
    const n = this.h.length;
    while (true) {
      let s = i, l = 2*i+1, r = 2*i+2;
      if (l < n && this.h[l].p < this.h[s].p) s = l;
      if (r < n && this.h[r].p < this.h[s].p) s = r;
      if (s === i) break;
      [this.h[s], this.h[i]] = [this.h[i], this.h[s]]; i = s;
    }
  }
}

// 히트맵 가우시안 커널 사전 계산 (매 프레임 exp() 연산 제거)
const _GRAV = 3, _GSIG = 1.4;
const GAUSS_OFFSETS = [];
for (let dy = -_GRAV; dy <= _GRAV; dy++)
  for (let dx = -_GRAV; dx <= _GRAV; dx++)
    GAUSS_OFFSETS.push({ dx, dy, w: Math.exp(-(dx*dx + dy*dy) / (2 * _GSIG * _GSIG)) });

// ═══════════════════════════════════════════════════════════════════════════
// A* 경로탐색
// ═══════════════════════════════════════════════════════════════════════════

function buildPFGrid() {
  const g = new Uint8Array(PF_W * PF_H);
  const markRect = (x, y, w, h, margin) => {
    const gx0 = Math.max(0, Math.floor((x - margin) / PF_CELL));
    const gx1 = Math.min(PF_W - 1, Math.ceil((x + w + margin) / PF_CELL));
    const gy0 = Math.max(0, Math.floor((y - margin) / PF_CELL));
    const gy1 = Math.min(PF_H - 1, Math.ceil((y + h + margin) / PF_CELL));
    for (let gy = gy0; gy <= gy1; gy++)
      for (let gx = gx0; gx <= gx1; gx++)
        g[gy * PF_W + gx] = 1;
  };
  ZONES.forEach(z => markRect(z.x, z.y, z.w, z.h, 8));          // 매대 (여유 8px)
  _customBarriers.forEach(b => markRect(b.x, b.y, b.w, b.h, 4)); // 사용자 장애물
  return g;
}

function getPFGrid() {
  if (_pfDirty || !_pfGrid) { _pfGrid = buildPFGrid(); _pfDirty = false; }
  return _pfGrid;
}

// 두 점 사이에 장애물이 있는지 확인 (경로 스무딩용)
function lineBlocked(grid, x1, y1, x2, y2) {
  const steps = Math.ceil(Math.hypot(x2 - x1, y2 - y1) / PF_CELL) + 1;
  for (let s = 1; s < steps; s++) {
    const t = s / steps;
    const gx = Math.round((x1 + (x2 - x1) * t) / PF_CELL);
    const gy = Math.round((y1 + (y2 - y1) * t) / PF_CELL);
    if (gx < 0 || gx >= PF_W || gy < 0 || gy >= PF_H || grid[gy * PF_W + gx]) return true;
  }
  return false;
}

// 시야선 기반 경로 단순화 (zig-zag 제거)
function smoothPath(path, grid) {
  if (path.length <= 2) return path;
  const out = [path[0]];
  let i = 0;
  while (i < path.length - 1) {
    let j = path.length - 1;
    while (j > i + 1 && lineBlocked(grid, path[i].x, path[i].y, path[j].x, path[j].y)) j--;
    out.push(path[j]);
    i = j;
  }
  return out;
}

// 벽 안에 있으면 인접한 빈 셀로 nudge
function nudgeToWalkable(grid, gx, gy) {
  if (gx >= 0 && gx < PF_W && gy >= 0 && gy < PF_H && !grid[gy * PF_W + gx]) return { gx, gy };
  for (let r = 1; r <= 4; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const nx = gx + dx, ny = gy + dy;
        if (nx >= 0 && nx < PF_W && ny >= 0 && ny < PF_H && !grid[ny * PF_W + nx])
          return { gx: nx, gy: ny };
      }
    }
  }
  return { gx, gy };
}

// A* 메인 함수: 월드 좌표 → 웨이포인트 배열 or null
function findPath(sx, sy, tx, ty) {
  const grid = getPFGrid();
  const cl   = (v, lo, hi) => Math.max(lo, Math.min(hi, Math.round(v)));

  let { gx: gsx, gy: gsy } = nudgeToWalkable(grid, cl(sx / PF_CELL, 0, PF_W - 1), cl(sy / PF_CELL, 0, PF_H - 1));
  let { gx: gex, gy: gey } = nudgeToWalkable(grid, cl(tx / PF_CELL, 0, PF_W - 1), cl(ty / PF_CELL, 0, PF_H - 1));

  const N   = PF_W * PF_H;
  const gSc = new Float32Array(N).fill(1e9);
  const fSc = new Float32Array(N).fill(1e9);
  const par = new Int32Array(N).fill(-1);
  const cls = new Uint8Array(N);

  const si = gsy * PF_W + gsx, ei = gey * PF_W + gex;
  if (si === ei) return [{ x: tx, y: ty }];

  gSc[si] = 0;
  fSc[si] = Math.hypot(gex - gsx, gey - gsy);
  const heap = new MinHeap();
  heap.push(si, fSc[si]);

  const DIRS = [[1,0,1],[-1,0,1],[0,1,1],[0,-1,1],[1,1,1.414],[1,-1,1.414],[-1,1,1.414],[-1,-1,1.414]];
  let found = false;

  for (let iter = 0; heap.size > 0 && iter < 5000; iter++) {
    // MinHeap으로 O(log n) 최소 f 노드 추출
    const cur = heap.pop();
    if (cls[cur]) continue; // stale entry (중복 push 된 경우)
    if (cur === ei) { found = true; break; }
    cls[cur] = 1;

    const cx = cur % PF_W, cy = (cur / PF_W) | 0;
    for (const [dx, dy, w] of DIRS) {
      const nx = cx + dx, ny = cy + dy;
      if (nx < 0 || nx >= PF_W || ny < 0 || ny >= PF_H) continue;
      const ni = ny * PF_W + nx;
      if (cls[ni] || grid[ni]) continue;
      // 대각 이동: 양옆 셀이 막히면 통과 불가
      if (dx !== 0 && dy !== 0 && (grid[cy * PF_W + nx] || grid[ny * PF_W + cx])) continue;
      const ng = gSc[cur] + w;
      if (ng < gSc[ni]) {
        par[ni] = cur; gSc[ni] = ng;
        fSc[ni] = ng + Math.hypot(gex - nx, gey - ny);
        heap.push(ni, fSc[ni]);
      }
    }
  }

  if (!found) return null;

  // 경로 복원
  const raw = [];
  let c = ei;
  while (c !== si && c !== -1) {
    raw.unshift({ x: (c % PF_W) * PF_CELL + PF_CELL * 0.5, y: ((c / PF_W) | 0) * PF_CELL + PF_CELL * 0.5 });
    c = par[c];
  }
  raw.push({ x: tx, y: ty });
  return smoothPath(raw, grid);
}

// ═══════════════════════════════════════════════════════════════════════════
// Viridis 컬러맵
// ═══════════════════════════════════════════════════════════════════════════

const VIRIDIS = [
  [68,1,84],[72,40,120],[62,73,137],[49,104,142],
  [38,130,142],[31,158,137],[53,183,121],[110,206,88],
  [181,222,43],[253,231,37],
];

function viridisColor(t) {
  const idx = Math.min(t * (VIRIDIS.length - 1), VIRIDIS.length - 1.001);
  const lo = Math.floor(idx), hi = lo + 1;
  const f = idx - lo;
  return VIRIDIS[lo].map((c, i) => Math.round(c + f * (VIRIDIS[hi][i] - c)));
}

// ═══════════════════════════════════════════════════════════════════════════
// 에이전트 헬퍼 (모듈 레벨)
// ═══════════════════════════════════════════════════════════════════════════

let _uid = 0;

function pickClass(ratios) {
  const total = Object.values(ratios).reduce((a, b) => a + b, 0);
  if (total === 0) return CLASS_KEYS[0];
  let r = Math.random() * total;
  for (const [cls, w] of Object.entries(ratios)) { r -= w; if (r <= 0) return cls; }
  return CLASS_KEYS[0];
}

function buildQueue(classType) {
  const pref    = ZONES.filter(z => z.preferred.includes(classType));
  const nonPref = ZONES.filter(z => !z.preferred.includes(classType));
  const shuffle = arr => [...arr].sort(() => Math.random() - 0.5);
  const nPref   = 1 + Math.floor(Math.random() * Math.min(pref.length, 3));
  const nNon    = 1 + Math.floor(Math.random() * 2);
  return [...shuffle(pref).slice(0, nPref), ...shuffle(nonPref).slice(0, nNon)];
}

function spawnAgent(classType) {
  const minor = classType.startsWith('minor');
  return {
    id: ++_uid,
    classType,
    x: ENT_X + (Math.random() - 0.5) * 30,
    y: ENT_Y,
    vx: 0, vy: 0,
    state: 'entering',
    targetX: ENT_X + (Math.random() - 0.5) * 80,
    targetY: 350 + Math.random() * 60,
    browseTarget: null,
    currentZone: null,
    dwellTimer: 0, dwellDuration: 0,
    checkoutTimer: 0,
    checkoutDuration: 0,
    pendingZones: buildQueue(classType),
    visitedZones: [], dwellTime: {}, purchaseHistory: [],
    speed: minor ? 1.1 + Math.random() * 0.9 : 0.7 + Math.random() * 0.5,
    height_cm: minor ? 130 + Math.random() * 30 : 155 + Math.random() * 30,
    entryTime: 0,
    // A* 경로탐색용
    path: [],            // 남은 웨이포인트 [{x,y}]
    pathTarget: null,    // 현재 path가 향하는 최종 목표
    stuckTimer: 0,       // 정체 감지 타이머
    pathCooldown: 0,
  };
}

function fmtTime(secs) {
  const s = Math.floor(secs);
  return `${String(Math.floor(s / 60)).padStart(2,'0')}:${String(s % 60).padStart(2,'0')}`;
}

// 에이전트 업데이트 (매 프레임 호출)
function stepAgent(a, allAgents, purchases, simSecs, dt) {
  const minor = a.classType.startsWith('minor');

  // ── 상태 머신 ──────────────────────────────────────────────────────────
  switch (a.state) {

    case 'entering': {
      const dx = a.targetX - a.x, dy = a.targetY - a.y;
      if (Math.hypot(dx, dy) < 30) {
        a.pathCooldown = 0;
        a.state = 'browsing';
      }
      break;
    }

    case 'browsing':
      if (a.pendingZones.length === 0) {
        const co = CHECKOUTS[Math.floor(Math.random() * CHECKOUTS.length)];
        a.targetX = co.x + co.w / 2;
        a.targetY = co.y + co.h / 2;
        a.browseTarget = null;
        a.pathCooldown = 0;
        a.state = 'checkout';
        break;
      }
      if (!a.browseTarget) {
        const z = a.pendingZones[0];
        a.browseTarget = {
          x: z.x + z.w / 2 + (Math.random() - 0.5) * 30,
          y: z.y + z.h + 18 + Math.random() * 14,
        };
        a.targetX = a.browseTarget.x;
        a.targetY = a.browseTarget.y;
      }
      {
        const dx = a.targetX - a.x, dy = a.targetY - a.y;
        if (Math.hypot(dx, dy) < 22) {
          const z = a.pendingZones.shift();
          a.currentZone  = z.id;
          a.browseTarget = null;
          a.state        = 'shopping';
          a.dwellTimer   = 0;
          a.dwellDuration = (minor ? 1 + Math.random() * 3 : 3 + Math.random() * 5) * _simConfig.dwellScale;
        }
      }
      break;

    case 'shopping': {
      a.dwellTimer += dt;
      if (Math.random() < 0.015) {
        const z = ZONES.find(z => z.id === a.currentZone);
        if (z) {
          a.targetX = z.x + z.w / 2 + (Math.random() - 0.5) * 50;
          a.targetY = z.y + z.h + 14 + Math.random() * 24;
        }
      }
      if (a.dwellTimer >= a.dwellDuration) {
        const z = ZONES.find(z => z.id === a.currentZone);
        if (z) {
          if (!a.visitedZones.includes(z.id)) a.visitedZones.push(z.id);
          a.dwellTime[z.id] = (a.dwellTime[z.id] || 0) + a.dwellTimer;
          const isPref = z.preferred.includes(a.classType);
          const buyProb = Math.min(0.85, Math.max(0.02, (isPref ? 0.35 : 0.08) + _simConfig.purchaseBonus));
          if (Math.random() < buyProb) {
            const item = z.items[Math.floor(Math.random() * z.items.length)];
            const rec = {
              agentId: a.id, classType: a.classType,
              zoneId: z.id, zoneLabel: z.label,
              item: item.name, price: item.price,
              dwellTime: a.dwellTimer, time: fmtTime(simSecs), simSecAt: simSecs,
            };
            purchases.push(rec);
            a.purchaseHistory.push(rec);
          }
        }
        a.currentZone = null;
        a.pathCooldown = 0;
        a.state = 'browsing';
      }
      break;
    }

    case 'checkout': {
      const dx = a.targetX - a.x, dy = a.targetY - a.y;
      if (Math.hypot(dx, dy) < 22) {
        if (a.checkoutDuration === 0) a.checkoutDuration = 2 + Math.random() * 4;
        a.checkoutTimer += dt;
        if (a.checkoutTimer > a.checkoutDuration) {
          a.targetX = ENT_X + (Math.random() - 0.5) * 40;
          a.targetY = ENT_Y;
          a.state = 'leaving';
        }
      }
      break;
    }

    case 'leaving': {
      const dx = a.targetX - a.x, dy = a.targetY - a.y;
      if (Math.hypot(dx, dy) < 20) a.state = 'removed';
      break;
    }
  }

  // ── A* 경로 추종 + 조향 ─────────────────────────────────────────────────
  // 목표가 바뀌었으면 경로 재계산
  a.pathCooldown = Math.max(0, (a.pathCooldown || 0) - dt);
  const tgtChanged = !a.pathTarget ||
    Math.hypot(a.targetX - a.pathTarget.x, a.targetY - a.pathTarget.y) > 18;
  if (tgtChanged && a.pathCooldown <= 0) {
    a.pathTarget = { x: a.targetX, y: a.targetY };
    const p = findPath(a.x, a.y, a.targetX, a.targetY);
    a.path = p || [];
    a.stuckTimer = 0;
    a.pathCooldown = 0.5;
  }

  // 도달한 웨이포인트 제거
  while (a.path.length > 1 && Math.hypot(a.path[0].x - a.x, a.path[0].y - a.y) < PF_CELL * 2.0) {
    a.path.shift();
  }

  // 실제 조향 목표 (웨이포인트 or 최종 목표)
  const steerX = a.path.length > 0 ? a.path[0].x : a.targetX;
  const steerY = a.path.length > 0 ? a.path[0].y : a.targetY;

  const tdx   = steerX - a.x, tdy = steerY - a.y;
  const tdist = Math.hypot(tdx, tdy) || 0.001;
  const nx    = tdx / tdist, ny = tdy / tdist;

  // 에이전트 간 분리 (Separation)
  let sepX = 0, sepY = 0;
  const SEP = 14;
  for (const other of allAgents) {
    if (other === a || other.state === 'removed') continue;
    const sdx = a.x - other.x, sdy = a.y - other.y;
    const sd = Math.hypot(sdx, sdy);
    if (sd < SEP && sd > 0) {
      const f = (SEP - sd) / SEP;
      sepX += (sdx / sd) * f;
      sepY += (sdy / sd) * f;
    }
  }

  // Arrive 감속
  const slowR  = 50;
  const sm     = tdist < slowR ? 0.15 + 0.85 * (tdist / slowR) : 1;
  const maxSpd = a.speed * 60 * dt;

  a.vx = (nx + sepX * 0.35) * maxSpd * sm;
  a.vy = (ny + sepY * 0.35) * maxSpd * sm;

  // 정체 감지: 움직임 없으면 경로 강제 재계산 (장애물에 끼인 경우)
  if (Math.hypot(a.vx, a.vy) < 0.08 && tdist > 30) {
    a.stuckTimer = (a.stuckTimer || 0) + dt;
    if (a.stuckTimer > 1.2) {
      a.pathTarget = null; // 경로 재계산 트리거
      a.stuckTimer = 0;
      // 랜덤 탈출 impulse
      a.vx += (Math.random() - 0.5) * 2;
      a.vy += (Math.random() - 0.5) * 2;
    }
  } else {
    a.stuckTimer = 0;
  }

  {
    const newX = Math.max(8, Math.min(CANVAS_W - 8, a.x + a.vx));
    const newY = Math.max(8, Math.min(CANVAS_H - 8, a.y + a.vy));
    const grid = getPFGrid();
    const gx = Math.floor(newX / PF_CELL);
    const gy = Math.floor(newY / PF_CELL);
    if (gx >= 0 && gx < PF_W && gy >= 0 && gy < PF_H && !grid[gy * PF_W + gx]) {
      a.x = newX;
      a.y = newY;
    } else {
      // 축별 슬라이딩
      const sgx = Math.floor(newX / PF_CELL);
      const sgyOld = Math.floor(a.y / PF_CELL);
      if (sgx >= 0 && sgx < PF_W && sgyOld >= 0 && sgyOld < PF_H && !grid[sgyOld * PF_W + sgx]) {
        a.x = newX;
      } else {
        const sgxOld = Math.floor(a.x / PF_CELL);
        const sgy = Math.floor(newY / PF_CELL);
        if (sgxOld >= 0 && sgxOld < PF_W && sgy >= 0 && sgy < PF_H && !grid[sgy * PF_W + sgxOld]) {
          a.y = newY;
        }
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// CSV 유틸
// ═══════════════════════════════════════════════════════════════════════════

function dlCSV(filename, content) {
  const blob = new Blob(['\uFEFF' + content], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

// ═══════════════════════════════════════════════════════════════════════════
// KPI 프리셋 (한국 소매 업태별 실증 데이터 기반)
// ═══════════════════════════════════════════════════════════════════════════

const KPI_PRESETS = {
  convenience: {
    label: '편의점',
    desc:  '일 방문객 ~300명 | 구매 전환율 ~88% | 평균 구매액 4,500원',
    // 한국 편의점(GS25/CU 평균): 남성 55%, 여성 45%, 20-30대 집중
    spawnRate:    5,
    classRatios:  { adult_male: 38, adult_female: 32, minor_male: 17, minor_female: 13 },
    speed:        1.4,   // 목적 구매, 빠른 동선
    dwellScale:   0.65,  // 매대 체류 짧음
    purchaseBonus: 0.18, // 높은 전환율
    dailyVisitors: 300,
    operatingHours: 16,
    weekendBoost: 1.05,
    availableZones: ['snack', 'beverage', 'daily', 'frozen'],  // 편의점: 과자, 음료, 생활용품, 냉동만
    convRate:     88,
    avgBasket:    4500,
  },
  supermarket: {
    label: '슈퍼마켓',
    desc:  '일 방문객 ~500명 | 구매 전환율 ~74% | 평균 구매액 18,000원',
    // 한국 슈퍼마켓: 주부 비중 높음, 30-40대 중심
    spawnRate:    3,
    classRatios:  { adult_male: 27, adult_female: 48, minor_male: 12, minor_female: 13 },
    speed:        1.0,
    dwellScale:   1.3,
    purchaseBonus: 0.04,
    dailyVisitors: 500,
    operatingHours: 12,
    weekendBoost: 1.25,
    availableZones: ['snack', 'beverage', 'daily', 'cosmetic', 'frozen', 'premium', 'bakery'],  // 슈퍼: 문구 제외
    convRate:     74,
    avgBasket:    18000,
  },
  mall: {
    label: '쇼핑몰/복합매장',
    desc:  '일 방문객 ~800명 | 구매 전환율 ~38% | 평균 구매액 35,000원',
    // 쇼핑몰: 20-30대 여성 중심, 탐색형 소비
    spawnRate:    4,
    classRatios:  { adult_male: 22, adult_female: 48, minor_male: 15, minor_female: 15 },
    speed:        0.75,
    dwellScale:   1.9,
    purchaseBonus: -0.18,
    dailyVisitors: 800,
    operatingHours: 10,
    weekendBoost: 1.40,
    availableZones: ['snack', 'beverage', 'daily', 'cosmetic', 'stationery', 'frozen', 'premium', 'bakery'],  // 쇼핑몰: 전체
    convRate:     38,
    avgBasket:    35000,
  },
  school: {
    label: '학교 앞 문구/분식',
    desc:  '일 방문객 ~200명 | 구매 전환율 ~80% | 평균 구매액 3,500원',
    // 하교 후 집중: 미성년 비율 80%, 빠른 구매 결정
    spawnRate:    5,
    classRatios:  { adult_male: 8, adult_female: 12, minor_male: 40, minor_female: 40 },
    speed:        1.35,
    dwellScale:   0.75,
    purchaseBonus: 0.12,
    dailyVisitors: 200,
    operatingHours: 6,
    weekendBoost: 0.30,
    availableZones: ['snack', 'beverage', 'stationery', 'bakery'],  // 학교앞: 과자, 음료, 문구, 빵만
    convRate:     80,
    avgBasket:    3500,
  },
};

// ═══════════════════════════════════════════════════════════════════════════
// 메인 컴포넌트
// ═══════════════════════════════════════════════════════════════════════════

export default function RetailDashboard() {
  const canvasRef    = useRef(null);
  const rafRef       = useRef(null);
  const heatOffRef   = useRef(null); // offscreen canvas for heatmap

  // 시뮬레이션 데이터 (React state 아님 → 리렌더 없이 매 프레임 갱신)
  const simRef = useRef({
    running: false,
    agents:  [],
    tick:    0,
    spawnTimer:    0,
    simSecs:       0,
    allPurchases:  [],
    totalVisitors: 0,  // 누적 입장 고객 수 (KPI용)
    heatGrids: {
      adult_male:   new Float32Array(HEAT_W * HEAT_H),
      adult_female: new Float32Array(HEAT_W * HEAT_H),
      minor_male:   new Float32Array(HEAT_W * HEAT_H),
      minor_female: new Float32Array(HEAT_W * HEAT_H),
    },
    speed:       1,
    spawnRate:   3,
    classRatios: { adult_male: 25, adult_female: 25, minor_male: 25, minor_female: 25 },
  });

  // UI state
  const [running,      setRunning]      = useState(false);
  const [speed,        setSpeed]        = useState(1);
  const [spawnRate,    setSpawnRate]    = useState(3);
  const [classRatios,  setClassRatios]  = useState({ adult_male: 25, adult_female: 25, minor_male: 25, minor_female: 25 });
  const [showHeatmap,  setShowHeatmap]  = useState(true);
  const [showROI,      setShowROI]      = useState(true);
  const [heatmapMode,  setHeatmapMode]  = useState('combined');
  const [barrierMode,  setBarrierMode]  = useState(false);
  const [barrierCount, setBarrierCount] = useState(0);
  const [kpiPreset,    setKpiPreset]    = useState('supermarket');
  const [stats, setStats] = useState({
    counts:        { adult_male:0, adult_female:0, minor_male:0, minor_female:0 },
    total:         0,
    checkoutCnt:   0,
    topZones:      [],
    revByClass:    [],
    dwellByZone:   [],
    insights:      [],
    // KPI 지표
    totalVisitors: 0,
    convRate:      0,
    avgBasket:     0,
    revenuePerHr:  0,
    simTime:      '00:00',
    totalRev:     0,
  });

  // 커스텀 파라미터 상태
  const ALL_ZONE_IDS = ZONES.map(z => z.id);
  const [dailyVisitors,  setDailyVisitors]  = useState(500);
  const [operatingHours, setOperatingHours] = useState(12);
  const [dwellScale,     setDwellScale]     = useState(1.3);
  const [purchaseBonus,  setPurchaseBonus]  = useState(0.04);
  const [weekendBoost,   setWeekendBoost]   = useState(1.15);
  const [availableZones, setAvailableZones] = useState(ALL_ZONE_IDS);
  const [isCustom,       setIsCustom]       = useState(false);
  const [showAdvanced,   setShowAdvanced]   = useState(false);

  // 배치 시뮬레이션 상태
  const [activeTab,     setActiveTab]     = useState('simulation'); // 'simulation' | 'analytics'
  const [batchStatus,   setBatchStatus]   = useState('idle');       // 'idle' | 'running' | 'done'
  const [batchProgress, setBatchProgress] = useState(0);
  const [batchData,     setBatchData]     = useState(null);
  const workerRef = useRef(null);

  // ref 동기화 (canvas 루프에서 최신값 읽기)
  const showHeatmapRef = useRef(showHeatmap);
  const showROIRef     = useRef(showROI);
  const heatmapModeRef = useRef(heatmapMode);
  const barrierModeRef = useRef(barrierMode);
  useEffect(() => { showHeatmapRef.current = showHeatmap; },  [showHeatmap]);
  useEffect(() => { showROIRef.current     = showROI; },      [showROI]);
  useEffect(() => { heatmapModeRef.current = heatmapMode; },  [heatmapMode]);
  useEffect(() => { barrierModeRef.current = barrierMode; },  [barrierMode]);
  useEffect(() => { simRef.current.speed       = speed; },       [speed]);
  useEffect(() => { simRef.current.spawnRate   = spawnRate; },   [spawnRate]);
  useEffect(() => { simRef.current.classRatios = { ...classRatios }; }, [classRatios]);
  useEffect(() => { _simConfig.dwellScale    = dwellScale; },    [dwellScale]);
  useEffect(() => { _simConfig.purchaseBonus = purchaseBonus; }, [purchaseBonus]);

  // KPI 프리셋 적용 → _simConfig + UI 슬라이더 동기화
  const applyPreset = useCallback((presetKey) => {
    const p = KPI_PRESETS[presetKey];
    if (!p) return;
    _simConfig.dwellScale    = p.dwellScale;
    _simConfig.purchaseBonus = p.purchaseBonus;
    setSpawnRate(p.spawnRate);
    setClassRatios({ ...p.classRatios });
    setSpeed(p.speed);
    setKpiPreset(presetKey);
    // 커스텀 파라미터 동기화
    setDailyVisitors(p.dailyVisitors);
    setOperatingHours(p.operatingHours);
    setDwellScale(p.dwellScale);
    setPurchaseBonus(p.purchaseBonus);
    setWeekendBoost(p.weekendBoost);
    setAvailableZones(p.availableZones || ALL_ZONE_IDS);
    setIsCustom(false);
  }, []);

  // 최초 마운트 시 기본 프리셋 적용
  useEffect(() => { applyPreset('supermarket'); }, []);

  // offscreen 히트맵 캔버스 초기화
  useEffect(() => {
    const oc = document.createElement('canvas');
    oc.width = HEAT_W; oc.height = HEAT_H;
    heatOffRef.current = oc;
  }, []);

  // ── 통계 계산 ────────────────────────────────────────────────────────────
  const computeStats = useCallback(() => {
    const { agents, allPurchases, simSecs, totalVisitors } = simRef.current;

    const counts = { adult_male:0, adult_female:0, minor_male:0, minor_female:0 };
    let checkoutCnt = 0;
    agents.forEach(a => {
      if (a.state !== 'removed') {
        counts[a.classType]++;
        if (a.state === 'checkout') checkoutCnt++;
      }
    });
    const total = Object.values(counts).reduce((s,v) => s+v, 0);

    // 현재 쇼핑 중인 매대별 인원
    const zoneMap = {};
    agents.forEach(a => { if (a.currentZone) zoneMap[a.currentZone] = (zoneMap[a.currentZone]||0) + 1; });
    const topZones = ZONES
      .map(z => ({ label: z.label, count: zoneMap[z.id]||0 }))
      .filter(z => z.count > 0)
      .sort((a,b) => b.count - a.count)
      .slice(0, 3);

    // 클래스별 매출
    const rev = {};
    allPurchases.forEach(p => rev[p.classType] = (rev[p.classType]||0) + p.price);
    const revByClass = CLASS_KEYS.map(cls => ({
      name:  CUSTOMER_CLASSES[cls].short,
      value: rev[cls] || 0,
      color: CUSTOMER_CLASSES[cls].color,
    }));
    const totalRev = Object.values(rev).reduce((s,v) => s+v, 0);

    // 매대별 평균 체류
    const dwS = {}, dwC = {};
    allPurchases.forEach(p => {
      dwS[p.zoneId] = (dwS[p.zoneId]||0) + (p.dwellTime||0);
      dwC[p.zoneId] = (dwC[p.zoneId]||0) + 1;
    });
    const dwellByZone = ZONES
      .map(z => ({ name: z.label, avg: dwC[z.id] ? +(dwS[z.id]/dwC[z.id]).toFixed(1) : 0 }))
      .filter(z => z.avg > 0)
      .sort((a,b) => b.avg - a.avg)
      .slice(0, 6);

    // 자동 인사이트
    const insights = [];
    if (allPurchases.length >= 5) {
      // ── 수익 최적화 ───────────────────────────────────────────────────────
      // [1] 객단가: 클래스별 평균 구매액 비교
      const clsRevMap = {};
      const clsBuyerMap = {};
      allPurchases.forEach(p => {
        clsRevMap[p.classType] = (clsRevMap[p.classType]||0) + p.price;
        if (!clsBuyerMap[p.classType]) clsBuyerMap[p.classType] = new Set();
        clsBuyerMap[p.classType].add(p.agentId);
      });
      const clsAvgBasket = CLASS_KEYS
        .filter(k => clsBuyerMap[k] && clsBuyerMap[k].size > 0)
        .map(k => ({ k, avg: Math.round(clsRevMap[k] / clsBuyerMap[k].size) }))
        .sort((a,b) => b.avg - a.avg);
      if (clsAvgBasket.length >= 2) {
        const top = clsAvgBasket[0];
        const bot = clsAvgBasket[clsAvgBasket.length-1];
        insights.push(`객단가: ${CUSTOMER_CLASSES[top.k].label} ₩${top.avg.toLocaleString()} (최고) vs ${CUSTOMER_CLASSES[bot.k].label} ₩${bot.avg.toLocaleString()} (최저) → 고단가 고객군 타겟 상품 강화 권장`);
      }

      // [2] 시간당 매출 효율 (초반 절반 vs 후반 절반)
      if (simSecs > 60) {
        const midTime = simSecs / 2;
        const earlyRev = allPurchases.filter(p => (p.simSecAt||0) < midTime).reduce((s,p)=>s+p.price,0);
        const lateRev  = allPurchases.filter(p => (p.simSecAt||0) >= midTime).reduce((s,p)=>s+p.price,0);
        if (earlyRev > 0) {
          const growthPct = Math.round((lateRev - earlyRev) / earlyRev * 100);
          if (Math.abs(growthPct) >= 10) {
            insights.push(growthPct > 0
              ? `매출 성장 추세: 후반 매출이 전반 대비 ${growthPct}% 증가 → 현재 고객 유입 전략 유지 권장`
              : `매출 정체 경보: 후반 매출이 전반 대비 ${Math.abs(growthPct)}% 감소 → 프로모션 또는 유입 강화 필요`);
          }
        }
      }

      // [3] 크로스셀링 기회: 2개 이상 매대에서 구매한 고객 비율
      const agentZoneBuy = {};
      allPurchases.forEach(p => {
        if (!agentZoneBuy[p.agentId]) agentZoneBuy[p.agentId] = new Set();
        agentZoneBuy[p.agentId].add(p.zoneId);
      });
      const crossBuyers = Object.values(agentZoneBuy).filter(s => s.size >= 2).length;
      const totalBuyers = Object.keys(agentZoneBuy).length;
      if (totalBuyers >= 3) {
        const crossPct = Math.round(crossBuyers / totalBuyers * 100);
        if (crossPct >= 30) {
          insights.push(`구매 고객의 ${crossPct}%가 2개 이상 매대에서 구매 → 연관 매대 인접 배치로 크로스셀링 극대화 가능`);
        } else {
          insights.push(`크로스셀링 비율 ${crossPct}% (저조) → 연관 상품 번들 기획 또는 매대 간 동선 단축 검토 필요`);
        }
      }

      // [4] 매대별 매출 집중도: 상위 2개 매대가 전체 매출의 X%
      const zoneRevMap = {};
      allPurchases.forEach(p => { zoneRevMap[p.zoneLabel] = (zoneRevMap[p.zoneLabel]||0) + p.price; });
      const zoneRevArr = Object.entries(zoneRevMap).sort((a,b)=>b[1]-a[1]);
      if (zoneRevArr.length >= 3 && totalRev > 0) {
        const top2Rev = zoneRevArr.slice(0,2).reduce((s,[,v])=>s+v,0);
        const top2Pct = Math.round(top2Rev / totalRev * 100);
        if (top2Pct >= 60) {
          insights.push(`"${zoneRevArr[0][0]}", "${zoneRevArr[1][0]}" 2개 매대가 전체 매출의 ${top2Pct}% 집중 → 나머지 매대 활성화 전략 수립 필요`);
        }
      }

      // ── 동선/배치 최적화 ─────────────────────────────────────────────────
      // [5] 체류 대비 구매전환율이 낮은 매대 (방문 많은데 구매 적음)
      const zoneVisitMap = {};
      const zonePurchaseMap = {};
      allPurchases.forEach(p => { zonePurchaseMap[p.zoneLabel] = (zonePurchaseMap[p.zoneLabel]||0)+1; });
      agents.forEach(a => {
        (a.visitedZones||[]).forEach(zid => {
          const z = ZONES.find(z=>z.id===zid);
          if (z) zoneVisitMap[z.label] = (zoneVisitMap[z.label]||0)+1;
        });
      });
      const lowConvZones = Object.entries(zoneVisitMap)
        .filter(([label, visits]) => visits >= 5)
        .map(([label, visits]) => {
          const buys = zonePurchaseMap[label]||0;
          return { label, visits, buys, rate: buys/visits };
        })
        .filter(z => z.rate < 0.2)
        .sort((a,b) => a.rate - b.rate);
      if (lowConvZones.length > 0) {
        const lc = lowConvZones[0];
        insights.push(`"${lc.label}" 방문 ${lc.visits}회 대비 구매전환 ${Math.round(lc.rate*100)}% (저조) → 상품 구성 또는 가격대 재검토 권장`);
      }

      // [6] 혼잡 매대 인접 비인기 매대 배치 제안
      const busyZoneId = Object.entries(zoneMap).sort((a,b)=>b[1]-a[1])[0];
      const emptyZone = ZONES.find(z => !zoneMap[z.id] || zoneMap[z.id] === 0);
      if (busyZoneId && emptyZone) {
        const busyLabel = ZONES.find(z=>z.id===busyZoneId[0])?.label;
        if (busyLabel && busyZoneId[1] >= 3) {
          insights.push(`"${busyLabel}" 혼잡 (현재 ${busyZoneId[1]}명) 인근에 "${emptyZone.label}" 배치 시 자연스러운 동선 분산 효과 기대`);
        }
      }

      // [7] 매대 간 시너지: A 매대 구매자가 B도 자주 구매하는 쌍 찾기
      const pairCount = {};
      Object.values(agentZoneBuy).forEach(zoneSet => {
        const zonesArr = [...zoneSet];
        for (let i = 0; i < zonesArr.length; i++) {
          for (let j = i+1; j < zonesArr.length; j++) {
            const pair = [zonesArr[i], zonesArr[j]].sort().join('||');
            pairCount[pair] = (pairCount[pair]||0)+1;
          }
        }
      });
      const topPair = Object.entries(pairCount).sort((a,b)=>b[1]-a[1])[0];
      if (topPair && topPair[1] >= 3) {
        const [zA, zB] = topPair[0].split('||');
        const labelA = ZONES.find(z=>z.id===zA)?.label || zA;
        const labelB = ZONES.find(z=>z.id===zB)?.label || zB;
        insights.push(`"${labelA}"와 "${labelB}" 동시 구매 고객 ${topPair[1]}명 → 두 매대 인접 배치 또는 묶음 프로모션 검토`);
      }

      // ── 고객 행동 분석 ───────────────────────────────────────────────────
      // [8] 미성년 vs 성년 구매 패턴 비교
      const adultP  = allPurchases.filter(p => p.classType.startsWith('adult'));
      const minorP  = allPurchases.filter(p => p.classType.startsWith('minor'));
      const adultBuyers = new Set(adultP.map(p=>p.agentId)).size;
      const minorBuyers = new Set(minorP.map(p=>p.agentId)).size;
      if (adultBuyers > 0 && minorBuyers > 0) {
        const adultAvg = Math.round(adultP.reduce((s,p)=>s+p.price,0)/adultBuyers);
        const minorAvg = Math.round(minorP.reduce((s,p)=>s+p.price,0)/minorBuyers);
        if (adultAvg > minorAvg * 1.3) {
          insights.push(`성년 객단가(₩${adultAvg.toLocaleString()})가 미성년(₩${minorAvg.toLocaleString()})보다 ${Math.round(adultAvg/minorAvg*10)/10}배 높음 → 성년 고객 체류 시간 연장 전략 우선 투자 권장`);
        }
      }

      // [9] 빠르게 이탈하는 고객군 (방문 매대 수 1개 이하 + 구매 없음)
      const clsNoBuyEarlyExit = {};
      agents.forEach(a => {
        const hasBuy = allPurchases.some(p=>p.agentId===a.id);
        if (!hasBuy && (a.visitedZones||[]).length <= 1 && a.state === 'leaving') {
          clsNoBuyEarlyExit[a.classType] = (clsNoBuyEarlyExit[a.classType]||0)+1;
        }
      });
      const topExitCls = Object.entries(clsNoBuyEarlyExit).sort((a,b)=>b[1]-a[1])[0];
      if (topExitCls && topExitCls[1] >= 3) {
        insights.push(`${CUSTOMER_CLASSES[topExitCls[0]].label} ${topExitCls[1]}명이 구매 없이 조기 이탈 → 입구 근처 해당 고객 선호 매대 배치 또는 입점 유도 콘텐츠 필요`);
      }

      // [10] 재방문 가치 높은 고객군 (구매 건수 대비 금액 높음)
      const clsValueMap = CLASS_KEYS
        .filter(k => clsBuyerMap[k] && clsBuyerMap[k].size > 0)
        .map(k => {
          const buyCnt = allPurchases.filter(p=>p.classType===k).length;
          const buyerCnt = clsBuyerMap[k].size;
          return { k, valuePerTxn: buyCnt > 0 ? Math.round((clsRevMap[k]||0)/buyCnt) : 0, buyerCnt };
        })
        .sort((a,b) => b.valuePerTxn - a.valuePerTxn);
      if (clsValueMap.length >= 2 && clsValueMap[0].valuePerTxn > 0) {
        const hv = clsValueMap[0];
        insights.push(`건당 구매액 최고: ${CUSTOMER_CLASSES[hv.k].label} ₩${hv.valuePerTxn.toLocaleString()} → 해당 고객군 재방문 유도(쿠폰/멤버십) 시 매출 레버리지 극대화`);
      }

      // ── 운영 효율 ────────────────────────────────────────────────────────
      // [11] 계산대 혼잡도
      if (checkoutCnt > 4) {
        insights.push(`계산대 혼잡 (${checkoutCnt}명 대기) → 추가 계산대 운영 또는 셀프계산대 도입 검토`);
      } else if (checkoutCnt === 0 && total > 5) {
        insights.push(`계산대 대기 없음 (매장 내 ${total}명) → 현재 계산대 운영 인력 효율적, 피크 시간 대비 인력 배치 최적화 유지`);
      }

      // [12] 피크/비피크: 현재 혼잡도 vs 전체 평균
      if (totalVisitors > 10 && simSecs > 30) {
        const avgDensity = totalVisitors / (simSecs / 60);
        const curDensity = total;
        if (curDensity > avgDensity * 1.4) {
          insights.push(`현재 매장 밀도(${total}명)가 시간 평균 대비 ${Math.round(curDensity/avgDensity*10)/10}배 높음 → 피크 구간 진입, 보충 인력 및 재고 점검 권장`);
        }
      }
    }

    // ── KPI 지표 계산 ─────────────────────────────────────────────────────
    // 구매 전환율: 1건 이상 구매한 고객 / 총 방문 고객
    const buyerSet = new Set(allPurchases.map(p => p.agentId));
    const convRate = totalVisitors > 0 ? Math.round(buyerSet.size / totalVisitors * 100) : 0;
    // 평균 구매액 (구매한 고객 기준)
    const avgBasket = buyerSet.size > 0 ? Math.round(totalRev / buyerSet.size) : 0;
    // 시간당 예상 매출
    const revenuePerHr = simSecs > 0 ? Math.round(totalRev / simSecs * 3600) : 0;

    setStats({ counts, total, checkoutCnt, topZones, revByClass, dwellByZone, insights, simTime: fmtTime(simSecs), totalRev,
               totalVisitors, convRate, avgBasket, revenuePerHr });
  }, []);

  // ── 히트맵 그리기 ──────────────────────────────────────────────────────────
  function drawHeatmap(ctx) {
    const sim  = simRef.current;
    const mode = heatmapModeRef.current;
    const oc   = heatOffRef.current;
    if (!oc) return;

    let grid;
    if (mode === 'combined') {
      grid = new Float32Array(HEAT_W * HEAT_H);
      for (const cls of CLASS_KEYS) { const s = sim.heatGrids[cls]; for (let i=0;i<grid.length;i++) grid[i]+=s[i]; }
    } else {
      grid = sim.heatGrids[mode];
    }
    let maxVal = 0;
    for (let i=0;i<grid.length;i++) if (grid[i]>maxVal) maxVal=grid[i];
    if (maxVal === 0) return;

    const offCtx = oc.getContext('2d');
    const id     = offCtx.createImageData(HEAT_W, HEAT_H);
    const d      = id.data;
    for (let i=0; i<HEAT_W*HEAT_H; i++) {
      const v = grid[i] / maxVal;
      if (v > 0.005) {
        const [r,g,b] = viridisColor(Math.min(v, 1));
        // 낮은 값도 잘 보이도록 alpha를 감마 보정 후 최소 40 보장
        const alpha = Math.max(40, Math.floor(Math.pow(v, 0.5) * 210));
        d[i*4]=r; d[i*4+1]=g; d[i*4+2]=b; d[i*4+3]=alpha;
      }
    }
    offCtx.putImageData(id, 0, 0);

    ctx.imageSmoothingEnabled  = true;
    ctx.imageSmoothingQuality  = 'medium';
    ctx.drawImage(oc, 0, 0, CANVAS_W, CANVAS_H);
  }

  // ── Canvas 렌더링 ─────────────────────────────────────────────────────────
  const drawFrame = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const sim = simRef.current;

    // 배경
    ctx.fillStyle = '#0f0f1e';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    // 바닥 격자
    ctx.strokeStyle = '#181828'; ctx.lineWidth = 1;
    for (let x=0; x<=CANVAS_W; x+=40) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,CANVAS_H); ctx.stroke(); }
    for (let y=0; y<=CANVAS_H; y+=40) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(CANVAS_W,y); ctx.stroke(); }

    // 히트맵 오버레이
    if (showHeatmapRef.current) drawHeatmap(ctx);

    // ROI 점선
    if (showROIRef.current) {
      ctx.setLineDash([4,3]); ctx.strokeStyle = 'rgba(80,160,255,0.3)'; ctx.lineWidth = 1;
      ZONES.forEach(z => ctx.strokeRect(z.x-12, z.y-8, z.w+24, z.h+32));
      ctx.setLineDash([]);
    }

    // 매대 (진열대)
    ZONES.forEach(z => {
      ctx.fillStyle   = '#1e1e3a'; ctx.strokeStyle = '#3a3a6a'; ctx.lineWidth = 2;
      ctx.fillRect(z.x, z.y, z.w, z.h); ctx.strokeRect(z.x, z.y, z.w, z.h);
      ctx.fillStyle = '#9090c0'; ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(z.label, z.x + z.w/2, z.y + z.h/2 + 5);
    });

    // 계산대
    CHECKOUTS.forEach((co, i) => {
      ctx.fillStyle = '#1a2e1a'; ctx.strokeStyle = '#3a6a3a'; ctx.lineWidth = 2;
      ctx.fillRect(co.x, co.y, co.w, co.h); ctx.strokeRect(co.x, co.y, co.w, co.h);
      ctx.fillStyle = '#70c070'; ctx.font = 'bold 10px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(`계산대${i+1}`, co.x + co.w/2, co.y + co.h/2 + 4);
    });

    // 입구/출구
    ctx.fillStyle = '#1a3a1a';
    ctx.fillRect(ENT_X - 40, CANVAS_H - 18, 80, 18);
    ctx.fillStyle = '#80ff80'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('입구 / 출구', ENT_X, CANVAS_H - 4);

    // 커스텀 장애물
    _customBarriers.forEach(b => {
      ctx.fillStyle = '#2a0a0a';
      ctx.strokeStyle = '#ff4444';
      ctx.lineWidth = 2;
      ctx.fillRect(b.x, b.y, b.w, b.h);
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      // X 표시
      ctx.strokeStyle = '#ff6666'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(b.x+4, b.y+4); ctx.lineTo(b.x+b.w-4, b.y+b.h-4); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(b.x+b.w-4, b.y+4); ctx.lineTo(b.x+4, b.y+b.h-4); ctx.stroke();
    });

    // 에이전트
    sim.agents.forEach(a => {
      if (a.state === 'removed') return;
      const cls = CUSTOMER_CLASSES[a.classType];
      const r   = 6;

      // 그림자
      ctx.beginPath(); ctx.arc(a.x+1, a.y+1, r, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(0,0,0,0.4)'; ctx.fill();

      // 원
      ctx.beginPath(); ctx.arc(a.x, a.y, r, 0, Math.PI*2);
      ctx.fillStyle   = cls.color + 'bb'; ctx.fill();
      ctx.strokeStyle = cls.color; ctx.lineWidth = 1.5; ctx.stroke();

      // 방향 화살표
      const spd = Math.hypot(a.vx, a.vy);
      if (spd > 0.3) {
        ctx.beginPath();
        ctx.moveTo(a.x + (a.vx/spd)*(r+1), a.y + (a.vy/spd)*(r+1));
        ctx.lineTo(a.x + (a.vx/spd)*(r+6), a.y + (a.vy/spd)*(r+6));
        ctx.strokeStyle = cls.color; ctx.lineWidth = 1.5; ctx.stroke();
      }

      // 쇼핑 진행 링 (노란색 arc)
      if (a.state === 'shopping' && a.dwellDuration > 0) {
        const prog = Math.min(a.dwellTimer / a.dwellDuration, 1);
        ctx.beginPath();
        ctx.arc(a.x, a.y, r+3, -Math.PI/2, -Math.PI/2 + prog * Math.PI*2);
        ctx.strokeStyle = '#ffee00'; ctx.lineWidth = 1.5; ctx.stroke();
      }
    });

    // 에이전트 수 오버레이
    const alive = sim.agents.filter(a => a.state !== 'removed').length;
    ctx.fillStyle = 'rgba(10,10,25,0.75)';
    ctx.fillRect(CANVAS_W - 165, 8, 157, 22);
    ctx.fillStyle = '#8080cc'; ctx.font = '11px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText(`에이전트: ${alive}명  |  ${fmtTime(sim.simSecs)}`, CANVAS_W - 10, 23);
    ctx.textAlign = 'left';
  }, []);

  // ── 시뮬레이션 스텝 ───────────────────────────────────────────────────────
  const simStep = useCallback(() => {
    const sim = simRef.current;
    if (!sim.running) return;

    const dt = (1/60) * sim.speed;
    sim.simSecs += dt;
    sim.tick++;

    // 에이전트 스폰
    sim.spawnTimer += dt;
    const interval = 60 / Math.max(sim.spawnRate, 0.1);
    const aliveCount = sim.agents.filter(a => a.state !== 'removed').length;
    if (sim.spawnTimer >= interval && aliveCount < 80) {
      const a = spawnAgent(pickClass(sim.classRatios));
      a.entryTime = sim.simSecs;
      sim.agents.push(a);
      sim.totalVisitors++;
      sim.spawnTimer = 0;
    }

    // 에이전트 업데이트
    for (const a of sim.agents) {
      if (a.state !== 'removed') stepAgent(a, sim.agents, sim.allPurchases, sim.simSecs, dt);
    }

    // 제거된 에이전트 정리 (120프레임마다)
    if (sim.tick % 120 === 0) sim.agents = sim.agents.filter(a => a.state !== 'removed');

    // 히트맵 감쇠 (60프레임마다 = ~1초 간격)
    if (sim.tick % 60 === 0) {
      for (const key of CLASS_KEYS) {
        const g = sim.heatGrids[key];
        for (let i = 0, len = g.length; i < len; i++) g[i] *= 0.97;
      }
    }

    // 히트맵 누적 (사전 계산된 가우시안 커널 → exp() 연산 제거)
    for (const a of sim.agents) {
      if (a.state === 'leaving' || a.state === 'removed') continue;
      const hcx = Math.floor(a.x / 4), hcy = Math.floor(a.y / 4);
      for (const { dx, dy, w } of GAUSS_OFFSETS) {
        const nx = hcx + dx, ny = hcy + dy;
        if (nx >= 0 && nx < HEAT_W && ny >= 0 && ny < HEAT_H)
          sim.heatGrids[a.classType][ny * HEAT_W + nx] += 0.06 * w;
      }
    }

    if (sim.tick % 30 === 0) computeStats();
  }, [computeStats]);

  // ── 애니메이션 루프 ────────────────────────────────────────────────────────
  useEffect(() => {
    let last = 0;
    function loop(t) {
      if (t - last >= 16) { simStep(); drawFrame(); last = t; }
      rafRef.current = requestAnimationFrame(loop);
    }
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [simStep, drawFrame]);

  // ── 컨트롤 핸들러 ─────────────────────────────────────────────────────────
  // ── 캔버스 좌표 변환 헬퍼 ──────────────────────────────────────────────────
  const toCanvasCoords = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = CANVAS_W / rect.width;
    const scaleY = CANVAS_H / rect.height;
    return {
      cx: (e.clientX - rect.left) * scaleX,
      cy: (e.clientY - rect.top)  * scaleY,
    };
  };

  // ── 장애물 클릭 핸들러 (좌클릭: 추가/제거, 우클릭: 제거) ──────────────────
  // 장애물 변경 후 공통 처리: PF 그리드 무효화 + 모든 에이전트 경로 초기화
  const invalidatePaths = useCallback(() => {
    _pfDirty = true;
    simRef.current.agents.forEach(a => { a.pathTarget = null; a.path = []; });
    setBarrierCount(_customBarriers.length);
  }, []);

  const handleCanvasClick = useCallback((e) => {
    if (!barrierModeRef.current) return;
    const { cx, cy } = toCanvasCoords(e);
    const bx = Math.round(cx / 20) * 20 - BARRIER_SIZE / 2;
    const by = Math.round(cy / 20) * 20 - BARRIER_SIZE / 2;
    const idx = _customBarriers.findIndex(b => Math.abs(b.x - bx) < 4 && Math.abs(b.y - by) < 4);
    if (idx >= 0) _customBarriers.splice(idx, 1);
    else _customBarriers.push({ x: bx, y: by, w: BARRIER_SIZE, h: BARRIER_SIZE });
    invalidatePaths();
  }, [invalidatePaths]);

  const handleCanvasContextMenu = useCallback((e) => {
    e.preventDefault();
    if (!barrierModeRef.current) return;
    const { cx, cy } = toCanvasCoords(e);
    const idx = _customBarriers.findIndex(b =>
      cx >= b.x && cx <= b.x + b.w && cy >= b.y && cy <= b.y + b.h
    );
    if (idx >= 0) { _customBarriers.splice(idx, 1); invalidatePaths(); }
  }, [invalidatePaths]);

  // ── 배치 시뮬레이션 핸들러 ───────────────────────────────────────────────
  const handleRunBatch = useCallback(() => {
    if (batchStatus === 'running') {
      workerRef.current?.postMessage({ type: 'abort' });
      workerRef.current?.terminate();
      workerRef.current = null;
      setBatchStatus('idle');
      return;
    }
    setBatchStatus('running');
    setBatchProgress(0);
    setBatchData(null);

    const worker = new Worker(
      new URL('./src/simulationWorker.js', import.meta.url),
      { type: 'module' }
    );
    workerRef.current = worker;

    worker.postMessage({
      type: 'run',
      config: {
        dailyVisitors,
        operatingHours,
        classRatios:   { ...classRatios },
        dwellScale,
        purchaseBonus,
        weekendBoost,
        availableZones,
      },
      days: 120,
    });

    worker.onmessage = ({ data }) => {
      if (data.type === 'progress') {
        setBatchProgress(data.pct);
      } else if (data.type === 'done') {
        // ArrayBuffer → Float32Array 복원
        const heatGridsRestored = {};
        const CLASS_KEYS_LOCAL = ['adult_male', 'adult_female', 'minor_male', 'minor_female'];
        CLASS_KEYS_LOCAL.forEach(cls => {
          if (data.heatGrids[cls]) heatGridsRestored[cls] = data.heatGrids[cls];
        });
        setBatchData({ ...data, heatGrids: heatGridsRestored });
        setBatchStatus('done');
        setBatchProgress(100);
        setActiveTab('analytics');
        worker.terminate();
        workerRef.current = null;
      } else if (data.type === 'error') {
        console.error('Worker error:', data.message);
        setBatchStatus('idle');
        worker.terminate();
        workerRef.current = null;
      }
    };

    worker.onerror = (e) => {
      console.error('Worker error:', e);
      setBatchStatus('idle');
      worker.terminate();
      workerRef.current = null;
    };
  }, [batchStatus, dailyVisitors, operatingHours, classRatios, dwellScale, purchaseBonus, weekendBoost, availableZones]);

  const handleToggle = () => {
    simRef.current.running = !simRef.current.running;
    setRunning(simRef.current.running);
  };

  const handleReset = () => {
    const sim = simRef.current;
    sim.running = false; sim.agents = []; sim.tick = 0;
    sim.spawnTimer = 0; sim.simSecs = 0; sim.allPurchases = []; sim.totalVisitors = 0;
    for (const cls of CLASS_KEYS) sim.heatGrids[cls] = new Float32Array(HEAT_W * HEAT_H);
    _customBarriers.length = 0;
    _pfDirty = true; _pfGrid = null;
    _uid = 0;
    setRunning(false);
    setBarrierCount(0);
    setStats({ counts:{ adult_male:0,adult_female:0,minor_male:0,minor_female:0 }, total:0, checkoutCnt:0, topZones:[], revByClass:[], dwellByZone:[], insights:[], simTime:'00:00', totalRev:0,
               totalVisitors:0, convRate:0, avgBasket:0, revenuePerHr:0 });
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) { ctx.fillStyle = '#0f0f1e'; ctx.fillRect(0,0,CANVAS_W,CANVAS_H); }
  };

  // ── CSV 다운로드 ──────────────────────────────────────────────────────────
  const downloadPurchaseCSV = () => {
    const header = '고객ID,고객분류,고객분류명,방문매대,상품명,가격,체류시간_초,구매시각\n';
    const rows = simRef.current.allPurchases.map(p =>
      `${p.agentId},${p.classType},${CUSTOMER_CLASSES[p.classType].label},${p.zoneLabel},${p.item},${p.price},${(p.dwellTime||0).toFixed(1)},${p.time}`
    ).join('\n');
    dlCSV('purchase_history.csv', header + rows);
  };

  const downloadSummaryCSV = () => {
    const header = '분류,분류명,총구매건수,총구매금액,평균체류시간_초,가장많이방문한매대\n';
    const rows = CLASS_KEYS.map(cls => {
      const p   = simRef.current.allPurchases.filter(x => x.classType === cls);
      const tot = p.reduce((s,x) => s+x.price, 0);
      const avg = p.length ? (p.reduce((s,x)=>s+(x.dwellTime||0),0)/p.length).toFixed(1) : 0;
      const zc  = {};
      p.forEach(x => zc[x.zoneLabel] = (zc[x.zoneLabel]||0)+1);
      const topZ = Object.entries(zc).sort((a,b)=>b[1]-a[1])[0]?.[0] || '-';
      return `${cls},${CUSTOMER_CLASSES[cls].label},${p.length},${tot},${avg},${topZ}`;
    }).join('\n');
    dlCSV('purchase_summary.csv', header + rows);
  };

  const downloadDwellCSV = () => {
    const header = '매대,' + CLASS_KEYS.map(cls => CUSTOMER_CLASSES[cls].short).join(',') + '\n';
    const rows = ZONES.map(z => {
      const cols = CLASS_KEYS.map(cls => {
        const p = simRef.current.allPurchases.filter(x => x.classType === cls && x.zoneId === z.id);
        return p.length ? (p.reduce((s,x)=>s+(x.dwellTime||0),0)/p.length).toFixed(1) : 0;
      });
      return `${z.label},${cols.join(',')}`;
    }).join('\n');
    dlCSV('dwell_analysis.csv', header + rows);
  };

  // 히트맵 이미지 저장 (클래스별 + 통합 + 2×2 비교)
  const saveHeatmapImages = () => {
    const ts = Date.now();

    // 단일 히트맵 저장 함수
    const saveOne = (mode, filename) => {
      const oc  = document.createElement('canvas');
      oc.width = CANVAS_W; oc.height = CANVAS_H;
      const ctx = oc.getContext('2d');
      ctx.fillStyle = '#0f0f1e'; ctx.fillRect(0,0,CANVAS_W,CANVAS_H);
      ZONES.forEach(z => {
        ctx.fillStyle='#1e1e3a'; ctx.strokeStyle='#3a3a6a'; ctx.lineWidth=2;
        ctx.fillRect(z.x,z.y,z.w,z.h); ctx.strokeRect(z.x,z.y,z.w,z.h);
        ctx.fillStyle='#9090c0'; ctx.font='bold 12px sans-serif'; ctx.textAlign='center';
        ctx.fillText(z.label, z.x+z.w/2, z.y+z.h/2+5);
      });
      const sim = simRef.current;
      let grid;
      if (mode === 'combined') {
        grid = new Float32Array(HEAT_W*HEAT_H);
        for (const cls of CLASS_KEYS) { const s=sim.heatGrids[cls]; for(let i=0;i<grid.length;i++) grid[i]+=s[i]; }
      } else {
        grid = sim.heatGrids[mode];
      }
      let maxV = 0; for (let i=0;i<grid.length;i++) if(grid[i]>maxV) maxV=grid[i];
      if (maxV > 0) {
        const id = ctx.createImageData(HEAT_W, HEAT_H);
        const d  = id.data;
        for (let i=0;i<HEAT_W*HEAT_H;i++) {
          const v = grid[i]/maxV;
          if (v>0.005) { const [r,g,b]=viridisColor(Math.min(v,1)); const alpha=Math.max(40,Math.floor(Math.pow(v,0.5)*210)); d[i*4]=r;d[i*4+1]=g;d[i*4+2]=b;d[i*4+3]=alpha; }
        }
        const tmpOc = document.createElement('canvas');
        tmpOc.width=HEAT_W; tmpOc.height=HEAT_H;
        tmpOc.getContext('2d').putImageData(id, 0, 0);
        ctx.imageSmoothingEnabled=true; ctx.imageSmoothingQuality='medium';
        ctx.drawImage(tmpOc, 0, 0, CANVAS_W, CANVAS_H);
      }
      // 타이틀
      ctx.fillStyle='rgba(0,0,0,0.5)'; ctx.fillRect(0,0,CANVAS_W,30);
      ctx.fillStyle='#e0e0ff'; ctx.font='bold 14px sans-serif'; ctx.textAlign='left';
      const label = mode === 'combined' ? '통합 히트맵' : CUSTOMER_CLASSES[mode].label;
      ctx.fillText(`히트맵: ${label}`, 10, 20); ctx.textAlign='left';
      const link = document.createElement('a');
      link.download = filename; link.href = oc.toDataURL('image/png'); link.click();
    };

    saveOne('combined', `heatmap_combined_${ts}.png`);
    setTimeout(() => {
      CLASS_KEYS.forEach((cls, i) =>
        setTimeout(() => saveOne(cls, `heatmap_${cls}_${ts}.png`), i * 300)
      );
    }, 300);

    // 2×2 비교
    setTimeout(() => {
      const BIG_W = CANVAS_W * 2, BIG_H = CANVAS_H * 2;
      const bigOc  = document.createElement('canvas');
      bigOc.width = BIG_W; bigOc.height = BIG_H;
      const bigCtx = bigOc.getContext('2d');
      bigCtx.fillStyle = '#0a0a18'; bigCtx.fillRect(0,0,BIG_W,BIG_H);
      CLASS_KEYS.forEach((cls, idx) => {
        const ox = (idx % 2) * CANVAS_W, oy = Math.floor(idx / 2) * CANVAS_H;
        const miniOc = document.createElement('canvas');
        miniOc.width = CANVAS_W; miniOc.height = CANVAS_H;
        const mCtx = miniOc.getContext('2d');
        mCtx.fillStyle='#0f0f1e'; mCtx.fillRect(0,0,CANVAS_W,CANVAS_H);
        const sim  = simRef.current;
        const grid = sim.heatGrids[cls];
        let maxV=0; for(let i=0;i<grid.length;i++) if(grid[i]>maxV) maxV=grid[i];
        if (maxV > 0) {
          const id = mCtx.createImageData(HEAT_W,HEAT_H);
          const d  = id.data;
          for (let i=0;i<HEAT_W*HEAT_H;i++) {
            const v = grid[i]/maxV;
            if (v>0.005){const[r,g,b]=viridisColor(Math.min(v,1));const alpha=Math.max(40,Math.floor(Math.pow(v,0.5)*210));d[i*4]=r;d[i*4+1]=g;d[i*4+2]=b;d[i*4+3]=alpha;}
          }
          const tmp=document.createElement('canvas'); tmp.width=HEAT_W; tmp.height=HEAT_H;
          tmp.getContext('2d').putImageData(id,0,0);
          mCtx.imageSmoothingEnabled=true; mCtx.drawImage(tmp,0,0,CANVAS_W,CANVAS_H);
        }
        ZONES.forEach(z=>{mCtx.fillStyle='rgba(30,30,58,0.7)';mCtx.fillRect(z.x,z.y,z.w,z.h);mCtx.strokeStyle='#3a3a6a';mCtx.strokeRect(z.x,z.y,z.w,z.h);});
        mCtx.fillStyle='rgba(0,0,0,0.55)'; mCtx.fillRect(0,0,CANVAS_W,26);
        mCtx.fillStyle=CUSTOMER_CLASSES[cls].color; mCtx.font='bold 13px sans-serif'; mCtx.textAlign='left';
        mCtx.fillText(`${CUSTOMER_CLASSES[cls].icon} ${CUSTOMER_CLASSES[cls].label}`, 8, 18);
        bigCtx.drawImage(miniOc, ox, oy);
      });
      const link=document.createElement('a');
      link.download=`heatmap_comparison_${ts}.png`; link.href=bigOc.toDataURL('image/png'); link.click();
    }, 1800);
  };

  // ── 렌더 ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ display:'flex', height:'100vh', minWidth:'1200px', background:'#0a0a18', color:'#e0e0ff', fontFamily:'system-ui,sans-serif', overflow:'hidden' }}>

      {/* ── 좌측 사이드바 ── */}
      <aside style={{ width:'280px', flexShrink:0, background:'#10101e', borderRight:'1px solid #222244', overflowY:'auto', padding:'12px', display:'flex', flexDirection:'column', gap:'10px' }}>
        <div style={{ textAlign:'center', fontSize:'15px', fontWeight:'bold', color:'#b0b0e0', padding:'6px 0 2px' }}>
          🏪 매장 시뮬레이션
        </div>

        {/* KPI 프리셋 */}
        <Card>
          <SecTitle>📊 업태 KPI 프리셋</SecTitle>
          <div style={{ display:'flex', flexDirection:'column', gap:'5px', marginTop:'4px' }}>
            {Object.entries(KPI_PRESETS).map(([key, p]) => (
              <button key={key} onClick={() => applyPreset(key)}
                style={{
                  background: kpiPreset === key ? '#1e2e4a' : '#12121e',
                  color: kpiPreset === key ? '#80b0ff' : '#8080aa',
                  border: `1px solid ${kpiPreset === key ? '#3a5aaa' : '#222244'}`,
                  borderRadius:'4px', padding:'6px 8px', cursor:'pointer',
                  textAlign:'left', fontSize:'11px',
                }}>
                <div style={{ fontWeight:'bold', marginBottom:'2px' }}>{p.label}</div>
                <div style={{ fontSize:'10px', opacity:0.8 }}>{p.desc}</div>
              </button>
            ))}
          </div>
        </Card>

        {/* 매장 파라미터 커스텀 */}
        <Card>
          <SecTitle style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
            <span>매장 파라미터</span>
            {isCustom && <span style={{ fontSize:'9px', color:'#ffaa40', background:'#2a1a0a', padding:'1px 6px', borderRadius:'8px' }}>커스텀</span>}
          </SecTitle>
          <div style={{ display:'flex', flexDirection:'column', gap:'8px', marginTop:'6px' }}>
            {/* 일 방문객 수 */}
            <div>
              <div style={{ fontSize:'10px', color:'#8080aa', marginBottom:'3px' }}>일 방문객 수</div>
              <div style={{ display:'flex', alignItems:'center', gap:'6px' }}>
                <input type="number" min={50} max={5000} step={50} value={dailyVisitors}
                  onChange={e => { setDailyVisitors(Math.max(50, +e.target.value || 50)); setIsCustom(true); setKpiPreset(null); }}
                  style={{ width:'70px', background:'#12121e', color:'#e0e0ff', border:'1px solid #333366', borderRadius:'3px', padding:'3px 6px', fontSize:'12px', textAlign:'right' }}
                />
                <span style={{ fontSize:'11px', color:'#8080aa' }}>명/일</span>
              </div>
            </div>
            {/* 영업시간 */}
            <div>
              <div style={{ fontSize:'10px', color:'#8080aa', marginBottom:'3px' }}>영업시간: {operatingHours}시간</div>
              <input type="range" min={4} max={24} step={1} value={operatingHours}
                onChange={e => { setOperatingHours(+e.target.value); setIsCustom(true); setKpiPreset(null); }}
                style={{ width:'100%', accentColor:'#5a7aff' }}
              />
              <div style={{ display:'flex', justifyContent:'space-between', fontSize:'9px', color:'#606080' }}>
                <span>4h</span><span>24h</span>
              </div>
            </div>
            {/* 고급 설정 토글 */}
            <button onClick={() => setShowAdvanced(v => !v)}
              style={{ background:'none', border:'none', color:'#6080cc', cursor:'pointer', fontSize:'11px', textAlign:'left', padding:'2px 0' }}>
              {showAdvanced ? '▼' : '▶'} 고급 설정
            </button>
            {showAdvanced && (
              <div style={{ display:'flex', flexDirection:'column', gap:'8px', padding:'6px', background:'#0a0a16', borderRadius:'4px', border:'1px solid #1a1a3a' }}>
                {/* 체류시간 배율 */}
                <div>
                  <div style={{ fontSize:'10px', color:'#8080aa', marginBottom:'2px' }}>체류시간 배율: {dwellScale.toFixed(2)}x</div>
                  <input type="range" min={0.3} max={3.0} step={0.05} value={dwellScale}
                    onChange={e => { setDwellScale(+e.target.value); setIsCustom(true); setKpiPreset(null); }}
                    style={{ width:'100%', accentColor:'#5a7aff' }}
                  />
                </div>
                {/* 구매확률 보정 */}
                <div>
                  <div style={{ fontSize:'10px', color:'#8080aa', marginBottom:'2px' }}>구매확률 보정: {purchaseBonus >= 0 ? '+' : ''}{(purchaseBonus * 100).toFixed(0)}%</div>
                  <input type="range" min={-0.3} max={0.3} step={0.01} value={purchaseBonus}
                    onChange={e => { setPurchaseBonus(+e.target.value); setIsCustom(true); setKpiPreset(null); }}
                    style={{ width:'100%', accentColor:'#5a7aff' }}
                  />
                </div>
                {/* 주말 배율 */}
                <div>
                  <div style={{ fontSize:'10px', color:'#8080aa', marginBottom:'2px' }}>주말 배율: {weekendBoost.toFixed(2)}x</div>
                  <input type="range" min={0.1} max={2.0} step={0.05} value={weekendBoost}
                    onChange={e => { setWeekendBoost(+e.target.value); setIsCustom(true); setKpiPreset(null); }}
                    style={{ width:'100%', accentColor:'#5a7aff' }}
                  />
                </div>
                {/* 활성 구역 */}
                <div>
                  <div style={{ fontSize:'10px', color:'#8080aa', marginBottom:'4px' }}>매장 구역 ({availableZones.length}/{ZONES.length})</div>
                  <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'2px' }}>
                    {ZONES.map(z => (
                      <label key={z.id} style={{ fontSize:'10px', color: availableZones.includes(z.id) ? '#c0c0e0' : '#505070', cursor:'pointer', display:'flex', alignItems:'center', gap:'3px' }}>
                        <input type="checkbox" checked={availableZones.includes(z.id)}
                          onChange={e => {
                            setAvailableZones(prev => e.target.checked ? [...prev, z.id] : prev.filter(id => id !== z.id));
                            setIsCustom(true); setKpiPreset(null);
                          }}
                          style={{ accentColor:'#5a7aff', width:'12px', height:'12px' }}
                        />
                        {z.label}
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* 배치 시뮬레이션 */}
        <Card>
          <SecTitle>🚀 4개월 배치 시뮬레이션</SecTitle>
          <Btn
            bg={batchStatus === 'running' ? '#4a1a1a' : batchStatus === 'done' ? '#1a3a2a' : '#1a2a4a'}
            onClick={handleRunBatch}
          >
            {batchStatus === 'running'
              ? `⏹ 중단 (${batchProgress}%)`
              : batchStatus === 'done'
              ? '🔄 재실행 (120일)'
              : '▶ 120일 배치 실행'}
          </Btn>
          {batchStatus === 'running' && (
            <div style={{ marginTop: '8px' }}>
              <div style={{ height: '5px', background: '#1a1a2a', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${batchProgress}%`, background: 'linear-gradient(90deg,#3a7aff,#7a3aff)', borderRadius: '3px', transition: 'width 0.3s' }} />
              </div>
              <div style={{ fontSize: '10px', color: '#6060aa', marginTop: '3px', textAlign: 'center' }}>{batchProgress}% 완료</div>
            </div>
          )}
          {batchStatus === 'done' && (
            <div style={{ fontSize: '11px', color: '#40cc80', marginTop: '5px' }}>
              ✓ 완료! 분석 탭에서 결과 확인
            </div>
          )}
          {batchData && (
            <button onClick={() => setActiveTab(t => t === 'analytics' ? 'simulation' : 'analytics')}
              style={{ marginTop: '6px', width: '100%', background: activeTab === 'analytics' ? '#2a1a4a' : '#1a1a3a', color: activeTab === 'analytics' ? '#b080ff' : '#8080aa', border: '1px solid #2a2a5a', borderRadius: '4px', padding: '5px', cursor: 'pointer', fontSize: '11px' }}>
              {activeTab === 'analytics' ? '📺 시뮬레이션 탭으로' : '📊 분석 탭으로'}
            </button>
          )}
        </Card>

        {/* 컨트롤 */}
        <Card>
          <div style={{ display:'flex', gap:'8px', justifyContent:'center' }}>
            <Btn bg={running ? '#4a1a5a' : '#1a4a1a'} onClick={handleToggle}>
              {running ? '⏸ 일시정지' : '▶ 시작'}
            </Btn>
            <Btn bg='#2a1a1a' onClick={handleReset}>🔄 리셋</Btn>
          </div>
          <SliderRow label={`속도: ${speed}x`} min={0.5} max={5} step={0.5} value={speed} onChange={setSpeed} />
          <div style={{ fontSize:'11px', color:'#6060aa', textAlign:'center', marginTop:'6px' }}>
            {stats.simTime} ·&nbsp;매장 내 {stats.total}명 ·&nbsp;누적 {stats.totalVisitors}명
          </div>
        </Card>

        {/* 유입 설정 */}
        <Card>
          <SecTitle>고객 유입 설정</SecTitle>
          <SliderRow label={`유입 속도: ${spawnRate}/분`} min={1} max={10} value={spawnRate} onChange={setSpawnRate} />
          <div style={{ fontSize:'11px', color:'#6060aa', margin:'8px 0 4px' }}>클래스 비율</div>
          {CLASS_KEYS.map(cls => (
            <SliderRow
              key={cls}
              label={`${CUSTOMER_CLASSES[cls].icon} ${CUSTOMER_CLASSES[cls].short}: ${classRatios[cls]}%`}
              labelColor={CUSTOMER_CLASSES[cls].color}
              min={0} max={100}
              value={classRatios[cls]}
              onChange={v => setClassRatios(r => ({ ...r, [cls]: v }))}
            />
          ))}
        </Card>

        {/* 표시 옵션 */}
        <Card>
          <SecTitle>표시 옵션</SecTitle>
          <CheckRow label='히트맵 오버레이' checked={showHeatmap} onChange={setShowHeatmap} />
          <CheckRow label='매대 ROI 표시'   checked={showROI}     onChange={setShowROI} />
          <div style={{ marginTop:'8px' }}>
            <div style={{ fontSize:'12px', color:'#8080aa', marginBottom:'4px' }}>히트맵 모드</div>
            <select value={heatmapMode} onChange={e => setHeatmapMode(e.target.value)} style={selectSt}>
              <option value='combined'>통합</option>
              {CLASS_KEYS.map(cls => <option key={cls} value={cls}>{CUSTOMER_CLASSES[cls].label}</option>)}
            </select>
          </div>
        </Card>

        {/* 장애물 설정 */}
        <Card>
          <SecTitle>🚧 장애물 설정</SecTitle>
          <CheckRow
            label={`장애물 배치 모드${barrierCount > 0 ? ` (${barrierCount}개)` : ''}`}
            checked={barrierMode}
            onChange={setBarrierMode}
          />
          {barrierMode && (
            <div style={{ fontSize:'11px', color:'#ffaa44', marginTop:'4px', lineHeight:'1.5' }}>
              좌클릭: 장애물 추가/제거<br />
              우클릭: 장애물 제거
            </div>
          )}
          <div style={{ marginTop:'8px' }}>
            <Btn bg='#2a1a1a' onClick={() => {
              _customBarriers.length = 0;
              setBarrierCount(0);
            }}>
              🗑 장애물 전체 초기화
            </Btn>
          </div>
        </Card>

        {/* 이벤트 */}
        <Card>
          <SecTitle>이벤트</SecTitle>
          <div style={{ display:'flex', flexDirection:'column', gap:'6px' }}>
            <Btn bg='#3a2a10' onClick={() => setSpawnRate(r => Math.min(10, r + 3))}>🏷 세일 시작 (유입 증가)</Btn>
            <Btn bg='#102a3a' onClick={() => setSpawnRate(r => Math.max(1, r - 2))}>🌙 한산 시간 (유입 감소)</Btn>
            <Btn bg='#1a2a1a' onClick={() => {
              // 계산대 안내 방송 → 쇼핑 중인 에이전트 일부를 checkout으로 전환
              simRef.current.agents.forEach(a => {
                if (a.state === 'shopping' && Math.random() < 0.4) {
                  const co = CHECKOUTS[Math.floor(Math.random() * CHECKOUTS.length)];
                  a.state = 'checkout';
                  a.currentZone = null;
                  a.targetX = co.x + co.w/2;
                  a.targetY = co.y + co.h/2;
                }
              });
            }}>📢 안내 방송 (계산대 유도)</Btn>
          </div>
        </Card>
      </aside>

      {/* ── 중앙: 탭 전환 (Canvas / 배치 분석) ── */}
      {activeTab === 'simulation' ? (
        <main style={{ flex:1, display:'flex', alignItems:'center', justifyContent:'center', padding:'10px', overflow:'hidden' }}>
          <canvas
            ref={canvasRef}
            width={CANVAS_W}
            height={CANVAS_H}
            onClick={handleCanvasClick}
            onContextMenu={handleCanvasContextMenu}
            style={{
              border: barrierMode ? '2px solid #ff4444' : '1px solid #222244',
              borderRadius:'6px', maxWidth:'100%', maxHeight:'100%', objectFit:'contain',
              cursor: barrierMode ? 'crosshair' : 'default',
            }}
          />
        </main>
      ) : (
        /* 분석 탭: Canvas + 우측 패널 영역 전체 사용 */
        <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden' }}>
          <div style={{ padding:'8px 14px 0', background:'#0a0a18', borderBottom:'1px solid #1a1a30', display:'flex', alignItems:'center', gap:'12px' }}>
            <div style={{ fontSize:'13px', fontWeight:'bold', color:'#b0b0e0' }}>📊 {batchData?.dailyStats?.length || 120}일 배치 분석 결과</div>
            <div style={{ fontSize:'11px', color:'#5050aa' }}>
              {batchData ? `총 방문객 ${batchData.totalVisitors?.toLocaleString()}명 · 구매 ${batchData.purchases?.length?.toLocaleString()}건` : ''}
            </div>
          </div>
          {batchData && <BatchAnalytics data={batchData} />}
        </div>
      )}

      {/* ── 우측 패널 (시뮬레이션 탭일 때만) ── */}
      {activeTab !== 'simulation' ? null : (
      /* ── 우측 패널 ── */
      <aside style={{ width:'320px', flexShrink:0, background:'#10101e', borderLeft:'1px solid #222244', overflowY:'auto', padding:'12px', display:'flex', flexDirection:'column', gap:'10px' }}>
        <div style={{ fontSize:'14px', fontWeight:'bold', color:'#b0b0e0' }}>📊 실시간 분석</div>

        {/* KPI 지표 */}
        <Card>
          <SecTitle>📈 핵심 KPI 지표</SecTitle>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'6px', marginTop:'6px' }}>
            {[
              { label:'누적 방문객',  value: `${stats.totalVisitors}명`,            color:'#7090ff' },
              { label:'구매 전환율',  value: `${stats.convRate}%`,                   color: stats.convRate >= 60 ? '#40cc80' : stats.convRate >= 30 ? '#ffaa44' : '#ff6060' },
              { label:'평균 구매액',  value: `₩${(stats.avgBasket||0).toLocaleString()}`, color:'#c080ff' },
              { label:'시간당 매출',  value: `₩${Math.round((stats.revenuePerHr||0)/1000)}K`, color:'#ffcc44' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ background:'#0e0e22', borderRadius:'5px', padding:'7px 8px', border:'1px solid #1e1e44' }}>
                <div style={{ fontSize:'10px', color:'#6060aa', marginBottom:'2px' }}>{label}</div>
                <div style={{ fontSize:'15px', fontWeight:'bold', color }}>{value}</div>
              </div>
            ))}
          </div>
          {stats.totalVisitors > 0 && (
            <div style={{ fontSize:'10px', color:'#5050888', marginTop:'6px', textAlign:'center' }}>
              참조 전환율: {KPI_PRESETS[kpiPreset]?.convRate ?? '-'}% (업태 기준)
            </div>
          )}
        </Card>

        {/* 인원 */}
        <Card>
          <SecTitle>현재 매장 인원: {stats.total}명</SecTitle>
          <div style={{ display:'flex', flexWrap:'wrap', gap:'5px', marginTop:'6px' }}>
            {CLASS_KEYS.map(cls => (
              <div key={cls} style={{
                display:'flex', alignItems:'center', gap:'4px', fontSize:'12px',
                background: CUSTOMER_CLASSES[cls].color + '22',
                border: `1px solid ${CUSTOMER_CLASSES[cls].color}44`,
                borderRadius:'4px', padding:'3px 8px',
              }}>
                <span>{CUSTOMER_CLASSES[cls].icon}</span>
                <span style={{ color: CUSTOMER_CLASSES[cls].color }}>{CUSTOMER_CLASSES[cls].short}</span>
                <span style={{ fontWeight:'bold' }}>{stats.counts[cls]}</span>
              </div>
            ))}
          </div>
          {stats.checkoutCnt > 0 && (
            <div style={{ fontSize:'11px', color:'#ffaa44', marginTop:'6px' }}>
              🛒 계산대 대기: {stats.checkoutCnt}명
            </div>
          )}
        </Card>

        {/* 혼잡 매대 */}
        {stats.topZones.length > 0 && (
          <Card>
            <SecTitle>🔥 혼잡 매대 Top {stats.topZones.length}</SecTitle>
            {stats.topZones.map((z, i) => (
              <div key={z.label} style={{ display:'flex', justifyContent:'space-between', fontSize:'13px', marginTop:'5px' }}>
                <span style={{ color:'#c0c0e0' }}>{i+1}. {z.label}</span>
                <span style={{ color:'#ff8040', fontWeight:'bold' }}>{z.count}명</span>
              </div>
            ))}
          </Card>
        )}

        {/* 매출 차트 */}
        <Card>
          <SecTitle>💰 클래스별 예상 매출</SecTitle>
          <div style={{ fontSize:'13px', marginBottom:'4px', color:'#e0e0ff' }}>
            총 ₩{stats.totalRev.toLocaleString()}
          </div>
          <ResponsiveContainer width='100%' height={115}>
            <BarChart data={stats.revByClass} margin={{ top:0, right:8, left:-20, bottom:0 }}>
              <XAxis dataKey='name' tick={{ fill:'#7070aa', fontSize:10 }} />
              <YAxis tick={{ fill:'#7070aa', fontSize:9 }} tickFormatter={v => v>=1000 ? `${(v/1000).toFixed(0)}K` : v} />
              <Tooltip
                contentStyle={{ background:'#161628', border:'1px solid #2a2a5a', color:'#e0e0ff', fontSize:'11px' }}
                formatter={v => [`₩${v.toLocaleString()}`, '매출']}
              />
              <Bar dataKey='value' radius={[3,3,0,0]}>
                {stats.revByClass.map((e,i) => <Cell key={i} fill={e.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* 체류 시간 차트 */}
        {stats.dwellByZone.length > 0 && (
          <Card>
            <SecTitle>⏱ 매대별 평균 체류 (초)</SecTitle>
            <ResponsiveContainer width='100%' height={145}>
              <BarChart data={stats.dwellByZone} layout='vertical' margin={{ top:0, right:28, left:26, bottom:0 }}>
                <XAxis type='number' tick={{ fill:'#7070aa', fontSize:9 }} />
                <YAxis type='category' dataKey='name' tick={{ fill:'#8080c0', fontSize:9 }} width={54} />
                <Tooltip
                  contentStyle={{ background:'#161628', border:'1px solid #2a2a5a', color:'#e0e0ff', fontSize:'11px' }}
                  formatter={v => [`${v}초`, '평균 체류']}
                />
                <Bar dataKey='avg' fill='#5a7aff' radius={[0,3,3,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        )}

        {/* 인사이트 */}
        {stats.insights.length > 0 && (
          <Card>
            <SecTitle>💡 추천 인사이트</SecTitle>
            {stats.insights.map((ins, i) => (
              <div key={i} style={{
                fontSize:'11px', marginTop:'6px', padding:'6px 8px', lineHeight:'1.5',
                background:'#161630', borderRadius:'4px', borderLeft:'3px solid #4a6aff',
              }}>
                {ins}
              </div>
            ))}
          </Card>
        )}

        {/* 다운로드 */}
        <Card>
          <SecTitle>📥 데이터 다운로드</SecTitle>
          <div style={{ display:'flex', flexDirection:'column', gap:'7px', marginTop:'6px' }}>
            <Btn bg='#1a2a4a' onClick={saveHeatmapImages}>
              🗺 히트맵 이미지 저장 (통합 + 4종 + 비교)
            </Btn>
            <Btn bg='#1a3a1a' onClick={downloadPurchaseCSV}>📋 구매 내역 CSV</Btn>
            <Btn bg='#1a3a1a' onClick={downloadSummaryCSV}>📊 구매 통계 CSV</Btn>
            <Btn bg='#1a3a1a' onClick={downloadDwellCSV}>⏱ 체류 분석 CSV</Btn>
          </div>
        </Card>
      </aside>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// UI 서브 컴포넌트
// ═══════════════════════════════════════════════════════════════════════════

function Card({ children }) {
  return (
    <div style={{ background:'#161628', border:'1px solid #222244', borderRadius:'6px', padding:'10px' }}>
      {children}
    </div>
  );
}

function SecTitle({ children }) {
  return (
    <div style={{ fontSize:'11px', fontWeight:'bold', color:'#7070b0', textTransform:'uppercase', letterSpacing:'0.5px', marginBottom:'5px' }}>
      {children}
    </div>
  );
}

function Btn({ bg, onClick, children }) {
  return (
    <button onClick={onClick} style={{
      background: bg, color:'#e0e0ff', border:'none', borderRadius:'4px',
      padding:'7px 10px', cursor:'pointer', fontSize:'12px', fontWeight:'bold',
      textAlign:'left', transition:'opacity .15s',
    }}
      onMouseEnter={e => e.currentTarget.style.opacity='0.8'}
      onMouseLeave={e => e.currentTarget.style.opacity='1'}
    >
      {children}
    </button>
  );
}

function SliderRow({ label, labelColor, min, max, step=1, value, onChange }) {
  return (
    <div style={{ marginTop:'6px' }}>
      <div style={{ fontSize:'12px', color: labelColor||'#9090c0', marginBottom:'2px' }}>{label}</div>
      <input
        type='range' min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width:'100%', accentColor:'#5a7aff', cursor:'pointer' }}
      />
    </div>
  );
}

function CheckRow({ label, checked, onChange }) {
  return (
    <label style={{ display:'flex', alignItems:'center', gap:'6px', cursor:'pointer', fontSize:'13px', marginTop:'5px' }}>
      <input type='checkbox' checked={checked} onChange={e => onChange(e.target.checked)} style={{ accentColor:'#5a7aff' }} />
      {label}
    </label>
  );
}

const selectSt = {
  width:'100%', background:'#161628', color:'#e0e0ff',
  border:'1px solid #2a2a5a', padding:'5px 6px', borderRadius:'4px', fontSize:'12px',
};
