// DQN 매대 배치 최적화 (TensorFlow.js)
// State: 9개 슬롯에 어떤 zone이 배치됐는지 (81차원 one-hot)
// Action: 슬롯 pair swap, C(9,2) = 36개 이산 액션
// Reward: Σ accessWeight[slot] × expectedRevenue[zoneId at slot]

const N = 9; // 슬롯/존 수
const N_ACTIONS = 36; // C(9,2)
const STATE_DIM = 81; // N × N

const ZONE_IDS = ['ice1', 'ice2', 'ice3', 'ice4', 'ice5', 'ice6', 'beverage', 'snack1', 'snack2'];
const ZONE_LABELS = {
  ice1: '아이스크림1', ice2: '아이스크림2', ice3: '아이스크림3',
  ice4: '아이스크림4', ice5: '아이스크림5', ice6: '아이스크림6',
  beverage: '음료', snack1: '과자1', snack2: '과자2',
};

// CANVAS 800×600, 입구 (400, 584) 기준 접근성 가중치 (3×3 그리드, ZONE_IDS 순서와 동일)
const SLOT_POSITIONS = [
  { cx: 135, cy:  77.5 }, // ice1    (x:15,  y:45,  w:240, h:65)
  { cx: 395, cy:  77.5 }, // ice2    (x:275, y:45)
  { cx: 655, cy:  77.5 }, // ice3    (x:535, y:45)
  { cx: 135, cy: 197.5 }, // ice4    (x:15,  y:165)
  { cx: 395, cy: 197.5 }, // ice5    (x:275, y:165)
  { cx: 655, cy: 197.5 }, // ice6    (x:535, y:165)
  { cx: 135, cy: 317.5 }, // beverage(x:15,  y:285)
  { cx: 395, cy: 317.5 }, // snack1  (x:275, y:285)
  { cx: 655, cy: 317.5 }, // snack2  (x:535, y:285)
];
const ENT_CX = 400, ENT_CY = 584;

function computeAccessWeights() {
  const raw = SLOT_POSITIONS.map(({ cx, cy }) =>
    1 / (1 + Math.sqrt((cx - ENT_CX) ** 2 + (cy - ENT_CY) ** 2))
  );
  const sum = raw.reduce((s, v) => s + v, 0);
  return raw.map(v => v / sum);
}
const ACCESS_WEIGHTS = computeAccessWeights();

// 36개 swap actions (i, j) where i < j
const ACTIONS = [];
for (let i = 0; i < N; i++)
  for (let j = i + 1; j < N; j++)
    ACTIONS.push([i, j]);

function permToState(perm) {
  // one-hot 81차원: slot × zone 인덱스
  const state = new Float32Array(STATE_DIM);
  for (let slot = 0; slot < N; slot++) {
    const zIdx = ZONE_IDS.indexOf(perm[slot]);
    if (zIdx >= 0) state[slot * N + zIdx] = 1;
  }
  return state;
}

function computeReward(perm, expectedRevenue) {
  let r = 0;
  for (let slot = 0; slot < N; slot++) {
    r += ACCESS_WEIGHTS[slot] * (expectedRevenue[perm[slot]] ?? 0);
  }
  return r;
}

function applyAction(perm, actionIdx) {
  const next = [...perm];
  const [i, j] = ACTIONS[actionIdx];
  [next[i], next[j]] = [next[j], next[i]];
  return next;
}

class ReplayBuffer {
  constructor(capacity) {
    this.buf = [];
    this.capacity = capacity;
    this.idx = 0;
  }
  push(item) {
    if (this.buf.length < this.capacity) this.buf.push(item);
    else { this.buf[this.idx] = item; this.idx = (this.idx + 1) % this.capacity; }
  }
  sample(n) {
    const result = [];
    for (let i = 0; i < n; i++)
      result.push(this.buf[Math.floor(Math.random() * this.buf.length)]);
    return result;
  }
  get size() { return this.buf.length; }
}

export async function trainDQNOptimizer(batchData, { episodes = 500, onProgress } = {}) {
  const tf = (await import('@tensorflow/tfjs')).default ?? (await import('@tensorflow/tfjs'));

  // expectedRevenue: zone별 일평균 매출
  const expectedRevenue = {};
  const dayCount = batchData.dailyStats?.length || 1;
  ZONE_IDS.forEach(zId => {
    const total = batchData.dailyStats?.reduce((s, d) => s + (d.revenueByZone?.[zId] ?? 0), 0) ?? 0;
    expectedRevenue[zId] = total / dayCount;
  });

  // 초기 permutation: 현재 배치 순서 (ZONE_IDS 그대로)
  const initialPerm = [...ZONE_IDS];
  const baselineReward = computeReward(initialPerm, expectedRevenue);

  // Q-network 빌드
  function buildQNet() {
    const m = tf.sequential();
    m.add(tf.layers.dense({ inputShape: [STATE_DIM], units: 64, activation: 'relu' }));
    m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    m.add(tf.layers.dense({ units: N_ACTIONS }));
    const huberLoss = (yTrue, yPred) => tf.tidy(() => {
      const delta = tf.scalar(1);
      const error = tf.sub(yTrue, yPred);
      const absError = tf.abs(error);
      const quadratic = tf.minimum(absError, delta);
      const linear = tf.sub(absError, quadratic);
      const losses = tf.add(tf.mul(0.5, tf.square(quadratic)), tf.mul(delta, linear));
      return tf.mean(losses);
    });
    m.compile({ optimizer: tf.train.adam(0.001), loss: huberLoss });
    return m;
  }

  const qNet    = buildQNet();
  const qTarget = buildQNet();

  // target 초기 동기화
  qTarget.setWeights(qNet.getWeights());

  const buffer = new ReplayBuffer(1000);
  const BATCH   = 32;
  const GAMMA   = 0.95;
  const EPS_START = 1.0, EPS_END = 0.05;
  const STEPS_PER_EP = 20;
  const TARGET_UPDATE = 5;

  const convergenceData = [];
  let bestPerm = [...initialPerm];
  let bestReward = baselineReward;

  for (let ep = 0; ep < episodes; ep++) {
    const eps = EPS_START + (EPS_END - EPS_START) * (ep / (episodes - 1));
    let perm = [...initialPerm];
    let epReward = computeReward(perm, expectedRevenue);

    for (let step = 0; step < STEPS_PER_EP; step++) {
      const state = permToState(perm);
      let actionIdx;
      if (Math.random() < eps) {
        actionIdx = Math.floor(Math.random() * N_ACTIONS);
      } else {
        const sTensor = tf.tensor2d([Array.from(state)], [1, STATE_DIM]);
        const qVals = Array.from(qNet.predict(sTensor).dataSync());
        sTensor.dispose();
        actionIdx = qVals.indexOf(Math.max(...qVals));
      }

      const nextPerm = applyAction(perm, actionIdx);
      const nextReward = computeReward(nextPerm, expectedRevenue);
      const done = step === STEPS_PER_EP - 1;

      buffer.push({ state, actionIdx, reward: nextReward, nextState: permToState(nextPerm), done });

      if (nextReward > epReward) { perm = nextPerm; epReward = nextReward; }
      if (nextReward > bestReward) { bestReward = nextReward; bestPerm = [...nextPerm]; }

      // 학습
      if (buffer.size >= BATCH) {
        const batch = buffer.sample(BATCH);
        const sArr  = batch.map(b => Array.from(b.state));
        const nsArr = batch.map(b => Array.from(b.nextState));

        const sTensor  = tf.tensor2d(sArr,  [BATCH, STATE_DIM]);
        const nsTensor = tf.tensor2d(nsArr, [BATCH, STATE_DIM]);

        const currentQTensor = qNet.predict(sTensor);
        const nextQTensor = qTarget.predict(nsTensor);
        const currentQs = Array.from(currentQTensor.arraySync());
        const nextQs = Array.from(nextQTensor.arraySync());
        currentQTensor.dispose();
        nextQTensor.dispose();

        const targets = currentQs.map((q, i) => {
          const { actionIdx: a, reward: r, done: d } = batch[i];
          const updated = [...q];
          updated[a] = d ? r : r + GAMMA * Math.max(...nextQs[i]);
          return updated;
        });

        const tTensor = tf.tensor2d(targets, [BATCH, N_ACTIONS]);
        await qNet.fit(sTensor, tTensor, { epochs: 1, verbose: 0 });
        [sTensor, nsTensor, tTensor].forEach(t => t.dispose());
      }
    }

    if ((ep + 1) % TARGET_UPDATE === 0) qTarget.setWeights(qNet.getWeights());

    if ((ep + 1) % 10 === 0) {
      convergenceData.push({ episode: ep + 1, reward: +epReward.toFixed(0), epsilon: +eps.toFixed(3) });
    }
    if (onProgress) onProgress((ep + 1) / episodes);
    // 에피소드마다 메인 스레드 양보 → React 상태 업데이트 반영
    if ((ep + 1) % 5 === 0) await tf.nextFrame();
  }

  qNet.dispose(); qTarget.dispose();

  const improvementPct = baselineReward > 0
    ? +((bestReward - baselineReward) / baselineReward * 100).toFixed(2)
    : 0;

  // swap 내역 (현재 배치 vs DQN 추천)
  const topActions = [];
  for (let slot = 0; slot < N; slot++) {
    if (bestPerm[slot] !== initialPerm[slot]) {
      topActions.push({
        slot,
        from: ZONE_LABELS[initialPerm[slot]],
        to:   ZONE_LABELS[bestPerm[slot]],
        fromId: initialPerm[slot],
        toId:   bestPerm[slot],
      });
    }
  }

  const recommendedPerm = bestPerm.map((zId, slot) => ({
    slot,
    zoneId:          zId,
    label:           ZONE_LABELS[zId],
    originalZoneId:  initialPerm[slot],
    originalLabel:   ZONE_LABELS[initialPerm[slot]],
    changed:         zId !== initialPerm[slot],
    accessWeight:    +ACCESS_WEIGHTS[slot].toFixed(4),
    expectedRevenue: +expectedRevenue[zId].toFixed(0),
  }));

  return { recommendedPerm, baselineReward: +baselineReward.toFixed(0), optimizedReward: +bestReward.toFixed(0), improvementPct, convergenceData, topActions, accessWeights: ACCESS_WEIGHTS };
}
