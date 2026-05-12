// LSTM 시계열 매출 예측 (TensorFlow.js)
// 입력: dailyStats 배열 [{ revenue, visitors, dayOfWeek, convRate }]
// 출력: 14일 예측 + baseline 비교 지표

const WINDOW = 7;
const FEATURES = 4; // revenue_norm, visitors_norm, dow/6, convRate

function minMaxNorm(arr) {
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const range = max - min || 1;
  return { norm: arr.map(v => (v - min) / range), min, max, range };
}

function denorm(v, min, range) {
  return v * range + min;
}

// 선형 트렌드 baseline: 마지막 WINDOW일로 선형회귀 → horizon일 예측
function linearTrendForecast(values, horizon) {
  const n = values.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += i; sy += values[i]; sxx += i * i; sxy += i * values[i];
  }
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx || 1);
  const intercept = (sy - slope * sx) / n;
  return Array.from({ length: horizon }, (_, i) => intercept + slope * (n + i));
}

function calcMetrics(actual, predicted) {
  const n = actual.length;
  if (n === 0) return { mape: 0, rmse: 0, r2: 0 };
  let sumMape = 0, sumSq = 0, sumTot = 0;
  const mean = actual.reduce((s, v) => s + v, 0) / n;
  for (let i = 0; i < n; i++) {
    sumMape += Math.abs((actual[i] - predicted[i]) / (actual[i] || 1));
    const e = actual[i] - predicted[i];
    sumSq += e * e;
    const t = actual[i] - mean;
    sumTot += t * t;
  }
  return {
    mape: +(sumMape / n * 100).toFixed(2),
    rmse: +Math.sqrt(sumSq / n).toFixed(2),
    r2: +(1 - sumSq / (sumTot || 1)).toFixed(4),
  };
}

export async function trainLSTMForecaster(dailyStats, { horizon = 14, onProgress } = {}) {
  const tf = (await import('@tensorflow/tfjs')).default ?? (await import('@tensorflow/tfjs'));

  const revenues  = dailyStats.map(d => d.revenue);
  const visitors  = dailyStats.map(d => d.visitors);
  const dows      = dailyStats.map(d => (d.dayOfWeek ?? 0) / 6);
  const convRates = dailyStats.map(d => d.convRate ?? 0);

  const revN = minMaxNorm(revenues);
  const visN = minMaxNorm(visitors);

  // 슬라이딩 윈도우 샘플 생성
  const n = dailyStats.length;
  const testSize = Math.min(7, Math.floor(n * 0.2));
  const trainEnd = n - testSize;

  if (trainEnd < WINDOW + 1) {
    return {
      model: null,
      forecast: [],
      metrics: { mape: 0, rmse: 0, r2: 0 },
      baselineMetrics: { mape: 0, rmse: 0, r2: 0 },
      history: [],
      insufficient: true,
    };
  }

  const xs = [], ys = [];
  for (let i = WINDOW; i < n; i++) {
    const seq = [];
    for (let j = i - WINDOW; j < i; j++) {
      seq.push([revN.norm[j], visN.norm[j], dows[j], convRates[j]]);
    }
    xs.push(seq);
    ys.push(revN.norm[i]);
  }

  const trainXs = xs.slice(0, trainEnd - WINDOW);
  const trainYs = ys.slice(0, trainEnd - WINDOW);
  const testXs  = xs.slice(trainEnd - WINDOW);
  const testYs  = ys.slice(trainEnd - WINDOW);

  const xTensor = tf.tensor3d(trainXs, [trainXs.length, WINDOW, FEATURES]);
  const yTensor = tf.tensor2d(trainYs, [trainYs.length, 1]);

  // 모델 정의
  const model = tf.sequential();
  model.add(tf.layers.lstm({ units: 32, returnSequences: true, inputShape: [WINDOW, FEATURES] }));
  model.add(tf.layers.lstm({ units: 16, returnSequences: false }));
  model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));
  model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });

  const historyLog = [];

  await model.fit(xTensor, yTensor, {
    epochs: 50,
    batchSize: 8,
    validationSplit: 0.15,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        historyLog.push({ epoch: epoch + 1, loss: +logs.loss.toFixed(6), valLoss: +(logs.val_loss ?? logs.loss).toFixed(6) });
        if (onProgress) onProgress((epoch + 1) / 50);
        // 메인 스레드 양보 → React 상태 업데이트 반영
        await tf.nextFrame();
      },
    },
  });

  xTensor.dispose();
  yTensor.dispose();

  // 테스트셋 예측
  const testXTensor = tf.tensor3d(testXs, [testXs.length, WINDOW, FEATURES]);
  const testPredNorm = Array.from((await model.predict(testXTensor)).dataSync());
  testXTensor.dispose();

  const testActual    = testYs.map(v => denorm(v, revN.min, revN.range));
  const testPredicted = testPredNorm.map(v => denorm(v, revN.min, revN.range));
  const metrics = calcMetrics(testActual, testPredicted);

  // Baseline: 테스트 직전 WINDOW일 선형 트렌드로 testSize일 예측
  const baselineInput = revenues.slice(trainEnd - WINDOW, trainEnd);
  const baselinePreds = linearTrendForecast(baselineInput, testSize);
  const baselineMetrics = calcMetrics(testActual, baselinePreds);

  // Horizon 예측: 마지막 WINDOW일 입력으로 롤링 예측
  const forecast = [];
  const buf = {
    rev:  revN.norm.slice(-WINDOW),
    vis:  visN.norm.slice(-WINDOW),
    dow:  dows.slice(-WINDOW),
    conv: convRates.slice(-WINDOW),
  };

  // 예측 불확실성: test set 잔차의 std 사용
  const residuals = testActual.map((a, i) => a - testPredicted[i]);
  const resMean = residuals.reduce((s, v) => s + v, 0) / (residuals.length || 1);
  const resStd  = Math.sqrt(residuals.reduce((s, v) => s + (v - resMean) ** 2, 0) / (residuals.length || 1));

  for (let step = 0; step < horizon; step++) {
    const seq = [];
    for (let j = 0; j < WINDOW; j++) {
      seq.push([buf.rev[j], buf.vis[j], buf.dow[j], buf.conv[j]]);
    }
    const inp = tf.tensor3d([seq], [1, WINDOW, FEATURES]);
    const predNorm = (await model.predict(inp)).dataSync()[0];
    inp.dispose();

    const revenue = Math.max(0, denorm(predNorm, revN.min, revN.range));
    forecast.push({
      day: n + step,
      label: `+${step + 1}일`,
      revenue: Math.round(revenue),
      lower:   Math.round(Math.max(0, revenue - 1.64 * resStd)),
      upper:   Math.round(revenue + 1.64 * resStd),
    });

    // 롤링: 새 예측값을 버퍼 끝에 추가
    buf.rev.shift();  buf.rev.push(predNorm);
    buf.vis.shift();  buf.vis.push(visN.norm[visN.norm.length - 1]);
    buf.dow.shift();  buf.dow.push(dows[dows.length - 1]);
    buf.conv.shift(); buf.conv.push(convRates[convRates.length - 1]);
  }

  return { model, forecast, metrics, baselineMetrics, history: historyLog };
}
