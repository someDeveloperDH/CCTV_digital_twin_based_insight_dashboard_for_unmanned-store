/**
 * KPI 예측 서버(localhost:8000)와의 fetch 래퍼.
 * Vite dev server proxy(/api → :8000)를 통해 호출된다.
 */

const TIMEOUT_MS = 30_000;

export async function predictKPI(data, { signal } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  const combinedSignal = signal
    ? anyAborted([signal, controller.signal])
    : controller.signal;

  try {
    const res = await fetch('/api/predict-kpi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        operatingHours: data.operatingHours ?? 12,
        purchases:      data.purchases ?? [],
        zoneStats:      data.zoneStats  ?? {},
        hourlyStats:    aggregateHourlyStats(data.hourlyStats ?? []),
      }),
      signal: combinedSignal,
    });

    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail ?? detail; } catch { /* ignore */ }
      throw new Error(detail);
    }

    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

/** hourlyStats: [{ hour, visitors, buyers }] 배열을 0-base hour로 집계 */
function aggregateHourlyStats(hourlyStats) {
  const map = {};
  for (const hs of hourlyStats) {
    const h = hs.hour ?? 0;
    if (!map[h]) map[h] = { hour: h, visitors: 0, buyers: 0 };
    map[h].visitors += hs.visitors ?? 0;
    map[h].buyers   += hs.buyers   ?? 0;
  }
  return Object.values(map);
}

/** 여러 AbortSignal 중 하나라도 abort되면 abort되는 signal 생성 */
function anyAborted(signals) {
  const ctrl = new AbortController();
  for (const s of signals) {
    if (s.aborted) { ctrl.abort(); break; }
    s.addEventListener('abort', () => ctrl.abort(), { once: true });
  }
  return ctrl.signal;
}
