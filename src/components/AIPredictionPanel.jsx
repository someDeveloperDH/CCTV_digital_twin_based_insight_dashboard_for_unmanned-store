import React, { useEffect, useState, useRef } from 'react';
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer, Cell,
} from 'recharts';
import { predictKPI } from '../services/predictionApi.js';

const TT = {
  contentStyle: { background: '#fff', border: '1px solid #e5e7eb', color: '#1f2937', fontSize: '11px', borderRadius: '6px' },
  itemStyle:    { color: '#374151' },
  labelStyle:   { color: '#6b7280', fontWeight: 'bold', marginBottom: '2px' },
};
const TICK = { fill: '#6b7280', fontSize: 10 };

const OURS = '#8b7bf0';

function InefficientCard({ rank, label, value }) {
  const styles = [
    'bg-red-50 border-red-200 text-red-700',
    'bg-orange-50 border-orange-200 text-orange-700',
    'bg-amber-50 border-amber-200 text-amber-700',
  ];
  return (
    <div className={`border rounded-xl p-3 text-center flex-1 ${styles[rank] ?? 'bg-gray-50 border-border text-text-muted'}`}>
      <div className="text-xs opacity-70">#{rank + 1} 비효율</div>
      <div className="text-sm font-bold mt-1">{label}</div>
      <div className="text-base font-bold mt-1">{(value * 100).toFixed(1)}%</div>
      <div className="text-xs opacity-70 mt-0.5">비효율 지수</div>
    </div>
  );
}

export default function AIPredictionPanel({ data, kpis }) {
  const [state, setState] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError]   = useState('');
  const abortRef = useRef(null);

  const runPrediction = () => {
    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setState('loading');
    setResult(null);
    setError('');

    predictKPI(data, { signal: ctrl.signal })
      .then(r  => { if (!ctrl.signal.aborted) { setResult(r); setState('done'); } })
      .catch(e => {
        if (e.name === 'AbortError') return;
        setError(e.message ?? '알 수 없는 오류');
        setState('error');
      });
  };

  useEffect(() => {
    setState('idle');
    setResult(null);
    if (abortRef.current) abortRef.current.abort();
  }, [data]);

  const actualZoneMap = {};
  const actualHourMap = {};
  if (kpis) {
    kpis.zoneKPIs.forEach(z => { actualZoneMap[z.id] = z; });
    kpis.hourlyAgg.forEach(h => { actualHourMap[h.hour] = h; });
  }

  const zoneConvData = result
    ? result.zoneKPI.map(z => {
        const actual = actualZoneMap[z.zoneId];
        return {
          name:   z.label,
          AI예측: +(z.convRate * 100).toFixed(1),
          실측평균: actual ? +(actual.convRate * 100).toFixed(1) : undefined,
        };
      }).sort((a, b) => b.AI예측 - a.AI예측)
    : [];

  const hourConvData = result
    ? result.hourKPI.map(h => {
        const actual = actualHourMap[h.hour];
        return {
          name:   h.label,
          AI예측: +(h.convRate * 100).toFixed(1),
          실측평균: actual ? +(actual.convRate * 100).toFixed(1) : undefined,
        };
      })
    : [];

  const ineffTop3 = result
    ? [...result.zoneKPI].sort((a, b) => b.inefficiency - a.inefficiency).slice(0, 3)
    : [];

  const dwellTable = result
    ? [...result.zoneKPI].sort((a, b) => b.avgDwell - a.avgDwell)
    : [];

  return (
    <div className="card p-4">
      {/* 헤더 */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="section-title mb-0">딥러닝 KPI 예측</p>
          {result && (
            <span className="text-xs text-text-muted">
              Day {result.predictionDay} 예측 (최근 {result.windowDays}일 기반)
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="badge bg-accent-light text-accent text-xs">Transformer 4.0~5.4%</span>
          <span className="badge bg-violet-100 text-violet-700 text-xs">Crossformer 3.9~8.6%</span>
          <button
            onClick={runPrediction}
            disabled={state === 'loading'}
            className={`btn-primary text-xs py-1 px-3 ${state === 'loading' ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {state === 'loading' ? '예측 중…' : state === 'done' ? '다시 예측' : '예측 실행'}
          </button>
        </div>
      </div>

      {state === 'idle' && (
        <div className="py-8 text-center text-text-muted text-sm">
          "예측 실행" 버튼을 눌러 다음날 KPI를 AI로 예측하세요.
          <br />
          <span className="text-xs text-text-muted/60">
            최소 7일 시뮬레이션 데이터 필요 · KPI 서버가 실행 중이어야 합니다
          </span>
        </div>
      )}

      {state === 'loading' && (
        <div className="py-8 text-center text-accent text-sm">
          모델 inference 중…
        </div>
      )}

      {state === 'error' && (
        <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-danger text-sm">
          {error}
          {(error.includes('연결') || error.includes('fetch')) && (
            <div className="mt-2 text-xs text-danger/70">
              KPI 서버가 실행 중인지 확인하세요:<br />
              <code className="text-warn">uvicorn simul.server.main:app --port 8000</code>
            </div>
          )}
        </div>
      )}

      {state === 'done' && result?.warnings?.length > 0 && (
        <div className="mb-3 p-2 bg-amber-50 border border-amber-200 rounded text-xs text-warn">
          {result.warnings.join(' / ')}
        </div>
      )}

      {state === 'done' && result?.unknownZones?.length > 0 && (
        <div className="mb-3 text-xs text-text-muted">
          다음 구역은 학습 데이터에 없어 예측에서 제외됨: {result.unknownZones.join(', ')}
        </div>
      )}

      {state === 'done' && result && (
        <div className="grid grid-cols-2 gap-4">

          {/* KPI #1 — 구역별 전환율 */}
          <div className="bg-gray-50 rounded-lg p-3 border border-border">
            <p className="section-title">KPI #1 — 구역별 구매전환율 (Transformer)</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={zoneConvData} layout="vertical" margin={{ top: 0, right: 30, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis type="number" domain={[0, 'auto']} tick={TICK} tickFormatter={v => `${v}%`} />
                <YAxis type="category" dataKey="name" tick={TICK} width={65} />
                <Tooltip {...TT} formatter={v => [`${v}%`, '']} />
                <Legend wrapperStyle={{ fontSize: '10px', color: '#6b7280' }} />
                <Bar dataKey="AI예측"   fill={OURS} radius={[0,6,6,0]} />
                <Bar dataKey="실측평균" fill="#d1d5db" radius={[0,6,6,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* KPI #6 — 시간대별 전환율 */}
          <div className="bg-gray-50 rounded-lg p-3 border border-border">
            <p className="section-title">KPI #6 — 시간대별 구매전환율 (Transformer)</p>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={hourConvData} margin={{ top: 4, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="name" tick={TICK} />
                <YAxis tick={TICK} tickFormatter={v => `${v}%`} />
                <Tooltip {...TT} formatter={v => [`${v}%`, '']} />
                <Legend wrapperStyle={{ fontSize: '10px', color: '#6b7280' }} />
                <Line type="monotone" dataKey="AI예측"   stroke="#6d5ce7" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="실측평균" stroke="#d1d5db" strokeWidth={1} dot={false} strokeDasharray="4 2" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* KPI #3 + #7 */}
          <div className="bg-gray-50 rounded-lg p-3 border border-border">
            <p className="section-title">KPI #3/#7 — 구역별 체류시간 · 전환효율 (Crossformer)</p>
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    {['구역', '체류(초)', '전환율', '효율지수'].map(h => (
                      <th key={h}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dwellTable.map(z => {
                    const actual = actualZoneMap[z.zoneId];
                    return (
                      <tr key={z.zoneId}>
                        <td>{z.label}</td>
                        <td className="text-ok font-medium">
                          {z.avgDwell.toFixed(1)}
                          {actual && <span className="text-text-muted ml-1">/ {actual.avgDwell.toFixed(1)}</span>}
                        </td>
                        <td className="col-ours">{(z.convRate * 100).toFixed(1)}%</td>
                        <td className="text-warn font-medium">{z.efficiency.toFixed(4)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-text-muted mt-2">체류시간: AI예측 / 실측평균 · 효율지수 = 전환율 / 체류시간</p>
          </div>

          {/* KPI #8 — 비효율 구역 TOP3 */}
          <div className="bg-gray-50 rounded-lg p-3 border border-border">
            <p className="section-title">KPI #8 — 비효율 구역 TOP3 (Crossformer)</p>
            <div className="flex gap-2 mb-3">
              {ineffTop3.map((z, i) => (
                <InefficientCard key={z.zoneId} rank={i} label={z.label} value={z.inefficiency} />
              ))}
            </div>
            <p className="text-xs text-text-muted">
              비효율 지수 = (체류시간/max체류) × (1 − 전환율/max전환율)<br />
              높을수록 체류 대비 구매 전환이 낮은 개선 우선 구역
            </p>
          </div>

        </div>
      )}
    </div>
  );
}
