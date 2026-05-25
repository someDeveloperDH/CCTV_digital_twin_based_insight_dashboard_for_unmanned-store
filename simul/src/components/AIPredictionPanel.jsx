import React, { useEffect, useState, useRef } from 'react';
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer, Cell,
} from 'recharts';
import { predictKPI } from '../services/predictionApi.js';

const DARK = {
  bg:      '#0e0e22',
  border:  '1px solid #1e1e44',
  label:   { fill: '#7070aa', fontSize: 10 },
  tooltip: {
    contentStyle: { background: '#161628', border: '1px solid #2a2a5a', color: '#e0e0ff', fontSize: '11px', borderRadius: '4px' },
    itemStyle:    { color: '#e0e0ff' },
    labelStyle:   { color: '#a0a0d0', fontWeight: 'bold', marginBottom: '2px' },
  },
};

function Badge({ label, value, color = '#40cc80' }) {
  return (
    <span style={{
      display: 'inline-block', background: '#0a1a2a', border: `1px solid ${color}33`,
      borderRadius: '10px', padding: '1px 8px', fontSize: '10px', color, marginLeft: '6px',
    }}>
      {label} {value}
    </span>
  );
}

function SectionTitle({ children }) {
  return (
    <div style={{ fontSize: '11px', fontWeight: 'bold', color: '#a0a0d0', marginBottom: '8px' }}>
      {children}
    </div>
  );
}

function InefficientCard({ rank, label, value }) {
  const colors = ['#ff6060', '#ff9060', '#ffc060'];
  const color  = colors[rank] ?? '#8090cc';
  return (
    <div style={{ background: '#0a1a2a', border: `1px solid ${color}55`, borderRadius: '6px', padding: '8px 12px', flex: 1 }}>
      <div style={{ fontSize: '10px', color: '#5070aa' }}>#{rank + 1} 비효율</div>
      <div style={{ fontSize: '13px', fontWeight: 'bold', color, marginTop: '2px' }}>{label}</div>
      <div style={{ fontSize: '10px', color: '#7070aa', marginTop: '2px' }}>
        비효율 지수 <span style={{ color }}>{(value * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}

export default function AIPredictionPanel({ data, kpis }) {
  const [state, setState] = useState('idle');   // idle | loading | done | error
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

  // data가 바뀌면 이전 결과 초기화
  useEffect(() => {
    setState('idle');
    setResult(null);
    if (abortRef.current) abortRef.current.abort();
  }, [data]);

  // ── 실측 KPI (simul 전체 기간 평균) ──
  const actualZoneMap = {};
  const actualHourMap = {};
  if (kpis) {
    kpis.zoneKPIs.forEach(z => { actualZoneMap[z.id] = z; });
    kpis.hourlyAgg.forEach(h => { actualHourMap[h.hour] = h; });
  }

  // ── 구역별 전환율 차트 데이터 ──
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

  // ── 시간대별 전환율 차트 데이터 ──
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

  // ── 비효율 TOP3 ──
  const ineffTop3 = result
    ? [...result.zoneKPI].sort((a, b) => b.inefficiency - a.inefficiency).slice(0, 3)
    : [];

  // ── 체류+효율 테이블 ──
  const dwellTable = result
    ? [...result.zoneKPI].sort((a, b) => b.avgDwell - a.avgDwell)
    : [];

  return (
    <div style={{ background: DARK.bg, borderRadius: '8px', padding: '14px', border: DARK.border }}>

      {/* ── 헤더 ── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
        <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#c0c0e0' }}>
          🔮 딥러닝 KPI 예측
          {result && (
            <span style={{ fontSize: '10px', color: '#6070aa', fontWeight: 'normal', marginLeft: '8px' }}>
              Day {result.predictionDay} 예측 (최근 {result.windowDays}일 기반)
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Badge label="Transformer MAPE" value="4.0~5.4%" color="#7090ff" />
          <Badge label="Crossformer MAPE" value="3.9~8.6%" color="#40cc80" />
          <button
            onClick={runPrediction}
            disabled={state === 'loading'}
            style={{
              background: state === 'loading' ? '#1a1a3a' : '#1a2a5a',
              color: state === 'loading' ? '#5060aa' : '#7090ff',
              border: '1px solid #2a3a7a', borderRadius: '4px',
              padding: '4px 12px', fontSize: '11px', cursor: state === 'loading' ? 'not-allowed' : 'pointer',
            }}
          >
            {state === 'loading' ? '예측 중…' : state === 'done' ? '다시 예측' : '예측 실행'}
          </button>
        </div>
      </div>

      {/* ── idle 안내 ── */}
      {state === 'idle' && (
        <div style={{ padding: '20px', textAlign: 'center', color: '#5060aa', fontSize: '12px' }}>
          "예측 실행" 버튼을 눌러 다음날 KPI를 AI로 예측하세요.
          <br />
          <span style={{ fontSize: '10px', color: '#3a4a7a' }}>
            (최소 7일 시뮬레이션 데이터 필요 / KPI 서버가 실행 중이어야 합니다)
          </span>
        </div>
      )}

      {/* ── 로딩 ── */}
      {state === 'loading' && (
        <div style={{ padding: '20px', textAlign: 'center', color: '#7090ff', fontSize: '12px' }}>
          모델 inference 중…
        </div>
      )}

      {/* ── 오류 ── */}
      {state === 'error' && (
        <div style={{ padding: '12px', borderRadius: '6px', background: '#1a0a0a', border: '1px solid #cc404055', color: '#ff8080', fontSize: '11px' }}>
          ⚠️ {error}
          {error.includes('연결') || error.includes('fetch') ? (
            <div style={{ marginTop: '6px', color: '#cc6060', fontSize: '10px' }}>
              KPI 서버가 실행 중인지 확인하세요:<br />
              <code style={{ color: '#ff9060' }}>
                uvicorn simul.server.main:app --port 8000
              </code>
            </div>
          ) : null}
        </div>
      )}

      {/* ── 경고 ── */}
      {state === 'done' && result?.warnings?.length > 0 && (
        <div style={{ marginBottom: '10px', padding: '8px 12px', background: '#1a1a0a', border: '1px solid #aa880055', borderRadius: '4px', fontSize: '10px', color: '#cc9940' }}>
          ⚠️ {result.warnings.join(' / ')}
        </div>
      )}

      {/* ── 알 수 없는 구역 안내 ── */}
      {state === 'done' && result?.unknownZones?.length > 0 && (
        <div style={{ marginBottom: '10px', fontSize: '10px', color: '#5060aa' }}>
          다음 구역은 학습 데이터에 없어 예측에서 제외됨: {result.unknownZones.join(', ')}
        </div>
      )}

      {/* ── 결과 패널 ── */}
      {state === 'done' && result && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>

          {/* KPI #1 — 구역별 전환율 */}
          <div style={{ background: '#0a0a1e', borderRadius: '6px', padding: '10px', border: '1px solid #14143a' }}>
            <SectionTitle>📊 KPI #1 — 구역별 구매전환율 (Transformer)</SectionTitle>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={zoneConvData} layout="vertical" margin={{ top: 0, right: 30, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a1a3a" />
                <XAxis type="number" domain={[0, 'auto']} tick={DARK.label} tickFormatter={v => `${v}%`} />
                <YAxis type="category" dataKey="name" tick={DARK.label} width={65} />
                <Tooltip {...DARK.tooltip} formatter={v => [`${v}%`, '']} />
                <Legend wrapperStyle={{ fontSize: '10px', color: '#7070aa' }} />
                <Bar dataKey="AI예측"   fill="#7090ff" radius={[0,3,3,0]} />
                <Bar dataKey="실측평균" fill="#404060" radius={[0,3,3,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* KPI #6 — 시간대별 전환율 */}
          <div style={{ background: '#0a0a1e', borderRadius: '6px', padding: '10px', border: '1px solid #14143a' }}>
            <SectionTitle>⏰ KPI #6 — 시간대별 구매전환율 (Transformer)</SectionTitle>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={hourConvData} margin={{ top: 4, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a1a3a" />
                <XAxis dataKey="name" tick={DARK.label} />
                <YAxis tick={DARK.label} tickFormatter={v => `${v}%`} />
                <Tooltip {...DARK.tooltip} formatter={v => [`${v}%`, '']} />
                <Legend wrapperStyle={{ fontSize: '10px', color: '#7070aa' }} />
                <Line type="monotone" dataKey="AI예측"   stroke="#7090ff" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="실측평균" stroke="#404060" strokeWidth={1} dot={false} strokeDasharray="4 2" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* KPI #3 + #7 — 체류시간 + 효율 */}
          <div style={{ background: '#0a0a1e', borderRadius: '6px', padding: '10px', border: '1px solid #14143a' }}>
            <SectionTitle>🕐 KPI #3/#7 — 구역별 체류시간 · 전환효율 (Crossformer)</SectionTitle>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '10px', color: '#a0a0d0' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #1e1e44' }}>
                    {['구역', '체류(초)', '전환율', '효율지수'].map(h => (
                      <th key={h} style={{ padding: '4px 8px', textAlign: 'left', color: '#6070aa' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dwellTable.map(z => {
                    const actual = actualZoneMap[z.zoneId];
                    return (
                      <tr key={z.zoneId} style={{ borderBottom: '1px solid #14143a' }}>
                        <td style={{ padding: '4px 8px' }}>{z.label}</td>
                        <td style={{ padding: '4px 8px', color: '#40cc80' }}>
                          {z.avgDwell.toFixed(1)}
                          {actual && <span style={{ color: '#404060', marginLeft: '4px' }}>/ {actual.avgDwell.toFixed(1)}</span>}
                        </td>
                        <td style={{ padding: '4px 8px', color: '#7090ff' }}>
                          {(z.convRate * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '4px 8px', color: '#ffc060' }}>
                          {z.efficiency.toFixed(4)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div style={{ fontSize: '9px', color: '#3a4a7a', marginTop: '6px' }}>
              체류시간: AI예측 / 실측평균 &nbsp;|&nbsp; 효율지수 = 전환율 / 체류시간
            </div>
          </div>

          {/* KPI #8 — 비효율 구역 TOP3 */}
          <div style={{ background: '#0a0a1e', borderRadius: '6px', padding: '10px', border: '1px solid #14143a' }}>
            <SectionTitle>🔴 KPI #8 — 비효율 구역 TOP3 (Crossformer)</SectionTitle>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
              {ineffTop3.map((z, i) => (
                <InefficientCard key={z.zoneId} rank={i} label={z.label} value={z.inefficiency} />
              ))}
            </div>
            <div style={{ fontSize: '9px', color: '#3a4a7a' }}>
              비효율 지수 = (체류시간 / max체류) × (1 − 전환율 / max전환율)
              <br />높을수록 체류 대비 구매 전환이 낮은 개선 우선 구역
            </div>
          </div>

        </div>
      )}
    </div>
  );
}
