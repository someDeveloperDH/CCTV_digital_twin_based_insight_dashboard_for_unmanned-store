import React from 'react'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

const TICK  = { fill: '#6b7280', fontSize: 11 }
const TT    = { contentStyle: { background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, fontSize: 12 } }

function RhoChart({ comparison }) {
  const chartData = comparison.spearmanRho
    .filter(m => m.zoneRho !== null)
    .map(m => ({ name: m.model, zone: m.zoneRho, hour: m.hourRho }))
  return (
    <>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="name" tick={TICK} />
          <YAxis domain={[0.9, 1.01]} tick={TICK} tickFormatter={v => v.toFixed(2)} />
          <Tooltip {...TT} formatter={v => v.toFixed(3)} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="zone" name="구역 순위 ρ" fill="#6d5ce7" radius={[6,6,0,0]} />
          <Bar dataKey="hour" name="시간대 ρ"    fill="#8b7bf0" radius={[6,6,0,0]} />
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-text-muted mt-1">LSTM: 셀별 구매비중 예측 모델 — 구역·시간대 순위 ρ 비교 불가</p>
    </>
  )
}

function MapeChart({ transformer, crossformer, lstm }) {
  const data = [
    { name: 'Transformer\n(구역)',  mape: transformer.metrics.zoneMAPE,           fill: '#6d5ce7' },
    { name: 'Transformer\n(시간)',  mape: transformer.metrics.hourMAPE,           fill: '#8b7bf0' },
    { name: 'Crossformer\n(효율)',  mape: crossformer.metrics.kpi7EfficiencyMAPE, fill: '#7c3aed' },
    { name: 'Crossformer\n(체류)',  mape: crossformer.metrics.kpi3DwellMAPE,      fill: '#9333ea' },
    { name: 'Crossformer\n(전환)',  mape: crossformer.metrics.kpi2ConvMAPE,       fill: '#a855f7' },
    { name: 'LSTM',                 mape: lstm.metrics.mape,                      fill: '#d1d5db' },
  ]
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} layout="vertical" margin={{ top: 8, right: 40, left: 80, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" horizontal={false} />
        <XAxis type="number" tick={TICK} tickFormatter={v => `${v}%`} />
        <YAxis type="category" dataKey="name" tick={{ ...TICK, fontSize: 10 }} width={80} />
        <Tooltip {...TT} formatter={v => `${v}%`} />
        <Bar dataKey="mape" name="MAPE" radius={[0,6,6,0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

const FLAG_STYLE = {
  best: 'bg-accent text-white',
  good: 'bg-violet-100 text-violet-700',
  bad:  'bg-gray-100 text-gray-500',
}

export default function ModelComparison({ comparison, transformer, crossformer, lstm }) {
  const cfMapeVals = [
    crossformer.metrics.kpi2ConvMAPE,
    crossformer.metrics.kpi3DwellMAPE,
    crossformer.metrics.kpi5CustomerMAPE,
    crossformer.metrics.kpi7EfficiencyMAPE,
  ].filter(v => v != null)
  const cfMapeMin = Math.min(...cfMapeVals)
  const cfMapeMax = Math.max(...cfMapeVals)

  const cfRhoVals = [
    crossformer.metrics.kpi3DwellRho,
    crossformer.metrics.kpi5CustomerRho,
    crossformer.metrics.kpi7EfficiencyRho,
    crossformer.metrics.kpi8InefficientRho,
  ].filter(v => v != null)
  const cfRhoMin = Math.min(...cfRhoVals)
  const cfRhoMax = Math.max(...cfRhoVals)

  const tfMapeMin = Math.min(transformer.metrics.zoneMAPE, transformer.metrics.hourMAPE)
  const tfMapeMax = Math.max(transformer.metrics.zoneMAPE, transformer.metrics.hourMAPE)
  const tfRhoMin  = Math.min(transformer.metrics.zoneSpearmanRho, transformer.metrics.hourSpearmanRho)
  const tfRhoMax  = Math.max(transformer.metrics.zoneSpearmanRho, transformer.metrics.hourSpearmanRho)

  const tableRows = [
    {
      model:    'LSTM',
      target:   '셀별 구매비중',
      kpi:      '시간대 비중',
      mape:     `${lstm.metrics.mape}% (비중)`,
      rho:      'N/A',
      baseline: `${lstm.metrics.baselineImprovement}% (열위)`,
      epoch:    lstm.training.earlyStoppingEpoch,
      flag:     'bad',
    },
    {
      model:    'Crossformer',
      target:   '구매수 + 체류시간',
      kpi:      '5종 KPI',
      mape:     `${cfMapeMin} ~ ${cfMapeMax}%`,
      rho:      `${cfRhoMin.toFixed(3)} ~ ${cfRhoMax.toFixed(3)}`,
      baseline: 'N/A',
      epoch:    crossformer.training.earlyStoppingEpoch,
      flag:     'good',
    },
    {
      model:    'Transformer',
      target:   '셀별 구매수',
      kpi:      '구역·시간대',
      mape:     `${tfMapeMin} ~ ${tfMapeMax}%`,
      rho:      `${tfRhoMin.toFixed(3)} ~ ${tfRhoMax.toFixed(3)}`,
      baseline: `+${transformer.metrics.baselineImprovement}% (우위)`,
      epoch:    transformer.training.earlyStoppingEpoch,
      flag:     'best',
    },
  ]

  return (
    <div className="space-y-6">
      <div>
        <p className="section-title">3-모델 성능 비교표</p>
        <div className="card overflow-auto">
          <table className="w-full text-sm">
            <thead>
              <tr style={{ background: 'linear-gradient(135deg, #8b7bf0, #a78bfa)' }}>
                {['모델', '예측 타겟', 'KPI 커버리지', 'MAPE', 'Spearman ρ', 'Baseline 대비', '수렴 epoch'].map(h => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-white whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.map((r, idx) => (
                <tr key={r.model} className="border-b border-border last:border-0 hover:bg-gray-50" style={idx % 2 === 1 ? { background: '#faf9fe' } : {}}>
                  <td className="px-4 py-3 font-medium whitespace-nowrap">
                    <span className={`badge mr-2 ${FLAG_STYLE[r.flag]}`}>
                      {r.flag === 'best' ? '1위' : r.flag === 'good' ? '2위' : '3위'}
                    </span>
                    {r.model}
                  </td>
                  <td className="px-4 py-3 text-text-muted">{r.target}</td>
                  <td className="px-4 py-3 text-text-muted">{r.kpi}</td>
                  <td className={`px-4 py-3 font-semibold ${r.flag === 'best' ? 'text-accent' : r.flag === 'bad' ? 'text-gray-400' : 'text-violet-600'}`}>{r.mape}</td>
                  <td className={`px-4 py-3 font-semibold ${r.flag !== 'bad' ? 'text-ok' : 'text-gray-400'}`}>{r.rho}</td>
                  <td className={`px-4 py-3 font-semibold ${r.baseline.includes('+') ? 'text-ok' : r.baseline.includes('열위') ? 'text-danger' : 'text-text-muted'}`}>{r.baseline}</td>
                  <td className="px-4 py-3 text-text-muted">{r.epoch}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-text-muted mt-2 text-right">
          {comparison.summary}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card p-4">
          <p className="section-title">Spearman ρ 비교 (순위 예측 정확도)</p>
          <RhoChart comparison={comparison} />
          <p className="text-xs text-text-muted mt-2 text-center">값이 높을수록 순위 예측 정확</p>
        </div>
        <div className="card p-4">
          <p className="section-title">MAPE 비교 (낮을수록 오차 적음)</p>
          <MapeChart transformer={transformer} crossformer={crossformer} lstm={lstm} />
          <p className="text-xs text-text-muted mt-2 text-center">막대가 짧을수록 예측 오차가 적음</p>
        </div>
      </div>
    </div>
  )
}
