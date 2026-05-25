import React, { useState } from 'react'
import {
  LineChart, Line, BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

const TICK = { fill: '#6b7280', fontSize: 11 }
const TT   = { contentStyle: { background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, fontSize: 12 } }
const OURS = '#8b7bf0'
const BASE = '#cbd5e1'
const SOFT_RAMP = ['#a78bfa','#9277ee','#8b7bf0','#b794f6','#c8a8ee','#dcb6e6','#f0abfc']
const softAt = (i) => SOFT_RAMP[i % SOFT_RAMP.length]

const INEFF_COLORS = [
  'bg-red-50 border-red-200 text-red-700',
  'bg-amber-50 border-amber-200 text-amber-700',
  'bg-yellow-50 border-yellow-200 text-yellow-700',
]

function HourlyChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis dataKey="hour" tick={TICK} tickFormatter={v => `${v}시`} />
        <YAxis tick={TICK} />
        <Tooltip {...TT} labelFormatter={v => `${v}시`} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line type="monotone" dataKey="actual" name="실측" stroke={BASE} strokeWidth={2} dot={{ r: 3, fill: BASE }} />
        <Line type="monotone" dataKey="pred"   name="예측" stroke="#6d5ce7" strokeWidth={2} strokeDasharray="5 3" dot={{ r: 3, fill: '#6d5ce7' }} />
      </LineChart>
    </ResponsiveContainer>
  )
}

function ZoneChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis dataKey="label" tick={{ ...TICK, fontSize: 10 }} />
        <YAxis tick={TICK} tickFormatter={v => `${v}%`} />
        <Tooltip {...TT} formatter={v => `${v}%`} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Bar dataKey="actual" name="실측 비중" fill={BASE} radius={[6,6,0,0]} />
        <Bar dataKey="pred" name="예측 비중" radius={[6,6,0,0]}>
          {data.map((_, i) => <Cell key={i} fill={softAt(i)} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function ImprovementChart({ improvement }) {
  const data = [
    { metric: '셀 MAE',   v1: improvement.cellMAE.v1,  v2: improvement.cellMAE.v2  },
    { metric: '구역 MAE', v1: improvement.zoneMAE.v1,  v2: improvement.zoneMAE.v2  },
    { metric: '시간 MAE', v1: improvement.hourMAE.v1,  v2: improvement.hourMAE.v2  },
  ]
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis dataKey="metric" tick={TICK} />
        <YAxis tick={TICK} />
        <Tooltip {...TT} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Bar dataKey="v1" name="v1 (개선 전)" fill={BASE} radius={[6,6,0,0]} />
        <Bar dataKey="v2" name="v2 (개선 후)" fill={OURS} radius={[6,6,0,0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export default function PredVsActual({ transformer, crossformer }) {
  const [tab, setTab] = useState('hourly')

  return (
    <div className="space-y-6">
      <div className="card p-4">
        <div className="flex items-center justify-between mb-4">
          <p className="section-title mb-0">Transformer — 예측 vs 실측</p>
          <div className="flex gap-1">
            {['hourly', 'zone', 'improvement'].map(t => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`tab-item text-xs py-1 px-3 ${tab === t ? 'active' : ''}`}
              >
                {t === 'hourly' ? '시간대별' : t === 'zone' ? '구역별 비중' : 'v1→v2 개선'}
              </button>
            ))}
          </div>
        </div>

        {tab === 'hourly' && (
          <>
            <HourlyChart data={transformer.hourlyPredVsActual} />
            <p className="text-xs text-text-muted mt-2 text-center">
              피크 패턴 정확히 재현 — 시간대 MAPE {transformer.metrics.hourMAPE}%
            </p>
          </>
        )}
        {tab === 'zone' && (
          <>
            <ZoneChart data={transformer.zoneShare} />
            <p className="text-xs text-text-muted mt-2 text-center">
              구역별 구매 비중 — 구역 MAPE {transformer.metrics.zoneMAPE}% / Spearman ρ {transformer.metrics.zoneSpearmanRho.toFixed(3)}
            </p>
          </>
        )}
        {tab === 'improvement' && (
          <>
            <ImprovementChart improvement={transformer.improvement} />
            <p className="text-xs text-text-muted mt-2 text-center">
              v1 → v2 개선: 정규화·log1p·분리 임베딩·SmoothL1 적용 후 구역 MAE {transformer.improvement.zoneMAE.changePct}%
            </p>
          </>
        )}
      </div>

      <div className="card p-4">
        <p className="section-title">Crossformer — 비효율 구역 Top 3 예측 (KPI #8, ρ={crossformer.metrics.kpi8InefficientRho})</p>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {crossformer.inefficientTop3.map((z, i) => (
            <div key={z.zone} className={`border rounded-xl p-4 text-center ${INEFF_COLORS[i]}`}>
              <div className="text-2xl font-bold">#{i + 1}</div>
              <div className="font-semibold mt-1">{z.label}</div>
              <div className="text-xl font-bold mt-1">{z.score.toFixed(3)}</div>
              <div className="text-xs mt-0.5 opacity-70">비효율 지수</div>
            </div>
          ))}
        </div>
        <p className="text-xs text-text-muted mt-3">
          비효율 지수 = 체류 대비 전환이 낮은 구역. 상품 재배치 또는 프로모션 대상으로 식별됨.
        </p>
      </div>
    </div>
  )
}
