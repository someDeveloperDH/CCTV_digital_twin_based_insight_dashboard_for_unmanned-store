import React from 'react'
import { TrendingUp, Target, Zap, CheckCircle } from 'lucide-react'

function MetricCard({ icon: Icon, label, value, sub, color = 'text-accent', bg = 'bg-accent-light' }) {
  return (
    <div className="card p-4 flex items-start gap-3">
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${bg}`}>
        <Icon className={`w-5 h-5 ${color}`} />
      </div>
      <div>
        <div className="text-xs text-text-muted">{label}</div>
        <div className={`text-2xl font-bold mt-0.5 ${color}`}>{value}</div>
        {sub && <div className="text-xs text-text-muted mt-0.5">{sub}</div>}
      </div>
    </div>
  )
}

export default function MetricCards({ transformer, crossformer }) {
  return (
    <div className="space-y-4">
      <div>
        <p className="section-title">Transformer — 핵심 지표</p>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <MetricCard
            icon={Target}
            label="구역별 MAPE"
            value={`${transformer.metrics.zoneMAPE}%`}
            sub="예측 오차율 (낮을수록 우수)"
            color="text-accent"
            bg="bg-accent-light"
          />
          <MetricCard
            icon={Target}
            label="시간대별 MAPE"
            value={`${transformer.metrics.hourMAPE}%`}
            sub="예측 오차율 (낮을수록 우수)"
            color="text-accent"
            bg="bg-accent-light"
          />
          <MetricCard
            icon={CheckCircle}
            label="구역 순위 ρ"
            value={transformer.metrics.zoneSpearmanRho.toFixed(3)}
            sub={`${transformer.zoneShare.length}개 구역 순위 일치`}
            color="text-ok"
            bg="bg-green-50"
          />
          <MetricCard
            icon={TrendingUp}
            label="Baseline 대비"
            value={`+${transformer.metrics.baselineImprovement}%`}
            sub={`Persistence MAE ${transformer.metrics.baselineCellMAE} → ${transformer.metrics.cellMAE}`}
            color="text-ok"
            bg="bg-green-50"
          />
        </div>
      </div>

      <div>
        <p className="section-title">Crossformer — KPI별 지표</p>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <MetricCard
            icon={Zap}
            label="전환효율 MAPE"
            value={`${crossformer.metrics.kpi7EfficiencyMAPE}%`}
            sub={`Spearman ρ ${crossformer.metrics.kpi7EfficiencyRho}`}
            color="text-violet-600"
            bg="bg-violet-50"
          />
          <MetricCard
            icon={Zap}
            label="체류시간 MAPE"
            value={`${crossformer.metrics.kpi3DwellMAPE}%`}
            sub={`순위 일치도 ρ ${crossformer.metrics.kpi3DwellRho}`}
            color="text-violet-600"
            bg="bg-violet-50"
          />
          <MetricCard
            icon={CheckCircle}
            label="고객군 순위 ρ"
            value={crossformer.metrics.kpi5CustomerRho.toFixed(3)}
            sub="4개 고객군 순위 일치"
            color="text-ok"
            bg="bg-green-50"
          />
          <MetricCard
            icon={CheckCircle}
            label="비효율 구역 ρ"
            value={crossformer.metrics.kpi8InefficientRho.toFixed(3)}
            sub="체류 대비 전환이 낮은 구역 순위 예측"
            color="text-ok"
            bg="bg-green-50"
          />
        </div>
      </div>
    </div>
  )
}
