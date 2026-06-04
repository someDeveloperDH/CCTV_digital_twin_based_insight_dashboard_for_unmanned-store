import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import metricsIcecream    from './data/metrics_icecream.json'
import metricsConvenience from './data/metrics_convenience.json'
import metricsSupermarket from './data/metrics_supermarket.json'
import ArchitectureDiagram from './components/ArchitectureDiagram.jsx'
import MetricCards         from './components/MetricCards.jsx'
import ModelComparison     from './components/ModelComparison.jsx'
import PredVsActual        from './components/PredVsActual.jsx'

const STORE_METRICS = {
  icecream:    metricsIcecream,
  convenience: metricsConvenience,
  supermarket: metricsSupermarket,
}

const STORE_TABS = [
  { key: 'icecream',    label: '🍦 아이스크림 무인매장' },
  { key: 'convenience', label: '🏪 편의점' },
  { key: 'supermarket', label: '🛒 슈퍼마켓' },
]

function SectionHeader({ number, title, sub }) {
  return (
    <div className="flex items-start gap-4 mb-5">
      <div className="w-9 h-9 rounded-full bg-accent flex items-center justify-center text-white text-sm font-bold shrink-0">
        {number}
      </div>
      <div>
        <h2 className="text-lg font-bold text-text">{title}</h2>
        {sub && <p className="text-sm text-text-muted mt-0.5">{sub}</p>}
      </div>
    </div>
  )
}

export default function ModelShowcase() {
  const [storeType, setStoreType] = useState(() => {
    const saved = localStorage.getItem('lastStoreType')
    return STORE_METRICS[saved] ? saved : 'icecream'
  })

  const metrics = STORE_METRICS[storeType]
  const { transformer, crossformer, comparison } = metrics

  const handleTabChange = (key) => {
    setStoreType(key)
    localStorage.setItem('lastStoreType', key)
  }

  return (
    <div className="min-h-screen bg-bg">
      {/* 최상단 헤더 */}
      <header className="bg-white border-b border-border sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link to="/" className="btn-ghost flex items-center gap-1.5 text-sm">
              <ArrowLeft className="w-4 h-4" />
              대시보드로
            </Link>
            <div className="w-px h-4 bg-border" />
            <span className="text-sm font-medium text-text">AI 모델 성능 검증</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="badge bg-green-100 text-ok">실측 결과 · {metrics.experimentDate}</span>
            <span className="badge bg-gray-100 text-text-muted">{metrics.dataInfo.source}</span>
          </div>
        </div>

        {/* 매장 타입 탭 */}
        <div className="max-w-6xl mx-auto px-6 pb-0 flex gap-1 border-t border-border">
          {STORE_TABS.map(tab => (
            <button
              key={tab.key}
              onClick={() => handleTabChange(tab.key)}
              className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                storeType === tab.key
                  ? 'border-accent text-accent'
                  : 'border-transparent text-text-muted hover:text-text'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-8 space-y-10">

        {/* 히어로 */}
        <div className="card p-8 bg-gradient-to-br from-accent-light to-white border-accent/20">
          <div className="max-w-3xl">
            <div className="flex items-center gap-2 mb-3">
              <span className="badge bg-accent text-white">캡스톤 디자인</span>
              <span className="text-text-muted text-sm">2026</span>
              <span className="badge bg-gray-100 text-text-muted">{metrics.storeLabel}</span>
            </div>
            <h1 className="text-2xl font-bold text-text leading-snug">
              매장 CCTV 영상에서의 실시간 고객 속성 분류<br />
              및 체류 히트맵 기반 상품 배치 최적화
            </h1>
            <p className="text-text-muted mt-3">
              Transformer · Crossformer · LSTM 3개 모델의 KPI 예측 성능 실측 비교 결과 — {metrics.storeLabel}
            </p>
            <div className="mt-5 p-4 bg-white rounded-xl border border-border">
              <p className="text-base font-semibold text-text">
                {comparison.headline}
              </p>
              <p className="text-xs text-text-muted mt-1">{comparison.summary}</p>
            </div>
          </div>
        </div>

        {/* 실험 데이터 요약 */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: '학습 데이터', value: metrics.dataInfo.source },
            { label: '총 구매 이벤트', value: `${(metrics.dataInfo.totalRows / 1000).toFixed(0)}K 건` },
            { label: '학습 / 검증 / 테스트', value: `${metrics.dataInfo.trainDays} / ${metrics.dataInfo.valDays} / ${metrics.dataInfo.testDays}일` },
            { label: '비교 모델 수', value: '3개 (Transformer · Crossformer · LSTM)' },
          ].map(d => (
            <div key={d.label} className="card p-4">
              <div className="text-xs text-text-muted">{d.label}</div>
              <div className="text-sm font-semibold text-text mt-1">{d.value}</div>
            </div>
          ))}
        </div>

        {/* Section 1: 모델 아키텍처 */}
        <section>
          <SectionHeader
            number={1}
            title="모델 아키텍처"
            sub="Transformer(SpaceTimeBlock) vs Crossformer(TwoStageAttention) 구조 비교"
          />
          <ArchitectureDiagram />
        </section>

        {/* Section 2: 핵심 지표 카드 */}
        <section>
          <SectionHeader
            number={2}
            title="핵심 성능 지표"
            sub="MAPE 10% 이하 = 실무 사용 가능 기준 | Spearman ρ 0.7 이상 = 신뢰 가능"
          />
          <MetricCards transformer={transformer} crossformer={crossformer} />
        </section>

        {/* Section 3: 3모델 비교 */}
        <section>
          <SectionHeader
            number={3}
            title="3-모델 종합 비교"
            sub="같은 데이터 · 같은 test split(105~120일)에서 실행한 결과"
          />
          <ModelComparison comparison={comparison} transformer={transformer} crossformer={crossformer} lstm={metrics.lstm} />
        </section>

        {/* Section 4: 예측 vs 실측 */}
        <section>
          <SectionHeader
            number={4}
            title="예측 vs 실측 상세"
            sub="시간대별 구매수 패턴 및 구역별 구매 비중 — Transformer 모델 (test 16일)"
          />
          <PredVsActual transformer={transformer} crossformer={crossformer} />
        </section>

        {/* 푸터 */}
        <footer className="text-center text-xs text-text-muted pb-8 border-t border-border pt-6">
          실험 환경: CPU only · torch 2.11.0+cpu · seed=42 · 2026-05-07<br />
          학습 데이터: {metrics.dataInfo.source}
        </footer>
      </div>
    </div>
  )
}
