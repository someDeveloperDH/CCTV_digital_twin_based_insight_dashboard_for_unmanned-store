import React from 'react'

function Block({ label, sub, color = 'bg-accent-light text-accent border-accent/30' }) {
  return (
    <div className={`rounded-lg border px-3 py-2 text-center ${color}`}>
      <div className="text-sm font-semibold">{label}</div>
      {sub && <div className="text-xs text-text-muted mt-0.5">{sub}</div>}
    </div>
  )
}

function Arrow() {
  return <div className="flex justify-center text-border text-lg leading-none">↓</div>
}

function RepeatBlock({ label, sub, n }) {
  return (
    <div className="relative rounded-xl border-2 border-accent/40 bg-accent-light/50 px-4 py-3">
      <span className="absolute -top-2.5 right-3 bg-white border border-border text-xs px-2 py-0.5 rounded-full text-accent font-semibold">
        × {n}
      </span>
      <div className="text-sm font-semibold text-accent text-center">{label}</div>
      {sub && <div className="text-xs text-text-muted text-center mt-0.5">{sub}</div>}
    </div>
  )
}

function TransformerArch() {
  return (
    <div className="flex flex-col gap-2 w-full max-w-xs mx-auto">
      <Block label="Input" sub="[B, L=7일, N=84 tokens, F=8 features]" color="bg-gray-50 border-border text-text" />
      <Arrow />
      <div className="grid grid-cols-2 gap-2">
        <Block label="Zone Emb" sub="[7 × 96]" color="bg-violet-50 border-violet-200 text-violet-700" />
        <Block label="Hour Emb" sub="[12 × 96]" color="bg-violet-50 border-violet-200 text-violet-700" />
        <Block label="DoW Emb" sub="[7 × 96]" color="bg-violet-50 border-violet-200 text-violet-700" />
        <Block label="Pos Emb" sub="[L × 96]" color="bg-violet-50 border-violet-200 text-violet-700" />
      </div>
      <Arrow />
      <RepeatBlock
        label="SpaceTimeBlock"
        sub="Spatial Attn → Temporal Attn → FFN"
        n={3}
      />
      <Arrow />
      <Block label="Head" sub="LayerNorm → Linear(96 → 1)" color="bg-emerald-50 border-emerald-200 text-emerald-700" />
      <Arrow />
      <Block label="Output" sub="[B, H=1, N=84]  (구역×시간대 구매수)" color="bg-green-50 border-green-200 text-green-700" />
    </div>
  )
}

function CrossformerArch() {
  return (
    <div className="flex flex-col gap-2 w-full max-w-xs mx-auto">
      <Block label="Input" sub="[B, L=5일, Z×T×C tokens, F=4]" color="bg-gray-50 border-border text-text" />
      <Arrow />
      <RepeatBlock
        label="TwoStageAttentionLayer"
        sub="① Time Attn  →  ② Router Cross-Dim Attn"
        n={2}
      />
      <Arrow />
      <div className="grid grid-cols-2 gap-2">
        <Block label="Purchase Head" sub="구매수 예측" color="bg-violet-50 border-violet-200 text-violet-700" />
        <Block label="Dwell Head" sub="체류시간 예측" color="bg-violet-50 border-violet-200 text-violet-700" />
      </div>
      <Arrow />
      <Block
        label="Output (5 KPIs)"
        sub="전환율 · 체류 · 효율 · 비효율 · 고객군"
        color="bg-green-50 border-green-200 text-green-700"
      />
    </div>
  )
}

export default function ArchitectureDiagram() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="card p-5">
        <div className="text-center mb-4">
          <span className="inline-block bg-accent text-white text-xs font-bold px-3 py-1 rounded-full">Transformer</span>
          <h3 className="text-base font-bold text-text mt-2">PurchaseGridTransformer</h3>
          <p className="text-xs text-text-muted mt-1">d=96 · heads=4 · layers=3 · dropout=0.1</p>
        </div>
        <TransformerArch />
      </div>

      <div className="card p-5">
        <div className="text-center mb-4">
          <span className="inline-block bg-violet-600 text-white text-xs font-bold px-3 py-1 rounded-full">Crossformer</span>
          <h3 className="text-base font-bold text-text mt-2">KPI Crossformer</h3>
          <p className="text-xs text-text-muted mt-1">d=64 · heads=4 · layers=2 · routers=8</p>
        </div>
        <CrossformerArch />
      </div>
    </div>
  )
}
