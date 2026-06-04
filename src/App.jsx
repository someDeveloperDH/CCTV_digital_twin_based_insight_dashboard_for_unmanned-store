import React, { Suspense, lazy } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'

const RetailDashboard = lazy(() => import('../Dashboard.jsx'))
const ModelShowcase   = lazy(() => import('./model-showcase/ModelShowcase.jsx'))

function Spinner() {
  return (
    <div className="flex items-center justify-center h-screen bg-bg">
      <div className="w-8 h-8 border-4 border-accent border-t-transparent rounded-full animate-spin" />
    </div>
  )
}

export default function App() {
  return (
    <Suspense fallback={<Spinner />}>
      <Routes>
        <Route path="/"       element={<RetailDashboard />} />
        <Route path="/model"  element={<ModelShowcase />} />
        <Route path="*"       element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  )
}
