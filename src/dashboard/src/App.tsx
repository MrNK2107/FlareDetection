import React, { useState, useEffect, useCallback, useRef } from 'react'
import { FlareWebSocket, PredictionOutput } from './api/websocket'
import { TimeSeriesPlot } from './components/TimeSeriesPlot'
import { ProbabilityGauge } from './components/ProbabilityGauge'
import { StateIndicator } from './components/StateIndicator'
import { ShapChart } from './components/ShapChart'
import { ExplanationCard } from './components/ExplanationCard'

interface TimePoint {
  timestamp: string
  soft_flux: number
  hard_flux: number
}

export default function App() {
  const [prediction, setPrediction] = useState<PredictionOutput | null>(null)
  const [timeSeries, setTimeSeries] = useState<TimePoint[]>([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<FlareWebSocket | null>(null)

  const onPrediction = useCallback((data: PredictionOutput) => {
    setPrediction(data)
    setTimeSeries(prev => {
      const now = new Date().toISOString()
      const point: TimePoint = {
        timestamp: now,
        soft_flux: data.severity_probs.B + data.severity_probs.C,
        hard_flux: data.flare_probability,
      }
      const next = [...prev, point]
      return next.length > 360 ? next.slice(-360) : next
    })
  }, [])

  const onError = useCallback((error: string) => {
    console.error('WS error:', error)
    setConnected(false)
  }, [])

  useEffect(() => {
    const apiUrl = import.meta.env.VITE_API_URL || ''
    if (apiUrl) {
      const protocol = apiUrl.startsWith('https') ? 'wss:' : 'ws:'
      const host = apiUrl.replace(/^https?:\/\//, '')
      var wsUrl = `${protocol}//${host}/ws/stream`
    } else if (window.location.hostname === 'localhost') {
      var wsUrl = 'ws://localhost:8000/ws/stream'
    } else {
      var wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/stream`
    }
    const ws = new FlareWebSocket(wsUrl, onPrediction, onError)
    wsRef.current = ws
    ws.connect()
    setConnected(true)
    return () => { ws.disconnect() }
  }, [onPrediction, onError])

  useEffect(() => {
    if (!connected || !prediction) return
    const interval = setInterval(() => {
      const features = new Array(43).fill(0)
      const featureNames = Array.from({ length: 43 }, (_, i) => `f${i}`)
      wsRef.current?.sendFeatures(features, featureNames)
    }, 10000)
    return () => clearInterval(interval)
  }, [connected, prediction])

  return (
    <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto' }}>
      <header style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc' }}>
            FlareClassifier
          </h1>
          <p style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
            Aditya-L1 Solar Flare Forecast — Real-Time Monitor
          </p>
        </div>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <span style={{
            width: '8px', height: '8px', borderRadius: '50%',
            background: connected ? '#22c55e' : '#ef4444',
            display: 'inline-block',
          }} />
          <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '16px', marginBottom: '16px' }}>
        <TimeSeriesPlot data={timeSeries} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <ProbabilityGauge probability={prediction?.flare_probability ?? 0} />
          <StateIndicator
            currentState={prediction?.solar_state ?? 'Awaiting data'}
            transitionProbs={prediction?.state_transition_probs ?? null}
            leadTimeMinutes={prediction?.expected_lead_time_min ?? null}
          />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <ShapChart
          features={[
            { name: 'dsoft_dt_mean', value: 0, shap: 0.32, direction: 'increasing' },
            { name: 'peak_lag_s', value: 0, shap: -0.18, direction: 'decreasing' },
            { name: 'thermal_fraction', value: 0, shap: 0.12, direction: 'increasing' },
            { name: 'soft_flux_mean', value: 0, shap: -0.08, direction: 'decreasing' },
            { name: 'dhard_dt_mean', value: 0, shap: 0.05, direction: 'increasing' },
          ]}
        />
        <ExplanationCard
          text={prediction?.explanation_text ?? 'Waiting for data...'}
          uncertainty={prediction?.model_uncertainty ?? 0}
        />
      </div>

      <footer style={{ marginTop: '24px', textAlign: 'center', fontSize: '0.75rem', color: '#475569' }}>
        FlareClassifier v0.1.0 — MVP
      </footer>
    </div>
  )
}
