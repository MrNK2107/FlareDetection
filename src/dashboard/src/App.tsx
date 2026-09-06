import React, { useState, useEffect, useCallback, useRef } from 'react'
import {
  FlareWebSocket, ServerMessage, PredictionOutput, TelemetryPoint,
} from './api/websocket'
import { TimeSeriesPlot } from './components/TimeSeriesPlot'
import { ProbabilityGauge } from './components/ProbabilityGauge'
import { StateIndicator } from './components/StateIndicator'
import { ShapChart } from './components/ShapChart'
import { ExplanationCard } from './components/ExplanationCard'
import { AttentionHeatmap } from './components/AttentionHeatmap'
import { CountdownTimer } from './components/CountdownTimer'
import { HistoryView } from './components/HistoryView'

interface TimePoint {
  timestamp: string
  soft_flux: number
  hard_flux: number
}

interface ShapFeature {
  name: string
  value: number
  shap: number
  direction: 'increasing' | 'decreasing'
}

interface AlertBanner {
  id: number
  probability: number
  text: string
  leadTime: number | null
}

export default function App() {
  const [prediction, setPrediction] = useState<PredictionOutput | null>(null)
  const [timeSeries, setTimeSeries] = useState<TimePoint[]>([])
  const [shapFeatures, setShapFeatures] = useState<ShapFeature[]>([])
  const [alerts, setAlerts] = useState<AlertBanner[]>([])
  const [connected, setConnected] = useState(false)
  const [paused, setPaused] = useState(false)
  const [speed, setSpeed] = useState(1.0)
  const wsRef = useRef<FlareWebSocket | null>(null)
  const alertIdRef = useRef(0)

  const handleMessage = useCallback((msg: ServerMessage) => {
    switch (msg.type) {
      case 'telemetry': {
        const incoming = (msg.points as TelemetryPoint[]).map((p) => ({
          timestamp: p.timestamp,
          soft_flux: p.soft_flux,
          hard_flux: p.hard_flux,
        }))
        setTimeSeries((prev) => {
          const next = [...prev, ...incoming]
          // de-dupe by timestamp, then keep the last 360 points (60 min)
          const seen = new Set<string>()
          const deduped = next.filter((p) => {
            if (seen.has(p.timestamp)) return false
            seen.add(p.timestamp)
            return true
          })
          return deduped.slice(-360)
        })
        break
      }
      case 'prediction': {
        const pred = msg as unknown as PredictionOutput
        setPrediction(pred)
        const top = (pred as any).top_features
        if (Array.isArray(top) && top.length > 0) {
          setShapFeatures(top.map((f: any) => ({
            name: f.name,
            value: f.value,
            shap: f.shap,
            direction: f.shap >= 0 ? 'increasing' : 'decreasing',
          })))
        }
        break
      }
      case 'alert': {
        const a = msg.payload
        alertIdRef.current += 1
        const id = alertIdRef.current
        setAlerts((prev) => [...prev, {
          id, probability: a.flare_probability, text: a.explanation_text, leadTime: a.expected_lead_time_min,
        }])
        if (typeof Notification !== 'undefined') {
          if (Notification.permission === 'granted') {
            new Notification(`Flare alert: ${(a.flare_probability * 100).toFixed(0)}%`, {
              body: a.explanation_text,
            })
          } else if (Notification.permission !== 'denied') {
            Notification.requestPermission()
          }
        }
        setTimeout(() => {
          setAlerts((prev) => prev.filter((x) => x.id !== id))
        }, 30000)
        break
      }
      case 'error':
        console.error('WS error:', msg.error)
        setConnected(false)
        break
      default:
        break
    }
  }, [])

  useEffect(() => {
    const apiUrl = import.meta.env.VITE_API_URL || ''
    let wsUrl: string
    if (apiUrl) {
      const protocol = apiUrl.startsWith('https') ? 'wss:' : 'ws:'
      const host = apiUrl.replace(/^https?:\/\//, '')
      wsUrl = `${protocol}//${host}/ws/stream`
    } else if (window.location.hostname === 'localhost') {
      wsUrl = 'ws://localhost:8000/ws/stream'
    } else {
      wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/stream`
    }
    const ws = new FlareWebSocket(wsUrl, handleMessage)
    wsRef.current = ws
    ws.connect()
    setConnected(true)
    return () => ws.disconnect()
  }, [handleMessage])

  const togglePause = () => {
    const next = !paused
    setPaused(next)
    wsRef.current?.sendControl(next ? 'pause' : 'resume')
  }

  const changeSpeed = (s: number) => {
    setSpeed(s)
    wsRef.current?.sendControl('set_speed', s)
  }

  return (
    <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto' }}>
      <header style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc', margin: 0 }}>
            FlareClassifier
          </h1>
          <p style={{ fontSize: '0.875rem', color: '#94a3b8', margin: 0 }}>
            Aditya-L1 Solar Flare Forecast — Real-Time Monitor
          </p>
        </div>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          {prediction?.model_version && (
            <span style={{
              fontSize: '0.625rem', color: '#94a3b8', background: '#0f172a',
              padding: '2px 8px', borderRadius: '999px', border: '1px solid #334155',
            }}>
              {prediction.model_version}
            </span>
          )}
          <button onClick={togglePause} style={{
            background: '#334155', color: '#e2e8f0', border: 'none', borderRadius: '6px',
            padding: '4px 12px', fontSize: '0.75rem', cursor: 'pointer',
          }}>
            {paused ? '▶ Resume' : '⏸ Pause'}
          </button>
          <select
            value={speed}
            onChange={(e) => changeSpeed(Number(e.target.value))}
            style={{
              background: '#0f172a', color: '#e2e8f0', border: '1px solid #334155',
              borderRadius: '6px', padding: '4px 8px', fontSize: '0.75rem',
            }}
          >
            {[1, 10, 60].map((s) => (
              <option key={s} value={s}>×{s}</option>
            ))}
          </select>
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

      {alerts.map((a) => (
        <div key={a.id} style={{
          background: '#450a0a', border: '1px solid #ef4444', borderRadius: '10px',
          padding: '12px 16px', marginBottom: '12px', color: '#fecaca',
        }}>
          <strong>⚠ Flare alert — {(a.probability * 100).toFixed(0)}%</strong>
          {a.leadTime !== null && <> · peak in ~{a.leadTime.toFixed(0)} min</>}
          <div style={{ fontSize: '0.75rem', marginTop: '4px' }}>{a.text}</div>
        </div>
      ))}

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '16px', marginBottom: '16px' }}>
        <TimeSeriesPlot data={timeSeries} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <ProbabilityGauge probability={prediction?.flare_probability ?? 0} />
          <StateIndicator
            currentState={prediction?.solar_state ?? 'Awaiting data'}
            transitionProbs={prediction?.state_transition_probs ?? null}
            leadTimeMinutes={prediction?.expected_lead_time_min ?? null}
          />
          <CountdownTimer
            leadTimeMinutes={prediction?.expected_lead_time_min ?? null}
            ci90={prediction?.lead_time_ci_90 ?? null}
            flareProbability={prediction?.flare_probability ?? 0}
            referenceTimestamp={prediction?.inference_timestamp_utc}
          />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <ShapChart features={shapFeatures} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <AttentionHeatmap
            weights={prediction?.attention_weights ?? null}
          />
          <ExplanationCard
            text={prediction?.explanation_text ?? 'Waiting for data...'}
            uncertainty={prediction?.model_uncertainty ?? 0}
          />
        </div>
      </div>

      <HistoryView />

      <footer style={{ marginTop: '24px', textAlign: 'center', fontSize: '0.75rem', color: '#475569' }}>
        FlareClassifier v1.0.0 — {prediction ? prediction.model_version : 'connecting…'}
      </footer>
    </div>
  )
}
