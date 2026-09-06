import React, { useCallback, useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  ReferenceDot, Legend,
} from 'recharts'

interface HistoryEvent {
  peak_utc: string
  flare_class: string
}

interface HistoryAlert {
  ts_utc: string
  flare_probability: number
  expected_lead_time_min: number | null
  explanation_text: string
}

/** Historical view (docs/08 §4): hindcast probability curve vs actual flare
 * onsets, recent alerts, CSV export. */
export function HistoryView() {
  const [days, setDays] = useState(7)
  const [points, setPoints] = useState<Array<Record<string, number | string>>>([])
  const [events, setEvents] = useState<HistoryEvent[]>([])
  const [alerts, setAlerts] = useState<HistoryAlert[]>([])
  const [loading, setLoading] = useState(false)

  const apiBase = import.meta.env.VITE_API_URL || ''

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const end = new Date()
      const start = new Date(Date.now() - days * 86_400_000)
      const [predRes, evRes, alertRes] = await Promise.all([
        fetch(`${apiBase}/history/predictions?start=${start.toISOString()}&end=${end.toISOString()}`),
        fetch(`${apiBase}/history/events?start=${start.toISOString()}&end=${end.toISOString()}`),
        fetch(`${apiBase}/history/alerts?start=${start.toISOString()}&end=${end.toISOString()}`),
      ])
      const predJson = await predRes.json()
      const evJson = await evRes.json()
      const alertJson = await alertRes.json()
      setPoints((predJson.predictions ?? []).map((p: any) => ({
        timestamp: p.ts_utc,
        probability: p.flare_probability,
      })))
      setEvents(evJson.events ?? [])
      setAlerts(alertJson.alerts ?? [])
    } catch (e) {
      console.error('History load failed', e)
    } finally {
      setLoading(false)
    }
  }, [apiBase, days])

  useEffect(() => {
    load()
  }, [load])

  return (
    <div style={{
      background: '#1e293b', borderRadius: '12px', padding: '20px',
      border: '1px solid #334155', marginTop: '16px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ fontSize: '0.875rem', fontWeight: 600, color: '#94a3b8', margin: 0 }}>
          Historical View — Hindcast vs Events
        </h3>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            style={{
              background: '#0f172a', color: '#e2e8f0', border: '1px solid #334155',
              borderRadius: '6px', padding: '4px 8px', fontSize: '0.75rem',
            }}
          >
            <option value={1}>Last 24h</option>
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
          </select>
          <button
            onClick={load}
            style={{
              background: '#334155', color: '#e2e8f0', border: 'none', borderRadius: '6px',
              padding: '4px 12px', fontSize: '0.75rem', cursor: 'pointer',
            }}
          >
            Refresh
          </button>
          <a
            href={`${apiBase}/history/predictions?format=csv&start=${new Date(Date.now() - days * 86_400_000).toISOString()}&end=${new Date().toISOString()}`}
            style={{
              background: '#0ea5e9', color: '#fff', border: 'none', borderRadius: '6px',
              padding: '4px 12px', fontSize: '0.75rem', textDecoration: 'none',
            }}
          >
            Export CSV
          </a>
        </div>
      </div>
      {loading ? (
        <div style={{ color: '#64748b', fontSize: '0.875rem', padding: '24px 0' }}>Loading history…</div>
      ) : points.length === 0 ? (
        <div style={{ color: '#64748b', fontSize: '0.875rem', padding: '24px 0' }}>
          No predictions recorded yet — the model writes history as it runs.
        </div>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={points}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                dataKey="timestamp"
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                tickFormatter={(v: string) => new Date(v).toLocaleString()}
                stroke="#475569"
              />
              <YAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 10 }} stroke="#475569" />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
              />
              <Legend />
              <Line type="monotone" dataKey="probability" name="Flare probability"
                stroke="#0ea5e9" dot={false} strokeWidth={2} />
              {events.map((ev, i) => (
                <ReferenceDot
                  key={i}
                  x={ev.peak_utc}
                  y={1}
                  r={5}
                  fill={ev.flare_class === 'X' ? '#dc2626' : ev.flare_class === 'M' ? '#ef4444' : '#f97316'}
                  stroke="#fff"
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
          <div style={{ fontSize: '0.6875rem', color: '#64748b', marginTop: '4px' }}>
            Markers: flare peaks from the catalogue (orange C, red M, dark red X)
          </div>
          {alerts.length > 0 && (
            <div style={{ marginTop: '12px' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '6px' }}>
                Recent alerts
              </div>
              {alerts.slice(-5).reverse().map((a, i) => (
                <div key={i} style={{
                  display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem',
                  color: '#cbd5e1', padding: '4px 0', borderBottom: '1px solid #0f172a',
                }}>
                  <span>{new Date(a.ts_utc).toLocaleString()} — p={a.flare_probability.toFixed(2)}</span>
                  <span style={{ color: '#64748b' }}>
                    {a.expected_lead_time_min ? `lead ${a.expected_lead_time_min.toFixed(0)} min` : ''}
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}
