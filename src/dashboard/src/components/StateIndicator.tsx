import React from 'react'

interface StateIndicatorProps {
  currentState: string
  transitionProbs: Record<string, number> | null
  leadTimeMinutes: number | null
}

const STATE_COLORS: Record<string, string> = {
  'Quiet': '#22c55e',
  'Energy Accumulation': '#eab308',
  'Precursor': '#f97316',
  'Initiation': '#ef4444',
  'Peak': '#dc2626',
  'Decay': '#a855f7',
  'Unknown': '#64748b',
  'Awaiting data': '#64748b',
}

export function StateIndicator({ currentState, transitionProbs, leadTimeMinutes }: StateIndicatorProps) {
  const color = STATE_COLORS[currentState] ?? '#64748b'
  const mostProbableNext = transitionProbs
    ? Object.entries(transitionProbs).sort(([, a], [, b]) => b - a)[0]
    : null

  return (
    <div style={{
      background: '#1e293b', borderRadius: '12px', padding: '20px',
      border: '1px solid #334155',
    }}>
      <h3 style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Solar State
      </h3>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
        <span style={{
          width: '12px', height: '12px', borderRadius: '50%',
          background: color, display: 'inline-block',
        }} />
        <span style={{ fontSize: '1.125rem', fontWeight: 600, color: '#f8fafc' }}>
          {currentState}
        </span>
      </div>
      {mostProbableNext && (
        <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '8px' }}>
          Next likely: <span style={{ color: '#e2e8f0' }}>{mostProbableNext[0]}</span>
          {' '}({(mostProbableNext[1] * 100).toFixed(0)}%)
        </div>
      )}
      {leadTimeMinutes !== null && (
        <div style={{
          background: '#0f172a', borderRadius: '8px', padding: '8px 12px',
          marginTop: '8px',
        }}>
          <div style={{ fontSize: '0.625rem', color: '#64748b', textTransform: 'uppercase' }}>Lead Time</div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>
            {leadTimeMinutes.toFixed(0)} <span style={{ fontSize: '0.75rem', fontWeight: 400, color: '#94a3b8' }}>min</span>
          </div>
        </div>
      )}
    </div>
  )
}
