import React from 'react'

interface AttentionHeatmapProps {
  weights: number[] | null | undefined
  windowMinutes?: number
}

/** Attention heatmap over the input window (docs/08 §2.2):
 * bright yellow/white = high attention; driver windows (weight > 0.1) marked. */
export function AttentionHeatmap({ weights, windowMinutes = 20 }: AttentionHeatmapProps) {
  const hasData = Array.isArray(weights) && weights.length > 0
  const max = hasData ? Math.max(...(weights as number[])) : 1
  const n = hasData ? (weights as number[]).length : 0

  return (
    <div style={{
      background: '#1e293b', borderRadius: '12px', padding: '20px',
      border: '1px solid #334155',
    }}>
      <h3 style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Attention — Input Window
      </h3>
      {!hasData ? (
        <div style={{ color: '#64748b', fontSize: '0.75rem', padding: '16px 0' }}>
          Attention weights available with the Transformer model.
        </div>
      ) : (
        <>
          <div style={{ display: 'flex', height: '28px', borderRadius: '6px', overflow: 'hidden' }}>
            {(weights as number[]).map((w, i) => {
              const intensity = max > 0 ? w / max : 0
              // dark -> amber -> bright yellow/white
              const color = `rgba(${40 + intensity * 215}, ${40 + intensity * 195}, ${20 + intensity * 60}, ${0.25 + intensity * 0.75})`
              return (
                <div
                  key={i}
                  title={`T-${((n - i) * windowMinutes / n).toFixed(1)} min — weight ${w.toFixed(3)}`}
                  style={{ flex: 1, background: color, minWidth: '2px' }}
                />
              )
            })}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px' }}>
            <span style={{ fontSize: '0.625rem', color: '#64748b' }}>T-{windowMinutes} min</span>
            <span style={{ fontSize: '0.625rem', color: '#64748b' }}>now</span>
          </div>
          <div style={{ marginTop: '6px', fontSize: '0.6875rem', color: '#94a3b8' }}>
            Driver windows (weight &gt; 0.1):{' '}
            <span style={{ color: '#fbbf24', fontWeight: 600 }}>
              {(weights as number[]).filter((w) => w > 0.1).length} of {n}
            </span>
          </div>
        </>
      )}
    </div>
  )
}
