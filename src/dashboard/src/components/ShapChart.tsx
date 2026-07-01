import React from 'react'

interface ShapFeature {
  name: string
  value: number
  shap: number
  direction: 'increasing' | 'decreasing'
}

interface ShapChartProps {
  features: ShapFeature[]
}

export function ShapChart({ features }: ShapChartProps) {
  if (!features.length) {
    return (
      <div style={{
        background: '#1e293b', borderRadius: '12px', padding: '20px',
        border: '1px solid #334155',
      }}>
        <h3 style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Feature Attribution (SHAP)
        </h3>
        <div style={{ color: '#64748b', fontSize: '0.875rem', textAlign: 'center', padding: '20px' }}>
          No SHAP data available
        </div>
      </div>
    )
  }

  const maxAbsShap = Math.max(...features.map(f => Math.abs(f.shap)), 0.001)

  return (
    <div style={{
      background: '#1e293b', borderRadius: '12px', padding: '20px',
      border: '1px solid #334155',
    }}>
      <h3 style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Feature Attribution (SHAP)
      </h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {features.map((f) => {
          const widthPct = Math.abs(f.shap) / maxAbsShap * 100
          const isPos = f.direction === 'increasing'
          return (
            <div key={f.name}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                <span style={{ fontSize: '0.75rem', color: '#cbd5e1' }}>
                  {f.name.length > 25 ? f.name.slice(0, 25) + '...' : f.name}
                </span>
                <span style={{ fontSize: '0.75rem', color: isPos ? '#fb923c' : '#60a5fa' }}>
                  {isPos ? '+' : ''}{f.shap.toFixed(3)}
                </span>
              </div>
              <div style={{ background: '#0f172a', borderRadius: '4px', height: '8px', position: 'relative', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${widthPct}%`,
                  background: isPos ? '#fb923c' : '#60a5fa',
                  borderRadius: '4px',
                  float: isPos ? 'left' : 'right',
                  transition: 'width 0.3s ease',
                }} />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
