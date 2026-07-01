import React from 'react'

interface ExplanationCardProps {
  text: string
  uncertainty: number
}

export function ExplanationCard({ text, uncertainty }: ExplanationCardProps) {
  const uncertaintyLabel = uncertainty < 0.1 ? 'Low' : uncertainty < 0.3 ? 'Medium' : 'High'
  const uncertaintyColor = uncertainty < 0.1 ? '#22c55e' : uncertainty < 0.3 ? '#eab308' : '#ef4444'

  return (
    <div style={{
      background: '#1e293b', borderRadius: '12px', padding: '20px',
      border: '1px solid #334155',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Explanation
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{
            width: '6px', height: '6px', borderRadius: '50%',
            background: uncertaintyColor, display: 'inline-block',
          }} />
          <span style={{ fontSize: '0.625rem', color: uncertaintyColor }}>
            {uncertaintyLabel} Uncertainty
          </span>
        </div>
      </div>
      <p style={{
        fontSize: '0.875rem', color: '#e2e8f0', lineHeight: '1.6',
        fontStyle: text === 'Waiting for data...' ? 'italic' : 'normal',
      }}>
        {text}
      </p>
    </div>
  )
}
