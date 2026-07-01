import React from 'react'

interface ProbabilityGaugeProps {
  probability: number
}

export function ProbabilityGauge({ probability }: ProbabilityGaugeProps) {
  const pct = Math.round(probability * 100)
  const color = probability < 0.3 ? '#22c55e' : probability < 0.7 ? '#eab308' : '#ef4444'
  const bgColor = probability < 0.3 ? '#052e16' : probability < 0.7 ? '#422006' : '#450a0a'
  const radius = 70
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - probability)

  return (
    <div style={{
      background: '#1e293b', borderRadius: '12px', padding: '20px',
      border: '1px solid #334155', textAlign: 'center', flex: 1,
    }}>
      <h3 style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Flare Probability
      </h3>
      <svg width="160" height="160" viewBox="0 0 160 160">
        <circle cx="80" cy="80" r={radius} fill="none" stroke="#334155" strokeWidth="12" />
        <circle
          cx="80" cy="80" r={radius}
          fill="none" stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 80 80)"
          style={{ transition: 'stroke-dashoffset 0.5s ease, stroke 0.3s ease' }}
        />
        <text x="80" y="75" textAnchor="middle" fill="#f8fafc" fontSize="32" fontWeight="700">
          {isNaN(pct) ? '--' : `${pct}%`}
        </text>
        <text x="80" y="95" textAnchor="middle" fill={color} fontSize="11" fontWeight="500">
          {probability < 0.3 ? 'LOW' : probability < 0.7 ? 'MEDIUM' : 'HIGH'}
        </text>
      </svg>
      <div style={{ display: 'flex', justifyContent: 'center', gap: '4px', marginTop: '8px' }}>
        {['green', 'amber', 'red'].map((band) => (
          <span key={band} style={{
            width: '40px', height: '4px', borderRadius: '2px',
            background: band === 'green' ? '#22c55e' : band === 'amber' ? '#eab308' : '#ef4444',
            opacity: band === 'green' ? (probability < 0.3 ? 1 : 0.3) :
                     band === 'amber' ? (probability >= 0.3 && probability < 0.7 ? 1 : 0.3) :
                     (probability >= 0.7 ? 1 : 0.3),
            transition: 'opacity 0.3s ease',
          }} />
        ))}
      </div>
    </div>
  )
}
