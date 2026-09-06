import React, { useEffect, useState } from 'react'

interface CountdownTimerProps {
  leadTimeMinutes: number | null
  ci90: [number, number] | null
  flareProbability: number
  referenceTimestamp?: string
}

/** Lead time countdown (docs/08 §1.4): visible only when
 * flare_probability > 0.3; MM:SS format with 90% CI band. */
export function CountdownTimer({ leadTimeMinutes, ci90, flareProbability, referenceTimestamp }: CountdownTimerProps) {
  const [remainingSec, setRemainingSec] = useState<number | null>(null)

  useEffect(() => {
    if (flareProbability <= 0.3 || leadTimeMinutes === null) {
      setRemainingSec(null)
      return
    }
    const reference = referenceTimestamp ? new Date(referenceTimestamp).getTime() : Date.now()
    const target = reference + leadTimeMinutes * 60_000
    const update = () => setRemainingSec(Math.max(0, Math.round((target - Date.now()) / 1000)))
    update()
    const id = setInterval(update, 1000)
    return () => clearInterval(id)
  }, [leadTimeMinutes, flareProbability, referenceTimestamp])

  if (remainingSec === null) return null

  const mm = Math.floor(remainingSec / 60).toString().padStart(2, '0')
  const ss = (remainingSec % 60).toString().padStart(2, '0')
  const urgent = remainingSec < 300

  return (
    <div style={{
      background: '#1e293b', borderRadius: '12px', padding: '16px 20px',
      border: `1px solid ${urgent ? '#ef4444' : '#f97316'}`,
    }}>
      <div style={{ fontSize: '0.625rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Estimated Time to Flare Peak
      </div>
      <div style={{ fontSize: '2rem', fontWeight: 700, color: urgent ? '#ef4444' : '#f97316', fontVariantNumeric: 'tabular-nums' }}>
        {mm}:{ss}
      </div>
      {ci90 && (
        <div style={{ fontSize: '0.6875rem', color: '#94a3b8' }}>
          90% CI: {ci90[0].toFixed(0)}–{ci90[1].toFixed(0)} min
        </div>
      )}
    </div>
  )
}
