import React, { useEffect, useState } from 'react'

interface CountdownTimerProps {
  leadTimeMinutes: number | null
  ci90: [number, number] | null
  flareProbability: number
  referenceTimestamp?: string
}

/**
 * Lead time countdown (docs/08 §1.4): visible only when flare_probability
 * > 0.3; MM:SS format with 90% CI band.
 *
 * Replay-aware: the server replays historical telemetry, so
 * `inference_timestamp_utc` may be far in the past. The countdown is only
 * live when the prediction is fresh (reference within 2 minutes of now);
 * for replayed predictions it renders the lead time statically instead of
 * showing a bogus 00:00 that is always expired.
 */
const FRESH_THRESHOLD_MS = 2 * 60_000

export function CountdownTimer({ leadTimeMinutes, ci90, flareProbability, referenceTimestamp }: CountdownTimerProps) {
  const [remainingSec, setRemainingSec] = useState<number | null>(null)
  const [isFresh, setIsFresh] = useState(true)

  useEffect(() => {
    if (flareProbability <= 0.3 || leadTimeMinutes === null) {
      setRemainingSec(null)
      return
    }
    const referenceMs = referenceTimestamp ? new Date(referenceTimestamp).getTime() : Date.now()
    const fresh = Number.isFinite(referenceMs) && Math.abs(Date.now() - referenceMs) < FRESH_THRESHOLD_MS
    setIsFresh(fresh)
    if (!fresh) {
      // Replay mode: static display, no ticking
      setRemainingSec(Math.round(leadTimeMinutes * 60))
      return
    }
    const target = referenceMs + leadTimeMinutes * 60_000
    const update = () => setRemainingSec(Math.max(0, Math.round((target - Date.now()) / 1000)))
    update()
    const id = setInterval(update, 1000)
    return () => clearInterval(id)
  }, [leadTimeMinutes, flareProbability, referenceTimestamp])

  if (remainingSec === null) return null

  const mm = Math.floor(remainingSec / 60).toString().padStart(2, '0')
  const ss = (remainingSec % 60).toString().padStart(2, '0')
  const urgent = isFresh && remainingSec < 300

  return (
    <div style={{
      background: '#1e293b', borderRadius: '12px', padding: '16px 20px',
      border: `1px solid ${urgent ? '#ef4444' : '#f97316'}`,
    }}>
      <div style={{ fontSize: '0.625rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Estimated Time to Flare Peak {isFresh ? '' : '(replay)'}
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
