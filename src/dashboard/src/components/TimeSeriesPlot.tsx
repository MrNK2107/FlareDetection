import React from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'

interface TimePoint {
  timestamp: string
  soft_flux: number
  hard_flux: number
}

interface TimeSeriesPlotProps {
  data: TimePoint[]
}

export function TimeSeriesPlot({ data }: TimeSeriesPlotProps) {
  return (
    <div style={{
      background: '#1e293b', borderRadius: '12px', padding: '20px',
      border: '1px solid #334155',
    }}>
      <h3 style={{ fontSize: '0.875rem', fontWeight: 600, color: '#94a3b8', marginBottom: '12px' }}>
        X-Ray Flux — Last 60 Minutes
      </h3>
      {data.length === 0 ? (
        <div style={{
          height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#64748b', fontSize: '0.875rem',
        }}>
          Waiting for telemetry data...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="timestamp"
              tick={{ fill: '#94a3b8', fontSize: 10 }}
              tickFormatter={(v: string) => new Date(v).toLocaleTimeString()}
              stroke="#475569"
            />
            <YAxis
              yAxisId="left"
              tick={{ fill: '#94a3b8', fontSize: 10 }}
              stroke="#fb923c"
              label={{ value: 'Soft X-ray', angle: -90, position: 'insideLeft', fill: '#fb923c', fontSize: 11 }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              tick={{ fill: '#94a3b8', fontSize: 10 }}
              stroke="#ef4444"
              label={{ value: 'Hard X-ray', angle: 90, position: 'insideRight', fill: '#ef4444', fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
              labelStyle={{ color: '#e2e8f0' }}
            />
            <Legend />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="soft_flux"
              stroke="#fb923c"
              name="Soft X-ray"
              dot={false}
              strokeWidth={2}
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="hard_flux"
              stroke="#ef4444"
              name="Hard X-ray"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
