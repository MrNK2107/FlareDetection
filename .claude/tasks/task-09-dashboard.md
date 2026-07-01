# Task 9: Dashboard

**Dependencies**: Task 8 (Inference API)
**Estimated effort**: Large
**PRD Reference**: Phase 7 — Dashboard & Alerting (§3.7)

## Objective

Build a React dashboard with Recharts and shadcn/ui that connects to the FastAPI WebSocket for real-time solar flare monitoring.

## Files to Create

- `src/dashboard/package.json`
- `src/dashboard/vite.config.ts`
- `src/dashboard/tsconfig.json`
- `src/dashboard/tsconfig.node.json`
- `src/dashboard/index.html`
- `src/dashboard/tailwind.config.js`
- `src/dashboard/postcss.config.js`
- `src/dashboard/src/main.tsx`
- `src/dashboard/src/App.tsx`
- `src/dashboard/src/api/websocket.ts`
- `src/dashboard/src/components/TimeSeriesPlot.tsx`
- `src/dashboard/src/components/ProbabilityGauge.tsx`
- `src/dashboard/src/components/StateIndicator.tsx`
- `src/dashboard/src/components/ShapChart.tsx`
- `src/dashboard/src/components/ExplanationCard.tsx`
- `src/dashboard/src/components/Layout.tsx`

## Implementation Steps

### 1. Project Setup

Initialize a Vite + React + TypeScript project:

```json
// package.json — key dependencies
{
  "dependencies": {
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "recharts": "^2.12.0",
    "lucide-react": "^0.400.0",
    "tailwindcss": "^3.4.0",
    "@radix-ui/react-progress": "^1.1.0",
    "@radix-ui/react-card": "^1.0.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.0",
    "vite": "^6.0.0",
    "typescript": "^5.5.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0"
  }
}
```

### 2. WebSocket Client (`src/api/websocket.ts`)

```typescript
export interface PredictionOutput {
  flare_probability: number;
  severity_probs: Record<string, number>;
  expected_lead_time_min: number | null;
  lead_time_ci_90: [number, number] | null;
  solar_state: string;
  state_transition_probs: Record<string, number> | null;
  dominant_feature: string;
  explanation_text: string;
  model_uncertainty: number;
  inference_timestamp_utc: string;
}

export class FlareWebSocket {
  private ws: WebSocket | null = null;
  private onPrediction: (data: PredictionOutput) => void;
  private onError: (error: Event) => void;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;

  constructor(
    url: string,
    onPrediction: (data: PredictionOutput) => void,
    onError: (error: Event) => void
  ) { ... }

  connect(): void { ... }
  disconnect(): void { ... }
  sendFeatures(features: number[], featureNames: string[]): void { ... }
  private handleMessage(event: MessageEvent): void { ... }
  private reconnect(): void { ... }
}
```

**Handling edge cases**:
- WebSocket disconnects → exponential backoff reconnection (1s, 2s, 4s, ...)
- Server returns error → display error state, don't crash
- Empty/malformed message → log and ignore

### 3. Main Layout (`src/App.tsx`)

```
┌─────────────────────────────────────────────┐
│  Header: FlareClassifier — Real-Time Monitor │
├──────────────────┬──────────────────────────┤
│                  │                          │
│   Time Series    │   Probability Gauge      │
│   Plot           │   + State Indicator      │
│   (60 min)       │   + Lead Time            │
│                  │                          │
├──────────────────┴──────────────────────────┤
│   SHAP Bar Chart   │   Explanation Card     │
│                    │                        │
└─────────────────────────────────────────────┘
```

### 4. `TimeSeriesPlot.tsx`

```typescript
interface TimeSeriesPlotProps {
  data: Array<{
    timestamp: string;
    soft_flux: number;
    hard_flux: number;
  }>;
  attentionWeights?: Array<{ time: string; weight: number }>;
}
```

- Dual Y-axis: soft X-ray (left) and hard X-ray (right) using Recharts
- Autoscrolling: latest 60 minutes, updates every 10 seconds
- Color scheme: soft=orange, hard=red
- Attention overlay: highlight regions where attention weight > 0.1
- Responsive container

**Edge cases**:
- No data → show "Waiting for data..." placeholder
- All zero values → axis auto-scales
- Extreme values → auto-scale Y-axis (use domain=['auto', 'auto'])

### 5. `ProbabilityGauge.tsx`

```typescript
interface ProbabilityGaugeProps {
  probability: number;  // 0-1
}
```

- Circular gauge with color bands:
  - Green: < 0.3 (low)
  - Amber: 0.3–0.7 (medium)
  - Red: > 0.7 (high)
- Animated needle or arc fill
- Numeric label in center: "87%"
- Use SVG (not a library) for the gauge to keep it lightweight

**Edge cases**:
- probability = 0 → fully green, needle at 0
- probability = 1 → fully red, needle at max
- probability = NaN → show "—" (dash)

### 6. `StateIndicator.tsx`

```typescript
interface StateIndicatorProps {
  currentState: string;           // 'Quiet', 'Energy Accumulation', etc.
  transitionProbs: Record<string, number> | null;
  leadTimeMinutes: number | null;
}
```

- Current state label with color coding
- Animated transition arrow to most probable next state
- Lead time countdown (only when flare_probability > 0.3)

**Edge cases**:
- Unknown state → gray with "Awaiting data"
- No transition probs → hide transition arrow
- leadTimeMinutes = null → hide countdown section

### 7. `ShapChart.tsx`

```typescript
interface ShapChartProps {
  features: Array<{
    name: string;
    value: number;
    shap: number;
    direction: 'increasing' | 'decreasing';
  }>;
}
```

- Horizontal bar chart (top 5 features)
- Positive SHAP = red bar (increasing probability)
- Negative SHAP = blue bar (decreasing probability)
- Sort by |SHAP| descending
- Use Recharts horizontal BarChart

**Edge cases**:
- Empty features array → "No SHAP values available"
- All SHAP values near zero → "All features near baseline"
- Very long feature names → truncate with ellipsis + tooltip

### 8. `ExplanationCard.tsx`

```typescript
interface ExplanationCardProps {
  text: string;
  uncertainty: number;
}
```

- Card with explanation text
- Uncertainty indicator (low/medium/high badge)
- Timestamp display

**Edge cases**:
- Empty text → "Explanation not available"
- Very high uncertainty → add "⚠️ High uncertainty" warning badge

### 9. Alerts (Browser Notification)

```typescript
export function requestNotificationPermission(): void { ... }

export function sendBrowserAlert(prediction: PredictionOutput): void {
  if (Notification.permission === 'granted') {
    new Notification('⚠️ Flare Alert', {
      body: `Probability: ${(prediction.flare_probability * 100).toFixed(0)}% | ` +
             `Lead time: ${prediction.expected_lead_time_min?.toFixed(0)} min | ` +
             `State: ${prediction.solar_state}`,
      icon: '/flare-icon.png'
    });
  }
}
```

## Acceptance Criteria

- [ ] Dashboard loads on `npm run dev` (Vite dev server)
- [ ] Connects to FastAPI WebSocket at `ws://localhost:8000/ws/stream`
- [ ] Time series plot updates in real-time (autoscrolling)
- [ ] Probability gauge shows correct color band and value
- [ ] State indicator displays current state
- [ ] SHAP bar chart renders top 5 features
- [ ] Explanation card shows readable text
- [ ] Lead time countdown appears when probability > 0.3
- [ ] Browser notifications trigger on alert threshold
- [ ] All components handle empty/error states gracefully

## Verification

```bash
# Build and verify
cd src/dashboard
npm install
npm run build
echo "Dashboard build complete: check src/dashboard/dist/"
```
