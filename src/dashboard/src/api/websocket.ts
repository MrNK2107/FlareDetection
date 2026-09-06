export interface SeverityProbs {
  B: number
  C: number
  M: number
  X: number
}

export interface PredictionOutput {
  flare_probability: number
  severity_probs: SeverityProbs
  expected_lead_time_min: number | null
  lead_time_ci_90: [number, number] | null
  solar_state: string
  state_transition_probs: Record<string, number> | null
  dominant_feature: string
  explanation_text: string
  model_uncertainty: number
  inference_timestamp_utc: string
  attention_weights?: number[] | null
  top_features?: Array<{ name: string; value: number; shap: number; direction: string }> | null
  model_version?: string
}

export interface TelemetryPoint {
  timestamp: string
  soft_flux: number
  hard_flux: number
  flare_class?: string
}

export interface AlertMessage {
  flare_probability: number
  severity_probs: SeverityProbs
  expected_lead_time_min: number | null
  explanation_text: string
  deeplink: string
  timestamp_utc: string
}

export type ServerMessage =
  | { type: 'prediction'; [key: string]: unknown }
  | { type: 'telemetry'; window_timestamp_utc: string; points: TelemetryPoint[] }
  | { type: 'alert'; payload: AlertMessage }
  | { type: 'control_ack'; action: string; speed: number; paused: boolean }
  | { type: 'error'; error: string }

export class FlareWebSocket {
  private ws: WebSocket | null = null
  private onMessage: (msg: ServerMessage) => void
  private reconnectAttempts = 0
  private maxReconnectAttempts = 10
  private url: string
  private manuallyClosed = false

  constructor(url: string, onMessage: (msg: ServerMessage) => void) {
    this.url = url
    this.onMessage = onMessage
  }

  connect(): void {
    this.manuallyClosed = false
    try {
      this.ws = new WebSocket(this.url)
      this.ws.onopen = () => {
        this.reconnectAttempts = 0
      }
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          this.onMessage(data as ServerMessage)
        } catch {
          this.onMessage({ type: 'error', error: 'Failed to parse message' })
        }
      }
      this.ws.onerror = () => {
        this.onMessage({ type: 'error', error: 'WebSocket error' })
      }
      this.ws.onclose = () => {
        if (!this.manuallyClosed) this.reconnect()
      }
    } catch {
      this.onMessage({ type: 'error', error: 'Failed to connect' })
      this.reconnect()
    }
  }

  disconnect(): void {
    this.manuallyClosed = true
    this.maxReconnectAttempts = 0
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  sendControl(action: string, value?: number): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'control', action, value }))
    }
  }

  private reconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return
    this.reconnectAttempts++
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000)
    setTimeout(() => this.connect(), delay)
  }
}
