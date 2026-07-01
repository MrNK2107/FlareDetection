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
  private onError: (error: string) => void;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private url: string;

  constructor(
    url: string,
    onPrediction: (data: PredictionOutput) => void,
    onError: (error: string) => void,
  ) {
    this.url = url;
    this.onPrediction = onPrediction;
    this.onError = onError;
  }

  connect(): void {
    try {
      this.ws = new WebSocket(this.url);
      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
      };
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.error) {
            this.onError(data.error);
          } else {
            this.onPrediction(data as PredictionOutput);
          }
        } catch {
          this.onError('Failed to parse message');
        }
      };
      this.ws.onerror = () => {
        this.onError('WebSocket error');
      };
      this.ws.onclose = () => {
        this.reconnect();
      };
    } catch (e) {
      this.onError('Failed to connect');
      this.reconnect();
    }
  }

  disconnect(): void {
    this.maxReconnectAttempts = 0;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  sendFeatures(features: number[], featureNames: string[]): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ features, feature_names: featureNames }));
    }
  }

  private reconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return;
    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000);
    setTimeout(() => this.connect(), delay);
  }
}
