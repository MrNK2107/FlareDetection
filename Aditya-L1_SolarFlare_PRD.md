**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

## **PRODUCT REQUIREMENTS DOCUMENT** 

## **Aditya-L1 Solar Flare Forecasting System** 

AI-Powered Early Warning Engine for Solar Events 

|**Version**|1.0|
|---|---|
|**Status**|Draft — For Developer<br>Review|
|**Data Source**|ISRO Aditya-L1 (SoLEXS<br>+ HEL1OS)|
|**Audience**|Engineering & ML Team|



Page 1 

Confidential — Internal Use Only 

**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

## **1. Overview & Concept Validation** 

This document defines the requirements for a solar flare forecasting system built on real-time X-ray telemetry from ISRO's Aditya-L1 spacecraft. The system ingests dual-channel X-ray data (soft and hard), processes it through a multi-stage signal and ML pipeline, and produces probabilistic flare forecasts with explainable outputs. 

## **1.1  Why This Concept Is Sound** 

The core insight is borrowed directly from clinical monitoring: just as cardiologists read an ECG not for raw voltage but for peaks, slopes, rhythm, and cross-signal relationships, this system treats solar X-ray telemetry the same way. That analogy is technically valid for three reasons: 

- Solar flares follow pre-event signatures. Soft X-ray flux typically rises minutes before hard X-ray emission accelerates, giving a measurable precursor window. 

- The two channels are complementary, not redundant. Soft X-rays (SoLEXS) capture thermal plasma emission; hard X-rays (HEL1OS) reflect non-thermal electron acceleration. Their lag correlation carries physical meaning. 

- Signal structure, not raw amplitude, is the predictive signal. Rates of change, frequency-domain patterns, and cross-channel phase differences are more predictive than instantaneous flux values alone. 

## **Validation Verdict** 

The proposal is technically grounded. The ECG analogy, multi-stream encoding, and state-machine framing are all defensible. The key risks — label scarcity, class imbalance, and unsupervised state inference — are real but solvable with standard techniques documented in this PRD. 

## **1.2  Risk Register** 

|**Risk**|**Description**|**Mitigation**|
|---|---|---|
|Class Imbalance|Sun is quiet >95% of the time. Naive<br>accuracy is misleading.|Focal loss, SMOTE on feature<br>space, anomaly-aware sampling.|
|Hidden State Labels|No ground truth for 'precursor' or<br>'energy accumulation' states.|Unsupervised clustering + weak<br>supervision from GOES flare<br>catalogue.|
|Small Labeled Dataset|Scientific datasets are small. Deep<br>models may overfit.|Feature engineering + classical<br>baselines first. Deep learning<br>second.|
|Sensor Gaps|Aditya-L1 may have telemetry|Robust imputation; flag-uncertain|



Page 2 

Confidential — Internal Use Only 

**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

|**Risk**|**Description**|**Mitigation**|
|---|---|---|
||dropouts or calibration drift.|outputs during gap windows.|
|Solar Cycle Drift|Model trained on Cycle 25 data may<br>degrade over years.|Continuous learning pipeline with<br>periodic evaluation and retraining.|



Page 3 

Confidential — Internal Use Only 

**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

## **2. System Architecture** 

The system is divided into seven loosely coupled stages. Each stage has defined inputs, outputs, and owning workstream. This separation allows parallel development and independent testing. 

|**#**|**Stage**|**Input → Output**|**Owner**|**Key Tech**|
|---|---|---|---|---|
|1|Data Ingestion &<br>Alignment|Raw telemetry →<br>Synchronized DataFrame|Data Eng|pandas, HDF5,<br>netCDF4|
|2|Signal Processing|Synchronized DF →<br>Windowed feature arrays|Data Eng|scipy,<br>PyWavelets|
|3|Feature Engineering|Feature arrays → Enriched<br>feature matrix|Physics + ML|tsfresh, ruptures|
|4|Physics Layer|Feature matrix → Physics-<br>aware features|Physics|Custom formulae|
|5|ML Model|Features → Probability +<br>lead time|ML|PyTorch,<br>Hugging Face|
|6|Explainability|Model outputs → Human-<br>readable reasons|ML|SHAP, Captum|
|7|Dashboard & Alerts|All outputs → Operator<br>interface|Frontend|React, FastAPI,<br>WebSockets|



Page 4 

Confidential — Internal Use Only 

**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

## **3. Phase-by-Phase Developer Requirements** 

## **Phase 1 — Data Engineering** 

This is the most important phase. Poor alignment or inconsistent normalization will corrupt every downstream component. Do not skip to modeling until this pipeline passes all validation checks. 

## **3.1.1  Timestamp Synchronization** 

- Resample both SoLEXS (soft X-ray) and HEL1OS (hard X-ray) streams to a common UTCaligned frequency of 1 second or the instrument's native cadence, whichever is coarser. 

- Use forward-fill for gaps shorter than 5 seconds. Flag and isolate gaps longer than 5 seconds for downstream uncertainty propagation. 

- Store synchronized data in a single Parquet or HDF5 file with columns: [timestamp_utc, soft_flux, hard_flux, soft_quality_flag, hard_quality_flag]. 

## **3.1.2  Cleaning** 

- Remove rows where either quality flag indicates sensor glitch or calibration event. 

- Detect and remove duplicate timestamps (keep first occurrence). 

- Clip values to physically plausible ranges (defined per instrument specification). Log clipped rows. 

## **3.1.3  Normalization** 

- Do NOT use global min-max normalization. Solar flux varies strongly with the solar cycle. 

- Use rolling z-score normalization with a window of 6 hours: z(t) = (x(t) - mean(t-6h to t)) / std(t-6h to t). 

- Apply normalization independently to each channel. 

## **3.1.4  Windowing** 

- Generate sliding windows of 20 minutes with a stride of 10 seconds for training data. 

- Label each window by the maximum GOES class occurring in the 30-minute future window (B, C, M, X, or None). Source labels from the NOAA/GOES flare catalogue. 

- Store windows as numpy memmap or torch Dataset for memory efficiency on large datasets. 

## **Acceptance Criterion — Phase 1** 

100% of windows must have aligned timestamps. Missing data rate after cleaning must be < 2%. Unit tests must assert that rolling z-score mean ≈ 0 and std ≈ 1 for any 6-hour slice. 

Page 5 

Confidential — Internal Use Only 

**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

## **Phase 2 — Feature Engineering** 

Feature engineering is where this system will separate itself from baseline approaches. Implement features in layers. Validate each layer independently before adding the next. 

|**Layer**|**Features**|**Implementation Notes**|
|---|---|---|
|L1 — Raw|soft_flux, hard_flux|Direct from cleaned pipeline.|
|L2 —<br>Dynamics|dSoft/dt, dHard/dt, d²Soft/dt²,<br>rolling_mean(60s, 300s),<br>rolling_std(60s, 300s),<br>rolling_variance(300s)|Use np.gradient for derivatives. Rolling with<br>pandas.|
|L3 — Cross-<br>channel|soft/hard ratio, soft-hard<br>difference, Pearson<br>correlation(lag=0s..120s), phase<br>difference via Hilbert transform|Lag-correlation sweep across 0–120s window.<br>Store peak lag as feature.|
|L4 —<br>Frequency|FFT power in bands [0.001–0.01<br>Hz, 0.01–0.1 Hz, 0.1–1 Hz],<br>wavelet energy (Morlet, scales 8<br>–256s), spectrogram entropy|Use scipy.signal.welch for PSD. PyWavelets for<br>CWT.|
|L5 —<br>Change-<br>point|CUSUM alarm flag, Bayesian<br>change-point posterior, ruptures<br>breakpoint distance (in seconds)|Use ruptures library with PELT search. Treat each<br>detected break as a binary flag and a distance<br>feature.|



## **Key Physics Observation** 

The lag at which the cross-channel Pearson correlation peaks — measured over a 2-minute sliding window — is one of the strongest single precursor features. Prioritize this in feature importance analysis. 

## **Phase 3 — Physics-Aware Feature Layer** 

This layer encodes solar physics domain knowledge as additional features. The goal is not to hardcode rules, but to give the model informative inputs that it might not discover from raw data alone. 

## **3.3.1  Stage Indicators** 

- Compute soft_rising: 1 if dSoft/dt > threshold for 3 consecutive windows. Threshold = 95th percentile of dSoft/dt in quiet-Sun baseline. 

- Compute hard_accelerating: 1 if d²Hard/dt² > threshold for 2 consecutive windows. 

- Compute precursor_candidate: soft_rising AND hard_flux < 1.5 * hard_quiet_baseline. 

Page 6 

Confidential — Internal Use Only 

**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

## **3.3.2  Derived Ratios** 

- Compute thermal_fraction: soft_flux / (soft_flux + hard_flux), smoothed over 30s. 

- Compute nonthermal_index: rate of change of hard_flux relative to soft_flux — (dHard/dt) / (dSoft/dt + epsilon). 

## **3.3.3  Historical Context** 

- Include time-since-last-flare (seconds) as a feature. Solar active regions often produce repeated events. 

- Include current solar cycle phase proxy: day-of-cycle normalized to [0, 1] using estimated solar maximum date. 

Page 7 

Confidential — Internal Use Only 

**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

## **Phase 4 — ML Model Architecture** 

Implement models in increasing complexity order. Each model must be evaluated against the same metric suite before the next is implemented. Do not skip the baseline. 

## **3.4.1  Baseline Models (Required First)** 

- Logistic Regression on L1+L2 features — establishes statistical lower bound. 

- Random Forest on all feature layers — strong classical baseline, interpretable feature importance. 

- LSTM (2-layer, hidden size 128) on raw windowed signals — neural baseline. 

## **3.4.2  Primary Architecture: Dual-Stream Temporal Transformer** 

This is the target architecture once baselines are established. The key design principle is to process soft and hard X-ray streams through independent encoders before fusion, allowing each encoder to specialize. 

|**Component**|**Specification**|
|---|---|
|Soft Encoder|1D Temporal CNN (kernel sizes 3, 7, 15) → Positional encoding →<br>Transformer encoder (4 heads, 2 layers, d_model=128)|
|Hard Encoder|Identical architecture to Soft Encoder. Weights NOT shared.|
|Cross-Attention Layer|Q from Hard encoder, K and V from Soft encoder. Attention weights stored<br>for explainability.|
|Feature Fusion|Concatenate cross-attended embeddings + engineered feature vector.<br>Linear projection to 256-dim.|
|Output Head|3 parallel heads: (1) Flare probability sigmoid, (2) Lead time regression<br>(log-scale), (3) Severity class softmax [B/C/M/X].|



## **3.4.3  Loss Function** 

- Flare probability head: Focal Loss (gamma=2, alpha=0.75) to handle class imbalance. Do not use BCE. 

- Lead time head: Huber loss on log-transformed minutes. 

- Severity head: Weighted cross-entropy with class weights inversely proportional to class frequency. 

- Total loss: L_total = 0.5 * L_focal + 0.3 * L_severity + 0.2 * L_leadtime. Tune weights via heldout validation. 

Page 8 

Confidential — Internal Use Only 

**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

## **3.4.4  Novel Extension — Solar State Machine** 

After the primary architecture is implemented, implement the hidden-state model as a second forecasting path. Define 6 latent states: Quiet (S0), Energy Accumulation (S1), Precursor (S2), Initiation (S3), Peak (S4), Decay (S5). 

- Train a discrete-state HMM or VQ-VAE to infer state labels in an unsupervised manner from the feature matrix. 

- Validate inferred states against known flare events: S3/S4 states must co-occur with GOES catalogue events at >70% rate. 

- Use inferred state sequence as an additional input to the forecasting head: state embedding (6dim one-hot) concatenated into fusion layer. 

- Expose current state and transition probabilities as a separate output for the dashboard. 

## **Phase 5 — Forecasting Output Specification** 

The model must not output a binary prediction. Every inference call must return the following structured payload: 

|**Field**|**Type**|**Description**|
|---|---|---|
|flare_probability|float [0,1]|Probability of any flare (>=B class) in next 30<br>minutes.|
|severity_probs|dict {B,C,M,X}|Per-class probabilities, must sum to <=1.|
|expected_lead_time_min|float|Expected minutes until flare peak. Null if<br>flare_probability < 0.3.|
|lead_time_ci_90|[float, float]|90% confidence interval on lead time estimate.|
|solar_state|str|Current inferred state label (e.g., 'Precursor').|
|state_transition_probs|dict|P(next_state | current_state) for all 6 states.|
|dominant_feature|str|Human-readable name of top SHAP feature.|
|explanation_text|str|Auto-generated natural language reason string.|
|model_uncertainty|float|Epistemic uncertainty estimate (MC Dropout std).|
|inference_timestamp_utc|datetime|UTC timestamp of when inference was run.|



Page 9 

Confidential — Internal Use Only 

**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

## **Phase 6 — Explainability** 

Explainability is a hard requirement, not a nice-to-have. Scientists will not operationally trust a system that cannot explain its predictions. All three mechanisms must be implemented. 

## **3.6.1  SHAP for Feature Attribution** 

- Use shap.DeepExplainer or shap.GradientExplainer for the Transformer model. 

- Compute SHAP values for every inference call. Cache for dashboard rendering. 

- Surface top-3 contributing features with direction (increasing/decreasing) and magnitude. 

## **3.6.2  Attention Visualization** 

- Store cross-attention weights from the fusion layer on every forward pass. 

- Dashboard must render attention as a heatmap overlaid on the 20-minute input window. 

- Highlight time steps where attention weight > 0.1 as 'driver windows'. 

## **3.6.3  Auto-Generated Explanation Text** 

- Use a template engine (not another LLM call) to produce explanation_text from structured SHAP + attention outputs. 

- Example output: 'Prediction driven by rapid soft X-ray rise over the past 6 minutes (SHAP rank 1) combined with emerging hard X-ray acceleration (attention peak at T-3 min). Cross-channel lag narrowed from 90s to 12s in the last window.' 

- All template variables must be traced to specific computed features — no hallucinated reasons. 

## **Phase 7 — Dashboard & Alerting** 

The dashboard is the primary operator interface. It must be usable by non-ML scientists who understand solar physics but not model internals. 

## **3.7.1  Real-Time View** 

- Live time-series plot: dual-axis showing soft and hard X-ray flux for the past 60 minutes. Autoscrolling. 

- Probability gauge: circular gauge showing current flare_probability with color bands (green <30%, amber 30–70%, red >70%). 

- State indicator: current solar_state with animated transition arrow showing most probable next state. 

Page 10 

Confidential — Internal Use Only 

**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

- Lead time countdown: displayed when flare_probability > 0.3. Shows expected_lead_time_min with CI bands. 

## **3.7.2  Explainability Panel** 

- Horizontal SHAP bar chart: top-5 features with positive/negative contribution colors. 

- Attention heatmap: 20-minute window x dual channel. Rendered as color gradient overlay on the time-series plot. 

- Explanation text block: displays auto-generated explanation_text in a readable card. 

## **3.7.3  Alert System** 

- Push alert when flare_probability crosses 0.5 threshold (rising edge only — no repeat alerts within 15 minutes). 

- Alert payload must include: severity_probs, expected_lead_time_min, explanation_text, and a deeplink to the dashboard at the triggering timestamp. 

- Alert channels: email (SMTP), webhook (JSON POST), and browser notification (Web Push API). 

- Alert suppression: configurable quiet window. Default 15 minutes after any alert. 

## **3.7.4  Historical View** 

- Timeline of past 30 days with flare events overlaid (sourced from GOES catalogue). 

- Hindcast overlay: model's historical probability curve vs actual flare onset. Used to visually evaluate calibration. 

- Exportable CSV for any selected date range: all inference payloads at 10-second cadence. 

Page 11 

Confidential — Internal Use Only 

**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

## **4. Evaluation Framework** 

All models must be evaluated on a held-out test set covering at least one 6-month period not seen during training. Use temporal split — never random split (which leaks future data into training). 

|**Metric**|**Target**|**Reasoning**|**Failure Mode to Avoid**|
|---|---|---|---|
|True Skill Statistic<br>(TSS)|>0.6|Accounts for imbalance.<br>Standard in solar<br>forecasting literature.|Optimizing accuracy<br>instead — TSS is ~0 for<br>always-quiet<br>predictions.|
|Probability Calibration<br>(Brier Score)|<0.08|Ensures probability<br>outputs are physically<br>meaningful.|Overconfident<br>predictions that degrade<br>operator trust.|
|Lead Time Error (MAE)|<5 min for M+X|Operational value<br>depends on useful lead<br>time.|Predicting flares after<br>onset has no<br>operational value.|
|False Alarm Rate|<25%|High FAR causes<br>operator alarm fatigue.|Maximizing recall while<br>ignoring precision.|
|Detection Rate (M+X<br>flares)|>80%|Missing major flares is<br>the primary operational<br>risk.|Tuning threshold only<br>on C-class events.|



## **Evaluation Rule** 

Never report accuracy as a primary metric. The baseline accuracy of predicting 'no flare' at all times exceeds 95% due to class imbalance. Use TSS and Brier Score as primary metrics in all experiment logs and model comparison reports. 

## **5. Continuous Learning Pipeline** 

The 11-year solar cycle means a model trained today may drift significantly within 2–3 years. The continuous learning pipeline is required for long-term operational viability. 

## **Retraining Trigger Conditions** 

- Scheduled: Monthly evaluation job compares rolling 30-day TSS against deployment TSS baseline. 

- Alert-triggered: If TSS drops more than 0.1 below baseline on any 7-day window, trigger immediate retraining with latest data. 

- Data volume trigger: When new labeled flare events accumulate to >500 new samples, queue retraining. 

Page 12 

Confidential — Internal Use Only 

**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

## **Deployment Gate** 

- New model must outperform current production model on the held-out evaluation set on all 5 metrics before promotion. 

- Shadow deployment: run new model in parallel for 72 hours, comparing outputs without serving them to the dashboard. 

- Version all models with timestamp and data window. Never overwrite production weights. 

Page 13 

Confidential — Internal Use Only 

**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

## **6. Development Workstreams** 

Four parallel workstreams should be staffed independently. Each has a defined interface contract so they can develop concurrently without blocking each other. 

|**Workstream**|**Deliverable**|**Interface Output**|**Depends On**|
|---|---|---|---|
|WS-1: Data &<br>Signal|Cleaning pipeline,<br>windowing, feature L1–<br>L2|Parquet files + numpy<br>arrays with schema doc|Raw telemetry access|
|WS-2: Physics &<br>Features|Feature L3–L5 +<br>physics layer|Feature matrix appended to<br>WS-1 output|WS-1 output|
|WS-3: AI Models|Baseline + Transformer<br>+ state machine|Trained model weights +<br>inference API (FastAPI)|WS-2 feature matrix|
|WS-4:<br>Dashboard|React frontend +<br>WebSocket backend +<br>alert system|Deployed web application|WS-3 inference API<br>contract|



## **7. Recommended Technology Stack** 

|**Layer**|**Technology**|**Rationale**|
|---|---|---|
|Data storage|Parquet + HDF5|Efficient columnar storage for time-series.<br>HDF5 for large arrays.|
|Signal processing|scipy, PyWavelets, ruptures|Well-maintained. Wavelet and change-<br>point support.|
|Feature engineering|tsfresh, pandas|tsfresh automates time-series feature<br>generation for L2 features.|
|ML framework|PyTorch + PyTorch Lightning|Lightning reduces boilerplate. Native<br>Transformer support.|
|Explainability|SHAP, Captum|Captum for Transformer attribution.<br>SHAP for classical models.|
|Inference API|FastAPI + Uvicorn|Async WebSocket support. Low latency.|
|Frontend|React + Recharts + shadcn/ui|Recharts for time-series plots. shadcn for<br>dashboard components.|
|Alerts|SMTP + generic webhook|Avoid vendor lock-in. Teams/Slack<br>webhooks via generic JSON POST.|
|Experiment tracking|MLflow or Weights & Biases|Required for model versioning and metric<br>history.|
|Continuous learning|Prefect or Airflow DAG|Scheduled retraining and evaluation jobs.|



Page 14 

Confidential — Internal Use Only 

**PRODUCT REQUIREMENTS DOCUMENT** 

Aditya-L1 Solar Flare Forecasting System 

## **8. Out of Scope (v1.0)** 

- Magnetogram or EUV image data integration — optical data requires a separate CV pipeline. 

- Multi-spacecraft data fusion (SDO, STEREO) — complicates alignment without proportionate gain in v1. 

- Flare location prediction on the solar disk — requires spatial modeling beyond 1D time series. 

- Mobile native application — web-responsive dashboard is sufficient for v1. 

- User authentication and multi-tenant access — single-operator deployment assumed for v1. 

## **9. Open Questions for Team Resolution** 

|**#**|**Question**|**Owner / Deadline**|
|---|---|---|
|1|What is the exact native sampling cadence of<br>SoLEXS and HEL1OS? This determines the<br>resampling target frequency.|WS-1 / Sprint 1|
|2|Are quality flags already in the telemetry<br>headers, or must they be inferred from variance<br>thresholds?|WS-1 / Sprint 1|
|3|What is the available labeled flare catalogue for<br>Aditya-L1 data? Is GOES cross-referencing<br>sufficient?|WS-2 / Sprint 1|
|4|Is 30-minute forecast horizon operationally<br>correct, or does the use case require 60-minute<br>or 2-hour windows?|All / Sprint 0|
|5|What is the minimum acceptable inference<br>latency? Does the model need to run faster<br>than real time?|WS-3 & WS-4 / Sprint 0|



Page 15 

Confidential — Internal Use Only 

