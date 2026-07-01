param(
    [string]$ConfigPath = "config/config.yaml"
)

$rootDir = $PWD.Path

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  FlareClassifier - Starting System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not (Test-Path "models/random_forest.pkl")) {
    Write-Host "Models not found. Running pipeline..." -ForegroundColor Yellow
    python scripts/run_pipeline.py --config $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pipeline failed. Exiting." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Starting API server on port 8000..." -ForegroundColor Green
$apiJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location $dir
    python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --log-level info
} -ArgumentList $rootDir
Start-Sleep 4

$apiRunning = $null
$apiOutput = Receive-Job $apiJob -Keep 2>&1
if ($apiOutput -match "Application startup complete") {
    $apiRunning = $true
    Write-Host "  API server is running." -ForegroundColor Green
} else {
    Write-Host "  API output: $apiOutput" -ForegroundColor Yellow
}

Write-Host "Starting dashboard on port 5173..." -ForegroundColor Green
$dashJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location "$dir/src/dashboard"
    npm run dev
} -ArgumentList $rootDir
Start-Sleep 4

$dashOutput = Receive-Job $dashJob -Keep 2>&1
if ($dashOutput -match "Local:") {
    Write-Host "  Dashboard is running." -ForegroundColor Green
} else {
    Write-Host "  Dashboard output: $dashOutput" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  FlareClassifier is running!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  API:        http://localhost:8000" -ForegroundColor Yellow
Write-Host "  API Docs:   http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "  Dashboard:  http://localhost:5173" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Health:     http://localhost:8000/health" -ForegroundColor Gray
Write-Host "  WebSocket:  ws://localhost:8000/ws/stream" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Cyan

try {
    while ($true) { Start-Sleep 1 }
} finally {
    Write-Host "`nShutting down..." -ForegroundColor Yellow
    Write-Host "  Stopping API..." -ForegroundColor Gray
    Stop-Job $apiJob -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $apiJob -ErrorAction SilentlyContinue | Out-Null
    Write-Host "  Stopping Dashboard..." -ForegroundColor Gray
    Stop-Job $dashJob -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $dashJob -ErrorAction SilentlyContinue | Out-Null
    Write-Host "Stopped." -ForegroundColor Green
}
