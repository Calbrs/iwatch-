# start.ps1 - Start the Clinic Doctor Time Tracker with LIVE output in the
# SAME console, and make sure stopping this script (Ctrl+C) also stops the
# tracker - no orphaned processes.
# Run:  powershell -ExecutionPolicy Bypass -File start.ps1
#
# Note: no ngrok tunnel is used anymore - for remote/device access deploy
# to Render (free) which provides the public HTTPS endpoint. This script runs
# the tracker locally, reachable at http://localhost:5000.

$ErrorActionPreference = "SilentlyContinue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$tracker = $null

function Stop-Tree([int]$ProcessId) {
    if ($ProcessId -le 0) { return }
    try { taskkill /PID $ProcessId /T /F 2>&1 | Out-Null } catch { }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

# ---------- 0. Clean up ANY previous instance --------------
# A leftover server on :5000 (from a killed console, not from start.ps1)
# silently shadows the new one and breaks streaming.
Write-Host "Cleaning up leftover processes..."
try {
    Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique |
        ForEach-Object {
            Write-Host "  killing old process holding :5000 (PID $_)"
            Stop-Tree $_
        }
} catch { }
Start-Sleep -Milliseconds 700

# ---------- 1. Install requirements if missing --------------
Write-Host "Checking dependencies..."
$depsOk = & python -c "import ultralytics, cv2, flask, numpy; print('ok')" 2>$null
if ($LASTEXITCODE -ne 0 -or "$depsOk".Trim() -ne 'ok') {
    Write-Host "Installing dependencies..."
    & python -m pip install -r requirements.txt
}

# ---------- 2. Start tracker in THIS console (live output) ----------
Write-Host "Starting tracker (its log prints below, live)..."
$tracker = Start-Process -FilePath "python" -ArgumentList "tracker.py" `
    -WorkingDirectory $Root -NoNewWindow -PassThru

# ---------- 3. Wait for the Flask API ----------
$up = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    if ($tracker.HasExited) { Write-Host "[tracker process ended early]"; break }
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:5000/api/cameras" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $up = $true; break }
    } catch { }
}
if (-not $up) {
    Write-Host "ERROR: tracker did not become ready within 60s."
    Write-Host "       (look at the python traceback printed above; a port conflict was pre-cleaned.)"
}

Write-Host ""
Write-Host "========================================================"
Write-Host "  Clinic Doctor Time Tracker is RUNNING - LIVE OUTPUT:"
Write-Host "  Local:   http://localhost:5000"
Write-Host "  Dashboard:      http://localhost:5000"
Write-Host "  Enroll doctors: http://localhost:5000/enroll"
Write-Host ""
Write-Host "  Remote/device access (public HTTPS endpoint):"
Write-Host "    - Render:    https://<your-service>.onrender.com"
Write-Host "    - HF Space:  https://<user>-<space>.hf.space"
Write-Host ""
Write-Host "  Tracker logs appear below in real time."
Write-Host "  Press Ctrl+C here to STOP the tracker."
Write-Host "========================================================"

# ---------- 4. Stay alive streaming output until stopped ----------
try {
    while ($true) {
        Start-Sleep -Milliseconds 500
        if ($tracker.HasExited) { Write-Host "[tracker process ended]"; break }
    }
} finally {
    Write-Host ""
    Write-Host "Stopping tracker..."
    if ($tracker) { Stop-Tree $tracker.Id }
    Start-Sleep -Milliseconds 400
    Write-Host "Stopped. No background processes left."
}