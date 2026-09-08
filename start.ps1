# start.ps1 - Start the Clinic Doctor Time Tracker + ngrok tunnel with LIVE
# output in the SAME console, and make sure stopping this script (Ctrl+C)
# also stops the tracker and the tunnel - no orphaned processes.
# Run:  powershell -ExecutionPolicy Bypass -File start.ps1

$ErrorActionPreference = "SilentlyContinue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$tracker = $null
$ngrok = $null

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
try {
    Get-Process ngrok -ErrorAction SilentlyContinue | ForEach-Object { Stop-Tree $_.Id; Write-Host "  killing old ngrok (PID $($_.Id))" }
} catch { }
Start-Sleep -Milliseconds 700

# ---------- 1. Install requirements if missing --------------
Write-Host "Checking dependencies..."
$depsOk = & python -c "import ultralytics, cv2, flask, numpy; print('ok')" 2>$null
if ($LASTEXITCODE -ne 0 -or "$depsOk".Trim() -ne 'ok') {
    Write-Host "Installing dependencies..."
    & python -m pip install -r requirements.txt
}

# ---------- 2. Start tracker + ngrok in THIS console (live output) ----------
Write-Host "Starting tracker (its log prints below, live)..."
$tracker = Start-Process -FilePath "python" -ArgumentList "tracker.py" `
    -WorkingDirectory $Root -NoNewWindow -PassThru

Write-Host "Starting ngrok tunnel..."
$ngrok = Start-Process -FilePath "ngrok" -ArgumentList @("http", "5000") `
    -WorkingDirectory $Root -NoNewWindow -PassThru

# ---------- 3. Wait for the Flask API ----------
$up = $false
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Seconds 1
    if ($tracker.HasExited) { Write-Host "[tracker process ended early]"; break }
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:5000/api/cameras" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $up = $true; break }
    } catch { }
}
if (-not $up) {
    Write-Host "ERROR: tracker did not become ready within 40s."
    Write-Host "       (look at the python traceback printed above; a port conflict was pre-cleaned.)"
}

# ---------- 4. Wait for the public HTTPS tunnel ----------
$tunnelUrl = $null
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $tunnels = (Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 3).tunnels
        foreach ($t in $tunnels) {
            if ($t.proto -eq "https") { $tunnelUrl = $t.public_url }
        }
    } catch { }
    if ($tunnelUrl) { break }
}

if ($tunnelUrl) {
    Set-Content -Path "$Root\public_url.txt" -Value $tunnelUrl -Encoding ascii
    Write-Host ""
    Write-Host "========================================================"
    Write-Host "  Clinic Doctor Time Tracker is RUNNING - LIVE OUTPUT:"
    Write-Host "  Local:   http://localhost:5000"
    Write-Host "  Public:  $tunnelUrl"
    Write-Host "  Dashboard:      $tunnelUrl"
    Write-Host "  Enroll doctors: $tunnelUrl/enroll"
    Write-Host ""
    Write-Host "  Tracker `& ngrok logs appear below in real time."
    Write-Host "  Press Ctrl+C here to STOP everything."
    Write-Host "========================================================"
} else {
    Set-Content -Path "$Root\public_url.txt" -Value "" -Encoding ascii
    Write-Host "WARNING: no public ngrok URL yet. Use local: http://localhost:5000"
}

# ---------- 5. Stay alive streaming output until stopped ----------
try {
    while ($true) {
        Start-Sleep -Milliseconds 500
        if ($tracker.HasExited)    { Write-Host "[tracker process ended]"; break }
        if ($ngrok.HasExited)      { Write-Host "[ngrok process ended]"; break }
    }
} finally {
    Write-Host ""
    Write-Host "Stopping tracker and tunnel..."
    if ($tracker) { Stop-Tree $tracker.Id }
    if ($ngrok)   { Stop-Tree $ngrok.Id }
    Start-Sleep -Milliseconds 400
    Write-Host "Stopped. No background processes left."
}