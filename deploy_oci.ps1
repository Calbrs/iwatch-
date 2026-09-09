# deploy_oci.ps1 - Push to GitHub, then pull + restart the iwatch-tracker
# service on the Oracle box, so the OCI code always matches this repo.
# Requires: ~/Downloads/modles ai.key (the OCI SSH key) and the box up.
#
# Run:  powershell -ExecutionPolicy Bypass -File deploy_oci.ps1
$ErrorActionPreference = "Stop"

$repo = "D:\Calbrs Projects\I Watch\clinic_tracker"
$key  = "C:\Users\7014\Downloads\modles ai.key"
$host = "ubuntu@84.12.95.253"

Set-Location -LiteralPath $repo

Write-Host "== 1/4 push to GitHub =="
git push origin master

Write-Host "== 2/4 pull + restart on the box =="
ssh -i $key -o BatchMode=yes -o StrictHostKeyChecking=no $host "cd ~/iwatch && git pull --ff-only && sudo systemctl restart iwatch-tracker"

Write-Host "== 3/4 wait for service =="
Start-Sleep -Seconds 5

Write-Host "== 4/4 verify deployed commit =="
ssh -i $key -o BatchMode=yes -o StrictHostKeyChecking=no $host "cd ~/iwatch && git log --oneline -1 && sudo systemctl is-active iwatch-tracker"
Write-Host "DONE"