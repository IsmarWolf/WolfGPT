# WolfGPT - Phase 0, step C (Windows side). Run in PowerShell AFTER rebooting.
# Idempotent: does nothing if Ubuntu is already registered.
$ErrorActionPreference = "Stop"
Write-Host "== WSL status ==" ; wsl --status
$installed = (wsl -l -q 2>$null) -replace "`0","" | Where-Object { $_ -match "Ubuntu" }
if ($installed) {
    Write-Host "Ubuntu already installed: $installed -> nothing to do."
} else {
    Write-Host "Installing Ubuntu (you will be asked for a Linux username/password)..."
    wsl --set-default-version 2
    wsl --install -d Ubuntu
}
Write-Host "== Distros ==" ; wsl -l -v
