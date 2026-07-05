# Run this in Windows PowerShell (Admin optional) before using WSL proxy.
# Purpose: remind how to enable Clash Allow LAN and open firewall for WSL.

Write-Host "Clash for Windows checklist for WSL:" -ForegroundColor Cyan
Write-Host "1. Open Clash for Windows"
Write-Host "2. Turn ON 'Allow LAN'"
Write-Host "3. Turn ON 'System Proxy' (or use TUN mode)"
Write-Host "4. Confirm ports: HTTP 7890, SOCKS 7891 (check General -> Port)"
Write-Host ""
Write-Host "Optional: allow inbound on Clash HTTP port for WSL subnet"

$ruleName = "Clash HTTP for WSL"
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if (-not $existing) {
  New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort 7890 | Out-Null
  Write-Host "Created firewall rule: $ruleName (TCP 7890 inbound)" -ForegroundColor Green
} else {
  Write-Host "Firewall rule already exists: $ruleName" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Then in WSL run:"
Write-Host "  source ~/projects/motion-score-benchmark/scripts/setup_wsl_proxy.sh --persist --test"
