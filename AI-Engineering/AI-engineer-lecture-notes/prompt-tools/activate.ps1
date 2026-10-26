# Activate the sprint-level Python venv (PowerShell).
# Usage:  . .\activate.ps1
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$here\..\..\..\.venv\Scripts\Activate.ps1"
Write-Host "sprint venv active -> $here\..\..\..\.venv"
