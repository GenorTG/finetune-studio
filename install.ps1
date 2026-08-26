# Finetune Studio Installer (Windows PowerShell)
$ErrorActionPreference = "Stop"
Write-Host "=== Finetune Studio Installer ===" -ForegroundColor Cyan

# Detect Python (prefer 3.13, accept 3.12)
$pythonCmd = $null
foreach ($cmd in @("python3.13", "python3.12", "python")) {
    try {
        $ver = & $cmd --version 2>&1 | Select-String -Pattern "\d+\.\d+" | ForEach-Object { $_.Matches.Value }
        $parts = $ver -split "\."
        if ([int]$parts[0] -ge 3 -and [int]$parts[1] -ge 12) {
            $pythonCmd = $cmd
            Write-Host "Found: $cmd ($ver)"
            break
        }
    } catch { continue }
}

if (-not $pythonCmd) {
    Write-Host "ERROR: Python 3.12+ not found." -ForegroundColor Red
    Write-Host "Install from https://www.python.org/downloads/"
    exit 1
}

# Install UV if missing
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing UV..."
    irm https://astral.sh/uv/install.ps1 | iex
}
Write-Host "UV: $(uv --version)"

# Create venv
Write-Host "Creating venv with $pythonCmd..."
uv venv .venv --python $pythonCmd
& .\.venv\Scripts\Activate.ps1

# Install packages
Write-Host "Installing packages..."
uv pip install -e "."

# Generate lock file if missing
if (-not (Test-Path uv.lock)) {
    Write-Host "Generating lock file..."
    uv lock
}

if (-not (Test-Path data)) { New-Item -ItemType Directory -Name data | Out-Null }

Write-Host ""
Write-Host "=== Install complete! ===" -ForegroundColor Green
Write-Host "Run: .\.venv\Scripts\activate.ps1; python -m finetune_studio"
