# Finetune Studio Installer (Windows PowerShell)
# GPU-aware like install.sh: scripts\accel_plan.py detects the accelerator (NVIDIA CUDA,
# Intel XPU; CPU wheels only when no GPU exists) and installs the newest matching stack.
# Usage:  .\install.ps1 [-Cpu] [-Gpu nvidia|amd|intel|none] [-NoGguf] [-NoLlamaCpp]
param(
    [switch]$Cpu,
    [ValidateSet("", "nvidia", "amd", "intel", "none")][string]$Gpu = "",
    [switch]$NoGguf,
    [switch]$NoLlamaCpp
)
$ErrorActionPreference = "Stop"
Write-Host "=== Finetune Studio Installer ===" -ForegroundColor Cyan

# Detect Python (prefer 3.13, accept 3.12)
$pythonCmd = $null
foreach ($cmd in @("python3.13", "python3.12", "python")) {
    try {
        $ver = & $cmd --version 2>&1 | Select-String -Pattern "\d+\.\d+" | ForEach-Object { $_.Matches.Value }
        $version = [version]$ver
        if ($version -ge [version]"3.12" -and $version -lt [version]"3.14") {
            $pythonCmd = $cmd
            Write-Host "Found: $cmd ($ver)"
            break
        }
    } catch { continue }
}

if (-not $pythonCmd) {
    Write-Host "ERROR: Python 3.12 or 3.13 not found." -ForegroundColor Red
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
$venvPy = ".\.venv\Scripts\python.exe"

# Accelerator flags shared by every accel_plan.py call
$accel = @()
if ($Cpu) { $accel += "--cpu" }
if ($Gpu) { $accel += @("--gpu", $Gpu) }
function Invoke-Accel {
    param([string]$Sub, [string[]]$More = @())
    & $venvPy scripts\accel_plan.py $Sub @accel --venv .venv @More
    return $LASTEXITCODE
}

Write-Host "Detected accelerator plan:" -ForegroundColor Cyan
Invoke-Accel plan | Out-Host

# 1. PyTorch: newest CUDA/XPU index the driver + GPU support; CPU only when no GPU of any vendor.
#    Writes .venv\torch-constraints.txt so later resolves cannot swap torch off its GPU build.
if ((Invoke-Accel install @("torch")) -ne 0) {
    Write-Host "ERROR: PyTorch install failed (see above). Fix the driver, or re-run with -Cpu." -ForegroundColor Red
    exit 1
}
$constraints = @()
if (Test-Path .venv\torch-constraints.txt) { $constraints = @("-c", ".venv\torch-constraints.txt") }

# 2. llama-cpp-python with GPU offload (prebuilt wheel, else source build with the right GGML_* flags)
if (-not $NoGguf) {
    if ((Invoke-Accel install @("llama-cpp-python")) -ne 0) {
        Write-Host "WARN: llama-cpp-python install failed - GGUF inference unavailable." -ForegroundColor Yellow
    }
}

# 3. App packages under the torch pin
Write-Host "Installing packages..."
uv pip install @constraints -e ".[parsers]"

# 4. Optional extras: bitsandbytes (CUDA/XPU backends), unsloth when a compatible release exists
uv pip install @constraints "numpy>=1.24.0" "scipy>=1.10.0"
Invoke-Accel install @("bitsandbytes") | Out-Null
Invoke-Accel install @("unsloth") | Out-Null

# 5. llama.cpp CLI (convert_hf_to_gguf.py + llama-quantize + llama-cli) with the same backend.
#    Needs git + cmake + Visual Studio Build Tools (and the CUDA toolkit for the CUDA backend).
if (-not $NoLlamaCpp) {
    if ((Invoke-Accel build-llama-cli) -ne 0) {
        Write-Host "WARN: llama.cpp CLI not built (needs git, cmake, VS Build Tools); GGUF export unavailable." -ForegroundColor Yellow
    }
}

# Generate lock file if missing
if (-not (Test-Path uv.lock)) {
    Write-Host "Generating lock file..."
    uv lock
}

if (-not (Test-Path data)) { New-Item -ItemType Directory -Name data | Out-Null }

Write-Host ""
Write-Host "=== Install complete! ===" -ForegroundColor Green
Write-Host "Run the WebUI with: .\.venv\Scripts\python.exe -m finetune_studio webui"
