@echo off
echo === Finetune Studio Installer ===

REM Detect Python (prefer 3.13, accept 3.12)
set PYTHON_CMD=
for %%V in (python3.13 python3.12 python3 python) do (
    where %%V >nul 2>&1
    if not errorlevel 1 (
        %%V -c "import sys; raise SystemExit(0 if (3,12) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>&1
        if not errorlevel 1 (
            set PYTHON_CMD=%%V
            goto :found_python
        )
    )
)

echo ERROR: Python 3.12 or 3.13 not found.
echo Install from https://www.python.org/downloads/
pause
exit /b 1

:found_python
echo Found: %PYTHON_CMD%

REM Install UV if missing
where uv >nul 2>&1
if errorlevel 1 (
    echo Installing UV...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set PATH=%USERPROFILE%\.local\bin;%PATH%
)
echo UV: & uv --version

REM Create venv
echo Creating venv...
uv venv .venv --python %PYTHON_CMD%
call .venv\Scripts\activate.bat

REM Accelerator flags: set FTS_CPU=1 to force CPU wheels, FTS_GPU=nvidia|amd|intel|none to override detection
set ACCEL_FLAGS=
if "%FTS_CPU%"=="1" set ACCEL_FLAGS=--cpu
if not "%FTS_GPU%"=="" set ACCEL_FLAGS=%ACCEL_FLAGS% --gpu %FTS_GPU%
set VENV_PY=.venv\Scripts\python.exe

echo Detected accelerator plan:
%VENV_PY% scripts\accel_plan.py plan %ACCEL_FLAGS% --venv .venv

REM 1. PyTorch: newest CUDA/XPU index the driver + GPU support; CPU only when no GPU exists.
REM    Writes .venv\torch-constraints.txt so later resolves cannot swap torch off its GPU build.
%VENV_PY% scripts\accel_plan.py install torch %ACCEL_FLAGS% --venv .venv
if errorlevel 1 (
    echo ERROR: PyTorch install failed. Fix the GPU driver or re-run with FTS_CPU=1.
    pause
    exit /b 1
)
set CONSTRAINTS=
if exist .venv\torch-constraints.txt set CONSTRAINTS=-c .venv\torch-constraints.txt

REM 2. llama-cpp-python with GPU offload (prebuilt wheel, else source build)
%VENV_PY% scripts\accel_plan.py install llama-cpp-python %ACCEL_FLAGS% --venv .venv
if errorlevel 1 echo WARN: llama-cpp-python install failed - GGUF inference unavailable.

REM 3. App packages under the torch pin (a lock-file sync would re-resolve torch to the default build)
if exist uv.lock (
    echo Installing from pyproject.toml with the torch pin ^(lock file ignored to keep the GPU build^)...
)
uv pip install %CONSTRAINTS% -e ".[parsers]"

REM 4. Optional extras
uv pip install %CONSTRAINTS% "numpy>=1.24.0" "scipy>=1.10.0"
%VENV_PY% scripts\accel_plan.py install bitsandbytes %ACCEL_FLAGS% --venv .venv
%VENV_PY% scripts\accel_plan.py install unsloth %ACCEL_FLAGS% --venv .venv

REM 5. llama.cpp CLI (needs git + cmake + Visual Studio Build Tools)
%VENV_PY% scripts\accel_plan.py build-llama-cli %ACCEL_FLAGS% --venv .venv
if errorlevel 1 echo WARN: llama.cpp CLI not built ^(needs git, cmake, VS Build Tools^); GGUF export unavailable.

if not exist data mkdir data
echo.
echo === Install complete! ===
echo Run: run.bat
pause
