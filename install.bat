@echo off
echo === Finetune Studio Installer ===

REM Detect Python (prefer 3.13, accept 3.12)
set PYTHON_CMD=
for %%V in (python3.13 python3.12 python3 python) do (
    where %%V >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=2 delims= " %%A in ('%%V --version 2^>^&1') do (
            set VER=%%A
            set MAJOR=!VER:~0,1!
            set MINOR=!VER:~2,2!
        )
        set PYTHON_CMD=%%V
        goto :found_python
    )
)

echo ERROR: Python 3.12+ not found.
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

REM Install
if exist uv.lock (
    echo Installing from lock file...
    uv sync
) else (
    echo Installing from pyproject.toml...
    uv sync
    uv lock
)

if not exist data mkdir data
echo.
echo === Install complete! ===
echo Run: run.bat
pause
