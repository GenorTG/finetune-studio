@echo off
cd /d "%~dp0"
if not exist .venv (
    echo No venv found. Running install...
    call install.bat
)
call .venv\Scripts\activate.bat
echo Starting Finetune Studio on http://localhost:7860
python -m finetune_studio %*
