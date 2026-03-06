@echo off
cd /d "%~dp0"
if not exist ".\.venv\Scripts\python.exe" (
  echo Python runtime not found in .venv
  echo Create env and install deps:
  echo   python -m venv .venv
  echo   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

set "APP_URL=http://127.0.0.1:8501"
powershell -NoProfile -Command ^
  "try { Invoke-WebRequest -Uri '%APP_URL%' -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
  echo APRIS server is already running on %APP_URL%
  exit /b 0
)

".\.venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
