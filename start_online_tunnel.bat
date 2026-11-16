@echo off
setlocal
title CyberSleuth Online Tunnel
cd /d "%~dp0"
if "%PORT%"=="" set "PORT=5000"
call :check_port
if errorlevel 1 (
  echo Port %PORT% already has a listener. Stop that service or set PORT to a free port, then run this script again.
  exit /b 1
)
where cloudflared >nul 2>&1
if errorlevel 1 (
  echo cloudflared was not found on PATH. Install Cloudflare Tunnel and run this script again.
  exit /b 1
)

echo Building the frontend...
pushd game
call npm run build
if errorlevel 1 (
  echo Frontend build failed.
  popd
  exit /b 1
)
popd

set "PYTHON_CMD="
set "PYTHON_FALLBACK="
set "CUDA_PYTHON="

if defined CONDA_PREFIX call :probe_python "%CONDA_PREFIX%\python.exe"
for %%P in ("%~dp0.venv\Scripts\python.exe" "%~dp0venv\Scripts\python.exe" "%~dp0env\Scripts\python.exe") do if exist "%%~P" call :probe_python "%%~P"
for /f "delims=" %%P in ('where python 2^>nul') do call :probe_python "%%P"
where py >nul 2>&1
if not errorlevel 1 call :probe_python py

if defined CUDA_PYTHON set "PYTHON_CMD=%CUDA_PYTHON%"
if not defined PYTHON_CMD set "PYTHON_CMD=%PYTHON_FALLBACK%"
if not defined PYTHON_CMD (
  echo No Python environment with Flask, PyTorch, and Transformers was found.
  echo Activate your project or CUDA virtual environment, then run this script again.
  pause
  exit /b 1
)

if not defined CS_ADMIN_1_PASSWORD set "GENERATE_ADMIN_1=1"
if not defined CS_ADMIN_2_PASSWORD set "GENERATE_ADMIN_2=1"
if defined GENERATE_ADMIN_1 for /f "delims=" %%P in ('powershell -NoProfile -Command "$bytes = New-Object byte[] 24; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes); [Convert]::ToBase64String($bytes)"') do set "CS_ADMIN_1_PASSWORD=%%P"
if defined GENERATE_ADMIN_2 for /f "delims=" %%P in ('powershell -NoProfile -Command "$bytes = New-Object byte[] 24; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes); [Convert]::ToBase64String($bytes)"') do set "CS_ADMIN_2_PASSWORD=%%P"
if not defined CS_ADMIN_1_PASSWORD (
  echo Failed to create CS_ADMIN_1_PASSWORD. Check PowerShell availability.
  exit /b 1
)
if not defined CS_ADMIN_2_PASSWORD (
  echo Failed to create CS_ADMIN_2_PASSWORD. Check PowerShell availability.
  exit /b 1
)

set "HOST=0.0.0.0"
set "DEPLOYMENT_MODE=online"
echo Using Python: %PYTHON_CMD%
if defined CUDA_PYTHON (
  echo CUDA-enabled PyTorch environment detected.
) else (
  echo WARNING: No CUDA-enabled PyTorch environment was found; using the first compatible Python.
)
if defined GENERATE_ADMIN_1 echo Temporary Admin_1 password: %CS_ADMIN_1_PASSWORD%
if defined GENERATE_ADMIN_2 echo Temporary Admin_2 password: %CS_ADMIN_2_PASSWORD%
echo Keep these credentials private. They are regenerated unless supplied via environment variables.
echo Starting CyberSleuth on http://localhost:%PORT% ...
start "CyberSleuth Backend" cmd /k ""%PYTHON_CMD%" app.py"

call :wait_for_server
if errorlevel 1 (
  echo Backend did not become healthy with the required models at http://127.0.0.1:%PORT%/health.
  echo Check that email_distilbert, url_bilstm, and qr_bilstm are true in /health, then review the Backend window.
  exit /b 1
)

start "CyberSleuth Local" "http://localhost:%PORT%"
echo Health check passed. Starting one Cloudflare Quick Tunnel.
echo Keep this window and the backend window open. The public trycloudflare.com URL will appear below.
cloudflared tunnel --url http://127.0.0.1:%PORT%
exit /b %errorlevel%

:probe_python
set "CANDIDATE=%~1"
"%CANDIDATE%" -c "import flask, numpy, PIL, torch, transformers" >nul 2>&1
if errorlevel 1 exit /b 0
if not defined PYTHON_FALLBACK set "PYTHON_FALLBACK=%CANDIDATE%"
"%CANDIDATE%" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 if not defined CUDA_PYTHON set "CUDA_PYTHON=%CANDIDATE%"
exit /b 0

:wait_for_server
for /L %%I in (1,1,45) do (
  powershell -NoProfile -Command "try { $response = Invoke-RestMethod -Uri 'http://127.0.0.1:%PORT%/health' -TimeoutSec 2; if ($response.status -eq 'ok' -and $response.active_models.email_distilbert -and $response.active_models.url_bilstm -and $response.active_models.qr_bilstm) { exit 0 } } catch {}; exit 1" >nul 2>&1
  if not errorlevel 1 exit /b 0
  timeout /t 2 /nobreak >nul
)
exit /b 1

:check_port
powershell -NoProfile -Command "if (Get-NetTCPConnection -State Listen -LocalPort %PORT% -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }" >nul 2>&1
exit /b %errorlevel%
