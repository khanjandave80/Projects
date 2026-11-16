@echo off
setlocal
title CyberSleuth SOC Platform Backend
cd /d "%~dp0"

if "%PORT%"=="" set "PORT=5000"
set "HOST=0.0.0.0"
set "DEPLOYMENT_MODE=offline"

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

powershell -NoProfile -Command "if (Get-NetTCPConnection -State Listen -LocalPort %PORT% -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }" >nul 2>&1
if not errorlevel 1 goto port_available
echo Port %PORT% already has a listener. Stop that service or set PORT to a free port, then run this script again.
pause
exit /b 1

:port_available
echo Using Python: %PYTHON_CMD%
if defined CUDA_PYTHON (
  echo CUDA-enabled PyTorch environment detected.
) else (
  echo WARNING: No CUDA-enabled PyTorch environment was found; using the first compatible Python.
)
echo Starting CyberSleuth on http://localhost:%PORT% ...
"%PYTHON_CMD%" app.py
set "EXIT_CODE=%errorlevel%"
if not "%EXIT_CODE%"=="0" echo Backend exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%

:probe_python
set "CANDIDATE=%~1"
"%CANDIDATE%" -c "import flask, numpy, PIL, torch, transformers" >nul 2>&1
if errorlevel 1 exit /b 0
if not defined PYTHON_FALLBACK set "PYTHON_FALLBACK=%CANDIDATE%"
"%CANDIDATE%" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 if not defined CUDA_PYTHON set "CUDA_PYTHON=%CANDIDATE%"
exit /b 0
