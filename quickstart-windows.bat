@echo off
setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"

echo [1/4] Checking Python 3...
set "PY="

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 -c "import sys; sys.exit(0 if sys.version_info>=(3,8) else 1)" >nul 2>nul
  if %errorlevel%==0 set "PY=py -3"
)

if not defined PY (
  where python >nul 2>nul
  if %errorlevel%==0 (
    python -c "import sys; sys.exit(0 if sys.version_info>=(3,8) else 1)" >nul 2>nul
    if %errorlevel%==0 set "PY=python"
  )
)

if not defined PY (
  echo Python 3.8+ not found. Please install Python from https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist "tetris_rl.py" (
  echo tetris_rl.py not found in this folder.
  pause
  exit /b 1
)

echo [2/4] Setting up venv...
if not exist ".venv\\Scripts\\python.exe" (
  %PY% -m venv ".venv"
  if errorlevel 1 (
    echo Failed to create venv.
    pause
    exit /b 1
  )
)

echo [3/4] Installing dependencies...
".venv\\Scripts\\python.exe" -m pip install pygame
if errorlevel 1 (
  echo Failed to install pygame.
  pause
  exit /b 1
)

echo [4/4] Launching...
".venv\\Scripts\\python.exe" "tetris_rl.py"
pause
