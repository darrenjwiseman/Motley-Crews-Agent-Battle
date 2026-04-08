@echo off
setlocal
cd /d "%~dp0.."

set "VENV=%cd%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"

if not exist "%PY%" (
  echo Creating virtual environment in .venv ...
  where py >nul 2>nul && py -3 -m venv .venv || python -m venv .venv
)

if "%SKIP_PIP_INSTALL%"=="1" goto launch

echo Checking dependencies ^(requirements-play.txt^) ...
"%PIP%" install --disable-pip-version-check -r "%cd%\requirements-play.txt"
if errorlevel 1 exit /b 1

:launch
"%PY%" -m motley_crews_play --ui %*
if errorlevel 1 pause
