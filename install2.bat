@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (set "PY=python") else (set "PY=py -3")
%PY% -m venv .venv2
if errorlevel 1 goto :error
".venv2\Scripts\python.exe" -m pip install --upgrade pip
".venv2\Scripts\python.exe" -m pip install -r requirements-ui2.txt
if errorlevel 1 goto :error
echo.
echo 1.0 workspace installed. Double-click run2.bat to start.
pause
exit /b 0
:error
echo Install failed. Check Python 3.10+ and network.
pause
exit /b 1
