@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (set "PY=python") else (set "PY=py -3")
%PY% -m venv .venv
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo.
echo Install complete. Double-click run.bat to start.
echo Put adb.exe and scrcpy.exe in tools\ or add them to PATH.
pause
exit /b 0
:error
echo Install failed. Please install Python 3.10+ and enable Add Python to PATH.
pause
exit /b 1
