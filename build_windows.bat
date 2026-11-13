@echo off
setlocal
cd /d "%~dp0"
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
    where py >nul 2>nul
    if errorlevel 1 (set "PY=python") else (set "PY=py -3")
)
%PY% -m pip install "pyinstaller>=6.0"
%PY% -m PyInstaller --noconfirm --clean --windowed --name scrcpy-flow main.py
if errorlevel 1 goto :error
echo.
echo Build complete: dist\scrcpy-flow\
echo Copy tools\ into dist\scrcpy-flow\tools\ for a portable package.
pause
exit /b 0
:error
echo Build failed.
pause
exit /b 1
