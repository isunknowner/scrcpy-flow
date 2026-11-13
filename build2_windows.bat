@echo off
setlocal
cd /d "%~dp0"
if exist ".venv2\Scripts\python.exe" (set "PY=.venv2\Scripts\python.exe") else (set "PY=py -3")
%PY% -m pip install "pyinstaller>=6.0"
%PY% -m PyInstaller --noconfirm --clean --windowed --name scrcpy-flow-1.0 main.py
if errorlevel 1 goto :error
echo.
echo Build complete: dist\scrcpy-flow-1.0\
echo Copy tools\ into dist\scrcpy-flow-1.0\tools\ for portable use.
pause
exit /b 0
:error
echo Build failed.
pause
exit /b 1
