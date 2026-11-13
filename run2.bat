@echo off
setlocal
cd /d "%~dp0"
if exist ".venv2\Scripts\python.exe" (
    ".venv2\Scripts\python.exe" -c "import PySide6" >nul 2>nul
    if not errorlevel 1 (
        ".venv2\Scripts\python.exe" -m scflow %*
        goto :done
    )
)
py -3 -m scflow %*
if errorlevel 9009 python -m scflow %*
:done
endlocal
