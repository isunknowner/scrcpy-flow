@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import numpy, PIL, tkinter" >nul 2>nul
    if not errorlevel 1 (
        ".venv\Scripts\python.exe" -m scflow %*
        goto :done
    )
)
py -3 -m scflow %*
if errorlevel 9009 python -m scflow %*
:done
endlocal
