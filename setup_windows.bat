@echo off
cd /d "%~dp0"
echo Creating Lumen Python 3.12 environment...
uv venv .venv --python 3.12
if errorlevel 1 goto :error

echo Installing required packages...
uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Setup complete. Starting Lumen...
".venv\Scripts\python.exe" app.py
exit /b 0

:error
echo.
echo Setup did not complete. Check the error message above.
pause
exit /b 1
