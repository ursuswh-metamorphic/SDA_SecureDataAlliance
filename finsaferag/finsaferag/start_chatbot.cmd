@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo    Federated RAG Chatbot with Privacy Protection
echo ==================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [X] Python not found
    pause
    exit /b 1
)
echo [OK] Python:
python --version
echo.

if not exist "config.toml" (
    echo [X] config.toml not found. Run from project root
    pause
    exit /b 1
)

if not exist "logs" mkdir logs
echo [OK] Logs directory ready
echo.

REM Start Flower in a new window (UTF-8 to avoid emoji encoding errors)
echo Starting Flower Federated Server...
echo ==================================================
set PYTHONIOENCODING=utf-8
start "Flower Federated Server" cmd /k "cd /d "%~dp0" && set PYTHONIOENCODING=utf-8 && flwr run . 1>logs\flower.log 2>&1"
echo [OK] Flower started in separate window. Logs: logs\flower.log
echo.
echo Waiting for server and clients (max 40s)...

set waited=0
:wait_loop
timeout /t 1 /nobreak >nul
findstr /C:"Federated RAG bridge initialized" logs\flower.log >nul 2>nul
if %errorlevel% equ 0 goto flower_ready
findstr /C:"Server ready" logs\flower.log >nul 2>nul
if %errorlevel% equ 0 goto flower_ready
findstr /C:"Successfully started run" logs\flower.log >nul 2>nul
if %errorlevel% equ 0 goto flower_ready
findstr /C:"Uvicorn" logs\flower.log >nul 2>nul
if %errorlevel% equ 0 goto flower_ready
set /a waited+=1
if %waited% lss 60 goto wait_loop
echo [WARNING] Timeout waiting for Flower. Check logs\flower.log for errors.

:flower_ready
echo [OK] Flower ready!
timeout /t 2 /nobreak >nul
echo.

echo Starting Streamlit UI...
echo ==================================================
echo.
echo Open: http://localhost:8501
echo.
echo Press Ctrl+C to stop Streamlit. Then close the "Flower Federated Server" window to stop the server.
echo Logs: logs\flower.log
echo ==================================================
echo.

cd ui
streamlit run streamlit_app.py

echo.
echo To stop the federated server, close the "Flower Federated Server" window.
pause
