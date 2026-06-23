@echo off
setlocal EnableDelayedExpansion

if not defined STROKEAI_RUNNING (
    set STROKEAI_RUNNING=1
    cmd /k ""%~f0""
    exit /b
)

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

cls
echo.
echo  ==========================================
echo   StrokeAI v4 VinceNet - Starting
echo   Port: 9000
echo  ==========================================
echo.

:: Checks
if not exist "%ROOT%\backend\venv\Scripts\activate.bat" (
    echo  [ERROR] Virtual environment not found.
    echo  Please run setup.bat first!
    echo.
    pause >nul & exit /b 1
)
if not exist "%ROOT%\frontend\index.html" (
    echo  [ERROR] frontend\index.html missing!
    pause >nul & exit /b 1
)

:: Model check
if exist "%ROOT%\backend\vincenet_stroke_f32.tflite" (
    echo  [OK] VinceNet f32 model found
) else if exist "%ROOT%\backend\vincenet_stroke_int8.tflite" (
    echo  [OK] VinceNet int8 model found
) else (
    echo  [WARNING] No VinceNet model found - will run in DEMO mode
    echo  Copy vincenet_stroke_f32.tflite to: %ROOT%\backend\
)
echo.

:: Kill anything on port 9000
echo  Clearing port 9000...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":9000 "') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 2 /nobreak >nul
echo  [OK] Port 9000 clear
echo.

:: Start server
echo  Starting VinceNet server on port 9000...
start "StrokeAI-VinceNet" cmd /k "title StrokeAI v4 VinceNet [9000] && cd /d "%ROOT%\backend" && call venv\Scripts\activate.bat && echo. && echo ============================================ && echo  StrokeAI v4 VinceNet RUNNING && echo  Open browser: http://localhost:9000 && echo  Health check: http://localhost:9000/health && echo  Press Ctrl+C to stop && echo ============================================ && echo. && uvicorn main:app --host 0.0.0.0 --port 9000 --reload"

:: Wait for ready
echo  Waiting for server to start...
set /a T=0
:wait
timeout /t 2 /nobreak >nul
curl -s http://localhost:9000/health >nul 2>&1
if not errorlevel 1 goto :ready
set /a T+=1
if !T! lss 15 goto :wait
echo  Server is taking longer than expected...
goto :open

:ready
echo  [OK] Server is ready!

:open
echo.
echo  Opening http://localhost:9000 ...
start "" "http://localhost:9000"

echo.
echo  ==========================================
echo   RUNNING at http://localhost:9000
echo  ==========================================
echo.
echo  Check the "StrokeAI-VinceNet" window for logs.
echo  Press any key to close this launcher...
pause >nul
