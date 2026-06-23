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
echo  ============================================
echo   StrokeAI v4 VinceNet - Setup
echo   Port: 9000
echo  ============================================
echo.
echo  Press any key to begin...
pause >nul

:: ── Find Python 3.11 ─────────────────────────────────────────────────
echo.
echo  [1/3] Finding Python 3.11...
set "PY="
if exist "C:\Python311\python.exe"                                          set "PY=C:\Python311\python.exe"
if exist "C:\Program Files\Python311\python.exe"                            set "PY=C:\Program Files\Python311\python.exe"
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"              set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe" set "PY=%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe"

if not defined PY (
    py -3.11 --version >nul 2>&1
    if not errorlevel 1 (
        for /f "usebackq tokens=*" %%i in (`py -3.11 -c "import sys; print(sys.executable)"`) do set "PY=%%i"
    )
)

if not defined PY (
    echo  [ERROR] Python 3.11 not found.
    echo  Download: https://www.python.org/downloads/releases/python-3119/
    echo  CHECK "Add Python to PATH" during install.
    start "" "https://www.python.org/downloads/releases/python-3119/"
    pause >nul & exit /b 1
)
echo  [OK] !PY!
"!PY!" --version
echo.
echo  Press any key for next step...
pause >nul

:: ── Backend venv ─────────────────────────────────────────────────────
echo.
echo  [2/3] Setting up Python environment...
if exist "%ROOT%\backend\venv" (
    echo  Removing old venv...
    rmdir /s /q "%ROOT%\backend\venv"
)
cd /d "%ROOT%\backend"
"!PY!" -m venv venv
if errorlevel 1 ( echo  [ERROR] venv failed. & pause >nul & exit /b 1 )

call venv\Scripts\activate.bat
echo  Python in venv:
python --version

echo  Upgrading pip...
python -m pip install --upgrade pip -q

echo  Installing setuptools + wheel...
pip install setuptools==69.5.1 wheel==0.43.0 -q

echo  Installing Pillow (pre-built)...
pip install Pillow==10.2.0 --only-binary=:all: -q

echo  Installing numpy...
pip install numpy==1.26.4 -q

echo  Installing TensorFlow (large download, be patient)...
pip install tensorflow==2.16.1
if errorlevel 1 (
    pip install tensorflow-cpu==2.16.1
    if errorlevel 1 ( echo  [ERROR] TensorFlow failed. & pause >nul & exit /b 1 )
)

echo  Installing FastAPI + uvicorn + other packages...
pip install fastapi==0.111.0 -q
pip install uvicorn[standard]==0.30.1 -q
pip install python-multipart==0.0.9 -q
pip install anthropic==0.28.0 -q
pip install python-dotenv==1.0.1 -q

echo  [OK] All packages installed.
echo.
echo  Press any key for next step...
pause >nul

:: ── Check files ───────────────────────────────────────────────────────
echo.
echo  [3/3] Checking project files...

set "OK=1"
if exist "%ROOT%\backend\vincenet_stroke_f32.tflite" (
    echo  [OK] vincenet_stroke_f32.tflite found
) else (
    echo  [MISSING] vincenet_stroke_f32.tflite
    echo  Copy it to: %ROOT%\backend\vincenet_stroke_f32.tflite
    set "OK=0"
)
if exist "%ROOT%\frontend\index.html" (
    echo  [OK] frontend\index.html found
) else (
    echo  [ERROR] frontend\index.html MISSING
    set "OK=0"
)
if exist "%ROOT%\backend\model_meta.json" (
    echo  [OK] model_meta.json found
) else (
    echo  [WARNING] model_meta.json not found - using defaults
)

echo.
echo  ============================================
if "!OK!"=="1" (
    echo   SETUP COMPLETE!
    echo.
    echo   Double-click start.bat to launch.
    echo   App opens at: http://localhost:9000
) else (
    echo   SETUP DONE - but some files are missing.
    echo   Add the missing files then run start.bat.
)
echo  ============================================
echo.
echo  Press any key to close...
pause >nul
