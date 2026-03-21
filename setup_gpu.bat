@echo off
echo =========================================================
echo   CKD AI - Python 3.12 Downgrade and GPU Training Setup
echo =========================================================

echo.
echo [1] Downloading Python 3.12.3 Installer (Silently)...
curl -L -o python-3.12.3-amd64.exe https://www.python.org/ftp/python/3.12.3/python-3.12.3-amd64.exe
if not exist "python-3.12.3-amd64.exe" (
    echo [ERROR] Download Failed! Check your internet connection.
    pause
    exit /b 1
)

echo.
echo [2] Installing Python 3.12.3 to your user profile...
echo (You may see a User Account Control prompt. Please allow it.)
start /wait python-3.12.3-amd64.exe /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
echo [✓] Python installation complete!

echo.
echo [3] Removing your broken Python 3.13 "venv" folder...
if exist "venv" (
    rmdir /s /q venv
)
echo [✓] Removed.

echo.
echo [4] Building a brand new virtual environment using Python 3.12...
SET PY312_PATH="%LocalAppData%\Programs\Python\Python312\python.exe"

if exist %PY312_PATH% (
    %PY312_PATH% -m venv venv
) else (
    echo [WARNING] Exact path not found. Attempting py launcher fallback...
    py -3.12 -m venv venv
)

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv failed to build! Your system might require manual Python 3.12 installation.
    pause
    exit /b 1
)
echo [✓] Virtual environment rebuilt successfully!

echo.
echo [5] Installing Core Project Dependencies (skipping default torch)...
call venv\Scripts\python.exe -m pip install --upgrade pip
call venv\Scripts\python.exe -m pip install pandas scikit-learn efficientnet_pytorch opencv-python joblib fastapi uvicorn python-multipart jinja2 matplotlib

echo.
echo [6] Force-Installing Windows CUDA 12.1 GPU PyTorch Wheels...
call venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo =========================================================
echo [SUCCESS] Your GPU Machine Learning Setup is Complete!
echo =========================================================
echo Let's verify if your GPU is finally detected...
echo.
call venv\Scripts\python.exe -c "import torch; print('PyTorch version: ' + torch.__version__); print('GPU CUDA Available: ' + str(torch.cuda.is_available()))"
echo.
echo We are done here! You can delete the "python-3.12.3-amd64.exe" installer file.
echo Please close your terminal, reopen a new one, type "venv\Scripts\activate" and launch your training script again!
pause
