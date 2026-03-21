@echo off
setlocal enabledelayedexpansion

:: Check terminal width a bit, though simple echo will suffice
title CKD AI - Project Info

set "ESC="
set "GREEN=!ESC![92m"
set "CYAN=!ESC![96m"
set "YELLOW=!ESC![93m"
set "BOLD=!ESC![1m"
set "RESET=!ESC![0m"

echo.
echo %CYAN%=========================================================%RESET%
echo %CYAN%%BOLD%   CKD AI - Chronic Kidney Disease Prediction System   %RESET%
echo %CYAN%=========================================================%RESET%
echo.
echo %GREEN%%BOLD%[PROJECT STATUS]%RESET%
echo  - Dual-Branch Late Fusion Architecture: %GREEN%ACTIVE%RESET%
echo  - Ocular Model (EfficientNet-B3): %GREEN%5-FOLD ENSEMBLE%RESET%
echo  - Clinical Model (Random Forest): %GREEN%DYNAMIC FEATURE MAPPING%RESET%
echo  - Explainability (Grad-CAM++): %GREEN%INTEGRATED%RESET%
echo.
echo %YELLOW%%BOLD%[QUICK RUN COMMANDS]%RESET%
echo.
echo %BOLD% 1. To Train Clinical Model:%RESET%
echo    %CYAN%venv\Scripts\python.exe scripts\clinical_model.py%RESET%
echo.
echo %BOLD% 2. To Train Ocular Ensemble (GPU):%RESET%
echo    %CYAN%venv\Scripts\python.exe scripts\train_ocular_v2.py%RESET%
echo.
echo %BOLD% 3. To Run Web Application Dashboard:%RESET%
echo    %CYAN%venv\Scripts\python.exe web_app\main.py%RESET%
echo.
echo %YELLOW%%BOLD%[ACCESS URL]%RESET%
echo  Local Dashboard: %BOLD%http://localhost:8000%RESET%
echo.
echo %CYAN%---------------------------------------------------------%RESET%
echo  This project uses %BOLD%Focal Loss%RESET% and %BOLD%Graham Preprocessing%RESET%.
echo  Ensure your virtual environment is active!
echo %CYAN%---------------------------------------------------------%RESET%
echo.

pause
