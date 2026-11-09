@echo off
chcp 65001 > nul
title LoRA Trainer 설치
cd /d "%~dp0"
echo ============================================
echo   LoRA Trainer 설치 (Python 3.10~3.13 필요)
echo ============================================

set "COMFY_PY=D:\AI\ComfyUI_windows_portable\python_embeded\python.exe"
if exist "%COMFY_PY%" (
  echo [감지] 로컬 ComfyUI 포터블 Python ^(AMD ROCm / GPU 환경^)을 감지했습니다.
  if not exist "venv\Scripts\python.exe" (
    echo [1/3] ROCm 연동 가상환경 구성...
    mkdir "venv\Scripts" 2>nul
    mkdir "venv\Lib\site-packages" 2>nul
    copy /y "D:\AI\ComfyUI_windows_portable\python_embeded\*.*" "venv\Scripts\" >nul
    powershell -NoProfile -Command "$pth = @('%~dp0'.TrimEnd('\'), '%~dp0venv\Lib\site-packages', 'D:\AI\ComfyUI_windows_portable\python_embeded\python312.zip', 'D:\AI\ComfyUI_windows_portable\python_embeded', 'D:\AI\ComfyUI_windows_portable\python_embeded\Lib\site-packages', 'D:\AI\ComfyUI_windows_portable\ComfyUI', '.', 'import site') -join \"`r`n\"; [System.IO.File]::WriteAllText('%~dp0venv\Scripts\python312._pth', $pth, [System.Text.UTF8Encoding]::new($false))"
  )
  echo [2/3] GUI 및 학습 패키지 설치...
  venv\Scripts\python.exe -m pip install --target "%~dp0venv\Lib\site-packages" --no-warn-script-location PySide6 pytest iniconfig pluggy
  venv\Scripts\python.exe -m pip install --target "%~dp0venv\Lib\site-packages" --no-deps --no-warn-script-location diffusers accelerate prodigyopt prodigy-plus-schedule-free schedulefree dadaptation came-pytorch
  venv\Scripts\python.exe -c "import torch;print('GPU 사용 가능:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
  echo.
  echo 설치 완료! run.bat 으로 실행하세요.
  pause
  exit /b 0
)

where python > nul 2>&1 || (echo [오류] python 을 찾을 수 없습니다. https://www.python.org 에서 설치하세요. & pause & exit /b 1)
if not exist venv (
  echo [1/3] 가상환경 생성...
  python -m venv venv || (echo [오류] 가상환경 생성 실패 & pause & exit /b 1)
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo [2/3] PyTorch (CUDA 12.8, RTX 30/40/50 지원) 설치...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || (echo [오류] PyTorch 설치 실패 & pause & exit /b 1)
echo [3/3] 나머지 패키지 설치...
pip install -r requirements.txt || (echo [오류] 패키지 설치 실패 & pause & exit /b 1)
python -c "import torch;print('CUDA 사용 가능:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo.
echo 설치 완료! run.bat 으로 실행하세요.
pause
