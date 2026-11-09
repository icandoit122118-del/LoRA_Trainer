@echo off
chcp 65001 > nul
title LoRA Trainer 내 깃허브 원격 업로드
cd /d "%~dp0"
echo ========================================================
echo   LoRA Trainer -> 내 깃허브 (icandoit122118-del) 업로드
echo ========================================================

gh auth status >nul 2>&1
if errorlevel 1 (
  echo [1/3] 깃허브 계정 웹 로그인을 진행합니다...
  gh auth login -h github.com -p https -w
  if errorlevel 1 (
    echo [오류] 깃허브 로그인에 실패했습니다.
    pause
    exit /b 1
  )
)

gh auth setup-git >nul 2>&1

echo [2/3] 원격 저장소(icandoit122118-del/LoRA_Trainer) 확인 및 생성...
gh repo view icandoit122118-del/LoRA_Trainer >nul 2>&1
if errorlevel 1 (
  gh repo create icandoit122118-del/LoRA_Trainer --public --description "ComfyUI Krea 2 & MOSS-Image LoRA Trainer (AMD ROCm 8GB FP8 & Block-Swap Optimized)"
)

git remote set-url origin https://github.com/icandoit122118-del/LoRA_Trainer.git 2>nul || git remote add origin https://github.com/icandoit122118-del/LoRA_Trainer.git

echo [3/3] 커밋 및 태그 원격 푸시 중...
git push -u origin main --tags
if errorlevel 1 (
  echo [오류] 원격 푸시에 실패했습니다.
  pause
  exit /b 1
)

echo.
echo 완료! https://github.com/icandoit122118-del/LoRA_Trainer 에 반영되었습니다.
pause
