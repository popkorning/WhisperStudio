@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title Whisper Studio: установка

echo Whisper Studio: установка в папку %CD%
echo Python и библиотеки ставятся внутрь этой папки, система не меняется.
echo.

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo [!] Не найден драйвер NVIDIA ^(nvidia-smi^). Приложению нужна видеокарта NVIDIA с CUDA.
  echo.
)

set "UV=uv"
where uv >nul 2>nul
if errorlevel 1 (
  set "UV=%~dp0.tools\uv\uv.exe"
  if not exist "%~dp0.tools\uv\uv.exe" call :get_uv || goto :fail
)
set "UV_PYTHON_INSTALL_DIR=%~dp0.tools\python"
set "UV_NO_CACHE=1"
set "PY=%~dp0.venv\Scripts\python.exe"

echo [1/4] Python 3.12 и окружение .venv
if not exist "%PY%" (
  "%UV%" venv .venv --python 3.12 --seed || goto :fail
)

echo.
echo [2/4] PyTorch с CUDA 12.8, около 3 ГБ
"%UV%" pip install --python "%PY%" torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 || goto :fail

echo.
echo [3/4] Остальные библиотеки
"%UV%" pip install --python "%PY%" -r requirements.txt || goto :fail

echo.
echo [4/4] Модели
echo Базовый набор моделей занимает около 13 ГБ. Его можно скачать и позже: в окне
echo программы на листе «Модели» или командой: ws models install-recommended
set "ANS=y"
set /p "ANS=Скачать сейчас? [Y/n] "
if /i "%ANS%"=="n" goto :done
call "%~dp0ws.cmd" models install-recommended || goto :fail

:done
echo.
echo Готово. Запуск: Whisper Studio.bat
if not defined WS_INSTALL_NOPAUSE pause
exit /b 0

:get_uv
echo Скачиваю uv ^(установщик Python и библиотек^)
if not exist "%~dp0.tools" mkdir "%~dp0.tools"
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $t='%~dp0.tools'; $z=Join-Path $t 'uv.zip'; Invoke-WebRequest -UseBasicParsing 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $z; Expand-Archive -Force $z (Join-Path $t 'uv'); Remove-Item $z"
exit /b %errorlevel%

:fail
echo.
echo Установка прервалась. Запустите install.cmd ещё раз: готовые шаги повторно не скачиваются.
if not defined WS_INSTALL_NOPAUSE pause
exit /b 1
