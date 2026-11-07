@echo off
set "PYTHONPATH=%~dp0"
set "PY=pythonw"
if exist "%~dp0.venv\Scripts\pythonw.exe" set "PY=%~dp0.venv\Scripts\pythonw.exe"
start "" "%PY%" -m whisper_studio ui
