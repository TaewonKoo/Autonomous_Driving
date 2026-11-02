@echo off
cd /d "%~dp0"
python -B view_map.py
if errorlevel 1 pause
