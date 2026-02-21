@echo off
echo Starting LexiScan Auto API Production Server...
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
pause
