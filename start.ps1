$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& "$PSScriptRoot/.venv/Scripts/python.exe" -m uvicorn app.api:app --host 127.0.0.1 --port 8012
