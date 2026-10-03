$ErrorActionPreference = "Continue"

Write-Host "=== TTsbyHoaf LOCAL ENV CHECK ===" -ForegroundColor Cyan

Write-Host "`n=== DRIVE E ===" -ForegroundColor Yellow
Get-PSDrive E | Format-Table Name, @{N='UsedGB';E={[math]::Round($_.Used/1GB,2)}}, @{N='FreeGB';E={[math]::Round($_.Free/1GB,2)}} -AutoSize

Write-Host "`n=== GIT ===" -ForegroundColor Yellow
git --version

Write-Host "`n=== PYTHON (must be 3.11 for this repo) ===" -ForegroundColor Yellow
python --version
python -c "import sys; print(sys.executable); print('version_info=', sys.version_info[:3])"

Write-Host "`n=== VIRTUALENV ===" -ForegroundColor Yellow
if (Test-Path ".\.venv\Scripts\python.exe") {
    .\.venv\Scripts\python.exe --version
} else {
    Write-Host "MISSING: .venv" -ForegroundColor Red
}

Write-Host "`n=== FFMPEG ===" -ForegroundColor Yellow
ffmpeg -version | Select-Object -First 1
ffprobe -version | Select-Object -First 1
where.exe ffmpeg
where.exe ffprobe

Write-Host "`n=== KAGGLE ===" -ForegroundColor Yellow
try {
    kaggle --version
    kaggle config view
} catch {
    Write-Host "MISSING or not on PATH: kaggle" -ForegroundColor Red
}

Write-Host "`n=== HUGGING FACE ===" -ForegroundColor Yellow
python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"

Write-Host "`n=== GIT STATUS ===" -ForegroundColor Yellow
git status --short
