param(
    [switch]$PersistUserPath
)

$ErrorActionPreference = "Stop"

$wingetRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
$candidates = Get-ChildItem -Path $wingetRoot -Directory -Filter "Gyan.FFmpeg_*" -ErrorAction SilentlyContinue |
    ForEach-Object {
        Get-ChildItem -Path $_.FullName -Directory -Filter "ffmpeg-*-full_build" -ErrorAction SilentlyContinue
    } |
    ForEach-Object {
        Join-Path $_.FullName "bin"
    } |
    Where-Object {
        Test-Path (Join-Path $_ "ffmpeg.exe")
    }

if (-not $candidates) {
    throw "Could not find a WinGet Gyan.FFmpeg installation."
}

# Pick newest folder by LastWriteTime.
$ffBin = $candidates |
    Sort-Object { (Get-Item $_).LastWriteTime } -Descending |
    Select-Object -First 1

# Current terminal/session.
$env:Path = "$ffBin;$env:Path"

Write-Host "Using FFmpeg from:" -ForegroundColor Cyan
Write-Host $ffBin

Write-Host "`nffmpeg:" -ForegroundColor Yellow
ffmpeg -version | Select-Object -First 1

Write-Host "`nffprobe:" -ForegroundColor Yellow
ffprobe -version | Select-Object -First 1

Write-Host "`nResolution order:" -ForegroundColor Yellow
where.exe ffmpeg
where.exe ffprobe

if ($PersistUserPath) {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $parts = @($userPath -split ";" | Where-Object {
        $_ -and ($_ -ne $ffBin)
    })

    $newUserPath = $ffBin + ";" + ($parts -join ";")
    [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")

    Write-Host "`nPersisted Gyan FFmpeg at the front of the USER Path." -ForegroundColor Green
    Write-Host "Close and reopen VS Code/PowerShell for the persistent PATH change to take effect."
}
