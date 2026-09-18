param([switch]$CpuOnly)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $projectPython)) {
        $installedPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
        if (Test-Path -LiteralPath $installedPython) {
            & $installedPython -m venv .venv
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.12 -m venv .venv
        } else {
            throw 'Install Python 3.12 first: winget install --exact --id Python.Python.3.12'
        }
        if ($LASTEXITCODE -ne 0) { throw 'Creating the Python environment failed.' }
    }

    $projectExtras = if ($CpuOnly) { '.[dev]' } else { '.[gpu,dev]' }
    & $projectPython -m pip install -c constraints-win-py312.txt -e $projectExtras
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

    & $projectPython -m pokefly download
    if ($LASTEXITCODE -ne 0) { throw 'Connectome download failed.' }

    & $projectPython -m pokefly vision-prepare
    if ($LASTEXITCODE -ne 0) { throw 'Pixel-to-photoreceptor mapping failed.' }

    $projectDevice = if ($CpuOnly) { 'cpu' } else { 'auto' }
    & $projectPython -m pokefly doctor --brain --device $projectDevice
    if ($LASTEXITCODE -ne 0) { throw 'The environment did not pass the neural probe.' }
    if (-not (Test-Path -LiteralPath 'fly-data\intrinsic-neutral-v1.npz')) {
        & $projectPython scripts/probe_intrinsic.py --calibration-only --device $projectDevice --export fly-data/intrinsic-neutral-v1.npz
        if ($LASTEXITCODE -ne 0) { throw 'Neutral intrinsic calibration failed.' }
    }
    Write-Host 'Ready. Internal experiment: .\scripts\start.ps1 -Intro'
    Write-Host 'Open http://127.0.0.1:8777. Experimental plasticity, not proven gameplay learning.'
    Write-Host 'Frozen/manual diagnostic: .\.venv\Scripts\python.exe -m pokefly watch --manual'
} finally {
    Pop-Location
}
