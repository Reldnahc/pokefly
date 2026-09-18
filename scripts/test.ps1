param(
    [switch]$Live,
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$validationRoot = Join-Path $projectRoot ('runs\verify-' + [Guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $validationRoot) { throw 'Validation directory already exists.' }
$null = New-Item -ItemType Directory -Path $validationRoot

Push-Location $projectRoot
try {
    # Isolate temp/cache permissions between normal users and sandboxed test runs.
    $validationTemp = Join-Path $validationRoot 'temp'
    $validationCache = Join-Path $validationRoot 'cache'
    & $projectPython -m pytest -q --basetemp $validationTemp -o "cache_dir=$validationCache"
    if ($LASTEXITCODE -ne 0) { throw 'Python tests failed.' }
    & $projectPython -m ruff check src tests scripts
    if ($LASTEXITCODE -ne 0) { throw 'Lint checks failed.' }
    if (Get-Command node -ErrorAction SilentlyContinue) {
        & node --test tests/dashboard.test.cjs
        if ($LASTEXITCODE -ne 0) { throw 'JavaScript unit tests failed.' }
    } else {
        Write-Host 'JavaScript unit tests skipped: install Node to run them.'
    }
    if ($Live) {
        & $projectPython scripts/smoke_watch.py --device $Device
        if ($LASTEXITCODE -ne 0) { throw 'Real-ROM pixel-stream smoke test failed.' }
        & $projectPython scripts/smoke_watch.py --device $Device --autonomous
        if ($LASTEXITCODE -ne 0) { throw 'Autonomous pixel-stream smoke test failed.' }
        & $projectPython scripts/smoke_internal.py --device $Device
        if ($LASTEXITCODE -ne 0) { throw 'Internal-learning/resume smoke test failed.' }
        & $projectPython scripts/smoke_watch.py --device $Device --autonomous --config configs/hybrid-v1.json
        if ($LASTEXITCODE -ne 0) { throw 'Hybrid graded-activity stream smoke test failed.' }
        & $projectPython scripts/smoke_internal.py --device $Device --config configs/hybrid-v1.json
        if ($LASTEXITCODE -ne 0) { throw 'Hybrid exact-resume smoke test failed.' }
        & $projectPython scripts/smoke_watch.py --device $Device --autonomous --config configs/sensory-isolated-v1.json
        if ($LASTEXITCODE -ne 0) { throw 'Sensory-isolated stream smoke test failed.' }
        & $projectPython scripts/smoke_internal.py --device $Device --config configs/sensory-isolated-v1.json
        if ($LASTEXITCODE -ne 0) { throw 'Sensory-isolated exact-resume smoke test failed.' }
        foreach ($candidateProfile in @('stream-v1', 'endpoint-v1', 'quiescent-v1',
            'compartment-ema-v1', 'intrinsic-v1', 'sensorimotor-v1',
            'sensorimotor-normalized-v1', 'sensorimotor-bounded-v1',
            'sensorimotor-bounded-serial-v2', 'sensorimotor-budget-v1',
            'sensorimotor-perturb-v1', 'sensorimotor-perturb-v2', 'sensorimotor-perturb-v3',
            'sensorimotor-reset-v2', 'compartment-reset-v2', 'sensorimotor-low-noise-v1',
            'sensorimotor-delayed-v1', 'sensorimotor-dual-v1', 'sensorimotor-centered-v1',
            'sensorimotor-score-v1', 'sensorimotor-score-v2', 'sensorimotor-score-v3',
            'sensorimotor-escape-v1', 'sensorimotor-sustained-v1', 'sensorimotor-sustained-menu-v1',
            'sensorimotor-impulse-dual-v1', 'sensorimotor-perturb-normalized-v3',
            'sensorimotor-noise-dual-v1', 'sensorimotor-event-dual-v1', 'sensorimotor-serial-v1',
            'sensorimotor-score-delayed-v2', 'sensorimotor-outcome-v3',
            'sensorimotor-serial-outcome-v3', 'sensorimotor-score-centered-v4',
            'sensorimotor-adaptive-v1', 'sensorimotor-premotor-v1',
            'sensorimotor-score-projected-v5', 'sensorimotor-gain12-v1',
            'visual-release-wide-v3')) {
            $candidateConfig = "configs/$candidateProfile.json"
            $candidateCalibration = (Get-Content -Raw -LiteralPath $candidateConfig | ConvertFrom-Json).brain.intrinsic_calibration
            if ($candidateCalibration -and -not (Test-Path -LiteralPath $candidateCalibration)) {
                Write-Host "SKIPPED ${candidateProfile}: generate its optional calibration first (MODEL_VARIANTS.md)."
                continue
            }
            & $projectPython scripts/smoke_watch.py --device $Device --autonomous --config $candidateConfig
            if ($LASTEXITCODE -ne 0) { throw "$candidateProfile live-display smoke test failed." }
            & $projectPython scripts/smoke_internal.py --device $Device --config $candidateConfig
            if ($LASTEXITCODE -ne 0) { throw "$candidateProfile exact-resume smoke test failed." }
        }
    }
    Write-Host "Validation files: $validationRoot"
    Write-Host 'These checks do not replace real-browser visual verification.'
} finally {
    Pop-Location
}
