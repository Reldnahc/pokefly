param(
    [switch]$Intro,
    [ValidateRange(0, 2147483647)][int]$Steps = 0,
    [ValidateRange(0, 3000)][double]$Hz = 5,
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto',
    [ValidateSet('learn', 'frozen', 'no-reward')][string]$Mode = 'learn',
    [ValidateRange(0, 65535)][int]$Port = 8777,
    [ValidateSet('baseline', 'hybrid-v1', 'sensory-isolated-v1', 'stream-v1',
        'endpoint-v1', 'quiescent-v1', 'compartment-ema-v1', 'intrinsic-v1',
        'sensorimotor-v1', 'sensorimotor-normalized-v1', 'sensorimotor-bounded-v1',
        'sensorimotor-bounded-serial-v2',
        'sensorimotor-budget-v1', 'sensorimotor-perturb-v1', 'sensorimotor-perturb-v2',
        'sensorimotor-perturb-v3', 'sensorimotor-reset-v2', 'compartment-reset-v2',
        'sensorimotor-low-noise-v1', 'sensorimotor-delayed-v1', 'sensorimotor-dual-v1',
        'sensorimotor-centered-v1', 'sensorimotor-score-v1',
        'sensorimotor-score-v2', 'sensorimotor-score-v3', 'sensorimotor-escape-v1',
        'sensorimotor-sustained-v1', 'sensorimotor-sustained-menu-v1',
        'sensorimotor-impulse-dual-v1', 'sensorimotor-noise-dual-v1',
        'sensorimotor-event-dual-v1', 'sensorimotor-serial-v1',
        'sensorimotor-score-delayed-v2', 'sensorimotor-outcome-v3',
        'sensorimotor-serial-outcome-v3', 'sensorimotor-score-centered-v4',
        'sensorimotor-adaptive-v1', 'sensorimotor-premotor-v1',
        'sensorimotor-score-projected-v5', 'sensorimotor-gain12-v1',
        'visual-rate-v1', 'visual-release-wide-v3',
        'visual-wide-projected-v4')][string]$Profile = 'visual-release-wide-v3',
    [string]$LoadState,
    [string]$Resume,
    [string]$Weights
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Run scripts\setup.ps1 first.'
}
if ($Intro -and $Resume) { throw 'Choose Intro or Resume, not both.' }
if ($LoadState -and ($Intro -or $Resume)) { throw 'LoadState cannot combine with Intro or Resume.' }
if ($Resume -and $Weights) { throw 'Choose Resume or Weights, not both.' }
$launchArgs = @('-m', 'pokefly', 'train', '--device', $Device, '--mode', $Mode,
                '--steps', $Steps, '--port', $Port, '--hz', $Hz.ToString([cultureinfo]::InvariantCulture))
if ($Intro) { $launchArgs += '--intro' }
if ($LoadState) { $launchArgs += @('--load-state', (Resolve-Path -LiteralPath $LoadState).Path) }
if ($Resume) { $launchArgs += @('--resume', (Resolve-Path -LiteralPath $Resume).Path) }
if ($Weights) { $launchArgs += @('--weights', (Resolve-Path -LiteralPath $Weights).Path) }
if (-not $Resume -and -not $Weights -and $Profile -ne 'baseline') {
    $launchArgs += @('--config', (Join-Path $projectRoot "configs\$Profile.json"))
}
if (($Resume -or $Weights) -and $PSBoundParameters.ContainsKey('Profile')) {
    throw 'Resume/Weights restore the saved model profile. Omit Profile when loading a brain.'
}
Push-Location $projectRoot
try {
    Write-Host 'Experimental fly controller. Open the Dashboard URL printed below.'
    if ($Resume -or $Weights) {
        Write-Host 'Model: saved checkpoint settings (not upgraded to the fresh-run default).'
    } else {
        Write-Host "Model: $Profile"
    }
    if ($Resume) {
        Write-Host 'Learning history: continuing the saved brain AND its saved game.'
    } elseif ($Weights) {
        Write-Host 'Learning history: retaining learned synapses for this new game attempt.'
    } else {
        Write-Host 'Learning history: NEW brain, original weights. Earlier checkpoints are not loaded.'
        Write-Host 'Use -Resume to continue a saved game, or -Weights <checkpoint> -Intro for a new game with prior learning.'
    }
    Write-Host 'Ctrl+C completes the current decision and saves a checkpoint.'
    & $projectPython @launchArgs
    if ($LASTEXITCODE -ne 0) { throw "Pokefly stopped with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
}
