<#
.SYNOPSIS
    Runs the full pipeline in order and stops at the first failing stage.

.DESCRIPTION
    Stages: inspect_rasters -> prepare_dataset -> split_dataset ->
            create_patches -> train -> evaluate -> predict
    (split_dataset runs BEFORE create_patches: the geographic split is fixed
    before any patch is cut.)
    $LASTEXITCODE is checked after EVERY stage. Any non-zero exit code
    stops the pipeline immediately with a clear message.

.PARAMETER SmokeTest
    Passes --smoke-test to every stage (tiny subset, 1 epoch).

.PARAMETER Config
    Path to the YAML config. Default: configs/config.yaml

.PARAMETER Python
    Python executable. Default: .venv\Scripts\python.exe if it exists, else python.

.EXAMPLE
    .\run_pipeline.ps1
    .\run_pipeline.ps1 -SmokeTest
#>
[CmdletBinding()]
param(
    [switch]$SmokeTest,
    [string]$Config = "configs/config.yaml",
    [string]$Python = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Run from the project root regardless of the caller's location.
Set-Location -Path $PSScriptRoot
$env:TF_CPP_MIN_LOG_LEVEL = "2"

# Prefer the project venv when no interpreter was given explicitly.
if ($Python -eq "python" -and (Test-Path -Path ".venv\Scripts\python.exe" -PathType Leaf)) {
    $Python = ".venv\Scripts\python.exe"
}
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Host "[pipeline] FAILED: Python executable '$Python' not found. Activate the venv first." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path -Path $Config -PathType Leaf)) {
    Write-Host "[pipeline] FAILED: config file not found: $Config" -ForegroundColor Red
    exit 1
}

$stages = @(
    "inspect_rasters",
    "prepare_dataset",
    "split_dataset",
    "create_patches",
    "train",
    "evaluate",
    "predict"
)

$mode = if ($SmokeTest) { "SMOKE TEST" } else { "FULL" }
Write-Host "[pipeline] Mode: $mode | Config: $Config" -ForegroundColor Cyan

$i = 0
foreach ($stage in $stages) {
    $i++
    $script = Join-Path -Path "scripts" -ChildPath "$stage.py"
    if (-not (Test-Path -Path $script -PathType Leaf)) {
        Write-Host "[pipeline] FAILED: missing stage script $script" -ForegroundColor Red
        exit 1
    }

    $stageArgs = @($script, "--config", $Config)
    if ($SmokeTest) { $stageArgs += "--smoke-test" }

    Write-Host ""
    Write-Host "[pipeline] ($i/$($stages.Count)) Running stage: $stage" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    & $Python @stageArgs
    $code = $LASTEXITCODE
    $sw.Stop()

    if ($code -ne 0) {
        Write-Host "[pipeline] FAILED at stage '$stage' (exit code $code) after $([math]::Round($sw.Elapsed.TotalSeconds, 1)) s. Stopping." -ForegroundColor Red
        exit $code
    }
    Write-Host "[pipeline] Stage '$stage' OK ($([math]::Round($sw.Elapsed.TotalSeconds, 1)) s)" -ForegroundColor Green
}

Write-Host ""
Write-Host "[pipeline] All $($stages.Count) stages completed successfully." -ForegroundColor Green
exit 0
