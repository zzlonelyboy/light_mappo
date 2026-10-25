param(
    [string]$Python = "python",
    [string]$DatasetDir = "dataCubic/d1_5000_scurve_cubic_rcs6_seed20260714_interuav",
    [string]$FeatureDir = "experimentsCubic/deep_ensemble_3gpp_e150_features",
    [string]$RunDir = "runs/deep_ensemble_full_train_streaming_mlp_threshold0p5",
    [string]$PretrainedPerBSDir = "runs/deep_ensemble_train4000_val500_threshold0p5/per_bs_models",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$previousPythonPath = $env:PYTHONPATH
$sourcePath = Join-Path $repoRoot "src"
if ([string]::IsNullOrWhiteSpace($previousPythonPath)) {
    $env:PYTHONPATH = $sourcePath
}
else {
    $env:PYTHONPATH = $sourcePath + [IO.Path]::PathSeparator + $previousPythonPath
}
Push-Location $repoRoot
try {
    $commonArguments = @(
        "--config", "configs/external_baselines/deep_ensemble.json",
        "--dataset-dir", $DatasetDir,
        "--feature-dir", $FeatureDir,
        "--run-dir", $RunDir
    )
    if (-not [string]::IsNullOrWhiteSpace($PretrainedPerBSDir)) {
        if (-not (Test-Path -LiteralPath $PretrainedPerBSDir)) {
            throw "The pretrained BS-level model directory does not exist at $PretrainedPerBSDir."
        }
        $commonArguments += @("--pretrained-per-bs-dir", $PretrainedPerBSDir)
    }
    if ($Force) {
        $trainArguments = @("train") + $commonArguments + @("--force")
    }
    else {
        $trainArguments = @("train") + $commonArguments
    }

    Write-Host "Training one streaming meta MLP on every train-split slot."
    & $Python -m isac_spoofing.cli.run_deep_ensemble_baseline @trainArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Deep Ensemble training failed with exit code $LASTEXITCODE."
    }

    Write-Host "Evaluating the full-train meta MLP on validation and test episodes."
    & $Python -m isac_spoofing.cli.run_deep_ensemble_baseline evaluate @commonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Deep Ensemble evaluation failed with exit code $LASTEXITCODE."
    }

    Write-Host "Results were written to $RunDir."
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
