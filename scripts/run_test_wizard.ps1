# Test wizard harness launcher. Run from repo root:  .\scripts\run_test_wizard.ps1
# Starts the backend on :8081 (TEST_MODE) and serves the test wizard on :8091.
# NEVER touches production Cloud Run.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$envFile = Join-Path $repo ".env.test"
if (-not (Test-Path $envFile)) {
    Write-Error ".env.test not found. Copy .env.test.example to .env.test and fill DRIVE_* ids (run scripts/setup_test_drive.py first)."
}

# Export .env.test into the process env BEFORE python starts, so module-level
# os.getenv reads (config dir, models, drafts) pick up the test values.
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
        $idx = $line.IndexOf("=")
        $key = $line.Substring(0, $idx).Trim()
        $val = $line.Substring($idx + 1).Trim()
        Set-Item -Path "Env:$key" -Value $val
    }
}

# Bootstrap test dirs — mirror prod config + models so the test wizard starts
# with the same packagings and loadable model weights.
$testRoot = Join-Path $repo "data\test"
$testConfig = Join-Path $testRoot "config"
$testModels = Join-Path $testRoot "models"
New-Item -ItemType Directory -Force -Path $testRoot, $testModels, (Join-Path $testRoot "drafts"), (Join-Path $testRoot "crops") | Out-Null

if (-not (Test-Path $testConfig)) {
    Copy-Item -Recurse -Path (Join-Path $repo "config") -Destination $testConfig
    Write-Host "Seeded data/test/config from config/"
}
foreach ($m in @("classifier.pt", "detector.pt")) {
    $dst = Join-Path $testModels $m
    $src = Join-Path $repo "models\$m"
    if ((-not (Test-Path $dst)) -and (Test-Path $src)) {
        Copy-Item -Path $src -Destination $dst
        Write-Host "Seeded data/test/models/$m"
    }
}

Write-Host "TEST_MODE backend -> http://localhost:8081"
Write-Host "Test wizard       -> http://localhost:8091/wizard.html"

# Serve the test wizard (static) in the background, then run the backend.
$wizardDir = Join-Path $repo "test wizzard"
Start-Process -NoNewWindow python -ArgumentList @("-m", "http.server", "8091", "--directory", "`"$wizardDir`"")

python -m uvicorn main:app --port 8081
