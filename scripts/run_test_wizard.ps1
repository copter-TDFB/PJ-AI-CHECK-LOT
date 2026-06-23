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

# Regenerate the harness wizard from web/wizard.html every start so it never
# drifts (a stale copy hid the TEST_MODE simulate button + showed wrong badges).
# Inject the :8081 API_BASE_OVERRIDE exactly like build_portable_bundle.ps1.
$wizardDir = Join-Path $repo "test wizzard"
New-Item -ItemType Directory -Force -Path $wizardDir | Out-Null
$wiz = Get-Content (Join-Path $repo "web\wizard.html") -Raw -Encoding UTF8
$inject = "<head>`r`n<script>`r`n  // Test wizard harness - backend runs locally on :8081.`r`n  window.API_BASE_OVERRIDE = 'http://localhost:8081';`r`n</script>"
if ($wiz -notmatch [regex]::Escape("API_BASE_OVERRIDE = 'http://localhost:8081'")) {
    $wiz = [regex]::new("<head>").Replace($wiz, $inject, 1)
}
Set-Content -Path (Join-Path $wizardDir "wizard.html") -Value $wiz -Encoding UTF8
Write-Host "Synced test wizzard/wizard.html from web/wizard.html"

Write-Host "TEST_MODE backend -> http://localhost:8081"
Write-Host "Test wizard       -> http://localhost:8091/wizard.html"

# Serve the test wizard (static) in the background, then run the backend.
Start-Process -NoNewWindow python -ArgumentList @("-m", "http.server", "8091", "--directory", "`"$wizardDir`"")

python -m uvicorn main:app --port 8081
