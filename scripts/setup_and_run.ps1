# One-command setup + run for the test wizard harness (tdfb.co Workspace users).
# Recipient runs this once from the project root:  .\scripts\setup_and_run.ps1
# It creates a venv + installs deps, runs Google consent + creates your own test
# Drive (first run only), then launches the backend (:8081) and wizard (:8091).
# NEVER touches production Cloud Run.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# 1. Python present?
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Install Python 3.11 (https://www.python.org/downloads/) and re-run."
}

# 2. virtualenv + dependencies (one-time)
$venv = Join-Path $repo ".venv"
$python = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Creating virtualenv and installing dependencies (one-time, a few minutes)..."
    python -m venv $venv
    & $python -m pip install --quiet --upgrade pip
    & $python -m pip install --quiet -r (Join-Path $repo "requirements.txt")
}

# 3. model weights present?
foreach ($m in @("classifier.pt", "detector.pt")) {
    if (-not (Test-Path (Join-Path $repo "models\$m"))) {
        Write-Error "models/$m is missing. Get the model files from the sender and place them in the models/ folder, then re-run."
    }
}

# 4. first-run: Google consent + create your own test Drive + write .env.test
if (-not (Test-Path (Join-Path $repo ".env.test"))) {
    if (-not (Test-Path (Join-Path $repo "oauth_client.json")) -and
        -not (Get-ChildItem $repo -Filter "client_secret_*.json" -ErrorAction SilentlyContinue)) {
        Write-Error "oauth_client.json not found. Ask the sender for it, put it in this folder, then re-run."
    }
    Write-Host "First-time setup: a browser will open for Google consent (sign in with your tdfb.co account)..."
    & $python (Join-Path $repo "scripts\bootstrap_test_env.py")
}

# 5. launch — activate the venv so run_test_wizard's `python` is the venv one
& (Join-Path $venv "Scripts\Activate.ps1")
Write-Host ""
Write-Host "Once you see 'Application startup complete', open: http://localhost:8091/wizard.html"
Write-Host ""
& (Join-Path $repo "scripts\run_test_wizard.ps1")
