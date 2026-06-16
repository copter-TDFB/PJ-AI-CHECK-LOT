# Build a fully self-contained, NO-INSTALL Windows test bundle under
# dist\portable-bundle\.  It bundles an embeddable Python 3.11 + all deps +
# models + the backend + the wizard + the caller's Drive creds, so a Windows
# recipient just unzips and double-clicks START.bat — no Docker, no Python, no
# downloads.
#
# It is a TEST harness: TEST_MODE=1 makes deploy skip the Cloud Run trigger and
# the DRIVE_* ids point at the OCR-LOT-TEST folder — production is untouched.
#
# SECURITY: the bundled .env + gcp-key.json hold real Drive/Vision creds. Anyone
# with the zip can act as you on Drive and spend your Vision quota. Send only to
# trusted people; revoke the OAuth token afterwards if needed.
#
# Run from the repo root (needs .env.test for ids/creds):
#     powershell -ExecutionPolicy Bypass -File scripts\build_portable_bundle.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$build  = Join-Path $repo "dist\portable-build"
$py     = Join-Path $build "python"
$pyexe  = Join-Path $py "python.exe"
$bundle = Join-Path $repo "dist\portable-bundle"

# ---------------------------------------------------------------------------
# 1. Build the portable Python env (download + deps) if not already present.
# ---------------------------------------------------------------------------
if (-not (Test-Path $pyexe)) {
    Write-Host "Building portable Python env (one-time, downloads ~1.5GB)..."
    New-Item -ItemType Directory -Force -Path $py | Out-Null
    $embed = Join-Path $build "embed.zip"
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip" -OutFile $embed
    Expand-Archive -Path $embed -DestinationPath $py -Force
    Remove-Item $embed
    # `..` puts the bundle root (one level above python.exe) on sys.path so the
    # app modules (main, api, pipeline, ...) import. With a ._pth present the
    # embeddable interpreter ignores cwd/PYTHONPATH, so this line is required.
    @"
python311.zip
.
..
Lib\site-packages
import site
"@ | Set-Content -Path (Join-Path $py "python311._pth") -Encoding ascii
    $getpip = Join-Path $build "get-pip.py"
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getpip
    & $pyexe $getpip --no-warn-script-location
    Remove-Item $getpip
    & $pyexe -m pip install --no-warn-script-location torch torchvision --index-url https://download.pytorch.org/whl/cpu
    & $pyexe -m pip install --no-warn-script-location -r requirements.txt
} else {
    Write-Host "Reusing existing portable Python env at $py"
}

# verify the env imports the app's heavy deps
& $pyexe -c "import torch, cv2, ultralytics, fastapi, uvicorn, googleapiclient, PIL, yaml, dotenv; print('env imports OK')"
if ($LASTEXITCODE -ne 0) { throw "portable env failed import check" }

# ---------------------------------------------------------------------------
# 2. Assemble the bundle.
# ---------------------------------------------------------------------------
if (Test-Path $bundle) { Remove-Item $bundle -Recurse -Force }
New-Item -ItemType Directory -Force -Path $bundle | Out-Null

Write-Host "Copying Python runtime..."
Copy-Item -Recurse $py (Join-Path $bundle "python")
# strip pip caches / __pycache__ to shrink
Get-ChildItem -Path (Join-Path $bundle "python") -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Copying backend source + models..."
foreach ($f in @("main.py", "requirements.txt")) { Copy-Item (Join-Path $repo $f) (Join-Path $bundle $f) }
foreach ($d in @("api", "pipeline", "services", "utils", "config")) {
    Copy-Item -Recurse (Join-Path $repo $d) (Join-Path $bundle $d)
}
Get-ChildItem -Path $bundle -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notlike "*\python\*" } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Join-Path $bundle "models") | Out-Null
Copy-Item (Join-Path $repo "models\classifier.pt") (Join-Path $bundle "models\")
Copy-Item (Join-Path $repo "models\detector.pt") (Join-Path $bundle "models\")
if (Test-Path (Join-Path $repo "gcp-key.json")) { Copy-Item (Join-Path $repo "gcp-key.json") (Join-Path $bundle "gcp-key.json") }

Write-Host "Building wizard (pinned to :8081)..."
$wiz = Get-Content (Join-Path $repo "web\wizard.html") -Raw
$inject = "<head>`r`n<script>`r`n  // Portable test bundle - backend runs locally on :8081.`r`n  window.API_BASE_OVERRIDE = 'http://localhost:8081';`r`n</script>"
if ($wiz -notmatch [regex]::Escape("API_BASE_OVERRIDE = 'http://localhost:8081'")) {
    $wiz = [regex]::new("<head>").Replace($wiz, $inject, 1)
}
New-Item -ItemType Directory -Force -Path (Join-Path $bundle "static") | Out-Null
Set-Content -Path (Join-Path $bundle "static\wizard.html") -Value $wiz -Encoding UTF8

Write-Host "Writing .env (TEST_MODE + test Drive ids + creds)..."
$envtest = @{}
foreach ($line in Get-Content (Join-Path $repo ".env.test")) {
    $t = $line.Trim()
    if ($t -and -not $t.StartsWith("#") -and $t.Contains("=")) {
        $i = $t.IndexOf("="); $envtest[$t.Substring(0, $i).Trim()] = $t.Substring($i + 1).Trim()
    }
}
$keys = @("DRIVE_DETECTOR_DATASET_FOLDER_ID","DRIVE_CLASSIFIER_DATASET_FOLDER_ID",
          "DRIVE_CONFIG_OVERRIDES_FILE_ID","DRIVE_OAUTH_CLIENT_ID","DRIVE_OAUTH_CLIENT_SECRET",
          "DRIVE_OAUTH_REFRESH_TOKEN","REFERENCE_DETECTOR_PATH","REFERENCE_CLASSIFIER_PATH")
$envLines = @("# Generated by build_portable_bundle.ps1 - TEST harness", "TEST_MODE=1",
              "DRIVE_MANIFEST_FILE_ID=", "GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json", "LOG_LEVEL=INFO")
foreach ($k in $keys) { $envLines += "$k=$($envtest[$k])" }
Set-Content -Path (Join-Path $bundle ".env") -Value ($envLines -join "`r`n") -Encoding ascii

Write-Host "Writing launchers + README..."
$startBat = @'
@echo off
cd /d "%~dp0"
echo ============================================
echo   OCR Lot Checker - Test Wizard (portable)
echo ============================================
echo Starting backend + wizard (no install needed)...
start "ocr-test-backend" /min "%~dp0python\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8081
start "ocr-test-wizard" /min "%~dp0python\python.exe" -m http.server 8091 --directory static
echo.
echo Waiting for the backend (first start loads models, ~20-40s)...
set /a n=0
:wait
set /a n+=1
curl -sf -o nul http://localhost:8081/openapi.json
if not errorlevel 1 goto ready
if %n% GEQ 120 ( echo [!] Backend did not start - see the "ocr-test-backend" window. & pause & exit /b 1 )
timeout /t 2 >nul
goto wait
:ready
start "" http://localhost:8091/wizard.html
echo.
echo   Wizard opened: http://localhost:8091/wizard.html
echo   To stop: double-click STOP.bat
echo.
timeout /t 6 >nul
'@
Set-Content -Path (Join-Path $bundle "START.bat") -Value $startBat -Encoding ascii

$stopBat = @'
@echo off
echo Stopping...
taskkill /fi "WINDOWTITLE eq ocr-test-backend*" /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq ocr-test-wizard*" /f >nul 2>&1
echo Stopped.
timeout /t 2 >nul
'@
Set-Content -Path (Join-Path $bundle "STOP.bat") -Value $stopBat -Encoding ascii

$readme = @'
# OCR Lot Checker - Test Wizard (Portable, Windows x64)

Fully self-contained TEST harness. **No install needed** - Python + all
dependencies are bundled. **Never touches production** (deploy is simulated;
Drive writes go only to the shared OCR-LOT-TEST test folder).

## Run
1. Unzip anywhere.
2. Double-click **START.bat**.
   - First run loads the models (~20-40s), then your browser opens automatically.
3. Wizard: http://localhost:8091/wizard.html

To stop: double-click **STOP.bat**.

## Requirements
- Windows 64-bit. That is all - nothing to install.

## Notes
- Drafts/images you create live inside this folder and can be deleted freely.
- This bundle contains the sender''s Google credentials; treat it as sensitive
  and do not re-share.
'@
Set-Content -Path (Join-Path $bundle "README.txt") -Value $readme -Encoding UTF8

# ---------------------------------------------------------------------------
# 3. Verify the bundled app imports with the bundled python.
# ---------------------------------------------------------------------------
Write-Host "Verifying bundled backend imports..."
Push-Location $bundle
& (Join-Path $bundle "python\python.exe") -c "import main; print('bundle backend imports OK:', main.app.title)"
$ok = $LASTEXITCODE
Pop-Location
if ($ok -ne 0) { throw "bundled backend failed to import" }

$size = (Get-ChildItem -Path $bundle -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("Built portable bundle: {0}  ({1:N0} MB)" -f $bundle, ($size / 1MB))
Write-Host "Next: zip it ->  Compress-Archive -Path '$bundle\*' -DestinationPath dist\portable-bundle.zip"
Write-Host "SECURITY: bundle holds real creds - send only to trusted recipients."
