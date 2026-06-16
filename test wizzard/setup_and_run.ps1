# Entry point for teammates — run this from inside the project.
#   Right-click → Run with PowerShell, or:  .\setup_and_run.ps1
#
# IMPORTANT: this 'test wizzard' folder is only the frontend. The backend is the
# whole project, so you must have received the ENTIRE project folder
# (pj ocr text check lot), not just this subfolder. This script checks that and
# then hands off to scripts/setup_and_run.ps1 (venv + deps + Google consent +
# your own test Drive + launch). NEVER touches production Cloud Run.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path (Join-Path $repo "main.py"))) {
    Write-Error @"
This needs the FULL project, not just the 'test wizzard' folder.
You appear to have only the 'test wizzard' subfolder.
Ask the sender for the whole project folder (pj ocr text check lot) — it contains
the backend (main.py, services/, models/, scripts/) that this wizard talks to.
"@
}

& (Join-Path $repo "scripts\setup_and_run.ps1")
