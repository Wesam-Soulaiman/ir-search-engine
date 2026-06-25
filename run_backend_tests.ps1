$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$backendDir = Join-Path -Path $repoRoot -ChildPath "backend"
$pythonPath = Join-Path -Path $repoRoot -ChildPath ".venv\Scripts\python.exe"

Set-Location -LiteralPath $backendDir
& $pythonPath manage.py test
exit $LASTEXITCODE
