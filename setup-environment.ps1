param(
    [switch]$UpdateLock
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$environmentPython = Join-Path $projectRoot '.venv/Scripts/python.exe'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required. Install uv before running this script.'
}

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $environmentPython)) {
        & uv venv --python 3.13 --no-python-downloads .venv
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create the project environment.' }
    }

    if ($UpdateLock) {
        & uv pip compile --quiet requirements.in --python $environmentPython --torch-backend cpu --generate-hashes --output-file requirements.txt
        if ($LASTEXITCODE -ne 0) { throw 'Failed to resolve requirements.' }
    }

    & uv pip sync requirements.txt --python $environmentPython --torch-backend cpu --require-hashes --strict --link-mode copy
    if ($LASTEXITCODE -ne 0) { throw 'Failed to synchronize the project environment.' }

    & uv pip check --python $environmentPython
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }

    Write-Output "Ready: $environmentPython"
} finally {
    Pop-Location
}
