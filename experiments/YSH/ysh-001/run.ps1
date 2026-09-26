param(
  [string]$AnalysisPython = (Join-Path $PSScriptRoot '../../../.venv/Scripts/python.exe'),
  [string]$PlotPython = (Join-Path $PSScriptRoot '../../../.venv/Scripts/python.exe')
)
$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $experimentRoot '../../..')).Path
$resultRoot = Join-Path $experimentRoot 'outputs'
& $AnalysisPython (Join-Path $experimentRoot 'analyze.py') --raw (Join-Path $repoRoot 'data/Welding_Data_Set_01.xlsx') --scaled (Join-Path $repoRoot 'data/scaled_data.xlsx') --output $resultRoot
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed' }
& $PlotPython (Join-Path $experimentRoot 'visualize.py') --output $resultRoot
if ($LASTEXITCODE -ne 0) { throw 'Visualization failed' }
& $AnalysisPython (Join-Path $experimentRoot 'verify.py')
if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
Write-Output 'Analysis and plots regenerated. The Markdown report is the reviewed 2026-09-22 snapshot; re-review it after changing inputs or code.'
