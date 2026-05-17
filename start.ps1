$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
$Port = if ($env:PORT) { $env:PORT } else { "8011" }
python -m uvicorn backend.main:app --host 0.0.0.0 --port $Port --reload
