Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
$env:SKIP_MODEL_LOAD = "1"

python -m pip install -r requirements-test.txt
python -m pytest @args
