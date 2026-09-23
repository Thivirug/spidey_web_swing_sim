param(
    [Parameter(Mandatory=$true)]
    [string]$scenario
)

$ErrorActionPreference = "stop"

Set-Location "$PSScriptRoot\.."

conda activate spidey 
python -m src.mainV2 scenarios=$scenario