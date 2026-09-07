$ErrorActionPreference = "stop"

Set-Location "$PSScriptRoot\.."

conda activate spidey
python -m src.simulation