$ErrorActionPreference = "stop"

Set-Location "$PSScriptRoot\.."

conda activate spidey
pytest