# get to main repo
$ErrorActionPreference = "stop"

Set-Location "$PSScriptRoot\.."

# if no arg default msg 
if ($args.Length -eq 0){
    $msg = "Auto commit: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
} 
else {
    $msg = $args -join " " # combine all args 
}

Write-Host "Staging all changes..." -ForegroundColor Cyan
git add .

Write-Host "Committing all changes with msg: '$msg' ..." -ForegroundColor Cyan
git commit -m $msg

Write-Host "Pushing to remote repo... " -ForegroundColor Cyan
git push

Write-Host "Done!" -ForegroundColor Cyan