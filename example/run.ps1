param(
    [ValidateSet('cpu','gpu')][string]$Backend = 'cpu',
    [ValidateSet('full','central')][string]$Profile = 'full',
    [string]$Distribution = 'Ubuntu-22.04'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$linuxRootResult = & wsl.exe -d $Distribution --exec wslpath -a $projectRoot.Replace('\','/')
if ($LASTEXITCODE -ne 0) { throw 'Cannot locate the project in WSL.' }
$linuxRoot = ($linuxRootResult -join "`n").Trim()
if (-not $linuxRoot) { throw 'WSL returned an empty project path.' }
& wsl.exe -d $Distribution --exec bash "$linuxRoot/example/run.sh" $Backend $Profile
if ($LASTEXITCODE -ne 0) { throw "Propagation failed with exit code $LASTEXITCODE." }
