param([string]$Distribution = 'Ubuntu-22.04')
$ErrorActionPreference = 'Stop'
$linuxRootResult = & wsl.exe -d $Distribution --exec wslpath -a $PSScriptRoot.Replace('\','/')
if ($LASTEXITCODE -ne 0) { throw 'Cannot locate the project in WSL.' }
$linuxRoot = ($linuxRootResult -join "`n").Trim()
if (-not $linuxRoot) { throw 'WSL returned an empty project path.' }
& wsl.exe -d $Distribution --exec bash "$linuxRoot/scripts/build.sh"
if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE." }
