# OPT-12: start a Python command outside the caller's process tree (Win32_Process.Create via WMI), so it survives the
# end of the Claude Code session or the VS Code terminal. No admin rights needed. stdout/stderr go to log files.
#   powershell -File scripts/opt12_launch.ps1 -Script scripts/opt12_night.py -ScriptArgs run -Log runner
param(
    [Parameter(Mandatory = $true)][string]$Script,
    [string]$ScriptArgs = "",
    [Parameter(Mandatory = $true)][string]$Log
)
$py   = 'C:\Users\nazar\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe'
$sp   = 'C:\UCU\AI\hw2\Multiple-Object-Video-Detection\.venv\Lib\site-packages'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$logs = Join-Path $repo 'outputs\opt12\logs'
New-Item -ItemType Directory -Force $logs | Out-Null
$cmd = 'cmd.exe /d /s /c "set "PYTHONPATH={0}" && set "PYTHONUNBUFFERED=1" && "{1}" "{2}\{3}" {4} 1>>"{5}\{6}.out" 2>>"{5}\{6}.err""' -f $sp, $py, $repo, $Script, $ScriptArgs, $logs, $Log
$si = New-CimInstance -CimClass (Get-CimClass Win32_ProcessStartup) -Property @{ ShowWindow = [uint16]0 } -ClientOnly
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $repo; ProcessStartupInformation = $si }
"ReturnValue=$($r.ReturnValue) wrapperPid=$($r.ProcessId)"
