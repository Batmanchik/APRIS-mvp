$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $projectRoot ".run\streamlit.pid"

$stopped = $false
if (Test-Path $pidFile) {
    $pidValue = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pidValue -and (Get-Process -Id $pidValue -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $pidValue -Force
        $stopped = $true
        Write-Host "APRIS stopped (PID=$pidValue)."
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

if (-not $stopped) {
    $fallback = Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -match '-m\s+streamlit\s+run\s+app\.py' -or
            $_.CommandLine -match 'streamlit(\.exe)?"?\s+run\s+app\.py'
        }
    foreach ($proc in $fallback) {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        $stopped = $true
    }
    if ($stopped) {
        Write-Host "APRIS stopped via fallback process search."
    } else {
        Write-Host "APRIS is not running."
    }
}
