param(
    [Parameter(Mandatory = $false, Position = 0)]
    [ValidateSet("start", "stop", "status", "open")]
    [string]$Command = "status"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunDir = Join-Path $ProjectRoot ".run"
$ApiPidFile = Join-Path $RunDir "api.pid"
$UiPidFile = Join-Path $RunDir "streamlit.pid"
$ApiOutLog = Join-Path $RunDir "api.out.log"
$ApiErrLog = Join-Path $RunDir "api.err.log"
$UiOutLog = Join-Path $RunDir "streamlit.out.log"
$UiErrLog = Join-Path $RunDir "streamlit.err.log"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ApiHealthUrl = "http://127.0.0.1:8000/api/v1/health"
$UiHealthUrl = "http://127.0.0.1:8501/_stcore/health"
$UiUrl = "http://127.0.0.1:8501"
$ApiBaseUrl = "http://127.0.0.1:8000"

function Test-HttpOk {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [Parameter(Mandatory = $false)]
        [int]$TimeoutSec = 2
    )

    try {
        $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSec
        return ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 400)
    } catch {
        return $false
    }
}

function Ensure-RunDir {
    if (-not (Test-Path $RunDir)) {
        New-Item -ItemType Directory -Path $RunDir | Out-Null
    }
}

function Resolve-BootstrapPython {
    $candidates = @(
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\python3.11.exe",
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe",
        "py",
        "python"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -eq "py" -or $candidate -eq "python") {
            try {
                & $candidate --version | Out-Null
                return $candidate
            } catch {
            }
            continue
        }

        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Ensure-Venv {
    if (Test-Path $PythonExe) {
        return
    }

    Write-Host "Python runtime not found in .venv. Bootstrapping environment..."
    $bootstrap = Resolve-BootstrapPython
    if (-not $bootstrap) {
        throw "Python 3.11+ was not found. Install Python and rerun scripts/app.ps1 start."
    }

    Set-Location $ProjectRoot
    if ($bootstrap -eq "py") {
        & py -3.11 -m venv .venv
    } else {
        & $bootstrap -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv."
    }

    & $PythonExe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip."
    }

    & $PythonExe -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install package from pyproject.toml."
    }
}

function Write-Pid {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [int]$ProcessId
    )

    Set-Content -Path $Path -Value $ProcessId
}

function Read-Pid {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        return $null
    }

    $raw = (Get-Content -Path $Path -ErrorAction SilentlyContinue | Select-Object -First 1)
    if (-not $raw) {
        return $null
    }

    [int]$parsedPid = 0
    if ([int]::TryParse($raw, [ref]$parsedPid)) {
        return $parsedPid
    }

    return $null
}

function Stop-ByPidFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PidFile
    )

    $procPid = Read-Pid -Path $PidFile
    if ($procPid) {
        $proc = Get-Process -Id $procPid -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id $procPid -Force -ErrorAction SilentlyContinue
        }
    }

    if (Test-Path $PidFile) {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }
}

function Start-Api {
    if (Test-HttpOk -Url $ApiHealthUrl) {
        Write-Host "API is already running on http://127.0.0.1:8000"
        return
    }

    Remove-Item $ApiOutLog, $ApiErrLog -Force -ErrorAction SilentlyContinue
    $apiProc = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList @("-m", "uvicorn", "apris.api.main:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $ApiOutLog `
        -RedirectStandardError $ApiErrLog `
        -WindowStyle Hidden `
        -PassThru
    Write-Pid -Path $ApiPidFile -ProcessId $apiProc.Id
}

function Start-Ui {
    if (Test-HttpOk -Url $UiHealthUrl) {
        Write-Host "UI is already running on $UiUrl"
        return
    }

    Remove-Item $UiOutLog, $UiErrLog -Force -ErrorAction SilentlyContinue
    $env:CHEOPS_API_BASE_URL = $ApiBaseUrl
    $uiProc = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList @("-m", "streamlit", "run", "app.py", "--server.address", "127.0.0.1", "--server.port", "8501") `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $UiOutLog `
        -RedirectStandardError $UiErrLog `
        -WindowStyle Hidden `
        -PassThru
    Write-Pid -Path $UiPidFile -ProcessId $uiProc.Id
}

function Show-Status {
    $apiUp = Test-HttpOk -Url $ApiHealthUrl
    $uiUp = Test-HttpOk -Url $UiHealthUrl
    $apiPid = Read-Pid -Path $ApiPidFile
    $uiPid = Read-Pid -Path $UiPidFile

    Write-Host ("API:       {0}" -f ($(if ($apiUp) { "running" } else { "stopped" })))
    Write-Host ("UI:        {0}" -f ($(if ($uiUp) { "running" } else { "stopped" })))
    Write-Host ("API PID:   {0}" -f ($(if ($apiPid) { $apiPid } else { "-" })))
    Write-Host ("UI PID:    {0}" -f ($(if ($uiPid) { $uiPid } else { "-" })))
    Write-Host ("UI URL:    {0}" -f $UiUrl)
    Write-Host ("API logs:  {0}, {1}" -f $ApiOutLog, $ApiErrLog)
    Write-Host ("UI logs:   {0}, {1}" -f $UiOutLog, $UiErrLog)
}

function Start-App {
    Ensure-RunDir
    Ensure-Venv

    Set-Location $ProjectRoot
    Start-Api
    Start-Ui

    for ($i = 0; $i -lt 45; $i++) {
        Start-Sleep -Seconds 1
        if ((Test-HttpOk -Url $ApiHealthUrl) -and (Test-HttpOk -Url $UiHealthUrl)) {
            Write-Host "Cheops AI started:"
            Show-Status
            return
        }
    }

    throw "Cheops AI process started but health-check timed out."
}

function Stop-App {
    Stop-ByPidFile -PidFile $ApiPidFile
    Stop-ByPidFile -PidFile $UiPidFile

    $fallback = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match "uvicorn\s+apris\.api\.main:app" -or
        $_.CommandLine -match 'streamlit(\.exe)?"?\s+run\s+app\.py'
    }
    foreach ($proc in $fallback) {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }

    Write-Host "Cheops AI stop command completed."
}

function Open-App {
    if (-not (Test-HttpOk -Url $UiHealthUrl)) {
        throw "UI is not running. Start it first: .\scripts\app.ps1 start"
    }
    Start-Process $UiUrl | Out-Null
    Write-Host "Opened $UiUrl"
}

switch ($Command) {
    "start" { Start-App }
    "stop" { Stop-App }
    "status" { Show-Status }
    "open" { Open-App }
    default { throw "Unsupported command: $Command" }
}
