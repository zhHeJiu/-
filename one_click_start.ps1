$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Set-Location $PSScriptRoot

$Port = if ($env:PORT) { $env:PORT } else { "8011" }
$HostName = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
$VenvDir = Join-Path $PSScriptRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $PSScriptRoot "requirements.txt"
$RequirementStamp = Join-Path $VenvDir ".requirements.sha256"
$EnvFile = Join-Path $PSScriptRoot ".env"
$EnvExample = Join-Path $PSScriptRoot ".env.example"
$FrontendPath = Join-Path $PSScriptRoot "frontend\index.html"

function Write-Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Warn($Message) {
    Write-Host "!! $Message" -ForegroundColor Yellow
}

function Test-CommandWorks($Command, [string[]]$Arguments = @()) {
    try {
        & $Command @Arguments --version *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Find-SystemPython {
    if (Test-CommandWorks "py" @("-3")) {
        return @{ Command = "py"; Args = @("-3") }
    }
    if (Test-CommandWorks "python") {
        return @{ Command = "python"; Args = @() }
    }
    if (Test-CommandWorks "python3") {
        return @{ Command = "python3"; Args = @() }
    }
    return $null
}

function Invoke-Python($PythonSpec, [string[]]$Arguments) {
    & $($PythonSpec.Command) @($PythonSpec.Args + $Arguments)
    if ($LASTEXITCODE -ne 0) {
        throw "Python 命令执行失败：$($Arguments -join ' ')"
    }
}

function Test-BackendHealth {
    try {
        $response = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/api/health" -TimeoutSec 1
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Open-FrontendWhenReady {
    param(
        [string]$Port,
        [string]$FrontendPath
    )

    Start-Job -ScriptBlock {
        param($JobPort, $JobFrontendPath)
        for ($i = 0; $i -lt 30; $i++) {
            try {
                $response = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$JobPort/api/health" -TimeoutSec 1
                if ($response.StatusCode -eq 200) { break }
            } catch {
                Start-Sleep -Seconds 1
            }
        }
        Start-Process $JobFrontendPath
    } -ArgumentList $Port, $FrontendPath | Out-Null
}

Write-Host "道德问题评估 - 一键启动" -ForegroundColor Green

if (!(Test-Path $EnvFile) -and (Test-Path $EnvExample)) {
    Copy-Item $EnvExample $EnvFile
    Write-Warn "未找到 .env，已从 .env.example 创建。大模型/API 配置请在网页模型配置面板填写。"
}

if (Test-BackendHealth) {
    Write-Step "检测到后端已经在 $Port 端口运行，直接打开前端"
    Start-Process $FrontendPath
    return
}

if (!(Test-Path $VenvPython)) {
    Write-Step "创建本地 Python 虚拟环境 .venv"
    $SystemPython = Find-SystemPython
    if ($null -eq $SystemPython) {
        Write-Host ""
        Write-Host "没有找到可用的 Python。" -ForegroundColor Red
        Write-Host "请先安装 Python 3.11+，并在安装时勾选 Add python.exe to PATH。"
        throw "Python not found"
    }
    Invoke-Python $SystemPython @("-m", "venv", ".venv")
}

$Python = @{ Command = $VenvPython; Args = @() }

$CurrentRequirementsHash = if (Test-Path $Requirements) {
    (Get-FileHash $Requirements -Algorithm SHA256).Hash
} else {
    ""
}
$InstalledRequirementsHash = if (Test-Path $RequirementStamp) {
    (Get-Content $RequirementStamp -Raw).Trim()
} else {
    ""
}

if ($CurrentRequirementsHash -and $CurrentRequirementsHash -ne $InstalledRequirementsHash) {
    Write-Step "安装或更新 Python 依赖"
    Invoke-Python $Python @("-m", "pip", "install", "-r", "requirements.txt")
    Set-Content -Path $RequirementStamp -Value $CurrentRequirementsHash -Encoding ASCII
}

Write-Step "启动后端并自动打开前端"
Write-Host "后端地址：http://127.0.0.1:$Port"
Write-Host "前端文件：$FrontendPath"
Write-Host "停止服务：在此窗口按 Ctrl+C"

Open-FrontendWhenReady -Port $Port -FrontendPath $FrontendPath

Invoke-Python $Python @("-m", "uvicorn", "backend.main:app", "--host", $HostName, "--port", $Port, "--reload")
