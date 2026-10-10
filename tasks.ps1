# Windows PowerShell task runner mirroring the Makefile.
# Usage: .\tasks.ps1 <setup|lint|format|test|train> [-Experiment baseline]
param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "lint", "format", "test", "train")]
    [string]$Task = "test",
    [string]$Experiment = "baseline"
)

$ErrorActionPreference = "Stop"
$Python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

function Invoke-Step([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "'$Exe $Arguments' failed with exit code $LASTEXITCODE" }
}

switch ($Task) {
    "setup" {
        Invoke-Step $Python @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-Step $Python @("-m", "pip", "install", "-e", ".[dev]")
        Invoke-Step $Python @("-m", "pre_commit", "install")
    }
    "lint" {
        Invoke-Step $Python @("-m", "ruff", "check", ".")
        Invoke-Step $Python @("-m", "black", "--check", ".")
    }
    "format" {
        Invoke-Step $Python @("-m", "ruff", "check", "--fix", ".")
        Invoke-Step $Python @("-m", "black", ".")
    }
    "test" { Invoke-Step $Python @("-m", "pytest", "-q") }
    "train" { Invoke-Step $Python @("scripts/train.py", "experiment=$Experiment") }
}
