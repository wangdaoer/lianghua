[CmdletBinding()]
param(
    [switch]$SkipPreflightSmoke,
    [switch]$SkipObservation,
    [switch]$SkipLivePreflight,
    [switch]$RetryPaperAccount,
    [switch]$FailOnBlocked,
    [switch]$FailOnProblemState,
    [switch]$StatusOnly,
    [string]$ObservationDate,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RefreshArgs = @()
)

$ErrorActionPreference = "Stop"

function Invoke-LoggedNativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Command
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$LogDir = Join-Path $ProjectRoot "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StartedAt = Get-Date
$StatusPath = Join-Path $LogDir "unified_daily_workflow_${Stamp}.status.json"
$WorkflowOutLog = Join-Path $LogDir "unified_daily_workflow_${Stamp}.workflow.out.log"
$WorkflowErrLog = Join-Path $LogDir "unified_daily_workflow_${Stamp}.workflow.err.log"
$DailyStatusOutLog = Join-Path $LogDir "unified_daily_workflow_${Stamp}.daily_status.out.log"
$DailyStatusErrLog = Join-Path $LogDir "unified_daily_workflow_${Stamp}.daily_status.err.log"
$DailyRunStatusSnapshotPath = Join-Path $ProjectRoot "outputs\research\daily_run_status_latest\daily_run_status_snapshot.json"
$DailyRunStatusReportPath = Join-Path $ProjectRoot "outputs\research\daily_run_status_latest\daily_run_status.md"

$WorkflowExitCode = $null
$DailyRunStatusExitCode = $null
$FinalExitCode = 1
$FinalStage = "starting"
$WrapperStatusPath = $null
$WrapperCommandText = $null
$DailyStatusCommandText = $null

try {
    if ($StatusOnly) {
        $WorkflowExitCode = 0
        $FinalStage = "status_only"
    } else {
        $WrapperScript = Join-Path $PSScriptRoot "run_daily_research_refresh_with_observation.ps1"
        $WrapperArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $WrapperScript)

        if (-not $SkipPreflightSmoke) {
            $WrapperArgs += "--run-daily-pipeline-preflight-smoke"
        }
        if ($ObservationDate) {
            $WrapperArgs += @("--observation-date", $ObservationDate)
        }
        if ($SkipObservation) {
            $WrapperArgs += "--skip-observation"
        }
        if ($SkipLivePreflight) {
            $WrapperArgs += "--skip-live-preflight"
        }
        if ($RetryPaperAccount) {
            $WrapperArgs += "--retry-paper-account"
        }
        if ($FailOnBlocked) {
            $WrapperArgs += "--live-preflight-fail-on-blocked"
        }
        if ($RefreshArgs) {
            $WrapperArgs += $RefreshArgs
        }

        $WrapperCommandText = "powershell " + ($WrapperArgs -join " ")
        $WorkflowExitCode = Invoke-LoggedNativeCommand -Command {
            & powershell @WrapperArgs 1> $WorkflowOutLog 2> $WorkflowErrLog
        }
    }

    $DailyStatusArgs = @("-m", "quant_etf_lab", "daily-run-status")
    $DailyStatusCommandText = "python -m quant_etf_lab daily-run-status"
    if ($FailOnProblemState) {
        $DailyStatusArgs += "--fail-on-problem-state"
        $DailyStatusCommandText += " --fail-on-problem-state"
    }

    $DailyRunStatusExitCode = Invoke-LoggedNativeCommand -Command {
        & python @DailyStatusArgs 1> $DailyStatusOutLog 2> $DailyStatusErrLog
    }

    $LatestWrapperStatus = Get-ChildItem -LiteralPath $LogDir -Filter "daily_research_refresh_with_observation_*.status.json" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($LatestWrapperStatus) {
        $WrapperStatusPath = $LatestWrapperStatus.FullName
    }

    if ($WorkflowExitCode -ne 0) {
        $FinalStage = "workflow_failed"
        $FinalExitCode = $WorkflowExitCode
    } elseif ($DailyRunStatusExitCode -ne 0) {
        $FinalStage = "daily_run_status_failed"
        $FinalExitCode = $DailyRunStatusExitCode
    } else {
        if ($FinalStage -ne "status_only") {
            $FinalStage = "completed"
        }
        $FinalExitCode = 0
    }
} catch {
    $FinalStage = "wrapper_failed"
    $FinalExitCode = 1
    $_ | Out-String | Set-Content -Path $WorkflowErrLog -Encoding UTF8
} finally {
    $FinishedAt = Get-Date
    [pscustomobject]@{
        schema_version = "quant-unified-daily-workflow-v1"
        started_at = $StartedAt.ToString("s")
        finished_at = $FinishedAt.ToString("s")
        final_stage = $FinalStage
        exit_code = $FinalExitCode
        workflow_exit_code = $WorkflowExitCode
        daily_run_status_exit_code = $DailyRunStatusExitCode
        project_root = [string]$ProjectRoot
        status_only = [bool]$StatusOnly
        skip_preflight_smoke = [bool]$SkipPreflightSmoke
        skip_observation = [bool]$SkipObservation
        skip_live_preflight = [bool]$SkipLivePreflight
        retry_paper_account = [bool]$RetryPaperAccount
        fail_on_blocked = [bool]$FailOnBlocked
        fail_on_problem_state = [bool]$FailOnProblemState
        observation_date = $ObservationDate
        refresh_args = $RefreshArgs
        wrapper_command = $WrapperCommandText
        wrapper_stdout_log = $WorkflowOutLog
        wrapper_stderr_log = $WorkflowErrLog
        wrapper_status_path = $WrapperStatusPath
        daily_run_status_command = $DailyStatusCommandText
        daily_run_status_stdout_log = $DailyStatusOutLog
        daily_run_status_stderr_log = $DailyStatusErrLog
        daily_run_status_snapshot_path = $DailyRunStatusSnapshotPath
        daily_run_status_report_path = $DailyRunStatusReportPath
        unified_status_path = $StatusPath
    } | ConvertTo-Json -Depth 5 | Set-Content -Path $StatusPath -Encoding UTF8
}

exit $FinalExitCode
