$ErrorActionPreference = "Stop"

$script:RefreshExitCode = $null
$script:ObservationExitCode = $null
$script:LivePreflightExitCode = $null
$script:DailyRunStatusExitCode = $null
$script:SkipLivePreflight = $false
$script:FinalStage = "starting"
$script:WrapperError = $null
$script:ObservationStatusPath = $null
$script:DailyRunStatusStatusPath = $null
$script:DailyRunStatusSnapshotPath = $null
$script:LivePreflightStatusPath = $null
$script:DailyPipelinePreflightSmokeExitCode = $null
$script:DailyPipelinePreflightSmokeStatusPath = $null

function Write-DailyResearchWrapperStatus {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StatusPath,
        [Parameter(Mandatory = $true)]
        [datetime]$StartedAt,
        [Parameter(Mandatory = $true)]
        [int]$ExitCode,
        [Parameter(Mandatory = $true)]
        [string]$ProjectRootPath,
        [Parameter(Mandatory = $true)]
        [string]$LockPath,
        [Parameter(Mandatory = $true)]
        [bool]$SkipLivePreflight
    )

    $FinishedAt = Get-Date
    [pscustomobject]@{
        started_at = $StartedAt.ToString("s")
        finished_at = $FinishedAt.ToString("s")
        exit_code = $ExitCode
        final_stage = $script:FinalStage
        project_root = $ProjectRootPath
        lock_path = $LockPath
        refresh_exit_code = $script:RefreshExitCode
        observation_exit_code = $script:ObservationExitCode
        daily_run_status_exit_code = $script:DailyRunStatusExitCode
        observation_status_path = $script:ObservationStatusPath
        daily_run_status_status_path = $script:DailyRunStatusStatusPath
        daily_run_status_snapshot_path = $script:DailyRunStatusSnapshotPath
        live_preflight_exit_code = $script:LivePreflightExitCode
        live_preflight_status_path = $script:LivePreflightStatusPath
        pipeline_preflight_smoke_exit_code = $script:DailyPipelinePreflightSmokeExitCode
        pipeline_preflight_smoke_status_path = $script:DailyPipelinePreflightSmokeStatusPath
        wrapper_error = $script:WrapperError
        skip_live_preflight = $SkipLivePreflight
        wrapper_status_path = $StatusPath
    } | ConvertTo-Json -Depth 3 | Set-Content -Path $StatusPath -Encoding UTF8
}

function Invoke-LivePreflight {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PipelineSnapshot,
        [Parameter(Mandatory = $true)]
        [string]$ProjectRootPath,
        [Parameter(Mandatory = $true)]
        [string]$LogDir,
        [Parameter(Mandatory = $true)]
        [string]$OutputDir,
        [string]$DailyRunStatusSnapshotPath,
        [switch]$FailOnBlocked
    )

    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutLog = Join-Path $LogDir "live_preflight_${Stamp}.out.log"
    $ErrLog = Join-Path $LogDir "live_preflight_${Stamp}.err.log"
    $StatusPath = Join-Path $LogDir "live_preflight_${Stamp}.status.json"
    $StartedAt = Get-Date

    $ArgsList = @(
        "-m", "quant_etf_lab", "live-preflight",
        "--pipeline-snapshot", $PipelineSnapshot,
        "--output-dir", $OutputDir
    )
    if ($DailyRunStatusSnapshotPath) {
        $ArgsList += @("--daily-run-status-snapshot", $DailyRunStatusSnapshotPath)
    }
    if ($FailOnBlocked) {
        $ArgsList += "--fail-on-blocked"
    }

    try {
        Push-Location $ProjectRootPath
        & python @ArgsList 1> $OutLog 2> $ErrLog
        $PreflightExitCode = $LASTEXITCODE
    } catch {
        $_ | Out-String | Set-Content -Path $ErrLog -Encoding UTF8
        $PreflightExitCode = 1
    } finally {
        Pop-Location
    }

    $FinishedAt = Get-Date
    $CommandText = "python -m quant_etf_lab live-preflight --pipeline-snapshot $PipelineSnapshot --output-dir $OutputDir"
    if ($DailyRunStatusSnapshotPath) {
        $CommandText += " --daily-run-status-snapshot $DailyRunStatusSnapshotPath"
    }
    if ($FailOnBlocked) {
        $CommandText += " --fail-on-blocked"
    }
    [pscustomobject]@{
        started_at = $StartedAt.ToString("s")
        finished_at = $FinishedAt.ToString("s")
        exit_code = $PreflightExitCode
        project_root = $ProjectRootPath
        pipeline_snapshot = $PipelineSnapshot
        output_dir = $OutputDir
        daily_run_status_snapshot = $DailyRunStatusSnapshotPath
        stdout_log = $OutLog
        stderr_log = $ErrLog
        command = $CommandText
    } | ConvertTo-Json -Depth 3 | Set-Content -Path $StatusPath -Encoding UTF8

    $script:LivePreflightStatusPath = $StatusPath
    return $PreflightExitCode
}

function Invoke-DailyPipelinePreflightSmoke {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRootPath,
        [Parameter(Mandatory = $true)]
        [string]$LogDir,
        [Parameter(Mandatory = $false)]
        [string]$Stamp
    )

    $Stamp = if ($Stamp) { $Stamp } else { Get-Date -Format "yyyyMMdd_HHmmss" }
    $OutLog = Join-Path $LogDir "pipeline_preflight_smoke_${Stamp}.out.log"
    $ErrLog = Join-Path $LogDir "pipeline_preflight_smoke_${Stamp}.err.log"
    $StatusPath = Join-Path $LogDir "pipeline_preflight_smoke_${Stamp}.status.json"
    $ScriptPath = Join-Path $PSScriptRoot "run_daily_pipeline_preflight_smoke.ps1"
    $StartedAt = Get-Date

    try {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $ScriptPath 1> $OutLog 2> $ErrLog
        $SmokeExitCode = $LASTEXITCODE
    } catch {
        $_ | Out-String | Set-Content -Path $ErrLog -Encoding UTF8
        $SmokeExitCode = 1
    }

    $FinishedAt = Get-Date
    [pscustomobject]@{
        started_at = $StartedAt.ToString("s")
        finished_at = $FinishedAt.ToString("s")
        exit_code = $SmokeExitCode
        project_root = $ProjectRootPath
        stdout_log = $OutLog
        stderr_log = $ErrLog
        script_path = $ScriptPath
        command = "powershell -NoProfile -ExecutionPolicy Bypass -File $ScriptPath"
    } | ConvertTo-Json -Depth 3 | Set-Content -Path $StatusPath -Encoding UTF8

    $script:DailyPipelinePreflightSmokeStatusPath = $StatusPath
    return $SmokeExitCode
}

function Invoke-DailyRunStatusSummary {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Reason,
        [Parameter(Mandatory = $true)]
        [string]$ProjectRootPath,
        [Parameter(Mandatory = $true)]
        [string]$LogDir
    )

    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutLog = Join-Path $LogDir "daily_run_status_${Stamp}.out.log"
    $ErrLog = Join-Path $LogDir "daily_run_status_${Stamp}.err.log"
    $StatusPath = Join-Path $LogDir "daily_run_status_${Stamp}.status.json"
    $StartedAt = Get-Date

    try {
        Push-Location $ProjectRootPath
        & python -m quant_etf_lab daily-run-status 1> $OutLog 2> $ErrLog
        $StatusExitCode = $LASTEXITCODE
    } catch {
        $_ | Out-String | Set-Content -Path $ErrLog -Encoding UTF8
        $StatusExitCode = 1
    } finally {
        Pop-Location
    }

    $FinishedAt = Get-Date
    $snapshotPath = (Join-Path $ProjectRootPath "outputs\research\daily_run_status_latest\daily_run_status_snapshot.json")
    [pscustomobject]@{
        started_at = $StartedAt.ToString("s")
        finished_at = $FinishedAt.ToString("s")
        exit_code = $StatusExitCode
        reason = $Reason
        project_root = $ProjectRootPath
        stdout_log = $OutLog
        stderr_log = $ErrLog
        report_path = (Join-Path $ProjectRootPath "outputs\research\daily_run_status_latest\daily_run_status.md")
        snapshot_path = $snapshotPath
        command = "python -m quant_etf_lab daily-run-status"
        daily_run_status_snapshot_path = $snapshotPath
    } | ConvertTo-Json -Depth 3 | Set-Content -Path $StatusPath -Encoding UTF8

    $script:DailyRunStatusStatusPath = $StatusPath
    $script:DailyRunStatusSnapshotPath = $snapshotPath
    return $StatusExitCode
}

function Resolve-LatestPipelineSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRootPath,
        [string]$DateStamp
    )

    $ResearchRoot = Join-Path $ProjectRootPath "outputs\research"
    $CandidateSnapshots = @()

    if ($DateStamp) {
        $CandidateSnapshots += Join-Path $ResearchRoot ("daily_pipeline_" + $DateStamp)
    }
    $CandidateSnapshots += Join-Path $ResearchRoot "daily_pipeline_latest"

    foreach ($Candidate in $CandidateSnapshots) {
        $CandidatePath = Join-Path $Candidate "daily_pipeline_snapshot.json"
        if (Test-Path -LiteralPath $CandidatePath) {
            return (Resolve-Path -LiteralPath $CandidatePath).Path
        }
    }

    $Fallback = Get-ChildItem -LiteralPath $ResearchRoot -Directory -Filter "daily_pipeline*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        ForEach-Object {
            $Snapshot = Join-Path $_.FullName "daily_pipeline_snapshot.json"
            if (Test-Path -LiteralPath $Snapshot) {
                Get-Item -LiteralPath $Snapshot
            }
        } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($Fallback) {
        return $Fallback.FullName
    }
    return $null
}

function Invoke-DailyPipelineRefresh {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRootPath,
        [string[]]$RefreshArgs = @(),
        [Parameter(Mandatory = $true)]
        [string]$LogDir,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $RefreshScript = Join-Path $PSScriptRoot "run_daily_research_refresh_logged.ps1"
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutLog = Join-Path $LogDir "daily_research_refresh_${Label}_${Stamp}.out.log"
    $ErrLog = Join-Path $LogDir "daily_research_refresh_${Label}_${Stamp}.err.log"
    $CommandArgs = @("-ExecutionPolicy", "Bypass", "-File", $RefreshScript) + $RefreshArgs
    try {
        Push-Location $ProjectRootPath
        & powershell @CommandArgs 1> $OutLog 2> $ErrLog
        return $LASTEXITCODE
    } catch {
        $_ | Out-String | Set-Content -Path $ErrLog -Encoding UTF8
        return 1
    } finally {
        Pop-Location
    }
}

function Pipeline-BlockedForPaperAccount {
    param([string]$PipelineSnapshotPath)

    if (-not (Test-Path -LiteralPath $PipelineSnapshotPath)) {
        return $false
    }
    $Snapshot = Get-Content -Path $PipelineSnapshotPath -Raw | ConvertFrom-Json
    return (
        [string]($Snapshot.pipeline_next_step_stage) -eq "retry_paper_account" -or
        [string]($Snapshot.pipeline_blocker) -eq "paper_account_step_failed"
    )
}

function Invoke-DailyResearchRefreshWithObservation {
    param(
        [string[]]$RawArgs
    )

    $LogDir = Join-Path $ProjectRoot "outputs\logs"
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    Set-Location $ProjectRoot
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $env:PYTHONIOENCODING = "utf-8"

    $ObservationDate = $null
    $SkipObservation = $false
    $SkipLivePreflight = $false
    $script:SkipLivePreflight = $false
    $LivePreflightFailOnBlocked = $false
    $LivePreflightOutputDir = $null
    $RunPipelinePreflightSmoke = $false
    $RetryPaperAccount = $false
    $RefreshArgs = @()
    $DateStamp = if ($ObservationDate) { $ObservationDate } else { Get-Date -Format "yyyyMMdd" }

    for ($Index = 0; $Index -lt $RawArgs.Count; $Index++) {
        $Arg = $RawArgs[$Index]
        if ($Arg -eq "--observation-date") {
            if ($Index + 1 -ge $RawArgs.Count) {
                throw "--observation-date requires YYYYMMDD."
            }
            $ObservationDate = $RawArgs[$Index + 1]
            $Index++
        } elseif ($Arg -eq "--skip-observation") {
            $SkipObservation = $true
        } elseif ($Arg -eq "--skip-live-preflight") {
            $SkipLivePreflight = $true
            $script:SkipLivePreflight = $true
        } elseif ($Arg -eq "--live-preflight-fail-on-blocked") {
            $LivePreflightFailOnBlocked = $true
        } elseif ($Arg -eq "--run-daily-pipeline-preflight-smoke") {
            $RunPipelinePreflightSmoke = $true
        } elseif ($Arg -eq "--retry-paper-account") {
            $RetryPaperAccount = $true
        } elseif ($Arg -eq "--live-preflight-output-dir") {
            if ($Index + 1 -ge $RawArgs.Count) {
                throw "--live-preflight-output-dir requires a directory."
            }
            $LivePreflightOutputDir = $RawArgs[$Index + 1]
            $Index++
        } else {
            $RefreshArgs += $Arg
        }
    }

    if ($ObservationDate -and $ObservationDate -notmatch "^\d{8}$") {
        throw "--observation-date must be YYYYMMDD."
    }

    $RefreshScript = Join-Path $PSScriptRoot "run_daily_research_refresh_logged.ps1"
    $RefreshExitCode = Invoke-DailyPipelineRefresh `
        -ProjectRootPath ([string]$ProjectRoot) `
        -RefreshArgs $RefreshArgs `
        -LogDir $LogDir `
        -Label "base"
    $script:RefreshExitCode = $RefreshExitCode
    if ($RefreshExitCode -ne 0) {
        $script:FinalStage = "refresh_failed"
        $script:DailyRunStatusExitCode = Invoke-DailyRunStatusSummary -Reason "refresh_failed" -ProjectRootPath ([string]$ProjectRoot) -LogDir $LogDir
        return $RefreshExitCode
    }

    $PipelineSnapshot = Resolve-LatestPipelineSnapshot -ProjectRootPath ([string]$ProjectRoot) -DateStamp $DateStamp
    if (-not $PipelineSnapshot) {
        throw "No daily_pipeline_snapshot.json found after refresh."
    }

    if ($RetryPaperAccount -and (Pipeline-BlockedForPaperAccount -PipelineSnapshotPath $PipelineSnapshot)) {
        Write-Host "Pipeline indicates paper-account failure. Retrying daily refresh with --skip-market-cap-update."
        $RetryArgs = $RefreshArgs
        if ($RetryArgs -notcontains "--skip-market-cap-update") {
            $RetryArgs += "--skip-market-cap-update"
        }
        $RetryExitCode = Invoke-DailyPipelineRefresh `
            -ProjectRootPath ([string]$ProjectRoot) `
            -RefreshArgs $RetryArgs `
            -LogDir $LogDir `
            -Label "retry"
        if ($RetryExitCode -ne 0) {
            $script:FinalStage = "retry_paper_account_failed"
            $script:RefreshExitCode = $RetryExitCode
            $script:DailyRunStatusExitCode = Invoke-DailyRunStatusSummary -Reason "paper_account_retry_failed" -ProjectRootPath ([string]$ProjectRoot) -LogDir $LogDir
            return $RetryExitCode
        }
        $RefreshExitCode = $RetryExitCode
        $script:RefreshExitCode = $RetryExitCode
        $PipelineSnapshot = Resolve-LatestPipelineSnapshot -ProjectRootPath ([string]$ProjectRoot) -DateStamp $DateStamp
        if ($PipelineSnapshot -and (Pipeline-BlockedForPaperAccount -PipelineSnapshotPath $PipelineSnapshot)) {
            Write-Host "Warning: paper-account blocker persists after one retry; continue with observation and/or live-preflight."
        }
    }

    if ($RunPipelinePreflightSmoke) {
        $script:DailyPipelinePreflightSmokeExitCode = Invoke-DailyPipelinePreflightSmoke -ProjectRootPath ([string]$ProjectRoot) -LogDir $LogDir
        if ($script:DailyPipelinePreflightSmokeExitCode -ne 0) {
            $script:FinalStage = "daily_pipeline_preflight_smoke_failed"
            return $script:DailyPipelinePreflightSmokeExitCode
        }
    }

    if ($SkipObservation) {
        $StatusExitCode = Invoke-DailyRunStatusSummary -Reason "observation_skipped" -ProjectRootPath ([string]$ProjectRoot) -LogDir $LogDir
        $script:DailyRunStatusExitCode = $StatusExitCode
        if ($StatusExitCode -ne 0) {
            $script:FinalStage = "daily_run_status_failed"
            return $StatusExitCode
        }
        $script:FinalStage = "observation_skipped"
        return 0
    }

    if (-not $PipelineSnapshot) {
        $PipelineSnapshot = Resolve-LatestPipelineSnapshot -ProjectRootPath ([string]$ProjectRoot) -DateStamp $DateStamp
    }
    if (-not $PipelineSnapshot) {
        throw "No daily_pipeline_snapshot.json found after refresh."
    }

    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $ObservationDir = Join-Path $ProjectRoot "outputs\research\allocator_observation_$DateStamp"
    $OutLog = Join-Path $LogDir "allocator_observation_${Stamp}.out.log"
    $ErrLog = Join-Path $LogDir "allocator_observation_${Stamp}.err.log"
    $StatusPath = Join-Path $LogDir "allocator_observation_${Stamp}.status.json"
    $script:ObservationStatusPath = $StatusPath
    $StartedAt = Get-Date

    try {
        & python -m quant_etf_lab allocator-observation --pipeline-snapshot $PipelineSnapshot --output-dir $ObservationDir --baseline-label "Rollback quality-v2" 1> $OutLog 2> $ErrLog
        $ObservationExitCode = $LASTEXITCODE
    } catch {
        $_ | Out-String | Set-Content -Path $ErrLog -Encoding UTF8
        $ObservationExitCode = 1
    }
    $script:ObservationExitCode = $ObservationExitCode

    $FinishedAt = Get-Date
    [pscustomobject]@{
        started_at = $StartedAt.ToString("s")
        finished_at = $FinishedAt.ToString("s")
        exit_code = $ObservationExitCode
        project_root = [string]$ProjectRoot
        pipeline_snapshot = $PipelineSnapshot
        observation_dir = $ObservationDir
        stdout_log = $OutLog
        stderr_log = $ErrLog
        command = "python -m quant_etf_lab allocator-observation --pipeline-snapshot $PipelineSnapshot --output-dir $ObservationDir --baseline-label `"Rollback quality-v2`""
    } | ConvertTo-Json -Depth 3 | Set-Content -Path $StatusPath -Encoding UTF8

    $StatusExitCode = Invoke-DailyRunStatusSummary -Reason "after_observation" -ProjectRootPath ([string]$ProjectRoot) -LogDir $LogDir
    $script:DailyRunStatusExitCode = $StatusExitCode
    if (-not $SkipLivePreflight -and $StatusExitCode -ne 0) {
        $script:FinalStage = "daily_run_status_failed"
        return $StatusExitCode
    }

    if (-not $SkipLivePreflight) {
        if (-not $LivePreflightOutputDir) {
            $LivePreflightOutputDir = Join-Path $ProjectRoot "outputs\research\live_preflight_$DateStamp"
        }
        $InvokeLivePreflightArgs = @{
            PipelineSnapshot = $PipelineSnapshot
            ProjectRootPath = [string]$ProjectRoot
            LogDir = $LogDir
            OutputDir = $LivePreflightOutputDir
            DailyRunStatusSnapshotPath = $script:DailyRunStatusSnapshotPath
        }
        if ($LivePreflightFailOnBlocked) {
            $LivePreflightExitCode = Invoke-LivePreflight @InvokeLivePreflightArgs -FailOnBlocked
        } else {
            $LivePreflightExitCode = Invoke-LivePreflight @InvokeLivePreflightArgs
        }
        $script:LivePreflightExitCode = $LivePreflightExitCode
        if ($LivePreflightExitCode -ne 0) {
            $script:FinalStage = if ($LivePreflightFailOnBlocked) { "live_preflight_blocked" } else { "live_preflight_failed" }
            return $LivePreflightExitCode
        }
    }

    if ($ObservationExitCode -eq 0 -and $StatusExitCode -ne 0) {
        $script:FinalStage = "daily_run_status_failed"
        return $StatusExitCode
    }
    if ($ObservationExitCode -ne 0) {
        $script:FinalStage = "observation_failed"
    } else {
        $script:FinalStage = "completed"
    }
    return $ObservationExitCode
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$LogDir = Join-Path $ProjectRoot "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$WrapperStartedAt = Get-Date
$WrapperStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$WrapperStatusPath = Join-Path $LogDir "daily_research_refresh_with_observation_${WrapperStamp}.status.json"
$LockDir = Join-Path $ProjectRoot "outputs\locks"
New-Item -ItemType Directory -Force -Path $LockDir | Out-Null
$LockPath = Join-Path $LockDir "daily_research_refresh_with_observation.lock"
$LockStream = $null
$ExitCode = 1

try {
    try {
        $LockStream = [System.IO.File]::Open(
            $LockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    } catch [System.IO.IOException] {
        [Console]::Error.WriteLine("Another daily research refresh with observation is already running. Lock: $LockPath")
        $script:FinalStage = "lock_conflict"
        $ExitCode = 75
    }

    if ($null -ne $LockStream) {
        $LockStream.SetLength(0)
        $LockPayload = "pid=$PID`nstarted_at=$((Get-Date).ToString("s"))`nscript=$PSCommandPath`n"
        $LockBytes = [System.Text.Encoding]::UTF8.GetBytes($LockPayload)
        $LockStream.Write($LockBytes, 0, $LockBytes.Length)
        $LockStream.Flush()

        $ExitCode = Invoke-DailyResearchRefreshWithObservation -RawArgs $args
    }
} catch {
    $script:WrapperError = ($_ | Out-String).Trim()
    $script:FinalStage = "wrapper_failed"
    [Console]::Error.WriteLine($script:WrapperError)
    $ExitCode = 1
} finally {
    if ($null -ne $LockStream) {
        $LockStream.Dispose()
        Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
    }
    Write-DailyResearchWrapperStatus `
        -StatusPath $WrapperStatusPath `
        -StartedAt $WrapperStartedAt `
        -ExitCode $ExitCode `
        -ProjectRootPath ([string]$ProjectRoot) `
        -LockPath $LockPath `
        -SkipLivePreflight $script:SkipLivePreflight
}

exit $ExitCode
