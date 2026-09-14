<#
.SYNOPSIS
  Runs the full experimental matrix for the GSA-TRA paper, sequentially, resumably.

.EXAMPLE
  # everything (5 seeds main configs, 3 seeds ablations) - several days on a laptop GPU
  .\run_all.ps1 -Data D:\data\VoiceBank_DEMAND_16k -Official D:\gtcrn\checkpoints\model_trained_on_vctk.tar

  # quick pass: 3 seeds, skips H=4 and the windowed-attention runs
  .\run_all.ps1 -Data D:\data\VoiceBank_DEMAND_16k -Official ...\model_trained_on_vctk.tar -Quick

  # only some phases (comma separated): official, baseline, h1, h4, zeroinit, ablate, window, eval, gates, ood, collect
  .\run_all.ps1 -Data ... -Official ... -Phases baseline,h1,eval,collect

  # with the out-of-domain sets (each has clean\ and noisy\ subfolders)
  .\run_all.ps1 -Data ... -Official ... -Testsets D:\data\testsets\A_vb_esc50,D:\data\testsets\B_libri_esc50,D:\data\testsets\C_libri_esc50_lowsnr

.NOTES
  Every training run writes <Runs>\<name>\best.pt and is SKIPPED if that file already exists,
  so you can stop the script (Ctrl+C) and re-run the same command to continue.
  Failures are logged to <Runs>\run_all.log and do not stop the remaining runs.

  Training recipe (matches the official GTCRN / SEtrain recipe as closely as VoiceBank allows):
    full-length utterances, batch 16, 200 epochs, lr 1e-3 with 10% linear warmup then cosine to 1e-6,
    loss 70*mag + 30*(re+im) + 0.1*(-SI-SDR dB), no magnitude floor, fp32.
  With ~10.8k training files that is ~135k optimiser steps (SEtrain: 250k). Use -Epochs to change.
#>
param(
    [Parameter(Mandatory = $true)] [string] $Data,
    [string] $Official = "",
    [string] $Runs = "runs",
    [string[]] $Testsets = @(),
    [int] $Epochs = 200,
    [int] $Batch = 16,
    [double] $Segment = 0,
    [int] $Window = 64,
    [string[]] $Phases = @("official", "baseline", "h1", "h4", "zeroinit", "ablate", "window", "eval", "gates", "ood", "collect"),
    [switch] $Quick,
    [switch] $Amp
)

$ErrorActionPreference = "Continue"

# `powershell -File x.ps1 -Phases a,b,c` delivers ONE string "a,b,c" (not an array); normalise both forms
$Phases   = @($Phases   | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })
$Testsets = @($Testsets | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })

# refuse to start on bad paths - otherwise every phase "succeeds" in a second
if (-not (Test-Path (Join-Path $Data "clean_trainset_28spk_wav"))) {
    Write-Host "ERROR: -Data '$Data' has no clean_trainset_28spk_wav folder. Pass the 16 kHz VoiceBank+DEMAND root." -ForegroundColor Red
    exit 1
}
if ($Official -and -not (Test-Path $Official)) {
    Write-Host "ERROR: -Official '$Official' not found." -ForegroundColor Red
    exit 1
}
foreach ($ts in $Testsets) {
    if (-not (Test-Path (Join-Path $ts "clean"))) { Write-Host "ERROR: testset '$ts' has no clean\ folder." -ForegroundColor Red; exit 1 }
}

$seedsMain = if ($Quick) { 0, 1, 2 } else { 0, 1, 2, 3, 4 }
$seedsAbl  = 0, 1, 2
if ($Quick) { $Phases = $Phases | Where-Object { $_ -notin @("h4", "window") } }

New-Item -ItemType Directory -Force -Path $Runs | Out-Null
$log = Join-Path $Runs "run_all.log"
function Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $log -Value $line
}
Log ("phases: {0} | seeds: {1} | data: {2} | official: {3}" -f ($Phases -join ","), ($seedsMain -join ","), $Data, $(if ($Official) { $Official } else { "-" }))
function Run($what, $cmd) {
    Log "START $what"
    Log "  > $cmd"
    $t0 = Get-Date
    Invoke-Expression $cmd | Out-Host          # stream python's stdout to the terminal (it is also in <out>\log.txt)
    $ok = ($LASTEXITCODE -eq 0)
    $dt = (Get-Date) - $t0
    Log ("{0} {1} after {2:hh\:mm\:ss}" -f ($(if ($ok) { "DONE " } else { "FAILED" }), $what, $dt))
}

$common = "--data `"$Data`" --epochs $Epochs --batch_size $Batch --segment $Segment --lr 1e-3 --min_lr 1e-6 --warmup_frac 0.1 --w_time 0.1 --w_complex 100 --mag_floor 0 --no_learnable_erb"
if ($Amp) { $common += " --amp" }

function Train($name, $extra, $seed) {
    $out = Join-Path $Runs $name
    if (Test-Path (Join-Path $out "best.pt")) { Log "SKIP  $name (best.pt exists)"; return }
    $resume = ""
    if (Test-Path (Join-Path $out "last.pt")) { $resume = " --resume `"$(Join-Path $out 'last.pt')`"" }
    Run "train $name" "python train.py $common --out `"$out`" --seed $seed $extra$resume"
}

# ---------------------------------------------------------------- phases
$officialDir = Join-Path $Runs "official"
if ("official" -in $Phases) {
    if ($Official -and (Test-Path $Official)) {
        New-Item -ItemType Directory -Force -Path $officialDir | Out-Null
        if (-not (Test-Path (Join-Path $officialDir "test_metrics.csv"))) {
            Run "evaluate official checkpoint" "python evaluate.py --data `"$Data`" --ckpt `"$Official`" --official --csv `"$(Join-Path $officialDir 'test_metrics.csv')`" --macs_seconds 10"
        } else { Log "SKIP  official evaluation (exists)" }
    } else { Log "WARN  -Official not given or not found; skipping the released-checkpoint reference" }
}

if ("baseline" -in $Phases) { foreach ($s in $seedsMain) { Train "base_s$s"    "--baseline" $s } }
if ("h1"       -in $Phases) { foreach ($s in $seedsMain) { Train "h1_s$s"      "--heads 1" $s } }
if ("h4"       -in $Phases) { foreach ($s in $seedsMain) { Train "h4_s$s"      "--heads 4" $s } }
if ("zeroinit" -in $Phases) { foreach ($s in $seedsMain) { Train "h1zero_s$s"  "--heads 1 --zero_init_out" $s } }
if ("ablate"   -in $Phases) {
    foreach ($s in $seedsAbl) {
        Train "seonly_s$s"   "--no_mhtra" $s                       # + SE only
        Train "attnonly_s$s" "--heads 1 --no_se" $s                # + attention only (H=1)
        Train "h2_s$s"       "--heads 2" $s                        # head-count sweep
    }
}
if ("window"   -in $Phases) { foreach ($s in $seedsAbl) { Train "h1win${Window}_s$s" "--heads 1 --attn_window $Window" $s } }

# ---------------------------------------------------------------- evaluation
$allRuns = Get-ChildItem -Path $Runs -Directory | Where-Object { Test-Path (Join-Path $_.FullName "best.pt") }
if ("eval" -in $Phases) {
    foreach ($r in $allRuns) {
        $csv = Join-Path $r.FullName "test_metrics.csv"
        if (Test-Path $csv) { Log "SKIP  eval $($r.Name) (exists)"; continue }
        # paired against the baseline with the SAME seed, plus the official checkpoint
        $seed = ($r.Name -split "_s")[-1]
        $cmp = @()
        $b = Join-Path (Join-Path $Runs "base_s$seed") "test_metrics.csv"
        if ((Test-Path $b) -and ($r.Name -notlike "base_*")) { $cmp += "`"$b`"" }
        $o = Join-Path $officialDir "test_metrics.csv"
        if (Test-Path $o) { $cmp += "`"$o`"" }
        $cmpArg = if ($cmp.Count) { " --compare " + ($cmp -join " ") } else { "" }
        Run "eval $($r.Name)" "python evaluate.py --data `"$Data`" --ckpt `"$(Join-Path $r.FullName 'best.pt')`" --macs_seconds 10$cmpArg"
    }
}

# ---------------------------------------------------------------- gate statistics (mode A / B diagnostic)
if ("gates" -in $Phases) {
    $utt = Get-ChildItem (Join-Path $Data "noisy_testset_wav") -Filter *.wav | Select-Object -First 1
    if ($utt) {
        foreach ($r in $allRuns) {
            if ($r.Name -like "base_*" -or $r.Name -like "seonly_*") { continue }
            $csv = Join-Path $r.FullName "gate_stats.csv"
            if (Test-Path $csv) { continue }
            Run "gates $($r.Name)" "python gate_stats.py `"$(Join-Path $r.FullName 'best.pt')`" `"$($utt.FullName)`" --csv `"$csv`""
        }
    }
}

# ---------------------------------------------------------------- out-of-domain sets (all seeds, not just seed 0)
if ("ood" -in $Phases -and $Testsets.Count) {
    foreach ($ts in $Testsets) {
        $tsName = Split-Path $ts -Leaf
        foreach ($r in $allRuns) {
            if ($r.Name -notlike "base_*" -and $r.Name -notlike "h1_*" -and $r.Name -notlike "h1zero_*") { continue }
            $csv = Join-Path $r.FullName "$tsName.csv"
            if (Test-Path $csv) { continue }
            $seed = ($r.Name -split "_s")[-1]
            $b = Join-Path (Join-Path $Runs "base_s$seed") "$tsName.csv"
            $cmpArg = if ((Test-Path $b) -and ($r.Name -notlike "base_*")) { " --compare `"$b`"" } else { "" }
            Run "ood $tsName / $($r.Name)" "python evaluate_any.py --root `"$ts`" --ckpt `"$(Join-Path $r.FullName 'best.pt')`" --csv `"$csv`"$cmpArg"
        }
        if ($Official -and (Test-Path $Official)) {
            $csv = Join-Path $officialDir "$tsName.csv"
            if (-not (Test-Path $csv)) {
                Run "ood $tsName / official" "python evaluate_any.py --root `"$ts`" --ckpt `"$Official`" --official --csv `"$csv`""
            }
        }
    }
}

if ("collect" -in $Phases) {
    Run "collect results" "python collect_results.py --runs `"$Runs`" --out `"$(Join-Path $Runs 'summary.csv')`""
}
Log "ALL PHASES FINISHED - see $(Join-Path $Runs 'summary.csv') and $log"