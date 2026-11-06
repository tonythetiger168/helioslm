<#
distill_local_3070.ps1 - Qwen3.6-35B-A3B as a local distillation teacher on
an 8 GB laptop GPU (RTX 3070 Laptop, 40 GB RAM, Windows). v5.252.

llama-server serves the 4-bit GGUF with the MoE experts in system RAM and
everything else on the GPU; examples/distill_generate.py talks to it as a
self-hosted ("vllm") OpenAI-compatible endpoint, so the licence gate, the
thinking switch (chat_template_kwargs), resume and token stats all work as
on a rented pod. Being on 127.0.0.1, it also passes the medical-data guard
(v5.215): medical prompts never leave the machine.

Needs, already installed: llama.cpp with CUDA (llama-server on PATH or
-LlamaServer), Python 3 (standard library only for this step), and the
GGUF below (22.1 GB; -Download fetches it with the `hf` CLI).

    # 1. smoke test: 5 prompts, prints tokens/s
    powershell -ExecutionPolicy Bypass -File examples\distill_local_3070.ps1 `
        -Prompts runs\distill\prompts.jsonl -Out runs\distill\qwen36_smoke.jsonl -Limit 5
    # 2. a real run, stopped after 8 hours; re-running resumes
    powershell -ExecutionPolicy Bypass -File examples\distill_local_3070.ps1 `
        -Prompts runs\distill\prompts.jsonl -Out runs\distill\qwen36_answers.jsonl -MaxHours 8

Tuning (see docs/LOCAL_RUN.md section 7): -NCpuMoe N keeps only the experts
of the first N layers in RAM (default -1: all of them); lower N is faster
while it still fits in 8 GB. -Parallel sets concurrent requests (each gets
-CtxPerSlot tokens of context).
#>
param(
    [Parameter(Mandatory = $true)][string]$Prompts,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$ModelDir = "models\qwen3.6-35b-a3b",
    [string]$Quant = "UD-Q4_K_M",
    [string]$LlamaServer = "llama-server",
    [string]$Python = "python",
    [int]$Port = 8000,
    [int]$NCpuMoe = -1,
    [int]$Parallel = 2,
    [int]$CtxPerSlot = 8192,
    [int]$MaxTokens = 6144,
    [int]$Threads = 0,
    [int]$Limit = 0,
    [double]$MaxHours = 0,
    [switch]$NoThinking,
    [string]$DataManifest = "",
    [switch]$Medical,
    [switch]$Download
)
$ErrorActionPreference = "Stop"

# Pinned: the teacher alias in distill_gen.TEACHERS (licence row
# Qwen/Qwen3.6-35B-A3B, Apache-2.0) and the GGUF repo revision whose file
# list was read on 2026-11-05 (base_model Qwen/Qwen3.6-35B-A3B, Apache-2.0).
$Teacher = "qwen3.6-35b-a3b"
$ServedModel = "Qwen/Qwen3.6-35B-A3B"
$GgufRepo = "unsloth/Qwen3.6-35B-A3B-GGUF"
$GgufRevision = "a483e9e6cbd595906af30beda3187c2663a1118c"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$GgufName = "Qwen3.6-35B-A3B-$Quant.gguf"
$Gguf = Join-Path $ModelDir $GgufName

if (-not (Get-Command $LlamaServer -ErrorAction SilentlyContinue)) {
    throw "llama-server not found: install llama.cpp (CUDA build) or pass -LlamaServer <path>"
}
if (-not (Test-Path $Prompts)) { throw "prompt file not found: $Prompts" }
if (-not (Test-Path $Gguf)) {
    $dl = "hf download $GgufRepo $GgufName --revision $GgufRevision --local-dir $ModelDir"
    if (-not $Download) {
        throw "model not found: $Gguf`nDownload it (22 GB) with:`n  $dl`nor re-run with -Download"
    }
    if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
        throw "the hf CLI is missing (pip install -U huggingface_hub), or download by hand:`n  $dl"
    }
    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
    & hf download $GgufRepo $GgufName --revision $GgufRevision --local-dir $ModelDir
    if ($LASTEXITCODE -ne 0) { throw "download failed" }
}

$logDir = Join-Path (Split-Path -Parent $Out) "llama_server_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logOut = Join-Path $logDir "server_$stamp.out.log"
$logErr = Join-Path $logDir "server_$stamp.err.log"

$serverArgs = @(
    "-m", $Gguf, "--alias", $ServedModel,
    "--host", "127.0.0.1", "--port", "$Port",
    "-c", "$($CtxPerSlot * $Parallel)", "--parallel", "$Parallel",
    "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
    "--jinja", "--reasoning-format", "deepseek"
)
if ($NCpuMoe -lt 0) { $serverArgs += "--cpu-moe" }
elseif ($NCpuMoe -gt 0) { $serverArgs += @("--n-cpu-moe", "$NCpuMoe") }
if ($Threads -gt 0) { $serverArgs += @("-t", "$Threads") }

Write-Host "starting llama-server ($GgufName, logs in $logDir)"
$server = Start-Process -FilePath $LlamaServer -ArgumentList $serverArgs -PassThru `
    -NoNewWindow -RedirectStandardOutput $logOut -RedirectStandardError $logErr
try {
    # loading 22 GB from disk takes a few minutes on a laptop SSD
    $deadline = (Get-Date).AddMinutes(15)
    while ($true) {
        if ($server.HasExited) {
            Get-Content $logErr -Tail 30 | Write-Host
            throw "llama-server exited while loading (code $($server.ExitCode)); log above"
        }
        try {
            $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://127.0.0.1:$Port/health"
            if ($r.StatusCode -eq 200) { break }
        } catch { }
        if ((Get-Date) -gt $deadline) { throw "llama-server not ready after 15 minutes" }
        Start-Sleep -Seconds 5
    }
    Write-Host "llama-server ready on port $Port"

    $genArgs = @(
        "examples\distill_generate.py", "generate",
        "--prompts", $Prompts, "--out", $Out,
        "--teacher", $Teacher, "--endpoint", "vllm",
        "--base-url", "http://127.0.0.1:$Port/v1",
        "--max-tokens", "$MaxTokens", "--workers", "$Parallel"
    )
    if ($NoThinking) { $genArgs += "--no-thinking" }
    if ($Limit -gt 0) { $genArgs += @("--limit", "$Limit") }
    if ($MaxHours -gt 0) { $genArgs += @("--max-hours", "$MaxHours") }
    if ($DataManifest) { $genArgs += @("--data-manifest", $DataManifest) }
    if ($Medical) { $genArgs += "--medical" }

    $t0 = Get-Date
    & $Python @genArgs
    if ($LASTEXITCODE -ne 0) { throw "distill_generate.py generate failed (exit $LASTEXITCODE)" }
    $secs = ((Get-Date) - $t0).TotalSeconds

    & $Python examples\distill_generate.py stats --answers $Out
    # output tokens of the whole file / this run's wall time: an upper bound
    # when the file already held answers from an earlier (resumed) run
    $tok = 0
    Get-Content $Out -Encoding UTF8 | ForEach-Object {
        $u = ($_ | ConvertFrom-Json).usage
        if ($u -and $u.output_tokens) { $tok += [int]$u.output_tokens }
    }
    if ($secs -gt 0) {
        Write-Host ("{0} output tokens in the file; this run took {1:N0} s; about {2:N1} tokens/s" -f `
            $tok, $secs, ($tok / $secs))
    }
}
finally {
    if (-not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
        Write-Host "llama-server stopped"
    }
}
