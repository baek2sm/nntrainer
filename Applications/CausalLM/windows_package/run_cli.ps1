# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file run_cli.ps1
# @brief Run Qwen3 0.6B with the packaged NNTrainer CausalLM CLI.

param(
    [string]$Prompt = "The capital of France is",
    [int]$Threads = 4
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Bin = Join-Path $Root "bin"
$Model = Join-Path $Root "model"
$Exe = Join-Path $Bin "nntr_causallm.exe"

if (-Not (Test-Path $Exe)) {
    throw "Missing executable: $Exe"
}
if (-Not (Test-Path $Model)) {
    throw "Missing model directory: $Model"
}

$DllDirs = Get-ChildItem $Bin -Filter *.dll -Recurse |
    ForEach-Object { Split-Path -Parent $_.FullName } |
    Sort-Object -Unique

$env:PATH = (($DllDirs + @($Bin)) -join ";") + ";" + $env:PATH
$env:NNTR_NUM_THREADS = [string]$Threads

$Output = & $Exe $Model $Prompt 2>&1
$ExitCode = $LASTEXITCODE

if ($ExitCode -ne 0) {
    $Output
    exit $ExitCode
}

$Text = ($Output | Out-String)
$ThinkEnd = $Text.LastIndexOf("</think>")
if ($ThinkEnd -ge 0) {
    $Text = $Text.Substring($ThinkEnd + "</think>".Length)
}

$EndMarker = $Text.IndexOf("<|im_end|>")
if ($EndMarker -ge 0) {
    $Text = $Text.Substring(0, $EndMarker)
}

$ReplacementMarker = [string][char]0xfffd
$Utf8ReplacementMojibake = ([string][char]0x5360) + ([string][char]0xc3d9) + ([string][char]0xc619)
$LatinReplacementMojibake = ([string][char]0x00ef) + ([string][char]0x00bf) + ([string][char]0x00bd)

$Text = $Text.Replace("<think>", "").Replace("</think>", "")
$Text = $Text.Replace($ReplacementMarker, "")
$Text = $Text.Replace($Utf8ReplacementMojibake, "")
$Text = $Text.Replace($LatinReplacementMojibake, "")
$EscapedModel = [regex]::Escape($Model)
$EscapedPrompt = [regex]::Escape($Prompt)
$Line = $Text -split "\r?\n" |
    ForEach-Object { $_.Trim() } |
    Where-Object {
        $_ -and
        $_ -notmatch "^(=|prefill:|generation:|total:|peak memory:|\[e2e time\]|Max Resident)" -and
        $_ -notmatch "^$EscapedModel([/\\].*)?$" -and
        $_ -notmatch "^$EscapedPrompt$"
    } |
    Select-Object -First 1

if ($Line) {
    Write-Output $Line
} else {
    $Output
}
