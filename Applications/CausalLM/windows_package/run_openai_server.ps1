# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file run_openai_server.ps1
# @brief Start the packaged NNTrainer OpenAI-compatible CausalLM server.

param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [string]$ModelId = "qwen3-0.6b",
    [int]$Threads = 4,
    [switch]$EnableThinking
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Bin = Join-Path $Root "bin"
$Model = Join-Path $Root "model"
$Exe = Join-Path $Bin "nntr_causallm_openai_server.exe"

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

$Args = @(
    $Model,
    "--host", $HostAddress,
    "--port", [string]$Port,
    "--model", $ModelId
)

if ($EnableThinking) {
    $Args += "--enable-thinking"
}

& $Exe @Args
