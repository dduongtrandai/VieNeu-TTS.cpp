param(
    [string]$Text = "",
    [string]$TextFile = "scripts\v3-native-benchmark-text.txt",
    [ValidateSet("all", "example", "example_2", "example_3", "example_4")]
    [string]$Ref = "all",
    [string]$OutputDir = "outputs\voice-clone",
    [string]$LogDir = "outputs\voice-clone\logs",
    [int]$MaxNewFrames = 180,
    [float]$Temperature = 0.8,
    [int]$TopK = 25,
    [float]$TopP = 0.95,
    [int]$MaxChars = 384,
    [int]$Threads = 4,
    [string]$ModelDir = ".models\vieneu-v3-turbo",
    [string]$CodecDir = "",
    [string]$OnnxDir = ".models\vieneu-v3-turbo",
    [string]$VoicesJson = "",
    [string]$OnnxRuntimeRoot = "",
    [string]$LlamaDir = "third_party\llama.cpp",
    [switch]$NoBuild,
    [switch]$NoBenchmark,
    [switch]$UseDirectLinear,
    [switch]$UseGgmlLinear,
    [switch]$DisableGgmlHeads,
    [switch]$NoFuseFfn,
    [switch]$DisableQ8Ffn
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $RepoRoot "build-check"
$RefRoot = Join-Path $RepoRoot "examples\audio_ref"
$ManifestPath = Join-Path $RefRoot "voice_clone_refs.json"
$CliCandidates = @(
    (Join-Path $RepoRoot "build\vieneu-tts-cli.exe"),
    (Join-Path $RepoRoot "build\Release\vieneu-tts-cli.exe"),
    (Join-Path $RepoRoot "build-check\vieneu-tts-cli.exe"),
    (Join-Path $RepoRoot "build-check\Release\vieneu-tts-cli.exe")
)

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Resolve-RepoPath([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return Join-Path $RepoRoot $Path
}

function Find-CMake {
    $cmd = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $vsCMake = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if (Test-Path $vsCMake) {
        return $vsCMake
    }

    throw "Cannot find cmake.exe. Install CMake or Visual Studio CMake tools, then rerun this script."
}

function Find-OnnxRuntimeRoot {
    if (![string]::IsNullOrWhiteSpace($OnnxRuntimeRoot)) {
        return (Resolve-RepoPath $OnnxRuntimeRoot)
    }

    $ortSdkDir = Join-Path $RepoRoot "ort_sdk"
    if (Test-Path $ortSdkDir) {
        $candidate = Get-ChildItem -LiteralPath $ortSdkDir -Directory -Filter "onnxruntime-win-x64-*" -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            Select-Object -First 1
        if ($candidate) {
            return $candidate.FullName
        }
    }

    return ""
}

function Find-Cli {
    if (!$NoBuild) {
        return (Join-Path $BuildDir "Release\vieneu-tts-cli.exe")
    }
    foreach ($path in $CliCandidates) {
        if (Test-Path $path) {
            return $path
        }
    }
    return (Join-Path $BuildDir "Release\vieneu-tts-cli.exe")
}

function Ensure-Built([string]$CliExe) {
    if ($NoBuild -and (Test-Path $CliExe)) {
        return
    }
    if ($NoBuild) {
        throw "CLI is missing and -NoBuild was set: $CliExe"
    }

    $cmake = Find-CMake
    $resolvedLlamaDir = Resolve-RepoPath $LlamaDir
    $resolvedOrtRoot = Find-OnnxRuntimeRoot
    if (!(Test-Path (Join-Path $resolvedLlamaDir "CMakeLists.txt"))) {
        throw "llama.cpp was not found at $resolvedLlamaDir."
    }

    $configureArgs = @(
        "-S", $RepoRoot,
        "-B", $BuildDir,
        "-G", "Visual Studio 17 2022",
        "-A", "x64",
        "-DVIENEU_LLAMA_DIR=$resolvedLlamaDir"
    )

    if (![string]::IsNullOrWhiteSpace($resolvedOrtRoot)) {
        $configureArgs += "-DONNXRUNTIME_ROOT=$resolvedOrtRoot"
    }

    Write-Step "Configuring runtime"
    & $cmake @configureArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed."
    }

    Write-Step "Building vieneu-tts-cli"
    & $cmake --build $BuildDir --config Release --target vieneu-tts-cli
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed."
    }
}

function Copy-RuntimeDlls([string]$CliExe) {
    $exeDir = Split-Path -Parent $CliExe
    New-Item -ItemType Directory -Force -Path $exeDir | Out-Null

    $runtimeDirs = @(
        (Join-Path $BuildDir "bin\Release"),
        (Join-Path $BuildDir "bin"),
        (Join-Path $BuildDir "Release"),
        $BuildDir,
        (Join-Path $exeDir "bin\Release"),
        (Join-Path $exeDir "bin"),
        (Join-Path $exeDir "Release"),
        $exeDir
    ) | Select-Object -Unique

    foreach ($runtimeDir in $runtimeDirs) {
        if (Test-Path $runtimeDir) {
            Get-ChildItem -LiteralPath $runtimeDir -Filter "*.dll" -File -ErrorAction SilentlyContinue | ForEach-Object {
                if ($_.DirectoryName -ne $exeDir) {
                    Copy-Item -LiteralPath $_.FullName -Destination $exeDir -Force
                }
            }
        }
    }

    $resolvedOrt = Find-OnnxRuntimeRoot
    if (![string]::IsNullOrWhiteSpace($resolvedOrt) -and (Test-Path $resolvedOrt)) {
        foreach ($ortLib in @(
            (Join-Path $resolvedOrt "lib"),
            (Join-Path $resolvedOrt "bin"),
            (Join-Path $resolvedOrt "runtimes\win-x64\native")
        )) {
            if (Test-Path $ortLib) {
                foreach ($pattern in @("onnxruntime.dll", "onnxruntime_providers*.dll", "DirectML.dll", "openvino*.dll", "tbb*.dll")) {
                    Get-ChildItem -LiteralPath $ortLib -Filter $pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
                        if ($_.DirectoryName -ne $exeDir) {
                            Copy-Item -LiteralPath $_.FullName -Destination $exeDir -Force
                        }
                    }
                }
            }
        }
    }
}

function Assert-NativeAssets([string]$ResolvedModelDir, [string]$ResolvedCodecDir) {
    $required = @(
        (Join-Path $ResolvedModelDir "config.json"),
        (Join-Path $ResolvedModelDir "tokenizer.json"),
        (Join-Path $ResolvedModelDir "voices_v3_turbo.json"),
        (Join-Path $ResolvedModelDir "backbone.gguf"),
        (Join-Path $ResolvedModelDir "vieneu_v3_heads.npz"),
        (Join-Path $ResolvedModelDir "acoustic\vieneu_acoustic_weights.npz"),
        (Join-Path $ResolvedCodecDir "moss_audio_tokenizer_decode_full.onnx"),
        (Join-Path $ResolvedCodecDir "moss_audio_tokenizer_encode.onnx")
    )

    $missing = @($required | Where-Object { !(Test-Path $_) })
    if ($missing.Count -gt 0) {
        throw "Missing native voice cloning assets:`n$($missing -join "`n")`nExport native assets first, for example with scripts\export-v3-native-assets.py."
    }
}

function Quote-ProcessArg([string]$Value) {
    if ($null -eq $Value) {
        return '""'
    }
    if ($Value.Length -eq 0) {
        return '""'
    }
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    return '"' + ($Value -replace '\\(?=\\*")', '$0$0' -replace '"', '\"') + '"'
}

if (!(Test-Path $ManifestPath)) {
    throw "Voice clone reference manifest is missing: $ManifestPath"
}

$manifest = Get-Content -LiteralPath $ManifestPath -Encoding UTF8 -Raw | ConvertFrom-Json
$refs = @($manifest.refs)
if ($Ref -ne "all") {
    $refs = @($refs | Where-Object { $_.id -eq $Ref })
}
if ($refs.Count -eq 0) {
    throw "No voice reference matched: $Ref"
}

$CliExe = Find-Cli
$ResolvedModelDir = Resolve-RepoPath $ModelDir
$ResolvedOnnxDir = Resolve-RepoPath $OnnxDir
if ([string]::IsNullOrWhiteSpace($CodecDir)) {
    $ResolvedCodecDir = Join-Path $ResolvedModelDir "codec"
} else {
    $ResolvedCodecDir = Resolve-RepoPath $CodecDir
}
if ([string]::IsNullOrWhiteSpace($VoicesJson)) {
    $ResolvedVoicesJson = Join-Path $ResolvedModelDir "voices_v3_turbo.json"
} else {
    $ResolvedVoicesJson = Resolve-RepoPath $VoicesJson
}
$ResolvedOutputDir = Resolve-RepoPath $OutputDir
$ResolvedLogDir = Resolve-RepoPath $LogDir
$ResolvedTextFile = Resolve-RepoPath $TextFile
$TextSource = "inline -Text"
if ([string]::IsNullOrWhiteSpace($Text)) {
    if (!(Test-Path $ResolvedTextFile)) {
        throw "Text is empty and default text file is missing: $ResolvedTextFile"
    }
    $Text = (Get-Content -LiteralPath $ResolvedTextFile -Encoding UTF8 -Raw).Trim()
    $TextSource = $ResolvedTextFile
}

Push-Location $RepoRoot
try {
    Write-Step "Preparing native voice cloning benchmark"
    Ensure-Built $CliExe
    Copy-RuntimeDlls $CliExe
    Assert-NativeAssets $ResolvedModelDir $ResolvedCodecDir
    New-Item -ItemType Directory -Force -Path $ResolvedOutputDir, $ResolvedLogDir | Out-Null

    $oldBenchmark = $env:VIENEU_V3_NATIVE_BENCHMARK
    $oldAcousticDirect = $env:VIENEU_ACOUSTIC_DIRECT_LINEAR
    $oldAcousticGgml = $env:VIENEU_ACOUSTIC_GGML_LINEAR
    $oldGgmlHeads = $env:VIENEU_ACOUSTIC_GGML_HEADS
    $oldFuseFfn = $env:VIENEU_GGML_FUSE_FFN
    $oldQ8Ffn = $env:VIENEU_ACOUSTIC_Q8_FFN
    try {
        $env:VIENEU_V3_NATIVE_BENCHMARK = if ($NoBenchmark) { "0" } else { "1" }
        if ($UseGgmlLinear) {
            $env:VIENEU_ACOUSTIC_DIRECT_LINEAR = "0"
            $env:VIENEU_ACOUSTIC_GGML_LINEAR = "1"
        } elseif ($UseDirectLinear) {
            $env:VIENEU_ACOUSTIC_DIRECT_LINEAR = "1"
            $env:VIENEU_ACOUSTIC_GGML_LINEAR = "0"
        }
        $env:VIENEU_ACOUSTIC_GGML_HEADS = if ($DisableGgmlHeads) { "0" } else { "1" }
        $env:VIENEU_GGML_FUSE_FFN = if ($NoFuseFfn) { "0" } else { "1" }
        $env:VIENEU_ACOUSTIC_Q8_FFN = if ($DisableQ8Ffn) { "0" } else { "1" }

        $inv = [System.Globalization.CultureInfo]::InvariantCulture
        $outputs = @()
        $logs = @()
        foreach ($item in $refs) {
            $refAudio = Join-Path $RefRoot ([string]$item.audio)
            if (!(Test-Path $refAudio)) {
                throw "Missing reference audio: $refAudio"
            }

            $outputPath = Join-Path $ResolvedOutputDir ("{0}-clone.wav" -f $item.id)
            $logPath = Join-Path $ResolvedLogDir ("{0}-clone.log" -f $item.id)
            $argsList = @(
                "--profile", "vieneu-v3-native",
                "--model-dir", $ResolvedModelDir,
                "--codec-dir", $ResolvedCodecDir,
                "--voices-json", $ResolvedVoicesJson,
                "--text", $Text,
                "--ref-audio", $refAudio,
                "--output", $outputPath,
                "--temperature", $Temperature.ToString($inv),
                "--top-k", $TopK.ToString($inv),
                "--top-p", $TopP.ToString($inv),
                "--max-new-frames", $MaxNewFrames.ToString($inv),
                "--max-chars", $MaxChars.ToString($inv),
                "--threads", $Threads.ToString($inv)
            )
            if (![string]::IsNullOrWhiteSpace($ResolvedOnnxDir)) {
                $argsList += @("--onnx-dir", $ResolvedOnnxDir)
            }

            Write-Step "Cloning from $($item.audio)"
            Write-Host "Reference transcript: $($item.text)"
            Write-Host "Text source: $TextSource"
            Write-Host "& $CliExe $($argsList -join ' ')"
            Write-Host "Log: $logPath"

            $stdoutPath = Join-Path ([System.IO.Path]::GetTempPath()) ("vieneu-v3-native-clone-stdout-{0}.log" -f ([guid]::NewGuid()))
            $stderrPath = Join-Path ([System.IO.Path]::GetTempPath()) ("vieneu-v3-native-clone-stderr-{0}.log" -f ([guid]::NewGuid()))
            $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            try {
                $processArgs = ($argsList | ForEach-Object { Quote-ProcessArg $_ }) -join " "
                $process = Start-Process -FilePath $CliExe `
                    -ArgumentList $processArgs `
                    -NoNewWindow `
                    -Wait `
                    -PassThru `
                    -RedirectStandardOutput $stdoutPath `
                    -RedirectStandardError $stderrPath
                $exitCode = $process.ExitCode
            } finally {
                $stopwatch.Stop()
            }

            $runOutput = @()
            if (Test-Path $stdoutPath) {
                $runOutput += Get-Content -LiteralPath $stdoutPath -Encoding UTF8
            }
            if (Test-Path $stderrPath) {
                $runOutput += Get-Content -LiteralPath $stderrPath -Encoding UTF8
            }
            Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue

            $runOutput | Tee-Object -FilePath $logPath
            Add-Content -LiteralPath $logPath -Value ""
            Add-Content -LiteralPath $logPath -Value ("[VoiceCloneScript] Reference: {0}" -f $item.audio)
            Add-Content -LiteralPath $logPath -Value ("[VoiceCloneScript] Reference transcript: {0}" -f $item.text)
            Add-Content -LiteralPath $logPath -Value ("[VoiceCloneScript] Text source: {0}" -f $TextSource)
            Add-Content -LiteralPath $logPath -Value ("[VoiceCloneScript] Total process time: {0:F3}s" -f $stopwatch.Elapsed.TotalSeconds)

            if ($exitCode -ne 0) {
                throw "TTS command failed for $($item.id) with exit code $exitCode."
            }
            $outputs += $outputPath
            $logs += $logPath
        }

        Write-Step "Done"
        Write-Host "Generated clone outputs:" -ForegroundColor Green
        $outputs | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
        Write-Host "Benchmark logs:" -ForegroundColor Green
        $logs | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
    } finally {
        $env:VIENEU_V3_NATIVE_BENCHMARK = $oldBenchmark
        $env:VIENEU_ACOUSTIC_DIRECT_LINEAR = $oldAcousticDirect
        $env:VIENEU_ACOUSTIC_GGML_LINEAR = $oldAcousticGgml
        $env:VIENEU_ACOUSTIC_GGML_HEADS = $oldGgmlHeads
        $env:VIENEU_GGML_FUSE_FFN = $oldFuseFfn
        $env:VIENEU_ACOUSTIC_Q8_FFN = $oldQ8Ffn
    }
} finally {
    Pop-Location
}
