param(
    [string]$Text = "",
    [string]$TextFile = "scripts\v3-native-benchmark-text.txt",
    [string]$Voice = "",
    [string]$RefAudio = "",
    [string]$Style = "tu_nhien",
    [string]$Output = "outputs\vieneu-v3-native-benchmark.wav",
    [string]$Log = "outputs\vieneu-v3-native-benchmark.log",
    [int]$MaxNewFrames = 300,
    [float]$Temperature = 0.8,
    [int]$TopK = 25,
    [float]$TopP = 0.95,
    [int]$MaxChars = 384,
    [int]$Threads = 4,
    [string]$ModelDir = ".models\vieneu-v3-turbo-native",
    [string]$CodecDir = "",
    [string]$OnnxDir = "",
    [string]$VoicesJson = "",
    [string]$OnnxRuntimeRoot = "",
    [string]$LlamaDir = "third_party\llama.cpp",
    [switch]$NoBuild,
    [switch]$NoBenchmark,
    [switch]$UseDirectLinear,
    [switch]$UseGgmlLinear,
    [switch]$DisableGgmlHeads,
    [switch]$NoFuseFfn,
    [switch]$DisableQ8Ffn,
    [switch]$NoDenoiseRef,
    [switch]$NoRefCodes,
    [switch]$SkipDownload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$RepoRoot = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $RepoRoot "build"
$CliCandidates = @(
    (Join-Path $RepoRoot "build\vieneu-tts-cli.exe"),
    (Join-Path $RepoRoot "build\Release\vieneu-tts-cli.exe")
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

function Test-AssetIntegrity([string]$Path) {
    if ([System.IO.Path]::GetExtension($Path) -ne ".npz") {
        return $true
    }

    try {
        Add-Type -AssemblyName System.IO.Compression | Out-Null
        $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
        try {
            $zip = [System.IO.Compression.ZipArchive]::new($stream, [System.IO.Compression.ZipArchiveMode]::Read)
            try {
                $buffer = [byte[]]::new(65536)
                foreach ($entry in $zip.Entries) {
                    $entryStream = $entry.Open()
                    try {
                        while ($entryStream.Read($buffer, 0, $buffer.Length) -gt 0) {
                        }
                    } finally {
                        $entryStream.Dispose()
                    }
                }
            } finally {
                $zip.Dispose()
            }
        } finally {
            $stream.Dispose()
        }
        return $true
    } catch {
        Write-Host "Integrity check failed for ${Path}: $($_.Exception.Message)"
        return $false
    }
}

function Get-RemoteFileSize([string]$Url) {
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if (!$curl) {
        return -1
    }

    $headers = & $curl.Source -L -I --silent --show-error $Url
    if ($LASTEXITCODE -ne 0) {
        return -1
    }

    $linked = $headers | Select-String -Pattern '^\s*x-linked-size:\s*(\d+)\s*$' | Select-Object -Last 1
    if ($linked) {
        return [int64]$linked.Matches[0].Groups[1].Value
    }

    $content = $headers | Select-String -Pattern '^\s*content-length:\s*(\d+)\s*$' | Select-Object -Last 1
    if ($content) {
        return [int64]$content.Matches[0].Groups[1].Value
    }
    return -1
}

function Download-File([string]$Url, [string]$Destination) {
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source -L --fail --retry 5 --retry-delay 2 --connect-timeout 30 -C - -o $Destination $Url
        if ($LASTEXITCODE -eq 0) {
            return
        }

        Write-Host "Resume failed, retrying from scratch with curl..."
        if (Test-Path $Destination) {
            Remove-Item -LiteralPath $Destination -Force
        }
        & $curl.Source -L --fail --retry 5 --retry-delay 2 --connect-timeout 30 -o $Destination $Url
        if ($LASTEXITCODE -eq 0) {
            return
        }
        throw "curl failed to download $Url"
    }

    Invoke-WebRequest -Uri $Url -OutFile $Destination
}

function Ensure-OptionalFile([string]$Url, [string]$Destination) {
    try {
        Ensure-File $Url $Destination
    } catch {
        Write-Host "Optional asset unavailable: $Url" -ForegroundColor Yellow
        Write-Host "  $($_.Exception.Message)" -ForegroundColor DarkYellow
    }
}

function Ensure-File([string]$Url, [string]$Destination) {
    if (Test-Path $Destination) {
        $localSize = (Get-Item -LiteralPath $Destination).Length
        if ($localSize -gt 0) {
            Write-Host "Found: $Destination"
            return
        }
    }

    if ($SkipDownload) {
        throw "Missing required file and -SkipDownload was set: $Destination"
    }

    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null

    $remoteSize = Get-RemoteFileSize $Url
    if (Test-Path $Destination) {
        $localSize = (Get-Item -LiteralPath $Destination).Length
        if ($remoteSize -gt 0 -and $localSize -eq $remoteSize) {
            if (!(Test-AssetIntegrity $Destination)) {
                Write-Host "Local file has expected size but failed integrity check, redownloading: $Destination"
                Remove-Item -LiteralPath $Destination -Force
            } else {
                Write-Host "Found: $Destination ($localSize bytes)"
                return
            }
        }
        if ((Test-Path $Destination) -and $remoteSize -gt 0 -and $localSize -gt $remoteSize) {
            Write-Host "Local file is larger than remote metadata, redownloading: $Destination"
            Remove-Item -LiteralPath $Destination -Force
        } elseif (Test-Path $Destination) {
            Write-Host "Resuming: $Destination ($localSize / $remoteSize bytes)"
        }
    } else {
        Write-Host "Downloading: $Url"
        Write-Host "        to: $Destination"
    }

    Download-File $Url $Destination

    if ($remoteSize -gt 0) {
        $finalSize = (Get-Item -LiteralPath $Destination).Length
        if ($finalSize -ne $remoteSize) {
            throw "Downloaded size mismatch for $Destination. Expected $remoteSize bytes, got $finalSize bytes."
        }
    }

    if (!(Test-AssetIntegrity $Destination)) {
        Write-Host "Redownloading from scratch after failed integrity check: $Destination"
        Remove-Item -LiteralPath $Destination -Force
        Download-File $Url $Destination
        if ($remoteSize -gt 0) {
            $finalSize = (Get-Item -LiteralPath $Destination).Length
            if ($finalSize -ne $remoteSize) {
                throw "Downloaded size mismatch for $Destination. Expected $remoteSize bytes, got $finalSize bytes."
            }
        }
        if (!(Test-AssetIntegrity $Destination)) {
            throw "Downloaded file failed integrity check: $Destination"
        }
    }
}

function Ensure-NativeAssets([string]$ResolvedModelDir, [string]$ResolvedCodecDir, [string]$ResolvedVoicesJson) {
    $required = @(
        (Join-Path $ResolvedModelDir "config.json"),
        (Join-Path $ResolvedModelDir "tokenizer.json"),
        (Join-Path $ResolvedModelDir "speaker_encoder.onnx"),
        $ResolvedVoicesJson,
        (Join-Path $ResolvedModelDir "backbone.gguf"),
        (Join-Path $ResolvedModelDir "vieneu_v3_heads.npz"),
        (Join-Path $ResolvedModelDir "acoustic\vieneu_acoustic_weights.npz"),
        (Join-Path $ResolvedCodecDir "moss_audio_tokenizer_decode_full.onnx"),
        (Join-Path $ResolvedCodecDir "moss_audio_tokenizer_decode_shared.data"),
        (Join-Path $ResolvedCodecDir "moss_audio_tokenizer_encode.onnx"),
        (Join-Path $ResolvedCodecDir "moss_audio_tokenizer_encode.data")
    )

    $missing = @($required | Where-Object { !(Test-Path $_) })
    if ($missing.Count -eq 0) {
        return
    }

    if ($SkipDownload) {
        throw "Missing native benchmark assets and -SkipDownload was set:`n$($missing -join "`n")"
    }

    Write-Step "Downloading missing native benchmark assets from Hugging Face..."
    $hfCppBase = "https://huggingface.co/lastudio-community/VieNeu-TTS-v3-Turbo-CPP/resolve/main"
    $hfTurboBase = "https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo/resolve/main"
    $hfUpdateBase = "$hfTurboBase/update"

    Ensure-File "$hfUpdateBase/config.json" (Join-Path $ResolvedModelDir "config.json")
    Ensure-File "$hfUpdateBase/tokenizer.json" (Join-Path $ResolvedModelDir "tokenizer.json")
    Ensure-File "$hfTurboBase/speaker_encoder.onnx" (Join-Path $ResolvedModelDir "speaker_encoder.onnx")
    Ensure-File "$hfTurboBase/voices_v3_turbo.json" $ResolvedVoicesJson
    Ensure-OptionalFile "$hfTurboBase/denoiser.onnx" (Join-Path $ResolvedModelDir "denoiser.onnx")
    Ensure-OptionalFile "$hfUpdateBase/model.safetensors" (Join-Path $ResolvedModelDir "update\model.safetensors")

    Ensure-File "$hfCppBase/backbone.gguf" (Join-Path $ResolvedModelDir "backbone.gguf")
    Ensure-File "$hfCppBase/vieneu_v3_heads.npz" (Join-Path $ResolvedModelDir "vieneu_v3_heads.npz")
    Ensure-File "$hfCppBase/acoustic/vieneu_acoustic_weights.npz" (Join-Path $ResolvedModelDir "acoustic\vieneu_acoustic_weights.npz")
    
    Ensure-File "$hfCppBase/codec/moss_audio_tokenizer_decode_full.onnx" (Join-Path $ResolvedCodecDir "moss_audio_tokenizer_decode_full.onnx")
    Ensure-File "$hfCppBase/codec/moss_audio_tokenizer_decode_shared.data" (Join-Path $ResolvedCodecDir "moss_audio_tokenizer_decode_shared.data")
    Ensure-File "$hfCppBase/codec/moss_audio_tokenizer_encode.onnx" (Join-Path $ResolvedCodecDir "moss_audio_tokenizer_encode.onnx")
    Ensure-File "$hfCppBase/codec/moss_audio_tokenizer_encode.data" (Join-Path $ResolvedCodecDir "moss_audio_tokenizer_encode.data")
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
$OutputPath = Resolve-RepoPath $Output
$LogPath = Resolve-RepoPath $Log
$ResolvedTextFile = Resolve-RepoPath $TextFile
if ([string]::IsNullOrWhiteSpace($Text)) {
    if (!(Test-Path $ResolvedTextFile)) {
        throw "Text is empty and default text file is missing: $ResolvedTextFile"
    }
    $Text = (Get-Content -LiteralPath $ResolvedTextFile -Encoding UTF8 -Raw).Trim()
}

Push-Location $RepoRoot
try {
    Write-Step "Preparing native benchmark"
    Ensure-Built $CliExe
    Copy-RuntimeDlls $CliExe
    Ensure-NativeAssets $ResolvedModelDir $ResolvedCodecDir $ResolvedVoicesJson
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath), (Split-Path -Parent $LogPath) | Out-Null

    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    $argsList = @(
        "--profile", "vieneu-v3-native",
        "--model-dir", $ResolvedModelDir,
        "--codec-dir", $ResolvedCodecDir,
        "--voices-json", $ResolvedVoicesJson,
        "--text", $Text,
        "--output", $OutputPath,
        "--temperature", $Temperature.ToString($inv),
        "--top-k", $TopK.ToString($inv),
        "--top-p", $TopP.ToString($inv),
        "--max-new-frames", $MaxNewFrames.ToString($inv),
        "--max-chars", $MaxChars.ToString($inv),
        "--threads", $Threads.ToString($inv),
        "--style", $Style
    )

    if (![string]::IsNullOrWhiteSpace($ResolvedOnnxDir)) {
        $argsList += @("--onnx-dir", $ResolvedOnnxDir)
    }
    if ($Voice.Trim().Length -gt 0) {
        $argsList += @("--voice", $Voice)
    }
    if ($RefAudio.Trim().Length -gt 0) {
        $argsList += @("--ref-audio", (Resolve-RepoPath $RefAudio))
    }
    if ($NoDenoiseRef) {
        $argsList += "--no-denoise-ref"
    }
    if ($NoRefCodes) {
        $argsList += "--no-ref-codes"
    }

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

        Write-Step "Running vieneu-v3-native benchmark"
        $displayArgs = $argsList -join " "
        Write-Host "& $CliExe $displayArgs"
        Write-Host "Log: $LogPath"

        $stdoutPath = Join-Path ([System.IO.Path]::GetTempPath()) ("vieneu-v3-native-stdout-{0}.log" -f ([guid]::NewGuid()))
        $stderrPath = Join-Path ([System.IO.Path]::GetTempPath()) ("vieneu-v3-native-stderr-{0}.log" -f ([guid]::NewGuid()))
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

        $runOutput | Tee-Object -FilePath $LogPath
        Add-Content -LiteralPath $LogPath -Value ""
        Add-Content -LiteralPath $LogPath -Value ("[BenchmarkScript] Total process time: {0:F3}s" -f $stopwatch.Elapsed.TotalSeconds)

        if ($exitCode -ne 0) {
            throw "TTS command failed with exit code $exitCode."
        }

        Write-Step "Done"
        Write-Host ("Total process time: {0:F3}s" -f $stopwatch.Elapsed.TotalSeconds) -ForegroundColor Cyan
        Write-Host "Output WAV: $OutputPath" -ForegroundColor Green
        Write-Host "Benchmark log: $LogPath" -ForegroundColor Green
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


