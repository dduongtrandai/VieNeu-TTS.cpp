param(
    [string]$Text = "Xin chao, day la bai kiem tra VieNeu TTS v3 Turbo tu runtime C++.",
    [string]$Voice = "",
    [string]$RefAudio = "",
    [string]$Output = "outputs\vieneu-v3-test.wav",
    [int]$MaxNewFrames = 300,
    [float]$Temperature = 0.8,
    [int]$TopK = 25,
    [float]$TopP = 0.95,
    [string]$AssetRoot = ".models\vieneu-v3-turbo",
    [switch]$SkipDownload,
    [switch]$NoBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$RepoRoot = $PSScriptRoot
$BuildDir = Join-Path $RepoRoot "build-check"
$ExeDir = Join-Path $BuildDir "Release"
$CliExe = Join-Path $ExeDir "vieneu-tts-cli.exe"
$OrtRoot = Join-Path $RepoRoot "ort_sdk\onnxruntime-win-x64-1.20.1"
$LlamaDir = Join-Path $RepoRoot "llama.cpp"
$ModelDir = Join-Path $RepoRoot $AssetRoot
$OnnxDir = Join-Path $ModelDir "onnx"
$CodecDir = Join-Path $ModelDir "codec"
$VoicesJson = Join-Path $ModelDir "voices_v3_turbo.json"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
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

function Ensure-File([string]$Url, [string]$Destination) {
    if ($SkipDownload) {
        if (Test-Path $Destination) {
            if (!(Test-AssetIntegrity $Destination)) {
                throw "Existing file failed integrity check and -SkipDownload was set: $Destination"
            }
            Write-Host "Found: $Destination"
            return
        }
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

function Ensure-Built {
    if ((Test-Path $CliExe) -and $NoBuild) {
        return
    }
    if ((Test-Path $CliExe) -and (Test-Path (Join-Path $ExeDir "vieneu-tts.dll"))) {
        Write-Host "Found existing CLI: $CliExe"
        return
    }
    if ($NoBuild) {
        throw "CLI is missing and -NoBuild was set: $CliExe"
    }

    if (!(Test-Path (Join-Path $OrtRoot "include\onnxruntime_c_api.h"))) {
        throw "ONNX Runtime SDK was not found at $OrtRoot. Run .\build-local.ps1 once, or place the SDK there."
    }
    if (!(Test-Path (Join-Path $LlamaDir "CMakeLists.txt"))) {
        throw "llama.cpp was not found at $LlamaDir. Run .\build-local.ps1 once, or clone llama.cpp there."
    }

    $cmake = Find-CMake
    Write-Step "Configuring native runtime"
    & $cmake -S $RepoRoot -B $BuildDir -G "Visual Studio 17 2022" -A x64 `
        "-DONNXRUNTIME_ROOT=$OrtRoot" `
        "-DVIENEU_LLAMA_DIR=$LlamaDir"
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed."
    }

    Write-Step "Building vieneu-tts-cli"
    & $cmake --build $BuildDir --config Release --target vieneu-tts-cli
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed."
    }
}

function Ensure-RuntimeDlls {
    New-Item -ItemType Directory -Force -Path $ExeDir | Out-Null

    $binDir = Join-Path $BuildDir "bin\Release"
    if (Test-Path $binDir) {
        Get-ChildItem $binDir -Filter "*.dll" | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $ExeDir -Force
        }
    }

    $ortLib = Join-Path $OrtRoot "lib"
    foreach ($name in @("onnxruntime.dll", "onnxruntime_providers_shared.dll")) {
        $src = Join-Path $ortLib $name
        if (Test-Path $src) {
            Copy-Item -LiteralPath $src -Destination $ExeDir -Force
        }
    }
}

function Ensure-Assets {
    New-Item -ItemType Directory -Force -Path $ModelDir, $OnnxDir, $CodecDir | Out-Null

    $v3Base = "https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo/resolve/main"
    Ensure-File "$v3Base/config.json" (Join-Path $ModelDir "config.json")
    Ensure-File "$v3Base/tokenizer.json" (Join-Path $ModelDir "tokenizer.json")
    Ensure-File "$v3Base/onnx/vieneu_prefill.onnx" (Join-Path $OnnxDir "vieneu_prefill.onnx")
    Ensure-File "$v3Base/onnx/vieneu_decode_step.onnx" (Join-Path $OnnxDir "vieneu_decode_step.onnx")
    Ensure-File "$v3Base/onnx/vieneu_acoustic_cached.onnx" (Join-Path $OnnxDir "vieneu_acoustic_cached.onnx")
    Ensure-File "$v3Base/onnx/vieneu_backbone_shared.data" (Join-Path $OnnxDir "vieneu_backbone_shared.data")
    Ensure-File "$v3Base/onnx/vieneu_v3_heads.npz" (Join-Path $OnnxDir "vieneu_v3_heads.npz")

    $codecBase = "https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX/resolve/main"
    Ensure-File "$codecBase/moss_audio_tokenizer_decode_full.onnx" (Join-Path $CodecDir "moss_audio_tokenizer_decode_full.onnx")
    Ensure-File "$codecBase/moss_audio_tokenizer_decode_shared.data" (Join-Path $CodecDir "moss_audio_tokenizer_decode_shared.data")
    Ensure-File "$codecBase/moss_audio_tokenizer_encode.onnx" (Join-Path $CodecDir "moss_audio_tokenizer_encode.onnx")
    Ensure-File "$codecBase/moss_audio_tokenizer_encode.data" (Join-Path $CodecDir "moss_audio_tokenizer_encode.data")

    if (!(Test-Path $VoicesJson)) {
        $localVoices = "E:\dduongtrandai-github\VieNeu-TTS\src\vieneu\assets\voices_v3_turbo.json"
        if (Test-Path $localVoices) {
            Copy-Item -LiteralPath $localVoices -Destination $VoicesJson -Force
            Write-Host "Copied voices: $VoicesJson"
        } else {
            Ensure-File "https://raw.githubusercontent.com/pnnbao97/VieNeu-TTS/main/src/vieneu/assets/voices_v3_turbo.json" $VoicesJson
        }
    } else {
        Write-Host "Found: $VoicesJson"
    }
}

Push-Location $RepoRoot
try {
    Write-Step "Preparing runtime"
    Ensure-Built
    Ensure-RuntimeDlls

    Write-Step "Preparing VieNeu v3 assets"
    Ensure-Assets

    $OutputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
        $Output
    } else {
        Join-Path $RepoRoot $Output
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath) | Out-Null

    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    $argsList = @(
        "--profile", "vieneu-v3-onnx",
        "--model-dir", $ModelDir,
        "--onnx-dir", $OnnxDir,
        "--codec-dir", $CodecDir,
        "--voices-json", $VoicesJson,
        "--text", $Text,
        "--output", $OutputPath,
        "--temperature", $Temperature.ToString($inv),
        "--top-k", $TopK.ToString($inv),
        "--top-p", $TopP.ToString($inv),
        "--max-new-frames", $MaxNewFrames.ToString($inv)
    )
    if ($Voice.Trim().Length -gt 0) {
        $argsList += @("--voice", $Voice)
    }
    if ($RefAudio.Trim().Length -gt 0) {
        $refPath = if ([System.IO.Path]::IsPathRooted($RefAudio)) { $RefAudio } else { Join-Path $RepoRoot $RefAudio }
        $argsList += @("--ref-audio", $refPath)
    }

    Write-Step "Running TTS"
    Write-Host "& $CliExe $($argsList -join ' ')"
    & $CliExe @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "TTS command failed with exit code $LASTEXITCODE."
    }

    Write-Step "Done"
    Write-Host "Output WAV: $OutputPath" -ForegroundColor Green
} finally {
    Pop-Location
}
