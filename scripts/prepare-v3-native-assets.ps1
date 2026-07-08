param(
    [string]$ModelDir = ".models\vieneu-v3-turbo-native",
    [string]$SourceRepo = "D:\code\VieNeu-TTS",
    [string]$Checkpoint = "pnnbao-ump/VieNeu-TTS-v3-Turbo",
    [string]$Python = "",
    [string]$VenvDir = ".venv",
    [string]$HfToken = "",
    [switch]$SkipDownload,
    [switch]$SkipCodec,
    [switch]$SkipDependencyCheck,
    [switch]$InstallPythonDeps,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$RepoRoot = Split-Path -Parent $PSScriptRoot

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

function Download-File([string]$Url, [string]$Destination, [bool]$Optional = $false) {
    if ($Force -and (Test-Path $Destination)) {
        Remove-Item -LiteralPath $Destination -Force
    }
    if ((Test-Path $Destination) -and !$Force) {
        $existing = Get-Item -LiteralPath $Destination
        if ($existing.Length -gt 0) {
            Write-Host "Found: $Destination"
            return $true
        }
    }
    if ($SkipDownload) {
        if ($Optional) {
            Write-Host "Optional file missing and -SkipDownload was set: $Destination" -ForegroundColor Yellow
            return $false
        }
        throw "Missing required file and -SkipDownload was set: $Destination"
    }

    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Write-Host "Downloading: $Url"
    Write-Host "        to: $Destination"

    $headers = @{}
    if (![string]::IsNullOrWhiteSpace($HfToken)) {
        $headers["Authorization"] = "Bearer $HfToken"
    }

    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl -and [string]::IsNullOrWhiteSpace($HfToken)) {
        if (Test-Path $Destination) {
            & $curl.Source -L --fail --retry 5 --retry-delay 2 --connect-timeout 30 -C - -o $Destination $Url
            if ($LASTEXITCODE -eq 0) {
                return $true
            }
            Remove-Item -LiteralPath $Destination -Force
        }
        & $curl.Source -L --fail --retry 5 --retry-delay 2 --connect-timeout 30 -o $Destination $Url
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        if ($Optional) {
            Write-Host "Optional download failed: $Url" -ForegroundColor Yellow
            return $false
        }
        throw "curl failed to download $Url"
    }

    try {
        Invoke-WebRequest -Uri $Url -OutFile $Destination -Headers $headers
        return $true
    } catch {
        if ($Optional) {
            Write-Host "Optional download failed: $Url" -ForegroundColor Yellow
            Write-Host "  $($_.Exception.Message)" -ForegroundColor DarkYellow
            return $false
        }
        throw
    }
}

function Resolve-PythonCommand([string]$RequestedPython, [string]$ResolvedVenvDir) {
    if (![string]::IsNullOrWhiteSpace($RequestedPython)) {
        return $RequestedPython
    }
    return Join-Path $ResolvedVenvDir "Scripts\python.exe"
}

function Ensure-ProjectVenv([string]$PythonExe, [string]$ResolvedVenvDir) {
    if (Test-Path $PythonExe) {
        return
    }

    Write-Step "Creating project Python virtual environment"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ResolvedVenvDir) | Out-Null

    $basePython = Get-Command python -ErrorAction SilentlyContinue
    if (!$basePython) {
        $basePython = Get-Command py -ErrorAction SilentlyContinue
    }
    if (!$basePython) {
        throw "Cannot find Python. Install Python first, or pass -Python D:\path\to\python.exe."
    }

    if ($basePython.Name -eq "py.exe" -or $basePython.Source -like "*\py.exe") {
        & $basePython.Source -3 -m venv $ResolvedVenvDir
    } else {
        & $basePython.Source -m venv $ResolvedVenvDir
    }
    if ($LASTEXITCODE -ne 0 -or !(Test-Path $PythonExe)) {
        throw "Failed to create project venv at: $ResolvedVenvDir"
    }

    & $PythonExe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip in project venv."
    }
}

function Test-PythonModules([string]$PythonExe, [string[]]$Modules) {
    $code = "import importlib.util,sys; required=['numpy','safetensors','torch','transformers','huggingface_hub']; missing=[m for m in required if importlib.util.find_spec(m) is None]; print(','.join(missing)) if missing else None; sys.exit(1 if missing else 0)"
    $output = & $PythonExe -c $code 2>&1
    $ok = $LASTEXITCODE -eq 0
    if (!$ok -and $output) {
        Write-Host "Missing Python modules: $output" -ForegroundColor Yellow
    }
    return $ok
}

function Assert-ExportPythonDeps([string]$PythonExe, [string]$ResolvedSourceRepo) {
    if ($SkipDependencyCheck) {
        return
    }
    Write-Step "Checking Python export dependencies"
    $required = @("numpy", "safetensors", "torch", "transformers", "huggingface_hub")
    if (Test-PythonModules -PythonExe $PythonExe -Modules $required) {
        Write-Host "Python export dependencies OK: $PythonExe" -ForegroundColor Green
        return
    }

    if ($InstallPythonDeps) {
        Write-Step "Installing missing Python export dependencies"
        & $PythonExe -m pip install numpy safetensors transformers huggingface_hub torch
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Python export dependencies."
        }
        if (Test-PythonModules -PythonExe $PythonExe -Modules $required) {
            Write-Host "Python export dependencies OK: $PythonExe" -ForegroundColor Green
            return
        }
        throw "Python export dependencies are still missing after pip install."
    }

    $pip = "$PythonExe -m pip install numpy safetensors transformers huggingface_hub torch"
    throw @"
Python export environment is missing required modules.

Python used:
  $PythonExe

Install them into the local project venv with:
  $pip

Or let this script install them into the local project venv:
  .\scripts\prepare-v3-native-assets.ps1 -InstallPythonDeps

You can override the Python executable with:
  .\scripts\prepare-v3-native-assets.ps1 -Python D:\path\to\python.exe
"@
}

function Copy-IfExists([string]$Source, [string]$Destination, [bool]$Optional = $false) {
    if (!(Test-Path $Source)) {
        if ($Optional) {
            return $false
        }
        throw "Missing required file: $Source"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
    Write-Host "Copied: $Source -> $Destination"
    return $true
}

function Require-File([string]$Path) {
    if (!(Test-Path $Path)) {
        throw "Required asset was not produced: $Path"
    }
}

$ResolvedModelDir = Resolve-RepoPath $ModelDir
$ResolvedSourceRepo = Resolve-RepoPath $SourceRepo
$ResolvedVenvDir = Resolve-RepoPath $VenvDir
$Python = Resolve-PythonCommand $Python $ResolvedVenvDir
$UpdateDir = Join-Path $ResolvedModelDir "update"
$AcousticDir = Join-Path $ResolvedModelDir "acoustic"
$CodecDir = Join-Path $ResolvedModelDir "codec"

$hfBase = "https://huggingface.co/$Checkpoint/resolve/main"
$hfUpdateBase = "$hfBase/update"
$codecBase = "https://huggingface.co/lastudio-community/VieNeu-TTS-v3-Turbo-CPP/resolve/main/codec"

Push-Location $RepoRoot
try {
    if ([string]::IsNullOrWhiteSpace($PSBoundParameters["Python"])) {
        Ensure-ProjectVenv $Python $ResolvedVenvDir
    }
    Assert-ExportPythonDeps $Python $ResolvedSourceRepo

    Write-Step "Downloading latest v3 Turbo update checkpoint assets"
    New-Item -ItemType Directory -Force -Path $ResolvedModelDir, $UpdateDir, $AcousticDir | Out-Null

    Download-File "$hfUpdateBase/config.json" (Join-Path $UpdateDir "config.json") | Out-Null
    Download-File "$hfUpdateBase/tokenizer.json" (Join-Path $UpdateDir "tokenizer.json") | Out-Null
    Download-File "$hfUpdateBase/model.safetensors" (Join-Path $UpdateDir "model.safetensors") | Out-Null

    Copy-Item -LiteralPath (Join-Path $UpdateDir "config.json") -Destination (Join-Path $ResolvedModelDir "config.json") -Force
    Copy-Item -LiteralPath (Join-Path $UpdateDir "tokenizer.json") -Destination (Join-Path $ResolvedModelDir "tokenizer.json") -Force

    Download-File "$hfBase/speaker_encoder.onnx" (Join-Path $ResolvedModelDir "speaker_encoder.onnx") | Out-Null
    Download-File "$hfBase/denoiser.onnx" (Join-Path $ResolvedModelDir "denoiser.onnx") $true | Out-Null

    $localVoices = Join-Path $ResolvedSourceRepo "src\vieneu\assets\voices_v3_turbo.json"
    if (!(Copy-IfExists $localVoices (Join-Path $ResolvedModelDir "voices_v3_turbo.json") $true)) {
        Download-File "$hfBase/voices_v3_turbo.json" (Join-Path $ResolvedModelDir "voices_v3_turbo.json") | Out-Null
    }

    if (!$SkipCodec) {
        Write-Step "Ensuring MOSS codec assets"
        Download-File "$codecBase/moss_audio_tokenizer_decode_full.onnx" (Join-Path $CodecDir "moss_audio_tokenizer_decode_full.onnx") | Out-Null
        Download-File "$codecBase/moss_audio_tokenizer_decode_shared.data" (Join-Path $CodecDir "moss_audio_tokenizer_decode_shared.data") | Out-Null
        Download-File "$codecBase/moss_audio_tokenizer_encode.onnx" (Join-Path $CodecDir "moss_audio_tokenizer_encode.onnx") | Out-Null
        Download-File "$codecBase/moss_audio_tokenizer_encode.data" (Join-Path $CodecDir "moss_audio_tokenizer_encode.data") | Out-Null
    }

    Write-Step "Exporting semantic backbone GGUF and native heads"
    & $Python scripts\export-v3-native-assets.py `
        --model-dir $ResolvedModelDir `
        --safetensors (Join-Path $UpdateDir "model.safetensors") `
        --config (Join-Path $UpdateDir "config.json")
    if ($LASTEXITCODE -ne 0) {
        throw "export-v3-native-assets.py failed."
    }

    Write-Step "Exporting acoustic decoder weights"
    $acousticArgs = @(
        "scripts\export-v3-acoustic-weights.py",
        "--source-repo", $ResolvedSourceRepo,
        "--checkpoint", $ResolvedModelDir,
        "--subfolder", "update",
        "--output", (Join-Path $AcousticDir "vieneu_acoustic_weights.npz")
    )
    if (![string]::IsNullOrWhiteSpace($HfToken)) {
        $acousticArgs += @("--hf-token", $HfToken)
    }
    & $Python @acousticArgs
    if ($LASTEXITCODE -ne 0) {
        throw "export-v3-acoustic-weights.py failed."
    }

    Write-Step "Validating native asset layout"
    Require-File (Join-Path $ResolvedModelDir "config.json")
    Require-File (Join-Path $ResolvedModelDir "tokenizer.json")
    Require-File (Join-Path $ResolvedModelDir "voices_v3_turbo.json")
    Require-File (Join-Path $ResolvedModelDir "speaker_encoder.onnx")
    Require-File (Join-Path $ResolvedModelDir "backbone.gguf")
    Require-File (Join-Path $ResolvedModelDir "vieneu_v3_heads.npz")
    Require-File (Join-Path $AcousticDir "vieneu_acoustic_weights.npz")

    Write-Step "Done"
    Write-Host "Native v3 Turbo assets are ready at: $ResolvedModelDir" -ForegroundColor Green
} finally {
    Pop-Location
}
