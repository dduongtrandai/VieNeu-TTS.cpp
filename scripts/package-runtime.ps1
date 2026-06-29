param (
    [string]$BuildDir = "build",
    [string]$OutputDir = "dist",
    [string]$OnnxRuntimeRoot = $env:ONNXRUNTIME_ROOT,
    [ValidateSet("cpu", "directml", "openvino")]
    [string]$RuntimeFlavor = "cpu",
    [string]$OnnxRuntimeVersion = "",
    [string]$RuntimeVersion = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not $RuntimeVersion) {
    if ($env:GITHUB_REF_NAME) {
        $RuntimeVersion = $env:GITHUB_REF_NAME
    } else {
        $RuntimeVersion = "v0.1.0"
    }
}

Write-Host "========================================="
Write-Host "Packaging VieNeu TTS Runtime Release..."
Write-Host "========================================="
Write-Host "Runtime flavor: $RuntimeFlavor"

# Create output dir
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$ZipPath = Join-Path $OutputDir "vieneu-tts-win-$RuntimeFlavor.zip"
if (Test-Path $ZipPath) {
    Remove-Item $ZipPath
}

# Temporary staging dir
$StagingDir = Join-Path $OutputDir "stage"
if (Test-Path $StagingDir) {
    Remove-Item -Recurse -Force $StagingDir
}
New-Item -ItemType Directory -Path $StagingDir | Out-Null

function Resolve-BuildArtifactPath {
    param(
        [string]$BaseDir,
        [string]$Filename
    )

    $candidates = @(
        (Join-Path $BaseDir $Filename),
        (Join-Path (Join-Path $BaseDir "Release") $Filename),
        (Join-Path (Join-Path $BaseDir "RelWithDebInfo") $Filename),
        (Join-Path (Join-Path (Join-Path $BaseDir "bin") "Release") $Filename),
        (Join-Path (Join-Path (Join-Path $BaseDir "bin") "RelWithDebInfo") $Filename),
        (Join-Path (Join-Path $BaseDir "bin") $Filename)
    )

    foreach ($p in $candidates) {
        if (Test-Path $p) {
            return $p
        }
    }

    $recursiveHit = Get-ChildItem -Path $BaseDir -Filter $Filename -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($recursiveHit) {
        return $recursiveHit.FullName
    }

    return ""
}

function Copy-RuntimeDll {
    param(
        [string]$SourcePath,
        [string]$DestinationDir
    )

    if (-not $SourcePath -or -not (Test-Path $SourcePath)) {
        return
    }

    $dest = Join-Path $DestinationDir (Split-Path $SourcePath -Leaf)
    Write-Host "Copying runtime dependency: $SourcePath"
    Copy-Item $SourcePath $dest -Force
}

function Resolve-FirstExistingPath {
    param(
        [string[]]$Candidates
    )

    foreach ($p in $Candidates) {
        if ($p -and (Test-Path $p)) {
            return $p
        }
    }
    return ""
}

# Verify build artifacts
$DllPath = Resolve-BuildArtifactPath -BaseDir $BuildDir -Filename "vieneu-tts.dll"
$CliPath = Resolve-BuildArtifactPath -BaseDir $BuildDir -Filename "vieneu-tts-cli.exe"

if (-not $DllPath) {
    Write-Error "Could not find build library vieneu-tts.dll under '$BuildDir'. Build the project first."
}
if (-not $CliPath) {
    Write-Error "Could not find build CLI tool vieneu-tts-cli.exe under '$BuildDir'. Build the project first."
}

# Copy DLL and CLI
Copy-Item $DllPath $StagingDir
Copy-Item $CliPath $StagingDir

# Copy onnxruntime.dll
$OrtCandidates = @()
if ($OnnxRuntimeRoot) {
    $OrtCandidates += (Join-Path $OnnxRuntimeRoot "bin\onnxruntime.dll")
    $OrtCandidates += (Join-Path $OnnxRuntimeRoot "lib\onnxruntime.dll")
    $OrtCandidates += (Join-Path $OnnxRuntimeRoot "runtimes\win-x64\native\onnxruntime.dll")
    $OrtCandidates += (Join-Path $OnnxRuntimeRoot "onnxruntime.dll")
}
$OrtCandidates += (Join-Path $BuildDir "onnxruntime.dll")
$OrtCandidates += (Join-Path (Join-Path $BuildDir "Release") "onnxruntime.dll")
$OrtCandidates += (Join-Path (Join-Path $BuildDir "RelWithDebInfo") "onnxruntime.dll")
$OrtCandidates += (Join-Path (Join-Path (Join-Path $BuildDir "bin") "Release") "onnxruntime.dll")
$OrtCandidates += (Join-Path (Join-Path (Join-Path $BuildDir "bin") "RelWithDebInfo") "onnxruntime.dll")

$OrtDllSrc = Resolve-FirstExistingPath -Candidates $OrtCandidates
if (-not $OrtDllSrc) {
    $OrtFallback = Resolve-BuildArtifactPath -BaseDir $BuildDir -Filename "onnxruntime.dll"
    if ($OrtFallback) {
        $OrtDllSrc = $OrtFallback
    }
}

if (-not $OrtDllSrc) {
    $diag = ($OrtCandidates | ForEach-Object { " - $_" }) -join "`n"
    Write-Error "onnxruntime.dll is required but was not found. Checked:`n$diag"
}

Write-Host "Copying ONNX Runtime DLL from: $OrtDllSrc"
Copy-Item $OrtDllSrc $StagingDir

# Copy optional ONNX Runtime provider dependencies when the supplied SDK includes
# CUDA, OpenVINO, DirectML, or other EP-specific runtime DLLs.
$OrtProviderRoots = @()
if ($OnnxRuntimeRoot) {
    $OrtProviderRoots += (Join-Path $OnnxRuntimeRoot "bin")
    $OrtProviderRoots += (Join-Path $OnnxRuntimeRoot "lib")
    $OrtProviderRoots += (Join-Path $OnnxRuntimeRoot "runtimes\win-x64\native")
    $OrtProviderRoots += $OnnxRuntimeRoot
}
$OrtProviderRoots += $BuildDir
$OrtProviderRoots += (Join-Path $BuildDir "Release")
$OrtProviderRoots += (Join-Path $BuildDir "RelWithDebInfo")
$OrtProviderRoots += (Join-Path (Join-Path $BuildDir "bin") "Release")
$OrtProviderRoots += (Join-Path (Join-Path $BuildDir "bin") "RelWithDebInfo")

$OrtProviderPatterns = @(
    "onnxruntime_providers*.dll",
    "DirectML.dll",
    "openvino*.dll",
    "tbb*.dll"
)

foreach ($root in ($OrtProviderRoots | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique)) {
    foreach ($pattern in $OrtProviderPatterns) {
        Get-ChildItem -LiteralPath $root -Filter $pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-RuntimeDll -SourcePath $_.FullName -DestinationDir $StagingDir
        }
    }
}

# Copy llama.cpp shared DLLs if the build produced them. Visual Studio builds from
# llama.cpp usually place these under build/bin/Release, not the build root.
$LlamaDllPath = Resolve-BuildArtifactPath -BaseDir $BuildDir -Filename "llama.dll"
if ($LlamaDllPath) {
    Copy-RuntimeDll -SourcePath $LlamaDllPath -DestinationDir $StagingDir

    $dependencyRoots = @(
        (Split-Path $LlamaDllPath -Parent),
        $BuildDir,
        (Join-Path $BuildDir "Release"),
        (Join-Path $BuildDir "RelWithDebInfo"),
        (Join-Path (Join-Path (Join-Path $BuildDir "bin") "Release") ""),
        (Join-Path (Join-Path (Join-Path $BuildDir "bin") "RelWithDebInfo") "")
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

    $dependencyNames = @(
        "ggml.dll",
        "ggml-base.dll",
        "ggml-cpu.dll",
        "ggml-blas.dll",
        "ggml-rpc.dll"
    )

    foreach ($root in $dependencyRoots) {
        foreach ($name in $dependencyNames) {
            $candidate = Join-Path $root $name
            if (Test-Path $candidate) {
                Copy-RuntimeDll -SourcePath $candidate -DestinationDir $StagingDir
            }
        }
    }

    $ggmlDlls = Get-ChildItem -Path $BuildDir -Filter "ggml*.dll" -File -Recurse -ErrorAction SilentlyContinue
    foreach ($ggmlDll in $ggmlDlls) {
        Copy-RuntimeDll -SourcePath $ggmlDll.FullName -DestinationDir $StagingDir
    }
} else {
    Write-Host "llama.dll was not found under '$BuildDir'; assuming llama.cpp was linked statically."
}

# Copy LICENSE files if they exist
$LicensesDir = Join-Path $StagingDir "LICENSES"
New-Item -ItemType Directory -Path $LicensesDir | Out-Null
$LicenseFiles = @(
    (Join-Path $ProjectRoot "LICENSE"),
    (Join-Path $ProjectRoot "README.md"),
    (Join-Path $ProjectRoot "llama.cpp\LICENSE")
)
foreach ($lf in $LicenseFiles) {
    if (Test-Path $lf) {
        Copy-Item $lf $LicensesDir
    }
}

# Write backend-manifest.json
$ManifestPath = Join-Path $StagingDir "backend-manifest.json"
$runtimeLabel = switch ($RuntimeFlavor) {
    "cpu" { "CPU" }
    "directml" { "DirectML" }
    "openvino" { "OpenVINO" }
}

$ManifestContent = @{
    id = "vieneu-tts-win-x86_64-$RuntimeFlavor"
    name = "$runtimeLabel VieNeu TTS Pipeline (x64)"
    version = "v0.1.0"
    type = "tts"
    engineFamily = "vieneu-tts"
    variant = "win-x86_64-$RuntimeFlavor"
    library = "vieneu-tts.dll"
    metadata = @{
        backend = "vieneu-tts"
        components = @{
            "llama.cpp" = "b4600" # fallback/placeholder commit
            "onnxruntime" = if ($OnnxRuntimeVersion) { $OnnxRuntimeVersion } else { "unknown" }
        }
        profiles = @("vieneu-v2-turbo", "vieneu-v3-onnx")
    }
}
$ManifestContent.version = $RuntimeVersion

$ManifestContent | ConvertTo-Json -Depth 5 | Out-File -FilePath $ManifestPath -Encoding utf8

# Compress to zip file
Write-Host "Compressing stage files to: $ZipPath..."
Compress-Archive -Path (Join-Path $StagingDir "*") -DestinationPath $ZipPath

# Cleanup staging dir
Remove-Item -Recurse -Force $StagingDir

Write-Host "Successfully packaged: $ZipPath"
