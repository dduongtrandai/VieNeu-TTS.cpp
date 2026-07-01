param (
    [string]$BuildDir = "build",
    [string]$OutputDir = "dist",
    [string]$OnnxRuntimeRoot = $env:ONNXRUNTIME_ROOT,
    [ValidateSet("cpu", "directml", "openvino")]
    [string]$RuntimeFlavor = "cpu",
    [ValidateSet("cpu", "cuda", "vulkan", "metal")]
    [string]$NativeBackend = "cpu",
    [string]$PackageFlavor = "",
    [string]$PackagePlatform = "",
    [string]$OnnxRuntimeVersion = "",
    [string]$RuntimeVersion = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

function Resolve-InputPath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
}

$BuildDir = Resolve-InputPath $BuildDir
$OutputDir = Resolve-InputPath $OutputDir

if (-not $RuntimeVersion) {
    if ($env:GITHUB_REF_NAME) {
        $RuntimeVersion = $env:GITHUB_REF_NAME
    } else {
        $RuntimeVersion = "v0.1.0"
    }
}

function Get-HostPlatform {
    if ($IsWindows -or $env:OS -eq "Windows_NT") { return "win" }
    if ($IsMacOS) { return "macos" }
    return "linux"
}

function Get-ArtifactName {
    param([string]$Kind)

    $platform = Get-HostPlatform
    if ($Kind -eq "cli") {
        if ($platform.StartsWith("win")) { return "vieneu-tts-cli.exe" }
        return "vieneu-tts-cli"
    }
    if ($platform.StartsWith("win")) { return "vieneu-tts.dll" }
    if ($platform.StartsWith("macos")) { return "libvieneu-tts.dylib" }
    return "libvieneu-tts.so"
}

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
        if (Test-Path $p) { return (Resolve-Path $p).Path }
    }

    $recursiveHit = Get-ChildItem -Path $BaseDir -Filter $Filename -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($recursiveHit) { return $recursiveHit.FullName }
    return ""
}

function Copy-RuntimeFile {
    param(
        [string]$SourcePath,
        [string]$DestinationDir
    )

    if (-not $SourcePath -or -not (Test-Path $SourcePath)) { return }

    $dest = Join-Path $DestinationDir (Split-Path $SourcePath -Leaf)
    Write-Host "Copying runtime dependency: $SourcePath"
    Copy-Item -LiteralPath $SourcePath -Destination $dest -Force
}

function Copy-FirstRuntimeFile {
    param(
        [string[]]$Roots,
        [string[]]$Patterns,
        [string]$DestinationDir,
        [string]$DisplayName
    )

    foreach ($root in ($Roots | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique)) {
        foreach ($pattern in $Patterns) {
            $hit = Get-ChildItem -LiteralPath $root -Filter $pattern -File -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($hit) {
                Copy-RuntimeFile -SourcePath $hit.FullName -DestinationDir $DestinationDir
                return $hit.FullName
            }
        }
    }
    Write-Error "$DisplayName is required but was not found."
}

function Copy-RuntimePatterns {
    param(
        [string[]]$Roots,
        [string[]]$Patterns,
        [string]$DestinationDir
    )

    foreach ($root in ($Roots | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique)) {
        foreach ($pattern in $Patterns) {
            Get-ChildItem -LiteralPath $root -Filter $pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
                Copy-RuntimeFile -SourcePath $_.FullName -DestinationDir $DestinationDir
            }
        }
    }
}

function Copy-RequiredRuntimePatterns {
    param(
        [string[]]$Roots,
        [string[]]$Patterns,
        [string]$DestinationDir,
        [string]$DisplayName
    )

    foreach ($pattern in $Patterns) {
        $null = Copy-FirstRuntimeFile -Roots $Roots -Patterns @($pattern) -DestinationDir $DestinationDir -DisplayName "$DisplayName ($pattern)"
    }
}

if (-not $PackagePlatform) {
    $PackagePlatform = Get-HostPlatform
}
if (-not $PackageFlavor) {
    $PackageFlavor = switch ($NativeBackend) {
        "cuda" { "cuda-12" }
        "vulkan" { "vulkan" }
        "metal" { "metal" }
        default { $RuntimeFlavor }
    }
}

$isWindowsPackage = $PackagePlatform.StartsWith("win")
$archiveExt = if ($isWindowsPackage) { ".zip" } else { ".tar.gz" }
$archivePath = Join-Path $OutputDir "vieneu-tts-$PackagePlatform-$PackageFlavor$archiveExt"

Write-Host "========================================="
Write-Host "Packaging VieNeu TTS Runtime Release..."
Write-Host "========================================="
Write-Host "Platform: $PackagePlatform"
Write-Host "Runtime flavor: $RuntimeFlavor"
Write-Host "Native ggml backend: $NativeBackend"
Write-Host "Archive: $archivePath"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
if (Test-Path $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}

$StagingDir = Join-Path $OutputDir "stage-$PackagePlatform-$PackageFlavor"
if (Test-Path $StagingDir) {
    Remove-Item -Recurse -Force $StagingDir
}
New-Item -ItemType Directory -Path $StagingDir | Out-Null

$LibraryName = Get-ArtifactName -Kind "library"
$CliName = Get-ArtifactName -Kind "cli"
$LibraryPath = Resolve-BuildArtifactPath -BaseDir $BuildDir -Filename $LibraryName
$CliPath = Resolve-BuildArtifactPath -BaseDir $BuildDir -Filename $CliName

if (-not $LibraryPath) {
    Write-Error "Could not find build library $LibraryName under '$BuildDir'. Build the project first."
}
if (-not $CliPath) {
    Write-Error "Could not find CLI tool $CliName under '$BuildDir'. Build the project first."
}

Copy-Item -LiteralPath $LibraryPath -Destination $StagingDir -Force
Copy-Item -LiteralPath $CliPath -Destination $StagingDir -Force

$runtimeRoots = @()
if ($OnnxRuntimeRoot) {
    $runtimeRoots += (Join-Path $OnnxRuntimeRoot "bin")
    $runtimeRoots += (Join-Path $OnnxRuntimeRoot "lib")
    $runtimeRoots += (Join-Path (Join-Path (Join-Path $OnnxRuntimeRoot "runtimes") "win-x64") "native")
    $runtimeRoots += $OnnxRuntimeRoot
}
if ($NativeBackend -eq "cuda" -and $env:CUDA_PATH) {
    $runtimeRoots += (Join-Path $env:CUDA_PATH "bin")
    $runtimeRoots += (Join-Path $env:CUDA_PATH "lib64")
}
if ($NativeBackend -eq "cuda" -and $env:CUDA_HOME) {
    $runtimeRoots += (Join-Path $env:CUDA_HOME "bin")
    $runtimeRoots += (Join-Path $env:CUDA_HOME "lib64")
}
if ($NativeBackend -eq "cuda" -and (Test-Path "/usr/local/cuda")) {
    $runtimeRoots += "/usr/local/cuda/bin"
    $runtimeRoots += "/usr/local/cuda/lib64"
}
$runtimeRoots += $BuildDir
$runtimeRoots += (Join-Path $BuildDir "Release")
$runtimeRoots += (Join-Path $BuildDir "RelWithDebInfo")
$runtimeRoots += (Join-Path (Join-Path (Join-Path $BuildDir "bin") "Release") "")
$runtimeRoots += (Join-Path (Join-Path (Join-Path $BuildDir "bin") "RelWithDebInfo") "")
$runtimeRoots += (Join-Path $BuildDir "bin")

$onnxPatterns = if ($PackagePlatform.StartsWith("win")) {
    @("onnxruntime.dll")
} elseif ($PackagePlatform.StartsWith("macos")) {
    @("libonnxruntime*.dylib")
} else {
    @("libonnxruntime.so*")
}
$null = Copy-FirstRuntimeFile -Roots $runtimeRoots -Patterns $onnxPatterns -DestinationDir $StagingDir -DisplayName "ONNX Runtime library"

$dependencyPatterns = @(
    "onnxruntime_providers*.dll",
    "libonnxruntime_providers*",
    "DirectML.dll",
    "openvino*.dll",
    "libopenvino*",
    "tbb*.dll",
    "libtbb*",
    "cudart64_*.dll",
    "cublas64_*.dll",
    "cublasLt64_*.dll",
    "libcudart.so*",
    "libcublas.so*",
    "libcublasLt.so*",
    "libllama.*",
    "llama.dll",
    "libggml*",
    "ggml*.dll",
    "sea_g2p_rs.dll",
    "libsea_g2p_rs.so*",
    "libsea_g2p_rs.dylib"
)
Copy-RuntimePatterns -Roots $runtimeRoots -Patterns $dependencyPatterns -DestinationDir $StagingDir

if ($NativeBackend -eq "cuda") {
    if ($isWindowsPackage) {
        Copy-RequiredRuntimePatterns -Roots $runtimeRoots -Patterns @(
            "cudart64_12.dll",
            "cublas64_12.dll",
            "cublasLt64_12.dll"
        ) -DestinationDir $StagingDir -DisplayName "CUDA runtime library"
    } elseif ($PackagePlatform.StartsWith("linux")) {
        Copy-RequiredRuntimePatterns -Roots $runtimeRoots -Patterns @(
            "libcudart.so*",
            "libcublas.so*",
            "libcublasLt.so*"
        ) -DestinationDir $StagingDir -DisplayName "CUDA runtime library"
    }
}

$SeaG2pDictCandidates = @(
    (Join-Path $ProjectRoot "third_party/sea-g2p/python/sea_g2p/sea_g2p.bin")
)
$SeaG2pDict = $SeaG2pDictCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($SeaG2pDict) {
    Copy-RuntimeFile -SourcePath $SeaG2pDict -DestinationDir $StagingDir
}

$LicensesDir = Join-Path $StagingDir "LICENSES"
New-Item -ItemType Directory -Path $LicensesDir | Out-Null
$LicenseFiles = @(
    (Join-Path $ProjectRoot "LICENSE"),
    (Join-Path $ProjectRoot "README.md"),
    (Join-Path (Join-Path (Join-Path $ProjectRoot "third_party") "llama.cpp") "LICENSE"),
    (Join-Path (Join-Path (Join-Path $ProjectRoot "third_party") "sea-g2p") "LICENSE")
)
foreach ($lf in $LicenseFiles) {
    if (Test-Path $lf) {
        Copy-Item -LiteralPath $lf -Destination $LicensesDir -Force
    }
}

$nativeLabel = switch ($NativeBackend) {
    "cuda" { "CUDA 12" }
    "vulkan" { "Vulkan" }
    "metal" { "Metal" }
    default { "CPU" }
}

$ManifestContent = @{
    id = "vieneu-tts-$PackagePlatform-$PackageFlavor"
    name = "$nativeLabel VieNeu TTS Native Pipeline ($PackagePlatform)"
    version = $RuntimeVersion
    type = "tts"
    engineFamily = "vieneu-tts"
    variant = "$PackagePlatform-$PackageFlavor"
    library = $LibraryName
    executable = $CliName
    metadata = @{
        backend = "vieneu-tts"
        nativeBackend = $NativeBackend
        onnxRuntimeFlavor = $RuntimeFlavor
        components = @{
            "llama.cpp" = "bundled"
            "onnxruntime" = if ($OnnxRuntimeVersion) { $OnnxRuntimeVersion } else { "unknown" }
        }
        profiles = @("vieneu-v3-native")
    }
}

$ManifestPath = Join-Path $StagingDir "backend-manifest.json"
$ManifestContent | ConvertTo-Json -Depth 5 | Out-File -FilePath $ManifestPath -Encoding utf8

if ($isWindowsPackage) {
    Compress-Archive -Path (Join-Path $StagingDir "*") -DestinationPath $archivePath -Force
} else {
    $previousLocation = Get-Location
    try {
        Set-Location $StagingDir
        tar -czf $archivePath .
    } finally {
        Set-Location $previousLocation
    }
}

Remove-Item -Recurse -Force $StagingDir
Write-Host "Successfully packaged: $archivePath"
