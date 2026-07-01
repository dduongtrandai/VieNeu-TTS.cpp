<#
.SYNOPSIS
    Builds vieneu-tts.cpp and its dependencies locally on Windows.
.DESCRIPTION
    This script automates the process of checking and downloading the required
    dependencies (llama.cpp, ONNX Runtime SDK, and optionally Ninja),
    configuring the MSVC build environment, and building the DLLs.
#>

param (
    [string]$OnnxRuntimeVersion = "1.24.4",
    [ValidateSet("cpu", "directml", "openvino")]
    [string]$OnnxRuntimeFlavor = "cpu",
    [ValidateSet("cpu", "cuda", "vulkan", "all")]
    [string]$NativeBackend = "cpu",
    [string]$BuildDir = "build",
    [string]$OpenVINOOnnxRuntimeVersion = "1.24.1",
    [switch]$Clean,
    [switch]$NoPackage,
    [switch]$PortableCpu,
    [string]$LlamaCppRepo = "https://github.com/ggml-org/llama.cpp.git",
    [string]$Generator = "Ninja"
)

$ErrorActionPreference = "Stop"
$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent -Path $PSScriptRoot

if ($NativeBackend -eq "all") {
    Write-Host "Building all local native backends: cpu, cuda, vulkan" -ForegroundColor Cyan
    foreach ($backend in @("cpu", "cuda", "vulkan")) {
        $backendBuildDir = "build-$backend"
        Write-Host "Starting $backend build in $backendBuildDir..." -ForegroundColor Cyan
        & $PSCommandPath `
            -OnnxRuntimeVersion $OnnxRuntimeVersion `
            -OnnxRuntimeFlavor $OnnxRuntimeFlavor `
            -NativeBackend $backend `
            -BuildDir $backendBuildDir `
            -OpenVINOOnnxRuntimeVersion $OpenVINOOnnxRuntimeVersion `
            -LlamaCppRepo $LlamaCppRepo `
            -Generator $Generator `
            -Clean:$Clean `
            -NoPackage:$NoPackage `
            -PortableCpu:$PortableCpu
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    exit 0
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Starting Local Build Setup for VieNeu TTS..." -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "ONNX Runtime flavor: $OnnxRuntimeFlavor" -ForegroundColor Cyan
Write-Host "Native ggml backend: $NativeBackend" -ForegroundColor Cyan

# 1. Check and initialize llama.cpp submodule if missing
$LlamaDir = Join-Path $ProjectRoot "third_party\llama.cpp"
if (-not (Test-Path (Join-Path $LlamaDir "CMakeLists.txt"))) {
    Write-Host "llama.cpp submodule not found. Initializing submodules..." -ForegroundColor Yellow
    git submodule update --init --recursive --depth 1 third_party/llama.cpp
    if (-not (Test-Path (Join-Path $LlamaDir "CMakeLists.txt"))) {
        Write-Host "Submodule init did not produce llama.cpp. Cloning fallback from $LlamaCppRepo..." -ForegroundColor Yellow
        git clone --depth 1 $LlamaCppRepo $LlamaDir
    }
} else {
    Write-Host "Found llama.cpp source directory." -ForegroundColor Green
}

# 2. Check and Download ONNX Runtime SDK if missing
$OrtSdkDir = Join-Path $ProjectRoot "ort_sdk"

function Expand-NuGetPackage([string]$PackageId, [string]$Version, [string]$DestinationRoot) {
    if (Test-Path $DestinationRoot) {
        return
    }

    Write-Host "$PackageId v$Version not found. Downloading..." -ForegroundColor Yellow
    if (-not (Test-Path $OrtSdkDir)) {
        New-Item -ItemType Directory -Path $OrtSdkDir | Out-Null
    }

    $PackagePath = Join-Path $OrtSdkDir "$PackageId.$Version.zip"
    $Url = "https://www.nuget.org/api/v2/package/$PackageId/$Version"
    Write-Host "Downloading NuGet package from $Url..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $Url -OutFile $PackagePath

    Write-Host "Extracting NuGet package to $DestinationRoot..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null
    Expand-Archive -Path $PackagePath -DestinationPath $DestinationRoot -Force
    Remove-Item $PackagePath -Force
}

function Copy-RequiredRuntimePattern([string[]]$Roots, [string]$Pattern, [string]$DestinationDir, [string]$DisplayName) {
    foreach ($root in ($Roots | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique)) {
        $hit = Get-ChildItem -LiteralPath $root -Filter $Pattern -File -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) {
            Copy-Item -LiteralPath $hit.FullName -Destination $DestinationDir -Force
            return
        }
    }
    Write-Error "$DisplayName ($Pattern) is required but was not found."
}

if ($OnnxRuntimeFlavor -eq "cpu") {
    $OrtRootName = "onnxruntime-win-x64-$OnnxRuntimeVersion"
    $OrtRoot = Join-Path $OrtSdkDir $OrtRootName
    if (-not (Test-Path $OrtRoot)) {
        Write-Host "ONNX Runtime CPU SDK v$OnnxRuntimeVersion not found. Downloading..." -ForegroundColor Yellow
        if (-not (Test-Path $OrtSdkDir)) {
            New-Item -ItemType Directory -Path $OrtSdkDir | Out-Null
        }

        $Url = "https://github.com/microsoft/onnxruntime/releases/download/v$OnnxRuntimeVersion/$OrtRootName.zip"
        $ZipPath = Join-Path $OrtSdkDir "$OrtRootName.zip"

        Write-Host "Downloading ONNX Runtime SDK from $Url..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $Url -OutFile $ZipPath

        Write-Host "Extracting ONNX Runtime SDK to $OrtSdkDir..." -ForegroundColor Cyan
        Expand-Archive -Path $ZipPath -DestinationPath $OrtSdkDir -Force

        Remove-Item $ZipPath -Force
        Write-Host "ONNX Runtime CPU SDK successfully installed at $OrtRoot" -ForegroundColor Green
    } else {
        Write-Host "Found ONNX Runtime CPU SDK at $OrtRoot" -ForegroundColor Green
    }
} elseif ($OnnxRuntimeFlavor -eq "directml") {
    $OrtRoot = Join-Path $OrtSdkDir "Microsoft.ML.OnnxRuntime.DirectML-$OnnxRuntimeVersion"
    Expand-NuGetPackage "Microsoft.ML.OnnxRuntime.DirectML" $OnnxRuntimeVersion $OrtRoot
    Write-Host "Using ONNX Runtime DirectML package at $OrtRoot" -ForegroundColor Green
} else {
    $OrtRoot = Join-Path $OrtSdkDir "Intel.ML.OnnxRuntime.OpenVino-$OpenVINOOnnxRuntimeVersion"
    Expand-NuGetPackage "Intel.ML.OnnxRuntime.OpenVino" $OpenVINOOnnxRuntimeVersion $OrtRoot
    Write-Host "Using ONNX Runtime OpenVINO package at $OrtRoot" -ForegroundColor Green
}

$SelectedOnnxRuntimeVersion = if ($OnnxRuntimeFlavor -eq "openvino") { $OpenVINOOnnxRuntimeVersion } else { $OnnxRuntimeVersion }

# 3. Check for compiler tools and load VS environment if needed
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    Write-Host "cl.exe (MSVC Compiler) not found in PATH. Finding Visual Studio installation..." -ForegroundColor Yellow
    
    $vsWherePath = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vsWherePath)) {
        $vsWherePath = Join-Path ${env:ProgramFiles} "Microsoft Visual Studio\Installer\vswhere.exe"
    }
    
    if (Test-Path $vsWherePath) {
        $vsPath = & $vsWherePath -latest -property installationPath
        if ($vsPath) {
            Write-Host "Found Visual Studio at: $vsPath" -ForegroundColor Green
            $devShellPath = Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
            if (Test-Path $devShellPath) {
                Write-Host "Loading MSVC Developer Environment..." -ForegroundColor Cyan
                $originalLocation = Get-Location
                Import-Module $devShellPath -ErrorAction Stop
                # Initialize developer shell for x64
                Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64" -ErrorAction Stop
                # Restore original path in case it changed
                Set-Location $originalLocation
            } else {
                Write-Error "Could not find DevShell DLL at $devShellPath"
            }
        } else {
            Write-Error "No Visual Studio installation found via vswhere."
        }
    } else {
        Write-Error "vswhere.exe not found. Visual Studio might not be installed or in the standard path."
    }
} else {
    Write-Host "cl.exe (MSVC Compiler) is already available in PATH." -ForegroundColor Green
}

# 4. Check for Ninja generator tool if Ninja is selected
if ($Generator -eq "Ninja") {
    if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
        Write-Host "Ninja is selected but not found in PATH. Downloading ninja.exe..." -ForegroundColor Yellow
        $NinjaUrl = "https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-win.zip"
        $NinjaZip = Join-Path $ProjectRoot "ninja-win.zip"
        
        Write-Host "Downloading Ninja from $NinjaUrl..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $NinjaUrl -OutFile $NinjaZip
        
        Write-Host "Extracting Ninja..." -ForegroundColor Cyan
        Expand-Archive -Path $NinjaZip -DestinationPath $ProjectRoot -Force
        Remove-Item $NinjaZip -Force
        
        # Add the project root folder to PATH environment variable for the current session
        $env:PATH = "$ProjectRoot;" + $env:PATH
        Write-Host "Ninja successfully installed in current path." -ForegroundColor Green
    } else {
        Write-Host "Ninja tool is available in PATH." -ForegroundColor Green
    }
}

if ($NativeBackend -eq "cuda") {
    if (-not $env:CUDA_PATH -or -not (Test-Path $env:CUDA_PATH)) {
        Write-Error "NativeBackend=cuda requires NVIDIA CUDA Toolkit. Install CUDA 12.x or set CUDA_PATH to the toolkit root."
    }
    $NvccPath = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
    if (-not (Test-Path $NvccPath)) {
        Write-Error "Could not find nvcc.exe at '$NvccPath'. Check your CUDA Toolkit installation."
    }
    Write-Host "Found CUDA Toolkit at $env:CUDA_PATH" -ForegroundColor Green
}

# 5. Clean build directory if requested
$BuildPath = Join-Path $ProjectRoot $BuildDir
if ($Clean -and (Test-Path $BuildPath)) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildPath
}

# 6. Configure CMake
Write-Host "Configuring CMake..." -ForegroundColor Cyan
$cmakeArgs = @(
    "-B", $BuildPath,
    "-S", $ProjectRoot,
    "-G", $Generator,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DVIENEU_PORTABLE_CPU=$($PortableCpu.IsPresent)",
    "-DVIENEU_NATIVE_BACKEND=$NativeBackend",
    "-DVIENEU_LLAMA_DIR=$LlamaDir",
    "-DONNXRUNTIME_ROOT=$OrtRoot"
)

# Run configuration
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    if (Test-Path (Join-Path $BuildPath "CMakeCache.txt")) {
        Write-Host "CMake configuration failed. Stale CMakeCache.txt detected. Cleaning build directory and retrying..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $BuildPath
        & cmake @cmakeArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Error "CMake configuration failed again."
        }
    } else {
        Write-Error "CMake configuration failed."
    }
}

# 7. Build the Project
Write-Host "Building project..." -ForegroundColor Cyan
& cmake --build $BuildPath --config Release --parallel
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed."
}

# 8. Copy ONNX Runtime DLLs next to the locally built executables.
Write-Host "Copying ONNX Runtime runtime DLLs..." -ForegroundColor Cyan
$OrtRuntimeRoots = @(
    (Join-Path $OrtRoot "bin"),
    (Join-Path $OrtRoot "lib"),
    (Join-Path $OrtRoot "runtimes\win-x64\native"),
    $OrtRoot
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

$OrtRuntimePatterns = @(
    "onnxruntime.dll",
    "onnxruntime_providers*.dll",
    "DirectML.dll",
    "openvino*.dll",
    "tbb*.dll",
    "cudart64_*.dll",
    "cublas64_*.dll",
    "cublasLt64_*.dll"
)

foreach ($root in $OrtRuntimeRoots) {
    foreach ($pattern in $OrtRuntimePatterns) {
        Get-ChildItem -LiteralPath $root -Filter $pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $BuildPath -Force
        }
    }
}

if ($NativeBackend -eq "cuda") {
    $CudaBin = Join-Path $env:CUDA_PATH "bin"
    foreach ($pattern in @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")) {
        Copy-RequiredRuntimePattern -Roots @($CudaBin) -Pattern $pattern -DestinationDir $BuildPath -DisplayName "CUDA runtime library"
    }
}

# 9. Package the runtime if not disabled
if (-not $NoPackage) {
    Write-Host "Packaging runtime..." -ForegroundColor Cyan
    $PackageScript = Join-Path $PSScriptRoot "package-runtime.ps1"
    if (Test-Path $PackageScript) {
        $OutputDir = Join-Path $ProjectRoot "dist"
        $PackageFlavor = switch ($NativeBackend) {
            "cuda" { "cuda-12" }
            "vulkan" { "vulkan" }
            default { $OnnxRuntimeFlavor }
        }
        & $PackageScript -BuildDir $BuildPath -OutputDir $OutputDir -OnnxRuntimeRoot $OrtRoot -RuntimeFlavor $OnnxRuntimeFlavor -NativeBackend $NativeBackend -PackageFlavor $PackageFlavor -PackagePlatform "win" -OnnxRuntimeVersion $SelectedOnnxRuntimeVersion -RuntimeVersion "local"
    } else {
        Write-Warning "Could not find package script at $PackageScript"
    }
}

Write-Host "=========================================" -ForegroundColor Green
Write-Host "Local build process completed successfully!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
