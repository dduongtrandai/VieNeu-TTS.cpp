cmake_minimum_required(VERSION 3.14)
project(vieneu-tts-project LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)
find_package(OpenMP)

# DLL/Output Directory setup (same as build root)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# MSVC specific compile settings
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
endif()

# Configuration Options
set(VIENEU_LLAMA_DIR "third_party/llama.cpp" CACHE PATH "Path to llama.cpp source directory")
set(VIENEU_SEA_G2P_DIR "third_party/sea-g2p" CACHE PATH "Path to sea-g2p source directory")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime SDK root")
option(VIENEU_PORTABLE_CPU "Build portable CPU kernels instead of host-native SIMD kernels" OFF)
option(VIENEU_SEA_G2P "Build and link the native Rust sea-g2p phonemizer" OFF)
set(VIENEU_NATIVE_BACKEND "cpu" CACHE STRING "Native ggml backend to build: cpu, cuda, vulkan, or metal")
set_property(CACHE VIENEU_NATIVE_BACKEND PROPERTY STRINGS "cpu" "cuda" "vulkan" "metal")

if(UNIX)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Fallback environment check for ONNX Runtime
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()

# Find llama.cpp dependency
if(NOT IS_ABSOLUTE "${VIENEU_LLAMA_DIR}")
    get_filename_component(VIENEU_LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${VIENEU_LLAMA_DIR}" ABSOLUTE)
endif()
if(NOT EXISTS "${VIENEU_LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp CMakeLists.txt not found at ${VIENEU_LLAMA_DIR}")
endif()

if(VIENEU_PORTABLE_CPU)
    set(GGML_NATIVE OFF CACHE BOOL "Build host-native ggml CPU kernels" FORCE)
else()
    set(GGML_NATIVE ON CACHE BOOL "Build host-native ggml CPU kernels" FORCE)
endif()
message(STATUS "VieNeu portable CPU build: ${VIENEU_PORTABLE_CPU}")
message(STATUS "GGML native CPU kernels: ${GGML_NATIVE}")
string(TOLOWER "${VIENEU_NATIVE_BACKEND}" VIENEU_NATIVE_BACKEND_NORMALIZED)
if(NOT VIENEU_NATIVE_BACKEND_NORMALIZED MATCHES "^(cpu|cuda|vulkan|metal)$")
    message(FATAL_ERROR "Unsupported VIENEU_NATIVE_BACKEND='${VIENEU_NATIVE_BACKEND}'. Expected cpu, cuda, vulkan, or metal.")
endif()
set(GGML_CUDA OFF CACHE BOOL "Build ggml CUDA backend" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "Build ggml Vulkan backend" FORCE)
set(GGML_METAL OFF CACHE BOOL "Build ggml Metal backend" FORCE)
if(VIENEU_NATIVE_BACKEND_NORMALIZED STREQUAL "cuda")
    set(GGML_CUDA ON CACHE BOOL "Build ggml CUDA backend" FORCE)
    find_package(CUDAToolkit REQUIRED)
    if(WIN32)
        set(GGML_CUDA_NCCL OFF CACHE BOOL "Disable NCCL for local Windows CUDA builds" FORCE)
    endif()
elseif(VIENEU_NATIVE_BACKEND_NORMALIZED STREQUAL "vulkan")
    set(GGML_VULKAN ON CACHE BOOL "Build ggml Vulkan backend" FORCE)
elseif(VIENEU_NATIVE_BACKEND_NORMALIZED STREQUAL "metal")
    if(NOT APPLE)
        message(FATAL_ERROR "VIENEU_NATIVE_BACKEND=metal is only supported on Apple platforms.")
    endif()
    set(GGML_METAL ON CACHE BOOL "Build ggml Metal backend" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "Embed Metal kernels in the runtime library" FORCE)
endif()
message(STATUS "VieNeu native ggml backend: ${VIENEU_NATIVE_BACKEND_NORMALIZED}")
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Do not build llama.cpp tests" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Do not build llama.cpp tools" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Do not build llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Do not build llama.cpp server" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "Do not build llama.cpp app" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "Do not build llama.cpp UI" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "Do not fetch llama.cpp prebuilt UI" FORCE)
add_subdirectory(${VIENEU_LLAMA_DIR} llama_cpp_build EXCLUDE_FROM_ALL)

if(VIENEU_SEA_G2P)
    if(NOT IS_ABSOLUTE "${VIENEU_SEA_G2P_DIR}")
        get_filename_component(VIENEU_SEA_G2P_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${VIENEU_SEA_G2P_DIR}" ABSOLUTE)
    endif()
    if(NOT EXISTS "${VIENEU_SEA_G2P_DIR}/Cargo.toml")
        message(FATAL_ERROR "sea-g2p Cargo.toml not found at ${VIENEU_SEA_G2P_DIR}")
    endif()
    if(NOT EXISTS "${VIENEU_SEA_G2P_DIR}/include/sea_g2p.h")
        message(FATAL_ERROR "sea-g2p C ABI header not found at ${VIENEU_SEA_G2P_DIR}/include/sea_g2p.h")
    endif()
    if(NOT EXISTS "${VIENEU_SEA_G2P_DIR}/python/sea_g2p/sea_g2p.bin")
        message(FATAL_ERROR "sea-g2p dictionary not found at ${VIENEU_SEA_G2P_DIR}/python/sea_g2p/sea_g2p.bin")
    endif()

    find_program(CARGO_EXECUTABLE cargo REQUIRED)
    set(VIENEU_SEA_G2P_TARGET_DIR "${CMAKE_BINARY_DIR}/sea-g2p")
    if(WIN32)
        set(VIENEU_SEA_G2P_RUNTIME_LIB "${VIENEU_SEA_G2P_TARGET_DIR}/release/sea_g2p_rs.dll")
        set(VIENEU_SEA_G2P_IMPORT_LIB "${VIENEU_SEA_G2P_TARGET_DIR}/release/sea_g2p_rs.dll.lib")
        set(VIENEU_SEA_G2P_LINK_LIB "${VIENEU_SEA_G2P_IMPORT_LIB}")
    elseif(APPLE)
        set(VIENEU_SEA_G2P_RUNTIME_LIB "${VIENEU_SEA_G2P_TARGET_DIR}/release/libsea_g2p_rs.dylib")
        set(VIENEU_SEA_G2P_IMPORT_LIB "${VIENEU_SEA_G2P_RUNTIME_LIB}")
        set(VIENEU_SEA_G2P_LINK_LIB "${VIENEU_SEA_G2P_RUNTIME_LIB}")
    else()
        set(VIENEU_SEA_G2P_RUNTIME_LIB "${VIENEU_SEA_G2P_TARGET_DIR}/release/libsea_g2p_rs.so")
        set(VIENEU_SEA_G2P_IMPORT_LIB "${VIENEU_SEA_G2P_RUNTIME_LIB}")
        set(VIENEU_SEA_G2P_LINK_LIB "${VIENEU_SEA_G2P_RUNTIME_LIB}")
    endif()

    if(WIN32)
        add_custom_command(
            OUTPUT "${VIENEU_SEA_G2P_RUNTIME_LIB}"
            BYPRODUCTS "${VIENEU_SEA_G2P_IMPORT_LIB}"
            COMMAND "${CARGO_EXECUTABLE}" build --release --manifest-path "${VIENEU_SEA_G2P_DIR}/Cargo.toml" --target-dir "${VIENEU_SEA_G2P_TARGET_DIR}"
            WORKING_DIRECTORY "${VIENEU_SEA_G2P_DIR}"
            COMMENT "Building native sea-g2p Rust library"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${VIENEU_SEA_G2P_RUNTIME_LIB}"
            COMMAND "${CARGO_EXECUTABLE}" build --release --manifest-path "${VIENEU_SEA_G2P_DIR}/Cargo.toml" --target-dir "${VIENEU_SEA_G2P_TARGET_DIR}"
            WORKING_DIRECTORY "${VIENEU_SEA_G2P_DIR}"
            COMMENT "Building native sea-g2p Rust library"
            VERBATIM
        )
    endif()
    add_custom_target(vieneu-sea-g2p-rust DEPENDS "${VIENEU_SEA_G2P_RUNTIME_LIB}")
    add_library(vieneu-sea-g2p SHARED IMPORTED GLOBAL)
    set_target_properties(vieneu-sea-g2p PROPERTIES
        IMPORTED_LOCATION "${VIENEU_SEA_G2P_RUNTIME_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${VIENEU_SEA_G2P_DIR}/include"
    )
    if(WIN32)
        set_target_properties(vieneu-sea-g2p PROPERTIES
            IMPORTED_IMPLIB "${VIENEU_SEA_G2P_IMPORT_LIB}"
        )
    endif()
    message(STATUS "VieNeu sea-g2p phonemizer: ON (${VIENEU_SEA_G2P_DIR})")
else()
    message(STATUS "VieNeu sea-g2p phonemizer: OFF")
endif()

# Find ONNX Runtime dependency
unset(ONNXRUNTIME_LIB CACHE)
unset(ONNXRUNTIME_INCLUDE_DIR CACHE)
if(ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_INCLUDE_CANDIDATES
        "${ONNXRUNTIME_ROOT}/include"
        "${ONNXRUNTIME_ROOT}/build/native/include"
    )
    set(ONNXRUNTIME_LIB_CANDIDATES
        "${ONNXRUNTIME_ROOT}/lib"
        "${ONNXRUNTIME_ROOT}/bin"
        "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native"
        "${ONNXRUNTIME_ROOT}"
    )
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_c_api.h
        PATHS ${ONNXRUNTIME_INCLUDE_CANDIDATES}
        NO_DEFAULT_PATH
        REQUIRED
    )
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS ${ONNXRUNTIME_LIB_CANDIDATES}
        NO_DEFAULT_PATH
        REQUIRED
    )
    include_directories(SYSTEM "${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime library found at: ${ONNXRUNTIME_LIB}")
    message(STATUS "ONNX Runtime headers found at: ${ONNXRUNTIME_INCLUDE_DIR}")
else()
    find_library(ONNXRUNTIME_LIB NAMES onnxruntime REQUIRED)
endif()

set(VIENEU_V3_ONNX_SOURCES
    src/vieneu/v3_onnx/vieneu_v3_onnx_engine.cpp
    src/vieneu/v3_onnx/vieneu_v3_onnx_assets.cpp
    src/vieneu/v3_onnx/vieneu_v3_onnx_tokenizer.cpp
    src/vieneu/v3_onnx/vieneu_v3_onnx_audio.cpp
    src/vieneu/v3_onnx/vieneu_v3_onnx_inference.cpp
    src/vieneu/v3_onnx/vieneu_v3_onnx_acoustic_native.cpp
    src/vieneu/v3_onnx/vieneu_v3_onnx_sampling.cpp
    src/vieneu/v3_onnx/vieneu_v3_onnx_voice.cpp
    src/vieneu/v3_onnx/vieneu_v3_onnx_utils.cpp
)

set(VIENEU_V3_NATIVE_SOURCES
    src/vieneu/v3_native/v3_native_assets.cpp
    src/vieneu/v3_native/v3_native_tokenizer.cpp
    src/vieneu/v3_native/v3_native_prompt.cpp
    src/vieneu/v3_native/v3_native_sampler.cpp
    src/vieneu/v3_native/v3_native_backbone_llama.cpp
    src/vieneu/v3_native/v3_native_acoustic_ggml.cpp
    src/vieneu/v3_native/v3_native_moss_codec.cpp
    src/vieneu/v3_native/v3_native_reference.cpp
    src/vieneu/v3_native/v3_native_stream.cpp
    src/vieneu/v3_native/v3_native_batch.cpp
    src/vieneu/v3_native/vieneu_v3_native.cpp
)

# Core static library
add_library(vieneu-tts-core STATIC
    src/backends/llama_backend.cpp
    src/codecs/neucodec_onnx.cpp
    src/vieneu/vieneu.cpp
    ${VIENEU_V3_ONNX_SOURCES}
    ${VIENEU_V3_NATIVE_SOURCES}
)
set_target_properties(vieneu-tts-core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(vieneu-tts-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${VIENEU_LLAMA_DIR}/vendor
)

target_link_libraries(vieneu-tts-core PUBLIC
    llama
    ${ONNXRUNTIME_LIB}
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vieneu-tts-core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(VIENEU_SEA_G2P)
    add_dependencies(vieneu-tts-core vieneu-sea-g2p-rust)
    target_compile_definitions(vieneu-tts-core PRIVATE
        VIENEU_USE_SEA_G2P
        VIENEU_SEA_G2P_DEFAULT_DICT="${VIENEU_SEA_G2P_DIR}/python/sea_g2p/sea_g2p.bin"
    )
    target_link_libraries(vieneu-tts-core PUBLIC vieneu-sea-g2p)
endif()

# Shared library for ABI consumers (vieneu-tts.dll)
add_library(vieneu-tts SHARED
    src/vieneu/vieneu_tts.cpp
)
target_link_libraries(vieneu-tts PUBLIC vieneu-tts-core)
target_include_directories(vieneu-tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)
if(VIENEU_SEA_G2P)
    add_custom_command(TARGET vieneu-tts POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${VIENEU_SEA_G2P_RUNTIME_LIB}"
            "$<TARGET_FILE_DIR:vieneu-tts>"
        COMMENT "Copying sea-g2p runtime library next to vieneu-tts"
        VERBATIM
    )
endif()
if(WIN32 AND VIENEU_NATIVE_BACKEND_NORMALIZED STREQUAL "cuda")
    foreach(VIENEU_CUDA_RUNTIME_DLL cudart64_12.dll cublas64_12.dll cublasLt64_12.dll)
        if(EXISTS "${CUDAToolkit_BIN_DIR}/${VIENEU_CUDA_RUNTIME_DLL}")
            add_custom_command(TARGET vieneu-tts POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CUDAToolkit_BIN_DIR}/${VIENEU_CUDA_RUNTIME_DLL}"
                    "$<TARGET_FILE_DIR:vieneu-tts>"
                COMMENT "Copying CUDA runtime dependency ${VIENEU_CUDA_RUNTIME_DLL} next to vieneu-tts"
                VERBATIM
            )
        else()
            message(FATAL_ERROR "CUDA runtime dependency not found: ${CUDAToolkit_BIN_DIR}/${VIENEU_CUDA_RUNTIME_DLL}")
        endif()
    endforeach()
endif()

# Export options for shared library on Windows
if(MSVC)
    target_compile_definitions(vieneu-tts PRIVATE VIENEU_BUILD_DLL)
endif()

# CLI Executable
add_executable(vieneu-tts-cli
    tools/vieneu-tts-cli.cpp
)
target_link_libraries(vieneu-tts-cli PRIVATE vieneu-tts)

# C ABI verification executable
add_executable(test-abi-c
    tests/abi-c.c
)
set_target_properties(test-abi-c PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
)
if(MSVC)
    target_compile_options(test-abi-c PRIVATE /W4 /WX)
else()
    target_compile_options(test-abi-c PRIVATE -Wall -Werror -pedantic)
endif()
target_include_directories(test-abi-c PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test-abi-c PRIVATE vieneu-tts)

if(BUILD_TESTING)
    add_executable(test-vieneu-phonemizer
        tests/vieneu-phonemizer.cpp
    )
    target_link_libraries(test-vieneu-phonemizer PRIVATE vieneu-tts-core)
    add_test(NAME vieneu-phonemizer COMMAND test-vieneu-phonemizer)

    add_executable(test-v3-native-chunking
        tests/v3-native-chunking.cpp
    )
    target_link_libraries(test-v3-native-chunking PRIVATE vieneu-tts-core)
    add_test(NAME v3-native-chunking COMMAND test-v3-native-chunking)

endif()
