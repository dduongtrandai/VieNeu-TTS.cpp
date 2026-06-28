cmake_minimum_required(VERSION 3.14)
project(vieneu-tts-project LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)

# DLL/Output Directory setup (same as build root)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# MSVC specific compile settings
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
endif()

# Configuration Options
set(VIENEU_LLAMA_DIR "../llama.cpp" CACHE PATH "Path to llama.cpp source directory")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime SDK root")

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

set(GGML_NATIVE OFF CACHE BOOL "Build portable CPU kernels instead of host-native kernels" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Do not build llama.cpp tests" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Do not build llama.cpp tools" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Do not build llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Do not build llama.cpp server" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "Do not build llama.cpp app" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "Do not build llama.cpp UI" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "Do not fetch llama.cpp prebuilt UI" FORCE)
add_subdirectory(${VIENEU_LLAMA_DIR} llama_cpp_build EXCLUDE_FROM_ALL)

# Find ONNX Runtime dependency
if(ONNXRUNTIME_ROOT)
    include_directories(SYSTEM "${ONNXRUNTIME_ROOT}/include")
    link_directories("${ONNXRUNTIME_ROOT}/lib")
    find_library(ONNXRUNTIME_LIB NAMES onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)
    message(STATUS "ONNX Runtime library found at: ${ONNXRUNTIME_LIB}")
else()
    find_library(ONNXRUNTIME_LIB NAMES onnxruntime REQUIRED)
endif()

# Core static library
add_library(vieneu-tts-core STATIC
    src/backends/llama_backend.cpp
    src/codecs/neucodec_onnx.cpp
    src/vieneu/vieneu.cpp
    src/vieneu/vieneu_v3_onnx.cpp
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

# Shared library for ABI consumers (vieneu-tts.dll)
add_library(vieneu-tts SHARED
    src/vieneu/vieneu_tts.cpp
)
target_link_libraries(vieneu-tts PUBLIC vieneu-tts-core)
target_include_directories(vieneu-tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

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
endif()
