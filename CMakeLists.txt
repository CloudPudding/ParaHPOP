cmake_minimum_required(VERSION 3.22)
# Set this explicitly for other GPUs; 75 is the GTX 1650 used locally.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA compute capabilities")
endif()
project(paraHPOP VERSION 0.1.0 LANGUAGES CXX CUDA)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(PARAHPOP_CPU_ARCH "x86-64-v3" CACHE STRING "Host CPU ISA (empty for compiler default)")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# CMake 3.22 predates nvcc's C++20 support; CUDA 12.5 accepts these flags.
if(CMAKE_VERSION VERSION_LESS 3.25 AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
  set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "--std=c++20")
  set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "--std=c++20")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(nlohmann_json 3.10.5 CONFIG REQUIRED)

add_library(parahpop_options INTERFACE)
target_include_directories(parahpop_options INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/common/engine"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/feta"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/parm"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/brie")
target_link_libraries(parahpop_options INTERFACE
  nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX CUDA::cudart)
target_compile_definitions(parahpop_options INTERFACE
  PARAHPOP_BLOCKSIZE=32 PARAHPOP_OMP_THREADS=8 PARM_BLOCKSIZE=256 PARM_OMP_THREADS=8
  FETA_WARP_SIZE BRIE_INTO_SHARED_LIBRARY PARM_INTO_SHARED_LIBRARY FETA_INTO_SHARED_LIBRARY
  CUDA_API_PER_THREAD_DEFAULT_STREAM=1 JSON_DIAGNOSTICS=0 JSON_USE_IMPLICIT_CONVERSIONS=1)
target_compile_options(parahpop_options INTERFACE
  "$<$<COMPILE_LANGUAGE:CXX>:-ftemplate-backtrace-limit=100>"
  "$<$<COMPILE_LANGUAGE:CUDA>:--std=c++20;-Xcompiler=-fopenmp;-Xcompiler=-ftemplate-backtrace-limit=100;--display-error-number>"
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>")
if(PARAHPOP_CPU_ARCH)
  target_compile_options(parahpop_options INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:-march=${PARAHPOP_CPU_ARCH}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=${PARAHPOP_CPU_ARCH}>")
endif()

file(GLOB_RECURSE BRIE_CUDA CONFIGURE_DEPENDS third_party/brie/brie/*.cu)
file(GLOB_RECURSE BRIE_CPP CONFIGURE_DEPENDS third_party/brie/brie/*.cpp)
set_source_files_properties(${BRIE_CPP} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(third_party/brie/brie/util/cbor.cpp PROPERTIES LANGUAGE CXX)
file(GLOB_RECURSE ENGINE_CUDA CONFIGURE_DEPENDS
  src/common/engine/paraHPOP/model/accelerations/*.cu
  src/common/engine/paraHPOP/model/environment/*.cu
  src/common/engine/paraHPOP/model/math/*.cu
  src/common/engine/paraHPOP/model/physics/*.cu
  src/common/engine/paraHPOP/util/log.cu
  src/common/engine/paraHPOP/util/naming.cu)
list(FILTER ENGINE_CUDA EXCLUDE REGEX "NonDimensional")
# Build both host and device backends from source in one shared model library.
# Keeping one CUDA module avoids duplicate device-global symbol registration.
add_library(parahpop_engine SHARED ${BRIE_CUDA} ${BRIE_CPP} ${ENGINE_CUDA})
target_link_libraries(parahpop_engine PUBLIC parahpop_options)
set_target_properties(parahpop_engine PROPERTIES
  POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_source_files_properties(src/common/driver.cpp PROPERTIES LANGUAGE CUDA)
add_library(parahpop_cli STATIC src/common/driver.cpp)
target_link_libraries(parahpop_cli PUBLIC parahpop_engine)
set_target_properties(parahpop_cli PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
foreach(backend cpu gpu)
  add_executable(parahpop_${backend} src/${backend}/main.cpp)
  target_link_libraries(parahpop_${backend} PRIVATE parahpop_cli)
  set_target_properties(parahpop_${backend} PROPERTIES
    LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "$ORIGIN/../lib")
endforeach()
