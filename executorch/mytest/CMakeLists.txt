cmake_minimum_required(VERSION 3.19)
project(lora_fa_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(_common_compile_options -Wno-deprecated-declarations -ffunction-sections
                            -fdata-sections)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../third-party/gflags)
find_package(gflags REQUIRED)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
get_target_property(_executorch_imported executorch IMPORTED)
if(NOT _executorch_imported)
  executorch_target_link_options_shared_lib(executorch)
endif()

foreach(
  _required_target
  extension_module_static
  extension_tensor
  extension_data_loader
  extension_flat_tensor
  extension_named_data_map
)
  if(NOT TARGET ${_required_target})
    message(
      FATAL_ERROR
        "Missing ${_required_target}. Rebuild/install ExecuTorch with the Android preset."
    )
  endif()
endforeach()

set(_link_libraries
    executorch
    gflags
    extension_module_static
    extension_tensor
    extension_data_loader
    extension_flat_tensor
    extension_named_data_map
)

if(TARGET extension_threadpool)
  list(APPEND _link_libraries extension_threadpool)
endif()

if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND _link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
  if(TARGET optimized_kernels)
    list(APPEND _link_libraries optimized_kernels)
  endif()
  if(TARGET portable_kernels)
    list(APPEND _link_libraries portable_kernels)
  endif()
  get_target_property(_optimized_imported optimized_native_cpu_ops_lib IMPORTED)
  if(NOT _optimized_imported)
    executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
  endif()
elseif(TARGET portable_ops_lib)
  list(APPEND _link_libraries portable_ops_lib portable_kernels)
  get_target_property(_portable_imported portable_ops_lib IMPORTED)
  if(NOT _portable_imported)
    executorch_target_link_options_shared_lib(portable_ops_lib)
  endif()
else()
  message(FATAL_ERROR "No ExecuTorch CPU op registry target found.")
endif()

if(TARGET xnnpack_backend)
  set(_xnnpack_backend_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND _xnnpack_backend_libs kleidiai)
  endif()
  list(APPEND _link_libraries ${_xnnpack_backend_libs})
  get_target_property(_xnnpack_imported xnnpack_backend IMPORTED)
  if(NOT _xnnpack_imported)
    executorch_target_link_options_shared_lib(xnnpack_backend)
  endif()
endif()

if(TARGET extension_memory_allocator)
  list(APPEND _link_libraries extension_memory_allocator)
endif()

if(ANDROID)
  list(APPEND _link_libraries log)
endif()

add_executable(lora_fa_runner lora_fa_runner.cpp)
target_include_directories(lora_fa_runner PRIVATE ${EXECUTORCH_ROOT}/..)
target_link_libraries(lora_fa_runner PRIVATE ${_link_libraries})
target_compile_options(lora_fa_runner PRIVATE ${_common_compile_options})

add_executable(lora_fa_runner_sst2 lora_fa_runner_sst2.cpp)
target_include_directories(lora_fa_runner_sst2 PRIVATE ${EXECUTORCH_ROOT}/..)
target_link_libraries(lora_fa_runner_sst2 PRIVATE ${_link_libraries})
target_compile_options(lora_fa_runner_sst2 PRIVATE ${_common_compile_options})

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(lora_fa_runner)
  target_link_options_gc_sections(lora_fa_runner_sst2)
  if(NOT APPLE)
    target_link_options(lora_fa_runner PRIVATE "LINKER:-s")
    target_link_options(lora_fa_runner_sst2 PRIVATE "LINKER:-s")
  endif()
endif()
