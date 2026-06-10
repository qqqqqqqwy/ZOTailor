set(TARGET llama-zoo-zo-lora-fa)

add_executable(${TARGET} zo-lora-fa.cpp)

target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (GGML_HEXAGON)
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_HEXAGON=1)
endif()

install(TARGETS ${TARGET} RUNTIME)

set(TARGET llama-zoo-base)

add_executable(${TARGET} base.cpp)

target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (GGML_HEXAGON)
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_HEXAGON=1)
endif()

install(TARGETS ${TARGET} RUNTIME)

set(TARGET llama-zoo-base-lora)

add_executable(${TARGET} base_lora.cpp)

target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (GGML_HEXAGON)
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_HEXAGON=1)
endif()

install(TARGETS ${TARGET} RUNTIME)

set(TARGET llama-zoo-repack)

add_executable(${TARGET} repack.cpp)

target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (GGML_HEXAGON)
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_HEXAGON=1)
endif()

install(TARGETS ${TARGET} RUNTIME)

set(TARGET llama-zoo-mezo)

add_executable(${TARGET} mezo.cpp)

target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (GGML_HEXAGON)
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_HEXAGON=1)
endif()

install(TARGETS ${TARGET} RUNTIME)

set(TARGET llama-zoo-sst2-zo-lora)

add_executable(${TARGET} main.cpp)

target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (GGML_HEXAGON)
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_HEXAGON=1)
endif()

install(TARGETS ${TARGET} RUNTIME)
