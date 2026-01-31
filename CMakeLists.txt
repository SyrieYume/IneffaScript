# cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
# cmake --build build

cmake_minimum_required (VERSION 3.30)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置C++标准为: C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用链接时优化
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# 禁用RTTI
# add_compile_options(
#     $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-fno-rtti>
#     $<$<CXX_COMPILER_ID:MSVC>:/GR->
# )

# 开启所有警告
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 忽略部分警告
add_compile_options(
    -Wno-unused-variable
    -Wno-unused-value
    -Wno-unused-parameter
    -Wno-unused-private-field
)

# 设置输出目录
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/release")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
endif()

project(Ineffa.js LANGUAGES CXX)

add_executable(IneffaScript)

target_sources(IneffaScript PRIVATE
    "./src/main.cpp"
)