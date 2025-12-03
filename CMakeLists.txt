cmake_minimum_required(VERSION 3.20)
project(StarPU_FWI VERSION 1.0.0 LANGUAGES CXX C)

# C++ Standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)

# Find StarPU libraries via pkg-config
pkg_check_modules(STARPU REQUIRED IMPORTED_TARGET starpu-1.4)
pkg_check_modules(STARPUMPI REQUIRED IMPORTED_TARGET starpumpi-1.4)

# Main executable
add_executable(starfwi
    src/main.cpp
    src/utils/cli_parser.cpp
    src/utils/wavelet.cpp
    src/utils/segy_loader.cpp
    src/utils/snapshot_writer.cpp
    src/acoustics/finite_difference_solver.cpp
    src/codelets/forward_propagation.cpp
)

# Include directories
target_include_directories(starfwi PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Compiler and linker options
target_compile_options(starfwi PRIVATE -fPIC)
target_link_options(starfwi PRIVATE
    -rdynamic
    -Wl,--export-dynamic
    -Wl,-E
)

# Link libraries
target_link_libraries(starfwi PRIVATE
    PkgConfig::STARPUMPI
    PkgConfig::STARPU
    MPI::MPI_CXX
    dl
)

# Testing
enable_testing()
add_test(NAME LoadSegyModel COMMAND starfwi ${CMAKE_SOURCE_DIR}/data/example.segy)
