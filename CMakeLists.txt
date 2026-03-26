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

# Sources shared by both binaries
set(STARFWI_COMMON_SOURCES
    src/utils/cli_parser.cpp
    src/utils/wavelet.cpp
    src/utils/segy_loader.cpp
    src/utils/seismogram_io.cpp
    src/utils/snapshot_writer.cpp
    src/acoustics/finite_difference_solver.cpp
    src/acoustics/receiver_recorder.cpp
    src/acoustics/misfit.cpp
    src/codelets/forward_propagation.cpp
)

# Helper function to apply common build settings
function(starfwi_target_setup target)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(${target} PRIVATE -fPIC)
    target_link_options(${target} PRIVATE -rdynamic -Wl,--export-dynamic -Wl,-E)
    target_link_libraries(${target} PRIVATE
        PkgConfig::STARPUMPI
        PkgConfig::STARPU
        MPI::MPI_CXX
        dl
    )
endfunction()

# starfwi-modeling: forward-propagates true model, writes observed seismograms
add_executable(starfwi-modeling
    src/main_modeling.cpp
    ${STARFWI_COMMON_SOURCES}
    src/codelets/save_observed.cpp
)
starfwi_target_setup(starfwi-modeling)

# starfwi-fwi: loads observed data and runs the inversion loop
add_executable(starfwi-fwi
    src/main_fwi.cpp
    ${STARFWI_COMMON_SOURCES}
    src/codelets/compute_misfit.cpp
    src/codelets/backward_propagation.cpp
)
starfwi_target_setup(starfwi-fwi)

# Testing
enable_testing()
add_test(NAME LoadSegyModel COMMAND starfwi-modeling ${CMAKE_SOURCE_DIR}/data/example.segy)
