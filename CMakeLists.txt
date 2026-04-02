cmake_minimum_required(VERSION 3.20)
project(StarPU_FWI VERSION 1.0.0 LANGUAGES CXX C)

# C++ Standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Detect CUDA availability
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # nvcc 12.x supports up to GCC 13; use the system default gcc, not gcc14.
    # Set CMAKE_CUDA_HOST_COMPILER explicitly to avoid nvcc picking up gcc14.
    if(NOT CMAKE_CUDA_HOST_COMPILER)
        find_program(GCC_FOR_CUDA NAMES gcc)
        if(GCC_FOR_CUDA)
            set(CMAKE_CUDA_HOST_COMPILER "${GCC_FOR_CUDA}")
        endif()
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Ampere (sm_80/sm_86), Ada Lovelace (sm_89), Hopper (sm_90).
    # Adjust for your hardware or set to "native" for a local build.
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(STARPU_USE_CUDA)
    message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER}")
else()
    message(STATUS "CUDA not found — building CPU-only")
endif()

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
if(CMAKE_CUDA_COMPILER)
    list(APPEND STARFWI_COMMON_SOURCES src/codelets/forward_propagation.cu)
endif()

# Helper function to apply common build settings
function(starfwi_target_setup target)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fPIC>)
    # Prepend GCC 14's private lib directory to the search path so that its
    # libstdc++.so is found before the system's older version (GCC 11 on
    # Amazon Linux 2023), which lacks symbols such as _M_replace_cold added
    # in GCC 12 that our C++23 code references.
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-libgcc-file-name
        OUTPUT_VARIABLE _gcc14_libgcc
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    get_filename_component(_gcc14_libdir "${_gcc14_libgcc}" DIRECTORY)
    if(_gcc14_libdir AND EXISTS "${_gcc14_libdir}")
        target_link_options(${target} PRIVATE -L${_gcc14_libdir})
    endif()

    target_link_options(${target} PRIVATE -rdynamic -Wl,--export-dynamic -Wl,-E)
    target_link_libraries(${target} PRIVATE
        PkgConfig::STARPUMPI
        PkgConfig::STARPU
        MPI::MPI_CXX
        dl
    )
    if(CMAKE_CUDA_COMPILER)
        target_link_libraries(${target} PRIVATE CUDA::cudart)
        # Force gcc14-g++ (the CXX linker) to drive the final link step.
        # Without this, CMake picks nvcc with the system host gcc as the linker
        # driver for targets that contain CUDA sources, which links against the
        # system libstdc++ (GCC 11 on Amazon Linux 2023) and misses symbols
        # added in GCC 12/13/14 that our C++23 code references.
        set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
    endif()
endfunction()

# starfwi-modeling: forward-propagates true model, writes observed seismograms
add_executable(starfwi-modeling
    src/main_modeling.cpp
    ${STARFWI_COMMON_SOURCES}
    src/codelets/save_observed.cpp
)
starfwi_target_setup(starfwi-modeling)

# starfwi-fwi: loads observed data and runs the inversion loop
set(STARFWI_FWI_SOURCES
    src/main_fwi.cpp
    ${STARFWI_COMMON_SOURCES}
    src/codelets/compute_misfit.cpp
    src/codelets/backward_propagation.cpp
)
if(CMAKE_CUDA_COMPILER)
    list(APPEND STARFWI_FWI_SOURCES src/codelets/backward_propagation.cu)
endif()
add_executable(starfwi-fwi ${STARFWI_FWI_SOURCES})
starfwi_target_setup(starfwi-fwi)

# Testing
enable_testing()
add_test(NAME LoadSegyModel COMMAND starfwi-modeling ${CMAKE_SOURCE_DIR}/data/example.segy)
