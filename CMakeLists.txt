cmake_minimum_required(VERSION 3.16)
project(MyProject 
    VERSION 1.0.0
    LANGUAGES C CXX
    DESCRIPTION "Let us rejoice in the god of the machine"
)

include(CTest)
enable_testing()

# Проверка компилятора
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Generator: ${CMAKE_GENERATOR}")

# Настройка стандартов
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Настройка вариантов сборки
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug or Release)")
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang настройки
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    
    # Флаги для Debug сборки
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g3 -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0 -DDEBUG")
    
    # Флаги для Release сборки  
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2 -DNDEBUG -flto")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG -flto")
    
elseif(MSVC)
    # MSVC настройки
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /MP")
    
    # Флаги для Debug сборки
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Zi /Od /MDd /DDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od /MDd /DDEBUG")
    
    # Флаги для Release сборки
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /MD /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /MD /DNDEBUG")
    
    add_compile_options(/wd4996)
endif()

# SIMD поддержка
option(ENABLE_AVX "Enable AVX instructions" ON)
option(ENABLE_AVX2 "Enable AVX2 instructions" ON)
option(ENABLE_SSE "Enable SSE instructions" ON)

if(ENABLE_SSE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-msse -msse2 -msse3)
        message(STATUS "SSE support enabled for GCC/Clang")
    elseif(MSVC)
        add_compile_options(/arch:SSE2)
        message(STATUS "SSE support enabled for MSVC")
    endif()
endif()

if(ENABLE_AVX)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-mavx)
        message(STATUS "AVX support enabled for GCC/Clang")
    elseif(MSVC)
        add_compile_options(/arch:AVX)
        message(STATUS "AVX support enabled for MSVC")
    endif()
endif()

if(ENABLE_AVX2)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-mavx2 -mfma)
        message(STATUS "AVX2 support enabled for GCC/Clang")
    elseif(MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.10)
        add_compile_options(/arch:AVX2)
        message(STATUS "AVX2 support enabled for MSVC")
    else()
        message(WARNING "AVX2 not supported with this MSVC version")
    endif()
endif()

# Поиск OpenMP
find_package(OpenMP)
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP C support found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
else()
    message(WARNING "OpenMP C support not found")
endif()

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP C++ support found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP C++ support not found")
endif()

# Добавление поддиректорий
add_subdirectory(src)
add_subdirectory(tests)

# Установка для пакетов
include(GNUInstallDirs)

export(TARGETS MyProjectLib FILE "${CMAKE_BINARY_DIR}/MyProjectTargets.cmake")