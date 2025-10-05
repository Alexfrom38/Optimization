file(GLOB_RECURSE SRC_SOURCES 
    "*.c"
    "*.cpp"
    "*.cxx"
    "*.cc"
)

file(GLOB_RECURSE SRC_HEADERS 
    "*.h"
    "*.hpp"
    "*.hxx"
    "*.inl"
)

# Исключаем файлы из папки build и других нежелательных директорий
list(FILTER SRC_SOURCES EXCLUDE REGEX ".*/build/.*")
list(FILTER SRC_SOURCES EXCLUDE REGEX ".*/CMakeFiles/.*")
list(FILTER SRC_SOURCES EXCLUDE REGEX ".*/\\.git/.*")

list(FILTER SRC_HEADERS EXCLUDE REGEX ".*/build/.*")
list(FILTER SRC_HEADERS EXCLUDE REGEX ".*/CMakeFiles/.*")
list(FILTER SRC_HEADERS EXCLUDE REGEX ".*/\\.git/.*")

# Вывод информации о найденных файлах
message(STATUS "Found ${SRC_SOURCES} source files in src directory")
foreach(source_file ${SRC_SOURCES})
    message(STATUS "  - ${source_file}")
endforeach()

message(STATUS "Found ${SRC_HEADERS} header files in src directory")
foreach(header_file ${SRC_HEADERS})
    message(STATUS "  - ${header_file}")
endforeach()

# Проверка, что найдены исходные файлы
list(LENGTH SRC_SOURCES SRC_SOURCES_COUNT)
if(SRC_SOURCES_COUNT EQUAL 0)
    message(WARNING "No source files found in src directory!")
endif()

# Создание библиотеки со всеми исходными файлами
add_library(MyProjectLib ${SRC_SOURCES} ${SRC_HEADERS})

# Установка свойств для библиотеки
target_include_directories(MyProjectLib
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Добавление OpenMP к цели
if(OpenMP_CXX_FOUND)
    target_link_libraries(MyProjectLib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Установка свойств для SIMD
if(ENABLE_AVX2)
    target_compile_definitions(MyProjectLib PRIVATE USE_AVX2)
endif()

if(ENABLE_AVX)
    target_compile_definitions(MyProjectLib PRIVATE USE_AVX)
endif()

if(ENABLE_SSE)
    target_compile_definitions(MyProjectLib PRIVATE USE_SSE)
endif()

# Платформо-специфичные настройки
if(WIN32)
    target_compile_definitions(MyProjectLib PRIVATE OS_WINDOWS)
    if(MSVC)
        target_compile_options(MyProjectLib PRIVATE /MP)
    endif()
else()
    target_compile_definitions(MyProjectLib PRIVATE OS_UNIX)
    target_compile_options(MyProjectLib PRIVATE -pthread)
endif()

# Дополнительные флаги компилятора для разных конфигураций
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(MyProjectLib PRIVATE
        $<$<CONFIG:Debug>:-O0 -g3 -DDEBUG>
        $<$<CONFIG:Release>:-O2 -DNDEBUG -flto>
    )
elseif(MSVC)
    target_compile_options(MyProjectLib PRIVATE
        $<$<CONFIG:Debug>:/Zi /Od /MDd /DDEBUG>
        $<$<CONFIG:Release>:/O2 /MD /DNDEBUG>
    )
endif()

# Установка
install(TARGETS MyProjectLib
    EXPORT MyProjectTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${SRC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/MyProject
)