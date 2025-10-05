# Загрузка Google Test
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)

# Настройка Google Test
FetchContent_MakeAvailable(googletest)

# Автоматическое нахождение всех тестовых файлов, ИСКЛЮЧАЯ папку build
file(GLOB_RECURSE TEST_SOURCES 
    "*.cpp"
    "*.cxx"
    "*.cc"
    "*.c"
)

# Автоматическое нахождение заголовочных файлов тестов, ИСКЛЮЧАЯ папку build
file(GLOB_RECURSE TEST_HEADERS 
    "*.h"
    "*.hpp"
    "*.hxx"
)

# Исключаем файлы из папки build и других нежелательных директорий
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/build/.*")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/CMakeFiles/.*")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/\\.git/.*")

list(FILTER TEST_HEADERS EXCLUDE REGEX ".*/build/.*")
list(FILTER TEST_HEADERS EXCLUDE REGEX ".*/CMakeFiles/.*")
list(FILTER TEST_HEADERS EXCLUDE REGEX ".*/\\.git/.*")

# Вывод информации о найденных тестовых файлах
message(STATUS "Found ${TEST_SOURCES} test source files in tests directory")
foreach(test_file ${TEST_SOURCES})
    message(STATUS "  - ${test_file}")
endforeach()

# Проверка, что найдены тестовые файлы
list(LENGTH TEST_SOURCES TEST_SOURCES_COUNT)
if(TEST_SOURCES_COUNT EQUAL 0)
    message(WARNING "No test source files found in tests directory!")
endif()

# Создание тестового исполняемого файла со всеми тестовыми файлами
add_executable(MyProjectTests ${TEST_SOURCES} ${TEST_HEADERS})

# Настройка тестов
target_link_libraries(MyProjectTests
    PRIVATE
        MyProjectLib
        GTest::gtest_main
        GTest::gmock
)

# Добавление OpenMP к тестам
if(OpenMP_CXX_FOUND)
    target_link_libraries(MyProjectTests PRIVATE OpenMP::OpenMP_CXX)
endif()

# Включение директорий
target_include_directories(MyProjectTests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

# Дополнительные флаги для тестов
target_compile_definitions(MyProjectTests PRIVATE TESTING=1)

if(WIN32)
    target_compile_definitions(MyProjectTests PRIVATE OS_WINDOWS)
else()
    target_compile_definitions(MyProjectTests PRIVATE OS_UNIX)
endif()

if(TEST_SOURCES_COUNT GREATER 0)
    include(GoogleTest)
    
    # Регистрируем тесты для CTest
    gtest_discover_tests(MyProjectTests
        EXTRA_ARGS "--gtest_output=xml:${CMAKE_BINARY_DIR}/test_results/"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    
    # Альтернативно: можно добавить тест вручную
    add_test(
        NAME MyProject_All_Tests
        COMMAND MyProjectTests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    
    # Дополнительная цель для удобного запуска тестов
    add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS MyProjectTests
        COMMENT "Running tests..."
    )
    
    # Цель для запуска тестов с verbose
    add_custom_target(run_tests_verbose
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose
        DEPENDS MyProjectTests
        COMMENT "Running tests with verbose output..."
    )
endif()

# Установка тестов (опционально)
if(BUILD_TESTING AND TEST_SOURCES_COUNT GREATER 0)
    install(TARGETS MyProjectTests
        DESTINATION ${CMAKE_INSTALL_BINDIR}/tests
    )
endif()