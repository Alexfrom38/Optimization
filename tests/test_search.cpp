#include <gtest/gtest.h>
#include "search.h"
#include <chrono>
#include <vector>
#include <iostream>

class SearchPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Генерируем тестовые данные один раз для всех тестов
        int_data = ArraySearch::generate_test_data(1000000);
        float_data = ArraySearch::generate_float_test_data(1000000);
        
        // Устанавливаем целевые элементы (существующие и несуществующие)
        existing_int_target = int_data[int_data.size() / 2]; // элемент в середине
        non_existing_int_target = -1; // гарантированно не существует
        
        existing_float_target = float_data[float_data.size() / 2];
        non_existing_float_target = -1.0f;
    }
    
    std::vector<int> int_data;
    std::vector<float> float_data;
    int existing_int_target;
    int non_existing_int_target;
    float existing_float_target;
    float non_existing_float_target;
    
    // Вспомогательная функция для измерения времени
    template<typename Func, typename... Args>
    auto measure_time(Func&& func, Args&&... args) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = std::forward<Func>(func)(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        return std::make_pair(result, std::chrono::duration_cast<std::chrono::microseconds>(end - start));
    }
};

// Тест корректности для int
TEST_F(SearchPerformanceTest, CorrectnessInt) {
    int ref_result = ArraySearch::reference_find(int_data, existing_int_target);
    int opt_result = ArraySearch::optimized_find(int_data, existing_int_target);
    int simd_result = ArraySearch::optimized_find_int(int_data.data(), int_data.size(), existing_int_target);
    
    EXPECT_EQ(ref_result, opt_result);
    EXPECT_EQ(ref_result, simd_result);
    
    // Проверка для несуществующего элемента
    EXPECT_EQ(-1, ArraySearch::reference_find(int_data, non_existing_int_target));
    EXPECT_EQ(-1, ArraySearch::optimized_find(int_data, non_existing_int_target));
    EXPECT_EQ(-1, ArraySearch::optimized_find_int(int_data.data(), int_data.size(), non_existing_int_target));
}

// Тест корректности для float
TEST_F(SearchPerformanceTest, CorrectnessFloat) {
    int ref_result = ArraySearch::reference_find(float_data, existing_float_target);
    int opt_result = ArraySearch::optimized_find(float_data, existing_float_target);
    int simd_result = ArraySearch::optimized_find_float(float_data.data(), float_data.size(), existing_float_target);
    
    EXPECT_EQ(ref_result, opt_result);
    EXPECT_EQ(ref_result, simd_result);
}

// Тест производительности для int (существующий элемент)
TEST_F(SearchPerformanceTest, PerformanceIntExisting) {
    const int iterations = 100;
    
    // Референсная реализация
    auto [ref_result, ref_time] = measure_time([&]() {
        int result = 0;
        for (int i = 0; i < iterations; ++i) {
            result += ArraySearch::reference_find(int_data, existing_int_target);
        }
        return result;
    });
    
    // Оптимизированная реализация (развернутый цикл)
    auto [opt_result, opt_time] = measure_time([&]() {
        int result = 0;
        for (int i = 0; i < iterations; ++i) {
            result += ArraySearch::optimized_find(int_data, existing_int_target);
        }
        return result;
    });
    
    // SIMD реализация
    auto [simd_result, simd_time] = measure_time([&]() {
        int result = 0;
        for (int i = 0; i < iterations; ++i) {
            result += ArraySearch::optimized_find_int(int_data.data(), int_data.size(), existing_int_target);
        }
        return result;
    });
    
    EXPECT_EQ(ref_result, opt_result);
    EXPECT_EQ(ref_result, simd_result);
    
    std::cout << "INT EXISTING - Ref: " << ref_time.count() << "μs, "
              << "Opt: " << opt_time.count() << "μs, "
              << "SIMD: " << simd_time.count() << "μs" << std::endl;
    
    // Проверяем что оптимизированные версии быстрее (с запасом на погрешность)
    EXPECT_LT(opt_time.count(), ref_time.count() * 0.9) << "Optimized version should be faster";
    EXPECT_LT(simd_time.count(), ref_time.count() * 0.9) << "SIMD version should be faster";
}

// Тест производительности для int (несуществующий элемент)
TEST_F(SearchPerformanceTest, PerformanceIntNonExisting) {
    const int iterations = 100;
    
    auto [ref_result, ref_time] = measure_time([&]() {
        int result = 0;
        for (int i = 0; i < iterations; ++i) {
            result += ArraySearch::reference_find(int_data, non_existing_int_target);
        }
        return result;
    });
    
    auto [opt_result, opt_time] = measure_time([&]() {
        int result = 0;
        for (int i = 0; i < iterations; ++i) {
            result += ArraySearch::optimized_find(int_data, non_existing_int_target);
        }
        return result;
    });
    
    auto [simd_result, simd_time] = measure_time([&]() {
        int result = 0;
        for (int i = 0; i < iterations; ++i) {
            result += ArraySearch::optimized_find_int(int_data.data(), int_data.size(), non_existing_int_target);
        }
        return result;
    });
    
    EXPECT_EQ(-iterations, ref_result); // Все должны вернуть -1
    EXPECT_EQ(ref_result, opt_result);
    EXPECT_EQ(ref_result, simd_result);
    
    std::cout << "INT NON-EXISTING - Ref: " << ref_time.count() << "μs, "
              << "Opt: " << opt_time.count() << "μs, "
              << "SIMD: " << simd_time.count() << "μs" << std::endl;
}

// Тест производительности для float
TEST_F(SearchPerformanceTest, PerformanceFloat) {
    const int iterations = 100;
    
    auto [ref_result, ref_time] = measure_time([&]() {
        int result = 0;
        for (int i = 0; i < iterations; ++i) {
            result += ArraySearch::reference_find(float_data, existing_float_target);
        }
        return result;
    });
    
    auto [opt_result, opt_time] = measure_time([&]() {
        int result = 0;
        for (int i = 0; i < iterations; ++i) {
            result += ArraySearch::optimized_find(float_data, existing_float_target);
        }
        return result;
    });
    
    auto [simd_result, simd_time] = measure_time([&]() {
        int result = 0;
        for (int i = 0; i < iterations; ++i) {
            result += ArraySearch::optimized_find_float(float_data.data(), float_data.size(), existing_float_target);
        }
        return result;
    });
    
    EXPECT_EQ(ref_result, opt_result);
    EXPECT_EQ(ref_result, simd_result);
    
    std::cout << "FLOAT - Ref: " << ref_time.count() << "μs, "
              << "Opt: " << opt_time.count() << "μs, "
              << "SIMD: " << simd_time.count() << "μs" << std::endl;
}

// Тест с разными размерами массивов
TEST_F(SearchPerformanceTest, DifferentSizes) {
    std::vector<size_t> sizes = {100, 1000, 10000, 100000, 1000000};
    
    for (size_t size : sizes) {
        auto test_data = ArraySearch::generate_test_data(size);
        int target = test_data[size / 2];
        
        auto [ref_result, ref_time] = measure_time([&]() {
            return ArraySearch::reference_find(test_data, target);
        });
        
        auto [simd_result, simd_time] = measure_time([&]() {
            return ArraySearch::optimized_find_int(test_data.data(), test_data.size(), target);
        });
        
        EXPECT_EQ(ref_result, simd_result);
        
        std::cout << "Size " << size << " - Ref: " << ref_time.count() 
                  << "μs, SIMD: " << simd_time.count() << "μs, "
                  << "Speedup: " << (double)ref_time.count() / simd_time.count() << "x" << std::endl;
        
        // Для больших массивов ожидаем большее ускорение
        if (size >= 10000) {
            EXPECT_GT((double)ref_time.count() / simd_time.count(), 1.2) 
                << "SIMD should be significantly faster for large arrays";
        }
    }
}

// Бенчмарк-тест для детального анализа
TEST_F(SearchPerformanceTest, DetailedBenchmark) {
    const size_t size = 1000000;
    const int iterations = 1000;
    auto test_data = ArraySearch::generate_test_data(size);
    int target = test_data[size / 2];
    
    long long ref_total_time = 0;
    long long opt_total_time = 0;
    long long simd_total_time = 0;
    
    for (int i = 0; i < iterations; ++i) {
        auto [_, ref_time] = measure_time([&]() {
            return ArraySearch::reference_find(test_data, target);
        });
        
        auto [__, opt_time] = measure_time([&]() {
            return ArraySearch::optimized_find(test_data, target);
        });
        
        auto [___, simd_time] = measure_time([&]() {
            return ArraySearch::optimized_find_int(test_data.data(), test_data.size(), target);
        });
        
        ref_total_time += ref_time.count();
        opt_total_time += opt_time.count();
        simd_total_time += simd_time.count();
    }
    
    double ref_avg = static_cast<double>(ref_total_time) / iterations;
    double opt_avg = static_cast<double>(opt_total_time) / iterations;
    double simd_avg = static_cast<double>(simd_total_time) / iterations;
    
    std::cout << "\n=== DETAILED BENCHMARK ===" << std::endl;
    std::cout << "Reference avg: " << ref_avg << "μs" << std::endl;
    std::cout << "Optimized avg: " << opt_avg << "μs" << std::endl;
    std::cout << "SIMD avg: " << simd_avg << "μs" << std::endl;
    std::cout << "Optimized speedup: " << ref_avg / opt_avg << "x" << std::endl;
    std::cout << "SIMD speedup: " << ref_avg / simd_avg << "x" << std::endl;
    
    // Записываем результаты для CI/CD или анализа
    RecordProperty("ReferenceTime", ref_avg);
    RecordProperty("OptimizedTime", opt_avg);
    RecordProperty("SIMDTime", simd_avg);
    RecordProperty("OptimizedSpeedup", ref_avg / opt_avg);
    RecordProperty("SIMDSpeedup", ref_avg / simd_avg);
}