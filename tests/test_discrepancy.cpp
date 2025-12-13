#include <gtest/gtest.h>
#include "discrepancy.h"
#include <chrono>
#include <iostream>

const size_t LHS_SIZE = 1000;
const size_t RHS_SIZE = 2000;
const size_t ITER = 100;


class DiscrepancyPerformanceTest : public ::testing::Test {
protected:
    // Вспомогательная функция для измерения времени
    template<typename Func, typename... Args>
    auto measure_time(Func&& func, Args&&... args) {
        auto start = std::chrono::high_resolution_clock::now();
        std::forward<Func>(func)(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }
};

// Тест корректности для float
TEST_F(DiscrepancyPerformanceTest, CorrectnessFloat) {
    std::vector<float> test_mat = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> test_lhs = {1.0f, 2.0f, 3.0f};
    std::vector<float> test_rhs = {4.0f, 5.0f, 6.0f};

    float ref_result = DiscrepancyCalculator::reference_calculate(test_mat, test_lhs, test_rhs);
    float opt_result = DiscrepancyCalculator::optimized_calculate(test_mat, test_lhs, test_rhs);
    float simd_result = DiscrepancyCalculator::optimized_calculate_float(test_mat.data(), test_lhs.data(), test_rhs.data(), 3, 3);
    
    EXPECT_FLOAT_EQ(ref_result, opt_result);
    EXPECT_FLOAT_EQ(ref_result, simd_result);
}

// Тест корректности для double
TEST_F(DiscrepancyPerformanceTest, CorrectnessDouble) {
    std::vector<double> test_mat = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<double> test_lhs = {1.0, 2.0, 3.0};
    std::vector<double> test_rhs = {4.0, 5.0, 6.0};

    double ref_result = DiscrepancyCalculator::reference_calculate(test_mat, test_lhs, test_rhs);
    double opt_result = DiscrepancyCalculator::optimized_calculate(test_mat, test_lhs, test_rhs);
    double simd_result = DiscrepancyCalculator::optimized_calculate_double(test_mat.data(), test_lhs.data(), test_rhs.data(), 3, 3);
    
    EXPECT_DOUBLE_EQ(ref_result, opt_result);
    EXPECT_DOUBLE_EQ(ref_result, simd_result);
}

// Тест производительности для float
TEST_F(DiscrepancyPerformanceTest, PerformanceFloat) {
    std::vector<float> test_mat = DiscrepancyCalculator::generate_test_mat_float(LHS_SIZE, RHS_SIZE);
    std::vector<float> test_lhs = DiscrepancyCalculator::generate_test_vec_float(LHS_SIZE);
    std::vector<float> test_rhs = DiscrepancyCalculator::generate_test_vec_float(RHS_SIZE);
    
    auto ref_time = measure_time([&]() {
        for (size_t i = 0; i < ITER; i++) {
            DiscrepancyCalculator::reference_calculate(test_mat, test_lhs, test_rhs);
        }
    });
    
    auto opt_time = measure_time([&]() {
        for (size_t i = 0; i < ITER; i++) {
            DiscrepancyCalculator::optimized_calculate(test_mat, test_lhs, test_rhs);
        }
    });
    
    auto simd_time = measure_time([&]() {
        for (size_t i = 0; i < ITER; i++) {
            DiscrepancyCalculator::optimized_calculate_float(test_mat.data(), test_lhs.data(), test_rhs.data(), LHS_SIZE, RHS_SIZE);
        }
    });
    
    std::cout << "FLOAT - Ref: " << ref_time.count() << "μs, "
              << "Opt: " << opt_time.count() << "μs, "
              << "SIMD: " << simd_time.count() << "μs" << std::endl;
}

// Тест производительности для double
TEST_F(DiscrepancyPerformanceTest, PerformanceDouble) {
    std::vector<double> test_mat = DiscrepancyCalculator::generate_test_mat_double(LHS_SIZE, RHS_SIZE);
    std::vector<double> test_lhs = DiscrepancyCalculator::generate_test_vec_double(LHS_SIZE);
    std::vector<double> test_rhs = DiscrepancyCalculator::generate_test_vec_double(RHS_SIZE);
    
    auto ref_time = measure_time([&]() {
        for (size_t i = 0; i < ITER; i++) {
            DiscrepancyCalculator::reference_calculate(test_mat, test_lhs, test_rhs);
        }
    });
    
    auto opt_time = measure_time([&]() {
        for (size_t i = 0; i < ITER; i++) {
            DiscrepancyCalculator::optimized_calculate(test_mat, test_lhs, test_rhs);
        }
    });
    
    auto simd_time = measure_time([&]() {
        for (size_t i = 0; i < ITER; i++) {
            DiscrepancyCalculator::optimized_calculate_double(test_mat.data(), test_lhs.data(), test_rhs.data(), LHS_SIZE, RHS_SIZE);
        }
    });
    
    std::cout << "DOUBLE - Ref: " << ref_time.count() << "μs, "
              << "Opt: " << opt_time.count() << "μs, "
              << "SIMD: " << simd_time.count() << "μs" << std::endl;
}

// Тест с разными размерами массивов для float
TEST_F(DiscrepancyPerformanceTest, DifferentSizesFloat) {
    std::vector<size_t> sizes = {100, 500, 1000, 10000, 25000};
    
    std::cout << "FLOAT" << std::endl;
    for (size_t size : sizes) {
        auto test_mat = DiscrepancyCalculator::generate_test_mat_float(size, size);
        auto test_lhs = DiscrepancyCalculator::generate_test_vec_float(size);
        auto test_rhs = DiscrepancyCalculator::generate_test_vec_float(size);

        auto ref_time = measure_time([&]() {
            return DiscrepancyCalculator::reference_calculate(test_mat, test_lhs, test_rhs);
        });
        
        auto opt_time = measure_time([&]() {
            return DiscrepancyCalculator::optimized_calculate(test_mat, test_lhs, test_rhs);
        });
        
        auto simd_time = measure_time([&]() {
            return DiscrepancyCalculator::optimized_calculate_float(test_mat.data(), test_lhs.data(), test_rhs.data(), size, size);
        });

        std::cout << "Size " << size << " - Ref: " << ref_time.count() << "μs" << std::endl
                  << "Opt: " << opt_time.count() << "μs, "
                  << "Speedup: " << (double)ref_time.count() / opt_time.count() << "x" << std::endl
                  << "SIMD: " << simd_time.count() << "μs, "
                  << "Speedup: " << (double)ref_time.count() / simd_time.count() << "x" << std::endl;
    }
}

// Тест с разными размерами массивов для double
TEST_F(DiscrepancyPerformanceTest, DifferentSizesDouble) {
    std::vector<size_t> sizes = {100, 500, 1000, 10000, 25000};
    
    std::cout << "DOUBLE" << std::endl;
    for (size_t size : sizes) {
        auto test_mat = DiscrepancyCalculator::generate_test_mat_double(size, size);
        auto test_lhs = DiscrepancyCalculator::generate_test_vec_double(size);
        auto test_rhs = DiscrepancyCalculator::generate_test_vec_double(size);
        
        auto ref_time = measure_time([&]() {
            return DiscrepancyCalculator::reference_calculate(test_mat, test_lhs, test_rhs);
        });
        
        auto opt_time = measure_time([&]() {
            return DiscrepancyCalculator::optimized_calculate(test_mat, test_lhs, test_rhs);
        });
        
        auto simd_time = measure_time([&]() {
            return DiscrepancyCalculator::optimized_calculate_double(test_mat.data(), test_lhs.data(), test_rhs.data(), size, size);
        });

        std::cout << "Size " << size << " - Ref: " << ref_time.count() << "μs" << std::endl
                  << "Opt: " << opt_time.count() << "μs, "
                  << "Speedup: " << (double)ref_time.count() / opt_time.count() << "x" << std::endl
                  << "SIMD: " << simd_time.count() << "μs, "
                  << "Speedup: " << (double)ref_time.count() / simd_time.count() << "x" << std::endl;
    }
}

// Бенчмарк-тест для детального анализа для float
TEST_F(DiscrepancyPerformanceTest, DetailedBenchmark) {
    const size_t size = 10000;
    auto test_mat = DiscrepancyCalculator::generate_test_mat_float(size, size);
    auto test_lhs = DiscrepancyCalculator::generate_test_vec_float(size);
    auto test_rhs = DiscrepancyCalculator::generate_test_vec_float(size);
    
    long long ref_total_time = 0;
    long long opt_total_time = 0;
    long long simd_total_time = 0;
    
    for (size_t i = 0; i < ITER; ++i) {
        auto ref_time = measure_time([&]() {
            return DiscrepancyCalculator::reference_calculate(test_mat, test_lhs, test_rhs);
        });
        
        auto opt_time = measure_time([&]() {
            return DiscrepancyCalculator::optimized_calculate(test_mat, test_lhs, test_rhs);
        });
        
        auto simd_time = measure_time([&]() {
            return DiscrepancyCalculator::optimized_calculate_float(test_mat.data(), test_lhs.data(), test_rhs.data(), size, size);
        });
        
        ref_total_time += ref_time.count();
        opt_total_time += opt_time.count();
        simd_total_time += simd_time.count();
    }
    
    double ref_avg = static_cast<double>(ref_total_time) / ITER;
    double opt_avg = static_cast<double>(opt_total_time) / ITER;
    double simd_avg = static_cast<double>(simd_total_time) / ITER;
    
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