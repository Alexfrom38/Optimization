#pragma once

#include <vector>
#include <cstddef>

class DiscrepancyCalculator {
public:
    // Референсная реализация
    template<typename T>
    static T reference_calculate(const std::vector<T>& mat,
                                 const std::vector<T>& lhs,
                                 const std::vector<T>& rhs);

    // Референсная реализация для сырых массивов
    template<typename T>
    static T reference_calculate(const T* mat,
                                 const T* lhs,
                                 const T* rhs,
                                 size_t size_lhs,
                                 size_t size_rhs);

    // Оптимизированная реализация
    template<typename T>
    static T optimized_calculate(const std::vector<T>& mat,
                                 const std::vector<T>& lhs,
                                 const std::vector<T>& rhs);

    // Оптимизированная реализация для сырых массивов
    template<typename T>
    static T optimized_calculate(const T* mat,
                                 const T* lhs,
                                 const T* rhs,
                                 size_t size_lhs,
                                 size_t size_rhs);
    
    // SIMD-оптимизированная версия для float
    static float optimized_calculate_float(const float* mat,
                                           const float* lhs,
                                           const float* rhs,
                                           size_t size_lhs,
                                           size_t size_rhs);
    
    // SIMD-оптимизированная версия для double
    static double optimized_calculate_double(const double* mat,
                                             const double* lhs,
                                             const double* rhs,
                                             size_t size_lhs,
                                             size_t size_rhs);
    
    // Генерация тестовых данных
    static std::vector<float> generate_test_mat_float(size_t size_lhs, size_t size_rhs);
    static std::vector<float> generate_test_vec_float(size_t size);

    static std::vector<double> generate_test_mat_double(size_t size_lhs, size_t size_rhs);
    static std::vector<double> generate_test_vec_double(size_t size);
    
private:
    // Внутренние реализации
    static float optimized_calculate_float_avx2(const float* mat,
                                                const float* lhs,
                                                const float* rhs,
                                                size_t size_lhs,
                                                size_t size_rhs);
    static float optimized_calculate_float_sse(const float* mat,
                                               const float* lhs,
                                               const float* rhs,
                                               size_t size_lhs,
                                               size_t size_rhs);
    static double optimized_calculate_double_avx2(const double* mat,
                                                  const double* lhs,
                                                  const double* rhs,
                                                  size_t size_lhs,
                                                  size_t size_rhs);
    static double optimized_calculate_double_sse(const double* mat,
                                                 const double* lhs,
                                                 const double* rhs,
                                                 size_t size_lhs,
                                                 size_t size_rhs);
};