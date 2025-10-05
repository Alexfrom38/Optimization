#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

class ArraySearch {
public:
    // Референсная реализация (линейный поиск)
    template<typename T>
    static int reference_find(const std::vector<T>& arr, const T& target);
    
    // Референсная реализация для сырых массивов
    template<typename T>
    static int reference_find(const T* arr, size_t size, const T& target);
    
    // Оптимизированная реализация
    template<typename T>
    static int optimized_find(const std::vector<T>& arr, const T& target);
    
    // Оптимизированная реализация для сырых массивов
    template<typename T>
    static int optimized_find(const T* arr, size_t size, const T& target);
    
    // SIMD-оптимизированная версия для int (использует AVX2/SSE)
    static int optimized_find_int(const int* arr, size_t size, int target);
    
    // SIMD-оптимизированная версия для float
    static int optimized_find_float(const float* arr, size_t size, float target);
    
    // Вспомогательные функции
    static std::vector<int> generate_test_data(size_t size);
    static std::vector<float> generate_float_test_data(size_t size);
    
private:
    // Внутренние реализации
    static int optimized_find_int_avx2(const int* arr, size_t size, int target);
    static int optimized_find_int_sse(const int* arr, size_t size, int target);
    static int optimized_find_float_avx2(const float* arr, size_t size, float target);
    static int optimized_find_float_sse(const float* arr, size_t size, float target);
};