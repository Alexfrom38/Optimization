#include "search.h"
#include <immintrin.h>
#include <algorithm>
#include <random>

// Референсная реализация (линейный поиск)
template<typename T>
int ArraySearch::reference_find(const std::vector<T>& arr, const T& target) {
    return reference_find(arr.data(), arr.size(), target);
}

template<typename T>
int ArraySearch::reference_find(const T* arr, size_t size, const T& target) {
    for (size_t i = 0; i < size; ++i) {
        if (arr[i] == target) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// Оптимизированная реализация (развернутый цикл + предсказание ветвлений)
template<typename T>
int ArraySearch::optimized_find(const std::vector<T>& arr, const T& target) {
    return optimized_find(arr.data(), arr.size(), target);
}

template<typename T>
int ArraySearch::optimized_find(const T* arr, size_t size, const T& target) {
    const T* ptr = arr;
    const T* end = arr + size;
    
    // Обработка по 4 элемента за итерацию (развернутый цикл)
    while (ptr + 4 <= end) {
        if (ptr[0] == target) return static_cast<int>(ptr - arr);
        if (ptr[1] == target) return static_cast<int>(ptr - arr + 1);
        if (ptr[2] == target) return static_cast<int>(ptr - arr + 2);
        if (ptr[3] == target) return static_cast<int>(ptr - arr + 3);
        ptr += 4;
    }
    
    // Обработка оставшихся элементов
    while (ptr < end) {
        if (*ptr == target) return static_cast<int>(ptr - arr);
        ++ptr;
    }
    
    return -1;
}

// Специализированная SIMD реализация для int
int ArraySearch::optimized_find_int(const int* arr, size_t size, int target) {
#if defined(USE_AVX2)
    return optimized_find_int_avx2(arr, size, target);
#elif defined(USE_SSE)
    return optimized_find_int_sse(arr, size, target);
#else
    return optimized_find(arr, size, target);
#endif
}

// Специализированная SIMD реализация для float
int ArraySearch::optimized_find_float(const float* arr, size_t size, float target) {
#if defined(USE_AVX2)
    return optimized_find_float_avx2(arr, size, target);
#elif defined(USE_SSE)
    return optimized_find_float_sse(arr, size, target);
#else
    return optimized_find(arr, size, target);
#endif
}

// AVX2 реализация для int
int ArraySearch::optimized_find_int_avx2(const int* arr, size_t size, int target) {
#ifdef __AVX2__
    const int* ptr = arr;
    const int* end = arr + size;
    
    __m256i target_vec = _mm256_set1_epi32(target);
    
    // Обработка по 8 элементов за итерацию
    while (ptr + 8 <= end) {
        __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
        __m256i cmp = _mm256_cmpeq_epi32(data, target_vec);
        int mask = _mm256_movemask_epi8(cmp);
        
        if (mask != 0) {
            // Найдено совпадение
            for (int i = 0; i < 8; ++i) {
                if (ptr[i] == target) {
                    return static_cast<int>(ptr - arr + i);
                }
            }
        }
        ptr += 8;
    }
    
    // Обработка оставшихся элементов
    return optimized_find(ptr, end - ptr, target);
#else
    return optimized_find(arr, size, target);
#endif
}

// SSE реализация для int
int ArraySearch::optimized_find_int_sse(const int* arr, size_t size, int target) {
#ifdef __SSE4_1__
    const int* ptr = arr;
    const int* end = arr + size;
    
    __m128i target_vec = _mm_set1_epi32(target);
    
    // Обработка по 4 элемента за итерацию
    while (ptr + 4 <= end) {
        __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
        __m128i cmp = _mm_cmpeq_epi32(data, target_vec);
        int mask = _mm_movemask_ps(_mm_castsi128_ps(cmp));
        
        if (mask != 0) {
            // Найдено совпадение
            for (int i = 0; i < 4; ++i) {
                if (ptr[i] == target) {
                    return static_cast<int>(ptr - arr + i);
                }
            }
        }
        ptr += 4;
    }
    
    // Обработка оставшихся элементов
    return optimized_find(ptr, end - ptr, target);
#else
    return optimized_find(arr, size, target);
#endif
}

// AVX2 реализация для float
int ArraySearch::optimized_find_float_avx2(const float* arr, size_t size, float target) {
#ifdef __AVX2__
    const float* ptr = arr;
    const float* end = arr + size;
    
    __m256 target_vec = _mm256_set1_ps(target);
    
    // Обработка по 8 элементов за итерацию
    while (ptr + 8 <= end) {
        __m256 data = _mm256_loadu_ps(ptr);
        __m256 cmp = _mm256_cmp_ps(data, target_vec, _CMP_EQ_OQ);
        int mask = _mm256_movemask_ps(cmp);
        
        if (mask != 0) {
            // Найдено совпадение
            for (int i = 0; i < 8; ++i) {
                if (ptr[i] == target) {
                    return static_cast<int>(ptr - arr + i);
                }
            }
        }
        ptr += 8;
    }
    
    // Обработка оставшихся элементов
    return optimized_find(ptr, end - ptr, target);
#else
    return optimized_find(arr, size, target);
#endif
}

// SSE реализация для float
int ArraySearch::optimized_find_float_sse(const float* arr, size_t size, float target) {
#ifdef __SSE4_1__
    const float* ptr = arr;
    const float* end = arr + size;
    
    __m128 target_vec = _mm_set1_ps(target);
    
    // Обработка по 4 элемента за итерацию
    while (ptr + 4 <= end) {
        __m128 data = _mm_loadu_ps(ptr);
        __m128 cmp = _mm_cmpeq_ps(data, target_vec);
        int mask = _mm_movemask_ps(cmp);
        
        if (mask != 0) {
            // Найдено совпадение
            for (int i = 0; i < 4; ++i) {
                if (ptr[i] == target) {
                    return static_cast<int>(ptr - arr + i);
                }
            }
        }
        ptr += 4;
    }
    
    // Обработка оставшихся элементов
    return optimized_find(ptr, end - ptr, target);
#else
    return optimized_find(arr, size, target);
#endif
}

// Генерация тестовых данных
std::vector<int> ArraySearch::generate_test_data(size_t size) {
    std::vector<int> data(size);
    std::iota(data.begin(), data.end(), 1); // Заполняем последовательными числами
    
    // Перемешиваем для реалистичности
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
    
    return data;
}

std::vector<float> ArraySearch::generate_float_test_data(size_t size) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1000.0f);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    
    return data;
}

// Явная инстанциация шаблонов
template int ArraySearch::reference_find<int>(const std::vector<int>&, const int&);
template int ArraySearch::reference_find<float>(const std::vector<float>&, const float&);
template int ArraySearch::reference_find<int>(const int*, size_t, const int&);
template int ArraySearch::reference_find<float>(const float*, size_t, const float&);

template int ArraySearch::optimized_find<int>(const std::vector<int>&, const int&);
template int ArraySearch::optimized_find<float>(const std::vector<float>&, const float&);
template int ArraySearch::optimized_find<int>(const int*, size_t, const int&);
template int ArraySearch::optimized_find<float>(const float*, size_t, const float&);