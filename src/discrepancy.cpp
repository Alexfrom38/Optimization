#include "discrepancy.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <random>
#include <omp.h>
#include <iostream>

const int SEED = 1;

// Референсная реализация
template<typename T>
T DiscrepancyCalculator::reference_calculate(const std::vector<T>& mat, const std::vector<T>& lhs, const std::vector<T>& rhs) {
    return reference_calculate(mat.data(), lhs.data(), rhs.data(), lhs.size(), rhs.size());
}

// Референсная реализация для сырых массивов
template<typename T>
T DiscrepancyCalculator::reference_calculate(const T* mat, const T* lhs, const T* rhs, size_t size_lhs, size_t size_rhs) {
    T discrepancy_square = 0;

    for (size_t i = 0; i < size_rhs; i++) {
        T sum = 0;

        for (size_t j = 0; j < size_lhs; j++) {
            sum += mat[i * size_lhs + j] * lhs[j];
        }

        discrepancy_square += (sum - rhs[i]) * (sum - rhs[i]);
    }

    return sqrt(discrepancy_square);
}




// Оптимизированная реализация
template<typename T>
T DiscrepancyCalculator::optimized_calculate(const std::vector<T>& mat, const std::vector<T>& lhs, const std::vector<T>& rhs) {
    return optimized_calculate(mat.data(), lhs.data(), rhs.data(), lhs.size(), rhs.size());
}

// Оптимизированная реализация для сырых массивов
template<typename T>
T DiscrepancyCalculator::optimized_calculate(const T* mat, const T* lhs, const T* rhs, size_t size_lhs, size_t size_rhs) {
    T discrepancy_square = 0;

#pragma omp parallel for shared(mat, lhs, rhs, size_lhs, size_rhs) reduction(+:discrepancy_square)
    for (size_t i = 0; i < size_rhs; i++) {
        T sum = 0;

        for (size_t j = 0; j < size_lhs; j++) {
            sum += mat[i * size_lhs + j] * lhs[j];
        }

        discrepancy_square += (sum - rhs[i]) * (sum - rhs[i]);
    }

    return sqrt(discrepancy_square);
}




// Специализированная SIMD реализация для float
float DiscrepancyCalculator::optimized_calculate_float(const float* mat, const float* lhs, const float* rhs, size_t size_lhs, size_t size_rhs) {
#if defined(USE_AVX2)
    return optimized_calculate_float_avx2(mat, lhs, rhs, size_lhs, size_rhs);
#elif defined(USE_SSE)
    return optimized_calculate_float_sse(mat, lhs, rhs, size_lhs, size_rhs);
#else
    return optimized_calculate(mat, lhs, rhs, size_lhs, size_rhs);
#endif
}

// Специализированная SIMD реализация для double
double DiscrepancyCalculator::optimized_calculate_double(const double* mat, const double* lhs, const double* rhs, size_t size_lhs, size_t size_rhs) {
#if defined(USE_AVX2)
    return optimized_calculate_double_avx2(mat, lhs, rhs, size_lhs, size_rhs);
#elif defined(USE_SSE)
    return optimized_calculate_double_sse(mat, lhs, rhs, size_lhs, size_rhs);
#else
    return optimized_calculate(mat, lhs, rhs, size_lhs, size_rhs);
#endif
}

// AVX2 реализация для float
float DiscrepancyCalculator::optimized_calculate_float_avx2(const float* mat, const float* lhs, const float* rhs, size_t size_lhs, size_t size_rhs) {
#ifdef __AVX2__
    if (size_lhs >= 8) {
        float discrepancy_square = 0;

#pragma omp parallel for shared(mat, lhs, rhs, size_lhs, size_rhs) reduction(+:discrepancy_square)
        for (size_t i = 0; i < size_rhs; i++) {
            float sum = 0.0f;

            const float* mat_ptr = mat + i * size_lhs;
            const float* mat_end = mat + (i + 1) * size_lhs;
            const float* lhs_ptr = lhs;

            // Обработка по 8 элементов за итерацию
            __m256 vsum = _mm256_setzero_ps();
            while (mat_ptr + 8 <= mat_end) {
                __m256 mat_data = _mm256_loadu_ps(mat_ptr);
                __m256 lhs_data = _mm256_loadu_ps(lhs_ptr);

                __m256 incr = _mm256_mul_ps(mat_data, lhs_data);
                vsum = _mm256_add_ps(vsum, incr);

                // vsum = _mm256_fmadd_ps(va, vb, vsum);

                mat_ptr += 8;
                lhs_ptr += 8;
            }

            __m128 low128 = _mm256_castps256_ps128(vsum);
            __m128 high128 = _mm256_extractf128_ps(vsum, 1);
            __m128 sum128 = _mm_add_ps(low128, high128);
            __m128 sum64 = _mm_hadd_ps(sum128, sum128);
            __m128 sum32 = _mm_hadd_ps(sum64, sum64);
            sum = _mm_cvtss_f32(sum32);

            while (mat_ptr <= mat_end) {
                sum += (*mat_ptr) * (*lhs_ptr);

                mat_ptr += 1;
                lhs_ptr += 1;
            }

            discrepancy_square = discrepancy_square + (sum - rhs[i]) * (sum - rhs[i]);
        }

        return sqrt(discrepancy_square);
    } else {
        return optimized_calculate(mat, lhs, rhs, size_lhs, size_rhs);
    }
#else
    return optimized_calculate(mat, lhs, rhs, size_lhs, size_rhs);
#endif
}

// SSE реализация для float
float DiscrepancyCalculator::optimized_calculate_float_sse(const float* mat, const float* lhs, const float* rhs, size_t size_lhs, size_t size_rhs) {
#ifdef __SSE4_1__
    if (size_lhs >= 4) {
        float discrepancy_square = 0;

#pragma omp parallel for shared(mat, lhs, rhs, size_lhs, size_rhs) reduction(+:discrepancy_square)
        for (size_t i = 0; i < size_rhs; i++) {
            float sum = 0.0f;

            const float* mat_ptr = mat + i * size_lhs;
            const float* mat_end = mat + (i + 1) * size_lhs;
            const float* lhs_ptr = lhs;

            // Обработка по 4 элемента за итерацию
            __m128 vsum = _mm_setzero_ps();
            while (mat_ptr + 4 <= mat_end) {
                __m128 mat_data = _mm_loadu_ps(mat_ptr);
                __m128 lhs_data = _mm_loadu_ps(lhs_ptr);

                __m128 incr = _mm_mul_ps(mat_data, lhs_data);
                vsum = _mm_add_ps(vsum, incr);

                // vsum = _mm_fmadd_ps(va, vb, vsum);

                mat_ptr += 4;
                lhs_ptr += 4;
            }

            vsum = _mm_hadd_ps(vsum, vsum);
            vsum = _mm_hadd_ps(vsum, vsum);
            sum = _mm_cvtss_f32(vsum);

            while (mat_ptr <= mat_end) {
                sum += (*mat_ptr) * (*lhs_ptr);

                mat_ptr += 1;
                lhs_ptr += 1;
            }

            discrepancy_square = discrepancy_square + (sum - rhs[i]) * (sum - rhs[i]);
        }

        return sqrt(discrepancy_square);
    } else {
        return optimized_calculate(mat, lhs, rhs, size_lhs, size_rhs);
    }
#else
    return optimized_calculate(mat, lhs, rhs, size_lhs, size_rhs);
#endif
}

// AVX2 реализация для double
double DiscrepancyCalculator::optimized_calculate_double_avx2(const double* mat, const double* lhs, const double* rhs, size_t size_lhs, size_t size_rhs) {
#ifdef __AVX2__
    if (size_lhs >= 4) {
        float discrepancy_square = 0;

#pragma omp parallel for shared(mat, lhs, rhs, size_lhs, size_rhs) reduction(+:discrepancy_square)
        for (size_t i = 0; i < size_rhs; i++) {
            double sum = 0.0;

            const double* mat_ptr = mat + i * size_lhs;
            const double* mat_end = mat + (i + 1) * size_lhs;
            const double* lhs_ptr = lhs;

            // Обработка по 4 элементов за итерацию
            __m256d vsum = _mm256_setzero_pd();
            while (mat_ptr + 4 <= mat_end) {
                __m256d mat_data = _mm256_loadu_pd(mat_ptr);
                __m256d lhs_data = _mm256_loadu_pd(lhs_ptr);

                __m256d incr = _mm256_mul_pd(mat_data, lhs_data);
                vsum = _mm256_add_pd(vsum, incr);

                // vsum = _mm256_fmadd_pd(va, vb, vsum);

                mat_ptr += 4;
                lhs_ptr += 4;
            }

            __m128d low128 = _mm256_castpd256_pd128(vsum);
            __m128d high128 = _mm256_extractf128_pd(vsum, 1);
            __m128d sum128 = _mm_add_pd(low128, high128);
            __m128d sum64 = _mm_hadd_pd(sum128, sum128);
            sum = _mm_cvtsd_f64(sum64);

            while (mat_ptr <= mat_end) {
                sum += (*mat_ptr) * (*lhs_ptr);

                mat_ptr += 1;
                lhs_ptr += 1;
            }

            discrepancy_square = discrepancy_square + (sum - rhs[i]) * (sum - rhs[i]);
        }

        return sqrt(discrepancy_square);
    } else {
        return optimized_calculate(mat, lhs, rhs, size_lhs, size_rhs);
    }
#else
    return optimized_calculate(mat, lhs, rhs, size_lhs, size_rhs);
#endif
}

// SSE реализация для double
double DiscrepancyCalculator::optimized_calculate_double_sse(const double* mat, const double* lhs, const double* rhs, size_t size_lhs, size_t size_rhs) {
#ifdef __SSE4_1__
    if (size_lhs >= 2) {
        float discrepancy_square = 0;

#pragma omp parallel for shared(mat, lhs, rhs, size_lhs, size_rhs) reduction(+:discrepancy_square)
        for (size_t i = 0; i < size_rhs; i++) {
            double sum = 0.0;

            const double* mat_ptr = mat + i * size_lhs;
            const double* mat_end = mat + (i + 1) * size_lhs;
            const double* lhs_ptr = lhs;

            // Обработка по 2 элемента за итерацию
            __m128d vsum = _mm_setzero_pd();
            while (mat_ptr + 2 <= mat_end) {
                __m128d mat_data = _mm_loadu_pd(mat_ptr);
                __m128d lhs_data = _mm_loadu_pd(lhs_ptr);

                __m128d incr = _mm_mul_pd(mat_data, lhs_data);
                vsum = _mm_add_pd(vsum, incr);

                // vsum = _mm_fmadd_pd(va, vb, vsum);

                mat_ptr += 2;
                lhs_ptr += 2;
            }

            vsum = _mm_hadd_pd(vsum, vsum);
            sum = _mm_cvtsd_f64(vsum);

            while (mat_ptr <= mat_end) {
                sum += (*mat_ptr) * (*lhs_ptr);

                mat_ptr += 1;
                lhs_ptr += 1;
            }

            discrepancy_square = discrepancy_square + (sum - rhs[i]) * (sum - rhs[i]);
        }

        return sqrt(discrepancy_square);
    } else {
        return optimized_calculate(mat, lhs, rhs, size_lhs, size_rhs);
    }
#else
    return optimized_calculate(mat, lhs, rhs, size_lhs, size_rhs);
#endif
}




// Генерация тестовых данных
std::vector<float> DiscrepancyCalculator::generate_test_mat_float(size_t size_lhs, size_t size_rhs) {
    std::vector<float> mat(size_lhs * size_rhs);

    std::mt19937 generator(SEED);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (size_t i = 0; i < size_lhs * size_rhs; i++) {
        mat[i] = distribution(generator);
    }

    return mat;
}

std::vector<float> DiscrepancyCalculator::generate_test_vec_float(size_t size) {
    std::vector<float> lhs(size);

    std::mt19937 generator(SEED);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (size_t i = 0; i < size; i++) {
        lhs[i] = distribution(generator);
    }

    return lhs;
}

std::vector<double> DiscrepancyCalculator::generate_test_mat_double(size_t size_lhs, size_t size_rhs) {
    std::vector<double> mat(size_lhs * size_rhs);

    std::mt19937 generator(SEED);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (size_t i = 0; i < size_lhs * size_rhs; i++) {
        mat[i] = distribution(generator);
    }

    return mat;
}

std::vector<double> DiscrepancyCalculator::generate_test_vec_double(size_t size) {
    std::vector<double> lhs(size);

    std::mt19937 generator(SEED);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (size_t i = 0; i < size; i++) {
        lhs[i] = distribution(generator);
    }

    return lhs;
}




// Явная инстанциация шаблонов
template float DiscrepancyCalculator::reference_calculate<float>(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&);
template double DiscrepancyCalculator::reference_calculate<double>(const std::vector<double>&, const std::vector<double>&, const std::vector<double>&);
template float DiscrepancyCalculator::reference_calculate<float>(const float*, const float*, const float*, size_t, size_t);
template double DiscrepancyCalculator::reference_calculate<double>(const double*, const double*, const double*, size_t, size_t);

template float DiscrepancyCalculator::optimized_calculate<float>(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&);
template double DiscrepancyCalculator::optimized_calculate<double>(const std::vector<double>&, const std::vector<double>&, const std::vector<double>&);
template float DiscrepancyCalculator::optimized_calculate<float>(const float*, const float*, const float*, size_t, size_t);
template double DiscrepancyCalculator::optimized_calculate<double>(const double*, const double*, const double*, size_t, size_t);