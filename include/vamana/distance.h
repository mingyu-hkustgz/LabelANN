#ifndef DISTANCE
#define DISTANCE

#include <memory>
#include <immintrin.h>
#include <x86intrin.h>
#include <iostream>
#include "config.h"


namespace ANNS {

    // virtual class for distance functions
    class DistanceHandler {
    public:
        virtual float compute(const char *a, const char *b, IdxType dim) const = 0;

        virtual ~DistanceHandler() {}
    };

    // float L2 distance
    class FloatL2DistanceHandler : public DistanceHandler {
    public:
        float compute(const char *a, const char *b, IdxType dim) const {
            const float *x = reinterpret_cast<const float *>(a);
            const float *y = reinterpret_cast<const float *>(b);

            //  AVX-2
            __m256 msum0 = _mm256_setzero_ps();

            while (dim >= 8) {
                __m256 mx = _mm256_loadu_ps(x);
                x += 8;
                __m256 my = _mm256_loadu_ps(y);
                y += 8;
                const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
                msum0 = _mm256_add_ps(msum0, _mm256_mul_ps(a_m_b1, a_m_b1));
                dim -= 8;
            }

            __m128 msum1 = _mm256_extractf128_ps(msum0, 1);
            __m128 msum2 = _mm256_extractf128_ps(msum0, 0);
            msum1 = _mm_add_ps(msum1, msum2);

            if (dim >= 4) {
                __m128 mx = _mm_loadu_ps(x);
                x += 4;
                __m128 my = _mm_loadu_ps(y);
                y += 4;
                const __m128 a_m_b1 = _mm_sub_ps(mx, my);
                msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
                dim -= 4;
            }

            if (dim > 0) {
                __m128 mx = masked_read(dim, x);
                __m128 my = masked_read(dim, y);
                __m128 a_m_b1 = _mm_sub_ps(mx, my);
                msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
            }

            msum1 = _mm_hadd_ps(msum1, msum1);
            msum1 = _mm_hadd_ps(msum1, msum1);
            return _mm_cvtss_f32(msum1);
        }

    private:
        static inline __m128 masked_read(IdxType dim, const float *x) {
            __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
            switch (dim) {
                case 3:
                    buf[2] = x[2];
                case 2:
                    buf[1] = x[1];
                case 1:
                    buf[0] = x[0];
            }
            return _mm_load_ps(buf);
        }
    };

    // float IP distance
    class FloatIPDistanceHandler : public DistanceHandler {
    public:
        float compute(const char *a, const char *b, IdxType dim) const {
            const float *x = reinterpret_cast<const float *>(a);
            const float *y = reinterpret_cast<const float *>(b);

            //  AVX-2
            __m256 msum0 = _mm256_setzero_ps();

            while (dim >= 8) {
                __m256 mx = _mm256_loadu_ps(x);
                x += 8;
                __m256 my = _mm256_loadu_ps(y);
                y += 8;
                msum0 = _mm256_add_ps(msum0, _mm256_mul_ps(my, mx));
                dim -= 8;
            }

            __m128 msum1 = _mm256_extractf128_ps(msum0, 1);
            __m128 msum2 = _mm256_extractf128_ps(msum0, 0);
            msum1 = _mm_add_ps(msum1, msum2);

            if (dim >= 4) {
                __m128 mx = _mm_loadu_ps(x);
                x += 4;
                __m128 my = _mm_loadu_ps(y);
                y += 4;
                msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, my));
                dim -= 4;
            }

            if (dim > 0) {
                __m128 mx = masked_read(dim, x);
                __m128 my = masked_read(dim, y);
                msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, my));
            }

            msum1 = _mm_hadd_ps(msum1, msum1);
            msum1 = _mm_hadd_ps(msum1, msum1);
            return 1.0f - _mm_cvtss_f32(msum1);
        }

    private:
        static inline __m128 masked_read(IdxType dim, const float *x) {
            __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
            switch (dim) {
                case 3:
                    buf[2] = x[2];
                case 2:
                    buf[1] = x[1];
                case 1:
                    buf[0] = x[0];
            }
            return _mm_load_ps(buf);
        }
    };

    // get desired distance handler
    std::unique_ptr<DistanceHandler>
    get_distance_handler(const std::string &data_type, const std::string &dist_fn) {
        if (data_type == "float") {
            if (dist_fn == "L2")
                return std::make_unique<FloatL2DistanceHandler>();
            else if (dist_fn == "IP") {
                return std::make_unique<FloatIPDistanceHandler>();
            } else if (dist_fn == "cosine") {
                std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type
                          << std::endl;
                exit(-1);
            } else {
                std::cerr << "Error: invalid distance function: " << dist_fn << " and data type: " << data_type
                          << std::endl;
                exit(-1);
            }
        } else if (data_type == "int8") {
            std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type
                      << std::endl;
            exit(-1);
        } else if (data_type == "uint8") {
            std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type
                      << std::endl;
            exit(-1);
        } else {
            std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type
                      << std::endl;
            exit(-1);
        }
    }
}

#endif // DISTANCE