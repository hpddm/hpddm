/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2015-10-21

   Copyright (C) 2015      Eidgenössische Technische Hochschule Zürich
                 2016-     Centre National de la Recherche Scientifique

   HPDDM is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   HPDDM is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with HPDDM.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _HPDDM_BLAS_
#define _HPDDM_BLAS_

#define HPDDM_GENERATE_EXTERN_BLAS(C, T)                                                                     \
void    HPDDM_F77(C ## axpy)(const int*, const T*, const T*, const int*, T*, const int*);                    \
void    HPDDM_F77(C ## scal)(const int*, const T*, T*, const int*);                                          \
void    HPDDM_F77(C ## lacpy)(const char*, const int*, const int*, const T*, const int*, T*, const int*);    \
void    HPDDM_F77(C ## gemv)(const char*, const int*, const int*, const T*,                                  \
                             const T*, const int*, const T*, const int*,                                     \
                             const T*, T*, const int*);                                                      \
void    HPDDM_F77(C ## symv)(const char*, const int*, const T*, const T*, const int*,                        \
                             const T*, const int*, const T*, T*, const int*);                                \
void    HPDDM_F77(C ## trsv)(const char*, const char*, const char*, const int*, const T*, const int*,        \
                             T*, const int*);                                                                \
void    HPDDM_F77(C ## gemm)(const char*, const char*, const int*, const int*, const int*,                   \
                             const T*, const T*, const int*, const T*, const int*,                           \
                             const T*, T*, const int*);                                                      \
void    HPDDM_F77(C ## symm)(const char*, const char*, const int*, const int*,                               \
                             const T*, const T*, const int*, const T*, const int*,                           \
                             const T*, T*, const int*);                                                      \
void    HPDDM_F77(C ## trmm)(const char*, const char*, const char*, const char*, const int*, const int*,     \
                             const T*, const T*, const int*, T*, const int*);                                \
void    HPDDM_F77(C ## trsm)(const char*, const char*, const char*, const char*, const int*, const int*,     \
                             const T*, const T*, const int*, T*, const int*);
#define HPDDM_GENERATE_EXTERN_BLAS_COMPLEX_VOID(C, T, B, U)                                                  \
HPDDM_GENERATE_EXTERN_BLAS(B, U)                                                                             \
HPDDM_GENERATE_EXTERN_BLAS(C, T)                                                                             \
void HPDDM_F77(B ## syr)(const char* const, const int* const, const U* const, const U* const,                \
                         const int* const, U* const, const int* const);                                      \
void HPDDM_F77(C ## her)(const char* const, const int* const, const U* const, const T* const,                \
                         const int* const, T* const, const int* const);                                      \
void HPDDM_F77(B ## syr2)(const char* const, const int* const, const U* const, const U* const,               \
                          const int* const, const U* const, const int* const, U* const, const int* const);   \
void HPDDM_F77(C ## her2)(const char* const, const int* const, const T* const, const T* const,               \
                          const int* const, const T* const, const int* const, T* const, const int* const);   \
void HPDDM_F77(B ## syrk)(const char* const, const char* const, const int* const, const int* const,          \
                          const U* const, const U* const, const int* const, const U* const, U* const,        \
                          const int* const);                                                                 \
void HPDDM_F77(C ## herk)(const char* const, const char* const, const int* const, const int* const,          \
                          const U* const, const T* const, const int* const, const U* const, T* const,        \
                          const int* const);
#define HPDDM_GENERATE_EXTERN_BLAS_COMPLEX(C, T, B, U)                                                       \
U HPDDM_F77(B ## nrm2)(const int*, const U*, const int*);                                                    \
U HPDDM_F77(B ## C ## nrm2)(const int*, const T*, const int*);                                               \
U HPDDM_F77(B ## dot)(const int*, const U*, const int*, const U*, const int*);                               \
HPDDM_GENERATE_EXTERN_BLAS_COMPLEX_VOID(C, T, B, U)

#if HPDDM_MKL
# define HPDDM_GENERATE_EXTERN_GEMM3M(C, T)                                                                  \
void HPDDM_F77(C ## gemm3m)(const char*, const char*, const int*, const int*, const int*,                    \
                            const T*, const T*, const int*, const T*, const int*,                            \
                            const T*, T*, const int*);
# if !defined(INTEL_MKL_VERSION) || INTEL_MKL_VERSION < 110300
#  define HPDDM_GENERATE_EXTERN_GEMMT(C, T)
# else
#  define HPDDM_GENERATE_EXTERN_GEMMT(C, T)                                                                  \
void HPDDM_F77(C ## gemmt)(const char*, const char*, const char*, const int*, const int*,                    \
                           const T*, const T*, const int*, const T*, const int*,                             \
                           const T*, T*, const int*);
# endif
# define HPDDM_GENERATE_EXTERN_MKL_EXTENSIONS(C, T, B, U)                                                    \
HPDDM_GENERATE_EXTERN_GEMM3M(C, T)                                                                           \
HPDDM_GENERATE_EXTERN_GEMMT(B, U)                                                                            \
HPDDM_GENERATE_EXTERN_GEMMT(C, T)
#endif

#ifndef INTEL_MKL_VERSION
# ifdef __cplusplus
extern "C" {
HPDDM_GENERATE_EXTERN_BLAS_COMPLEX_VOID(c, std::complex<float>, s, float)
HPDDM_GENERATE_EXTERN_BLAS_COMPLEX(z, std::complex<double>, d, double)
#  if defined(__APPLE__) || HPDDM_MKL
#   if !HPDDM_MKL
#    define HPDDM_PREFIX_AXPBY(func) catlas_ ## func
#   else
HPDDM_GENERATE_EXTERN_MKL_EXTENSIONS(c, std::complex<float>, s, float)
HPDDM_GENERATE_EXTERN_MKL_EXTENSIONS(z, std::complex<double>, d, double)
#   endif
#   ifndef CBLAS_H
#    define HPDDM_GENERATE_EXTERN_AXPBY(C, T, B, U)                                                          \
void HPDDM_PREFIX_AXPBY(B ## axpby)(const int, const U, const U*,                                            \
                                    const int, const U, U*, const int);                                      \
void HPDDM_PREFIX_AXPBY(C ## axpby)(const int, const T*, const T*,                                           \
                                    const int, const T*, T*, const int);
HPDDM_GENERATE_EXTERN_AXPBY(c, std::complex<float>, s, float)
HPDDM_GENERATE_EXTERN_AXPBY(z, std::complex<double>, d, double)
#   endif
#  endif
}
# else
HPDDM_GENERATE_EXTERN_BLAS_COMPLEX_VOID(c, void, s, float)
HPDDM_GENERATE_EXTERN_BLAS_COMPLEX(z, void, d, double)
# endif // __cplusplus
#endif // INTEL_MKL_VERSION

#ifdef __cplusplus
namespace HPDDM {
/* Class: Blas
 *
 *  A class that wraps most of BLAS routines for dense linear algebra.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
struct Blas {
    /* Function: axpy
     *  Computes a scalar-vector product and adds the result to a vector. */
    static void axpy(const int* const, const K* const, const K* const, const int* const, K* const, const int* const);
    template<class U, class V, typename std::enable_if<!(std::is_same<U, V>::value && std::is_same<U, K>::value && std::is_same<V, K>::value)>::type* = nullptr>
    static void axpy(const int* const n, const K* const a, const U* const x, const int* const incx, V* const y, const int* const incy) {
        const U alpha(*a);
        for(int i = 0, j = 0, k = 0; i < *n; ++i, j += *incx, k += *incy) y[k] += alpha * x[j];
    }
    /* Function: axpby
     *  Computes two scalar-vector products. */
    static void axpby(const int&, const K&, const K* const, const int&, const K&, K* const, const int&);
    /* Function: scal
     *  Computes the product of a vector by a scalar. */
    static void scal(const int* const, const K* const, K* const, const int* const);
    /* Function: nrm2
     *  Computes the Euclidean norm of a vector. */
    template<class U, typename std::enable_if<!std::is_same<underlying_type<U>, float>::value>::type* = nullptr>
    static underlying_type<U> nrm2(const int* const, const U* const, const int* const);
    template<class U, typename std::enable_if<std::is_same<underlying_type<U>, float>::value>::type* = nullptr>
    static underlying_type<U> nrm2(const int* const n, const U* const x, const int* const incx) {
        underlying_type<U> sum = underlying_type<U>();
        for(int i = 0, j = 0; i < *n; ++i, j += *incx) sum += std::norm(x[j]);
        return std::sqrt(sum);
    }
    /* Function: dot
     *  Computes a vector-vector dot product. */
    template<class U, typename std::enable_if<std::is_same<U, double>::value>::type* = nullptr>
    static U dot(const int* const, const U* const, const int* const, const U* const, const int* const);
    template<class U, typename std::enable_if<std::is_same<U, float>::value>::type* = nullptr>
    static U dot(const int* const n, const U* const x, const int* const incx, const U* const y, const int* const incy) {
        U sum = U();
        for(int i = 0, j = 0, k = 0; i < *n; ++i, j += *incx, k += *incy) sum += x[j] * y[k];
        return sum;
    }
    template<class U, typename std::enable_if<!std::is_same<U, float>::value && !std::is_same<U, double>::value>::type* = nullptr>
    static U dot(const int* const n, const U* const x, const int* const incx, const U* const y, const int* const incy) {
        U sum = U();
        for(int i = 0, j = 0, k = 0; i < *n; ++i, j += *incx, k += *incy) sum += std::conj(x[j]) * y[k];
        return sum;
    }
    /* Function: lacpy
     *  Copies all or part of a two-dimensional matrix. */
    static void lacpy(const char* const, const int* const, const int* const, const K* const, const int* const, K* const, const int* const);

    /* Function: gemv
     *  Computes a scalar-matrix-vector product. */
    static void gemv(const char* const, const int* const, const int* const, const K* const, const K* const,
                     const int* const, const K* const, const int* const, const K* const, K* const, const int* const);
    /* Function: symv
     *  Computes a symmetric scalar-matrix-vector product. */
    static void symv(const char* const, const int* const, const K* const, const K* const, const int* const,
                     const K* const, const int* const, const K* const, K* const, const int* const);
    /* Function: trsv
     *  Solves a system of linear equations with a triangular matrix and a single right-hand side. */
    static void trsv(const char* const, const char* const, const char* const, const int* const, const K* const, const int* const, K* const, const int* const);
    /* Function: her
     *  Computes a rank-1 update of a symmetric or Hermitian matrix. */
    static void her(const char* const, const int* const, const underlying_type<K>* const, const K* const, const int* const, K* const, const int* const);
    /* Function: her2
     *  Computes a rank-2 update of a symmetric or Hermitian matrix. */
    static void her2(const char* const, const int* const, const K* const, const K* const, const int* const, const K* const, const int* const, K* const, const int* const);

    /* Function: gemm
     *  Computes a scalar-matrix-matrix product. */
    static void gemm(const char* const, const char* const, const int* const, const int* const, const int* const, const K* const, const K* const,
                     const int* const, const K* const, const int* const, const K* const, K* const, const int* const);
    /* Function: gemmt
     *  Computes a scalar-matrix-matrix product but updates only one triangular part of the output matrix. */
    static void gemmt(const char* const, const char* const, const char* const, const int* const, const int* const, const K* const, const K* const,
                      const int* const, const K* const, const int* const, const K* const, K* const, const int* const);
    /* Function: herk
     *  Computes a Hermitian rank-k update. */
    static void herk(const char* const, const char* const, const int* const, const int* const, const underlying_type<K>* const, const K* const,
                     const int* const, const underlying_type<K>* const, K* const, const int* const);
    /* Function: symm
     *  Computes a symmetric scalar-matrix-matrix product. */
    static void symm(const char* const, const char* const, const int* const, const int* const, const K* const, const K* const,
                     const int* const, const K* const, const int* const, const K* const, K* const, const int* const);
    /* Function: trmm
     *  Computes a triangular matrix-matrix product. */
    static void trmm(const char*, const char*, const char*, const char*, const int*, const int*,
                     const K*, const K*, const int*, K*, const int*);
    /* Function: trsm
     *  Solves a system of linear equations with a triangular matrix. */
    static void trsm(const char*, const char*, const char*, const char*, const int*, const int*,
                     const K*, const K*, const int*, K*, const int*);
};

# define HPDDM_GENERATE_GEMM(C, T)                                                                           \
template<>                                                                                                   \
inline void Blas<T>::gemm(const char* const transa, const char* const transb, const int* const m,            \
                          const int* const n, const int* const k, const T* const alpha, const T* const a,    \
                          const int* const lda, const T* const b, const int* const ldb, const T* const beta, \
                          T* const c, const int* const ldc) {                                                \
    HPDDM_F77(C ## gemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);                      \
}
# if !HPDDM_MKL || !defined(INTEL_MKL_VERSION) || INTEL_MKL_VERSION < 110300
#  define HPDDM_GENERATE_GEMMT(C, T)                                                                         \
template<>                                                                                                   \
inline void Blas<T>::gemmt(const char* const, const char* const transa, const char* const transb,            \
                           const int* const n, const int* const k, const T* const alpha, const T* const a,   \
                           const int* const lda, const T* const b, const int* const ldb, const T* const beta,\
                           T* const c, const int* const ldc) {                                               \
    HPDDM_F77(C ## gemm)(transa, transb, n, n, k, alpha, a, lda, b, ldb, beta, c, ldc);                      \
}
# else
#  define HPDDM_GENERATE_GEMMT(C, T)                                                                         \
template<>                                                                                                   \
inline void Blas<T>::gemmt(const char* const uplo, const char* const transa, const char* const transb,       \
                           const int* const n, const int* const k, const T* const alpha, const T* const a,   \
                           const int* const lda, const T* const b, const int* const ldb, const T* const beta,\
                           T* const c, const int* const ldc) {                                               \
    HPDDM_F77(C ## gemmt)(uplo, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);                  \
}
# endif
# if !HPDDM_MKL
#  define HPDDM_GENERATE_GEMM_COMPLEX(C, T) HPDDM_GENERATE_GEMM(C, T)
# else
#  define HPDDM_GENERATE_GEMM_COMPLEX(C, T)                                                                  \
template<>                                                                                                   \
inline void Blas<T>::gemm(const char* const transa, const char* const transb, const int* const m,            \
                          const int* const n, const int* const k, const T* const alpha, const T* const a,    \
                          const int* const lda, const T* const b, const int* const ldb, const T* const beta, \
                          T* const c, const int* const ldc) {                                                \
    HPDDM_F77(C ## gemm3m)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);                    \
}
# endif
# define HPDDM_GENERATE_BLAS(C, T)                                                                           \
template<>                                                                                                   \
inline void Blas<T>::axpy(const int* const n, const T* const a, const T* const x, const int* const incx,     \
                          T* const y, const int* const incy) {                                               \
    HPDDM_F77(C ## axpy)(n, a, x, incx, y, incy);                                                            \
}                                                                                                            \
template<>                                                                                                   \
inline void Blas<T>::scal(const int* const n, const T* const a, T* const x, const int* const incx) {         \
    HPDDM_F77(C ## scal)(n, a, x, incx);                                                                     \
}                                                                                                            \
template<>                                                                                                   \
inline void Blas<T>::lacpy(const char* const uplo, const int* const m, const int* const n,                   \
                           const T* const a, const int* const lda, T* const b, const int* const ldb) {       \
    HPDDM_F77(C ## lacpy)(uplo, m, n, a, lda, b, ldb);                                                       \
}                                                                                                            \
                                                                                                             \
template<>                                                                                                   \
inline void Blas<T>::gemv(const char* const trans, const int* const m, const int* const n,                   \
                          const T* const alpha, const T* const a, const int* const lda, const T* const x,    \
                          const int* const incx, const T* const beta, T* const y, const int* const incy) {   \
    HPDDM_F77(C ## gemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);                                \
}                                                                                                            \
template<>                                                                                                   \
inline void Blas<T>::symv(const char* const uplo, const int* const n, const T* const alpha, const T* const a,\
                          const int* const lda, const T* const x, const int* const incx, const T* const beta,\
                          T* const y, const int* const incy) {                                               \
    HPDDM_F77(C ## symv)(uplo, n, alpha, a, lda, x, incx, beta, y, incy);                                    \
}                                                                                                            \
template<>                                                                                                   \
inline void Blas<T>::trsv(const char* const uplo, const char* const trans, const char* const diag,           \
                          const int* const n, const T* const a, const int* const lda, T* const x,            \
                          const int* const incx) {                                                           \
    HPDDM_F77(C ## trsv)(uplo, trans, diag, n, a, lda, x, incx);                                             \
}                                                                                                            \
                                                                                                             \
HPDDM_GENERATE_GEMMT(C, T)                                                                                   \
template<>                                                                                                   \
inline void Blas<T>::symm(const char* const side, const char* const uplo, const int* const m,                \
                          const int* const n, const T* const alpha, const T* const a, const int* const lda,  \
                          const T* const b, const int* const ldb, const T* const beta,                       \
                          T* const c, const int* const ldc) {                                                \
    HPDDM_F77(C ## symm)(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);                             \
}                                                                                                            \
template<>                                                                                                   \
inline void Blas<T>::trmm(const char* const side, const char* const uplo, const char* const transa,          \
                          const char* const diag, const int* const m, const int* const n,                    \
                          const T* const alpha, const T* const a, const int* const lda,                      \
                          T* const b, const int* const ldb) {                                                \
    HPDDM_F77(C ## trmm)(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);                             \
}                                                                                                            \
template<>                                                                                                   \
inline void Blas<T>::trsm(const char* const side, const char* const uplo, const char* const transa,          \
                          const char* const diag, const int* const m, const int* const n,                    \
                          const T* const alpha, const T* const a, const int* const lda,                      \
                          T* const b, const int* const ldb) {                                                \
    HPDDM_F77(C ## trsm)(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);                             \
}
# if HPDDM_MKL || defined(__APPLE__)
#  define HPDDM_GENERATE_AXPBY(C, T, B, U)                                                                   \
template<>                                                                                                   \
inline void Blas<U>::axpby(const int& n, const U& alpha, const U* const u, const int& incx,                  \
                           const U& beta, U* const v, const int& incy) {                                     \
    HPDDM_PREFIX_AXPBY(B ## axpby)(n, alpha, u, incx, beta, v, incy);                                        \
}                                                                                                            \
template<>                                                                                                   \
inline void Blas<T>::axpby(const int& n, const T& alpha, const T* const u, const int& incx,                  \
                           const T& beta, T* const v, const int& incy) {                                     \
    HPDDM_PREFIX_AXPBY(C ## axpby)(n, &alpha, u, incx, &beta, v, incy);                                      \
}
# endif
# define HPDDM_GENERATE_BLAS_COMPLEX_VOID(C, T, B, U)                                                        \
HPDDM_GENERATE_BLAS(B, U)                                                                                    \
HPDDM_GENERATE_GEMM(B, U)                                                                                    \
HPDDM_GENERATE_GEMM_COMPLEX(C, T)                                                                            \
HPDDM_GENERATE_BLAS(C, T)                                                                                    \
                                                                                                             \
template<>                                                                                                   \
inline void Blas<U>::her(const char* const uplo, const int* const n, const U* const alpha,                   \
                         const U* const x, const int* const incx, U* const a, const int* const lda) {        \
    HPDDM_F77(B ## syr)(uplo, n, alpha, x, incx, a, lda);                                                    \
}                                                                                                            \
template<>                                                                                                   \
inline void Blas<T>::her(const char* const uplo, const int* const n, const U* const alpha,                   \
                         const T* const x, const int* const incx, T* const a, const int* const lda) {        \
    HPDDM_F77(C ## her)(uplo, n, alpha, x, incx, a, lda);                                                    \
}                                                                                                            \
template<>                                                                                                   \
inline void Blas<U>::her2(const char* const uplo, const int* const n, const U* const alpha,                  \
                          const U* const x, const int* const incx, const U* const y,                         \
                          const int* const incy, U* const a, const int* const lda) {                         \
    HPDDM_F77(B ## syr2)(uplo, n, alpha, x, incx, y, incy, a, lda);                                          \
}                                                                                                            \
template<>                                                                                                   \
inline void Blas<T>::her2(const char* const uplo, const int* const n, const T* const alpha,                  \
                          const T* const x, const int* const incx, const T* const y,                         \
                          const int* const incy, T* const a, const int* const lda) {                         \
    HPDDM_F77(C ## her2)(uplo, n, alpha, x, incx, y, incy, a, lda);                                          \
}                                                                                                            \
                                                                                                             \
template<>                                                                                                   \
inline void Blas<U>::herk(const char* const uplo, const char* const trans, const int* const n,               \
                          const int* const k, const U* const alpha, const U* const a, const int* const lda,  \
                          const U* const beta, U* const c, const int* const ldc) {                           \
    HPDDM_F77(B ## syrk)(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);                                    \
}                                                                                                            \
template<>                                                                                                   \
inline void Blas<T>::herk(const char* const uplo, const char* const trans, const int* const n,               \
                          const int* const k, const U* const alpha, const T* const a, const int* const lda,  \
                          const U* const beta, T* const c, const int* const ldc) {                           \
    HPDDM_F77(C ## herk)(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);                                    \
}
# define HPDDM_GENERATE_BLAS_COMPLEX(C, T, B, U)                                                             \
template<>                                                                                                   \
template<>                                                                                                   \
inline U Blas<U>::nrm2(const int* const n, const U* const x, const int* const incx) {                        \
    return HPDDM_F77(B ## nrm2)(n, x, incx);                                                                 \
}                                                                                                            \
template<>                                                                                                   \
template<>                                                                                                   \
inline U Blas<T>::nrm2(const int* const n, const T* const x, const int* const incx) {                        \
    return HPDDM_F77(B ## C ## nrm2)(n, x, incx);                                                            \
}                                                                                                            \
template<>                                                                                                   \
template<>                                                                                                   \
inline U Blas<U>::dot(const int* const n, const U* const x, const int* const incx,                           \
                      const U* const y, const int* const incy) {                                             \
    return HPDDM_F77(B ## dot)(n, x, incx, y, incy);                                                         \
}                                                                                                            \
HPDDM_GENERATE_BLAS_COMPLEX_VOID(C, T, B, U)
HPDDM_GENERATE_BLAS_COMPLEX_VOID(c, std::complex<float>, s, float)
HPDDM_GENERATE_BLAS_COMPLEX(z, std::complex<double>, d, double)
# if HPDDM_MKL || defined(__APPLE__)
HPDDM_GENERATE_AXPBY(c, std::complex<float>, s, float)
HPDDM_GENERATE_AXPBY(z, std::complex<double>, d, double)
# else
template<class K>
inline void Blas<K>::axpby(const int& n, const K& alpha, const K* const u, const int& incx, const K& beta, K* const v, const int& incy) {
    if(beta == K())
        for(int i = 0; i < n; ++i)
            v[i * incy] = alpha * u[i * incx];
    else
        for(int i = 0; i < n; ++i)
            v[i * incy] = alpha * u[i * incx] + beta * v[i * incy];
}
# endif // HPDDM_MKL || defined(__APPLE__)
} // HPDDM
#endif // __cplusplus
#endif // _HPDDM_BLAS_
