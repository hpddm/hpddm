/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2014-08-04

   Copyright (C) 2011-2014 Université de Grenoble
                 2015      Eidgenössische Technische Hochschule Zürich
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

#ifndef HPDDM_WRAPPER_HPP_
#define HPDDM_WRAPPER_HPP_

#if HPDDM_LIBXSMM
#include <libxsmm.h>
#endif

#define HPDDM_GENERATE_EXTERN_MKL(C, T)                                                                      \
void cblas_ ## C ## gthr(const int, const T*, T*, const int*);                                               \
void cblas_ ## C ## sctr(const int, const T*, const int*, T*);

#define HPDDM_GENERATE_CSRCSC                                                                                \
template<class K>                                                                                            \
template<char N, char M>                                                                                     \
inline void Wrapper<K>::csrcsc(const int* const n, const K* const a, const int* const ja,                    \
                               const int* const ia, K* const b, int* const jb, int* const ib) {              \
    unsigned int nnz = ia[*n] - (N == 'F');                                                                  \
    std::fill_n(ib, *n + 1, 0);                                                                              \
    for(unsigned int i = 0; i < nnz; ++i)                                                                    \
        ib[ja[i] + (N == 'C')]++;                                                                            \
    std::partial_sum(ib, ib + *n + 1, ib);                                                                   \
    for(unsigned int i = 0; i < *n; ++i)                                                                     \
        for(unsigned int j = ia[i] - (N == 'F'); j < ia[i + 1] - (N == 'F'); ++j) {                          \
            unsigned int k = ib[ja[j] - (N == 'F')]++;                                                       \
            jb[k] = i + (M == 'F');                                                                          \
            b[k] = a[j];                                                                                     \
        }                                                                                                    \
    for(unsigned int i = *n; i > 0; --i)                                                                     \
        ib[i] = ib[i - 1] + (M == 'F');                                                                      \
    ib[0] = (M == 'F');                                                                                      \
}

#if HPDDM_MKL
# include <mkl_spblas.h>
# include <mkl_vml.h>
# include <mkl_trans.h>
# if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION < 110201
#  define HPDDM_CONST(T, V) const_cast<T*>(V)
# else
#  define HPDDM_CONST(T, V) V
#  if !defined(INTEL_MKL_VERSION)
extern "C" {
HPDDM_GENERATE_EXTERN_MKL(s, float)
HPDDM_GENERATE_EXTERN_MKL(d, double)
HPDDM_GENERATE_EXTERN_MKL(c, std::complex<float>)
HPDDM_GENERATE_EXTERN_MKL(z, std::complex<double>)
}
#   include <mkl_service.h>
#   if !defined(MKL_ENABLE_AVX512_MIC) || MKL_ENABLE_AVX512_MIC == 3
#    include <mkl_version.h>
#   endif
#  endif
# endif
#endif // HPDDM_MKL

#ifdef I
#undef I
#endif

namespace HPDDM {
/* Class: Wrapper
 *
 *  A class for handling dense and sparse linear algebra.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
struct Wrapper {
#if HPDDM_MPI
    /* Function: mpi_type
     *  Returns the MPI datatype of the template parameter of <Wrapper>. */
    static MPI_Datatype mpi_type();
    /* Function: mpi_underlying_type
     *  Returns the MPI datatype of the underlying type of the template parameter of <Wrapper>. */
    static MPI_Datatype mpi_underlying_type() {
        return Wrapper<underlying_type<K>>::mpi_type();
    }
    /* Function: mpi_op
     *  Returns the MPI operation of the template parameter of <Wrapper>. */
    static MPI_Op mpi_op(const MPI_Op&);
#endif
    static constexpr bool is_complex = !std::is_same<typename std::remove_const<K>::type, underlying_type<K>>::value;
    /* Variable: transc
     *  Transposed real operators or Hermitian transposed complex operators. */
    static constexpr char transc = is_complex ? 'C' : 'T';
    /* Variable: I
     *  Numbering of a sparse <MatrixCSR>. */
#if HPDDM_MKL
    static constexpr char I = 'F';
#else
    static constexpr char I = HPDDM_NUMBERING;
#endif
    /* Variable: d__0
     *  Zero. */
    static constexpr K d__0 = std::is_floating_point<K>::value ? K(0.0) : 0;
    /* Variable: d__1
     *  One. */
    static constexpr K d__1 = std::is_floating_point<K>::value ? K(1.0) : 1;
    /* Variable: d__2
     *  Minus one. */
    static constexpr K d__2 = std::is_floating_point<K>::value ? K(-1.0) : -1;

    /* Function: csrmv(square)
     *  Computes a sparse square matrix-vector product. */
    template<char N = HPDDM_NUMBERING>
    static void csrmv(bool, const int* const, const K* const, const int* const, const int* const, const K* const, K* const);
    template<char N = HPDDM_NUMBERING>
    static void bsrmv(bool, const int* const, const int* const, const K* const, const int* const, const int* const, const K* const, K* const);
    /* Function: csrmv
     *  Computes a scalar-sparse matrix-vector product. */
    template<char N = HPDDM_NUMBERING>
    static void csrmv(const char* const, const int* const, const int* const, const K* const, bool,
                      const K* const, const int* const, const int* const, const K* const, const K* const, K* const);
    template<char N = HPDDM_NUMBERING>
    static void bsrmv(const char* const, const int* const, const int* const, const int* const, const K* const, bool,
                      const K* const, const int* const, const int* const, const K* const, const K* const, K* const);
    /* Function: csrmm(square)
     *  Computes a sparse square matrix-matrix product. */
    template<char N = HPDDM_NUMBERING>
    static void csrmm(bool, const int* const, const int* const, const K* const, const int* const, const int* const, const K* const, K* const);
    template<char N = HPDDM_NUMBERING>
    static void bsrmm(bool, const int* const, const int* const, const int* const, const K* const, const int* const, const int* const, const K* const, K* const);
    /* Function: csrmm
     *  Computes a scalar-sparse matrix-matrix product. */
    template<char N = HPDDM_NUMBERING>
    static void csrmm(const char* const, const int* const, const int* const, const int* const, const K* const, bool,
                      const K* const, const int* const, const int* const, const K* const, const K* const, K* const);
    template<char N = HPDDM_NUMBERING>
    static void bsrmm(const char* const, const int* const, const int* const, const int* const, const int* const, const K* const, bool,
                      const K* const, const int* const, const int* const, const K* const, const K* const, K* const);

    /* Function: csrcsc
     *  Converts a matrix stored in Compressed Sparse Row format into Compressed Sparse Column format. */
    template<char, char>
    static void csrcsc(const int* const, const K* const, const int* const, const int* const, K* const, int* const, int* const);
    /* Function: bsrcoo
     *  Converts a matrix stored in Block Compressed Sparse Row format into Coordinate format. */
    template<char, char, char>
    static void bsrcoo(const int, const unsigned short, K* const, const int* const, const int* const, K*&, int*&, int*&, const int& = 0);
    /* Function: gthr
     *  Gathers the elements of a full-storage sparse vector into compressed form. */
    static void gthr(const int&, const K* const, K* const, const int* const);
    /* Function: sctr
     *  Scatters the elements of a compressed sparse vector into full-storage form. */
    static void sctr(const int&, const K* const, const int* const, K* const);
    /* Function: diag(in-place)
     *  Computes a vector-matrix element-wise multiplication. */
    static void diag(const int&, const underlying_type<K>* const, K* const, const int& = 1);
    /* Function: diag
     *  Computes a vector-matrix element-wise multiplication. */
    static void diag(const int&, const underlying_type<K>* const, const K* const, K* const, const int& = 1);
    /* Function: conj
     *  Conjugates a real or complex number. */
    template<class T, typename std::enable_if<!Wrapper<T>::is_complex>::type* = nullptr>
    static T conj(const T& x) { return x; }
    template<class T, typename std::enable_if<Wrapper<T>::is_complex>::type* = nullptr>
    static T conj(const T& x) { return HPDDM::conj(x); }
    template<char O>
    static void cycle(const int n, const int m, K* const ab, const int k, const int lda = 0, const int ldb = 0) {
        if((O == 'T' || O == 'C') && (n > 1 || ldb > n) && (m > 1 || lda > m)) {
            if(lda > m)
                for(int i = 1; i < n; ++i)
                    std::copy_n(ab + i * lda * k, m * k, ab + i * m * k);
            const int size = n * m - 1;
            std::vector<char> b((size >> 3) + 1);
            b[0] |= 1;
            b[size >> 3] |= 1 << (size & 7);
            int i = 1;
            while(i < size) {
                int it = i;
                std::vector<K> t(ab + i * k, ab + (i + 1) * k);
                do {
                    int next = (i * n) % size;
                    std::swap_ranges(ab + next * k, ab + (next + 1) * k, t.begin());
                    b[i >> 3] |= 1 << (i & 7);
                    i = next;
                } while(i != it);

                for(i = 1; i < size && (b[i >> 3] & (1 << (i & 7))) != 0; ++i);
            }
            if(ldb > n) {
                if(O == 'C' && is_complex) {
                    for(int i = m; i > 0; --i)
                        for(unsigned int j = n * k; j-- > 0; )
                            ab[(i - 1) * ldb * k + j] = conj(ab[(i - 1) * n * k + j]);
                }
                else
                    for(int i = m; i > 0; --i)
                        std::copy_backward(ab + (i - 1) * n * k, ab + i * n * k, ab + ((i - 1) * ldb + n) * k);
            }
            else if(O == 'C' && is_complex)
                std::for_each(ab, ab + n * m * k, [](K& x) { x = conj(x); });
        }
    }
    /* Function: imatcopy
     *  Transforms (copy, transpose, conjugate transpose, conjugate) a dense matrix in-place. */
    template<char O>
    static void imatcopy(const int, const int, K* const, const int, const int);
    /* Function: omatcopy
     *  Transforms (copy, transpose, conjugate transpose, conjugate) a dense matrix out-place. */
    template<char O>
    static void omatcopy(const int, const int, const K* const, const int, K* const, const int);
};

#if HPDDM_MPI
template<>
inline MPI_Datatype Wrapper<float>::mpi_type() { return MPI_FLOAT; }
template<>
inline MPI_Datatype Wrapper<double>::mpi_type() { return MPI_DOUBLE; }
template<>
inline MPI_Datatype Wrapper<std::complex<float>>::mpi_type() { return MPI_C_COMPLEX; }
template<>
inline MPI_Datatype Wrapper<std::complex<double>>::mpi_type() { return MPI_C_DOUBLE_COMPLEX; }
template<class K>
inline MPI_Datatype Wrapper<K>::mpi_type() { static_assert(std::is_integral<K>::value, "Wrong type"); return sizeof(K) == sizeof(int) ? MPI_INT : sizeof(K) == sizeof(long long) ? MPI_LONG_LONG : MPI_BYTE; }
# if defined(PETSC_HAVE_REAL___FLOAT128) && !(defined(__NVCC__) || defined(__CUDACC__))
template<>
inline MPI_Datatype Wrapper<__float128>::mpi_type() { return MPIU___FLOAT128; }
template<>
inline MPI_Datatype Wrapper<__complex128>::mpi_type() { return MPIU___COMPLEX128; }
template<>
inline MPI_Op Wrapper<__float128>::mpi_op(const MPI_Op& op) { return op == MPI_SUM ? (!PetscDefined(USE_REAL___FLOAT128) ? MPIU_SUM___FP16___FLOAT128 : MPIU_SUM) : op; }
template<>
inline MPI_Op Wrapper<__complex128>::mpi_op(const MPI_Op& op) { return op == MPI_SUM ? (!PetscDefined(USE_REAL___FLOAT128) ? MPIU_SUM___FP16___FLOAT128 : MPIU_SUM) : op; }
# endif
# if defined(PETSC_HAVE_REAL___FP16)
template<>
inline MPI_Datatype Wrapper<__fp16>::mpi_type() { return MPIU___FP16; }
#  if defined(PETSC_HAVE_CXX_DIALECT_CXX14)
template<>
inline MPI_Datatype Wrapper<std::complex<__fp16>>::mpi_type() { return MPI_FLOAT; }
#  endif
template<>
inline MPI_Op Wrapper<__fp16>::mpi_op(const MPI_Op& op) { return op == MPI_SUM ? (!PetscDefined(USE_REAL___FP16) ? MPIU_SUM___FP16___FLOAT128 : MPIU_SUM) : op; }
# endif
template<class K>
inline MPI_Op Wrapper<K>::mpi_op(const MPI_Op& op) { return op; }
#endif

template<class K>
constexpr char Wrapper<K>::transc;

template<class K>
constexpr K Wrapper<K>::d__0;
template<class K>
constexpr K Wrapper<K>::d__1;
template<class K>
constexpr K Wrapper<K>::d__2;

template<class K>
inline void Wrapper<K>::diag(const int& m, const underlying_type<K>* const d, K* const in, const int& n) {
    if(d)
        diag(m, d, nullptr, in, n);
}
template<class K>
inline void Wrapper<K>::sctr(const int& n, const K* const x, const int* const indx, K* const y) {
    for(int i = 0; i < n; ++i)
        y[indx[i]] = x[i];
}

#if HPDDM_MKL
template<char N, char M = 'L'>
struct matdescr {
    static const char a[];
    static const char b[];
};

template<char N, char M>
const char matdescr<N, M>::a[4] { 'G', '0', '0', N };
template<char N, char M>
const char matdescr<N, M>::b[4] { 'S',  M , 'N', N };


#if INTEL_MKL_VERSION > 20180001 || __INTEL_MKL_BUILD_DATE > 20200000
#define HPDDM_GENERATE_SPARSE_MKL(C, T)                                                                      \
template<>                                                                                                   \
template<char N>                                                                                             \
inline void Wrapper<T>::csrmv(bool sym, const int* const n, const T* const a, const int* const ia,           \
                              const int* const ja, const T* const x, T* const y) {                           \
    csrmv<N>("N", n, n, &d__1, sym, a, ia, ja, x, &d__0, y);                                                 \
}                                                                                                            \
template<>                                                                                                   \
template<char N>                                                                                             \
inline void Wrapper<T>::bsrmv(bool sym, const int* const n, const int* const bs, const T* const a,           \
                              const int* const ia, const int* const ja, const T* const x, T* const y) {      \
    bsrmv<N>("N", n, n, bs, &d__1, sym, a, ia, ja, x, &d__0, y);                                             \
}                                                                                                            \
template<>                                                                                                   \
template<char N>                                                                                             \
inline void Wrapper<T>::csrmv(const char* const trans, const int* const m, const int* const k,               \
                              const T* const alpha, bool sym, const T* const a, const int* const ia,         \
                              const int* const ja, const T* const x, const T* const beta, T* const y) {      \
    struct matrix_descr descr;                                                                               \
    sparse_matrix_t       csr;                                                                               \
    mkl_sparse_ ## C ## _create_csr(&csr, N == 'C' ? SPARSE_INDEX_BASE_ZERO : SPARSE_INDEX_BASE_ONE, *m, *k, \
                                    const_cast<int*>(ia), const_cast<int*>(ia + 1), const_cast<int*>(ja),    \
                                    const_cast<T*>(a));                                                      \
    descr.type = sym ? SPARSE_MATRIX_TYPE_SYMMETRIC : SPARSE_MATRIX_TYPE_GENERAL;                            \
    descr.mode = SPARSE_FILL_MODE_LOWER;                                                                     \
    descr.diag = SPARSE_DIAG_NON_UNIT;                                                                       \
    mkl_sparse_ ## C ## _mv(*trans == 'N' ? SPARSE_OPERATION_NON_TRANSPOSE : SPARSE_OPERATION_TRANSPOSE,     \
                            *alpha, csr, descr, x, *beta, y);                                                \
    mkl_sparse_destroy(csr);                                                                                 \
}                                                                                                            \
template<>                                                                                                   \
template<char N>                                                                                             \
inline void Wrapper<T>::bsrmv(const char* const trans, const int* const m, const int* const k,               \
                              const int* const bs, const T* const alpha, bool sym, const T* const a,         \
                              const int* const ia, const int* const ja, const T* const x,                    \
                              const T* const beta, T* const y) {                                             \
    struct matrix_descr descr;                                                                               \
    sparse_matrix_t       bsr;                                                                               \
    mkl_sparse_ ## C ## _create_bsr(&bsr, N == 'C' ? SPARSE_INDEX_BASE_ZERO : SPARSE_INDEX_BASE_ONE,         \
                                    N == 'C' ? SPARSE_LAYOUT_ROW_MAJOR : SPARSE_LAYOUT_COLUMN_MAJOR, *m, *k, \
                                    *bs, const_cast<int*>(ia), const_cast<int*>(ia + 1),                     \
                                    const_cast<int*>(ja), const_cast<T*>(a));                                \
    descr.type = sym ? SPARSE_MATRIX_TYPE_SYMMETRIC : SPARSE_MATRIX_TYPE_GENERAL;                            \
    descr.mode = SPARSE_FILL_MODE_UPPER;                                                                     \
    descr.diag = SPARSE_DIAG_NON_UNIT;                                                                       \
    mkl_sparse_ ## C ## _mv(*trans == 'N' ? SPARSE_OPERATION_NON_TRANSPOSE : SPARSE_OPERATION_TRANSPOSE,     \
                            *alpha, bsr, descr, x, *beta, y);                                                \
    mkl_sparse_destroy(bsr);                                                                                 \
}                                                                                                            \
template<>                                                                                                   \
template<char N>                                                                                             \
inline void Wrapper<T>::csrmm(const char* const trans, const int* const m, const int* const n,               \
                              const int* const k, const T* const alpha, bool sym,                            \
                              const T* const a, const int* const ia, const int* const ja,                    \
                              const T* const x, const T* const beta, T* const y) {                           \
    if(*n != 1) {                                                                                            \
        if(N != 'F') {                                                                                       \
            std::for_each(const_cast<int*>(ja), const_cast<int*>(ja) + ia[*m], [](int& i) { ++i; });         \
            std::for_each(const_cast<int*>(ia), const_cast<int*>(ia) + *m + 1, [](int& i) { ++i; });         \
        }                                                                                                    \
        int ldb = (*trans == 'N' ? *k : *m);                                                                 \
        int ldc = (*trans == 'N' ? *m : *k);                                                                 \
        struct matrix_descr descr;                                                                           \
        sparse_matrix_t       csr;                                                                           \
        mkl_sparse_ ## C ## _create_csr(&csr, SPARSE_INDEX_BASE_ONE, *m,                                     \
                                        *k, const_cast<int*>(ia), const_cast<int*>(ia + 1),                  \
                                        const_cast<int*>(ja), const_cast<T*>(a));                            \
        descr.type = sym ? SPARSE_MATRIX_TYPE_SYMMETRIC : SPARSE_MATRIX_TYPE_GENERAL;                        \
        descr.mode = SPARSE_FILL_MODE_LOWER;                                                                 \
        descr.diag = SPARSE_DIAG_NON_UNIT;                                                                   \
        mkl_sparse_ ## C ## _mm(*trans == 'N' ? SPARSE_OPERATION_NON_TRANSPOSE : SPARSE_OPERATION_TRANSPOSE, \
                                *alpha, csr, descr, SPARSE_LAYOUT_COLUMN_MAJOR, x, *n, ldb, *beta, y, ldc);  \
        mkl_sparse_destroy(csr);                                                                             \
        if(N != 'F') {                                                                                       \
            std::for_each(const_cast<int*>(ia), const_cast<int*>(ia) + *m + 1, [](int& i) { --i; });         \
            std::for_each(const_cast<int*>(ja), const_cast<int*>(ja) + ia[*m], [](int& i) { --i; });         \
        }                                                                                                    \
    }                                                                                                        \
    else                                                                                                     \
        csrmv<N>(trans, m, k, alpha, sym, a, ia, ja, x, beta, y);                                            \
}
HPDDM_GENERATE_CSRCSC
#define HPDDM_GENERATE_MKL_BSRMM(C, T)                                                                       \
template<>                                                                                                   \
template<char N>                                                                                             \
inline void Wrapper<T>::bsrmm(const char* const trans, const int* const m, const int* const n,               \
                              const int* const k, const int* const bs, const T* const alpha, bool sym,       \
                              const T* const a, const int* const ia, const int* const ja,                    \
                              const T* const x, const T* const beta, T* const y) {                           \
    if(*k) {                                                                                                 \
        if(*n != 1) {                                                                                        \
            if(N != 'F') {                                                                                   \
                std::for_each(const_cast<int*>(ja), const_cast<int*>(ja) + ia[*m], [](int& i) { ++i; });     \
                std::for_each(const_cast<int*>(ia), const_cast<int*>(ia) + *m + 1, [](int& i) { ++i; });     \
            }                                                                                                \
            int ldb = *bs * (*trans == 'N' ? *k : *m);                                                       \
            int ldc = *bs * (*trans == 'N' ? *m : *k);                                                       \
            struct matrix_descr descr;                                                                       \
            sparse_matrix_t       bsr;                                                                       \
            mkl_sparse_ ## C ## _create_bsr(&bsr, SPARSE_INDEX_BASE_ONE, SPARSE_LAYOUT_COLUMN_MAJOR, *m, *k, \
                                            *bs, const_cast<int*>(ia), const_cast<int*>(ia + 1),             \
                                            const_cast<int*>(ja), const_cast<T*>(a));                        \
            descr.type = sym ? SPARSE_MATRIX_TYPE_SYMMETRIC : SPARSE_MATRIX_TYPE_GENERAL;                    \
            descr.mode = SPARSE_FILL_MODE_UPPER;                                                             \
            descr.diag = SPARSE_DIAG_NON_UNIT;                                                               \
            mkl_sparse_ ## C ## _mm(*trans == 'N' ? SPARSE_OPERATION_NON_TRANSPOSE :                         \
                                    SPARSE_OPERATION_TRANSPOSE, *alpha, bsr, descr,                          \
                                    SPARSE_LAYOUT_COLUMN_MAJOR, x, *n, ldb, *beta, y, ldc);                  \
            mkl_sparse_destroy(bsr);                                                                         \
            if(N != 'F') {                                                                                   \
                std::for_each(const_cast<int*>(ia), const_cast<int*>(ia) + *m + 1, [](int& i) { --i; });     \
                std::for_each(const_cast<int*>(ja), const_cast<int*>(ja) + ia[*m], [](int& i) { --i; });     \
            }                                                                                                \
        }                                                                                                    \
        else                                                                                                 \
            bsrmv<N>(trans, m, k, bs, alpha, sym, a, ia, ja, x, beta, y);                                    \
    }                                                                                                        \
}
#else
#define HPDDM_GENERATE_SPARSE_MKL(C, T)                                                                      \
template<>                                                                                                   \
template<char N>                                                                                             \
inline void Wrapper<T>::csrmv(bool sym, const int* const n, const T* const a, const int* const ia,           \
                              const int* const ja, const T* const x, T* const y) {                           \
    if(N == 'C') {                                                                                           \
        if(sym)                                                                                              \
            mkl_cspblas_ ## C ## csrsymv("L", HPDDM_CONST(int, n), HPDDM_CONST(T, a), HPDDM_CONST(int, ia),  \
                                         HPDDM_CONST(int, ja), HPDDM_CONST(T, x), y);                        \
        else                                                                                                 \
            mkl_cspblas_ ## C ## csrgemv("N", HPDDM_CONST(int, n), HPDDM_CONST(T, a), HPDDM_CONST(int, ia),  \
                                         HPDDM_CONST(int, ja), HPDDM_CONST(T, x), y);                        \
    }                                                                                                        \
    else {                                                                                                   \
        if(sym)                                                                                              \
            mkl_ ## C ## csrsymv("L", HPDDM_CONST(int, n), HPDDM_CONST(T, a), HPDDM_CONST(int, ia),          \
                                 HPDDM_CONST(int, ja), HPDDM_CONST(T, x), y);                                \
        else                                                                                                 \
            mkl_ ## C ## csrgemv("N", HPDDM_CONST(int, n), HPDDM_CONST(T, a), HPDDM_CONST(int, ia),          \
                                 HPDDM_CONST(int, ja), HPDDM_CONST(T, x), y);                                \
    }                                                                                                        \
}                                                                                                            \
template<>                                                                                                   \
template<char N>                                                                                             \
inline void Wrapper<T>::bsrmv(bool sym, const int* const n, const int* const bs, const T* const a,           \
                              const int* const ia, const int* const ja, const T* const x, T* const y) {      \
    if(N == 'C') {                                                                                           \
        if(sym)                                                                                              \
            mkl_cspblas_ ## C ## bsrsymv("U", HPDDM_CONST(int, n), HPDDM_CONST(int, bs), HPDDM_CONST(T, a),  \
                                         HPDDM_CONST(int, ia), HPDDM_CONST(int, ja), HPDDM_CONST(T, x), y);  \
        else                                                                                                 \
            mkl_cspblas_ ## C ## bsrgemv("N", HPDDM_CONST(int, n), HPDDM_CONST(int, bs), HPDDM_CONST(T, a),  \
                                         HPDDM_CONST(int, ia), HPDDM_CONST(int, ja), HPDDM_CONST(T, x), y);  \
    }                                                                                                        \
    else {                                                                                                   \
        if(sym)                                                                                              \
            mkl_ ## C ## bsrsymv("U", HPDDM_CONST(int, n), HPDDM_CONST(int, bs), HPDDM_CONST(T, a),          \
                                 HPDDM_CONST(int, ia), HPDDM_CONST(int, ja), HPDDM_CONST(T, x), y);          \
        else                                                                                                 \
            mkl_ ## C ## bsrgemv("N", HPDDM_CONST(int, n), HPDDM_CONST(int, bs), HPDDM_CONST(T, a),          \
                                 HPDDM_CONST(int, ia), HPDDM_CONST(int, ja), HPDDM_CONST(T, x), y);          \
    }                                                                                                        \
}                                                                                                            \
template<>                                                                                                   \
template<char N>                                                                                             \
inline void Wrapper<T>::csrmv(const char* const trans, const int* const m, const int* const k,               \
                              const T* const alpha, bool sym, const T* const a, const int* const ia,         \
                              const int* const ja, const T* const x, const T* const beta, T* const y) {      \
    mkl_ ## C ## csrmv(HPDDM_CONST(char, trans), HPDDM_CONST(int, m), HPDDM_CONST(int, k),                   \
                       HPDDM_CONST(T, alpha), HPDDM_CONST(char, sym ? matdescr<N>::b : matdescr<N>::a),      \
                       HPDDM_CONST(T, a), HPDDM_CONST(int, ja), HPDDM_CONST(int, ia),                        \
                       HPDDM_CONST(int, ia) + 1, HPDDM_CONST(T, x), HPDDM_CONST(T, beta), y);                \
}                                                                                                            \
template<>                                                                                                   \
template<char N>                                                                                             \
inline void Wrapper<T>::bsrmv(const char* const trans, const int* const m, const int* const k,               \
                              const int* const bs, const T* const alpha, bool sym, const T* const a,         \
                              const int* const ia, const int* const ja, const T* const x,                    \
                              const T* const beta, T* const y) {                                             \
    mkl_ ## C ## bsrmv(HPDDM_CONST(char, trans), HPDDM_CONST(int, m), HPDDM_CONST(int, k),                   \
                       HPDDM_CONST(int, bs), HPDDM_CONST(T, alpha),                                          \
                       HPDDM_CONST(char, sym ? (matdescr<N, 'U'>::b) : matdescr<N>::a), HPDDM_CONST(T, a),   \
                       HPDDM_CONST(int, ja), HPDDM_CONST(int, ia), HPDDM_CONST(int, ia) + 1,                 \
                       HPDDM_CONST(T, x), HPDDM_CONST(T, beta), y);                                          \
}                                                                                                            \
template<>                                                                                                   \
template<char N>                                                                                             \
inline void Wrapper<T>::csrmm(const char* const trans, const int* const m, const int* const n,               \
                              const int* const k, const T* const alpha, bool sym,                            \
                              const T* const a, const int* const ia, const int* const ja,                    \
                              const T* const x, const T* const beta, T* const y) {                           \
    if(*n != 1) {                                                                                            \
        if(N != 'F') {                                                                                       \
            std::for_each(const_cast<int*>(ja), const_cast<int*>(ja) + ia[*m], [](int& i) { ++i; });         \
            std::for_each(const_cast<int*>(ia), const_cast<int*>(ia) + *m + 1, [](int& i) { ++i; });         \
        }                                                                                                    \
        const int* const ldb = (*trans == 'N' ? k : m);                                                      \
        const int* const ldc = (*trans == 'N' ? m : k);                                                      \
        mkl_ ## C ## csrmm(HPDDM_CONST(char, trans), HPDDM_CONST(int, m), HPDDM_CONST(int, n),               \
                           HPDDM_CONST(int, k), HPDDM_CONST(T, alpha),                                       \
                           HPDDM_CONST(char, sym ? matdescr<'F'>::b : matdescr<'F'>::a), HPDDM_CONST(T, a),  \
                           HPDDM_CONST(int, ja), HPDDM_CONST(int, ia), HPDDM_CONST(int, ia) + 1,             \
                           HPDDM_CONST(T, x), HPDDM_CONST(int, ldb), HPDDM_CONST(T, beta),  y,               \
                           HPDDM_CONST(int, ldc));                                                           \
        if(N != 'F') {                                                                                       \
            std::for_each(const_cast<int*>(ia), const_cast<int*>(ia) + *m + 1, [](int& i) { --i; });         \
            std::for_each(const_cast<int*>(ja), const_cast<int*>(ja) + ia[*m], [](int& i) { --i; });         \
        }                                                                                                    \
    }                                                                                                        \
    else                                                                                                     \
        csrmv<N>(trans, m, k, alpha, sym, a, ia, ja, x, beta, y);                                            \
}                                                                                                            \
template<>                                                                                                   \
template<char N, char M>                                                                                     \
inline void Wrapper<T>::csrcsc(const int* const n, const T* const a, const int* const ja,                    \
                               const int* const ia, T* const b, int* const jb, int* const ib) {              \
    int job[6] { 0, N == 'F', M == 'F', 0, 0, 1 };                                                           \
    int error;                                                                                               \
    mkl_ ## C ## csrcsc(job, HPDDM_CONST(int, n), const_cast<T*>(a), const_cast<int*>(ja),                   \
                        const_cast<int*>(ia), b, jb, ib, &error);                                            \
}
#define HPDDM_GENERATE_MKL_BSRMM(C, T)                                                                       \
template<>                                                                                                   \
template<char N>                                                                                             \
inline void Wrapper<T>::bsrmm(const char* const trans, const int* const m, const int* const n,               \
                              const int* const k, const int* const bs, const T* const alpha, bool sym,       \
                              const T* const a, const int* const ia, const int* const ja,                    \
                              const T* const x, const T* const beta, T* const y) {                           \
    if(*n != 1) {                                                                                            \
        if(N != 'F') {                                                                                       \
            std::for_each(const_cast<int*>(ja), const_cast<int*>(ja) + ia[*m], [](int& i) { ++i; });         \
            std::for_each(const_cast<int*>(ia), const_cast<int*>(ia) + *m + 1, [](int& i) { ++i; });         \
        }                                                                                                    \
        int ldb = *bs * (*trans == 'N' ? *k : *m);                                                           \
        int ldc = *bs * (*trans == 'N' ? *m : *k);                                                           \
        mkl_ ## C ## bsrmm(HPDDM_CONST(char, trans), HPDDM_CONST(int, m), HPDDM_CONST(int, n),               \
                           HPDDM_CONST(int, k), HPDDM_CONST(int, bs), HPDDM_CONST(T, alpha),                 \
                           HPDDM_CONST(char, sym ? (matdescr<'F', 'U'>::b) : matdescr<'F'>::a),              \
                           HPDDM_CONST(T, a), HPDDM_CONST(int, ja), HPDDM_CONST(int, ia),                    \
                           HPDDM_CONST(int, ia) + 1, HPDDM_CONST(T, x), &ldb,                                \
                           HPDDM_CONST(T, beta), y, &ldc);                                                   \
        if(N != 'F') {                                                                                       \
            std::for_each(const_cast<int*>(ia), const_cast<int*>(ia) + *m + 1, [](int& i) { --i; });         \
            std::for_each(const_cast<int*>(ja), const_cast<int*>(ja) + ia[*m], [](int& i) { --i; });         \
        }                                                                                                    \
    }                                                                                                        \
    else                                                                                                     \
        bsrmv<N>(trans, m, k, bs, alpha, sym, a, ia, ja, x, beta, y);                                        \
}
#endif

#define HPDDM_GENERATE_MKL(C, T)                                                                             \
HPDDM_GENERATE_SPARSE_MKL(C, T)                                                                              \
template<>                                                                                                   \
inline void Wrapper<T>::gthr(const int& n, const T* const y, T* const x, const int* const indx) {            \
    cblas_ ## C ## gthr(n, y, x, indx);                                                                      \
}                                                                                                            \
template<>                                                                                                   \
inline void Wrapper<T>::sctr(const int& n, const T* const x, const int* const indx, T* const y) {            \
    cblas_ ## C ## sctr(n, x, indx, y);                                                                      \
}                                                                                                            \
template<>                                                                                                   \
template<char O>                                                                                             \
inline void Wrapper<T>::imatcopy(const int n, const int m, T* const ab, const int lda, const int ldb) {      \
    static_assert(O == 'N' || O == 'R' || O == 'T' || O == 'C', "Unknown operation");                        \
    if(lda && ldb)                                                                                           \
        mkl_ ## C ## imatcopy('C', O, m, n, d__1, ab, lda, ldb);                                             \
}                                                                                                            \
template<>                                                                                                   \
template<char O>                                                                                             \
inline void Wrapper<T>::omatcopy(const int n, const int m, const T* const a, const int lda,                  \
                                 T* const b, const int ldb) {                                                \
    static_assert(O == 'N' || O == 'R' || O == 'T' || O == 'C', "Unknown operation");                        \
    if(lda && ldb)                                                                                           \
        mkl_ ## C ## omatcopy('C', O, m, n, d__1, a, lda, b, ldb);                                           \
}
#define HPDDM_GENERATE_MKL_VML(C, T)                                                                         \
template<>                                                                                                   \
inline void Wrapper<T>::diag(const int& m, const T* const d,                                                 \
                             const T* const in, T* const out, const int& n) {                                \
    if(d) {                                                                                                  \
        if(in)                                                                                               \
            for(int i = 0; i < n; ++i)                                                                       \
                v ## C ## Mul(m, d, in + i * m, out + i * m);                                                \
        else                                                                                                 \
            for(int i = 0; i < n; ++i)                                                                       \
                v ## C ## Mul(m, d, out + i * m, out + i * m);                                               \
    }                                                                                                        \
    else if(in)                                                                                              \
        std::copy_n(in, n * m, out);                                                                         \
}
HPDDM_GENERATE_MKL(s, float)
HPDDM_GENERATE_MKL(d, double)
HPDDM_GENERATE_MKL(c, std::complex<float>)
HPDDM_GENERATE_MKL(z, std::complex<double>)
#if !HPDDM_LIBXSMM
HPDDM_GENERATE_MKL_BSRMM(s, float)
HPDDM_GENERATE_MKL_BSRMM(d, double)
#endif
HPDDM_GENERATE_MKL_BSRMM(c, std::complex<float>)
HPDDM_GENERATE_MKL_BSRMM(z, std::complex<double>)
HPDDM_GENERATE_MKL_VML(s, float)
HPDDM_GENERATE_MKL_VML(d, double)
#else
template<class K>
template<char N>
inline void Wrapper<K>::csrmv(bool sym, const int* const n, const K* const a, const int* const ia, const int* const ja, const K* const x, K* const y) {
    csrmv<N>("N", n, n, &d__1, sym, a, ia, ja, x, &d__0, y);
}
template<class K>
template<char N>
inline void Wrapper<K>::bsrmv(bool sym, const int* const n, const int* const bs, const K* const a,
                              const int* const ia, const int* const ja, const K* const x, K* const y) {
    bsrmv<N>("N", n, n, bs, &d__1, sym, a, ia, ja, x, &d__0, y);
}
template<class K>
template<char N>
inline void Wrapper<K>::csrmv(const char* const trans, const int* const m, const int* const k, const K* const alpha, bool sym,
                              const K* const a, const int* const ia, const int* const ja, const K* const x, const K* const beta, K* const y) {
    if(*trans == 'N' && !sym) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, HPDDM_GRANULARITY)
#endif
        for(int i = 0; i < *m; ++i) {
            y[i] *= *beta;
            for(int j = ia[i] - (N == 'F'); j < ia[i + 1] - (N == 'F'); ++j)
                y[i] += *alpha * a[j] * x[ja[j] - (N == 'F')];
        }
    }
    else {
        if(beta == &d__0)
            std::fill_n(y, *k, K());
        else if(beta != &d__1)
            Blas<K>::scal(k, beta, y, &i__1);
        if(sym) {
            for(int i = 0; i < *m; ++i)
                for(int l = ia[i] - (N == 'F'); l < ia[i + 1] - (N == 'F'); ++l) {
                    int j = ja[l] - (N == 'F');
                    const K scal = *alpha * (Wrapper<K>::is_complex && *trans == 'C' ? conj(a[l]) : a[l]);
                    y[i] += scal * x[j];
                    if(i != j)
                        y[j] += scal * x[i];
                }
        }
        else {
            for(int i = 0; i < *m; ++i)
                for(int j = ia[i] - (N == 'F'); j < ia[i + 1] - (N == 'F'); ++j) {
                    const K scal = *alpha * (Wrapper<K>::is_complex && *trans == 'C' ? conj(a[j]) : a[j]);
                    y[ja[j] - (N == 'F')] += scal * x[i];
                }
        }
    }
}
template<class K>
template<char N>
inline void Wrapper<K>::bsrmv(const char* const trans, const int* const m, const int* const k,
                              const int* const bs, const K* const alpha, bool sym, const K* const a,
                              const int* const ia, const int* const ja, const K* const x,
                              const K* const beta, K* const y) {
    if(*trans == 'N' && !sym) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, HPDDM_GRANULARITY)
#endif
        for(int i = 0; i < *m; ++i) {
            Blas<K>::scal(bs, beta, y + *bs * i, &i__1);
            for(int j = ia[i] - (N == 'F'); j < ia[i + 1] - (N == 'F'); ++j)
                Blas<K>::gemv(N == 'F' ? "N" : "T", bs, bs, alpha, a + *bs * *bs * j, bs, x + *bs * (ja[j] - (N == 'F')), &i__1, &(Wrapper<K>::d__1), y + *bs * i, &i__1);
        }
    }
    else {
        const int ldc = *bs * *k;
        if(beta == &d__0)
            std::fill_n(y, ldc, K());
        else if(beta != &d__1)
            Blas<K>::scal(&ldc, beta, y, &i__1);
        if(Wrapper<K>::is_complex && *trans == 'C' && (sym || N == 'C')) {
            K* const c = const_cast<K*>(a);
            for(int i = 0; i < *bs * *bs * (ia[*m] - (N == 'F')); ++i)
                c[i] = conj(c[i]);
        }
        if(sym) {
            for(int i = 0; i < *m; ++i) {
                for(int l = ia[i] - (N == 'F'); l < ia[i + 1] - (N == 'F'); ++l) {
                    int j = ja[l] - (N == 'F');
                    Blas<K>::gemv(N == 'F' ? "N" : "T", bs, bs, alpha, a + *bs * *bs * l, bs, x + *bs * j, &i__1, &(Wrapper<K>::d__1), y + *bs * i, &i__1);
                    if(i != j)
                        Blas<K>::gemv(N == 'F' ? "T" : "N", bs, bs, alpha, a + *bs * *bs * l, bs, x + *bs * i, &i__1, &(Wrapper<K>::d__1), y + *bs * j, &i__1);
                }
            }
        }
        else {
            for(int i = 0; i < *m; ++i)
                for(int j = ia[i] - (N == 'F'); j < ia[i + 1] - (N == 'F'); ++j)
                    Blas<K>::gemv(N == 'F' ? trans : "N", bs, bs, alpha, a + *bs * *bs * j, bs, x + *bs * i, &i__1, &(Wrapper<K>::d__1), y + *bs * (ja[j] - (N == 'F')), &i__1);
        }
        if(Wrapper<K>::is_complex && *trans == 'C' && (sym || N == 'C')) {
            K* const c = const_cast<K*>(a);
            for(int i = 0; i < *bs * *bs * (ia[*m] - (N == 'F')); ++i)
                c[i] = conj(c[i]);
        }
    }
}
template<class K>
template<char N>
inline void Wrapper<K>::csrmm(const char* const trans, const int* const m, const int* const n, const int* const k, const K* const alpha, bool sym,
                              const K* const a, const int* const ia, const int* const ja, const K* const x,  const K* const beta, K* const y) {
    if(*trans == 'N' || sym) {
        int j = *m * *n;
        if(beta == &d__0)
            std::fill_n(y, j, K());
        else if(beta != &d__1)
            Blas<K>::scal(&j, beta, y, &i__1);
        if(sym) {
            for(int i = 0; i < *m; ++i)
                for(int l = ia[i] - (N == 'F'); l < ia[i + 1] - (N == 'F'); ++l) {
                    j = ja[l] - (N == 'F');
                    const K scal = *alpha * (Wrapper<K>::is_complex && *trans == 'C' ? conj(a[l]) : a[l]);
                    Blas<K>::axpy(n, &scal, x + j, k, y + i, m);
                    if(i != j)
                        Blas<K>::axpy(n, &scal, x + i, k, y + j, m);
                }
        }
        else {
#ifdef _OPENMP
#pragma omp for schedule(static, HPDDM_GRANULARITY)
#endif
            for(int i = 0; i < *m; ++i)
                for(j = ia[i] - (N == 'F'); j < ia[i + 1] - (N == 'F'); ++j) {
                    const K scal = *alpha * a[j];
                    Blas<K>::axpy(n, &scal, x + ja[j] - (N == 'F'), k, y + i, m);
                }
        }
    }
    else {
        int j = *k * *n;
        if(beta == &d__0)
            std::fill_n(y, j, K());
        else if(beta != &d__1)
            Blas<K>::scal(&j, beta, y, &i__1);
        for(int i = 0; i < *m; ++i)
            for(int j = ia[i] - (N == 'F'); j < ia[i + 1] - (N == 'F'); ++j) {
                const K scal = *alpha * (Wrapper<K>::is_complex && *trans == 'C' ? conj(a[j]) : a[j]);
                Blas<K>::axpy(n, &scal, x + i, m, y + ja[j] - (N == 'F'), k);
            }
    }
}
template<class K>
template<char N>
inline void Wrapper<K>::bsrmm(const char* const trans, const int* const m, const int* const n, const int* const k, const int* const bs, const K* const alpha, bool sym,
                              const K* const a, const int* const ia, const int* const ja, const K* const x,  const K* const beta, K* const y) {
    if(*k) {
        if(*trans == 'N' && !sym) {
            const int ldb = *bs * *k;
            const int ldc = *bs * *m;
            int j = ldc * *n;
            if(beta == &d__0)
                std::fill_n(y, j, K());
            else if(beta != &d__1)
                Blas<K>::scal(&j, beta, y, &i__1);
#ifdef _OPENMP
#pragma omp for schedule(static, HPDDM_GRANULARITY)
#endif
            for(int i = 0; i < *m; ++i) {
                for(j = ia[i] - (N == 'F'); j < ia[i + 1] - (N == 'F'); ++j)
                    Blas<K>::gemm("N", "N", bs, n, bs, alpha, a + *bs * *bs * j, bs, x + *bs * (ja[j] - (N == 'F')), &ldb, &(Wrapper<K>::d__1), y + *bs * i, &ldc);
            }
        }
        else {
            const int ldb = *bs * *m;
            const int ldc = *bs * *k;
            int j = ldc * *n;
            if(beta == &d__0)
                std::fill_n(y, j, K());
            else if(beta != &d__1)
                Blas<K>::scal(&j, beta, y, &i__1);
            if(Wrapper<K>::is_complex && *trans == 'C' && (sym || N == 'C')) {
                K* const c = const_cast<K*>(a);
                for(int i = 0; i < *bs * *bs * (ia[*m] - (N == 'F')); ++i)
                    c[i] = conj(c[i]);
            }
            if(sym) {
                for(int i = 0; i < *m; ++i) {
                    for(int l = ia[i] - (N == 'F'); l < ia[i + 1] - (N == 'F'); ++l) {
                        j = ja[l] - (N == 'F');
                        Blas<K>::gemm("T", "N", bs, n, bs, alpha, a + *bs * *bs * l, bs, x + *bs * i, &ldb, &(Wrapper<K>::d__1), y + *bs * j, &ldc);
                        if(i != j)
                            Blas<K>::gemm("N", "N", bs, n, bs, alpha, a + *bs * *bs * l, bs, x + *bs * j, &ldb, &(Wrapper<K>::d__1), y + *bs * i, &ldc);
                    }
                }
            }
            else {
                for(int i = 0; i < *m; ++i)
                    for(int l = ia[i] - (N == 'F'); l < ia[i + 1] - (N == 'F'); ++l)
                        Blas<K>::gemm(N == 'F' ? trans : "T", "N", bs, n, bs, alpha, a + *bs * *bs * l, bs, x + *bs * i, &ldb, &(Wrapper<K>::d__1), y + *bs * (ja[l] - (N == 'F')), &ldc);
            }
            if(Wrapper<K>::is_complex && *trans == 'C' && (sym || N == 'C')) {
                K* const c = const_cast<K*>(a);
                for(int i = 0; i < *bs * *bs * (ia[*m] - (N == 'F')); ++i)
                    c[i] = conj(c[i]);
            }
        }
    }
}

HPDDM_GENERATE_CSRCSC
template<class K>
inline void Wrapper<K>::gthr(const int& n, const K* const y, K* const x, const int* const indx) {
    for(int i = 0; i < n; ++i)
        x[i] = y[indx[i]];
}
template<class K>
template<char O>
inline void Wrapper<K>::omatcopy(const int n, const int m, const K* const a, const int lda, K* const b, const int ldb) {
    static_assert(O == 'N' || O == 'R' || O == 'T' || O == 'C', "Unknown operation");
    if(O == 'T' || O == 'C')
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < m; ++j) {
                if(O == 'T')
                    b[j * ldb + i] = a[i * lda + j];
                else
                    b[j * ldb + i] = conj(a[i * lda + j]);
            }
    else if(O == 'R' && is_complex)
        for(int i = 0; i < n; ++i)
            std::transform(a + i * lda, a + i * lda + m, b + i * ldb, [](const K& z) { return conj(z); });
    else
        for(int i = 0; i < n; ++i)
            std::copy_n(a + i * lda, m, b + i * ldb);
}
template<class K>
template<char O>
inline void Wrapper<K>::imatcopy(const int n, const int m, K* const ab, const int lda, const int ldb) {
    static_assert(O == 'N' || O == 'R' || O == 'T' || O == 'C', "Unknown operation");
    if(O == 'T' || O == 'C') {
        if(n == m && lda == ldb) {
            for(int i = 0; i < n - 1; ++i)
                for(int j = i + 1; j < n; ++j) {
                    if(O == 'C' && is_complex) {
                        ab[i * lda + j] = conj(ab[i * lda + j]);
                        ab[j * lda + i] = conj(ab[j * lda + i]);
                        std::swap(ab[i * lda + j], ab[j * lda + i]);
                    }
                    else
                        std::swap(ab[i * lda + j], ab[j * lda + i]);
                }
        }
        else
            cycle<O>(n, m, ab, 1, lda, ldb);
    }
    else if(O == 'R' && is_complex) {
        if(lda == ldb) {
            for(int i = 0; i < n; ++i)
                std::for_each(ab + i * lda, ab + i * lda + m, [](K& z) { z = conj(z); });
        }
        else if (lda < ldb) {
            for(int i = n; i-- > 0; )
                for(int j = m; j-- > 0; )
                    ab[i * ldb + j] = conj(ab[i * lda + j]);
        }
        else {
            for(int i = 0; i < n; ++i)
                for(int j = 0; j < m; ++j)
                    ab[i * ldb + j] = conj(ab[i * lda + j]);
        }
    }
    else {
        if(lda < ldb)
            for(int i = n; i > 0; --i)
                std::copy_backward(ab + (i - 1) * lda, ab + (i - 1) * lda + m, ab + (i - 1) * ldb + m);
        else if(lda > ldb)
            for(int i = 1; i < n; ++i)
                std::copy_n(ab + i * lda, m, ab + i * ldb);
    }
}
#endif // HPDDM_MKL

template<class K>
template<char S, char N, char M>
inline void Wrapper<K>::bsrcoo(const int n, const unsigned short bs, K* const a, const int* const ia, const int* const ja, K*& b, int*& ib, int*& jb, const int& shift) {
    if(S != 'S' || bs == 1)
        b = a;
    unsigned int nnz = (ia[n] - (N == 'F')) * bs * bs;
    if(S == 'S' && bs > 1) {
        for(unsigned int i = 0; i < n; ++i)
            if(ja[ia[i] - (N == 'F')] - (N == 'F') == i + shift)
                nnz -= (bs * (bs - 1)) / 2;
        b = new K[nnz];
    }
    ib = new int[2 * nnz];
    jb = ib + nnz;
    nnz = 0;
    for(unsigned int i = 0; i < n; ++i) {
        for(unsigned int j = ia[i]; j < ia[i + 1]; ++j) {
            if(N == 'F') {
                for(unsigned short l = 0; l < bs; ++l) {
                    for(unsigned short k = 0; k < ((S == 'S' && bs > 1 && (ja[j - 1] - 1 == i + shift)) ? l + 1 : bs); ++k, ++nnz) {
                        ib[nnz] = (i + shift) * bs + k + (M == 'F');
                        jb[nnz] = (ja[j - 1] - 1) * bs + l + (M == 'F');
                        if(S == 'S' && bs > 1)
                            b[nnz] = a[(j - 1) * bs * bs + k + l * bs];
                    }
                }
            }
            else {
                for(unsigned short l = 0; l < bs; ++l) {
                    for(unsigned short k = 0; ((S == 'S' && bs > 1 && (ja[j] == i + shift)) ? l : 0) < bs; ++k, ++nnz) {
                        ib[nnz] = (i + shift) * bs + k + (M == 'F');
                        jb[nnz] = ja[j] * bs + l + (M == 'F');
                        if(S == 'S' && bs > 1)
                            b[nnz] = a[j * bs * bs + l + k * bs];
                    }
                }
            }
        }
    }
}
template<class K>
inline void Wrapper<K>::diag(const int& m, const underlying_type<K>* const d, const K* const in, K* const out, const int& n) {
    if(d) {
        if(in)
            for(int i = 0; i < n; ++i)
                for(int j = 0; j < m; ++j)
                    out[j + i * m] = d[j] * in[j + i * m];
        else
            for(int i = 0; i < n; ++i)
                for(int j = 0; j < m; ++j)
                    out[j + i * m] *= d[j];
    }
    else if(in)
        std::copy_n(in, n * m, out);
}
template<class K>
template<char N>
inline void Wrapper<K>::csrmm(bool sym, const int* const n, const int* const m, const K* const a, const int* const ia, const int* const ja, const K* const x, K* const y) {
    csrmm<N>("N", n, m, n, &d__1, sym, a, ia, ja, x, &d__0, y);
}
template<class K>
template<char N>
inline void Wrapper<K>::bsrmm(bool sym, const int* const n, const int* const m, const int* const bs, const K* const a, const int* const ia, const int* const ja, const K* const x, K* const y) {
    bsrmm<N>("N", n, m, n, bs, &d__1, sym, a, ia, ja, x, &d__0, y);
}
#if HPDDM_LIBXSMM
#define HPDDM_GENERATE_LIBXSMM(C, T)                                                                         \
template<>                                                                                                   \
template<char N>                                                                                             \
inline void Wrapper<T>::bsrmm(const char* const trans, const int* const m, const int* const n,               \
                              const int* const k, const int* const bs, const T* const alpha, bool sym,       \
                              const T* const a, const int* const ia, const int* const ja, const T* const x,  \
                              const T* const beta, T* const y) {                                             \
    if(*trans == 'N' && !sym) {                                                                              \
        const int ldb = *bs * *k;                                                                            \
        const int ldc = *bs * *m;                                                                            \
        int j = ldc * *n;                                                                                    \
        if(beta == &d__0)                                                                                    \
            std::fill_n(y, j, T());                                                                          \
        else if(beta != &d__1)                                                                               \
            Blas<T>::scal(&j, beta, y, &i__1);                                                               \
        for(int i = 0; i < *m; ++i) {                                                                        \
            for(j = ia[i] - (N == 'F'); j < ia[i + 1] - (N == 'F'); ++j)                                     \
                libxsmm_ ## C ## gemm("N", "N", bs, n, bs, alpha, a + *bs * *bs * j, bs, x + *bs * (ja[j] - (N == 'F')), &ldb, &(Wrapper<T>::d__1), y + *bs * i, &ldc); \
        }                                                                                                    \
    }                                                                                                        \
    else {                                                                                                   \
        const int ldb = *bs * *m;                                                                            \
        const int ldc = *bs * *k;                                                                            \
        int j = ldc * *n;                                                                                    \
        if(beta == &d__0)                                                                                    \
            std::fill_n(y, j, T());                                                                          \
        else if(beta != &d__1)                                                                               \
            Blas<T>::scal(&j, beta, y, &i__1);                                                               \
        if(Wrapper<T>::is_complex && *trans == 'C' && (sym || N == 'C')) {                                   \
            T* const c = const_cast<T*>(a);                                                                  \
            for(int i = 0; i < *bs * *bs * (ia[*m] - (N == 'F')); ++i)                                       \
                c[i] = conj(c[i]);                                                                           \
        }                                                                                                    \
        if(sym) {                                                                                            \
            for(int i = 0; i < *m; ++i) {                                                                    \
                for(int l = ia[i] - (N == 'F'); l < ia[i + 1] - (N == 'F'); ++l) {                           \
                    j = ja[l] - (N == 'F');                                                                  \
                    Blas<T>::gemm("T", "N", bs, n, bs, alpha, a + *bs * *bs * l, bs, x + *bs * i, &ldb,      \
                                  &(Wrapper<T>::d__1), y + *bs * j, &ldc);                                   \
                    if(i != j)                                                                               \
                        libxsmm_ ## C ## gemm("N", "N", bs, n, bs, alpha, a + *bs * *bs * l, bs, x + *bs * j, &ldb, &(Wrapper<T>::d__1), y + *bs * i, &ldc); \
                }                                                                                            \
            }                                                                                                \
        }                                                                                                    \
        else {                                                                                               \
            for(int i = 0; i < *m; ++i)                                                                      \
                for(int l = ia[i] - (N == 'F'); l < ia[i + 1] - (N == 'F'); ++l)                             \
                    Blas<T>::gemm(N == 'F' ? trans : "T", "N", bs, n, bs, alpha, a + *bs * *bs * l, bs,      \
                                  x + *bs * i, &ldb, &(Wrapper<T>::d__1), y + *bs * (ja[l] - (N == 'F')),    \
                                  &ldc);                                                                     \
        }                                                                                                    \
        if(Wrapper<T>::is_complex && *trans == 'C' && (sym || N == 'C')) {                                   \
            T* const c = const_cast<T*>(a);                                                                  \
            for(int i = 0; i < *bs * *bs * (ia[*m] - (N == 'F')); ++i)                                       \
                c[i] = conj(c[i]);                                                                           \
        }                                                                                                    \
    }                                                                                                        \
}
HPDDM_GENERATE_LIBXSMM(s, float)
HPDDM_GENERATE_LIBXSMM(d, double)
#endif

template<class Idx, class T>
inline void reorder(const Idx& i, const Idx& j, const T& v) {
    std::swap(v[i], v[j]);
}
template<class Idx, class First, class... Rest>
inline void reorder(const Idx& i, const Idx& j, const First& first, const Rest&... rest) {
    std::swap(first[i], first[j]);
    reorder(i, j, rest...);
}
/* Function: reorder
 *  Rearranges an arbitrary number of containers based on the permutation defined by the first argument. */
template<class T, class... Args>
inline void reorder(std::vector<T>& order, const Args&... arguments) {
    static_assert(sizeof...(arguments) > 0, "Nothing to reorder");
    for(T i = 0; i < order.size() - 1; ++i) {
        T j = order[i];
        if(j != i) {
            T k = i + 1;
            while(order[k] != i)
                ++k;
            std::swap(order[i], order[k]);
            reorder(i, j, arguments...);
        }
    }
}
} // HPDDM
#endif // HPDDM_WRAPPER_HPP_
