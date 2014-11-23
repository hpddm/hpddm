/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <jolivet@ann.jussieu.fr>
        Date: 2014-08-04

   Copyright (C) 2011-2014 Université de Grenoble

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

#ifndef _WRAPPER_
#define _WRAPPER_

namespace HPDDM {
/* Class: Wrapper
 *
 *  A class for handling all dense and sparse linear algebra.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Wrapper {
    public:
        /* Function: mpi_type
         *  Returns the MPI datatype of the template parameter of <Wrapper>. */
        static inline MPI_Datatype mpi_type();
        /* Variable: transc
         *  Transposed real operators or conjugated transposed complex operators. */
        static const char transc;
        /* Variable: I
         *  Numbering of a sparse <MatrixCSR>. */
        static const char I;
        /* Variable: d__0
         *  Zero. */
        static const K d__0;
        /* Variable: d__1
         *  One. */
        static const K d__1;
        /* Variable: d__2
         *  Minus one. */
        static const K d__2;

        /* Function: axpy
         *  Computes a scalar-vector product and adds the result to a vector. */
        static inline void axpy(const int* const, const K* const, const K* const, const int* const, K* const, const int* const);
        /* Function: scal
         *  Computes a scalar-vector product. */
        static inline void scal(const int* const, const K* const, K* const, const int* const);
        /* Function: nrm2
         *  Computes the Euclidean norm of a vector. */
        static inline double nrm2(const int* const, const K* const, const int* const);
        /* Function: dot
         *  Computes a vector-vector dot product. */
        static inline double dot(const int* const, const K* const, const int* const, const K* const, const int* const);
        /* Function: lacpy
         *  Copies all or part of a two-dimensional matrix. */
        static inline void lacpy(const char* const, const int* const, const int* const, const K* const, const int* const, K* const, const int* const);

        /* Function: symv
         *  Computes a symmetric scalar-matrix-vector product. */
        static inline void symv(const char* const, const int* const, const K* const, const K* const, const int* const,
                                const K* const, const int* const, const K* const, K* const, const int* const);
        /* Function: gemv
         *  Computes a scalar-matrix-vector product. */
        static inline void gemv(const char* const, const int* const, const int* const, const K* const, const K* const,
                                const int* const, const K* const, const int* const, const K* const, K* const, const int* const);
        /* Function: symm
         *  Computes a symmetric scalar-matrix-matrix product. */
        static inline void symm(const char* const, const char* const, const int* const, const int* const, const K* const, const K* const,
                                const int* const, const K* const, const int* const, const K* const, K* const, const int* const);
        /* Function: gemm
         *  Computes a scalar-matrix-matrix product. */
        static inline void gemm(const char* const, const char* const, const int* const, const int* const, const int* const, const K* const, const K* const,
                                const int* const, const K* const, const int* const, const K* const, K* const, const int* const);

        /* Function: csrmv
         *  Computes a sparse square matrix-vector product. */
        template<char>
        static inline void csrmv(bool, const int* const, const K* const, const int* const, const int* const, const K* const, K* const);
        /* Function: csrgemv
         *  Computes a scalar-sparse matrix-vector product. */
        template<char>
        static inline void csrgemv(const char* const, const int* const, const int* const, const K* const, bool,
                                   const K* const, const int* const, const int* const, const K* const, const K* const, K* const);
        /* Function: csrgemm
         *  Computes a scalar-sparse matrix-matrix product. */
        template<char>
        static inline void csrgemm(const char* const, const int* const, const int* const, const int* const, const K* const, bool,
                                   const K* const, const int* const, const int* const, const K* const, const int* const,
                                   const K* const, K* const, const int* const);

        /* Function: csrcsc
         *  Converts a matrix stored in Compressed Sparse Row format into Compressed Sparse Column format. */
        template<char>
        static inline void csrcsc(const int* const, K* const, int* const, int* const, K* const, int* const, int* const);
        /* Function: gthr
         *  Gathers the elements of a full-storage sparse vector into compressed form. */
        static inline void gthr(const int&, const K* const, K* const, const int* const);
        /* Function: sctr
         *  Scatters the elements of a compressed sparse vector into full-storage form. */
        static inline void sctr(const int&, const K* const, const int* const, K* const);
        /* Function: diagv(in-place)
         *  Computes a vector-vector element-wise multiplication. */
        static inline void diagv(const int&, const double* const, K* const);
        /* Function: diagv
         *  Computes a vector-vector element-wise multiplication. */
        static inline void diagv(const int&, const double* const, const K* const, K* const);
        /* Function: diagm
         *  Computes a vector-matrix element-wise multiplication. */
        static inline void diagm(const int&, const int&, const double* const, const K* const, K* const);
        /* Function: axpby
         *  Computes two scalar-vector products. */
        static inline void axpby(const int&, const K&, const K* const, const int&, const K&, K* const, const int&);
        /* Function: conjugate
         *  Conjugates all elements of a matrix. */
        static inline void conjugate(const int&, const int&, const int&, K* const);
};

template<>
const char Wrapper<double>::transc = 'T';
template<>
const char Wrapper<std::complex<double>>::transc = 'C';

template<class K>
const K Wrapper<K>::d__0 = K(0.0);
template<class K>
const K Wrapper<K>::d__1 = K(1.0);
template<class K>
const K Wrapper<K>::d__2 = K(-1.0);

template<>
inline MPI_Datatype Wrapper<double>::mpi_type() { return MPI_DOUBLE; }
template<>
inline MPI_Datatype Wrapper<std::complex<double>>::mpi_type() { return MPI_DOUBLE_COMPLEX; }

template<>
inline void Wrapper<double>::axpy(const int* const n, const double* const a, const double* const x, const int* const incx, double* const y, const int* const incy) {
    HPDDM_F77(daxpy)(n, a, x, incx, y, incy);
}
template<>
inline void Wrapper<std::complex<double>>::axpy(const int* const n, const std::complex<double>* const a, const std::complex<double>* const x, const int* const incx, std::complex<double>* const y, const int* const incy) {
    HPDDM_F77(zaxpy)(n, a, x, incx, y, incy);
}
template<>
inline void Wrapper<double>::scal(const int* const n, const double* const a, double* const x, const int* const incx) {
    HPDDM_F77(dscal)(n, a, x, incx);
}
template<>
inline void Wrapper<std::complex<double>>::scal(const int* const n, const std::complex<double>* const a, std::complex<double>* const x, const int* const incx) {
    HPDDM_F77(zscal)(n, a, x, incx);
}
template<>
inline double Wrapper<double>::nrm2(const int* const n, const double* const x, const int* const incx) {
    return HPDDM_F77(dnrm2)(n, x, incx);
}
template<>
inline double Wrapper<std::complex<double>>::nrm2(const int* const n, const std::complex<double>* const x, const int* const incx) {
    return HPDDM_F77(dznrm2)(n, x, incx);
}
template<>
inline double Wrapper<double>::dot(const int* const n, const double* const x, const int* const incx, const double* const y, const int* const incy) {
    return HPDDM_F77(ddot)(n, x, incx, y, incy);
}
template<>
inline double Wrapper<std::complex<double>>::dot(const int* const n, const std::complex<double>* const x, const int* const incx, const std::complex<double>* const y, const int* const incy) {
#if HPDDM_MKL || defined(__APPLE__)
    std::complex<double> res;
    zdotc(&res, n, x, incx, y, incy);
#else
    std::complex<double> res = HPDDM_F77(zdotc)(n, x, incx, y, incy);
#endif
    return std::real(res);
}
template<>
inline void Wrapper<double>::lacpy(const char* const uplo, const int* const m, const int* const n, const double* const a, const int* const lda, double* const b, const int* const ldb) {
    HPDDM_F77(dlacpy)(uplo, m, n, a, lda, b, ldb);
}
template<>
inline void Wrapper<std::complex<double>>::lacpy(const char* const uplo, const int* const m, const int* const n, const std::complex<double>* const a, const int* const lda, std::complex<double>* const b, const int* const ldb) {
    HPDDM_F77(zlacpy)(uplo, m, n, a, lda, b, ldb);
}

template<>
inline void Wrapper<double>::symv(const char* const uplo, const int* const n, const double* const alpha, const double* const a, const int* const lda, const double* const b, const int* const ldb, const double* const beta, double* const c, const int* const ldc) {
    HPDDM_F77(dsymv)(uplo, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
template<>
inline void Wrapper<double>::gemv(const char* const trans, const int* const m, const int* const n, const double* const alpha, const double* const a, const int* const lda, const double* const b, const int* const ldb, const double* const beta, double* const c, const int* const ldc) {
    HPDDM_F77(dgemv)(trans, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
template<>
inline void Wrapper<std::complex<double>>::gemv(const char* const trans, const int* const m, const int* const n, const std::complex<double>* const alpha, const std::complex<double>* const a, const int* const lda, const std::complex<double>* const b, const int* const ldb, const std::complex<double>* const beta, std::complex<double>* const c, const int* const ldc) {
    HPDDM_F77(zgemv)(trans, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
template<>
inline void Wrapper<double>::symm(const char* const side, const char* const uplo, const int* const m, const int* const n, const double* const alpha, const double* const a, const int* const lda, const double* const b, const int* const ldb, const double* const beta, double* const c, const int* const ldc) {
    HPDDM_F77(dsymm)(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
template<>
inline void Wrapper<double>::gemm(const char* const transa, const char* const transb, const int* const m, const int* const n, const int* const k, const double* const alpha, const double* const a, const int* const lda, const double* const b, const int* const ldb, const double* const beta, double* const c, const int* const ldc) {
    HPDDM_F77(dgemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
template<>
inline void Wrapper<std::complex<double>>::gemm(const char* const transa, const char* const transb, const int* const m, const int* const n, const int* const k, const std::complex<double>* const alpha, const std::complex<double>* const a, const int* const lda, const std::complex<double>* const b, const int* const ldb, const std::complex<double>* const beta, std::complex<double>* const c, const int* const ldc) {
    HPDDM_F77(zgemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template<class K>
inline void Wrapper<K>::diagv(const int& n, const double* const d, K* const in) {
    diagv(n, d, nullptr, in);
}

#if HPDDM_MKL
template<class K>
const char Wrapper<K>::I = 'F';

template<char N>
struct matdescr {
    static const char a[];
    static const char b[];
};

template<char N>
const char matdescr<N>::a[6] = { 'G', '0', '0', N, '0', '0' };
template<char N>
const char matdescr<N>::b[6] = { 'S', 'L', 'N', N, '0', '0' };

template<>
template<char N>
inline void Wrapper<double>::csrmv(bool sym, const int* const n, const double* const a, const int* const ia, const int* const ja, const double* const x, double* const y) {
    static_assert(N == 'C', "Unsupported matrix indexing");
    if(sym)
        mkl_cspblas_dcsrsymv(&uplo, n, a, ia, ja, x, y);
    else
        mkl_cspblas_dcsrgemv(&transa, n, a, ia, ja, x, y);
}
template<>
template<char N>
inline void Wrapper<std::complex<double>>::csrmv(bool sym, const int* const n, const std::complex<double>* const a, const int* const ia, const int* const ja, const std::complex<double>* const x, std::complex<double>* const y) {
    static_assert(N == 'C', "Unsupported matrix indexing");
    if(sym)
        mkl_cspblas_zcsrsymv(&uplo, n, a, ia, ja, x, y);
    else
        mkl_cspblas_zcsrgemv(&transa, n, a, ia, ja, x, y);
}
template<>
template<char N>
inline void Wrapper<double>::csrgemv(const char* const trans, const int* const m, const int* const k, const double* const alpha, bool sym,
                                     const double* const a, const int* const ia, const int* const ja, const double* const x, const double* const beta, double* const y) {
    mkl_dcsrmv(trans, m, k, alpha, sym ? matdescr<N>::b : matdescr<N>::a, a, ja, ia, ia + 1, x, beta, y);
}
template<>
template<char N>
inline void Wrapper<std::complex<double>>::csrgemv(const char* const trans, const int* const m, const int* const k, const std::complex<double>* const alpha, bool sym,
                                     const std::complex<double>* const a, const int* const ia, const int* const ja, const std::complex<double>* const x, const std::complex<double>* const beta, std::complex<double>* const y) {
    mkl_zcsrmv(trans, m, k, alpha, sym ? matdescr<N>::b : matdescr<N>::a, a, ja, ia, ia + 1, x, beta, y);
}
template<>
template<char N>
inline void Wrapper<double>::csrgemm(const char* const trans, const int* const m, const int* const n, const int* const k, const double* const alpha, bool sym,
                                     const double* const a, const int* const ia, const int* const ja, const double* const x, const int* const ldb, const double* const beta, double* const y, const int* const ldc) {
    mkl_dcsrmm(trans, m, n, k, alpha, sym ? matdescr<N>::b : matdescr<N>::a, a, ja, ia, ia + 1, x, ldb, beta, y, ldc);
}
template<>
template<char N>
inline void Wrapper<std::complex<double>>::csrgemm(const char* const trans, const int* const m, const int* const n, const int* const k, const std::complex<double>* const alpha, bool sym,
                                     const std::complex<double>* const a, const int* const ia, const int* const ja, const std::complex<double>* const x, const int* const ldb, const std::complex<double>* const beta, std::complex<double>* const y, const int* const ldc) {
    mkl_zcsrmm(trans, m, n, k, alpha, sym ? matdescr<N>::b : matdescr<N>::a, a, ja, ia, ia + 1, x, ldb, beta, y, ldc);
}

template<>
template<char N>
inline void Wrapper<double>::csrcsc(const int* const n, double* const a, int* const ja, int* const ia, double* const b, int* const jb, int* const ib) {
    int job[6] = { 0, 0, N == 'F', 0, 0, 1 };
    int error;
    mkl_dcsrcsc(job, n, a, ja, ia, b, jb, ib, &error);
}
template<>
template<char N>
inline void Wrapper<std::complex<double>>::csrcsc(const int* const n, std::complex<double>* const a, int* const ja, int* const ia, std::complex<double>* const b, int* const jb, int* const ib) {
    int job[6] = { 0, 0, N == 'F', 0, 0, 1 };
    int error;
    mkl_zcsrcsc(job, n, a, ja, ia, b, jb, ib, &error);
}
template<>
inline void Wrapper<double>::gthr(const int& n, const double* const y, double* const x, const int* const indx) {
    cblas_dgthr(n, y, x, indx);
}
template<>
inline void Wrapper<std::complex<double>>::gthr(const int& n, const std::complex<double>* const y, std::complex<double>* const x, const int* const indx) {
    cblas_zgthr(n, y, x, indx);
}
template<>
inline void Wrapper<double>::sctr(const int& n, const double* const x, const int* const indx, double* const y) {
    cblas_dsctr(n, x, indx, y);
}
template<>
inline void Wrapper<std::complex<double>>::sctr(const int& n, const std::complex<double>* const x, const int* const indx, std::complex<double>* const y) {
    cblas_zsctr(n, x, indx, y);
}
template<>
inline void Wrapper<double>::diagv(const int& n, const double* const d, const double* const in, double* const out) {
    if(in)
        vdMul(n, d, in, out);
    else
        vdMul(n, d, out, out);
}
template<>
inline void Wrapper<double>::diagm(const int& m, const int& n, const double* const d, const double* const in, double* const out) {
    for(int i = 0; i < n; ++i)
        vdMul(m, d, in + i * m, out + i * m);
}
template<>
inline void Wrapper<double>::axpby(const int& n, const double& alpha, const double* const u, const int& incx, const double& beta, double* const v, const int& incy) {
    cblas_daxpby(n, alpha, u, incx, beta, v, incy);
}
template<>
inline void Wrapper<std::complex<double>>::axpby(const int& n, const std::complex<double>& alpha, const std::complex<double>* const u, const int& incx, const std::complex<double>& beta, std::complex<double>* const v, const int& incy) {
    cblas_zaxpby(n, &alpha, u, incx, &beta, v, incy);
}
#else
template<class K>
const char Wrapper<K>::I = 'C';

template<class K>
template<char N>
inline void Wrapper<K>::csrmv(bool sym, const int* const n, const K* const a, const int* const ia, const int* const ja, const K* const x, K* const y) {
    int i, j, k;
    K res;
    if(sym) {
        std::fill(y, y + *n, 0.0);
#pragma omp parallel for private(k, j, res) schedule(static, HPDDM_GRANULARITY)
        for(i = 0; i < *n; ++i) {
            if(ia[i + 1] != ia[i]) {
                res = K();
                for(k = ia[i] - (N == 'F'); k < ia[i + 1] - 1 - (N == 'F'); ++k) {
                    j = ja[k] - (N == 'F');
                    res += a[k] * x[j];
                    y[j] += a[k] * x[i];
                }
                y[i] = res + a[ia[i + 1] - 1 - (N == 'F')] * x[i];
            }
        }
    }
    else {
        for(i = 0; i < *n; ++i) {
            res = K();
            for(k = ia[i] - (N == 'F'); k < ia[i + 1] - (N == 'F'); ++k) {
                j = ja[k] - (N == 'F');
                res += a[k] * x[j];
            }
            y[i] = res;
        }
    }
}
template<class K>
template<char N>
inline void Wrapper<K>::csrgemv(const char* const trans, const int* const m, const int* const k, const K* const alpha, bool sym,
                                const K* const a, const int* const ia, const int* const ja, const K* const x, const K* const beta, K* const y) {
    int i, j, l;
    K res;
    if(trans == &transa) {
        if(sym) {
            if(beta == &(Wrapper<K>::d__0))
                std::fill(y, y + *m, K());
            else if(beta != &(Wrapper<K>::d__1))
                Wrapper<K>::scal(m, beta, y, &i__1);
            for(i = 0; i < *m; ++i) {
                if(ia[i + 1] != ia[i]){
                    res = K();
                    for(l = ia[i] - (N == 'F'); l < ia[i + 1] - 1 - (N == 'F'); ++l) {
                        j = ja[l] - (N == 'F');
                        res += a[l] * x[j];
                        y[j] += *alpha * a[l] * x[i];
                    }
                    y[i] += *alpha * (res + a[ia[i + 1] - 1 - (N == 'F')] * x[i]);
                }
            }
        }
        else {
            if(beta == &(Wrapper<K>::d__0))
                std::fill(y, y + *m, K());
            for(i = 0; i < *m; ++i) {
                res = K();
                for(l = ia[i] - (N == 'F'); l < ia[i + 1] - (N == 'F'); ++l)
                    res += a[l] * x[ja[l] - (N == 'F')];
                y[i] = *alpha * res + *beta * y[i];
            }
        }
    }
    else {
        if(beta == &(Wrapper<K>::d__0))
            std::fill(y, y + *k, K());
        else if(beta != &(Wrapper<K>::d__1))
            Wrapper<K>::scal(k, beta, y, &i__1);
        if(sym) {
#pragma omp parallel for private(l, j, res) schedule(static, HPDDM_GRANULARITY)
            for(i = 0; i < *m; ++i) {
                res = K();
                for(l = ia[i] - (N == 'F'); l < ia[i + 1] - (N == 'F'); ++l) {
                    j = ja[l] - (N == 'F');
                    y[j] += *alpha * a[l] * x[i];
                    if(i != j)
                        res += a[l] * x[j];
                }
                y[i] += *alpha * res;
            }
        }
        else {
            for(i = 0; i < *m; ++i) {
                for(l = ia[i] - (N == 'F'); l < ia[i + 1] - (N == 'F'); ++l)
                    y[ja[l] - (N == 'F')] += *alpha * a[l] * x[i];
            }
        }
    }
}
template<class K>
template<char N>
inline void Wrapper<K>::csrgemm(const char* const trans, const int* const m, const int* const n, const int* const k, const K* const alpha, bool sym,
                                const K* const a, const int* const ia, const int* const ja, const K* const x, const int* const ldb, const K* const beta, K* const y, const int* const ldc) {
    int i, l;
    if(trans == &transa) {
        int dimY = *m;
        K* res;
        if(sym) {
            int j;
            int dimX = *k;
            int dimNY = dimY * *n;
            if(beta == &(Wrapper<K>::d__0))
                std::fill(y, y + dimNY, K());
            else if(beta != &(Wrapper<K>::d__1))
                Wrapper<K>::scal(&dimNY, beta, y, &i__1);
            res = new K[*n];
            for(i = 0; i < dimY; ++i) {
                std::fill(res, res + *n, K());
                for(l = ia[i] - (N == 'F'); l < ia[i + 1] - (N == 'F'); ++l) {
                    j = ja[l] - (N == 'F');
                    if(i != j)
                        for(int r = 0; r < *n; ++r) {
                            res[r] += a[l] * x[j + r * dimX];
                            y[j + r * dimY] += *alpha * a[l] * x[i + r * dimX];
                        }
                    else
                        Wrapper<K>::axpy(n, a + l, x + j, k, res, &i__1);
                }
                Wrapper<K>::axpy(n, alpha, res, &i__1, y + i, m);
            }
            delete [] res;
        }
        else {
#pragma omp parallel private(res)
            {
                res = new K[*n];
#pragma omp for private(l) schedule(static, HPDDM_GRANULARITY)
                for(i = 0; i < dimY; ++i) {
                    std::fill(res, res + *n, K());
                    for(l = ia[i] - (N == 'F'); l < ia[i + 1] - (N == 'F'); ++l)
                        Wrapper<K>::axpy(n, a + l, x + ja[l] - (N == 'F'), k, res, &i__1);
                    Wrapper<K>::axpby(*n, *alpha, res, 1, *beta, y + i, dimY);
                }
                delete [] res;
            }
        }
    }
    else {
        int dimX = *m;
        int dimY = *k;
        int dimNY = dimY * *n;
        /*
        for(int i = 0; i < *n; ++i)
            csrgemv<N>(trans, m, k, alpha, sym, a, ia, ja, x + i * dimX, beta, y + i * dimY);
        */
        if(beta == &(Wrapper<K>::d__0))
            std::fill(y, y + dimNY, K());
        else if(beta != &(Wrapper<K>::d__1))
            Wrapper<K>::scal(&dimNY, beta, y, &i__1);
        if(sym) {
            K* res = new K[*n];
            for(i = 0; i < *m; ++i) {
                std::fill(res, res + *n, K());
                for(l = ia[i] - (N == 'F'); l < ia[i + 1] - (N == 'F'); ++l) {
                    int j = ja[l] - (N == 'F');
                    if(i != j)
                        for(int r = 0; r < *n; ++r) {
                            y[j + r * dimY] += *alpha * a[l] * x[i + r * dimX];
                            res[r] += a[l] * x[j + r * dimX];
                        }
                    else {
                        const K scal = *alpha * a[l];
                        Wrapper<K>::axpy(n, &scal, x + i, m, y + j, k);
                    }
                }
                Wrapper<K>::axpy(n, alpha, res, &i__1, y + i, k);
            }
            delete [] res;
        }
        else {
            for(i = 0; i < *m; ++i) {
                for(l = ia[i] - (N == 'F'); l < ia[i + 1] - (N == 'F'); ++l) {
                    const K scal = *alpha * a[l];
                    Wrapper<K>::axpy(n, &scal, x + i, m, y + ja[l] - (N == 'F'), k);
                }
            }
        }
    }
}

template<class K>
template<char N>
inline void Wrapper<K>::csrcsc(const int* const n, K* const a, int* const ja, int* const ia, K* const b, int* const jb, int* const ib) {
    const int n_ = *n;
    int nz = ia[n_];
    int i;
    int* ptr;

    std::fill(ib, ib + n_ + 1, 0);

    for(i = 0; i < nz; ++i)
        ib[ja[i] + 1]++;

    for(i = 0; i < n_; ++i)
        ib[i + 1] += ib[i];

    for(i = 0, ptr = ia; i < n_; ++i, ++ptr)
        for(int j = *ptr; j < *(ptr + 1); ++j) {
            int k = ib[ja[j]]++;
            jb[k] = i + (N == 'F');
            b[k] = a[j];
        }

    for(int i = n_; i > 0; i--)
        ib[i] = ib[i - 1] + (N == 'F');

    ib[0] = (N == 'F');
}
template<class K>
inline void Wrapper<K>::gthr(const int& n, const K* const y, K* const x, const int* const indx) {
    for(int i = 0; i < n; ++i)
        x[i] = y[indx[i]];
}
template<class K>
inline void Wrapper<K>::sctr(const int& n, const K* const x, const int* const indx, K* const y) {
    for(int i = 0; i < n; ++i)
        y[indx[i]] = x[i];
}
#ifdef __APPLE__
template<>
inline void Wrapper<double>::axpby(const int& n, const double& alpha, const double* const u, const int& incx, const double& beta, double* const v, const int& incy) {
    catlas_daxpby(n, alpha, u, incx, beta, v, incy);
}
template<>
inline void Wrapper<std::complex<double>>::axpby(const int& n, const std::complex<double>& alpha, const std::complex<double>* const u, const int& incx, const std::complex<double>& beta, std::complex<double>* const v, const int& incy) {
    catlas_zaxpby(n, &alpha, u, incx, &beta, v, incy);
}
#else
template<class K>
inline void Wrapper<K>::axpby(const int& n, const K& alpha, const K* const u, const int& incx, const K& beta, K* const v, const int& incy) {
    if(beta == Wrapper<K>::d__0)
        for(unsigned int i = 0; i < n; ++i)
            v[i * incy] = alpha * u[i * incx];
    else
        for(unsigned int i = 0; i < n; ++i)
            v[i * incy] = alpha * u[i * incx] + beta * v[i * incy];
}
#endif // __APPLE__
#endif // HPDDM_MKL

template<class K>
inline void Wrapper<K>::diagv(const int& n, const double* const d, const K* const in, K* const out) {
    if(in)
        for(unsigned int i = 0; i < n; ++i)
            out[i] = d[i] * in[i];
    else
        for(unsigned int i = 0; i < n; ++i)
            out[i] *= d[i];
}
template<class K>
inline void Wrapper<K>::diagm(const int& m, const int& n, const double* const d, const K* const in, K* const out) {
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < m; ++j)
            out[j + i * m] = d[j] * in[j + i * m];
}
template<class K>
inline void conjugate(const int&, const int&, const int&, K* const) { }
template<>
inline void conjugate(const int& m, const int& n, const int& ld, std::complex<double>* const in) {
    for(int i = 0; i < n; ++i)
        std::transform(in + i * ld, in + i * ld + m, in + i * ld, [](std::complex<double> const& z) { return std::conj(z); });
}
} // HPDDM
#endif // _WRAPPER_
