/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2014-03-16

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

#ifndef HPDDM_LAPACK_HPP_
#define HPDDM_LAPACK_HPP_

#define HPDDM_GENERATE_EXTERN_LAPACK(C, T, U, SYM, ORT)                                                      \
void HPDDM_F77(C ## lapmt)(const int*, const int*, const int*, T*, const int*, int*);                        \
U    HPDDM_F77(C ## lange)(const char*, const int*, const int*, const T*, const int*, U*);                   \
U    HPDDM_F77(C ## lan ## SYM)(const char*, const char*, const int*, const T*, const int*, U*);             \
void HPDDM_F77(C ## SYM ## gst)(const int*, const char*, const int*, T*, const int*,                         \
                                const T*, const int*, int*);                                                 \
void HPDDM_F77(C ## SYM ## trd)(const char*, const int*, T*, const int*, U*, U*, T*, T*, const int*, int*);  \
void HPDDM_F77(C ## stein)(const int*, const U*, const U*, const int*, const U*, const int*,                 \
                           const int*, T*, const int*, U*, int*, int*, int*);                                \
void HPDDM_F77(C ## ORT ## mtr)(const char*, const char*, const char*, const int*, const int*,               \
                                const T*, const int*, const T*, T*, const int*, T*, const int*, int*);       \
void HPDDM_F77(C ## gehrd)(const int*, const int*, const int*, T*, const int*, T*, T*, const int*, int*);    \
void HPDDM_F77(C ## ORT ## mhr)(const char*, const char*, const int*, const int*, const int*, const int*,    \
                                const T*, const int*, const T*, T*, const int*, T*, const int*, int*);       \
void HPDDM_F77(C ## getrf)(const int*, const int*, T*, const int*, int*, int*);                              \
void HPDDM_F77(C ## getrs)(const char*, const int*, const int*, const T*, const int*, const int*, T*,        \
                           const int*, int*);                                                                \
void HPDDM_F77(C ## getri)(const int*, T*, const int*, const int*, T*, const int*, int*);                    \
void HPDDM_F77(C ## sytrf)(const char*, const int*, T*, const int*, int*, T*, int*, int*);                   \
void HPDDM_F77(C ## sytrs)(const char*, const int*, const int*, const T*, const int*, const int*, T*,        \
                           const int*, int*);                                                                \
void HPDDM_F77(C ## sytri)(const char*, const int*, T*, const int*, int*, T*, int*);                         \
void HPDDM_F77(C ## potrf)(const char*, const int*, T*, const int*, int*);                                   \
void HPDDM_F77(C ## potrs)(const char*, const int*, const int*, const T*, const int*, T*, const int*, int*); \
void HPDDM_F77(C ## potri)(const char*, const int*, T*, const int*, int*);                                   \
void HPDDM_F77(C ## pstrf)(const char*, const int*, T*, const int*, int*, int*, const U*, U*, int*);         \
void HPDDM_F77(C ## trtrs)(const char*, const char*, const char*, const int*, const int*, const T*,          \
                           const int*, T*, const int*, int*);                                                \
void HPDDM_F77(C ## posv)(const char*, const int*, const int*, T*, const int*, T*, const int*, int*);        \
void HPDDM_F77(C ## pptrf)(const char*, const int*, T*, int*);                                               \
void HPDDM_F77(C ## pptrs)(const char*, const int*, const int*, T*, T*, const int*, int*);                   \
void HPDDM_F77(C ## ppsv)(const char*, const int*, const int*, T*, T*, const int*, int*);                    \
void HPDDM_F77(C ## SYM ## sv)(const char*, const int*, const int*, T*, const int*, int*, T*, const int*,    \
                               T*, int*, int*);                                                              \
void HPDDM_F77(C ## geqrf)(const int*, const int*, T*, const int*, T*, T*, const int*, int*);                \
void HPDDM_F77(C ## geqrt)(const int*, const int*, const int*, T*, const int*, T*, const int*, T*, int*);    \
void HPDDM_F77(C ## gemqrt)(const char*, const char*, const int*, const int*, const int*, const int*,        \
                            const T*, const int*, const T*, const int*, T*, const int*, T*, int*);
#define HPDDM_GENERATE_EXTERN_LAPACK_COMPLEX(C, T, B, U)                                                     \
HPDDM_GENERATE_EXTERN_LAPACK(B, U, U, sy, or)                                                                \
HPDDM_GENERATE_EXTERN_LAPACK(C, T, U, he, un)                                                                \
void HPDDM_F77(B ## stebz)(const char*, const char*, const int*, const U*, const U*, const int*, const int*, \
                           const U*, const U*, const U*, int*, int*, U*, int*, int*, U*, int*, int*);        \
void HPDDM_F77(B ## pocon)(const char*, const int*, const U*, const int*, U*, U*, U*, int*, int*);           \
void HPDDM_F77(C ## pocon)(const char*, const int*, const T*, const int*, U*, U*, T*, U*, int*);             \
void HPDDM_F77(B ## geqp3)(const int*, const int*, U*, const int*, const int*, U*, U*, const int*, int*);    \
void HPDDM_F77(C ## geqp3)(const int*, const int*, T*, const int*, const int*, T*, T*, const int*, U*, int*);\
void HPDDM_F77(B ## ormqr)(const char*, const char*, const int*, const int*, const int*, const U*,           \
                           const int*, const U*, U*, const int*, U*, const int*, int*);                      \
void HPDDM_F77(C ## unmqr)(const char*, const char*, const int*, const int*, const int*, const T*,           \
                           const int*, const T*, T*, const int*, T*, const int*, int*);                      \
void HPDDM_F77(B ## hseqr)(const char*, const char*, const int*, const int*, const int*, U*, const int*, U*, \
                           U*, U*, const int*, U*, const int*, int*);                                        \
void HPDDM_F77(C ## hseqr)(const char*, const char*, const int*, const int*, const int*, T*, const int*, T*, \
                           T*, const int*, T*, const int*, int*);                                            \
void HPDDM_F77(B ## hsein)(const char*, const char*, const char*, int*, const int*, U*, const int*, U*,      \
                           const U*, U*, const int*, U*, const int*, const int*, int*, U*, int*, int*, int*);\
void HPDDM_F77(C ## hsein)(const char*, const char*, const char*, int*, const int*, T*, const int*, T*, T*,  \
                           const int*, T*, const int*, const int*, int*, T*, U*, int*, int*, int*);          \
void HPDDM_F77(B ## geev)(const char*, const char*, const int*, U*, const int*, U*, U*, U*, const int*, U*,  \
                          const int*, U*, const int*, int*);                                                 \
void HPDDM_F77(C ## geev)(const char*, const char*, const int*, T*, const int*, T*, T*, const int*, T*,      \
                          const int*, T*, const int*, U*, int*);                                             \
void HPDDM_F77(B ## ggev)(const char*, const char*, const int*, U*, const int*, U*, const int*, U*, U*, U*,  \
                          U*, const int*, U*, const int*, U*, const int*, int*);                             \
void HPDDM_F77(C ## ggev)(const char*, const char*, const int*, T*, const int*, T*, const int*, T*, T*,      \
                          T*, const int*, T*, const int*, T*, const int*, U*, int*);                         \
void HPDDM_F77(B ## gesvd)(const char*, const char*, const int*, const int*, U*, const int*, U*, U*,         \
                           const int*, U*, const int*, U*, const int*, int*);                                \
void HPDDM_F77(C ## gesvd)(const char*, const char*, const int*, const int*, T*, const int*, U*, T*,         \
                           const int*, T*, const int*, T*, const int*, U*, int*);                            \
void HPDDM_F77(B ## gesdd)(const char*, const int*, const int*, U*, const int*, U*, U*, const int*, U*,      \
                           const int*, U*, const int*, int*, int*);                                          \
void HPDDM_F77(C ## gesdd)(const char*, const int*, const int*, T*, const int*, U*, T*, const int*, T*,      \
                           const int*, T*, const int*, U*, int*, int*);

#ifndef _MKL_H_
# ifdef __cplusplus
extern "C" {
HPDDM_GENERATE_EXTERN_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HPDDM_GENERATE_EXTERN_LAPACK_COMPLEX(z, std::complex<double>, d, double)
#  if defined(PETSCHPDDM_H)
#   if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_HAVE_F2CBLASLAPACK___FLOAT128_BINDINGS)
HPDDM_GENERATE_EXTERN_LAPACK_COMPLEX(w, __complex128, q, __float128)
#   endif
#   if defined(PETSC_USE_REAL___FP16) || defined(PETSC_HAVE_F2CBLASLAPACK___FP16_BINDINGS)
HPDDM_GENERATE_EXTERN_LAPACK_COMPLEX(k, std::complex<__fp16>, h, __fp16)
#   endif
#  endif
}
# else
HPDDM_GENERATE_EXTERN_LAPACK_COMPLEX(c, void, s, float)
HPDDM_GENERATE_EXTERN_LAPACK_COMPLEX(z, void, d, double)
# endif // __cplusplus
#endif // _MKL_H_

#ifdef __cplusplus
namespace HPDDM {
/* Class: Lapack
 *
 *  A class that wraps some LAPACK routines for dense linear algebra.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
struct Lapack {
    /* Function: lapmt
     *  Performs a forward or backward permutation of the columns of a matrix. */
    static void lapmt(const int*, const int*, const int*, K*, const int*, int*);
    /* Function: lange
     *  Computes the norm of a general rectangular matrix. */
    static underlying_type<K> lange(const char*, const int*, const int*, const K*, const int*, underlying_type<K>*);
    /* Function: lan
     *  Computes the norm of a symmetric or Hermitian matrix. */
    static underlying_type<K> lan(const char*, const char*, const int*, const K*, const int*, underlying_type<K>*);
    /* Function: getrf
     *  Computes an LU factorization of a general rectangular matrix. */
    static void getrf(const int*, const int*, K*, const int*, int*, int*);
    /* Function: getrs
     *  Solves a system of linear equations with an LU-factored matrix. */
    static void getrs(const char*, const int*, const int*, const K*, const int*, const int*, K*, const int*, int*);
    /* Function: getri
     *  Computes the inverse of an LU-factored matrix. */
    static void getri(const int*, K*, const int*, int*, K*, const int*, int*);
    /* Function: sytrf
     *  Computes the Bunch--Kaufman factorization of a symmetric matrix. */
    static void sytrf(const char*, const int*, K*, const int*, int*, K*, int*, int*);
    /* Function: sytrs
     *  Solves a system of linear equations with an LDLT-factored matrix. */
    static void sytrs(const char*, const int*, const int*, const K*, const int*, const int*, K*, const int*, int*);
    /* Function: sytri
     *  Computes the inverse of an LDLT-factored matrix. */
    static void sytri(const char*, const int*, K*, const int*, int*, K*, int*);
    /* Function: potrf
     *  Computes the Cholesky factorization of a symmetric or Hermitian positive definite matrix. */
    static void potrf(const char*, const int*, K*, const int*, int*);
    /* Function: pocon
     *  Estimates the reciprocal of the condition number of a symmetric or Hermitian positive definite matrix. */
    static void pocon(const char*, const int*, const K*, const int*, underlying_type<K>*, underlying_type<K>*, K*, typename std::conditional<Wrapper<K>::is_complex, underlying_type<K>*, int*>::type, int*);
    /* Function: potrs
     *  Solves a system of linear equations with a Cholesky-factored matrix. */
    static void potrs(const char*, const int*, const int*, const K*, const int*, K*, const int*, int*);
    /* Function: potri
     *  Computes the inverse of a Cholesky-factored matrix. */
    static void potri(const char*, const int*, K*, const int*, int*);
    /* Function: pstrf
     *  Computes the Cholesky factorization of a symmetric or Hermitian positive semidefinite matrix with pivoting. */
    static void pstrf(const char*, const int*, K*, const int*, int*, int*, const underlying_type<K>*, underlying_type<K>*, int*);
    /* Function: trtrs
     *  Solves a system of linear equations with a triangular matrix. */
    static void trtrs(const char*, const char*, const char*, const int*, const int*, const K*, const int*, K*, const int*, int*);
    /* Function: posv
     *  Solves a system of linear equations with a symmetric or Hermitian positive definite matrix. */
    static void posv(const char*, const int*, const int*, K*, const int*, K*, const int*, int*);
    static void pptrf(const char*, const int*, K*, int*);
    static void pptrs(const char*, const int*, const int*, K*, K*, const int*, int*);
    /* Function: ppsv
     *  Solves a system of linear equations with a packed symmetric or Hermitian positive definite matrix. */
    static void ppsv(const char*, const int*, const int*, K*, K*, const int*, int*);
    /* Function: sv
     *  Solves a system of linear equations with a symmetric or Hermitian indefinite matrix. */
    static void sv(const char*, const int*, const int*, K*, const int*, int*, K*, const int*, K*, int*, int*);
    /* Function: geqp3
     *  Computes a QR decomposition of a rectangular matrix with column pivoting. */
    static void geqp3(const int*, const int*, K*, const int*, int*, K*, K*, const int*, underlying_type<K>*, int*);
    /* Function: geqrf
     *  Computes a QR decomposition of a rectangular matrix. */
    static void geqrf(const int*, const int*, K*, const int*, K*, K*, const int*, int*);
    /* Function: geqrt
     *  Computes a blocked QR decomposition of a rectangular matrix using the compact WY representation of Q. */
    static void geqrt(const int*, const int*, const int*, K*, const int*, K*, const int*, K*, int*);
    /* Function: gemqrt
     *  Multiplies a matrix by an orthogonal or unitary matrix obtained with <Lapack::geqrt>. */
    static void gemqrt(const char*, const char*, const int*, const int*, const int*, const int*, const K*, const int*, const K*, const int*, K*, const int*, K*, int*);
    /* Function: mqr
     *  Multiplies a matrix by an orthogonal or unitary matrix obtained with <Lapack::geq>. */
    static void mqr(const char*, const char*, const int*, const int*, const int*, const K*, const int*, const K*, K*, const int*, K*, const int*, int*);
    /* Function: gehrd
     *  Reduces a matrix to an upper Hessenberg matrix. */
    static void gehrd(const int*, const int*, const int*, K*, const int*, K*, K*, const int*, int*);
    /* Function: hseqr
     *  Computes all eigenvalues and (optionally) the Schur factorization of an upper Hessenberg matrix. */
    static void hseqr(const char*, const char*, const int*, const int*, const int*, K*, const int*, K*, K*, K*, const int*, K*, const int*, int*);
    /* Function: hsein
     *  Computes selected eigenvectors of an upper Hessenberg matrix that correspond to specified eigenvalues. */
    static void hsein(const char*, const char*, const char*, int*, const int*, K*, const int*, K*, const K*, K*, const int*, K*, const int*, const int*, int*, K*, underlying_type<K>*, int*, int*, int*);
    /* Function: mhr
     *  Multiplies a matrix by an orthogonal or unitary matrix obtained with <Lapack::gehrd>. */
    static void mhr(const char*, const char*, const int*, const int*, const int*, const int*, const K*, const int*, const K*, K*, const int*, K*, const int*, int*);
    /* Function: geev
     *  Computes the eigenvalues and the eigenvectors of a nonsymmetric eigenvalue problem. */
    static void geev(const char*, const char*, const int*, K*, const int*, K*, K*, K*, const int*, K*, const int*, K*, const int*, underlying_type<K>*, int*);
    /* Function: ggev
     *  Computes the eigenvalues and the eigenvectors of a nonsymmetric generalized eigenvalue problem. */
    static void ggev(const char*, const char*, const int*, K*, const int*, K*, const int*, K*, K*, K*, K*, const int*, K*, const int*, K*, const int*, underlying_type<K>*, int*);
    /* Function: gst
     *  Reduces a symmetric or Hermitian definite generalized eigenvalue problem to a standard form. */
    static void gst(const int*, const char*, const int*, K*, const int*, K*, const int*, int*);
    /* Function: trd
     *  Reduces a symmetric or Hermitian matrix to a tridiagonal form. */
    static void trd(const char*, const int*, K*, const int*, underlying_type<K>*, underlying_type<K>*, K*, K*, const int*, int*);
    /* Function: stein
     *  Computes the eigenvectors corresponding to specified eigenvalues of a symmetric tridiagonal matrix. */
    static void stein(const int*, const underlying_type<K>*, const underlying_type<K>*, const int*, const underlying_type<K>*, const int*, const int*, K*, const int*, underlying_type<K>*, int*, int*, int*);
    /* Function: stebz
     *  Computes selected eigenvalues of a symmetric tridiagonal matrix by bisection. */
    static void stebz(const char*, const char*, const int*, const underlying_type<K>*, const underlying_type<K>*, const int*, const int*, const underlying_type<K>*, const underlying_type<K>*, const underlying_type<K>*, int*, int*, underlying_type<K>*, int*, int*, underlying_type<K>*, int*, int*);
    /* Function: mtr
     *  Multiplies a matrix by an orthogonal or unitary matrix obtained with <Lapack::trd>. */
    static void mtr(const char*, const char*, const char*, const int*, const int*, const K*, const int*, const K*, K*, const int*, K*, const int*, int*);
    /* Function: gesvd
     *  Computes the singular value decomposition of a rectangular matrix. */
    static void gesvd(const char*, const char*, const int*, const int*, K*, const int*, underlying_type<K>*, K*, const int*, K*, const int*, K*, const int*, underlying_type<K>*, int*);
    /* Function: gesdd
     *  Computes the singular value decomposition of a rectangular matrix, and optionally the left and/or right singular vectors, using a divide and conquer algorithm. */
    static void gesdd(const char*, const int*, const int*, K*, const int*, underlying_type<K>*, K*, const int*, K*, const int*, K*, const int*, underlying_type<K>*, int*, int*);
};

/* Class: QR
 *
 *  A class to use LAPACK for computing QR decompositions.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class QR {
    private:
        int                               _n;
        int                           _lwork;
        K* const                          _a;
        K* const                        _tau;
        K* const                       _work;
# if HPDDM_QR == 1
        std::vector<int>               _jpvt;
        int                            _rank;
# endif
        /* Function: workspace
         *  Returns the optimal size of the workspace array. */
        int workspace() const {
            int info;
            int lwork[2] { -1, -1 };
            K wkopt;
# if HPDDM_QR == 1
            Lapack<K>::geqp3(&_n, &_n, nullptr, &_n, nullptr, nullptr, &wkopt, lwork, nullptr, &info);
# else
            Lapack<K>::geqrf(&_n, &_n, nullptr, &_n, nullptr, &wkopt, lwork, &info);
# endif
            lwork[0] = static_cast<int>(std::real(wkopt));
            Lapack<K>::mqr("L", "T", &_n, &i__1, &_n, nullptr, &_n, nullptr, nullptr, &_n, &wkopt, lwork + 1, &info);
            lwork[1] = static_cast<int>(std::real(wkopt));
            return *std::max_element(lwork, lwork + 1);
        }
    public:
        QR(int n, const K* const cpy = nullptr) : _n(n), _lwork(workspace()), _a(new K[_n * (_n + 1) + _lwork]), _tau(_a + _n * _n), _work(_tau + _n) {
# if HPDDM_QR == 1
            _jpvt.resize(_n);
            _rank = n;
# endif
            if(cpy)
                for(unsigned int i = 0; i < _n; ++i) {
                    _a[i * (_n + 1)] = cpy[i * (_n + 1)];
                    for(unsigned int j = i + 1; j < _n; ++j)
                        _a[j * _n + i] = _a[i * _n + j] = cpy[i * _n + j];
                }
        }
        QR(const QR&) = delete;
        ~QR() {
            delete [] _a;
        }
        /* Function: getPointer
         *  Returns the pointer <QR::a>. */
        K* getPointer() const { return _a; }
        void decompose() {
            int info;
# if HPDDM_QR == 1
            underlying_type<K>* rwork = Wrapper<K>::is_complex ? new underlying_type<K>[2 * _n] : nullptr;
            Lapack<K>::geqp3(&_n, &_n, _a, &_n, _jpvt.data(), _tau, _work, &_lwork, rwork, &info);
            delete [] rwork;
            while(std::abs(_a[(_rank - 1) * (_n + 1)]) < HPDDM_EPS * std::abs(_a[0]) && _rank-- > 0);
# else
            Lapack<K>::geqrf(&_n, &_n, _a, &_n, _tau, _work, &_lwork, &info);
            std::vector<int> jpvt;
            jpvt.reserve(6);
            underlying_type<K> max = std::abs(_a[0]);
            for(unsigned int i = 1; i < _n; ++i) {
                if(std::abs(_a[(_n + 1) * i]) < max * 1.0e-6)
                    jpvt.emplace_back(i);
                else if(std::abs(_a[(_n + 1) * i]) > max / 1.0e-6) {
                    jpvt.clear();
                    max = std::abs(_a[(_n + 1) * i]);
                    i = 0;
                }
                else
                    max = std::max(std::abs(_a[(i + 1) * _n]), max);
            }
            std::for_each(jpvt.cbegin(), jpvt.cend(), [&](const int i) { std::fill_n(_a + _n * i, i, K()); _a[(_n + 1) * i] = Wrapper<K>::d__1; });
# endif
        }
        /* Function: solve
         *  Computes the solution of a least squares problem. */
        void solve(K* const x) const {
            int info;
            Lapack<K>::mqr("L", "T", &_n, &i__1, &_n, _a, &_n, _tau, x, &_n, _work, &_lwork, &info);
# if HPDDM_QR == 1
            Lapack<K>::trtrs("U", "N", "N", &_rank, &i__1, _a, &_n, x, &_n, &info);
            Lapack<K>::lapmt(&i__0, &i__1, &_n, x, &i__1, const_cast<int*>(_jpvt.data()));
# else
            Lapack<K>::trtrs("U", "N", "N", &_n, &i__1, _a, &_n, x, &_n, &info);
# endif
        }
};

#if defined(DLAPACK) || defined(LAPACKSUB)
#ifdef LAPACKSUB
#undef HPDDM_CHECK_COARSEOPERATOR
#define HPDDM_CHECK_SUBDOMAIN
#include "HPDDM_preprocessor_check.hpp"
#define SUBDOMAIN HPDDM::LapackTRSub
#endif
template<class K>
class LapackTRSub {
    private:
        K*                _a;
        int*           _ipiv;
        int               _n;
        unsigned short _type;
    public:
        LapackTRSub() : _a(), _ipiv(), _n(), _type() { }
        LapackTRSub(const LapackTRSub&) = delete;
        ~LapackTRSub() { dtor(); }
        static constexpr char _numbering = 'F';
        void dtor() {
            delete [] _a;
            _a = nullptr;
            delete [] _ipiv;
            _ipiv = nullptr;
        }
        template<char N = HPDDM_NUMBERING, bool transpose = false>
        void numfact(MatrixCSR<K>* const& A, bool detection = false, K* const& schur = nullptr) {
            _n = A->_n;
            _a = new K[_n * _n]();
            if(A->_nnz == _n * _n) {
                if(N == 'C')
                    Wrapper<K>::template omatcopy<'T'>(_n, _n, A->_a, _n, _a, _n);
                else
                    std::copy_n(A->_a, A->_nnz, _a);
            }
            else {
                for(unsigned int i = 0; i < A->_n; ++i) {
                    for(unsigned int j = A->_ia[i] - (N == 'F'); j < A->_ia[i + 1] - (N == 'F'); ++j) {
                        if(!transpose)
                            _a[i + (A->_ja[j] - (N == 'F')) * _n] = A->_a[j];
                        else
                            _a[i * _n + (A->_ja[j] - (N == 'F'))] = A->_a[j];
                    }
                }
            }
            int info;
            if(!A->_sym) {
                _ipiv = new int[_n];
                Lapack<K>::getrf(&_n, &_n, _a, &_n, _ipiv, &info);
            }
            else {
                _type = 1 + (Option::get()->val<char>("operator_spd", 0) && !detection);
                if(_type == 1) {
                    int lwork = -1;
                    _ipiv = new int[_n];
                    K wkopt;
                    Lapack<K>::sytrf("L", &_n, _a, &_n, _ipiv, &wkopt, &lwork, &info);
                    if(info == 0) {
                        lwork = static_cast<int>(std::real(wkopt));
                        K* work = new K[lwork];
                        Lapack<K>::sytrf("L", &_n, _a, &_n, _ipiv, work, &lwork, &info);
                        delete [] work;
                    }
                }
                else
                    Lapack<K>::potrf("L", &_n, _a, &_n, &info);
            }
        }
        template<char N = HPDDM_NUMBERING>
        int inertia(MatrixCSR<K>* const& A) {
            return 0;
        }
        unsigned short deficiency() const { return 0; }
        void solve(K* const x, const unsigned short& n = 1) const {
            int nrhs = n, info;
            if(_type == 1)
                Lapack<K>::sytrs("L", &_n, &nrhs, _a, &_n, _ipiv, x, &_n, &info);
            else if(_type == 2)
                Lapack<K>::potrs("L", &_n, &nrhs, _a, &_n, x, &_n, &info);
            else
                Lapack<K>::getrs("N", &_n, &nrhs, _a, &_n, _ipiv, x, &_n, &info);
        }
        void solve(const K* const b, K* const x, const unsigned short& n = 1) const {
            std::copy_n(b, n * _n, x);
            solve(x, n);
        }
};

#ifdef DLAPACK
#undef HPDDM_CHECK_SUBDOMAIN
#define HPDDM_CHECK_COARSEOPERATOR
#include "HPDDM_preprocessor_check.hpp"
#define COARSEOPERATOR HPDDM::LapackTR
template<class K>
class LapackTR : public DMatrix, public LapackTRSub<K> {
    private:
        typedef LapackTRSub<K> super;
    protected:
        /* Variable: numbering
         *  0-based indexing. */
        static constexpr char _numbering = 'C';
    public:
        template<char S>
        void numfact(unsigned int n, int* I, int* J, K* C) {
            MatrixCSR<K>* E;
            if(I == nullptr && J == nullptr)
                E = new MatrixCSR<K>(n, n, n * n, C, nullptr, nullptr, S == 'S');
            else
                E = new MatrixCSR<K>(n, n, I[n] - (_numbering == 'F'), C, I, J, S == 'S');
            this->super::template numfact<_numbering, true>(E);
            delete E;
            delete [] I;
        }

};
#endif
#endif

# define HPDDM_GENERATE_LAPACK(C, T, B, U, SYM, ORT)                                                         \
template<>                                                                                                   \
inline void Lapack<T>::lapmt(const int* forwrd, const int* m, const int* n, T* x, const int* ldx, int* k) {  \
    HPDDM_F77(C ## lapmt)(forwrd, m, n, x, ldx, k);                                                          \
}                                                                                                            \
template<>                                                                                                   \
inline U Lapack<T>::lange(const char* norm, const int* m, const int* n, const T* a, const int* lda,          \
                          U* work) {                                                                         \
    return HPDDM_F77(C ## lange)(norm, m, n, a, lda, work);                                                  \
}                                                                                                            \
template<>                                                                                                   \
inline U Lapack<T>::lan(const char* norm, const char* uplo, const int* m, const T* a, const int* lda,        \
                        U* work) {                                                                           \
    return HPDDM_F77(C ## lan ## SYM)(norm, uplo, m, a, lda, work);                                          \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::gst(const int* itype, const char* uplo, const int* n,                                 \
                           T* a, const int* lda, T* b, const int* ldb, int* info) {                          \
    HPDDM_F77(C ## SYM ## gst)(itype, uplo, n, a, lda, b, ldb, info);                                        \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::trd(const char* uplo, const int* n, T* a, const int* lda,                             \
                           U* d, U* e, T* tau, T* work, const int* lwork, int* info) {                       \
    HPDDM_F77(C ## SYM ## trd)(uplo, n, a, lda, d, e, tau, work, lwork, info);                               \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::stein(const int* n, const U* d, const U* e, const int* m, const U* w,                 \
                             const int* iblock, const int* isplit, T* z, const int* ldz,                     \
                             U* work, int* iwork, int* ifailv, int* info) {                                  \
    HPDDM_F77(C ## stein)(n, d, e, m, w, iblock, isplit, z, ldz, work, iwork, ifailv, info);                 \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::stebz(const char* range, const char* order, const int* n, const U* vl, const U* vu,   \
                             const int* il, const int* iu, const U* abstol, const U* d, const U* e, int* m,  \
                             int* nsplit, U* w, int* iblock, int* isplit, U* work, int* iwork, int* info) {  \
    HPDDM_F77(B ## stebz)(range, order, n, vl, vu, il, iu, abstol, d, e, m, nsplit, w, iblock, isplit,       \
                          work, iwork, info);                                                                \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::mtr(const char* side, const char* uplo, const char* trans, const int* m,              \
                           const int* n, const T* a, const int* lda, const T* tau, T* c, const int* ldc,     \
                           T* work, const int* lwork, int* info) {                                           \
    HPDDM_F77(C ## ORT ## mtr)(side, uplo, trans, m, n, a, lda, tau, c, ldc, work, lwork, info);             \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::gehrd(const int* n, const int* ilo, const int* ihi, T* a, const int* lda, T* tau,     \
                             T* work, const int* lwork, int* info) {                                         \
    HPDDM_F77(C ## gehrd)(n, ilo, ihi, a, lda, tau, work, lwork, info);                                      \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::getrf(const int* m, const int* n, T* a, const int* lda, int* ipiv, int* info) {       \
    HPDDM_F77(C ## getrf)(m, n, a, lda, ipiv, info);                                                         \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::getrs(const char* trans, const int* n, const int* nrhs, const T* a, const int* lda,   \
                             const int* ipiv, T* b, const int* ldb, int* info) {                             \
    HPDDM_F77(C ## getrs)(trans, n, nrhs, a, lda, ipiv, b, ldb, info);                                       \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::getri(const int* n, T* a, const int* lda, int* ipiv, T* work, const int* lwork,       \
                             int* info) {                                                                    \
    HPDDM_F77(C ## getri)(n, a, lda, ipiv, work, lwork, info);                                               \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::sytrf(const char* uplo, const int* n, T* a, const int* lda, int* ipiv, T* work,       \
                             int* lwork, int* info) {                                                        \
    HPDDM_F77(C ## sytrf)(uplo, n, a, lda, ipiv, work, lwork, info);                                         \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::sytrs(const char* uplo, const int* n, const int* nrhs, const T* a, const int* lda,    \
                             const int* ipiv, T* b, const int* ldb, int* info) {                             \
    HPDDM_F77(C ## sytrs)(uplo, n, nrhs, a, lda, ipiv, b, ldb, info);                                        \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::sytri(const char* uplo, const int* n, T* a, const int* lda, int* ipiv, T* work,       \
                             int* info) {                                                                    \
    HPDDM_F77(C ## sytri)(uplo, n, a, lda, ipiv, work, info);                                                \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::potrf(const char* uplo, const int* n, T* a, const int* lda, int* info) {              \
    HPDDM_F77(C ## potrf)(uplo, n, a, lda, info);                                                            \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::potrs(const char* uplo, const int* n, const int* nrhs, const T* a, const int* lda,    \
                             T* b, const int* ldb, int* info) {                                              \
    HPDDM_F77(C ## potrs)(uplo, n, nrhs, a, lda, b, ldb, info);                                              \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::potri(const char* uplo, const int* n, T* a, const int* lda, int* info) {              \
    HPDDM_F77(C ## potri)(uplo, n, a, lda, info);                                                            \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::pstrf(const char* uplo, const int* n, T* a, const int* lda, int* piv, int* rank,      \
                             const U* tol, U* work, int* info) {                                             \
    HPDDM_F77(C ## pstrf)(uplo, n, a, lda, piv, rank, tol, work, info);                                      \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::trtrs(const char* uplo, const char* trans, const char* diag, const int* n,            \
                             const int* nrhs, const T* a, const int* lda, T* b, const int* ldb, int* info) { \
    HPDDM_F77(C ## trtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);                                 \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::posv(const char* uplo, const int* n, const int* nrhs, T* a, const int* lda,           \
                            T* b, const int* ldb, int* info) {                                               \
    HPDDM_F77(C ## posv)(uplo, n, nrhs, a, lda, b, ldb, info);                                               \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::pptrf(const char* uplo, const int* n, T* ap, int* info) {                             \
    HPDDM_F77(C ## pptrf)(uplo, n, ap, info);                                                                \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::pptrs(const char* uplo, const int* n, const int* nrhs, T* ap, T* b,                   \
                             const int* ldb, int* info) {                                                    \
    HPDDM_F77(C ## pptrs)(uplo, n, nrhs, ap, b, ldb, info);                                                  \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::ppsv(const char* uplo, const int* n, const int* nrhs, T* ap, T* b,                    \
                            const int* ldb, int* info) {                                                     \
    HPDDM_F77(C ## ppsv)(uplo, n, nrhs, ap, b, ldb, info);                                                   \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::sv(const char* uplo, const int* n, const int* nrhs, T* a, const int* lda, int* ipiv,  \
                          T* b, const int* ldb, T* work, int* lwork, int* info) {                            \
    HPDDM_F77(C ## SYM ## sv)(uplo, n, nrhs, a, lda, ipiv, b, ldb, work, lwork, info);                       \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::geqrf(const int* m, const int* n, T* a, const int* lda, T* tau, T* work,              \
                             const int* lwork, int* info) {                                                  \
    HPDDM_F77(C ## geqrf)(m, n, a, lda, tau, work, lwork, info);                                             \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::geqrt(const int* m, const int* n, const int* nb, T* a, const int* lda, T* t,          \
                             const int* ldt, T* work, int* info) {                                           \
    HPDDM_F77(C ## geqrt)(m, n, nb, a, lda, t, ldt, work, info);                                             \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::gemqrt(const char* side, const char* trans, const int* m, const int* n, const int* k, \
                              const int* nb, const T* v, const int* ldv, const T* t, const int* ldt, T* c,   \
                              const int* ldc, T* work, int* info) {                                          \
    HPDDM_F77(C ## gemqrt)(side, trans, m, n, k, nb, v, ldv, t, ldt, c, ldc, work, info);                    \
}
# define HPDDM_GENERATE_LAPACK_COMPLEX(C, T, B, U)                                                           \
HPDDM_GENERATE_LAPACK(B, U, B, U, sy, or)                                                                    \
HPDDM_GENERATE_LAPACK(C, T, B, U, he, un)                                                                    \
template<>                                                                                                   \
inline void Lapack<U>::pocon(const char* uplo, const int* n, const U* a, const int* lda, U* anorm, U* rcond, \
                             U* work, int* iwork, int* info) {                                               \
    HPDDM_F77(B ## pocon)(uplo, n, a, lda, anorm, rcond, work, iwork, info);                                 \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::pocon(const char* uplo, const int* n, const T* a, const int* lda, U* anorm, U* rcond, \
                             T* work, U* rwork, int* info) {                                                 \
    HPDDM_F77(C ## pocon)(uplo, n, a, lda, anorm, rcond, work, rwork, info);                                 \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<U>::geqp3(const int* m, const int* n, U* a, const int* lda, int* jpvt, U* tau, U* work,   \
                             const int* lwork, U*, int* info) {                                              \
    HPDDM_F77(B ## geqp3)(m, n, a, lda, jpvt, tau, work, lwork, info);                                       \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::geqp3(const int* m, const int* n, T* a, const int* lda, int* jpvt, T* tau, T* work,   \
                             const int* lwork, U* rwork, int* info) {                                        \
    HPDDM_F77(C ## geqp3)(m, n, a, lda, jpvt, tau, work, lwork, rwork, info);                                \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<U>::mqr(const char* side, const char* trans, const int* m, const int* n, const int* k,    \
                           const U* a, const int* lda, const U* tau, U* c, const int* ldc, U* work,          \
                           const int* lwork, int* info) {                                                    \
    HPDDM_F77(B ## ormqr)(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);                     \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::mqr(const char* side, const char* trans, const int* m, const int* n, const int* k,    \
                           const T* a, const int* lda, const T* tau, T* c, const int* ldc, T* work,          \
                           const int* lwork, int* info) {                                                    \
    HPDDM_F77(C ## unmqr)(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);                     \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<U>::mhr(const char* side, const char* trans, const int* m, const int* n, const int* ilo,  \
                           const int* ihi, const U* a,  const int* lda, const U* tau, U* c, const int* ldc,  \
                           U* work, const int* lwork, int* info) {                                           \
    HPDDM_F77(B ## ormhr)(side, trans, m, n, ilo, ihi, a, lda, tau, c, ldc, work, lwork, info);              \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::mhr(const char* side, const char* trans, const int* m, const int* n, const int* ilo,  \
                           const int* ihi, const T* a,  const int* lda, const T* tau, T* c, const int* ldc,  \
                           T* work, const int* lwork, int* info) {                                           \
    HPDDM_F77(C ## unmhr)(side, trans, m, n, ilo, ihi, a, lda, tau, c, ldc, work, lwork, info);              \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<U>::hseqr(const char* job, const char* compz, const int* n, const int* ilo,               \
                             const int* ihi, U* h, const int* ldh, U* wr, U* wi, U* z, const int* ldz,       \
                             U* work, const int* lwork, int* info) {                                         \
    HPDDM_F77(B ## hseqr)(job, compz, n, ilo, ihi, h, ldh, wr, wi, z, ldz, work, lwork, info);               \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::hseqr(const char* job, const char* compz, const int* n, const int* ilo,               \
                             const int* ihi, T* h, const int* ldh, T* w, T*, T* z, const int* ldz,           \
                             T* work, const int* lwork, int* info) {                                         \
    HPDDM_F77(C ## hseqr)(job, compz, n, ilo, ihi, h, ldh, w, z, ldz, work, lwork, info);                    \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<U>::hsein(const char* side, const char* eigsrc, const char* initv, int* select,           \
                             const int* n, U* h, const int* ldh, U* wr, const U* wi, U* vl, const int* ldvl, \
                             U* vr, const int* ldvr, const int* mm, int* m, U* work, U*, int* ifaill,        \
                             int* ifailr, int* info) {                                                       \
    HPDDM_F77(B ## hsein)(side, eigsrc, initv, select, n, h, ldh, wr, wi, vl, ldvl, vr, ldvr, mm, m, work,   \
                          ifaill, ifailr, info);                                                             \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::hsein(const char* side, const char* eigsrc, const char* initv, int* select,           \
                             const int* n, T* h, const int* ldh, T* w, const T*, T* vl, const int* ldvl,     \
                             T* vr, const int* ldvr, const int* mm, int* m, T* work, U* rwork, int* ifaill,  \
                             int* ifailr, int* info) {                                                       \
    HPDDM_F77(C ## hsein)(side, eigsrc, initv, select, n, h, ldh, w, vl, ldvl, vr, ldvr, mm, m, work, rwork, \
                          ifaill, ifailr, info);                                                             \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<U>::geev(const char* jobvl, const char* jobvr, const int* n, U* a, const int* lda,        \
                            U* wr, U* wi, U* vl, const int* ldvl, U* vr, const int* ldvr, U* work,           \
                            const int* lwork, U*, int* info) {                                               \
    HPDDM_F77(B ## geev)(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);            \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::geev(const char* jobvl, const char* jobvr, const int* n, T* a, const int* lda, T* w,  \
                            T*, T* vl, const int* ldvl, T* vr, const int* ldvr, T* work, const int* lwork,   \
                            U* rwork, int* info) {                                                           \
    HPDDM_F77(C ## geev)(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);          \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<U>::ggev(const char* jobvl, const char* jobvr, const int* n, U* a, const int* lda, U* b,  \
                            const int* ldb, U* alphar, U* alphai, U* beta, U* vl, const int* ldvl, U* vr,    \
                            const int* ldvr, U* work, const int* lwork, U*, int* info) {                     \
    HPDDM_F77(B ## ggev)(jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work,    \
                         lwork, info);                                                                       \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::ggev(const char* jobvl, const char* jobvr, const int* n, T* a, const int* lda, T* b,  \
                            const int* ldb, T* alpha, T*, T* beta, T* vl, const int* ldvl, T* vr,            \
                            const int* ldvr, T* work, const int* lwork, U* rwork, int* info) {               \
    HPDDM_F77(C ## ggev)(jobvl, jobvr, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, lwork,      \
                         rwork, info);                                                                       \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<U>::gesvd(const char* jobu, const char* jobvt, const int* m, const int* n, U* a,          \
                             const int* lda, U* s, U* u, const int* ldu, U* vt, const int* ldvt, U* work,    \
                             const int* lwork, U*, int* info) {                                              \
    HPDDM_F77(B ## gesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);                \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::gesvd(const char* jobu, const char* jobvt, const int* m, const int* n, T* a,          \
                             const int* lda, U* s, T* u, const int* ldu, T* vt, const int* ldvt, T* work,    \
                             const int* lwork, U* rwork, int* info) {                                        \
    HPDDM_F77(C ## gesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);         \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<U>::gesdd(const char* jobz, const int* m, const int* n, U* a, const int* lda, U* s,       \
                             U* u, const int* ldu, U* vt, const int* ldvt, U* work, const int* lwork,        \
                             U*, int* iwork, int* info) {                                                    \
    HPDDM_F77(B ## gesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);                \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::gesdd(const char* jobz, const int* m, const int* n, T* a, const int* lda, U* s,       \
                             T* u, const int* ldu, T* vt, const int* ldvt, T* work, const int* lwork,        \
                             U* rwork, int* iwork, int* info) {                                              \
    HPDDM_F77(C ## gesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);         \
}
HPDDM_GENERATE_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HPDDM_GENERATE_LAPACK_COMPLEX(z, std::complex<double>, d, double)
# if defined(PETSCHPDDM_H)
#  if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_HAVE_F2CBLASLAPACK___FLOAT128_BINDINGS)
HPDDM_GENERATE_LAPACK_COMPLEX(w, __complex128, q, __float128)
#  endif
#  if defined(PETSC_USE_REAL___FP16) || defined(PETSC_HAVE_F2CBLASLAPACK___FP16_BINDINGS)
HPDDM_GENERATE_LAPACK_COMPLEX(k, std::complex<__fp16>, h, __fp16)
#  endif
# endif
} // HPDDM
#endif // __cplusplus
#endif // HPDDM_LAPACK_HPP_
