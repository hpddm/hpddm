/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@inf.ethz.ch>
        Date: 2014-03-16

   Copyright (C) 2011-2014 Université de Grenoble
                 2015      Eidgenössische Technische Hochschule Zürich

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

#ifndef _LAPACK_
#define _LAPACK_

#define HPDDM_GENERATE_EXTERN_LAPACK(C, T, U, SYM, ORT)                                                      \
void HPDDM_F77(C ## SYM ## gst)(const int*, const char*, const int*, T*, const int*,                         \
                                const T*, const int*, int*);                                                 \
void HPDDM_F77(C ## SYM ## trd)(const char*, const int*, T*, const int*, U*, U*, T*, T*, const int*, int*);  \
void HPDDM_F77(C ## stein)(const int*, const U*, const U*, const int*, const U*, const int*,                 \
                           const int*, T*, const int*, U*, int*, int*, int*);                                \
void HPDDM_F77(C ## ORT ## mtr)(const char*, const char*, const char*, const int*, const int*,               \
                                const T*, const int*, const T*, T*, const int*, T*, const int*, int*);       \
void HPDDM_F77(C ## geqrf)(const int*, const int*, T*, const int*, T*, T*, const int*, int*);                \
void HPDDM_F77(C ## geqrt)(const int*, const int*, const int*, T*, const int*, T*, const int*, T*, int*);    \
void HPDDM_F77(C ## gemqrt)(const char*, const char*, const int*, const int*, const int*, const int*,        \
                            const T*, const int*, const T*, const int*, T*, const int*, T*, int*);           \
void HPDDM_F77(C ## lapmt)(const int*, const int*, const int*, T*, const int*, int*);                        \
void HPDDM_F77(C ## trtrs)(const char*, const char*, const char*, const int*, const int*, const T*,          \
                           const int*, T*, const int*, int*);                                                \
void HPDDM_F77(C ## potrf)(const char*, const int*, T*, const int*, int*);                                   \
void HPDDM_F77(C ## potrs)(const char*, const int*, const int*, const T*, const int*, T*, const int*, int*);
#define HPDDM_GENERATE_EXTERN_LAPACK_COMPLEX(C, T, B, U)                                                     \
void HPDDM_F77(B ## stebz)(const char*, const char*, const int*, const U*, const U*, const int*, const int*, \
                           const U*, const U*, const U*, int*, int*, U*, int*, int*, U*, int*, int*);        \
void HPDDM_F77(B ## geqp3)(const int*, const int*, U*, const int*, const int*, U*, U*, const int*, int*);    \
void HPDDM_F77(C ## geqp3)(const int*, const int*, T*, const int*, const int*, T*, T*, const int*, U*, int*);\
void HPDDM_F77(B ## ormqr)(const char*, const char*, const int*, const int*, const int*, const U*,           \
                           const int*, const U*, U*, const int*, U*, const int*, int*);                      \
void HPDDM_F77(C ## unmqr)(const char*, const char*, const int*, const int*, const int*, const T*,           \
                           const int*, const T*, T*, const int*, T*, const int*, int*);

#if !defined(INTEL_MKL_VERSION)
extern "C" {
HPDDM_GENERATE_EXTERN_LAPACK(s, float, float, sy, or)
HPDDM_GENERATE_EXTERN_LAPACK(d, double, double, sy, or)
HPDDM_GENERATE_EXTERN_LAPACK(c, std::complex<float>, float, he, un)
HPDDM_GENERATE_EXTERN_LAPACK(z, std::complex<double>, double, he, un)
HPDDM_GENERATE_EXTERN_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HPDDM_GENERATE_EXTERN_LAPACK_COMPLEX(z, std::complex<double>, d, double)
}
#endif // INTEL_MKL_VERSION

namespace HPDDM {
template<class K> class QR;
/* Class: Lapack
 *
 *  A class inheriting from <Eigensolver> to use <Lapack>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Lapack : public Eigensolver<K> {
    private:
        friend class QR<K>;
        /* Function: gst
         *  Reduces a symmetric or Hermitian definite generalized eigenvalue problem to a standard form. */
        static void gst(const int*, const char*, const int*, K*, const int*, K*, const int*, int*);
        /* Function: trd
         *  Reduces a symmetric or Hermitian matrix to a tridiagonal form. */
        static void trd(const char*, const int*, K*, const int*, typename Wrapper<K>::ul_type*, typename Wrapper<K>::ul_type*, K*, K*, const int*, int*);
        /* Function: stebz
         *  Computes selected eigenvalues of a symmetric tridiagonal matrix by bisection. */
        static void stebz(const char*, const char*, const int*, const typename Wrapper<K>::ul_type*, const typename Wrapper<K>::ul_type*, const int*, const int*, const typename Wrapper<K>::ul_type*, const typename Wrapper<K>::ul_type*, const typename Wrapper<K>::ul_type*, int*, int*, typename Wrapper<K>::ul_type*, int*, int*, typename Wrapper<K>::ul_type*, int*, int*);
        /* Function: stein
         *  Computes the eigenvectors corresponding to specified eigenvalues of a symmetric tridiagonal matrix. */
        static void stein(const int*, const typename Wrapper<K>::ul_type*, const typename Wrapper<K>::ul_type*, const int*, const typename Wrapper<K>::ul_type*, const int*, const int*, K*, const int*, typename Wrapper<K>::ul_type*, int*, int*, int*);
        /* Function: mtr
         *  Multiplies a matrix by an orthogonal or unitary matrix obtained with <Lapack::trd>. */
        static void mtr(const char*, const char*, const char*, const int*, const int*, const K*, const int*, const K*, K*, const int*, K*, const int*, int*);
    public:
        Lapack(int n)                                                                                   : Eigensolver<K>(n) { }
        Lapack(int n, int nu)                                                                           : Eigensolver<K>(n, nu) { }
        Lapack(typename Wrapper<K>::ul_type threshold, int n, int nu)                                   : Eigensolver<K>(threshold, n, nu) { }
        Lapack(typename Wrapper<K>::ul_type tol, typename Wrapper<K>::ul_type threshold, int n, int nu) : Eigensolver<K>(tol, threshold, n, nu) { }
        /* Function: lapmt
         *  Performs a forward or backward permutation of the columns of a matrix. */
        static void lapmt(const int*, const int*, const int*, K*, const int*, int*);
        /* Function: trtrs
         *  Solves a system of linear equations with a triangular matrix. */
        static void trtrs(const char*, const char*, const char*, const int*, const int*, const K*, const int*, K*, const int*, int*);
        /* Function: potrf
         *  Computes the Cholesky factorization of a symmetric or Hermitian positive definite matrix. */
        static void potrf(const char*, const int*, K*, const int*, int*);
        /* Function: potrs
         *  Solves a system of linear equations with a Cholesky-factored matrix. */
        static void potrs(const char*, const int*, const int*, const K*, const int*, K*, const int*, int*);
        /* Function: workspace
         *  Returns the optimal size of the workspace array. */
        int workspace() const {
            int info;
            int lwork = -1;
            K wkopt;
            trd("L", &(Eigensolver<K>::_n), nullptr, &(Eigensolver<K>::_n), nullptr, nullptr, nullptr, &wkopt, &lwork, &info);
            return static_cast<int>(std::real(wkopt));
        }
        /* Function: reduce
         *
         *  Reduces a symmetric or Hermitian definite generalized eigenvalue problem to a standard problem after factorizing the right-hand side matrix.
         *
         * Parameters:
         *    A              - Left-hand side matrix.
         *    B              - Right-hand side matrix. */
        void reduce(K* const& A, K* const& B) const {
            int info;
            potrf("L", &(Eigensolver<K>::_n), B, &(Eigensolver<K>::_n), &info);
            gst(&i__1, "L", &(Eigensolver<K>::_n), A, &(Eigensolver<K>::_n), B, &(Eigensolver<K>::_n), &info);
        }
        /* Function: expand
         *
         *  Computes the eigenvectors of a generalized eigenvalue problem after completion of <Lapack::solve>.
         *
         * Parameters:
         *    B              - Right-hand side matrix.
         *    ev             - Array of eigenvectors. */
        void expand(K* const& B, K* const* const ev) const {
            int info;
            trtrs("L", "T", "N", &(Eigensolver<K>::_n), &(Eigensolver<K>::_nu), B, &(Eigensolver<K>::_n), *ev, &(Eigensolver<K>::_n), &info);
        }
        /* Function: solve
         *
         *  Computes eigenvectors of the standard eigenvalue problem Ax = l x.
         *
         * Parameters:
         *    A              - Left-hand side matrix.
         *    ev             - Array of eigenvectors.
         *    work           - Workspace array.
         *    lwork          - Size of the input workspace array.
         *    communicator   - MPI communicator for selecting the threshold criterion. */
        void solve(K* const& A, K**& ev, K* const& work, int& lwork, const MPI_Comm& communicator) {
            int info;
            K* tau = work + lwork;
            typename Wrapper<K>::ul_type* d = reinterpret_cast<typename Wrapper<K>::ul_type*>(tau + Eigensolver<K>::_n);
            typename Wrapper<K>::ul_type* e = d + Eigensolver<K>::_n;
            trd("L", &(Eigensolver<K>::_n), A, &(Eigensolver<K>::_n), d, e, tau, work, &lwork, &info);
            typename Wrapper<K>::ul_type vl = -1.0 / HPDDM_EPS;
            typename Wrapper<K>::ul_type vu = Eigensolver<K>::_threshold;
            int iu = Eigensolver<K>::_nu;
            int nsplit;
            typename Wrapper<K>::ul_type* evr = e + Eigensolver<K>::_n - 1;
            int* iblock = new int[5 * Eigensolver<K>::_n];
            int* isplit = iblock + Eigensolver<K>::_n;
            int* iwork = isplit + Eigensolver<K>::_n;
            char range = Eigensolver<K>::_threshold > 0.0 ? 'V' : 'I';
            stebz(&range, "B", &(Eigensolver<K>::_n), &vl, &vu, &i__1, &iu, &(Eigensolver<K>::_tol), d, e, &(Eigensolver<K>::_nu), &nsplit, evr, iblock, isplit, reinterpret_cast<typename Wrapper<K>::ul_type*>(work), iwork, &info);
            if(Eigensolver<K>::_nu) {
                ev = new K*[Eigensolver<K>::_nu];
                *ev = new K[Eigensolver<K>::_n * Eigensolver<K>::_nu];
                for(unsigned short i = 1; i < Eigensolver<K>::_nu; ++i)
                    ev[i] = *ev + i * Eigensolver<K>::_n;
                int* ifailv = new int[Eigensolver<K>::_nu];
                stein(&(Eigensolver<K>::_n), d, e, &(Eigensolver<K>::_nu), evr, iblock, isplit, *ev, &(Eigensolver<K>::_n), reinterpret_cast<typename Wrapper<K>::ul_type*>(work), iwork, ifailv, &info);
                delete [] ifailv;
                mtr("L", "L", &transa, &(Eigensolver<K>::_n), &(Eigensolver<K>::_nu), A, &(Eigensolver<K>::_n), tau, *ev, &(Eigensolver<K>::_n), work, &lwork, &info);
                if(std::is_same<K, typename Wrapper<K>::ul_type>::value)
                    lwork += 3 * Eigensolver<K>::_n - 1;
                else
                    lwork += 4 * Eigensolver<K>::_n - 1;
            }
            delete [] iblock;
        }
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
#if HPDDM_QR == 1
        std::vector<int>               _jpvt;
        int                            _rank;
#endif
        /* Function: geqp3
         *  Computes a QR decomposition of a rectangular matrix with column pivoting. */
        static void geqp3(const int*, const int*, K*, const int*, int*, K*, K*, const int*, typename Wrapper<K>::ul_type*, int*);
        /* Function: workspace
         *  Returns the optimal size of the workspace array. */
        int workspace() const {
            int info;
            int lwork[2] { -1, -1 };
            K wkopt;
#if HPDDM_QR == 1
            geqp3(&_n, &_n, nullptr, &_n, nullptr, nullptr, &wkopt, lwork, nullptr, &info);
#else
            geqrf(&_n, &_n, nullptr, &_n, nullptr, &wkopt, lwork, &info);
#endif
            lwork[0] = static_cast<int>(std::real(wkopt));
            mqr("L", "T", &_n, &i__1, &_n, nullptr, &_n, nullptr, nullptr, &_n, &wkopt, lwork + 1, &info);
            lwork[1] = static_cast<int>(std::real(wkopt));
            return *std::max_element(lwork, lwork + 1);
        }
    public:
        QR(int n, const K* const cpy = nullptr) : _n(n), _lwork(workspace()), _a(new K[_n * (_n + 1) + _lwork]), _tau(_a + _n * _n), _work(_tau + _n) {
#if HPDDM_QR == 1
            _jpvt.resize(_n);
            _rank = n;
#endif
            if(cpy)
                for(unsigned int i = 0; i < _n; ++i) {
                    _a[i * (_n + 1)] = cpy[i * (_n + 1)];
                    for(unsigned int j = i + 1; j < _n; ++j)
                        _a[j * _n + i] = _a[i * _n + j] = cpy[i * _n + j];
                }
        }
        ~QR() {
            delete [] _a;
        }
        /* Function: geqrf
         *  Computes a QR decomposition of a rectangular matrix. */
        static void geqrf(const int*, const int*, K*, const int*, K*, K*, const int*, int*);
        /* Function: geqrt
         *  Computes a blocked QR decomposition of a rectangular matrix using the compact WY representation of Q. */
        static void geqrt(const int*, const int*, const int*, K*, const int*, K*, const int*, K*, int*);
        /* Function: mqr
         *  Multiplies a matrix by an orthogonal or unitary matrix obtained with <QR::geq>. */
        static void mqr(const char*, const char*, const int*, const int*, const int*, const K*, const int*, const K*, K*, const int*, K*, const int*, int*);
        /* Function: gemqrt
         *  Multiplies a matrix by an orthogonal or unitary matrix obtained with <QR::geqrt>. */
        static void gemqrt(const char*, const char*, const int*, const int*, const int*, const int*, const K*, const int*, const K*, const int*, K*, const int*, K*, int*);
        /* Function: getPointer
         *  Returns the pointer <QR::a>. */
        K* getPointer() const { return _a; }
        void decompose() {
            int info;
#if HPDDM_QR == 1
            typename Wrapper<K>::ul_type* rwork = (std::is_same<K, typename Wrapper<K>::ul_type>::value ? nullptr : new typename Wrapper<K>::ul_type[2 * _n]);
            geqp3(&_n, &_n, _a, &_n, _jpvt.data(), _tau, _work, &_lwork, rwork, &info);
            delete [] rwork;
            while(std::abs(_a[(_rank - 1) * (_n + 1)]) < HPDDM_EPS * std::abs(_a[0]) && _rank-- > 0);
#else
            geqrf(&_n, &_n, _a, &_n, _tau, _work, &_lwork, &info);
            std::vector<int> jpvt;
            jpvt.reserve(6);
            typename Wrapper<K>::ul_type max = std::abs(_a[0]);
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
#endif
        }
        /* Function: solve
         *  Computes the solution of a least squares problem. */
        void solve(K* const x) const {
            int info;
            mqr("L", "T", &_n, &i__1, &_n, _a, &_n, _tau, x, &_n, _work, &_lwork, &info);
#if HPDDM_QR == 1
            Lapack<K>::trtrs("U", "N", "N", &_rank, &i__1, _a, &_n, x, &_n, &info);
            Lapack<K>::lapmt(&i__0, &i__1, &_n, x, &i__1, const_cast<int*>(_jpvt.data()));
#else
            Lapack<K>::trtrs("U", "N", "N", &_n, &i__1, _a, &_n, x, &_n, &info);
#endif
        }
};

#define HPDDM_GENERATE_LAPACK(C, T, B, U, SYM, ORT)                                                          \
template<>                                                                                                   \
inline void Lapack<T>::gst(const int* itype, const char* uplo, const int* n,                                 \
                           T* a, const int* lda, T* b, const int* ldb, int* info) {                          \
    HPDDM_F77(C ## SYM ## gst)(itype, uplo, n, a, lda, b, ldb, info);                                        \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::stein(const int* n, const U* d, const U* e, const int* m, const U* w,                 \
                             const int* iblock, const int* isplit, T* z, const int* ldz,                     \
                             U* work, int* iwork, int* ifailv, int* info) {                                  \
    HPDDM_F77(C ## stein)(n, d, e, m, w, iblock, isplit, z, ldz, work, iwork, ifailv, info);                 \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::trd(const char* uplo, const int* n, T* a, const int* lda,                             \
                           U* d, U* e, T* tau, T* work, const int* lwork, int* info) {                       \
    HPDDM_F77(C ## SYM ## trd)(uplo, n, a, lda, d, e, tau, work, lwork, info);                               \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::mtr(const char* side, const char* uplo, const char* trans, const int* m,              \
                           const int* n, const T* a, const int* lda, const T* tau, T* c, const int* ldc,     \
                           T* work, const int* lwork, int* info) {                                           \
    HPDDM_F77(C ## ORT ## mtr)(side, uplo, trans, m, n, a, lda, tau, c, ldc, work, lwork, info);             \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::stebz(const char* range, const char* order, const int* n, const U* vl, const U* vu,   \
                             const int* il, const int* iu, const U* abstol, const U* d, const U* e, int* m,  \
                             int* nsplit, U* w, int* iblock, int* isplit, U* work, int* iwork, int* info) {  \
    HPDDM_F77(B ## stebz)(range, order, n, vl, vu, il, iu, abstol, d, e, m, nsplit, w, iblock, isplit,       \
                          work, iwork, info);                                                                \
}                                                                                                            \
template<>                                                                                                   \
inline void QR<T>::geqrf(const int* m, const int* n, T* a, const int* lda, T* tau, T* work,                  \
                         const int* lwork, int* info) {                                                      \
    HPDDM_F77(C ## geqrf)(m, n, a, lda, tau, work, lwork, info);                                             \
}                                                                                                            \
template<>                                                                                                   \
inline void QR<T>::geqrt(const int* m, const int* n, const int* nb, T* a, const int* lda, T* t,              \
                         const int* ldt, T* work, int* info) {                                               \
    HPDDM_F77(C ## geqrt)(m, n, nb, a, lda, t, ldt, work, info);                                             \
}                                                                                                            \
template<>                                                                                                   \
inline void QR<T>::gemqrt(const char* side, const char* trans, const int* m, const int* n, const int* k,     \
                          const int* nb, const T* v, const int* ldv, const T* t, const int* ldt, T* c,       \
                          const int* ldc, T* work, int* info) {                                              \
    HPDDM_F77(C ## gemqrt)(side, trans, m, n, k, nb, v, ldv, t, ldt, c, ldc, work, info);                    \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::lapmt(const int* forwrd, const int* m, const int* n, T* x, const int* ldx,            \
                             int* k) {                                                                       \
    HPDDM_F77(C ## lapmt)(forwrd, m, n, x, ldx, k);                                                          \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::trtrs(const char* uplo, const char* trans, const char* diag, const int* n,            \
                             const int* nrhs, const T* a, const int* lda, T* b, const int* ldb, int* info) { \
    HPDDM_F77(C ## trtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);                                 \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::potrf(const char* uplo, const int* n, T* a, const int* lda, int* info) {              \
    HPDDM_F77(C ## potrf)(uplo, n, a, lda, info);                                                            \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::potrs(const char* uplo, const int* n, const int* nrhs, const T* a, const int* lda,    \
                             T* b, const int* ldb, int* info) {                                              \
    HPDDM_F77(C ## potrs)(uplo, n, nrhs, a, lda, b, ldb, info);                                              \
}
#define HPDDM_GENERATE_LAPACK_COMPLEX(C, T, B, U)                                                            \
template<>                                                                                                   \
inline void QR<U>::geqp3(const int* m, const int* n, U* a, const int* lda, int* jpvt, U* tau, U* work,       \
                         const int* lwork, U* rwork, int* info) {                                            \
    HPDDM_F77(B ## geqp3)(m, n, a, lda, jpvt, tau, work, lwork, info);                                       \
}                                                                                                            \
template<>                                                                                                   \
inline void QR<T>::geqp3(const int* m, const int* n, T* a, const int* lda, int* jpvt, T* tau, T* work,       \
                         const int* lwork, U* rwork, int* info) {                                            \
    HPDDM_F77(C ## geqp3)(m, n, a, lda, jpvt, tau, work, lwork, rwork, info);                                \
}                                                                                                            \
template<>                                                                                                   \
inline void QR<U>::mqr(const char* side, const char* trans, const int* m, const int* n, const int* k,        \
                       const U* a, const int* lda, const U* tau, U* c, const int* ldc, U* work,              \
                       const int* lwork, int* info) {                                                        \
    HPDDM_F77(B ## ormqr)(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);                     \
}                                                                                                            \
template<>                                                                                                   \
inline void QR<T>::mqr(const char* side, const char* trans, const int* m, const int* n, const int* k,        \
                       const T* a, const int* lda, const T* tau, T* c, const int* ldc, T* work,              \
                       const int* lwork, int* info) {                                                        \
    HPDDM_F77(C ## unmqr)(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);                     \
}
HPDDM_GENERATE_LAPACK(s, float, s, float, sy, or)
HPDDM_GENERATE_LAPACK(d, double, d, double, sy, or)
HPDDM_GENERATE_LAPACK(c, std::complex<float>, s, float, he, un)
HPDDM_GENERATE_LAPACK(z, std::complex<double>, d, double, he, un)
HPDDM_GENERATE_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HPDDM_GENERATE_LAPACK_COMPLEX(z, std::complex<double>, d, double)
} // HPDDM
#endif // _LAPACK_
