/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <jolivet@ann.jussieu.fr>
        Date: 2014-03-16

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

#ifndef _LAPACK_
#define _LAPACK_

#define HPDDM_GENERATE_EXTERN_LAPACK(C, T, U, SYM, ORT)                                                      \
void HPDDM_F77(C ## potrf)(const char*, const int*, T*, const int*, int*);                                   \
void HPDDM_F77(C ## trtrs)(const char*, const char*, const char*, const int*, const int*, const T*,          \
                           const int*, T*, const int*, int*);                                                \
void HPDDM_F77(C ## SYM ## gst)(const int*, const char*, const int*, T*, const int*,                         \
                                const T*, const int*, int*);                                                 \
void HPDDM_F77(C ## SYM ## trd)(const char*, const int*, T*, const int*, U*, U*, T*, T*, const int*, int*);  \
void HPDDM_F77(C ## stein)(const int*, const U*, const U*, const int*, const U*, const int*,                 \
                           const int*, T*, const int*, U*, int*, int*, int*);                                \
void HPDDM_F77(C ## ORT ## mtr)(const char*, const char*, const char*, const int*, const int*,               \
                                const T*, const int*, const T*, T*, const int*, T*, const int*, int*);

#if !defined(INTEL_MKL_VERSION)
extern "C" {
HPDDM_GENERATE_EXTERN_LAPACK(s, float, float, sy, or)
HPDDM_GENERATE_EXTERN_LAPACK(d, double, double, sy, or)
HPDDM_GENERATE_EXTERN_LAPACK(c, std::complex<float>, float, he, un)
HPDDM_GENERATE_EXTERN_LAPACK(z, std::complex<double>, double, he, un)
void HPDDM_F77(sstebz)(const char*, const char*, const int*, const float*, const float*, const int*, const int*, const float*, const float*, const float*, int*, int*, float*, int*, int*, float*, int*, int*);
void HPDDM_F77(dstebz)(const char*, const char*, const int*, const double*, const double*, const int*, const int*, const double*, const double*, const double*, int*, int*, double*, int*, int*, double*, int*, int*);
}
#endif // INTEL_MKL_VERSION

namespace HPDDM {
/* Class: Lapack
 *
 *  A class inheriting from <Eigensolver> to use <Lapack>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Lapack : public Eigensolver<K> {
    private:
        /* Function: potrf
         *  Computes the Cholesky factorization of a symmetric or Hermitian positive definite matrix. */
        static inline void potrf(const char*, const int*, K*, const int*, int*);
        /* Function: trtrs
         *  Solves a system of linear equations with a triangular matrix. */
        static inline void trtrs(const char*, const char*, const char*, const int*, const int*, const K*, const int*, K*, const int*, int*);
        /* Function: gst
         *  Reduces a symmetric or Hermitian definite generalized eigenvalue problem to a standard form. */
        static inline void gst(const int*, const char*, const int*, K*, const int*, K*, const int*, int*);
        /* Function: trd
         *  Reduces a symmetric or Hermitian matrix to a tridiagonal form. */
        static inline void trd(const char*, const int*, K*, const int*, typename Wrapper<K>::ul_type*, typename Wrapper<K>::ul_type*, K*, K*, const int*, int*);
        /* Function: stebz
         *  Computes selected eigenvalues of a symmetric tridiagonal matrix by bisection. */
        static inline void stebz(const char*, const char*, const int*, const typename Wrapper<K>::ul_type*, const typename Wrapper<K>::ul_type*, const int*, const int*, const typename Wrapper<K>::ul_type*, const typename Wrapper<K>::ul_type*, const typename Wrapper<K>::ul_type*, int*, int*, typename Wrapper<K>::ul_type*, int*, int*, typename Wrapper<K>::ul_type*, int*, int*);
        /* Function: stein
         *  Computes the eigenvectors corresponding to specified eigenvalues of a symmetric tridiagonal matrix. */
        static inline void stein(const int*, const typename Wrapper<K>::ul_type*, const typename Wrapper<K>::ul_type*, const int*, const typename Wrapper<K>::ul_type*, const int*, const int*, K*, const int*, typename Wrapper<K>::ul_type*, int*, int*, int*);
        /* Function: mtr
         *  Multiplies a matrix by a orthogonal or unitary matrix obtained with <Lapack::trd>. */
        static inline void mtr(const char*, const char*, const char*, const int*, const int*, const K*, const int*, const K*, K*, const int*, K*, const int*, int*);
    public:
        Lapack(int n, int nu)                                                                           : Eigensolver<K>(n, nu) { }
        Lapack(typename Wrapper<K>::ul_type threshold, int n, int nu)                                   : Eigensolver<K>(threshold, n, nu) { }
        Lapack(typename Wrapper<K>::ul_type tol, typename Wrapper<K>::ul_type threshold, int n, int nu) : Eigensolver<K>(tol, threshold, n, nu) { }
        /* Function: reduce
         *
         *  Reduces a symmetric or Hermitian definite generalized eigenvalue problem to a standard problem after factorizing the right-hand side matrix.
         *
         * Parameters:
         *    A              - Left-hand side matrix.
         *    B              - Right-hand side matrix. */
        inline void reduce(K* const& A, K* const& B) const {
            int info;
            potrf(&uplo, &(Eigensolver<K>::_n), B, &(Eigensolver<K>::_n), &info);
            gst(&i__1, &uplo, &(Eigensolver<K>::_n), A, &(Eigensolver<K>::_n), B, &(Eigensolver<K>::_n), &info);
        }
        /* Function: expand
         *
         *  Computes the eigenvectors of a generalized eigenvalue problem after completion of <Lapack::solve>.
         *
         * Parameters:
         *    B              - Right-hand side matrix.
         *    ev             - Array of eigenvectors. */
        inline void expand(K* const& B, K* const* const ev) {
            int info;
            trtrs(&uplo, &transb, &transa, &(Eigensolver<K>::_n), &(Eigensolver<K>::_nu), B, &(Eigensolver<K>::_n), *ev, &(Eigensolver<K>::_n), &info);
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
        inline void solve(K* const& A, K**& ev, K* const& work, int& lwork, const MPI_Comm& communicator) {
            int info;
            K* tau = work + lwork;
            typename Wrapper<K>::ul_type* d = reinterpret_cast<typename Wrapper<K>::ul_type*>(tau) + Eigensolver<K>::_n;
            typename Wrapper<K>::ul_type* e = d + Eigensolver<K>::_n;
            trd(&uplo, &(Eigensolver<K>::_n), A, &(Eigensolver<K>::_n), d, e, tau, work, &lwork, &info);
            char range = Eigensolver<K>::_threshold > 0.0 ? 'V' : 'I';
            char order = 'B';
            int il = 1;
            int iu = Eigensolver<K>::_nu;
            typename Wrapper<K>::ul_type vl = -2 * Eigensolver<K>::_tol;
            typename Wrapper<K>::ul_type vu = Eigensolver<K>::_threshold;
            int nsplit;
            typename Wrapper<K>::ul_type* evr = e + Eigensolver<K>::_n - 1;
            int* iblock = new int[5 * Eigensolver<K>::_n];
            int* isplit = iblock + Eigensolver<K>::_n;
            int* iwork = isplit + Eigensolver<K>::_n;
            stebz(&range, &order, &(Eigensolver<K>::_n), &vl, &vu, &il, &iu, &(Eigensolver<K>::_tol), d, e, &(Eigensolver<K>::_nu), &nsplit, evr, iblock, isplit, reinterpret_cast<typename Wrapper<K>::ul_type*>(work), iwork, &info);
            if(Eigensolver<K>::_threshold > 0.0)
                Eigensolver<K>::selectNu(evr, communicator);
            if(Eigensolver<K>::_nu) {
                ev = new K*[Eigensolver<K>::_nu];
                *ev = new K[Eigensolver<K>::_n * Eigensolver<K>::_nu];
                for(unsigned short i = 1; i < Eigensolver<K>::_nu; ++i)
                    ev[i] = *ev + i * Eigensolver<K>::_n;
                int* ifailv = new int[Eigensolver<K>::_nu];
                stein(&(Eigensolver<K>::_n), d, e, &(Eigensolver<K>::_nu), evr, iblock, isplit, *ev, &(Eigensolver<K>::_n), reinterpret_cast<typename Wrapper<K>::ul_type*>(work), iwork, ifailv, &info);
                delete [] ifailv;
                order = 'L';
                mtr(&order, &uplo, &transa, &(Eigensolver<K>::_n), &(Eigensolver<K>::_nu), A, &(Eigensolver<K>::_n), tau, *ev, &(Eigensolver<K>::_n), work, &lwork, &info);
                if(std::is_same<K, typename Wrapper<K>::ul_type>::value)
                    lwork += 3 * Eigensolver<K>::_n - 1;
                else
                    lwork += 4 * Eigensolver<K>::_n - 1;
            }
            delete [] iblock;
        }
};

#define HPDDM_GENERATE_LAPACK(C, T, B, U, SYM, ORT)                                                          \
template<>                                                                                                   \
inline void Lapack<T>::potrf(const char* uplo, const int* n, T* a, const int* lda, int* info) {              \
    HPDDM_F77(C ## potrf)(uplo, n, a, lda, info);                                                            \
}                                                                                                            \
template<>                                                                                                   \
inline void Lapack<T>::trtrs(const char* uplo, const char* trans, const char* diag, const int* n,            \
                             const int* nrhs, const T* a, const int* lda, T* b, const int* ldb, int* info) { \
    HPDDM_F77(C ## trtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);                                 \
}                                                                                                            \
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
}
HPDDM_GENERATE_LAPACK(s, float, s, float, sy, or)
HPDDM_GENERATE_LAPACK(d, double, d, double, sy, or)
HPDDM_GENERATE_LAPACK(c, std::complex<float>, s, float, he, un)
HPDDM_GENERATE_LAPACK(z, std::complex<double>, d, double, he, un)
} // HPDDM
#endif // _LAPACK_
