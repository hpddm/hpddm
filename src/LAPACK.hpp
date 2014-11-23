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

extern "C" {
void HPDDM_F77(dpotrf)(const char*, const int*, double*, const int*, int*);
void HPDDM_F77(zpotrf)(const char*, const int*, std::complex<double>*, const int*, int*);
void HPDDM_F77(dtrtrs)(const char*, const char*, const char*, const int*, const int*, const double*, const int*, double*, const int*, int*);
void HPDDM_F77(ztrtrs)(const char*, const char*, const char*, const int*, const int*, const std::complex<double>*, const int*, std::complex<double>*, const int*, int*);
void HPDDM_F77(dsygst)(const int*, const char*, const int*, double*, const int*, const double*, const int*, int*);
void HPDDM_F77(zhegst)(const int*, const char*, const int*, std::complex<double>*, const int*, const std::complex<double>*, const int*, int*);
void HPDDM_F77(dsytrd)(const char*, const int*, double*, const int*, double*, double*, double*, double*, const int*, int*);
void HPDDM_F77(zhetrd)(const char*, const int*, std::complex<double>*, const int*, double*, double*, std::complex<double>*, std::complex<double>*, const int*, int*);
void HPDDM_F77(dstebz)(const char*, const char*, const int*, const double*, const double*, const int*, const int*, const double*, const double*, const double*, int*, int*, double*, int*, int*, double*, int*, int*);
void HPDDM_F77(dstein)(const int*, const double*, const double*, const int*, const double*, const int*, const int*, double*, const int*, double*, int*, int*, int*);
void HPDDM_F77(zstein)(const int*, const double*, const double*, const int*, const double*, const int*, const int*, std::complex<double>*, const int*, double*, int*, int*, int*);
void HPDDM_F77(dormtr)(const char*, const char*, const char*, const int*, const int*, const double*, const int*, const double*, double*, const int*, double*, const int*, int*);
void HPDDM_F77(zunmtr)(const char*, const char*, const char*, const int*, const int*, const std::complex<double>*, const int*, const std::complex<double>*, std::complex<double>*, const int*, std::complex<double>*, const int*, int*);
}

namespace HPDDM {
/* Class: Lapack
 *
 *  A class inheriting from <Eigensolver> to use <Lapack>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Lapack : public Eigensolver {
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
        static inline void trd(const char*, const int*, K*, const int*, double*, double*, K*, K*, const int*, int*);
        /* Function: stebz
         *  Computes selected eigenvalues of a symmetric tridiagonal matrix by bisection. */
        static inline void stebz(const char*, const char*, const int*, const double*, const double*, const int*, const int*, const double*, const double*, const double*, int*, int*, double*, int*, int*, double*, int*, int*);
        /* Function: stein
         *  Computes the eigenvectors corresponding to specified eigenvalues of a symmetric tridiagonal matrix. */
        static inline void stein(const int*, const double*, const double*, const int*, const double*, const int*, const int*, K*, const int*, double*, int*, int*, int*);
        /* Function: mtr
         *  Multiplies a matrix by a orthogonal or unitary matrix obtained with <Lapack::trd>. */
        static inline void mtr(const char*, const char*, const char*, const int*, const int*, const K*, const int*, const K*, K*, const int*, K*, const int*, int*);
    public:
        Lapack(double tol, double threshold, int n, int nu) : Eigensolver(tol, threshold, n, nu) { }
        Lapack(double threshold, int n, int nu)             : Eigensolver(threshold, n, nu) { }
        Lapack(int n, int nu)                               : Eigensolver(n, nu) { }
        /* Function: reduce
         *
         *  Reduces a symmetric or Hermitian definite generalized eigenvalue problem to a standard problem after factorizing the right-hand side matrix.
         *
         * Parameters:
         *    A              - Left-hand side matrix.
         *    B              - Right-hand side matrix. */
        inline void reduce(K* const& A, K* const& B) const {
            int info;
            potrf(&uplo, &_n, B, &_n, &info);
            gst(&i__1, &uplo, &_n, A, &_n, B, &_n, &info);
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
            trtrs(&uplo, &transb, &transa, &_n, &_nu, B, &_n, *ev, &_n, &info);
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
            double* d = reinterpret_cast<double*>(tau) + _n;
            double* e = d + _n;
            trd(&uplo, &_n, A, &_n, d, e, tau, work, &lwork, &info);
            char range = _threshold > 0.0 ? 'V' : 'I';
            char order = 'B';
            int il = 1;
            int iu = _nu;
            double vl = -2 * _tol;
            double vu = _threshold;
            int nsplit;
            double* evr = e + _n - 1;
            int* iblock = new int[5 * _n];
            int* isplit = iblock + _n;
            int* iwork = isplit + _n;
            stebz(&range, &order, &_n, &vl, &vu, &il, &iu, &_tol, d, e, &_nu, &nsplit, evr, iblock, isplit, reinterpret_cast<double*>(work), iwork, &info);
            if(_nu) {
                ev = new K*[_nu];
                *ev = new K[_n * _nu];
                for(unsigned short i = 1; i < _nu; ++i)
                    ev[i] = *ev + i * _n;
                int* ifailv = new int[_nu];
                stein(&_n, d, e, &_nu, evr, iblock, isplit, *ev, &_n, reinterpret_cast<double*>(work), iwork, ifailv, &info);
                delete [] ifailv;
                order = 'L';
                mtr(&order, &uplo, &transa, &_n, &_nu, A, &_n, tau, *ev, &_n, work, &lwork, &info);
                if(std::is_same<K, double>::value)
                    lwork += 3 * _n - 1;
                else
                    lwork += 4 * _n - 1;
                if(_threshold > 0.0)
                    selectNu(evr, communicator);
            }
            delete [] iblock;
        }
};

template<>
inline void Lapack<double>::potrf(const char* uplo, const int* n, double* a, const int* lda, int* info) {
    HPDDM_F77(dpotrf)(uplo, n, a, lda, info);
}
template<>
inline void Lapack<std::complex<double>>::potrf(const char* uplo, const int* n, std::complex<double>* a, const int* lda, int* info) {
    HPDDM_F77(zpotrf)(uplo, n, a, lda, info);
}

template<>
inline void Lapack<double>::trtrs(const char* uplo, const char* trans, const char* diag, const int* n, const int* nrhs, const double* a, const int* lda, double* b, const int* ldb, int* info) {
    HPDDM_F77(dtrtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
}
template<>
inline void Lapack<std::complex<double>>::trtrs(const char* uplo, const char* trans, const char* diag, const int* n, const int* nrhs, const std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb, int* info) {
    HPDDM_F77(ztrtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
}

template<>
inline void Lapack<double>::gst(const int* itype, const char* uplo, const int* n, double* a, const int* lda, double* b, const int* ldb, int* info) {
    HPDDM_F77(dsygst)(itype, uplo, n, a, lda, b, ldb, info);
}
template<>
inline void Lapack<std::complex<double>>::gst(const int* itype, const char* uplo, const int* n, std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb, int* info) {
    HPDDM_F77(zhegst)(itype, uplo, n, a, lda, b, ldb, info);
}

template<>
inline void Lapack<double>::stein(const int* n, const double* d, const double* e, const int* m, const double* w, const int* iblock, const int* isplit, double* z, const int* ldz, double* work, int* iwork, int* ifail, int* info) {
    HPDDM_F77(dstein)(n, d, e, m, w, iblock, isplit, z, ldz, work, iwork, ifail, info);
}
template<>
inline void Lapack<std::complex<double>>::stein(const int* n, const double* d, const double* e, const int* m, const double* w, const int* iblock, const int* isplit, std::complex<double>* z, const int* ldz, double* work, int* iwork, int* ifailv, int* info) {
    HPDDM_F77(zstein)(n, d, e, m, w, iblock, isplit, z, ldz, work, iwork, ifailv, info);
}

template<>
inline void Lapack<double>::trd(const char* uplo, const int* n, double* a, const int* lda, double* d, double* e, double* tau, double* work, const int* lwork, int* info) {
    HPDDM_F77(dsytrd)(uplo, n, a, lda, d, e, tau, work, lwork, info);
}
template<>
inline void Lapack<std::complex<double>>::trd(const char* uplo, const int* n, std::complex<double>* a, const int* lda, double* d, double* e, std::complex<double>* tau, std::complex<double>* work, const int* lwork, int* info) {
    HPDDM_F77(zhetrd)(uplo, n, a, lda, d, e, tau, work, lwork, info);
}

template<class K>
inline void Lapack<K>::stebz(const char* range, const char* order, const int* n, const double* vl, const double* vu, const int* il, const int* iu, const double* abstol, const double* d, const double* e, int* m, int* nsplit, double* w, int* iblock, int* isplit, double* work, int* iwork, int* info) {
    HPDDM_F77(dstebz)(range, order, n, vl, vu, il, iu, abstol, d, e, m, nsplit, w, iblock, isplit, work, iwork, info);
}

template<>
inline void Lapack<double>::mtr(const char* side, const char* uplo, const char* trans, const int* m, const int* n, const double* a, const int* lda, const double* tau, double* c, const int* ldc, double* work, const int* lwork, int* info) {
    HPDDM_F77(dormtr)(side, uplo, trans, m, n, a, lda, tau, c, ldc, work, lwork, info);
}
template<>
inline void Lapack<std::complex<double>>::mtr(const char* side, const char* uplo, const char* trans, const int* m, const int* n, const std::complex<double>* a, const int* lda, const std::complex<double>* tau, std::complex<double>* c, const int* ldc, std::complex<double>* work, const int* lwork, int* info) {
    HPDDM_F77(zunmtr)(side, uplo, trans, m, n, a, lda, tau, c, ldc, work, lwork, info);
}
} // HPDDM
#endif // _LAPACK_
