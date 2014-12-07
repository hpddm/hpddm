/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <jolivet@ann.jussieu.fr>
        Date: 2012-12-15

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

#ifndef _ARPACK_
#define _ARPACK_

#define EIGENSOLVER HPDDM::Arpack

#define HPDDM_GENERATE_ARPACK_EXTERN_REAL(C, T)                                                               \
void C ## saupd_(int*, char*, int*, const char*, int*, T*, T*, int*, T*, int*,                                \
                 int*, int*, T*, T*, int*, int*, int, int);                                                   \
void C ## seupd_(const int*, char*, int*, T*, T*, int*, const T*, char*, int*,                                \
                 const char*, int*, T*, T*, int*, T*, int*, int*, int*,                                       \
                 T*, T*, int*, int*, int, int, int);
#define HPDDM_GENERATE_ARPACK_EXTERN_COMPLEX(C, T)                                                            \
void C ## naupd_(int*, char*, int*, const char*, int*, T*, std::complex<T>*, int*,                            \
                 std::complex<T>*, int*, int*, int*, std::complex<T>*,                                        \
                 std::complex<T>*, int*, T*, int*, int, int);                                                 \
void C ## neupd_(const int*, char*, int*, std::complex<T>*, std::complex<T>*, int*,                           \
                 const std::complex<T>*, std::complex<T>*, char*, int*, const char*, int*,                    \
                 T*, std::complex<T>*, int*, std::complex<T>*, int*, int*,                                    \
                 int*, std::complex<T>*, std::complex<T>*, int*, T*, int*, int, int, int);

extern "C" {
HPDDM_GENERATE_ARPACK_EXTERN_REAL(s, float)
HPDDM_GENERATE_ARPACK_EXTERN_REAL(d, double)
HPDDM_GENERATE_ARPACK_EXTERN_COMPLEX(c, float)
HPDDM_GENERATE_ARPACK_EXTERN_COMPLEX(z, double)
}

namespace HPDDM {
/* Class: Arpack
 *
 *  A class inheriting from <Eigensolver> to use <Arpack>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Arpack : public Eigensolver<K> {
    private:
        /* Variable: it
         *  Maximum number of iterations of the IRAM. */
        unsigned short                        _it;
        /* Variable: which
         *  Eigenpairs to retrieve. */
        static constexpr const char* const _which = std::is_same<K, typename Wrapper<K>::ul_type>::value ? "LM" : "LM";
        /* Function: aupd
         *  Iterates the implicitly restarted Arnoldi method. */
        static inline void aupd(int*, char*, int*, const char*, int*, typename Wrapper<K>::ul_type*, K*, int*, K*, int*, int*, K*, K*, int*, typename Wrapper<K>::ul_type*, int*);
        /* Function: eupd
         *  Post-processes the eigenpairs computed with <Arpack::aupd>. */
        static inline void eupd(const int*, char*, int*, K*, K*, int*, const K*, K*, char*, const char*, int*, typename Wrapper<K>::ul_type*, K*, int*, K*, int*, int*, K*, K*, int*, typename Wrapper<K>::ul_type*, int*);
    public:
        Arpack(int n, int nu)                                                                                              : Eigensolver<K>(n, nu), _it(100) { }
        Arpack(typename Wrapper<K>::ul_type threshold, int n, int nu)                                                      : Eigensolver<K>(threshold, n, nu), _it(100) { }
        Arpack(typename Wrapper<K>::ul_type tol, typename Wrapper<K>::ul_type threshold, int n, int nu, unsigned short it) : Eigensolver<K>(tol, threshold, n, nu), _it(it) { }
        /* Function: solve
         *
         *  Computes eigenvectors of the generalized eigenvalue problem Ax = l Bx.
         *
         * Parameters:
         *    A              - Left-hand side matrix.
         *    B              - Right-hand side matrix.
         *    ev             - Array of eigenvectors.
         *    communicator   - MPI communicator for selecting the threshold criterion. */
        template<template<class> class Solver>
        inline void solve(MatrixCSR<K>* const& A, MatrixCSR<K>* const& B, K**& ev, const MPI_Comm& communicator, Solver<K>* const& s = nullptr) {
            int ido = 0;
            char bmat = 'G';
            int iparam[11] = { 1, 0, _it, 1, 0, 0, 3, 0, 0, 0, 0 };
            int ipntr[std::is_same<K, typename Wrapper<K>::ul_type>::value ? 11 : 14] = { };
            int ncv = 2 * Eigensolver<K>::_nu + 1;
            int lworkl = ncv * (ncv + 8);
            K* workd;
            K* workev;
            if(std::is_same<K, typename Wrapper<K>::ul_type>::value) {
                workd = new K[lworkl + (ncv + 4) * Eigensolver<K>::_n];
                workev = nullptr;
            }
            else {
                lworkl += ncv * (2 * ncv - 3);
                workd = new K[lworkl + (ncv + 4) * Eigensolver<K>::_n + 2 * ncv];
                workev = workd + lworkl + (ncv + 4) * Eigensolver<K>::_n;
            }
            K* workl = workd + 3 * Eigensolver<K>::_n;
            K* vp = workl + lworkl;
            typename Wrapper<K>::ul_type* rwork = nullptr;
            if(!std::is_same<K, typename Wrapper<K>::ul_type>::value)
                rwork = new typename Wrapper<K>::ul_type[Eigensolver<K>::_n];
            int info = 0;
            K* vresid = vp + ncv * Eigensolver<K>::_n;
            Solver<K>* const prec = s ? s : new Solver<K>;
#if defined(MUMPSSUB)
            prec->numfact(A, false);
#else
            prec->numfact(A, true);
#endif
            while(1) {
                aupd(&ido, &bmat, &(Eigensolver<K>::_n), _which, &(Eigensolver<K>::_nu), &(Eigensolver<K>::_tol), vresid, &ncv,
                     vp, iparam, ipntr, workd, workl, &lworkl, rwork, &info);
                if(ido == 99)
                    break;
                else if(ido == -1) {
                    Wrapper<K>::template csrmv<'C'>(B->_sym, &(Eigensolver<K>::_n), B->_a, B->_ia, B->_ja, workd + ipntr[0] - 1, workd + ipntr[1] - 1);
                    prec->solve(workd + ipntr[1] - 1);
                }
                else if(ido == 1)
                    prec->solve(workd + ipntr[2] - 1, workd + ipntr[1] - 1);
                else
                    Wrapper<K>::template csrmv<'C'>(B->_sym, &(Eigensolver<K>::_n), B->_a, B->_ia, B->_ja, workd + ipntr[0] - 1, workd + ipntr[1] - 1);
            }
            if(s == nullptr)
                delete prec;
            Eigensolver<K>::_nu = iparam[4];
            if(Eigensolver<K>::_nu) {
                K* evr = new K[Eigensolver<K>::_nu];
                ev = new K*[Eigensolver<K>::_nu];
                *ev = new K[Eigensolver<K>::_n * Eigensolver<K>::_nu];
                for(unsigned short i = 1; i < Eigensolver<K>::_nu; ++i)
                    ev[i] = *ev + i * Eigensolver<K>::_n;
                char HowMny = 'A';
                int* select = new int[ncv];
                eupd(&i__1, &HowMny, select, evr, *ev, &(Eigensolver<K>::_n), &(Wrapper<K>::d__0), workev, &bmat,
                     _which, &(Eigensolver<K>::_nu), &(Eigensolver<K>::_tol), vresid, &ncv, vp, iparam,
                     ipntr, workd, workl, &lworkl, rwork, &info);
                delete [] select;
                if(Eigensolver<K>::_threshold > 0.0)
                    Eigensolver<K>::selectNu(evr, communicator);
                delete [] evr;
            }
            delete [] workd;
            delete [] rwork;
        }
};

#define HPDDM_GENERATE_ARPACK_REAL(C, T)                                                                     \
template<>                                                                                                   \
inline void Arpack<T>::aupd(int* ido, char* bmat, int* n, const char* which, int* nu, T* tol,                \
                            T* vresid, int* ncv, T* vp, int* iparam, int* ipntr, T* workd, T* workl,         \
                            int* lworkl, T* rwork, int* info) {                                              \
    C ## saupd_(ido, bmat, n, which, nu, tol, vresid, ncv, vp, n, iparam,                                    \
                ipntr, workd, workl, lworkl, info, 1, 2);                                                    \
}                                                                                                            \
template<>                                                                                                   \
inline void Arpack<T>::eupd(const int* rvec, char* HowMny, int* select, T* evr, T* ev, int* n,               \
                            const T* sigma, T* workev, char* bmat, const char* which, int* nu, T* tol,       \
                            T* vresid, int* necv, T* vp, int* iparam, int* ipntr,                            \
                            T* workd, T* workl, int* lworkl, T* rwork, int* info) {                          \
    C ## seupd_(rvec, HowMny, select, evr, ev, n, sigma, bmat,                                               \
                n, which, nu, tol, vresid, necv, vp, n, iparam,                                              \
                ipntr, workd, workl, lworkl, info, 1, 1, 2);                                                 \
}
#define HPDDM_GENERATE_ARPACK_COMPLEX(C, T)                                                                  \
template<>                                                                                                   \
inline void Arpack<std::complex<T>>::aupd(int* ido, char* bmat, int* n, const char* which, int* nu, T* tol,  \
                                          std::complex<T>* vresid, int* ncv, std::complex<T>* vp,            \
                                          int* iparam, int* ipntr, std::complex<T>* workd,                   \
                                          std::complex<T>* workl, int* lworkl, T* rwork, int* info) {        \
    C ## naupd_(ido, bmat, n, which, nu, tol, vresid, ncv, vp, n, iparam,                                    \
                ipntr, workd, workl, lworkl, rwork, info, 1, 2);                                             \
}                                                                                                            \
template<>                                                                                                   \
inline void Arpack<std::complex<T>>::eupd(const int* rvec, char* HowMny, int* select, std::complex<T>* evr,  \
                                          std::complex<T>* ev, int* n, const std::complex<T>* sigma,         \
                                          std::complex<T>* workev, char* bmat, const char* which,            \
                                          int* nu, T* tol, std::complex<T>* vresid, int* necv,               \
                                          std::complex<T>* vp, int* iparam, int* ipntr,                      \
                                          std::complex<T>* workd, std::complex<T>* workl,                    \
                                          int* lworkl, T* rwork, int* info) {                                \
    C ## neupd_(rvec, HowMny, select, evr, ev, n, sigma, workev, bmat,                                       \
                n, which, nu, tol, vresid, necv, vp, n, iparam,                                              \
                ipntr, workd, workl, lworkl, rwork, info, 1, 1, 2);                                          \
}
HPDDM_GENERATE_ARPACK_REAL(s, float)
HPDDM_GENERATE_ARPACK_REAL(d, double)
HPDDM_GENERATE_ARPACK_COMPLEX(c, float)
HPDDM_GENERATE_ARPACK_COMPLEX(z, double)
} // HPDDM
#endif // _ARPACK_
