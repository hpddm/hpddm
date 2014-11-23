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

extern "C" {
void dsaupd_(int*, char*, int*, const char*, int*, double*, double*, int*, double*, int*,
             int*, int*, double*, double*, int*, int*, int, int);
void znaupd_(int*, char*, int*, const char*, int*, double*, std::complex<double>*, int*,
             std::complex<double>*, int*, int*, int*, std::complex<double>*,
             std::complex<double>*, int*, double*, int*, int, int);
void dseupd_(int*, char*, int*, double*, double*, int*, const double*, char*, int*,
             const char*, int*, double*, double*, int*, double*, int*, int*, int*,
             double*, double*, int*, int*, int, int, int);
void zneupd_(int*, char*, int*, std::complex<double>*, std::complex<double>*, int*,
             const std::complex<double>*, std::complex<double>*, char*, int*, const char*, int*,
             double*, std::complex<double>*, int*, std::complex<double>*, int*, int*,
             int*, std::complex<double>*, std::complex<double>*, int*, double*, int*, int, int, int);
}

namespace HPDDM {
/* Class: Arpack
 *
 *  A class inheriting from <Eigensolver> to use <Arpack>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Arpack : public Eigensolver {
    private:
        /* Variable: it
         *  Maximum number of iterations of the IRAM. */
        unsigned short              _it;
        /* Variable: which
         *  Eigenpairs to retrieve. */
        static const char      _which[];
        /* Function: aupd
         *  Iterates the implicitly restarted Arnoldi method. */
        static inline void aupd(int*, char*, int*, const char*, int*, double*, K*, int*, K*, int*, int*, K*, K*, int*, double*, int*);
        /* Function: eupd
         *  Post-processes the eigenpairs computed with <Arpack::aupd>. */
        static inline void eupd(int*, char*, int*, K*, K*, int*, const K*, K*, char*, const char*, int*, double*, K*, int*, K*, int*, int*, K*, K*, int*, double*, int*);
    public:
        Arpack(double tol, double threshold, int n, int nu, unsigned short it) : Eigensolver(tol, threshold, n, nu), _it(it) { }
        Arpack(double threshold, int n, int nu)                                : Eigensolver(threshold, n, nu), _it(100) { }
        Arpack(int n, int nu)                                                  : Eigensolver(n, nu), _it(100) { }
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
        inline void solve(MatrixCSR<K>* const& A, MatrixCSR<K>* const& B, K**& ev, const MPI_Comm& communicator) {
            int ido = 0;
            char bmat = 'G';
            int iparam[11] = { 1, 0, _it, 1, 0, 0, 3, 0, 0, 0, 0 };
            int ipntr[std::is_same<K, double>::value ? 11 : 14] = { };
            int ncv = 2 * _nu + 1;
            int lworkl = ncv * (ncv + 8);
            K* workd;
            K* workev;
            if(std::is_same<K, double>::value) {
                workd = new K[lworkl + (ncv + 4) * _n];
                workev = nullptr;
            }
            else {
                lworkl += ncv * (2 * ncv - 3);
                workd = new K[lworkl + (ncv + 4) * _n + 2 * ncv];
                workev = workd + lworkl + (ncv + 4) * _n;
            }
            K* workl = workd + 3 * _n;
            K* vp = workl + lworkl;
            double* rwork = nullptr;
            if(!std::is_same<K, double>::value)
                rwork = new double[_n];
            int info = 0;
            K* vresid = vp + ncv * _n;
            Solver<K> s;
#if defined(MUMPSSUB)
            s.numfact(A, false);
#else
            s.numfact(A, true);
#endif
            while(1) {
                aupd(&ido, &bmat, &_n, _which, &_nu, &_tol, vresid, &ncv,
                     vp, iparam, ipntr, workd, workl, &lworkl, rwork, &info);
                if(ido == 99)
                    break;
                else if(ido == -1) {
                    Wrapper<K>::template csrmv<'C'>(B->_sym, &_n, B->_a, B->_ia, B->_ja, workd + ipntr[0] - 1, workd + ipntr[1] - 1);
                    s.solve(workd + ipntr[1] - 1);
                }
                else if(ido == 1)
                    s.solve(workd + ipntr[2] - 1, workd + ipntr[1] - 1);
                else
                    Wrapper<K>::template csrmv<'C'>(B->_sym, &_n, B->_a, B->_ia, B->_ja, workd + ipntr[0] - 1, workd + ipntr[1] - 1);
            }
            _nu = iparam[4];
            if(_nu) {
                K* evr = new K[_nu];
                ev = new K*[_nu];
                *ev = new K[_n * _nu];
                for(unsigned short i = 1; i < _nu; ++i)
                    ev[i] = *ev + i * _n;
                char HowMny = 'A';
                int rvec = 1;
                int necv = ncv;
                int* select = new int[ncv];
                eupd(&rvec, &HowMny, select, evr, *ev, &_n, &(Wrapper<K>::d__0), workev, &bmat,
                     _which, &_nu, &_tol, vresid, &necv, vp, iparam,
                     ipntr, workd, workl, &lworkl, rwork, &info);
                delete [] select;
                if(_threshold > 0.0)
                    selectNu(evr, communicator);
                delete [] evr;
            }
            delete [] workd;
            delete [] rwork;
        }
};

template<>
const char Arpack<double>::_which[3] = "LM";
template<>
const char Arpack<std::complex<double>>::_which[3] = "LM";

template<>
inline void Arpack<double>::aupd(int* ido, char* bmat, int* n, const char* which, int* nu, double* tol, double* vresid, int* ncv, double* vp, int* iparam, int* ipntr, double* workd, double* workl, int* lworkl, double* rwork, int* info) {
    dsaupd_(ido, bmat, n, which, nu, tol, vresid, ncv, vp, n, iparam,
            ipntr, workd, workl, lworkl, info, 1, 2);
}
template<>
inline void Arpack<std::complex<double>>::aupd(int* ido, char* bmat, int* n, const char* which, int* nu, double* tol, std::complex<double>* vresid, int* ncv, std::complex<double>* vp, int* iparam, int* ipntr, std::complex<double>* workd, std::complex<double>* workl, int* lworkl, double* rwork, int* info) {
    znaupd_(ido, bmat, n, which, nu, tol, vresid, ncv, vp, n, iparam,
            ipntr, workd, workl, lworkl, rwork, info, 1, 2);
}

template<>
inline void Arpack<double>::eupd(int* rvec, char* HowMny, int* select, double* evr, double* ev, int* n, const double* sigma, double* workev, char* bmat, const char* which, int* nu, double* tol, double* vresid, int* necv, double* vp, int* iparam, int* ipntr, double* workd, double* workl, int* lworkl, double* rwork, int* info) {
    dseupd_(rvec, HowMny, select, evr, ev, n, sigma, bmat,
            n, which, nu, tol, vresid, necv, vp, n, iparam,
            ipntr, workd, workl, lworkl, info, 1, 1, 2);
}
template<>
inline void Arpack<std::complex<double>>::eupd(int* rvec, char* HowMny, int* select, std::complex<double>* evr, std::complex<double>* ev, int* n, const std::complex<double>* sigma, std::complex<double>* workev, char* bmat, const char* which, int* nu, double* tol, std::complex<double>* vresid, int* necv, std::complex<double>* vp, int* iparam, int* ipntr, std::complex<double>* workd, std::complex<double>* workl, int* lworkl, double* rwork, int* info) {
    zneupd_(rvec, HowMny, select, evr, ev, n, sigma, workev, bmat,
            n, which, nu, tol, vresid, necv, vp, n, iparam,
            ipntr, workd, workl, lworkl, rwork, info, 1, 1, 2);
}
} // HPDDM
#endif // _ARPACK_
