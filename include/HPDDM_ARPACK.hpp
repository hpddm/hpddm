/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2012-12-15

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

#ifndef HPDDM_ARPACK_HPP_
#define HPDDM_ARPACK_HPP_

#define HPDDM_GENERATE_ARPACK_EXTERN(C, T, B, U)                                                              \
void HPDDM_F77(B ## saupd)(int*, const char*, const int*, const char*, const int*, const U*, U*, int*, U*,    \
                           const int*, int*, int*, U*, U*, int*, int*, int, int);                             \
void HPDDM_F77(B ## seupd)(const int*, const char*, int*, U*, U*, const int*, const U*, const char*,          \
                           const int*, const char*, const int*, const U*, U*, int*, U*, const int*, int*,     \
                           int*, U*, U*, int*, int*, int, int, int);                                          \
void HPDDM_F77(C ## naupd)(int*, const char*, const int*, const char*, const int*, const U*, T*, int*,        \
                           T*, const int*, int*, int*, T*, T*, int*, U*, int*, int, int);                     \
void HPDDM_F77(C ## neupd)(const int*, const char*, int*, T*, T*, const int*, const T*, T*, const char*,      \
                           const int*, const char*, const int*, const U*, T*, int*, T*, const int*, int*,     \
                           int*, T*, T*, int*, U*, int*, int, int, int);

extern "C" {
HPDDM_GENERATE_ARPACK_EXTERN(c, std::complex<float>, s, float)
HPDDM_GENERATE_ARPACK_EXTERN(z, std::complex<double>, d, double)
}

#include "HPDDM_eigensolver.hpp"

namespace HPDDM {
#ifdef MU_ARPACK
#undef HPDDM_CHECK_COARSEOPERATOR
#undef HPDDM_CHECK_SUBDOMAIN
#define HPDDM_CHECK_EIGENSOLVER
#include "HPDDM_preprocessor_check.hpp"
#define EIGENSOLVER HPDDM::Arpack
/* Class: Arpack
 *
 *  A class inheriting from <Eigensolver> to use <Arpack> for sparse eigenvalue problems.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Arpack : public Eigensolver<K> {
    private:
        /* Variable: it
         *  Maximum number of iterations of the IRAM. */
        unsigned short                        it_;
        /* Variable: which
         *  Eigenpairs to retrieve. */
        static constexpr const char* const which_ = Wrapper<K>::is_complex ? "LM" : "LM";
        /* Function: aupd
         *  Iterates the implicitly restarted Arnoldi method. */
        static void aupd(int*, const char*, const int*, const char*, const int*, const underlying_type<K>*, K*, int*, K*, int*, int*, K*, K*, int*, underlying_type<K>*, int*);
        /* Function: eupd
         *  Post-processes the eigenpairs computed with <Arpack::aupd>. */
        static void eupd(const int*, const char*, int*, K*, K*, const int*, const K*, K*, const char*, const char*, const int*, const underlying_type<K>*, K*, int*, K*, int*, int*, K*, K*, int*, underlying_type<K>*, int*);
    public:
        Arpack(int n, int nu)                                                                          : Eigensolver<K>(n, nu), it_(100) { }
        Arpack(underlying_type<K> threshold, int n, int nu)                                            : Eigensolver<K>(threshold, n, nu), it_(100) { }
        Arpack(underlying_type<K> tol, underlying_type<K> threshold, int n, int nu, unsigned short it) : Eigensolver<K>(tol, threshold, n, nu), it_(it) { }
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
        void solve(MatrixCSR<K>* const& A, MatrixCSR<K>* const& B, K**& ev, const MPI_Comm& communicator, Solver<K>* const& s = nullptr, std::ios_base::openmode mode = std::ios_base::out) {
            int iparam[11] { 1, 0, it_, 1, 0, 0, 3, 0, 0, 0, 0 };
            int ipntr[!Wrapper<K>::is_complex ? 11 : 14] { };
            if(4 * Eigensolver<K>::nu_ > Eigensolver<K>::n_)
                Eigensolver<K>::nu_ = std::max(1, Eigensolver<K>::n_ / 4);
            int ncv = std::min(Eigensolver<K>::n_, std::max(Option::get()->val<int>("arpack_ncv"), 2 * Eigensolver<K>::nu_ + 1));
            int lworkl = ncv * (ncv + 8);
            K* workd, *workev, *evr = nullptr;
            if(!Wrapper<K>::is_complex) {
                workd = new K[lworkl + (ncv + 4) * Eigensolver<K>::n_]();
                workev = nullptr;
            }
            else {
                lworkl += ncv * (2 * ncv - 3);
                workd = new K[lworkl + (ncv + 4) * Eigensolver<K>::n_ + 2 * ncv]();
                workev = workd + lworkl + (ncv + 4) * Eigensolver<K>::n_;
            }
            K* workl = workd + 3 * Eigensolver<K>::n_;
            K* vp = workl + lworkl;
            underlying_type<K>* rwork = nullptr;
            if(Wrapper<K>::is_complex)
                rwork = new underlying_type<K>[Eigensolver<K>::n_];
            K* vresid = vp + ncv * Eigensolver<K>::n_;
            int info;
            if(Eigensolver<K>::nu_) {
                Solver<K>* const prec = s ? s : new Solver<K>;
#ifdef MUMPSSUB
                prec->numfact(A, false);
#else
                prec->numfact(A, true);
#endif
                do {
                    const int* const n = &(Eigensolver<K>::n_), *const nu = &(Eigensolver<K>::nu_);
                    const underlying_type<K>* const tol = &(Eigensolver<K>::tol_);
                    auto loop = [&]() {
                        int ido = info = 0;
                        while(ido != 99) {
                            aupd(&ido, "G", n, which_, nu, tol, vresid, &ncv,
                                 vp, iparam, ipntr, workd, workl, &lworkl, rwork, &info);
                            if(ido == -1) {
                                if(B) {
                                    if(B->ia_ && B->ja_)
                                        Wrapper<K>::csrmv(B->sym_, n, B->a_, B->ia_, B->ja_, workd + ipntr[0] - 1, workd + ipntr[1] - 1);
                                    else {
                                        if(B->sym_)
                                            Blas<K>::symv("L", n, &(Wrapper<K>::d__1), B->a_, n, workd + ipntr[0] - 1, &i__1, &(Wrapper<K>::d__0), workd + ipntr[1] - 1, &i__1);
                                        else
                                            Blas<K>::gemv("N", n, n, &(Wrapper<K>::d__1), B->a_, n, workd + ipntr[0] - 1, &i__1, &(Wrapper<K>::d__0), workd + ipntr[1] - 1, &i__1);
                                    }
                                }
                                else
                                    std::copy_n(workd + ipntr[0] - 1, *n, workd + ipntr[1] - 1);
                                prec->solve(workd + ipntr[1] - 1);
                            }
                            else if(ido == 1)
                                prec->solve(workd + ipntr[2] - 1, workd + ipntr[1] - 1);
                            else {
                                if(B) {
                                    if(B->ia_ && B->ja_)
                                        Wrapper<K>::csrmv(B->sym_, n, B->a_, B->ia_, B->ja_, workd + ipntr[0] - 1, workd + ipntr[1] - 1);
                                    else {
                                        if(B->sym_)
                                            Blas<K>::symv("L", n, &(Wrapper<K>::d__1), B->a_, n, workd + ipntr[0] - 1, &i__1, &(Wrapper<K>::d__0), workd + ipntr[1] - 1, &i__1);
                                        else
                                            Blas<K>::gemv("N", n, n, &(Wrapper<K>::d__1), B->a_, n, workd + ipntr[0] - 1, &i__1, &(Wrapper<K>::d__0), workd + ipntr[1] - 1, &i__1);
                                    }
                                }
                                else
                                    std::copy_n(workd + ipntr[0] - 1, *n, workd + ipntr[1] - 1);
                            }
                        }
                    };
                    loop();
                    if(info == -9999) {
                        Eigensolver<K>::nu_ = std::ceil(Eigensolver<K>::nu_ / 3);
                        std::fill_n(iparam + 4, 7, 0);
                        iparam[2] = it_, iparam[6] = 3;
                        ncv = 2 * Eigensolver<K>::nu_ + 1;
                    }
                } while(info == -9999 && Eigensolver<K>::nu_ > 1);
                if(!s)
                    delete prec;
                Eigensolver<K>::nu_ = iparam[4];
            }
            if(Eigensolver<K>::nu_) {
                evr = new K[Eigensolver<K>::nu_];
                ev = new K*[Eigensolver<K>::nu_];
                *ev = new K[Eigensolver<K>::n_ * Eigensolver<K>::nu_];
                for(unsigned short i = 1; i < Eigensolver<K>::nu_; ++i)
                    ev[i] = *ev + i * Eigensolver<K>::n_;
                int* select = new int[ncv];
                eupd(&i__1, "A", select, evr, *ev, &(Eigensolver<K>::n_), &(Wrapper<K>::d__0), workev, "G",
                     which_, &(Eigensolver<K>::nu_), &(Eigensolver<K>::tol_), vresid, &ncv, vp, iparam,
                     ipntr, workd, workl, &lworkl, rwork, &info);
                delete [] select;
                std::string name = Eigensolver<K>::dump(evr, ev, communicator, mode);
                if(!name.empty()) {
                    std::ofstream output(name, std::fstream::in | std::fstream::out | std::fstream::app);
                    output << "ARPACK information:\n";
                    output << "\t" << iparam[4] << " Arnoldi update iteration" << (iparam[4] > 1 ? "s" : "") << "\n";
                    output << "\t" << iparam[8] << " (y = OP x) operation" << (iparam[8] > 1 ? "s" : "") << "\n";
                    output << "\t" << iparam[9] << " (y = B x) operation" << (iparam[9] > 1 ? "s" : "") << "\n";
                    output << "\t" << iparam[10] << " step" << (iparam[10] > 1 ? "s" : "") << " of re-orthogonalization\n";
                    output << "\n\n";
                }
            }
            else {
                ev = new K*[1];
                *ev = nullptr;
            }
            if(Eigensolver<K>::threshold_ > 0.0)
                Eigensolver<K>::selectNu(evr, ev, communicator);
            delete [] evr;
            delete [] workd;
            delete [] rwork;
        }
};
#endif // MU_ARPACK

#define HPDDM_GENERATE_ARPACK(C, T, B, U)                                                                    \
template<>                                                                                                   \
inline void Arpack<U>::aupd(int* ido, const char* bmat, const int* n, const char* which, const int* nu,      \
                            const U* tol, U* vresid, int* ncv, U* vp, int* iparam, int* ipntr, U* workd,     \
                            U* workl, int* lworkl, U*, int* info) {                                          \
    HPDDM_F77(B ## saupd)(ido, bmat, n, which, nu, tol, vresid, ncv, vp, n, iparam,                          \
                          ipntr, workd, workl, lworkl, info, 1, 2);                                          \
}                                                                                                            \
template<>                                                                                                   \
inline void Arpack<U>::eupd(const int* rvec, const char* HowMny, int* select, U* evr, U* ev, const int* n,   \
                            const U* sigma, U*, const char* bmat, const char* which, const int* nu,          \
                            const U* tol, U* vresid, int* necv, U* vp, int* iparam, int* ipntr,              \
                            U* workd, U* workl, int* lworkl, U*, int* info) {                                \
    HPDDM_F77(B ## seupd)(rvec, HowMny, select, evr, ev, n, sigma, bmat,                                     \
                          n, which, nu, tol, vresid, necv, vp, n, iparam,                                    \
                          ipntr, workd, workl, lworkl, info, 1, 1, 2);                                       \
}                                                                                                            \
template<>                                                                                                   \
inline void Arpack<T>::aupd(int* ido, const char* bmat, const int* n, const char* which, const int* nu,      \
                            const U* tol, T* vresid, int* ncv, T* vp, int* iparam, int* ipntr, T* workd,     \
                            T* workl, int* lworkl, U* rwork, int* info) {                                    \
    HPDDM_F77(C ## naupd)(ido, bmat, n, which, nu, tol, vresid, ncv, vp, n, iparam,                          \
                          ipntr, workd, workl, lworkl, rwork, info, 1, 2);                                   \
}                                                                                                            \
template<>                                                                                                   \
inline void Arpack<T>::eupd(const int* rvec, const char* HowMny, int* select, T* evr, T* ev, const int* n,   \
                            const T* sigma, T* workev, const char* bmat, const char* which, const int* nu,   \
                            const U* tol, T* vresid, int* necv, T* vp, int* iparam, int* ipntr,              \
                            T* workd, T* workl, int* lworkl, U* rwork, int* info) {                          \
    HPDDM_F77(C ## neupd)(rvec, HowMny, select, evr, ev, n, sigma, workev, bmat,                             \
                          n, which, nu, tol, vresid, necv, vp, n, iparam,                                    \
                          ipntr, workd, workl, lworkl, rwork, info, 1, 1, 2);                                \
}
HPDDM_GENERATE_ARPACK(c, std::complex<float>, s, float)
HPDDM_GENERATE_ARPACK(z, std::complex<double>, d, double)
} // HPDDM
#endif // HPDDM_ARPACK_HPP_
