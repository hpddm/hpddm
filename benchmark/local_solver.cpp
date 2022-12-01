/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2016-02-28

   Copyright (C) 2016-     Centre National de la Recherche Scientifique

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

#include <chrono>
#include <deque>
#if HPDDM_MKL
# include <complex>
# define MKL_Complex16         std::complex<double>
# define MKL_Complex8          std::complex<float>
# define MKL_INT               int
#endif
#define HPDDM_MINIMAL
#include <HPDDM.hpp>
#ifdef MUMPSSUB
# include "HPDDM_MUMPS.hpp"
#elif defined(MKL_PARDISOSUB)
# include "HPDDM_MKL_PARDISO.hpp"
#elif defined(PASTIXSUB)
# include "HPDDM_PaStiX.hpp"
#elif defined(SUITESPARSESUB)
# include "HPDDM_SuiteSparse.hpp"
#elif defined(DISSECTIONSUB)
# include "HPDDM_Dissection.hpp"
#endif

#ifdef FORCE_SINGLE
#ifdef FORCE_COMPLEX
typedef std::complex<float> K;
#else
typedef float K;
#endif
#else
#ifdef FORCE_COMPLEX
typedef std::complex<double> K;
#else
typedef double K;
#endif
#endif

int main(int argc, char** argv) {
    if(argc < 2)
        return 1;
    HPDDM::MatrixCSR<K>* A = nullptr;
    {
        auto tBegin = std::chrono::steady_clock::now();
        std::ifstream t(argv[1]);
        A = new HPDDM::MatrixCSR<K>(t);
        if(A->_n <= 0) {
            delete A;
            return 1;
        }
        auto tEnd = std::chrono::steady_clock::now();
        std::cout << "// matrix read from file in " << std::chrono::duration<double>(tEnd - tBegin).count() << " second(s)\n";
    }
    {
#if defined(MUMPSSUB) || defined(PASTIXSUB)
        MPI_Init(&argc, &argv);
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
        int size = 1;
#endif
#ifdef _OPENMP
        int th = omp_get_max_threads();
#else
        int th = 1;
#endif
        std::cout << "// " << size << " MPI process" << (size > 1 ? "es" : "") << " x " << th << " thread" << (th > 1 ? "s" : "") << " = " << (size * th) << " worker" << (size * th > 1 ? "s" : "") << std::endl;
    }
    HPDDM::Option& opt = *HPDDM::Option::get();
    opt.parse(argc, argv, false, {
        std::forward_as_tuple("warm_up=<1>", "Number of fake runs to prime the pump.", HPDDM::Option::Arg::integer),
        std::forward_as_tuple("trials=<3>", "Number of trial runs to time.", HPDDM::Option::Arg::integer),
        std::forward_as_tuple("rhs=<1>", "Number of generated random right-hand sides.", HPDDM::Option::Arg::positive),
        std::forward_as_tuple("solve_phase_only=(0|1)", "Benchmark only the solve phase.", HPDDM::Option::Arg::argument)
    });
    int mu = opt.app()["rhs"];
    bool solve = opt.app()["solve_phase_only"];
    K* rhs = new K[mu * A->_n];
    std::fill_n(rhs, mu * A->_n, K(1.0));
    std::streamsize old = std::cout.precision();
    SUBDOMAIN<K>* S = new SUBDOMAIN<K>;
    if(solve)
        S->numfact(A);
    std::cout << std::scientific;
    for(unsigned int begin = 0, end = opt.app()["warm_up"]; begin < end; ++begin) {
        if(!solve)
            S->numfact(A);
        S->solve(rhs, mu);
        if(!solve) {
            delete S;
            S = new SUBDOMAIN<K>;
        }
    }
    for(unsigned int begin = 0, end = opt.app()["trials"]; begin < end; ++begin) {
        if(!solve) {
            auto tBegin = std::chrono::steady_clock::now();
            S->numfact(A);
            auto tEnd = std::chrono::steady_clock::now();
            std::cout << std::setw(10) << std::setprecision(5) << std::chrono::duration<double>(tEnd - tBegin).count() << "\t";
        }
        std::deque<unsigned short> q;
        for(unsigned short nu = mu; nu >= 1; nu /= 2)
            q.emplace_front(nu);
        for(const unsigned short& nu : q) {
            auto tBegin = std::chrono::steady_clock::now();
            S->solve(rhs, nu);
            auto tEnd = std::chrono::steady_clock::now();
            std::cout << std::setw(10) << std::setprecision(5) << std::chrono::duration<double>(tEnd - tBegin).count() << (nu > 1 ? "\t" : "");
        }
        std::cout << "\n";
        if(!solve) {
            delete S;
            S = new SUBDOMAIN<K>;
        }
    }
    std::cout.unsetf(std::ios_base::scientific);
    std::cout.precision(old);
    delete S;
    delete [] rhs;
#if defined(MUMPSSUB) || defined(PASTIXSUB)
    MPI_Finalize();
#endif
    delete A;
    return 0;
}
