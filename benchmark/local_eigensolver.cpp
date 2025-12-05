/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2016-07-12

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
#if HPDDM_MKL
  #include <complex>
  #define MKL_Complex16 std::complex<double>
  #define MKL_Complex8  std::complex<float>
  #define MKL_INT       int
#endif
#define HPDDM_MINIMAL
#include <HPDDM.hpp>
#ifdef MUMPSSUB
  #include "HPDDM_MUMPS.hpp"
#elif defined(MKL_PARDISOSUB)
  #include "HPDDM_MKL_PARDISO.hpp"
#elif defined(PASTIXSUB)
  #include "HPDDM_PaStiX.hpp"
#elif defined(SUITESPARSESUB)
  #include "HPDDM_SuiteSparse.hpp"
#elif defined(DISSECTIONSUB)
  #include "HPDDM_Dissection.hpp"
#endif
#include "HPDDM_eigensolver.hpp"
#ifdef MU_ARPACK
  #include "HPDDM_ARPACK.hpp"
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

int main(int argc, char **argv)
{
  if (argc < 3) return 1;
  HPDDM::MatrixCSR<K> *A = nullptr;
  {
    auto          tBegin = std::chrono::steady_clock::now();
    std::ifstream t(argv[1]);
    A = new HPDDM::MatrixCSR<K>(t);
    if (A->n_ <= 0) {
      delete A;
      return 1;
    }
    auto tEnd = std::chrono::steady_clock::now();
    std::cout << "// left-hand side matrix read from file in " << std::chrono::duration<double, std::ratio<1>>(tEnd - tBegin).count() << " second(s)\n";
  }
  HPDDM::MatrixCSR<K> *B = nullptr;
  {
    auto          tBegin = std::chrono::steady_clock::now();
    std::ifstream t(argv[2]);
    B = new HPDDM::MatrixCSR<K>(t);
    if (B->n_ <= 0) {
      delete B;
      delete A;
      return 1;
    }
    auto tEnd = std::chrono::steady_clock::now();
    std::cout << "// right-hand side matrix read from file in " << std::chrono::duration<double, std::ratio<1>>(tEnd - tBegin).count() << " second(s)\n";
  }
  {
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size > 1) {
      delete B;
      delete A;
      MPI_Finalize();
      return 1;
    }
#ifdef _OPENMP
    int th = omp_get_max_threads();
#else
    int th = 1;
#endif
    std::cout << "// " << size << " MPI process" << (size > 1 ? "es" : "") << " x " << th << " thread" << (th > 1 ? "s" : "") << " = " << (size * th) << " worker" << (size * th > 1 ? "s" : "") << std::endl;
  }
  HPDDM::Option &opt = *HPDDM::Option::get();
  opt.parse(argc, argv, false,
            {
              std::forward_as_tuple("warm_up=<2>", "Number of fake runs to prime the pump.", HPDDM::Option::Arg::integer),
              std::forward_as_tuple("trials=<5>", "Number of trial runs to time.", HPDDM::Option::Arg::integer),
            });
  std::streamsize old = std::cout.precision();
  EIGENSOLVER<K> *S   = new EIGENSOLVER<K>(A->n_, HPDDM::Option::get()->val("geneo_nu", 20));
  std::cout << std::scientific;
  K **ev;
  for (unsigned int begin = 0, end = opt.app()["warm_up"]; begin < end; ++begin) {
    S->template solve<SUBDOMAIN>(A, B, ev, MPI_COMM_SELF);
    if (ev) delete[] *ev;
    delete[] ev;
    delete S;
    S = new EIGENSOLVER<K>(A->n_, HPDDM::Option::get()->val("geneo_nu", 20));
  }
  for (unsigned int begin = 0, end = opt.app()["trials"]; begin < end; ++begin) {
    auto tBegin = std::chrono::steady_clock::now();
    S->template solve<SUBDOMAIN>(A, B, ev, MPI_COMM_SELF);
    auto tEnd = std::chrono::steady_clock::now();
    std::cout << std::setw(10) << std::setprecision(5) << std::chrono::duration<double, std::milli>(tEnd - tBegin).count() << "\n";
    if (ev) delete[] *ev;
    delete[] ev;
    delete S;
    S = new EIGENSOLVER<K>(A->n_, HPDDM::Option::get()->val("geneo_nu", 20));
  }
  std::cout.unsetf(std::ios_base::scientific);
  std::cout.precision(old);
  delete S;
  MPI_Finalize();
  delete B;
  delete A;
  return 0;
}
