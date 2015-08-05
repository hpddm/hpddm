/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@inf.ethz.ch>
              Frédéric Nataf <nataf@ann.jussieu.fr>
        Date: 2013-07-14

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

#ifndef _HPDDM_
#define _HPDDM_

/* Constants: C-style preprocessor variables
 *
 *    HPDDM_VERSION       - Version of the framework.
 *    HPDDM_EPS           - Small positive number used internally for dropping values.
 *    HPDDM_PEN           - Large positive number used externally for penalization, e.g. for imposing Dirichlet boundary conditions.
 *    HPDDM_GRANULARITY   - Granularity for OpenMP scheduling.
 *    HPDDM_OUTPUT_CO     - If set to one, the coarse operator is saved to disk (for debugging only).
 *    HPDDM_MKL           - If not set to zero, Intel MKL is chosen as the linear algebra backend.
 *    HPDDM_SCHWARZ       - Overlapping Schwarz methods enabled.
 *    HPDDM_FETI          - FETI methods enabled.
 *    HPDDM_BDD           - BDD methods enabled.
 *    HPDDM_QR            - If not set to zero, pseudo-inverses of Schur complements are computed using dense QR decompositions (with pivoting if set to one, without pivoting otherwise).
 *    HPDDM_ICOLLECTIVE   - If possible, use nonblocking MPI collective operations.
 *    HPDDM_GMV           - For overlapping Schwarz methods, this can be used to reduce the volume of communication for computing global matrix-vector products. */
#define HPDDM_VERSION         000100
#define HPDDM_EPS             1.0e-12
#define HPDDM_PEN             1.0e+30
#define HPDDM_GRANULARITY     50000
#define HPDDM_OUTPUT_CO       0
#ifndef HPDDM_MKL
# ifdef INTEL_MKL_VERSION
#  define HPDDM_MKL           1
# else
#  define HPDDM_MKL           0
# endif
#endif
#ifndef HPDDM_SCHWARZ
# define HPDDM_SCHWARZ        1
#endif
#ifndef HPDDM_FETI
# define HPDDM_FETI           1
#endif
#ifndef HPDDM_BDD
# define HPDDM_BDD            1
#endif
#define HPDDM_QR              2
#define HPDDM_ICOLLECTIVE     0
#define HPDDM_GMV             0

#ifdef __MINGW32__
# include <inttypes.h>
#endif
#include <mpi.h>
#if HPDDM_ICOLLECTIVE
# if !((OMPI_MAJOR_VERSION > 1 || (OMPI_MAJOR_VERSION == 1 && OMPI_MINOR_VERSION >= 7)) || MPICH_NUMVERSION >= 30000000)
#  pragma message("You cannot use nonblocking MPI collective operations with that MPI implementation")
#  undef HPDDM_ICOLLECTIVE
#  define HPDDM_ICOLLECTIVE   0
# endif
#endif // HPDDM_ICOLLECTIVE

#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <complex>
static_assert(2 * sizeof(double) == sizeof(std::complex<double>) && 2 * sizeof(float) == sizeof(std::complex<float>) && 2 * sizeof(float) == sizeof(double), "Incorrect sizes");
#ifdef _OPENMP
# include <omp.h>
#endif
#ifdef __GNUG__
# include <cxxabi.h>
# include <memory>
#endif

#ifndef MKL_Complex16
# define MKL_Complex16 std::complex<double>
#endif
#ifndef MKL_Complex8
# define MKL_Complex8 std::complex<float>
#endif
#ifndef MKL_INT
# define MKL_INT int
#endif
#if HPDDM_MKL
# if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION < 110201
#  define HPDDM_CONST(T, V) const_cast<T*>(V)
# else
#  define HPDDM_CONST(T, V) V
# endif
# define HPDDM_PREFIX_AXPBY(func) cblas_ ## func
# include <mkl_spblas.h>
# include <mkl_vml.h>
# include <mkl_trans.h>
#else
# include <bitset>
#endif // HPDDM_MKL
#if defined(__powerpc__) || defined(INTEL_MKL_VERSION)
# define HPDDM_F77(func) func
#else
# define HPDDM_F77(func) func ## _
#endif

#define HPDDM_GENERATE_EXTERN_BLAS(C, T)                                                                     \
void    HPDDM_F77(C ## axpy)(const int*, const T*, const T*, const int*, T*, const int*);                    \
void    HPDDM_F77(C ## scal)(const int*, const T*, T*, const int*);                                          \
void   HPDDM_F77(C ## lacpy)(const char*, const int*, const int*, const T*, const int*, T*, const int*);     \
void    HPDDM_F77(C ## symv)(const char*, const int*, const T*, const T*, const int*,                        \
                             const T*, const int*, const T*, T*, const int*);                                \
void    HPDDM_F77(C ## gemv)(const char*, const int*, const int*, const T*,                                  \
                             const T*, const int*, const T*, const int*,                                     \
                             const T*, T*, const int*);                                                      \
void    HPDDM_F77(C ## symm)(const char*, const char*, const int*, const int*,                               \
                             const T*, const T*, const int*, const T*, const int*,                           \
                             const T*, T*, const int*);                                                      \
void    HPDDM_F77(C ## gemm)(const char*, const char*, const int*, const int*, const int*,                   \
                             const T*, const T*, const int*, const T*, const int*,                           \
                             const T*, T*, const int*);
#if !defined(__APPLE__) && !HPDDM_MKL
# define HPDDM_GENERATE_EXTERN_DOTC(C, T, U) U  _Complex HPDDM_F77(C ## dotc)(const int*, const T*, const int*, const T*, const int*);
#else
# define HPDDM_GENERATE_EXTERN_DOTC(C, T, U) void C ## dotc(T*, const int*, const T*, const int*, const T*, const int*);
#endif
#define HPDDM_GENERATE_EXTERN_BLAS_COMPLEX(C, T, B, U)                                                       \
U  HPDDM_F77(B ## nrm2)(const int*, const U*, const int*);                                                   \
U  HPDDM_F77(B ## C ## nrm2)(const int*, const T*, const int*);                                              \
U  HPDDM_F77(B ## dot)(const int*, const U*, const int*, const U*, const int*);                              \
HPDDM_GENERATE_EXTERN_DOTC(C, T, U)
#define HPDDM_GENERATE_EXTERN_MKL(C, T)                                                                      \
void cblas_ ## C ## gthr(const int, const T*, T*, const int*);                                               \
void cblas_ ## C ## sctr(const int, const T*, const int*, T*);

#if !defined(INTEL_MKL_VERSION)
extern "C" {
HPDDM_GENERATE_EXTERN_BLAS(s, float)
HPDDM_GENERATE_EXTERN_BLAS(d, double)
HPDDM_GENERATE_EXTERN_BLAS(c, std::complex<float>)
HPDDM_GENERATE_EXTERN_BLAS(z, std::complex<double>)
HPDDM_GENERATE_EXTERN_BLAS_COMPLEX(c, std::complex<float>, s, float)
HPDDM_GENERATE_EXTERN_BLAS_COMPLEX(z, std::complex<double>, d, double)
# if defined(__APPLE__) || HPDDM_MKL
#  if HPDDM_MKL
HPDDM_GENERATE_EXTERN_MKL(s, float)
HPDDM_GENERATE_EXTERN_MKL(d, double)
HPDDM_GENERATE_EXTERN_MKL(c, std::complex<float>)
HPDDM_GENERATE_EXTERN_MKL(z, std::complex<double>)
#  else
#   define HPDDM_PREFIX_AXPBY(func) catlas_ ## func
#  endif
#  if !defined(CBLAS_H)
#   define HPDDM_GENERATE_EXTERN_AXPBY(C, T, B, U)                                                           \
void HPDDM_PREFIX_AXPBY(B ## axpby)(const int, const U, const U*,                                            \
                                    const int, const U, U*, const int);                                      \
void HPDDM_PREFIX_AXPBY(C ## axpby)(const int, const T*, const T*,                                           \
                                    const int, const T*, T*, const int);
HPDDM_GENERATE_EXTERN_AXPBY(c, std::complex<float>, s, float)
HPDDM_GENERATE_EXTERN_AXPBY(z, std::complex<double>, d, double)
#  endif
# endif
}
#endif // INTEL_MKL_VERSION

#include <vector>
#include <algorithm>
#include <numeric>

namespace HPDDM {
/* Constants: BLAS constants
 *
 *    transa              - Untransposed operators.
 *    transb              - Transposed operators.
 *    i__0                - Zero.
 *    i__1                - One. */
static constexpr char transa =  'N';
static constexpr char transb =  'T';
static constexpr int i__0    =    0;
static constexpr int i__1    =    1;

typedef std::pair<unsigned short, std::vector<int>>  pairNeighbor; // MPI_Comm_size < MAX_UNSIGNED_SHORT
typedef std::vector<pairNeighbor>                  vectorNeighbor;
#ifdef __GNUG__
inline std::string demangle(const char* name) {
    int status;
    std::unique_ptr<char, void(*)(void*)> res { abi::__cxa_demangle(name, NULL, NULL, &status), std::free };
    return status == 0 ? res.get() : name;
}
#else
inline std::string demangle(const char* name) { return name; }
#endif // __GNUG__
#ifdef __MINGW32__
# include <sstream>
template<class T>
inline T sto(std::string s, typename std::enable_if<std::is_same<T, int>::value>::type* = nullptr) {
    return atoi(s.c_str());
}
template<class T>
inline T sto(std::string s, typename std::enable_if<std::is_same<T, double>::value || std::is_same<T, float>::value>::type* = nullptr) {
    return atof(s.c_str());
}
template<class T>
inline std::string to_string(const T& x) {
    std::ostringstream stm;
    stm << x;
    return stm.str();
}
#else
template<class T>
inline T sto(std::string s, typename std::enable_if<std::is_same<T, int>::value>::type* = nullptr) {
    return std::stoi(s);
}
template<class T>
inline T sto(std::string s, typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr) {
    return std::stof(s);
}
template<class T>
inline T sto(std::string s, typename std::enable_if<std::is_same<T, double>::value>::type* = nullptr) {
    return std::stod(s);
}
# define to_string(x) std::to_string(x)
#endif // __MINGW32__
template<class T>
using alias = T;
} // HPDDM
#if (!defined(__clang__) && defined(__GNUC__)) || (defined(__INTEL_COMPILER) && defined(__GNUC__))
# if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100) < 40900
#  define HPDDM_NO_REGEX     1
#  pragma message("Consider updating libstdc++ to a version that implements <regex> functionalities")
# endif
#endif
#include "option.hpp"
#include "enum.hpp"
#if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION < 110201
# ifdef __INTEL_COMPILER

# elif defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wc++11-compat-deprecated-writable-strings"
# elif defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wwrite-strings"
# endif
#endif
#include "wrapper.hpp"
#if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION < 110201
# ifdef __INTEL_COMPILER

# elif defined(__clang__)
#  pragma clang diagnostic pop
# elif defined(__GNUC__)
#  pragma GCC diagnostic pop
# endif
#endif
#include "matrix.hpp"
#include "dmatrix.hpp"

#if !HPDDM_MKL
# if defined(MKL_PARDISOSUB)
#  undef MKL_PARDISOSUB
#  define MUMPSSUB
# endif
# if defined(DMKL_PARDISO)
#  undef DMKL_PARDISO
#  define DMUMPS
# endif
#endif // HPDDM_MKL
#if defined(DMUMPS) || defined(MUMPSSUB)
# include "MUMPS.hpp"
#endif
#if defined(DMKL_PARDISO) || defined(MKL_PARDISOSUB)
# include "MKL_PARDISO.hpp"
#endif
#if defined(DPASTIX) || defined(PASTIXSUB)
# include "PaStiX.hpp"
#endif
#if defined(DHYPRE)
# include "Hypre.hpp"
#endif
#include "SuiteSparse.hpp"
#include "eigensolver.hpp"
#if HPDDM_BDD || HPDDM_FETI
# include "LAPACK.hpp"
#endif
#if HPDDM_SCHWARZ
# ifndef EIGENSOLVER
#  if defined(INTEL_MKL_VERSION)
#   undef HPDDM_F77
#   define HPDDM_F77(func) func ## _
#  endif
#  include "ARPACK.hpp"
# endif
#endif

#include "preconditioner.hpp"
#include "coarse_operator_impl.hpp"
#include "operator.hpp"
#include "option_impl.hpp"

#if HPDDM_SCHWARZ
# include "schwarz.hpp"
template<class K = double, char S = 'S'>
using HpSchwarz = HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, S, K>;
#endif
#if HPDDM_FETI
# include "FETI.hpp"
template<HPDDM::FetiPrcndtnr P, class K = double, char S = 'S'>
using HpFeti = HPDDM::Feti<SUBDOMAIN, COARSEOPERATOR, S, K, P>;
#endif
#if HPDDM_BDD
# include "BDD.hpp"
template<class K = double, char S = 'S'>
using HpBdd = HPDDM::Bdd<SUBDOMAIN, COARSEOPERATOR, S, K>;
#endif

#include "iterative.hpp"
#endif // _HPDDM_
