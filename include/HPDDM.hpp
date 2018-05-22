/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
              Frédéric Nataf <nataf@ann.jussieu.fr>
        Date: 2013-07-14

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

#ifndef _HPDDM_
#define _HPDDM_

/* Title: HPDDM */

/* Constants: C-style preprocessor variables
 *
 *    HPDDM_VERSION       - Version of the framework.
 *    HPDDM_EPS           - Small positive number used internally for dropping values.
 *    HPDDM_PEN           - Large positive number used externally for penalization, e.g. for imposing Dirichlet boundary conditions.
 *    HPDDM_GRANULARITY   - Granularity for OpenMP scheduling.
 *    HPDDM_MPI           - If not set to zero, MPI is supposed to be activated during compilation and for running the library.
 *    HPDDM_MKL           - If not set to zero, Intel MKL is chosen as the linear algebra backend.
 *    HPDDM_NUMBERING     - 0- or 1-based indexing of user-supplied matrices.
 *    HPDDM_SCHWARZ       - Overlapping Schwarz methods enabled.
 *    HPDDM_FETI          - FETI methods enabled.
 *    HPDDM_BDD           - BDD methods enabled.
 *    HPDDM_DENSE         - Methods for dense matrices enabled. Users must provide their own matrix--vector products.
 *    HPDDM_PETSC         - PETSc KSP interface enabled.
 *    HPDDM_QR            - If not set to zero, pseudo-inverses of Schur complements are computed using dense QR decompositions (with pivoting if set to one, without pivoting otherwise).
 *    HPDDM_ICOLLECTIVE   - If possible, use nonblocking MPI collective operations.
 *    HPDDM_MIXED_PRECISION - Use mixed precision arithmetic for the assembly of coarse operators.
 *    HPDDM_INEXACT_COARSE_OPERATOR - Solve coarse systems using a Krylov method.
 *    HPDDM_LIBXSMM       - Block sparse matrices products are computed using LIBXSMM. */
#define HPDDM_VERSION            991
#define HPDDM_EPS             1.0e-12
#define HPDDM_PEN             1.0e+30
#define HPDDM_GRANULARITY     50000
#ifndef HPDDM_NUMBERING
# pragma message("The numbering of user-supplied matrices has not been set, assuming 0-based indexing")
# define HPDDM_NUMBERING      'C'
#elif defined(__cplusplus)
static_assert(HPDDM_NUMBERING == 'C' || HPDDM_NUMBERING == 'F', "Unknown numbering");
#endif
#ifndef HPDDM_MPI
# define HPDDM_MPI            1
#elif !HPDDM_MPI && defined(MPI_VERSION)
# pragma message("You cannot deactivate MPI support and still include MPI headers")
# undef HPDDM_MPI
# define HPDDM_MPI            1
#endif
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
#ifndef HPDDM_DENSE
# define HPDDM_DENSE          0
#elif HPDDM_DENSE
# if defined(HPDDM_INEXACT_COARSE_OPERATOR) && HPDDM_INEXACT_COARSE_OPERATOR
#  undef HPDDM_INEXACT_COARSE_OPERATOR
#  pragma message("HPDDM_DENSE and HPDDM_INEXACT_COARSE_OPERATOR are mutually exclusive")
# endif
# if defined(HPDDM_SCHWARZ) && !HPDDM_SCHWARZ
#  undef HPDDM_SCHWARZ
# endif
# ifndef HPDDM_SCHWARZ
#  define HPDDM_SCHWARZ       1
# endif
#endif
#ifndef HPDDM_PETSC
# define HPDDM_PETSC          0
#endif
#define HPDDM_QR              2
#ifndef HPDDM_ICOLLECTIVE
# define HPDDM_ICOLLECTIVE    0
#endif
#ifndef HPDDM_MIXED_PRECISION
# define HPDDM_MIXED_PRECISION 0
#endif
#ifndef HPDDM_INEXACT_COARSE_OPERATOR
# define HPDDM_INEXACT_COARSE_OPERATOR 0
#endif
#ifndef HPDDM_LIBXSMM
# define HPDDM_LIBXSMM        0
#endif

#ifdef _MSC_VER
# ifndef _CRT_SECURE_NO_WARNINGS
#  define _CRT_SECURE_NO_WARNINGS
# endif
# ifndef _SCL_SECURE_NO_WARNINGS
#  define _SCL_SECURE_NO_WARNINGS
# endif
#endif
#ifdef __MINGW32__
# include <inttypes.h>
#endif
#if HPDDM_MPI
# ifndef MPI_VERSION
#  include <mpi.h>
# endif
# if HPDDM_ICOLLECTIVE
#  if !((OMPI_MAJOR_VERSION > 1 || (OMPI_MAJOR_VERSION == 1 && OMPI_MINOR_VERSION >= 7)) || MPICH_NUMVERSION >= 30000000)
#   pragma message("You cannot use nonblocking MPI collective operations with that MPI implementation")
#   undef HPDDM_ICOLLECTIVE
#   define HPDDM_ICOLLECTIVE  0
#  endif
# endif // HPDDM_ICOLLECTIVE
#else
# ifdef HPDDM_SCHWARZ
#  undef HPDDM_SCHWARZ
# endif
# ifdef HPDDM_FETI
#  undef HPDDM_FETI
# endif
# ifdef HPDDM_BDD
#  undef HPDDM_BDD
# endif
# define HPDDM_SCHWARZ        0
# define HPDDM_FETI           0
# define HPDDM_BDD            0
#endif // HPDDM_MPI

#if defined(__powerpc__) || defined(INTEL_MKL_VERSION)
# define HPDDM_F77(func) func
#else
# define HPDDM_F77(func) func ## _
#endif

#ifdef _OPENMP
# include <omp.h>
#endif
#ifdef __cplusplus
# include <complex>
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
# define HPDDM_PREFIX_AXPBY(func) cblas_ ## func
#endif // HPDDM_MKL

#include "preprocessor_check.hpp"
#ifdef __cplusplus
# include <iostream>
# include <fstream>
# include <iomanip>
# include <unordered_map>
# include <limits>
# include <algorithm>
# include <vector>
# include <numeric>
# include <functional>
# include <memory>
# if !__cpp_rtti && !defined(__GXX_RTTI) && !defined(__INTEL_RTTI__) && !defined(_CPPRTTI)
#  pragma message("Consider enabling RTTI support with your C++ compiler")
# endif
static_assert(2 * sizeof(double) == sizeof(std::complex<double>) && 2 * sizeof(float) == sizeof(std::complex<float>) && 2 * sizeof(float) == sizeof(double) && sizeof(char) == 1, "Incorrect sizes");
# ifdef __GNUG__
#  include <cxxabi.h>
# endif

namespace HPDDM {
/* Constants: BLAS constants
 *
 *    i__0                - Zero.
 *    i__1                - One. */
static constexpr int i__0 = 0;
static constexpr int i__1 = 1;

typedef std::pair<unsigned short, std::vector<int>>  pairNeighbor; // MPI_Comm_size < MAX_UNSIGNED_SHORT
typedef std::vector<pairNeighbor>                  vectorNeighbor;
# ifdef __GNUG__
inline std::string demangle(const char* name) {
    int status;
    std::unique_ptr<char, void(*)(void*)> res { abi::__cxa_demangle(name, NULL, NULL, &status), std::free };
    return status == 0 ? res.get() : name;
}
# else
inline std::string demangle(const char* name) { return name; }
# endif // __GNUG__
# ifdef __MINGW32__
#  include <sstream>
template<class T>
inline T sto(const std::string& s, typename std::enable_if<std::is_same<T, int>::value>::type* = nullptr) {
    return atoi(s.c_str());
}
template<class T>
inline T sto(const std::string& s, typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value>::type* = nullptr) {
    return atof(s.c_str());
}
template<class T>
inline std::string to_string(const T& x) {
    std::ostringstream stm;
    stm << x;
    return stm.str();
}
# else
template<class T>
inline T sto(const std::string& s, typename std::enable_if<std::numeric_limits<T>::is_integer>::type* = nullptr) {
    return std::stoi(s);
}
template<class T>
inline T sto(const std::string& s, typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr) {
    return std::stof(s);
}
template<class T>
inline T sto(const std::string& s, typename std::enable_if<std::is_same<T, double>::value>::type* = nullptr) {
    return std::stod(s);
}
template<class T>
inline std::string to_string(const T& x) { return std::to_string(x); }
# endif // __MINGW32__
template<class T>
inline T sto(const std::string& s, typename std::enable_if<std::is_same<T, std::complex<float>>::value || std::is_same<T, std::complex<double>>::value>::type* = nullptr) {
    std::istringstream stm(s);
    T cplx;
    stm >> cplx;
    return cplx;
}
template<class T>
inline std::string pts(const T* const s, const unsigned int k, typename std::enable_if<!std::is_same<T, void>::value>::type* = nullptr) {
    std::ostringstream stm;
    stm << s[k];
    return stm.str();
}
template<class T>
inline std::string pts(const T* const, const unsigned int, typename std::enable_if<std::is_same<T, void>::value>::type* = nullptr) {
    return "";
}
template<class U, class V>
inline U pow(U x, V y) {
    static_assert(std::is_integral<U>::value && std::is_integral<V>::value, "Only integral types are supported, consider using std::pow(base, exp)");
    U temp;
    if(y == 0)
        return 1;
    temp = pow(x, y / 2);
    if(y % 2)
        return x * temp * temp;
    else
        return temp * temp;
}
template<class T>
using alias = T;

template<class T>
struct underlying_type_spec {
    typedef T type;
};
template<class T>
struct underlying_type_spec<std::complex<T>> {
    typedef T type;
};
template<class T>
using underlying_type = typename underlying_type_spec<T>::type;
template<class T>
using pod_type = typename std::conditional<std::is_same<underlying_type<T>, T>::value, T, void*>::type;
template<class T>
using downscaled_type = typename std::conditional<HPDDM_MIXED_PRECISION && std::is_same<underlying_type<T>, T>::value, float, typename std::conditional<HPDDM_MIXED_PRECISION, std::complex<float>, T>::type>::type;

template<class>
struct hpddm_method_id { static constexpr char value = 0; };
template<class T>
struct is_substructuring_method { static constexpr bool value = (hpddm_method_id<T>::value == 2 || hpddm_method_id<T>::value == 3); };

template<class T>
inline void hash_range(std::size_t& seed, T begin, T end) {
    std::hash<typename std::remove_const<typename std::remove_reference<decltype(*begin)>::type>::type> hasher;
    while(begin != end)
        seed ^= hasher(*begin++) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
} // HPDDM
# if (!defined(__clang__) && defined(__GNUC__)) || (defined(__INTEL_COMPILER) && defined(__GNUC__))
#  if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100) < 40900
#   define HPDDM_NO_REGEX     1
#   pragma message("Consider updating libstdc++ to a version that implements <regex> functionalities")
#  endif
# endif
# include "option.hpp"
# if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION < 110201 && !defined(__INTEL_COMPILER)
#  ifdef __clang__
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wc++11-compat-deprecated-writable-strings"
#  elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wwrite-strings"
#  endif
# endif
# include "BLAS.hpp"
# include "wrapper.hpp"
# if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION < 110201 && !defined(__INTEL_COMPILER)
#  ifdef __clang__
#   pragma clang diagnostic pop
#  elif defined(__GNUC__)
#   pragma GCC diagnostic pop
#  endif
# endif
# include "enum.hpp"
# include "matrix.hpp"
# if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
#  include "dmatrix.hpp"
#  if !HPDDM_MKL
#   ifdef MKL_PARDISOSUB
#    undef MKL_PARDISOSUB
#    define MUMPSSUB
#   endif
#   ifdef DMKL_PARDISO
#    undef DMKL_PARDISO
#    define DMUMPS
#   endif
#  endif // HPDDM_MKL
#  if defined(DMUMPS) || defined(MUMPSSUB)
#   include "MUMPS.hpp"
#  endif
#  if defined(DMKL_PARDISO) || defined(MKL_PARDISOSUB)
#   include "MKL_PARDISO.hpp"
#  endif
#  if defined(DPASTIX) || defined(PASTIXSUB)
#   include "PaStiX.hpp"
#  endif
#  if defined(DHYPRE)
#   include "Hypre.hpp"
#  endif
#  if defined(DSUITESPARSE) || defined(SUITESPARSESUB)
#   include "SuiteSparse.hpp"
#  endif
#  ifdef DISSECTIONSUB
#   include "Dissection.hpp"
#  endif
#  ifdef LAPACKSUB
#   include "LAPACK.hpp"
#  endif
#  ifdef DELEMENTAL
#   include "Elemental.hpp"
#  endif
# endif
# if !defined(SUBDOMAIN) || !defined(COARSEOPERATOR)
#  undef HPDDM_SCHWARZ
#  undef HPDDM_FETI
#  undef HPDDM_BDD
#  undef HPDDM_DENSE
#  define HPDDM_SCHWARZ       0
#  define HPDDM_FETI          0
#  define HPDDM_BDD           0
#  define HPDDM_DENSE         0
# endif
# ifndef HPDDM_MINIMAL
#  include "LAPACK.hpp"
#  if HPDDM_MPI
#   if HPDDM_SCHWARZ
#    ifndef EIGENSOLVER
#     ifdef INTEL_MKL_VERSION
#      undef HPDDM_F77
#      define HPDDM_F77(func) func ## _
#     endif
#     ifdef MU_ARPACK
#      include "ARPACK.hpp"
#     elif defined(MU_FEAST)
#      include "FEAST.hpp"
#     endif
#    endif
#   endif
#  endif

#  if !HPDDM_MPI
#   define MPI_Allreduce(a, b, c, d, e, f) (void)f
#   define MPI_Comm_size(a, b) *b = 1
#   define MPI_Comm_rank(a, b) *b = 0
#   define MPI_COMM_SELF 0
typedef int MPI_Comm;
typedef int MPI_Request;
#  endif
#  include "GCRODR.hpp"
#  include "CG.hpp"
#  if !HPDDM_MPI
#   undef MPI_COMM_SELF
#   undef MPI_Comm_rank
#   undef MPI_Comm_size
#   undef MPI_Allreduce
#  else
#   include "schwarz.hpp"
template<class K = double, char S = 'S'>
using HpSchwarz = HPDDM::Schwarz<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
    SUBDOMAIN, COARSEOPERATOR,
#endif
    S, K>;
#   include "schur.hpp"
template<class K = double>
using HpSchur = HPDDM::Schur<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
    SUBDOMAIN, COARSEOPERATOR<K>,
#endif
    K>;
#   if HPDDM_FETI
#    include "FETI.hpp"
template<HPDDM::FetiPrcndtnr P, class K = double, char S = 'S'>
using HpFeti = HPDDM::Feti<SUBDOMAIN, COARSEOPERATOR, S, K, P>;
#   endif
#   if HPDDM_BDD
#    include "BDD.hpp"
template<class K = double, char S = 'S'>
using HpBdd = HPDDM::Bdd<SUBDOMAIN, COARSEOPERATOR, S, K>;
#   endif
#   if HPDDM_DENSE
#    include "dense.hpp"
template<class K = double, char S = 'S'>
using HpDense = HPDDM::Dense<SUBDOMAIN, COARSEOPERATOR, S, K>;
#   endif
#  endif // !HPDDM_MPI
# endif // !HPDDM_MINIMAL
# include "option_impl.hpp"
#else
# include "BLAS.hpp"
# include "LAPACK.hpp"
#endif // __cplusplus
#if HPDDM_PETSC
# include "PETSc.hpp"
#endif

#endif // _HPDDM_
