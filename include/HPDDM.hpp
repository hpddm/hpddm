/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
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

#pragma once

/* Title: HPDDM */

#include "HPDDM_define.hpp"

#ifdef __clang__
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wshadow"
  #pragma clang diagnostic ignored "-Wsign-compare"
#elif defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wshadow"
  #pragma GCC diagnostic ignored "-Wsign-compare"
#endif

#ifdef _MSC_VER
  #ifndef _CRT_SECURE_NO_WARNINGS
    #define _CRT_SECURE_NO_WARNINGS
  #endif
  #ifndef _SCL_SECURE_NO_WARNINGS
    #define _SCL_SECURE_NO_WARNINGS
  #endif
#endif
#ifdef __MINGW32__
  #include <inttypes.h>
#endif
#if HPDDM_MPI
  #if !defined(MPI_VERSION) && !defined(PETSC_HAVE_MPIUNI)
    #include <mpi.h>
  #endif
  #if HPDDM_ICOLLECTIVE
    #if !((OMPI_MAJOR_VERSION > 1 || (OMPI_MAJOR_VERSION == 1 && OMPI_MINOR_VERSION >= 7)) || MPICH_NUMVERSION >= 30000000)
      #pragma message("You cannot use nonblocking MPI collective operations with that MPI implementation")
      #undef HPDDM_ICOLLECTIVE
      #define HPDDM_ICOLLECTIVE 0
    #endif
  #endif // HPDDM_ICOLLECTIVE
#else
  #ifdef HPDDM_SCHWARZ
    #undef HPDDM_SCHWARZ
  #endif
  #ifdef HPDDM_FETI
    #undef HPDDM_FETI
  #endif
  #ifdef HPDDM_BDD
    #undef HPDDM_BDD
  #endif
  #define HPDDM_SCHWARZ 0
  #define HPDDM_FETI    0
  #define HPDDM_BDD     0
#endif // HPDDM_MPI

#if (defined(__powerpc__) && !defined(PETSC_BLASLAPACK_UNDERSCORE) && !defined(Add_)) || defined(INTEL_MKL_VERSION)
  #define HPDDM_F77(func) func
#else
  #define HPDDM_F77(func) func##_
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif
#ifdef __cplusplus
  #include <complex>
#endif
#ifndef MKL_Complex16
  #define MKL_Complex16 std::complex<double>
#endif
#ifndef MKL_Complex8
  #define MKL_Complex8 std::complex<float>
#endif
#ifndef MKL_INT
  #define MKL_INT int
#endif
#if HPDDM_MKL
  #if defined(__cplusplus) && (!defined(INTEL_MKL_VERSION) || INTEL_MKL_VERSION >= 20200004)
    #define HPDDM_NOEXCEPT noexcept
  #endif
  #define HPDDM_PREFIX_AXPBY(func) cblas_##func
#endif // HPDDM_MKL
#if !HPDDM_MKL || !defined(__cplusplus) || (defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION < 20200004)
  #define HPDDM_NOEXCEPT
#endif

#include "HPDDM_preprocessor_check.hpp"
#ifdef __cplusplus
  #include <iostream>
  #include <fstream>
  #include <iomanip>
  #include <unordered_map>
  #include <limits>
  #include <algorithm>
  #include <vector>
  #include <numeric>
  #include <functional>
  #include <memory>
  #include <set>
  #if !__cpp_rtti && !defined(__GXX_RTTI) && !defined(__INTEL_RTTI__) && !defined(_CPPRTTI)
    #pragma message("Consider enabling RTTI support with your C++ compiler")
  #endif
static_assert(2 * sizeof(double) == sizeof(std::complex<double>) && 2 * sizeof(float) == sizeof(std::complex<float>) && 2 * sizeof(float) == sizeof(double) && sizeof(char) == 1, "Unsupported scalar type");
  #ifdef __GNUG__
    #include <cxxabi.h>
  #endif
  #define HPDDM_HAS_MEMBER(member) \
    template <class T> \
    class has_##member { \
    private: \
      typedef char one; \
      typedef one (&two)[2]; \
      template <class C> \
      static one test(decltype(&C::member)); \
      template <class C> \
      static two test(...); \
\
    public: \
      static constexpr bool value = (sizeof(test<T>(nullptr)) == sizeof(one)); \
    };

namespace HPDDM
{
/* Constants: BLAS constants
 *
 *    i__0                - Zero.
 *    i__1                - One. */
static constexpr int i__0 = 0;
static constexpr int i__1 = 1;

typedef std::pair<unsigned short, std::vector<int>> pairNeighbor; // MPI_Comm_size < MAX_UNSIGNED_SHORT
typedef std::vector<pairNeighbor>                   vectorNeighbor;
template <class... T>
inline void ignore(const T &...)
{
}
  #ifdef __GNUG__
inline std::string demangle(const char *name)
{
  int                                     status;
  std::unique_ptr<char, void (*)(void *)> res{abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
  return status == 0 ? res.get() : name;
}
  #else
inline std::string demangle(const char *name)
{
  return name;
}
  #endif // __GNUG__
  #ifdef __MINGW32__
    #include <sstream>
template <class T>
inline T sto(const std::string &s, typename std::enable_if<std::is_same<T, int>::value>::type * = nullptr)
{
  return atoi(s.c_str());
}
template <class T>
inline T sto(const std::string &s, typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value>::type * = nullptr)
{
  return atof(s.c_str());
}
template <class T>
inline std::string to_string(const T &x)
{
  std::ostringstream stm;
  stm << x;
  return stm.str();
}
  #else
template <class T>
inline T sto(const std::string &s, typename std::enable_if<std::numeric_limits<T>::is_integer>::type * = nullptr)
{
  return std::stoi(s);
}
template <class T>
inline T sto(const std::string &s, typename std::enable_if<std::is_same<T, float>::value>::type * = nullptr)
{
  return std::stof(s);
}
template <class T>
inline T sto(const std::string &s, typename std::enable_if<std::is_same<T, double>::value>::type * = nullptr)
{
  return std::stod(s);
}
template <class T>
inline std::string to_string(const T &x)
{
  return std::to_string(x);
}
  #endif // __MINGW32__
template <class T>
inline T sto(const std::string &s, typename std::enable_if<std::is_same<T, std::complex<float>>::value || std::is_same<T, std::complex<double>>::value>::type * = nullptr)
{
  std::istringstream stm(s);
  T                  cplx;
  stm >> cplx;
  return cplx;
}
template <class U, class V>
inline U pow(U x, V y)
{
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value, "Only integral types are supported, consider using std::pow(base, exp)");
  U temp;
  if (y == 0) return 1;
  temp = pow(x, y / 2);
  if (y % 2) return x * temp * temp;
  else return temp * temp;
}
template <class T>
using alias = T;

template <class T>
struct underlying_type_spec {
  typedef T type;
};
template <class T>
struct underlying_type_spec<std::complex<T>> {
  typedef T type;
};
template <class T>
using underlying_type = typename underlying_type_spec<T>::type;
template <class T>
struct complex_spec {
  typedef std::complex<T> type;
};
template <class T>
using complex = typename complex_spec<T>::type;
template <class T>
using pod_type = typename std::conditional<std::is_same<underlying_type<T>, T>::value, T, void *>::type;
} // namespace HPDDM

  #if !defined(PETSC_HAVE_REAL___FLOAT128) || defined(PETSC_SKIP_REAL___FLOAT128) || defined(__NVCC__) || defined(__CUDACC__)
namespace HPDDM
{
template <class T>
inline underlying_type<T> norm(const T &v)
{
  return std::norm(v);
}
template <class T>
inline underlying_type<T> abs(const T &v)
{
  return std::abs(v);
}
template <class T>
inline underlying_type<T> real(const T &v)
{
  return std::real(v);
}
template <class T>
inline underlying_type<T> imag(const T &v)
{
  return std::imag(v);
}
} // namespace HPDDM
  #endif
  #if defined(PETSC_HAVE_REAL___FP16) && !defined(PETSC_SKIP_REAL___FP16)
namespace HPDDM
{
template <class T>
using downscaled_type = typename std::conditional<std::is_same<underlying_type<T>, T>::value, typename std::conditional<HPDDM_MIXED_PRECISION && std::is_same<T, double>::value, float, typename std::conditional<HPDDM_MIXED_PRECISION && std::is_same<T, float>::value, __fp16, T>::type>::type, typename std::conditional<HPDDM_MIXED_PRECISION && std::is_same<T, std::complex<double>>::value, std::complex<float>, typename std::conditional<HPDDM_MIXED_PRECISION && std::is_same<T, std::complex<float>>::value, std::complex<__fp16>, T>::type>::type>::type;
} // namespace HPDDM
    #if !defined(PETSC_HAVE_REAL___FLOAT128) || defined(PETSC_SKIP_REAL___FLOAT128)
      #include "HPDDM_specifications.hpp"
    #endif
  #else
namespace HPDDM
{
template <class T>
using downscaled_type = typename std::conditional<std::is_same<underlying_type<T>, T>::value, typename std::conditional<HPDDM_MIXED_PRECISION && std::is_same<T, double>::value, float, T>::type, typename std::conditional<HPDDM_MIXED_PRECISION && std::is_same<T, std::complex<double>>::value, std::complex<float>, T>::type>::type;
} // namespace HPDDM
  #endif
  #if !defined(PETSC_HAVE_REAL___FLOAT128) || defined(PETSC_SKIP_REAL___FLOAT128) || defined(__NVCC__) || defined(__CUDACC__)
namespace HPDDM
{
template <class T>
using upscaled_type = typename std::conditional<std::is_same<underlying_type<T>, T>::value, double, std::complex<double>>::type;
template <class T>
inline T sqrt(const T &v)
{
  return std::sqrt(v);
}
template <class T>
inline T conj(const T &v)
{
  return std::conj(v);
}
template <class T>
inline T hypot(const T &x, const T &y)
{
  return std::hypot(x, y);
}
template <class T>
inline int lround(const T &v)
{
  return std::lround(v);
}
template <class InputIt, class Size, class OutputIt>
inline OutputIt copy_n(InputIt in, Size n, OutputIt out)
{
  return std::copy_n(in, n, out);
}
} // namespace HPDDM
  #else
    #include <quadmath.h>
namespace HPDDM
{
template <>
struct underlying_type_spec<__complex128> {
  typedef __float128 type;
};
template <>
struct complex_spec<__float128> {
  typedef __complex128 type;
};
template <class T>
using upscaled_type = typename std::conditional<std::is_same<underlying_type<T>, T>::value, typename std::conditional<std::is_same<T, float>::value, double, __float128>::type, typename std::conditional<std::is_same<T, std::complex<float>>::value, std::complex<double>, __complex128>::type>::type;
template <class T, typename std::enable_if<!std::is_same<T, __float128>::value && !std::is_same<T, __complex128>::value>::type * = nullptr>
inline underlying_type<T> norm(const T &v)
{
  return std::norm(v);
}
template <class T, typename std::enable_if<std::is_same<T, __float128>::value>::type * = nullptr>
inline underlying_type<T> norm(const T &v)
{
  return v * v;
}
template <class T, typename std::enable_if<std::is_same<T, __complex128>::value>::type * = nullptr>
inline underlying_type<T> norm(const T &v)
{
  return crealq(v * conjq(v));
}
template <class T, typename std::enable_if<!std::is_same<T, __float128>::value && !std::is_same<T, __complex128>::value>::type * = nullptr>
inline underlying_type<T> abs(const T &v)
{
  return std::abs(v);
}
template <class T, typename std::enable_if<std::is_same<T, __float128>::value>::type * = nullptr>
inline underlying_type<T> abs(const T &v)
{
  return fabsq(v);
}
template <class T, typename std::enable_if<std::is_same<T, __complex128>::value>::type * = nullptr>
inline underlying_type<T> abs(const T &v)
{
  return cabsq(v);
}
template <class T, typename std::enable_if<!std::is_same<T, __float128>::value && !std::is_same<T, __complex128>::value>::type * = nullptr>
inline underlying_type<T> real(const T &v)
{
  return std::real(v);
}
template <class T, typename std::enable_if<std::is_same<T, __float128>::value>::type * = nullptr>
inline underlying_type<T> real(const T &v)
{
  return v;
}
template <class T, typename std::enable_if<std::is_same<T, __complex128>::value>::type * = nullptr>
inline underlying_type<T> real(const T &v)
{
  return crealq(v);
}
template <class T, typename std::enable_if<!std::is_same<T, __complex128>::value>::type * = nullptr>
inline underlying_type<T> imag(const T &v)
{
  return std::imag(v);
}
template <class T, typename std::enable_if<std::is_same<T, __complex128>::value>::type * = nullptr>
inline underlying_type<T> imag(const T &v)
{
  return cimagq(v);
}
template <class T, typename std::enable_if<!std::is_same<T, __float128>::value && !std::is_same<T, __complex128>::value>::type * = nullptr>
inline T sqrt(const T &v)
{
  return std::sqrt(v);
}
template <class T, typename std::enable_if<std::is_same<T, __float128>::value>::type * = nullptr>
inline T sqrt(const T &v)
{
  return sqrtq(v);
}
template <class T, typename std::enable_if<std::is_same<T, __complex128>::value>::type * = nullptr>
inline T sqrt(const T &v)
{
  return csqrtq(v);
}
template <class T, typename std::enable_if<!std::is_same<T, __complex128>::value>::type * = nullptr>
inline T conj(const T &v)
{
  return std::conj(v);
}
template <class T, typename std::enable_if<std::is_same<T, __complex128>::value>::type * = nullptr>
inline T conj(const T &v)
{
  return conjq(v);
}
template <class T, typename std::enable_if<!std::is_same<T, __float128>::value>::type * = nullptr>
inline T hypot(const T &x, const T &y)
{
  return std::hypot(x, y);
}
template <class T, typename std::enable_if<std::is_same<T, __float128>::value>::type * = nullptr>
inline T hypot(const T &x, const T &y)
{
  return hypotq(x, y);
}
template <class T, typename std::enable_if<!std::is_same<T, __float128>::value>::type * = nullptr>
inline long lround(const T &v)
{
  return std::lround(v);
}
template <class T, typename std::enable_if<std::is_same<T, __float128>::value>::type * = nullptr>
inline long lround(const T &v)
{
  return lroundq(v);
}
template <class InputIt, class Size, class OutputIt, typename std::enable_if<!std::is_same<OutputIt, __complex128 *>::value || std::is_same<typename std::remove_const<typename std::remove_pointer<InputIt>::type>::type, typename std::remove_const<typename std::remove_pointer<OutputIt>::type>::type>::value>::type * = nullptr>
inline void copy_n(InputIt in, Size n, OutputIt out)
{
  std::copy_n(in, n, out);
}
template <class InputIt, class Size, class OutputIt, typename std::enable_if<std::is_same<OutputIt, __complex128 *>::value && !std::is_same<typename std::remove_const<typename std::remove_pointer<InputIt>::type>::type, typename std::remove_const<typename std::remove_pointer<OutputIt>::type>::type>::value>::type * = nullptr>
inline void copy_n(InputIt in, Size n, OutputIt out)
{
  for (Size i = 0; i < n; ++i) {
    __real__ out[i] = in[i].real();
    __imag__ out[i] = in[i].imag();
  }
}
} // namespace HPDDM
    #include "HPDDM_specifications.hpp"
  #endif

namespace HPDDM
{
template <class T>
inline std::string pts(const T *const s, const unsigned int k, typename std::enable_if<!std::is_same<T, void>::value>::type * = nullptr)
{
  std::ostringstream stm;
  stm << std::scientific << std::setprecision(std::is_same<underlying_type<T>, float>::value ? 22 : 44);
  stm << s[k];
  return stm.str();
}
template <class T>
inline std::string pts(const T *const, const unsigned int, typename std::enable_if<std::is_same<T, void>::value>::type * = nullptr)
{
  return "";
}

template <class>
struct hpddm_method_id {
  static constexpr char value = 0;
};
template <class T>
struct is_substructuring_method {
  static constexpr bool value = (hpddm_method_id<T>::value == 2 || hpddm_method_id<T>::value == 3);
};

template <class T>
inline void hash_range(std::size_t &seed, T begin, T end)
{
  std::hash<typename std::remove_const<typename std::remove_reference<decltype(*begin)>::type>::type> hasher;
  while (begin != end) seed ^= hasher(*begin++) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
} // namespace HPDDM
  #if (!defined(__clang__) && defined(__GNUC__)) || (defined(__INTEL_COMPILER) && defined(__GNUC__))
    #if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100) < 40900
      #define HPDDM_NO_REGEX 1
      #pragma message("Consider updating libstdc++ to a version that implements <regex> functionalities")
    #endif
  #endif
  #include "HPDDM_option.hpp"
  #if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION < 110201 && !defined(__INTEL_COMPILER)
    #ifdef __clang__
      #pragma clang diagnostic push
      #pragma clang diagnostic ignored "-Wc++11-compat-deprecated-writable-strings"
    #elif defined(__GNUC__)
      #pragma GCC diagnostic push
      #pragma GCC diagnostic ignored "-Wwrite-strings"
    #endif
  #endif
  #include "HPDDM_BLAS.hpp"
  #include "HPDDM_wrapper.hpp"
  #if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION < 110201 && !defined(__INTEL_COMPILER)
    #ifdef __clang__
      #pragma clang diagnostic pop
    #elif defined(__GNUC__)
      #pragma GCC diagnostic pop
    #endif
  #endif
  #include "HPDDM_enum.hpp"
  #include "HPDDM_matrix.hpp"
  #if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
    #include "HPDDM_dmatrix.hpp"
    #if !HPDDM_MKL
      #ifdef MKL_PARDISOSUB
        #undef MKL_PARDISOSUB
        #define MUMPSSUB
      #endif
      #ifdef DMKL_PARDISO
        #undef DMKL_PARDISO
        #define DMUMPS
      #endif
    #endif // HPDDM_MKL
    #if defined(DMUMPS) || defined(MUMPSSUB)
      #include "HPDDM_MUMPS.hpp"
    #endif
    #if defined(DMKL_PARDISO) || defined(MKL_PARDISOSUB)
      #include "HPDDM_MKL_PARDISO.hpp"
    #endif
    #if defined(DPASTIX) || defined(PASTIXSUB)
      #include "HPDDM_PaStiX.hpp"
    #endif
    #if defined(DHYPRE)
      #include "HPDDM_hypre.hpp"
    #endif
    #if defined(DSUITESPARSE) || defined(SUITESPARSESUB)
      #include "HPDDM_SuiteSparse.hpp"
    #endif
    #ifdef DISSECTIONSUB
      #include "HPDDM_Dissection.hpp"
    #endif
    #if defined(DLAPACK) || defined(LAPACKSUB)
      #include "HPDDM_LAPACK.hpp"
    #endif
    #ifdef DELEMENTAL
      #include "HPDDM_Elemental.hpp"
    #endif
    #ifdef PETSCSUB
      #include "HPDDM_PETSc.hpp"
    #endif
  #endif
  #if !defined(SUBDOMAIN) || !defined(COARSEOPERATOR)
    #undef HPDDM_SCHWARZ
    #undef HPDDM_FETI
    #undef HPDDM_BDD
    #undef HPDDM_DENSE
    #define HPDDM_SCHWARZ 0
    #define HPDDM_FETI    0
    #define HPDDM_BDD     0
    #define HPDDM_DENSE   0
  #endif
  #ifndef HPDDM_MINIMAL
    #include "HPDDM_LAPACK.hpp"
    #if HPDDM_MPI
      #if HPDDM_SCHWARZ
        #ifndef EIGENSOLVER
          #ifdef INTEL_MKL_VERSION
            #undef HPDDM_F77
            #define HPDDM_F77(func) func##_
          #endif
          #ifdef MU_ARPACK
            #include "HPDDM_ARPACK.hpp"
          #elif defined(MU_SLEPC)
            #include "HPDDM_SLEPc.hpp"
          #endif
        #endif
      #endif
    #endif

    #if !HPDDM_MPI
      #define MPI_Allreduce(a, b, c, d, e, f) (void)f
      #define MPI_Comm_size(a, b)             *b = 1
      #define MPI_Comm_rank(a, b)             *b = 0
      #define MPI_COMM_SELF                   0
typedef int MPI_Comm;
typedef int MPI_Request;
    #endif
    #if !HPDDM_PETSC || defined(PETSC_PCHPDDM_MAXLEVELS)
      #include "HPDDM_GCRODR.hpp"
      #include "HPDDM_CG.hpp"
    #endif
    #if !HPDDM_MPI
      #undef MPI_COMM_SELF
      #undef MPI_Comm_rank
      #undef MPI_Comm_size
      #undef MPI_Allreduce
    #else
      #include "HPDDM_schwarz.hpp"
template <class K = double, char S = 'S'>
using HpSchwarz = HPDDM::Schwarz<
      #if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
  SUBDOMAIN, COARSEOPERATOR, S,
      #endif
  K>;
      #include "HPDDM_schur.hpp"
template <class K = double>
using HpSchur = HPDDM::Schur<
      #if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
  SUBDOMAIN, COARSEOPERATOR<K>,
      #endif
  K>;
      #if HPDDM_FETI
        #include "HPDDM_FETI.hpp"
template <HPDDM::FetiPrcndtnr P, class K = double, char S = 'S'>
using HpFeti = HPDDM::Feti<SUBDOMAIN, COARSEOPERATOR, S, K, P>;
      #endif
      #if HPDDM_BDD
        #include "HPDDM_BDD.hpp"
template <class K = double, char S = 'S'>
using HpBdd = HPDDM::Bdd<SUBDOMAIN, COARSEOPERATOR, S, K>;
      #endif
      #if HPDDM_DENSE
        #include "HPDDM_dense.hpp"
template <class K = double, char S = 'S'>
using HpDense = HPDDM::Dense<SUBDOMAIN, COARSEOPERATOR, S, K>;
      #endif
    #endif // !HPDDM_MPI
  #endif   // !HPDDM_MINIMAL
  #include "HPDDM_option_impl.hpp"
#else
  #include "HPDDM_BLAS.hpp"
  #include "HPDDM_LAPACK.hpp"
#endif // __cplusplus
#if HPDDM_PETSC
  #ifdef HPDDM_MINIMAL
    #include "HPDDM_iterative.hpp"
  #endif
  #include "HPDDM_PETSc.hpp"
#endif

#ifdef __clang__
  #pragma clang diagnostic pop
#elif defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif
