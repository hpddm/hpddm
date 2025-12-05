/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2014-11-06

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

#ifndef HPDDM_SUITESPARSE_HPP_
#define HPDDM_SUITESPARSE_HPP_

#include <cholmod.h>
#include <umfpack.h>

namespace HPDDM
{
template <class K>
struct stsprs {
  static_assert(std::is_same<double, underlying_type<K>>::value, "UMFPACK only supports double-precision floating-point scalars");
};

template <>
struct stsprs<double> {
  static void umfpack_defaults(double *control) { umfpack_di_defaults(control); }
  static void umfpack_report_info(const double *control, const double *info) { umfpack_di_report_info(control, info); }
  static int  umfpack_numeric(const int *ia, const int *ja, const double *a, void *symbolic, void **numeric, const double *control, double *info) { return umfpack_di_numeric(ia, ja, a, symbolic, numeric, control, info); }
  static int  umfpack_symbolic(int n, int m, const int *ia, const int *ja, const double *a, void **symbolic, const double *control, double *info) { return umfpack_di_symbolic(n, m, ia, ja, a, symbolic, control, info); }
  static void umfpack_free_symbolic(void **symbolic) { umfpack_di_free_symbolic(symbolic); }
  static int  umfpack_wsolve(int sys, const int *ia, const int *ja, const double *a, double *X, const double *B, void *numeric, const double *control, double *info, int *Wi, double *W)
  {
    return umfpack_di_wsolve(sys, ia, ja, a, X, B, numeric, control, info, Wi, W);
  }
  static void umfpack_free_numeric(void **numeric) { umfpack_di_free_numeric(numeric); }
};

template <>
struct stsprs<std::complex<double>> {
  static void umfpack_defaults(double *control) { umfpack_zi_defaults(control); }
  static void umfpack_report_info(const double *control, const double *info) { umfpack_zi_report_info(control, info); }
  static int  umfpack_numeric(const int *ia, const int *ja, const std::complex<double> *a, void *symbolic, void **numeric, const double *control, double *info)
  {
    return umfpack_zi_numeric(ia, ja, reinterpret_cast<const double *>(a), NULL, symbolic, numeric, control, info);
  }
  static int umfpack_symbolic(int n, int m, const int *ia, const int *ja, const std::complex<double> *a, void **symbolic, const double *control, double *info)
  {
    return umfpack_zi_symbolic(n, m, ia, ja, reinterpret_cast<const double *>(a), NULL, symbolic, control, info);
  }
  static void umfpack_free_symbolic(void **symbolic) { umfpack_zi_free_symbolic(symbolic); }
  static int  umfpack_wsolve(int sys, const int *ia, const int *ja, const std::complex<double> *a, std::complex<double> *X, const std::complex<double> *B, void *numeric, const double *control, double *info, int *Wi, std::complex<double> *W)
  {
    return umfpack_zi_wsolve(sys, ia, ja, reinterpret_cast<const double *>(a), NULL, reinterpret_cast<double *>(X), NULL, reinterpret_cast<const double *>(B), NULL, numeric, control, info, Wi, reinterpret_cast<double *>(W));
  }
  static void umfpack_free_numeric(void **numeric) { umfpack_zi_free_numeric(numeric); }
};

#ifdef DSUITESPARSE
  #undef HPDDM_CHECK_SUBDOMAIN
  #define HPDDM_CHECK_COARSEOPERATOR
  #include "HPDDM_preprocessor_check.hpp"
  #define COARSEOPERATOR HPDDM::SuiteSparse
/* Class: SuiteSparse
 *
 *  A class inheriting from <DMatrix> to use <SuiteSparse>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template <class K>
class SuiteSparse : public DMatrix {
private:
  /* Variable: L
         *  Factors returned by CHOLMOD. */
  cholmod_factor *L_;
  /* Variable: c
         *  Parameters, statistics, and workspace of CHOLMOD. */
  cholmod_common *c_;
  /* Variable: b
         *  Right-hand side matrix of CHOLMOD. */
  cholmod_dense *b_;
  /* Variable: x
         *  Solution matrix of CHOLMOD. */
  cholmod_dense *x_;
  /* Variable: Y
         *  Dense workspace matrix of CHOLMOD. */
  cholmod_dense *Y_;
  /* Variable: E
         *  Dense workspace matrix of CHOLMOD. */
  cholmod_dense *E_;
  /* Variable: numeric
         *  Opaque object for the numerical factorization of UMFPACK. */
  void *numeric_;
  /* Variable: control
         *  Array of double parameters. */
  double *control_;
  /* Variable: pattern
         *  Workspace integer array of UMFPACK. */
  int *pattern_;
  /* Variable: W
         *  Workspace double array of UMFPACK. */
  K *W_;
  /* Variable: tmp
         *  Workspace array. */
  K *tmp_;

protected:
  /* Variable: numbering
         *  0-based indexing. */
  static constexpr char numbering_ = 'C';

public:
  SuiteSparse() : L_(), c_(), b_(), x_(), Y_(), E_(), numeric_(), control_(), pattern_(), W_(), tmp_() { }
  ~SuiteSparse()
  {
    delete[] tmp_;
    W_ = nullptr;
    if (c_) {
      cholmod_free_factor(&L_, c_);
      cholmod_free(1, sizeof(cholmod_dense), b_, c_);
      cholmod_free(1, sizeof(cholmod_dense), x_, c_);
      cholmod_free_dense(&Y_, c_);
      cholmod_free_dense(&E_, c_);
      cholmod_finish(c_);
      delete c_;
    } else {
      delete[] pattern_;
      delete[] control_;
      stsprs<K>::umfpack_free_numeric(&numeric_);
    }
  }
  template <char S>
  void numfact(unsigned int ncol, int *I, int *J, K *C)
  {
    if (S == 'S') {
      c_ = new cholmod_common;
      cholmod_start(c_);
      c_->print         = 3;
      cholmod_sparse *M = static_cast<cholmod_sparse *>(cholmod_malloc(1, sizeof(cholmod_sparse), c_));
      M->nrow           = ncol;
      M->ncol           = ncol;
      M->nzmax          = I[ncol];
      M->sorted         = 1;
      M->packed         = 1;
      M->stype          = -1;
      M->xtype          = Wrapper<K>::is_complex ? CHOLMOD_COMPLEX : CHOLMOD_REAL;
      M->p              = I;
      M->i              = J;
      M->x              = C;
      M->dtype          = std::is_same<double, underlying_type<K>>::value ? CHOLMOD_DOUBLE : CHOLMOD_SINGLE;
      M->itype          = CHOLMOD_INT;
      L_                = cholmod_analyze(M, c_);
      if (Option::get()->val<char>("verbosity", 0) > 2) cholmod_print_common(NULL, c_);
      cholmod_factorize(M, L_, c_);
      b_        = static_cast<cholmod_dense *>(cholmod_malloc(1, sizeof(cholmod_dense), c_));
      b_->nrow  = M->nrow;
      b_->xtype = M->xtype;
      b_->dtype = M->dtype;
      b_->d     = b_->nrow;
      tmp_      = new K[b_->nrow];
      x_        = static_cast<cholmod_dense *>(cholmod_malloc(1, sizeof(cholmod_dense), c_));
      x_->nrow  = M->nrow;
      x_->x     = NULL;
      x_->xtype = M->xtype;
      x_->dtype = M->dtype;
      x_->d     = x_->nrow;
      cholmod_free(1, sizeof(cholmod_sparse), M, c_);
    } else {
      control_ = new double[UMFPACK_CONTROL];
      stsprs<K>::umfpack_defaults(control_);
      control_[UMFPACK_PRL]    = 2;
      control_[UMFPACK_IRSTEP] = 0;
      double *info             = new double[UMFPACK_INFO];
      pattern_                 = new int[ncol];
      tmp_                     = new K[6 * ncol];
      W_                       = tmp_ + ncol;
      numeric_                 = NULL;

      void *symbolic;
      stsprs<K>::umfpack_symbolic(ncol, ncol, I, J, C, &symbolic, control_, info);
      stsprs<K>::umfpack_numeric(I, J, C, symbolic, &numeric_, control_, info);
      if (Option::get()->val<char>("verbosity", 0) > 2) stsprs<K>::umfpack_report_info(control_, info);
      stsprs<K>::umfpack_free_symbolic(&symbolic);
      delete[] info;
    }
    delete[] I;
  }
  template <bool>
  void solve(K *rhs, const unsigned short &n = 1)
  {
    for (unsigned short nu = 0; nu < n; ++nu) {
      if (c_) {
        b_->ncol  = 1;
        b_->nzmax = x_->nrow;
        b_->x     = rhs + nu * DMatrix::n_;
        x_->ncol  = 1;
        x_->nzmax = x_->nrow;
        x_->x     = tmp_;
        cholmod_solve2(CHOLMOD_A, L_, b_, NULL, &x_, NULL, &Y_, &E_, c_);
      } else stsprs<K>::umfpack_wsolve(UMFPACK_Aat, NULL, NULL, NULL, tmp_, rhs + nu * DMatrix::n_, numeric_, control_, NULL, pattern_, W_);
      std::copy_n(tmp_, DMatrix::n_, rhs + nu * DMatrix::n_);
    }
  }
};
#endif // DSUITESPARSE

#ifdef SUITESPARSESUB
  #undef HPDDM_CHECK_COARSEOPERATOR
  #define HPDDM_CHECK_SUBDOMAIN
  #include "HPDDM_preprocessor_check.hpp"
  #define SUBDOMAIN HPDDM::SuiteSparseSub
template <class K>
class SuiteSparseSub {
private:
  cholmod_factor *L_;
  cholmod_common *c_;
  cholmod_dense  *b_;
  cholmod_dense  *x_;
  cholmod_dense  *Y_;
  cholmod_dense  *E_;
  void           *numeric_;
  double         *control_;
  int            *pattern_;
  K              *W_;
  K              *tmp_;

public:
  SuiteSparseSub() : L_(), c_(), b_(), x_(), Y_(), E_(), numeric_(), control_(), pattern_(), W_(), tmp_() { }
  SuiteSparseSub(const SuiteSparseSub &) = delete;
  ~SuiteSparseSub() { dtor(); }
  static constexpr char numbering_ = 'C';
  void                  dtor()
  {
    delete[] tmp_;
    tmp_ = nullptr;
    if (c_) {
      cholmod_free_factor(&L_, c_);
      cholmod_free(1, sizeof(cholmod_dense), b_, c_);
      cholmod_free(1, sizeof(cholmod_dense), x_, c_);
      cholmod_free_dense(&Y_, c_);
      cholmod_free_dense(&E_, c_);
      cholmod_finish(c_);
      delete c_;
      c_ = nullptr;
    } else if (control_) {
      delete[] pattern_;
      delete[] control_;
      control_ = nullptr;
      stsprs<K>::umfpack_free_numeric(&numeric_);
    }
  }
  template <char N = HPDDM_NUMBERING>
  void numfact(MatrixCSR<K> *const &A, bool detection = false)
  {
    static_assert(N == 'C', "Unsupported numbering");
    if (Option::get()->val<char>("operator_spd", 0) && A->sym_) {
      if (!c_) {
        c_ = new cholmod_common;
        cholmod_start(c_);
      }
      cholmod_sparse *M = static_cast<cholmod_sparse *>(cholmod_malloc(1, sizeof(cholmod_sparse), c_));
      M->nrow           = A->m_;
      M->ncol           = A->n_;
      M->nzmax          = A->nnz_;
      M->sorted         = 1;
      M->packed         = 1;
      M->stype          = 1;
      M->xtype          = Wrapper<K>::is_complex ? CHOLMOD_COMPLEX : CHOLMOD_REAL;
      M->p              = A->ia_;
      M->i              = A->ja_;
      M->x              = A->a_;
      M->dtype          = std::is_same<double, underlying_type<K>>::value ? CHOLMOD_DOUBLE : CHOLMOD_SINGLE;
      M->itype          = CHOLMOD_INT;
      if (L_) cholmod_free_factor(&L_, c_);
      L_ = cholmod_analyze(M, c_);
      cholmod_factorize(M, L_, c_);
      if (!b_) {
        b_        = static_cast<cholmod_dense *>(cholmod_malloc(1, sizeof(cholmod_dense), c_));
        b_->nrow  = M->nrow;
        b_->xtype = M->xtype;
        b_->dtype = M->dtype;
        b_->d     = b_->nrow;
        tmp_      = new K[b_->nrow];
        x_        = static_cast<cholmod_dense *>(cholmod_malloc(1, sizeof(cholmod_dense), c_));
        x_->nrow  = M->nrow;
        x_->x     = NULL;
        x_->xtype = M->xtype;
        x_->dtype = M->dtype;
        x_->d     = x_->nrow;
      }
      cholmod_free(1, sizeof(cholmod_sparse), M, c_);
    } else {
      if (!control_) {
        control_ = new double[UMFPACK_CONTROL];
        stsprs<K>::umfpack_defaults(control_);
        control_[UMFPACK_PRL]    = 0;
        control_[UMFPACK_IRSTEP] = 0;
        pattern_                 = new int[A->m_];
        tmp_                     = new K[6 * A->m_];
        W_                       = tmp_ + A->m_;
      }
      double *info     = new double[UMFPACK_INFO];
      void   *symbolic = NULL;
      K      *a;
      int    *ia;
      int    *ja;
      if (!A->sym_) {
        a  = A->a_;
        ia = A->ia_;
        ja = A->ja_;
      } else {
        std::vector<std::vector<std::pair<unsigned int, K>>> v(A->n_);
        unsigned int                                         nnz = ((A->nnz_ + A->n_ - 1) / A->n_) * 2;
        for (unsigned int i = 0; i < A->n_; ++i) v[i].reserve(nnz);
        nnz = 0;
        for (unsigned int i = 0; i < A->n_; ++i) {
          for (unsigned int j = A->ia_[i]; j < A->ia_[i + 1] - 1; ++j) {
            if (std::abs(A->a_[j]) > HPDDM_EPS) {
              v[i].emplace_back(A->ja_[j], A->a_[j]);
              v[A->ja_[j]].emplace_back(i, A->a_[j]);
              nnz += 2;
            }
          }
          v[i].emplace_back(i, A->a_[A->ia_[i + 1] - 1]);
          ++nnz;
        }
        ja  = new int[A->n_ + 1 + nnz];
        ia  = ja + nnz;
        a   = new K[nnz];
        nnz = 0;
        unsigned int i;
  #ifdef _OPENMP
    #pragma omp parallel for schedule(static, HPDDM_GRANULARITY)
  #endif
        for (i = 0; i < A->n_; ++i) std::sort(v[i].begin(), v[i].end(), [](const std::pair<unsigned int, K> &lhs, const std::pair<unsigned int, K> &rhs) { return lhs.first < rhs.first; });
        ia[0] = 0;
        for (i = 0; i < A->n_; ++i) {
          for (const std::pair<unsigned int, K> &p : v[i]) {
            ja[nnz]  = p.first;
            a[nnz++] = p.second;
          }
          ia[i + 1] = nnz;
        }
      }
      stsprs<K>::umfpack_symbolic(A->m_, A->n_, ia, ja, a, &symbolic, control_, info);
      if (numeric_) {
        stsprs<K>::umfpack_free_numeric(&numeric_);
        numeric_ = NULL;
      }
      stsprs<K>::umfpack_numeric(ia, ja, a, symbolic, &numeric_, control_, info);
      stsprs<K>::umfpack_report_info(control_, info);
      stsprs<K>::umfpack_free_symbolic(&symbolic);
      if (A->sym_) {
        delete[] ja;
        delete[] a;
      }
      delete[] info;
    }
  }
  void solve(K *const x) const
  {
    if (c_) {
      b_->ncol  = 1;
      b_->nzmax = x_->nrow;
      b_->x     = x;
      x_->ncol  = 1;
      x_->nzmax = x_->nrow;
      x_->x     = tmp_;
      cholmod_solve2(CHOLMOD_A, L_, b_, NULL, const_cast<cholmod_dense **>(&x_), NULL, const_cast<cholmod_dense **>(&Y_), const_cast<cholmod_dense **>(&E_), c_);
      std::copy_n(tmp_, x_->nrow, x);
    } else {
      stsprs<K>::umfpack_wsolve(UMFPACK_Aat, NULL, NULL, NULL, tmp_, x, numeric_, control_, NULL, pattern_, W_);
      std::copy(tmp_, W_, x);
    }
  }
  void solve(K *const x, const unsigned short &n) const
  {
    if (c_) {
      b_->ncol  = n;
      b_->nzmax = x_->nrow;
      b_->x     = x;
      x_->ncol  = n;
      x_->nzmax = x_->nrow;
      x_->x     = new K[n * x_->nrow];
      cholmod_solve2(CHOLMOD_A, L_, b_, NULL, const_cast<cholmod_dense **>(&x_), NULL, const_cast<cholmod_dense **>(&Y_), const_cast<cholmod_dense **>(&E_), c_);
      std::copy_n(static_cast<K *>(x_->x), n * x_->nrow, x);
      delete[] static_cast<K *>(x_->x);
      x_->x = NULL;
    } else {
      int ld = std::distance(tmp_, W_);
      for (unsigned short i = 0; i < n; ++i) {
        stsprs<K>::umfpack_wsolve(UMFPACK_Aat, NULL, NULL, NULL, tmp_, x + i * ld, numeric_, control_, NULL, pattern_, W_);
        std::copy(tmp_, W_, x + i * ld);
      }
    }
  }
  void solve(const K *const b, K *const x, const unsigned short &n = 1) const
  {
    if (c_) {
      b_->ncol  = n;
      b_->nzmax = x_->nrow;
      b_->x     = const_cast<K *>(b);
      x_->ncol  = n;
      x_->nzmax = x_->nrow;
      x_->x     = x;
      cholmod_solve2(CHOLMOD_A, L_, b_, NULL, const_cast<cholmod_dense **>(&x_), NULL, const_cast<cholmod_dense **>(&Y_), const_cast<cholmod_dense **>(&E_), c_);
    } else {
      int ld = std::distance(tmp_, W_);
      for (unsigned short i = 0; i < n; ++i) stsprs<K>::umfpack_wsolve(UMFPACK_Aat, NULL, NULL, NULL, x + i * ld, b + i * ld, numeric_, control_, NULL, pattern_, W_);
    }
  }
};
#endif // SUITESPARSESUB
} // namespace HPDDM
#endif // HPDDM_SUITESPARSE_HPP_
