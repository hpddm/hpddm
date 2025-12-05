/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2012-10-04

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

#ifndef HPDDM_PASTIX_HPP_
#define HPDDM_PASTIX_HPP_

#define HPDDM_GENERATE_PASTIX_EXTERN(C, T) \
  int C##_cscd_redispatch(pastix_int_t, pastix_int_t *, pastix_int_t *, T *, T *, pastix_int_t, pastix_int_t *, pastix_int_t, pastix_int_t **, pastix_int_t **, T **, T **, pastix_int_t *, MPI_Comm, pastix_int_t);

extern "C" {
#include <pastix.h>
HPDDM_GENERATE_PASTIX_EXTERN(s, float)
HPDDM_GENERATE_PASTIX_EXTERN(d, double)
HPDDM_GENERATE_PASTIX_EXTERN(c, std::complex<float>)
HPDDM_GENERATE_PASTIX_EXTERN(z, std::complex<double>)
}

#define HPDDM_GENERATE_PASTIX(C, T) \
  template <> \
  struct pstx<T> { \
    static void dist(pastix_data_t **pastix_data, MPI_Comm pastix_comm, pastix_int_t n, pastix_int_t *colptr, pastix_int_t *row, T *avals, pastix_int_t *loc2glob, pastix_int_t *perm, pastix_int_t *invp, T *b, pastix_int_t rhs, pastix_int_t *iparm, double *dparm) \
    { \
      C##_dpastix(pastix_data, pastix_comm, n, colptr, row, avals, loc2glob, perm, invp, b, rhs, iparm, dparm); \
    } \
    static void seq(pastix_data_t **pastix_data, MPI_Comm pastix_comm, pastix_int_t n, pastix_int_t *colptr, pastix_int_t *row, T *avals, pastix_int_t *perm, pastix_int_t *invp, T *b, pastix_int_t rhs, pastix_int_t *iparm, double *dparm) \
    { \
      C##_pastix(pastix_data, pastix_comm, n, colptr, row, avals, perm, invp, b, rhs, iparm, dparm); \
    } \
    static int cscd_redispatch(pastix_int_t n, pastix_int_t *ia, pastix_int_t *ja, T *a, T *rhs, pastix_int_t nrhs, pastix_int_t *l2g, pastix_int_t dn, pastix_int_t **dia, pastix_int_t **dja, T **da, T **drhs, pastix_int_t *dl2g, MPI_Comm comm, pastix_int_t dof) \
    { \
      return C##_cscd_redispatch(n, ia, ja, a, rhs, nrhs, l2g, dn, dia, dja, da, drhs, dl2g, comm, dof); \
    } \
    static void         initParam(pastix_int_t *iparm, double *dparm) { C##_pastix_initParam(iparm, dparm); } \
    static pastix_int_t getLocalNodeNbr(pastix_data_t **pastix_data) { return C##_pastix_getLocalNodeNbr(pastix_data); } \
    static pastix_int_t getLocalNodeLst(pastix_data_t **pastix_data, pastix_int_t *nodelst) { return C##_pastix_getLocalNodeLst(pastix_data, nodelst); } \
    static pastix_int_t setSchurUnknownList(pastix_data_t *pastix_data, pastix_int_t n, pastix_int_t *list) { return C##_pastix_setSchurUnknownList(pastix_data, n, list); } \
    static pastix_int_t setSchurArray(pastix_data_t *pastix_data, T *array) { return C##_pastix_setSchurArray(pastix_data, array); } \
  };

namespace HPDDM
{
template <class>
struct pstx { };
HPDDM_GENERATE_PASTIX(s, float)
HPDDM_GENERATE_PASTIX(d, double)
HPDDM_GENERATE_PASTIX(c, std::complex<float>)
HPDDM_GENERATE_PASTIX(z, std::complex<double>)

#ifdef DPASTIX
  #undef HPDDM_CHECK_SUBDOMAIN
  #define HPDDM_CHECK_COARSEOPERATOR
  #include "HPDDM_preprocessor_check.hpp"
  #define COARSEOPERATOR HPDDM::Pastix
/* Class: Pastix
 *
 *  A class inheriting from <DMatrix> to use <Pastix>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template <class K>
class Pastix : public DMatrix {
private:
  /* Variable: data
         *  Internal data pointer. */
  pastix_data_t *data_;
  /* Variable: values2
         *  Array of data. */
  K *values2_;
  /* Variable: dparm
         *  Array of double-precision floating-point parameters. */
  double *dparm_;
  /* Variable: ncol2
         *  Number of local rows. */
  pastix_int_t ncol2_;
  /* Variable: colptr2
         *  Array of row pointers. */
  pastix_int_t *colptr2_;
  /* Variable: rows2
         *  Array of column indices. */
  pastix_int_t *rows2_;
  /* Variable: loc2glob2
         *  Local to global numbering. */
  pastix_int_t *loc2glob2_;
  /* Variable: iparm
         *  Array of integer parameters. */
  pastix_int_t *iparm_;

protected:
  /* Variable: numbering
         *  1-based indexing. */
  static constexpr char numbering_ = 'F';

public:
  Pastix() : data_(), values2_(), dparm_(), colptr2_(), rows2_(), loc2glob2_(), iparm_() { }
  ~Pastix()
  {
    free(rows2_);
    free(values2_);
    delete[] loc2glob2_;
    free(colptr2_);
    if (iparm_) {
      iparm_[IPARM_START_TASK] = API_TASK_CLEAN;
      iparm_[IPARM_END_TASK]   = API_TASK_CLEAN;

      pstx<K>::dist(&data_, DMatrix::communicator_, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1, iparm_, dparm_);
      delete[] iparm_;
      delete[] dparm_;
    }
  }
  /* Function: numfact
         *
         *  Initializes <Pastix::iparm> and <Pastix::dparm>, and factorizes the supplied matrix.
         *
         * Template Parameter:
         *    S              - 'S'ymmetric or 'G'eneral factorization.
         *
         * Parameters:
         *    ncol           - Number of local rows.
         *    I              - Array of row pointers.
         *    loc2glob       - Local to global numbering.
         *    J              - Array of column indices.
         *    C              - Array of data. */
  template <char S>
  void numfact(unsigned int ncol, int *I, int *loc2glob, int *J, K *C)
  {
    iparm_ = new pastix_int_t[IPARM_SIZE];
    dparm_ = new double[DPARM_SIZE];

    pstx<K>::initParam(iparm_, dparm_);
    const Option &opt = *Option::get();
    const char    val = opt.val<char>("verbosity", 0);
    if (val < 3) iparm_[IPARM_VERBOSE] = API_VERBOSE_NOT;
    else iparm_[IPARM_VERBOSE] = val - 2;
    iparm_[IPARM_MATRIX_VERIFICATION] = API_NO;
    iparm_[IPARM_START_TASK]          = API_TASK_INIT;
    iparm_[IPARM_END_TASK]            = API_TASK_INIT;
    if (S == 'S') {
      iparm_[IPARM_SYM]           = API_SYM_YES;
      iparm_[IPARM_FACTORIZATION] = opt.val<char>("operator_spd", 0) ? API_FACT_LLT : API_FACT_LDLT;
    } else {
      iparm_[IPARM_SYM]           = API_SYM_NO;
      iparm_[IPARM_FACTORIZATION] = API_FACT_LU;
      if (Wrapper<K>::is_complex) iparm_[IPARM_TRANSPOSE_SOLVE] = API_YES;
    }
    iparm_[IPARM_RHSD_CHECK] = API_NO;
    pastix_int_t *perm       = new pastix_int_t[ncol];
    pstx<K>::dist(&data_, DMatrix::communicator_, ncol, I, J, NULL, loc2glob, perm, NULL, NULL, 1, iparm_, dparm_);

    iparm_[IPARM_START_TASK] = API_TASK_ORDERING;
    iparm_[IPARM_END_TASK]   = API_TASK_ANALYSE;

    pstx<K>::dist(&data_, DMatrix::communicator_, ncol, I, J, NULL, loc2glob, perm, NULL, NULL, 1, iparm_, dparm_);
    delete[] perm;

    iparm_[IPARM_VERBOSE] = API_VERBOSE_NOT;

    ncol2_ = pstx<K>::getLocalNodeNbr(&data_);

    loc2glob2_ = new pastix_int_t[ncol2_];
    pstx<K>::getLocalNodeLst(&data_, loc2glob2_);

    pstx<K>::cscd_redispatch(ncol, I, J, C, NULL, 0, loc2glob, ncol2_, &colptr2_, &rows2_, &values2_, NULL, loc2glob2_, DMatrix::communicator_, 1);

    iparm_[IPARM_START_TASK] = API_TASK_NUMFACT;
    iparm_[IPARM_END_TASK]   = API_TASK_NUMFACT;

    pstx<K>::dist(&data_, DMatrix::communicator_, ncol2_, colptr2_, rows2_, values2_, loc2glob2_, NULL, NULL, NULL, 1, iparm_, dparm_);

    iparm_[IPARM_CSCD_CORRECT] = API_YES;
    delete[] I;
    delete[] loc2glob;
  }
  /* Function: solve
         *
         *  Solves the system in-place.
         *
         * Parameters:
         *    rhs            - Input right-hand sides, solution vectors are stored in-place.
         *    n              - Number of right-hand sides. */
  void solve(K *rhs, const unsigned short &n)
  {
    K *rhs2 = new K[n * ncol2_];
    if (!DMatrix::mapOwn_ && !DMatrix::mapRecv_) {
      int nloc = DMatrix::ldistribution_[DMatrix::rank_];
      DMatrix::initializeMap<1>(ncol2_, loc2glob2_, rhs2, rhs);
      DMatrix::ldistribution_  = new int[1];
      *DMatrix::ldistribution_ = nloc;
    } else DMatrix::redistribute<1>(rhs2, rhs);
    for (unsigned short nu = 1; nu < n; ++nu) DMatrix::redistribute<1>(rhs2 + nu * ncol2_, rhs + nu * *DMatrix::ldistribution_);

    iparm_[IPARM_START_TASK] = API_TASK_SOLVE;
    iparm_[IPARM_END_TASK]   = API_TASK_SOLVE;
    pstx<K>::dist(&data_, DMatrix::communicator_, ncol2_, colptr2_, rows2_, values2_, loc2glob2_, NULL, NULL, rhs2, n, iparm_, dparm_);

    for (unsigned short nu = 0; nu < n; ++nu) DMatrix::redistribute<2>(rhs + nu * *DMatrix::ldistribution_, rhs2 + nu * ncol2_);
    delete[] rhs2;
  }
};
#endif // DPASTIX

#ifdef PASTIXSUB
  #undef HPDDM_CHECK_COARSEOPERATOR
  #define HPDDM_CHECK_SUBDOMAIN
  #include "HPDDM_preprocessor_check.hpp"
  #define SUBDOMAIN HPDDM::PastixSub
template <class K>
class PastixSub {
private:
  pastix_data_t *data_;
  K             *values_;
  double        *dparm_;
  pastix_int_t   ncol_;
  pastix_int_t  *colptr_;
  pastix_int_t  *rows_;
  pastix_int_t  *iparm_;

public:
  PastixSub() : data_(), values_(), dparm_(), colptr_(), rows_(), iparm_() { }
  PastixSub(const PastixSub &) = delete;
  ~PastixSub() { dtor(); }
  static constexpr char numbering_ = 'F';
  void                  dtor()
  {
    if (iparm_) {
      if (iparm_[IPARM_SYM] == API_SYM_YES || iparm_[IPARM_SYM] == API_SYM_HER) {
        delete[] rows_;
        delete[] colptr_;
        delete[] values_;
      }
      iparm_[IPARM_START_TASK] = API_TASK_CLEAN;
      iparm_[IPARM_END_TASK]   = API_TASK_CLEAN;
      pstx<K>::seq(&data_, MPI_COMM_SELF, 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, iparm_, dparm_);
      delete[] iparm_;
      iparm_ = nullptr;
      delete[] dparm_;
    }
  }
  template <char N = HPDDM_NUMBERING>
  void numfact(MatrixCSR<K> *const &A, bool detection = false, K *const &schur = nullptr)
  {
    static_assert(N == 'C' || N == 'F', "Unknown numbering");
    if (!iparm_) {
      iparm_ = new pastix_int_t[IPARM_SIZE];
      dparm_ = new double[DPARM_SIZE];
      ncol_  = A->n_;
      pstx<K>::initParam(iparm_, dparm_);
      iparm_[IPARM_VERBOSE]             = API_VERBOSE_NOT;
      iparm_[IPARM_MATRIX_VERIFICATION] = API_NO;
      iparm_[IPARM_START_TASK]          = API_TASK_INIT;
      iparm_[IPARM_END_TASK]            = API_TASK_INIT;
      iparm_[IPARM_SCHUR]               = schur ? API_YES : API_NO;
      iparm_[IPARM_RHSD_CHECK]          = API_NO;
      dparm_[DPARM_EPSILON_MAGN_CTRL]   = -1.0 / HPDDM_PEN;
      if (A->sym_) {
        values_           = new K[A->nnz_];
        colptr_           = new int[ncol_ + 1];
        rows_             = new int[A->nnz_];
        iparm_[IPARM_SYM] = API_SYM_YES;
      } else {
        iparm_[IPARM_SYM]           = API_SYM_NO;
        iparm_[IPARM_FACTORIZATION] = API_FACT_LU;
      }
    }
    const MatrixCSR<K> *B = A->sym_ ? nullptr : A->template symmetrizedStructure<N, 'F'>();
    if (A->sym_) {
      iparm_[IPARM_FACTORIZATION] = (Option::get()->val<char>("operator_spd", 0) && !detection) ? API_FACT_LLT : API_FACT_LDLT;
      Wrapper<K>::template csrcsc<N, 'F'>(&ncol_, A->a_, A->ja_, A->ia_, values_, rows_, colptr_);
    } else {
      values_ = B->a_;
      colptr_ = B->ia_;
      rows_   = B->ja_;
      if (B != A) iparm_[IPARM_TRANSPOSE_SOLVE] = API_YES;
    }
    pastix_int_t *perm    = new pastix_int_t[2 * ncol_];
    pastix_int_t *iperm   = perm + ncol_;
    int          *listvar = nullptr;
    if (iparm_[IPARM_START_TASK] == API_TASK_INIT) {
      pstx<K>::seq(&data_, MPI_COMM_SELF, ncol_, colptr_, rows_, NULL, NULL, NULL, NULL, 1, iparm_, dparm_);
      if (schur) {
        listvar = new int[static_cast<int>(std::real(schur[0]))];
        std::iota(listvar, listvar + static_cast<int>(std::real(schur[0])), static_cast<int>(std::real(schur[1])));
        pstx<K>::setSchurUnknownList(data_, static_cast<int>(std::real(schur[0])), listvar);
        pstx<K>::setSchurArray(data_, schur);
      }
      iparm_[IPARM_START_TASK] = API_TASK_ORDERING;
      iparm_[IPARM_END_TASK]   = API_TASK_NUMFACT;
    } else {
      iparm_[IPARM_START_TASK] = API_TASK_NUMFACT;
      iparm_[IPARM_END_TASK]   = API_TASK_NUMFACT;
    }
    pstx<K>::seq(&data_, MPI_COMM_SELF, ncol_, colptr_, rows_, values_, perm, iperm, NULL, 1, iparm_, dparm_);
    delete[] listvar;
    delete[] perm;
    if (iparm_[IPARM_SYM] == API_SYM_NO) {
      if (B == A) {
        if (N == 'C') {
          std::for_each(colptr_, colptr_ + ncol_ + 1, [](int &i) { --i; });
          std::for_each(rows_, rows_ + A->nnz_, [](int &i) { --i; });
        }
      } else delete B;
    }
  }
  void solve(K *const x, const unsigned short &n = 1) const
  {
    iparm_[IPARM_START_TASK] = API_TASK_SOLVE;
    iparm_[IPARM_END_TASK]   = API_TASK_SOLVE;
    pstx<K>::seq(const_cast<pastix_data_t **>(&data_), MPI_COMM_SELF, ncol_, NULL, NULL, NULL, NULL, NULL, x, n, iparm_, dparm_);
  }
  void solve(const K *const b, K *const x, const unsigned short &n = 1) const
  {
    std::copy_n(b, n * ncol_, x);
    solve(x, n);
  }
};
#endif // PASTIXSUB
} // namespace HPDDM
#endif // HPDDM_PASTIX_HPP_
