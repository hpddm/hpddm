/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2012-10-07

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

#ifndef HPDDM_MKL_PARDISO_HPP_
#define HPDDM_MKL_PARDISO_HPP_

#ifdef DMKL_PARDISO
  #include <mkl_cluster_sparse_solver.h>
#endif
#ifdef MKL_PARDISOSUB
  #include <mkl_pardiso.h>
#endif

namespace HPDDM
{
template <class K>
struct prds {
  static constexpr int SPD = !Wrapper<K>::is_complex ? 2 : 4;
  static constexpr int SYM = !Wrapper<K>::is_complex ? -2 : 6;
  static constexpr int SSY = !Wrapper<K>::is_complex ? 1 : 3;
  static constexpr int UNS = !Wrapper<K>::is_complex ? 11 : 13;
};

#ifdef DMKL_PARDISO
  #undef HPDDM_CHECK_SUBDOMAIN
  #define HPDDM_CHECK_COARSEOPERATOR
  #include "HPDDM_preprocessor_check.hpp"
  #define COARSEOPERATOR HPDDM::MklPardiso
/* Class: MKL Pardiso
 *
 *  A class inheriting from <DMatrix> to use <MKL Pardiso>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template <class K>
class MklPardiso : public DMatrix {
private:
  /* Variable: pt
         *  Internal data pointer. */
  void *pt_[64];
  /* Variable: a
         *  Array of data. */
  K *C_;
  /* Variable: I
         *  Array of row pointers. */
  int *I_;
  /* Variable: J
         *  Array of column indices. */
  int *J_;
  #if !HPDDM_INEXACT_COARSE_OPERATOR
  /* Variable: w
         *  Workspace array. */
  K *w_;
  #endif
  /* Variable: mtype
         *  Matrix type. */
  int mtype_;
  /* Variable: iparm
         *  Array of parameters. */
  mutable int iparm_[64];
  /* Variable: comm
         *  MPI communicator. */
  int comm_;

protected:
  /* Variable: numbering
         *  0-based indexing. */
  static constexpr char numbering_ = 'F';

public:
  MklPardiso() :
    pt_(),
    C_(),
    I_(),
    J_(),
  #if !HPDDM_INEXACT_COARSE_OPERATOR
    w_(),
  #endif
    mtype_(),
    iparm_(),
    comm_(-1)
  {
  }
  ~MklPardiso()
  {
  #if !HPDDM_INEXACT_COARSE_OPERATOR
    delete[] w_;
  #endif
    int phase = -1;
    int error;
    K   ddum;
    int idum;
    if (comm_ != -1) {
      int i__0 = 0;
      int i__1 = 1;
      CLUSTER_SPARSE_SOLVER(pt_, &i__1, &i__1, &mtype_, &phase, &(DMatrix::n_), &ddum, &idum, &idum, &i__1, &i__1, iparm_, &i__0, &ddum, &ddum, const_cast<int *>(&comm_), &error);
      comm_ = -1;
    }
    delete[] I_;
    delete[] C_;
  }
  /* Function: numfact
         *
         *  Initializes <MKL Pardiso::pt> and <MKL Pardiso::iparm>, and factorizes the supplied matrix.
         *
         * Template Parameter:
         *    S              - 'S'ymmetric or 'G'eneral factorization.
         *
         * Parameters:
         *    I              - Array of row pointers.
         *    loc2glob       - Lower and upper bounds of the local domain.
         *    J              - Array of column indices.
         *    C              - Array of data. */
  template <char S>
  void numfact(unsigned short bs, int *I, int *loc2glob, int *J, K *&C)
  {
    if (DMatrix::communicator_ != MPI_COMM_NULL && comm_ == -1) comm_ = MPI_Comm_c2f(DMatrix::communicator_);
    I_                = I;
    J_                = J;
    C_                = C;
    const Option &opt = *Option::get();
    if (S == 'S') mtype_ = opt.val<char>("operator_spd", 0) ? prds<K>::SPD : prds<K>::SYM;
    else mtype_ = prds<K>::SSY;
    int phase, error;
    K   ddum;
    std::fill_n(iparm_, 64, 0);
    iparm_[0] = 1;
    iparm_[1] = opt.val<int>("mkl_pardiso_iparm_2", 2);
  #if !HPDDM_INEXACT_COARSE_OPERATOR
    iparm_[5] = 1;
  #else
    iparm_[5] = 0;
  #endif
    iparm_[9]  = opt.val<int>("mkl_pardiso_iparm_10", S != 'S' ? 13 : 8);
    iparm_[10] = opt.val<int>("mkl_pardiso_iparm_11", S != 'S' ? 1 : 0);
    iparm_[12] = opt.val<int>("mkl_pardiso_iparm_13", S != 'S' ? 1 : 0);
    iparm_[20] = opt.val<int>("mkl_pardiso_iparm_21", 1);
    iparm_[26] = opt.val<int>("mkl_pardiso_iparm_27", 0);
    iparm_[27] = std::is_same<double, underlying_type<K>>::value ? 0 : 1;
    iparm_[34] = (numbering_ == 'C');
    iparm_[36] = bs;
    iparm_[39] = 2;
    iparm_[40] = loc2glob[0];
    iparm_[41] = loc2glob[1];
    phase      = 12;
    *loc2glob  = DMatrix::n_ / bs;

    int i__0 = 0;
    int i__1 = 1;
    CLUSTER_SPARSE_SOLVER(pt_, &i__1, &i__1, &mtype_, &phase, loc2glob, C, I_, J_, &i__1, &i__1, iparm_, opt.val<char>("verbosity", 0) < 3 ? &i__0 : &i__1, &ddum, &ddum, const_cast<int *>(&comm_), &error);
  #if !HPDDM_INEXACT_COARSE_OPERATOR
    w_ = new K[(iparm_[41] - iparm_[40] + 1) * bs];
  #endif
    C = nullptr;
    delete[] loc2glob;
  }
    /* Function: solve
         *
         *  Solves the system in-place.
         *
         * Parameters:
         *    rhs            - Input right-hand sides, solution vectors are stored in-place.
         *    n              - Number of right-hand sides. */
  #if !HPDDM_INEXACT_COARSE_OPERATOR
  void solve(K *rhs, const unsigned short &n) const {
  #else
  void solve(const K *const rhs, K *const x, const unsigned short &n) const
  {
  #endif
    int error;
  int phase = 33;
  int nrhs  = n;
  int i__0  = 0;
  int i__1  = 1;
  #if !HPDDM_INEXACT_COARSE_OPERATOR
  if (n != 1) {
    delete[] w_;
    K **ptr = const_cast<K **>(&w_);
    *ptr    = new K[(iparm_[41] - iparm_[40] + 1) * iparm_[36] * n];
  }
  CLUSTER_SPARSE_SOLVER(const_cast<void **>(pt_), &i__1, &i__1, &mtype_, &phase, &(DMatrix::n_), C_, I_, J_, &i__1, &nrhs, iparm_, &i__0, rhs, w_, const_cast<int *>(&comm_), &error);
  #else
    CLUSTER_SPARSE_SOLVER(const_cast<void **>(pt_), &i__1, &i__1, &mtype_, &phase, &(DMatrix::n_), C_, I_, J_, &i__1, &nrhs, iparm_, &i__0, const_cast<K *>(rhs), x, const_cast<int *>(&comm_), &error);
  #endif
}
};
#endif // DMKL_PARDISO

#ifdef MKL_PARDISOSUB
  #undef HPDDM_CHECK_COARSEOPERATOR
  #define HPDDM_CHECK_SUBDOMAIN
  #include "HPDDM_preprocessor_check.hpp"
  #define SUBDOMAIN HPDDM::MklPardisoSub
template <class K>
class MklPardisoSub {
private:
  void       *pt_[64];
  K          *C_;
  int        *I_;
  int        *J_;
  K          *w_;
  int         mtype_;
  mutable int iparm_[64];
  int         n_;
  int         partial_;

public:
  MklPardisoSub() : pt_(), C_(), I_(), J_(), w_(), mtype_(), iparm_(), n_(), partial_() { }
  MklPardisoSub(const MklPardisoSub &) = delete;
  ~MklPardisoSub() { dtor(); }
  static constexpr char numbering_ = 'F';
  void                  dtor()
  {
    delete[] w_;
    w_        = nullptr;
    int phase = -1;
    int error;
    int idum;
    K   ddum;
    n_ = 1;
    PARDISO(pt_, const_cast<int *>(&i__1), const_cast<int *>(&i__1), &mtype_, &phase, &n_, &ddum, &idum, &idum, const_cast<int *>(&i__1), const_cast<int *>(&i__1), iparm_, const_cast<int *>(&i__0), &ddum, &ddum, &error);
    if (mtype_ == prds<K>::SPD || mtype_ == prds<K>::SYM) {
      delete[] I_;
      delete[] J_;
      I_ = nullptr;
      J_ = nullptr;
    }
    if (mtype_ == prds<K>::SYM) {
      delete[] C_;
      C_ = nullptr;
    }
  }
  template <char N = HPDDM_NUMBERING>
  void numfact(MatrixCSR<K> *const &A, bool detection = false, K *const &schur = nullptr)
  {
    static_assert(N == 'C' || N == 'F', "Unknown numbering");
    int          *perm = nullptr;
    int           phase, error;
    K             ddum;
    const Option &opt = *Option::get();
    if (!w_) {
      n_ = A->n_;
      std::fill_n(iparm_, 64, 0);
      iparm_[0]  = 1;
      iparm_[1]  = opt.val<int>("mkl_pardiso_iparm_2", 2);
      iparm_[5]  = 1;
      iparm_[9]  = opt.val<int>("mkl_pardiso_iparm_10", !A->sym_ ? 13 : 8);
      iparm_[10] = opt.val<int>("mkl_pardiso_iparm_11", !A->sym_ ? 1 : 0);
      iparm_[12] = opt.val<int>("mkl_pardiso_iparm_13", !A->sym_ ? 1 : 0);
      iparm_[20] = opt.val<int>("mkl_pardiso_iparm_21", 1);
      iparm_[23] = opt.val<int>("mkl_pardiso_iparm_24", 0);
      iparm_[24] = opt.val<int>("mkl_pardiso_iparm_25", 0);
      iparm_[26] = opt.val<int>("mkl_pardiso_iparm_27", 0);
      iparm_[27] = std::is_same<double, underlying_type<K>>::value ? 0 : 1;
      iparm_[34] = (N == 'C');
      phase      = 12;
      if (A->sym_) {
        I_ = new int[n_ + 1];
        J_ = new int[A->nnz_];
        C_ = new K[A->nnz_];
      } else mtype_ = A->template structurallySymmetric<N>() ? prds<K>::SSY : prds<K>::UNS;
      if (schur) {
        iparm_[35] = 2;
        perm       = new int[n_];
        partial_   = static_cast<int>(std::real(schur[1]));
        std::fill_n(perm, partial_, 0);
        std::fill(perm + partial_, perm + n_, 1);
      }
      w_ = new K[n_];
    } else {
      if (mtype_ == prds<K>::SPD) C_ = new K[A->nnz_];
      phase = 22;
    }
    if (A->sym_) {
      mtype_ = (opt.val<char>("operator_spd", 0) && !detection) ? prds<K>::SPD : prds<K>::SYM;
      Wrapper<K>::template csrcsc<N, N>(&n_, A->a_, A->ja_, A->ia_, C_, J_, I_);
    } else {
      I_ = A->ia_;
      J_ = A->ja_;
      C_ = A->a_;
    }
    PARDISO(pt_, const_cast<int *>(&i__1), const_cast<int *>(&i__1), &mtype_, &phase, const_cast<int *>(&n_), C_, I_, J_, perm, const_cast<int *>(&i__1), iparm_, opt.val<char>("verbosity", 0) >= 4 ? const_cast<int *>(&i__1) : const_cast<int *>(&i__0), &ddum, schur, &error);
    delete[] perm;
    if (mtype_ == prds<K>::SPD) delete[] C_;
  }
  template <char N = HPDDM_NUMBERING>
  int inertia(MatrixCSR<K> *const &A)
  {
    numfact<N>(A, true);
    return iparm_[22];
  }
  void solve(K *x) const
  {
    int error;
    iparm_[5] = 1;
    if (!partial_) {
      int phase = 33;
      PARDISO(const_cast<void **>(pt_), const_cast<int *>(&i__1), const_cast<int *>(&i__1), const_cast<int *>(&mtype_), &phase, const_cast<int *>(&n_), C_, I_, J_, const_cast<int *>(&i__1), const_cast<int *>(&i__1), iparm_, const_cast<int *>(&i__0), x, const_cast<K *>(w_), &error);
    } else {
      int phase = 331;
      PARDISO(const_cast<void **>(pt_), const_cast<int *>(&i__1), const_cast<int *>(&i__1), const_cast<int *>(&mtype_), &phase, const_cast<int *>(&n_), C_, I_, J_, const_cast<int *>(&i__1), const_cast<int *>(&i__1), iparm_, const_cast<int *>(&i__0), x, const_cast<K *>(w_), &error);
      std::fill(x + partial_, x + n_, K());
      phase = 333;
      PARDISO(const_cast<void **>(pt_), const_cast<int *>(&i__1), const_cast<int *>(&i__1), const_cast<int *>(&mtype_), &phase, const_cast<int *>(&n_), C_, I_, J_, const_cast<int *>(&i__1), const_cast<int *>(&i__1), iparm_, const_cast<int *>(&i__0), x, const_cast<K *>(w_), &error);
    }
  }
  void solve(const K *const b, K *const x) const
  {
    int error;
    if (!partial_) {
      iparm_[5] = 0;
      int phase = 33;
      PARDISO(const_cast<void **>(pt_), const_cast<int *>(&i__1), const_cast<int *>(&i__1), const_cast<int *>(&mtype_), &phase, const_cast<int *>(&n_), C_, I_, J_, const_cast<int *>(&i__1), const_cast<int *>(&i__1), iparm_, const_cast<int *>(&i__0), const_cast<K *>(b), x, &error);
    } else {
      iparm_[5] = 1;
      int phase = 331;
      std::copy_n(b, partial_, x);
      PARDISO(const_cast<void **>(pt_), const_cast<int *>(&i__1), const_cast<int *>(&i__1), const_cast<int *>(&mtype_), &phase, const_cast<int *>(&n_), C_, I_, J_, const_cast<int *>(&i__1), const_cast<int *>(&i__1), iparm_, const_cast<int *>(&i__0), x, const_cast<K *>(w_), &error);
      std::fill(x + partial_, x + n_, K());
      phase = 333;
      PARDISO(const_cast<void **>(pt_), const_cast<int *>(&i__1), const_cast<int *>(&i__1), const_cast<int *>(&mtype_), &phase, const_cast<int *>(&n_), C_, I_, J_, const_cast<int *>(&i__1), const_cast<int *>(&i__1), iparm_, const_cast<int *>(&i__0), x, const_cast<K *>(w_), &error);
    }
  }
  void solve(K *const x, const unsigned short &n) const
  {
    int error;
    int phase = 33;
    int nrhs  = n;
    iparm_[5] = 1;
    K *w      = new K[n_ * n];
    PARDISO(const_cast<void **>(pt_), const_cast<int *>(&i__1), const_cast<int *>(&i__1), const_cast<int *>(&mtype_), &phase, const_cast<int *>(&n_), C_, I_, J_, const_cast<int *>(&i__1), &nrhs, iparm_, const_cast<int *>(&i__0), x, w, &error);
    delete[] w;
  }
  void solve(const K *const b, K *const x, const unsigned short &n) const
  {
    int error;
    int phase = 33;
    int nrhs  = n;
    iparm_[5] = 0;
    PARDISO(const_cast<void **>(pt_), const_cast<int *>(&i__1), const_cast<int *>(&i__1), const_cast<int *>(&mtype_), &phase, const_cast<int *>(&n_), C_, I_, J_, const_cast<int *>(&i__1), &nrhs, iparm_, const_cast<int *>(&i__0), const_cast<K *>(b), x, &error);
  }
};
#endif // MKL_PARDISOSUB
} // namespace HPDDM
#endif // HPDDM_MKL_PARDISO_HPP_
