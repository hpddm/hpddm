/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2013-06-03

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

#include "HPDDM_schur.hpp"

namespace HPDDM
{
/* Class: Feti
 *
 *  A class for solving problems using the FETI method.
 *
 * Template Parameters:
 *    Solver         - Solver used for the factorization of local matrices.
 *    CoarseOperator - Class of the coarse operator.
 *    S              - 'S'ymmetric or 'G'eneral coarse operator.
 *    K              - Scalar type. */
template <template <class> class Solver, template <class> class CoarseSolver, char S, class K, FetiPrcndtnr P>
class Feti : public Schur<Solver, CoarseOperator<CoarseSolver, S, K>, K> {
private:
  /* Variable: primal
         *  Storage for local primal unknowns. */
  K *primal_;
  /* Variable: dual
         *  Storage for local dual unknowns. */
  K **dual_;
  /* Variable: m
         *  Local partition of unity. */
  underlying_type<K> **m_;
  /* Function: A
         *
         *  Jump operator.
         *
         * Template Parameters:
         *    trans          - 'T' if the transposed jump operator should be applied, 'N' otherwise.
         *    scale          - True if the unknowns should be scale by <Feti::m>, false otherwise.
         *
         * Parameters:
         *    primal         - Primal unknowns.
         *    dual           - Dual unknowns. */
  template <char trans, bool scale>
  void A(K *const primal, K *const *const dual) const
  {
    static_assert(trans == 'T' || trans == 'N', "Unsupported value for argument 'trans'");
    if (trans == 'T') {
      std::fill_n(primal, Subdomain<K>::dof_, K());
      for (unsigned short i = 0; i < super::signed_; ++i)
        for (unsigned int j = 0; j < Subdomain<K>::map_[i].second.size(); ++j) primal[Subdomain<K>::map_[i].second[j]] -= scale ? m_[i][j] * dual[i][j] : dual[i][j];
      for (unsigned short i = super::signed_; i < Subdomain<K>::map_.size(); ++i)
        for (unsigned int j = 0; j < Subdomain<K>::map_[i].second.size(); ++j) primal[Subdomain<K>::map_[i].second[j]] += scale ? m_[i][j] * dual[i][j] : dual[i][j];
    } else {
      for (unsigned short i = 0; i < super::signed_; ++i) {
        MPI_Irecv(Subdomain<K>::buff_[i], Subdomain<K>::map_[i].second.size(), Wrapper<K>::mpi_type(), Subdomain<K>::map_[i].first, 0, Subdomain<K>::communicator_, Subdomain<K>::rq_ + i);
        for (unsigned int j = 0; j < Subdomain<K>::map_[i].second.size(); ++j) dual[i][j] = -(scale ? m_[i][j] * primal[Subdomain<K>::map_[i].second[j]] : primal[Subdomain<K>::map_[i].second[j]]);
        MPI_Isend(dual[i], Subdomain<K>::map_[i].second.size(), Wrapper<K>::mpi_type(), Subdomain<K>::map_[i].first, 0, Subdomain<K>::communicator_, Subdomain<K>::rq_ + Subdomain<K>::map_.size() + i);
      }
      for (unsigned short i = super::signed_; i < Subdomain<K>::map_.size(); ++i) {
        MPI_Irecv(Subdomain<K>::buff_[i], Subdomain<K>::map_[i].second.size(), Wrapper<K>::mpi_type(), Subdomain<K>::map_[i].first, 0, Subdomain<K>::communicator_, Subdomain<K>::rq_ + i);
        for (unsigned int j = 0; j < Subdomain<K>::map_[i].second.size(); ++j) dual[i][j] = (scale ? m_[i][j] * primal[Subdomain<K>::map_[i].second[j]] : primal[Subdomain<K>::map_[i].second[j]]);
        MPI_Isend(dual[i], Subdomain<K>::map_[i].second.size(), Wrapper<K>::mpi_type(), Subdomain<K>::map_[i].first, 0, Subdomain<K>::communicator_, Subdomain<K>::rq_ + Subdomain<K>::map_.size() + i);
      }
      MPI_Waitall(2 * Subdomain<K>::map_.size(), Subdomain<K>::rq_, MPI_STATUSES_IGNORE);
      Blas<K>::axpy(&(super::mult_), &(Wrapper<K>::d__1), Subdomain<K>::buff_[0], &i__1, *dual, &i__1);
    }
  }
  template <class U, typename std::enable_if<!Wrapper<U>::is_complex>::type * = nullptr>
  void allocate(U **&dual, underlying_type<U> **&m)
  {
    static_assert(std::is_same<U, K>::value, "Wrong types");
    dual = new U *[2 * Subdomain<K>::map_.size()];
    m    = dual + Subdomain<K>::map_.size();
  }
  template <class U, typename std::enable_if<Wrapper<U>::is_complex>::type * = nullptr>
  void allocate(U **&dual, underlying_type<U> **&m)
  {
    static_assert(std::is_same<U, K>::value, "Wrong types");
    dual = new U *[Subdomain<K>::map_.size()];
    m    = new underlying_type<U> *[Subdomain<K>::map_.size()];
  }

public:
  Feti() : primal_(), dual_(), m_() { }
  ~Feti()
  {
    delete[] super::schur_;
    super::schur_ = nullptr;
    if (m_) delete[] *m_;
    if (Wrapper<K>::is_complex) delete[] m_;
    delete[] dual_;
  }
  /* Typedef: super
         *  Type of the immediate parent class <Schur>. */
  typedef Schur<Solver, CoarseOperator<CoarseSolver, S, K>, K> super;
  /* Function: initialize
         *  Allocates <Feti::primal>, <Feti::dual>, and <Feti::m> and calls <Schur::initialize>. */
  void initialize()
  {
    super::template initialize<true>();
    primal_ = super::structure_ + super::bi_->m_;
    allocate(dual_, m_);
    *dual_ = super::work_;
    *m_    = new underlying_type<K>[super::mult_];
    for (unsigned short i = 1; i < Subdomain<K>::map_.size(); ++i) {
      dual_[i] = dual_[i - 1] + Subdomain<K>::map_[i - 1].second.size();
      m_[i]    = m_[i - 1] + Subdomain<K>::map_[i - 1].second.size();
    }
  }
  /* Function: start
         *
         *  Projected Conjugate Gradient initialization.
         *
         * Template Parameter:
         *    excluded       - True if the main processes are excluded from the domain decomposition, false otherwise.
         *
         * Parameters:
         *    f              - Right-hand side.
         *    x              - Solution vector.
         *    b              - Condensed right-hand side.
         *    r              - First residual. */
  template <bool excluded>
  bool start(const K *const f, K *const x, K *const *const l, K *const *const r) const
  {
    bool       allocate = Subdomain<K>::setBuffer();
    Solver<K> *p        = static_cast<Solver<K> *>(super::pinv_);
    if (super::co_) {
      super::start();
      if (!excluded) {
        if (super::ev_) {
          if (super::schur_) {
            super::condensateEffort(f, nullptr);
            Blas<K>::gemv(&(Wrapper<K>::transc), &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::dof_), super::structure_ + super::bi_->m_, &i__1, &(Wrapper<K>::d__0), super::uc_, &i__1); //     uc_ = R_b g
            super::co_->template callSolver<excluded>(super::uc_);                                                                                                                                  //     uc_ = (G Q G^T) \ R_b g
            Blas<K>::gemv("N", &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::dof_), super::uc_, &i__1, &(Wrapper<K>::d__0), primal_, &i__1); // primal_ = R_b (G Q G^T) \ R f
          } else {
            Blas<K>::gemv(&(Wrapper<K>::transc), &(Subdomain<K>::a_->n_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::a_->n_), f, &i__1, &(Wrapper<K>::d__0), super::uc_, &i__1);    //     uc_ = R f
            super::co_->template callSolver<excluded>(super::uc_);                                                                                                                                                     //     uc_ = (G Q G^T) \ R f
            Blas<K>::gemv("N", &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_ + super::bi_->m_, &(Subdomain<K>::a_->n_), super::uc_, &i__1, &(Wrapper<K>::d__0), primal_, &i__1); // primal_ = R_b (G Q G^T) \ R f
          }
        } else {
          super::co_->template callSolver<excluded>(super::uc_);
          std::fill_n(primal_, Subdomain<K>::dof_, K());
        }
        A<'N', 0>(primal_, l); //       l = A R_b (G Q G^T) \ R f
        precond(l);            //       l = Q A R_b (G Q G^T) \ R f
        A<'T', 0>(primal_, l); // primal_ = A^T Q A R_b (G Q G^T) \ R f
        std::fill_n(super::structure_, super::bi_->m_, K());
        p->solve(super::structure_); // primal_ = S \ A^T Q A R_b (G Q G^T) \ R f
      } else super::co_->template callSolver<excluded>(super::uc_);
    }
    if (!excluded) {
      p->solve(f, x); //       x = S \ f
      if (!super::co_) {
        A<'N', 0>(x + super::bi_->m_, r);   //       r = A S \ f
        std::fill_n(*l, super::mult_, K()); //       l = 0
      } else {
        Blas<K>::axpby(Subdomain<K>::dof_, 1.0, x + super::bi_->m_, 1, -1.0, primal_, 1); // primal_ = S \ (f - A^T Q A R_b (G Q G^T) \ R f)
        A<'N', 0>(primal_, r);                                                            //       r = A S \ (f - A^T Q A R_b (G Q G^T) \ R f)
        project<excluded, 'T'>(r);                                                        //       r = P^T r
      }
    } else if (super::co_) project<excluded, 'T'>(r);
    return allocate;
  }
  /* Function: allocateSingle
         *
         *  Allocates a single Lagrange multiplier.
         *
         * Parameter:
         *    mult           - Reference to a Lagrange multiplier. */
  void allocateSingle(K **&mult) const
  {
    mult  = new K *[Subdomain<K>::map_.size()];
    *mult = new K[super::mult_];
    for (unsigned short i = 1; i < Subdomain<K>::map_.size(); ++i) mult[i] = mult[i - 1] + Subdomain<K>::map_[i - 1].second.size();
  }
  /* Function: allocateArray
         *
         *  Allocates an array of multiple Lagrange multipliers.
         *
         * Template Parameter:
         *    N              - Size of the array.
         *
         * Parameter:
         *    array          - Reference to an array of Lagrange multipliers. */
  template <unsigned short N>
  void allocateArray(K **(&array)[N]) const
  {
    *array  = new K *[N * Subdomain<K>::map_.size()];
    **array = new K[N * super::mult_];
    for (unsigned short i = 0; i < N; ++i) {
      array[i]  = *array + i * Subdomain<K>::map_.size();
      *array[i] = **array + i * super::mult_;
      for (unsigned short j = 1; j < Subdomain<K>::map_.size(); ++j) array[i][j] = array[i][j - 1] + Subdomain<K>::map_[j - 1].second.size();
    }
  }
  /* Function: buildScaling
         *
         *  Builds the local partition of unity <Feti::m>.
         *
         * Parameters:
         *    scaling        - Type of scaling (multiplicity, stiffness or coefficient scaling).
         *    rho            - Physical local coefficients (optional). */
  template <class T>
  void buildScaling(T &scaling, const K *const &rho = nullptr)
  {
    initialize();
    std::vector<std::pair<unsigned short, unsigned int>> *array = new std::vector<std::pair<unsigned short, unsigned int>>[Subdomain<K>::dof_];
    for (const pairNeighbor &neighbor : Subdomain<K>::map_)
      for (unsigned int j = 0; j < neighbor.second.size(); ++j) array[neighbor.second[j]].emplace_back(neighbor.first, j);
    if ((scaling == 2 && rho) || scaling == 1) {
      if (scaling == 1) super::stiffnessScaling(primal_);
      else std::copy_n(rho + super::bi_->m_, Subdomain<K>::dof_, primal_);
      bool allocate = Subdomain<K>::setBuffer(super::work_, super::mult_ + super::bi_->m_);
      Subdomain<K>::exchange(primal_);
      for (unsigned short i = 0; i < Subdomain<K>::map_.size(); ++i)
        for (unsigned int j = 0; j < Subdomain<K>::map_[i].second.size(); ++j) m_[i][j] = std::real(Subdomain<K>::buff_[i][j] / primal_[Subdomain<K>::map_[i].second[j]]);
      Subdomain<K>::clearBuffer(allocate);
    } else {
      scaling = 0;
      for (unsigned short i = 0; i < Subdomain<K>::map_.size(); ++i)
        for (unsigned int j = 0; j < Subdomain<K>::map_[i].second.size(); ++j) m_[i][j] = 1.0 / (1.0 + array[Subdomain<K>::map_[i].second[j]].size());
    }
    delete[] array;
  }
  /* Function: apply
         *
         *  Applies the global FETI operator.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector (optional). */
  void apply(K *const *const in, K *const *const out = nullptr) const
  {
    A<'T', 0>(primal_, in);
    std::fill_n(super::structure_, super::bi_->m_, K());
    static_cast<Solver<K> *>(super::pinv_)->solve(super::structure_);
    A<'N', 0>(primal_, out ? out : in);
  }
  /* Function: applyLocalPreconditioner(n)
         *
         *  Applies the local preconditioner to multiple right-hand sides.
         *
         * Template Parameter:
         *    q              - Type of <FetiPrcndtnr> to apply.
         *
         * Parameters:
         *    u              - Input vectors.
         *    n              - Number of input vectors. */
  template <FetiPrcndtnr q = P>
  void applyLocalPreconditioner(K *&u, unsigned short n) const
  {
    switch (q) {
    case FetiPrcndtnr::DIRICHLET:
      super::applyLocalSchurComplement(u, n);
      break;
    case FetiPrcndtnr::LUMPED:
      super::applyLocalLumpedMatrix(u, n);
      break;
    case FetiPrcndtnr::SUPERLUMPED:
      super::applyLocalSuperlumpedMatrix(u, n);
      break;
    case FetiPrcndtnr::NONE:
      break;
    }
  }
  /* Function: applyLocalPreconditioner
         *
         *  Applies the local preconditioner to a single right-hand side.
         *
         * Template Parameter:
         *    q              - Type of <FetiPrcndtnr> to apply.
         *
         * Parameter:
         *    u              - Input vector. */
  template <FetiPrcndtnr q = P>
  void applyLocalPreconditioner(K *const u) const
  {
    switch (q) {
    case FetiPrcndtnr::DIRICHLET:
      super::applyLocalSchurComplement(u);
      break;
    case FetiPrcndtnr::LUMPED:
      super::applyLocalLumpedMatrix(u);
      break;
    case FetiPrcndtnr::SUPERLUMPED:
      super::applyLocalSuperlumpedMatrix(u);
      break;
    case FetiPrcndtnr::NONE:
      break;
    }
  }
  /* Function: precond
         *
         *  Applies the global preconditioner to a single right-hand side.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector (optional). */
  template <FetiPrcndtnr q = P>
  void precond(K *const *const in, K *const *const out = nullptr) const
  {
    A<'T', 1>(primal_, in);
    applyLocalPreconditioner<q>(primal_);
    A<'N', 1>(primal_, out ? out : in);
  }
  /* Function: project
         *
         *  Projects into the coarse space.
         *
         * Template Parameters:
         *    excluded       - True if the main processes are excluded from the domain decomposition, false otherwise.
         *    trans          - 'T' if the transposed projection should be applied, 'N' otherwise.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector (optional). */
  template <bool excluded, char trans>
  void project(K *const *const in, K *const *const out = nullptr) const
  {
    static_assert(trans == 'T' || trans == 'N', "Unsupported value for argument 'trans'");
    if (super::co_) {
      if (!excluded) {
        if (trans == 'T') precond(in, dual_);
        if (super::ev_) {
          if (trans == 'T') A<'T', 0>(primal_, dual_);
          else A<'T', 0>(primal_, in);
          if (super::schur_) {
            Blas<K>::gemv(&(Wrapper<K>::transc), &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::dof_), primal_, &i__1, &(Wrapper<K>::d__0), super::uc_, &i__1);
            super::co_->template callSolver<excluded>(super::uc_);
            Blas<K>::gemv("N", &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::dof_), super::uc_, &i__1, &(Wrapper<K>::d__0), primal_, &i__1);
          } else {
            Blas<K>::gemv(&(Wrapper<K>::transc), &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_ + super::bi_->m_, &(Subdomain<K>::a_->n_), primal_, &i__1, &(Wrapper<K>::d__0), super::uc_, &i__1);
            super::co_->template callSolver<excluded>(super::uc_);
            Blas<K>::gemv("N", &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_ + super::bi_->m_, &(Subdomain<K>::a_->n_), super::uc_, &i__1, &(Wrapper<K>::d__0), primal_, &i__1);
          }
        } else {
          super::co_->template callSolver<excluded>(super::uc_);
          std::fill_n(primal_, Subdomain<K>::dof_, K());
        }
        A<'N', 0>(primal_, dual_);
        if (trans == 'N') precond(dual_);
        if (out)
          for (unsigned int i = 0; i < super::mult_; ++i) (*out)[i] = (*in)[i] - (*dual_)[i];
        else Blas<K>::axpy(&(super::mult_), &(Wrapper<K>::d__2), *dual_, &i__1, *in, &i__1);
      } else super::co_->template callSolver<excluded>(super::uc_);
    } else if (!excluded && out) std::copy_n(*in, super::mult_, *out);
  }
  /* Function: buildTwo
         *
         *  Assembles and factorizes the coarse operator by calling <Preconditioner::buildTwo>.
         *
         * Template Parameter:
         *    excluded       - Greater than 0 if the main processes are excluded from the domain decomposition, equal to 0 otherwise.
         *
         * Parameter:
         *    comm           - Global MPI communicator.
         *
         * See also: <Bdd::buildTwo>, <Schwarz::buildTwo>.*/
  template <unsigned short excluded = 0>
  std::pair<MPI_Request, const K *> *buildTwo(const MPI_Comm &comm)
  {
    return super::template buildTwo<excluded, FetiProjection<decltype(*this), P, K>>(this, comm);
  }
  /* Function: computeSolution
         *
         *  Computes the solution after convergence of the Projected Conjugate Gradient.
         *
         * Template Parameter:
         *    excluded       - True if the main processes are excluded from the domain decomposition, false otherwise.
         *
         * Parameters:
         *    l              - Last iterate of the Lagrange multiplier.
         *    x              - Solution vector. */
  template <bool excluded>
  void computeSolution(K *const *const l, K *const x) const
  {
    if (!excluded) {
      A<'T', 0>(primal_, l); //    primal_ = A^T l
      std::fill_n(super::structure_, super::bi_->m_, K());
      static_cast<Solver<K> *>(super::pinv_)->solve(super::structure_);                                // structure_ = S \ A^T l
      Blas<K>::axpy(&(Subdomain<K>::a_->n_), &(Wrapper<K>::d__2), super::structure_, &i__1, x, &i__1); //          x = x - S \ A^T l
      if (super::co_) {
        A<'N', 0>(x + super::bi_->m_, dual_); //      dual_ = A (x - S \ A^T l)
        precond(dual_);                       //      dual_ = Q A (x - S \ A^T l)
        if (!super::ev_) super::co_->template callSolver<excluded>(super::uc_);
        else {
          A<'T', 0>(primal_, dual_); //    primal_ = A^T Q A (x - S \ A^T l)
          if (super::schur_) {
            Blas<K>::gemv(&(Wrapper<K>::transc), &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::dof_), primal_, &i__1, &(Wrapper<K>::d__0), super::uc_, &i__1); //        uc_ = R_b^T A^T Q A (x - S \ A^T l)
            super::co_->template callSolver<excluded>(super::uc_); //        uc_ = (G Q G^T) \ R_b^T A^T Q A (x - S \ A^T l)
            Blas<K>::gemv("N", &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::dof_), super::uc_, &i__1, &(Wrapper<K>::d__0), primal_, &i__1); //        x_b = x_b - R_b^T (G Q G^T) \ R_b^T A^T Q A (x - S \ A^T l)
            Wrapper<K>::template csrmv<Wrapper<K>::I>(&(Wrapper<K>::transc), &(Subdomain<K>::dof_), &(super::bi_->m_), &(Wrapper<K>::d__2), false, super::bi_->a_, super::bi_->ia_, super::bi_->ja_, primal_, &(Wrapper<K>::d__0), super::work_);
            if (super::bi_->m_) super::s_.solve(super::work_);
            Blas<K>::axpy(&(super::bi_->m_), &(Wrapper<K>::d__2), super::work_, &i__1, x, &i__1);
            Blas<K>::axpy(&(Subdomain<K>::dof_), &(Wrapper<K>::d__2), primal_, &i__1, x + super::bi_->m_, &i__1);
          } else {
            Blas<K>::gemv(&(Wrapper<K>::transc), &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_ + super::bi_->m_, &(Subdomain<K>::a_->n_), primal_, &i__1, &(Wrapper<K>::d__0), super::uc_, &i__1); //       uc_ = R A^T Q A (x - S \ A^T l)
            super::co_->template callSolver<excluded>(super::uc_);                                                                                                                                //       uc_ = (G Q G^T) \ R A^T Q A (x - S \ A^T l)
            Blas<K>::gemv("N", &(Subdomain<K>::a_->n_), super::co_->getAddrLocal(), &(Wrapper<K>::d__2), *super::ev_, &(Subdomain<K>::a_->n_), super::uc_, &i__1, &(Wrapper<K>::d__1), x, &i__1); //         x = x - R^T (G Q G^T) \ R A^T Q A (x - S \ A^T l)
          }
        }
      }
    } else if (super::co_) super::co_->template callSolver<excluded>(super::uc_);
  }
  template <bool>
  void computeSolution(const K *const, K *const) const
  {
  }
  /* Function: computeDot
         *
         *  Computes the dot product of two Lagrange multipliers.
         *
         * Template Parameter:
         *    excluded       - True if the main processes are excluded from the domain decomposition, false otherwise.
         *
         * Parameters:
         *    a              - Left-hand side.
         *    b              - Right-hand side. */
  template <bool excluded>
  void computeDot(underlying_type<K> *const val, const K *const *const a, const K *const *const b, const MPI_Comm &comm) const
  {
    if (!excluded) *val = std::real(Blas<K>::dot(&(super::mult_), *a, &i__1, *b, &i__1)) / 2.0;
    else *val = 0.0;
    MPI_Allreduce(MPI_IN_PLACE, val, 1, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
  }
  /* Function: getScaling
         *  Returns a constant pointer to <Feti::m>. */
  const underlying_type<K> *const *getScaling() const { return m_; }
  /* Function: solveGEVP
         *
         *  Solves the GenEO problem.
         *
         * Template Parameter:
         *    L              - 'S'ymmetric or 'G'eneral transfer of the local Schur complements. */
  template <char L = 'S'>
  void solveGEVP()
  {
    underlying_type<K> *const pt = reinterpret_cast<underlying_type<K> *>(primal_);
    for (unsigned short i = 0; i < Subdomain<K>::map_.size(); ++i)
      for (unsigned int j = 0; j < Subdomain<K>::map_[i].second.size(); ++j) pt[Subdomain<K>::map_[i].second[j]] = m_[i][j];
    super::template solveGEVP<L>(pt);
    if (super::deficiency_ == 0 && super::ev_) {
      delete[] *super::ev_;
      delete[] super::ev_;
      super::ev_ = nullptr;
    }
  }
};

template <template <class> class Solver, template <class> class CoarseSolver, char S, class K, FetiPrcndtnr P>
struct hpddm_method_id<Feti<Solver, CoarseSolver, S, K, P>> {
  static constexpr char value = 2;
};
} // namespace HPDDM
