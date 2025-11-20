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
/* Class: Bdd
 *
 *  A class for solving problems using the BDD method.
 *
 * Template Parameters:
 *    Solver         - Solver used for the factorization of local matrices.
 *    CoarseOperator - Class of the coarse operator.
 *    S              - 'S'ymmetric or 'G'eneral coarse operator.
 *    K              - Scalar type. */
template <template <class> class Solver, template <class> class CoarseSolver, char S, class K>
class Bdd : public Schur<Solver, CoarseOperator<CoarseSolver, S, K>, K> {
private:
  /* Variable: m
         *  Local partition of unity. */
  underlying_type<K> *m_;

public:
  Bdd() : m_() { }
  ~Bdd() { delete[] m_; }
  /* Typedef: super
         *  Type of the immediate parent class <Schur>. */
  typedef Schur<Solver, CoarseOperator<CoarseSolver, S, K>, K> super;
  /* Function: initialize
         *  Allocates <Bdd::m> and calls <Schur::initialize>. */
  void initialize()
  {
    super::template initialize<false>();
    m_ = new underlying_type<K>[Subdomain<K>::dof_];
  }
  void allocateSingle(K *&primal) const { primal = new K[Subdomain<K>::dof_]; }
  template <unsigned short N>
  void allocateArray(K *(&array)[N]) const
  {
    *array = new K[N * Subdomain<K>::dof_];
    for (unsigned short i = 1; i < N; ++i) array[i] = *array + i * Subdomain<K>::dof_;
  }
  /* Function: buildScaling
         *
         *  Builds the local partition of unity <Bdd::m>.
         *
         * Parameters:
         *    scaling        - Type of scaling (multiplicity, stiffness or coefficient scaling).
         *    rho            - Physical local coefficients (optional). */
  template <class T>
  void buildScaling(T &scaling, const K *const &rho = nullptr)
  {
    initialize();
    std::fill_n(m_, Subdomain<K>::dof_, 1.0);
    if ((scaling == 2 && rho) || scaling == 1) {
      if (scaling == 1) super::stiffnessScaling(super::work_);
      else std::copy_n(rho + super::bi_->m_, Subdomain<K>::dof_, super::work_);
      bool allocate = Subdomain<K>::setBuffer(super::structure_, Subdomain<K>::a_->n_);
      Subdomain<K>::recvBuffer(super::work_);
      for (unsigned short i = 0, size = Subdomain<K>::map_.size(); i < size; ++i)
        for (unsigned int j = 0; j < Subdomain<K>::map_[i].second.size(); ++j)
          m_[Subdomain<K>::map_[i].second[j]] *= std::real(Subdomain<K>::buff_[size + i][j]) / std::real(Subdomain<K>::buff_[size + i][j] + m_[Subdomain<K>::map_[i].second[j]] * Subdomain<K>::buff_[i][j]);
      Subdomain<K>::clearBuffer(allocate);
    } else {
      scaling = 0;
      for (const pairNeighbor &neighbor : Subdomain<K>::map_)
        for (pairNeighbor::second_type::const_reference p : neighbor.second) m_[p] /= 1.0 + m_[p];
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
  bool start(const K *const f, K *const x, K *const b, K *r) const
  {
    bool allocate = Subdomain<K>::setBuffer();
    if (super::co_) {
      super::start();
      if (!excluded) {
        super::condensateEffort(f, b);
        Subdomain<K>::exchange(b ? b : super::structure_ + super::bi_->m_);
        if (super::ev_) {
          std::copy_n(b ? b : super::structure_ + super::bi_->m_, Subdomain<K>::dof_, x);
          Wrapper<K>::diag(Subdomain<K>::dof_, m_, x);
          if (super::schur_) {
            Blas<K>::gemv(&(Wrapper<K>::transc), &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::dof_), x, &i__1, &(Wrapper<K>::d__0), super::uc_, &i__1);
            super::co_->template callSolver<excluded>(super::uc_);
            Blas<K>::gemv("N", &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::dof_), super::uc_, &i__1, &(Wrapper<K>::d__0), x, &i__1);
          } else {
            Blas<K>::gemv(&(Wrapper<K>::transc), &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_ + super::bi_->m_, &(Subdomain<K>::a_->n_), x, &i__1, &(Wrapper<K>::d__0), super::uc_, &i__1);
            super::co_->template callSolver<excluded>(super::uc_);
            Blas<K>::gemv("N", &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_ + super::bi_->m_, &(Subdomain<K>::a_->n_), super::uc_, &i__1, &(Wrapper<K>::d__0), x, &i__1);
          }
          Wrapper<K>::diag(Subdomain<K>::dof_, m_, x);
        } else {
          std::fill_n(x, Subdomain<K>::dof_, K());
          super::co_->template callSolver<excluded>(super::uc_);
        }
        Subdomain<K>::exchange(x);
        super::applyLocalSchurComplement(x, r);
        Subdomain<K>::exchange(r);
        Blas<K>::axpby(Subdomain<K>::dof_, Wrapper<K>::d__1, b ? b : super::structure_ + super::bi_->m_, 1, Wrapper<K>::d__2, r, 1);
      } else super::co_->template callSolver<excluded>(super::uc_);
    } else if (!excluded) {
      super::condensateEffort(f, r);
      Subdomain<K>::exchange(r);
      std::fill_n(x, Subdomain<K>::dof_, K());
    }
    return allocate;
  }
  /* Function: apply
         *
         *  Applies the global Schur complement to a single right-hand side.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector (optional). */
  void apply(K *const in, K *const out = nullptr) const
  {
    if (out) {
      super::applyLocalSchurComplement(in, out);
      Subdomain<K>::exchange(out);
    } else {
      super::applyLocalSchurComplement(in);
      Subdomain<K>::exchange(in);
    }
  }
  /* Function: precond
         *
         *  Applies the global preconditioner to a single right-hand side.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector (optional). */
  void precond(K *const in, K *const out = nullptr) const
  {
    Wrapper<K>::diag(Subdomain<K>::dof_, m_, in, super::work_ + super::bi_->m_);
    if (!HPDDM_QR || !super::schur_) {
      std::fill_n(super::work_, super::bi_->m_, K());
      static_cast<Solver<K> *>(super::pinv_)->solve(super::work_);
    } else {
      if (super::deficiency_) static_cast<QR<K> *>(super::pinv_)->solve(super::work_ + super::bi_->m_);
      else {
        int info;
        Lapack<K>::potrs("L", &(Subdomain<K>::dof_), &i__1, static_cast<const K *>(super::pinv_), &(Subdomain<K>::dof_), super::work_ + super::bi_->m_, &(Subdomain<K>::dof_), &info);
      }
    }
    if (out) {
      Wrapper<K>::diag(Subdomain<K>::dof_, m_, super::work_ + super::bi_->m_, out);
      Subdomain<K>::exchange(out);
    } else {
      Wrapper<K>::diag(Subdomain<K>::dof_, m_, super::work_ + super::bi_->m_, in);
      Subdomain<K>::exchange(in);
    }
  }
  /* Function: callNumfact
         *  Factorizes <Subdomain::a> or <Schur::schur> if available. */
  void callNumfact()
  {
    if (HPDDM_QR && super::schur_) {
      delete super::bb_;
      super::bb_ = nullptr;
      if (super::deficiency_) {
        super::pinv_ = new QR<K>(Subdomain<K>::dof_, super::schur_);
        QR<K> *qr    = static_cast<QR<K> *>(super::pinv_);
        qr->decompose();
      } else {
        super::pinv_ = new K[Subdomain<K>::dof_ * Subdomain<K>::dof_];
        Blas<K>::lacpy("L", &(Subdomain<K>::dof_), &(Subdomain<K>::dof_), super::schur_, &(Subdomain<K>::dof_), static_cast<K *>(super::pinv_), &(Subdomain<K>::dof_));
        int info;
        Lapack<K>::potrf("L", &(Subdomain<K>::dof_), static_cast<K *>(super::pinv_), &(Subdomain<K>::dof_), &info);
      }
    } else super::callNumfact();
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
  void project(K *const in, K *const out = nullptr) const
  {
    static_assert(trans == 'T' || trans == 'N', "Unsupported value for argument 'trans'");
    if (super::co_) {
      if (!excluded) {
        if (trans == 'N') apply(in, super::structure_ + super::bi_->m_);
        if (super::ev_) {
          if (trans == 'N') Wrapper<K>::diag(Subdomain<K>::dof_, m_, super::structure_ + super::bi_->m_);
          else Wrapper<K>::diag(Subdomain<K>::dof_, m_, in, super::structure_ + super::bi_->m_);
          if (super::schur_) {
            Blas<K>::gemv(&(Wrapper<K>::transc), &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::dof_), super::structure_ + super::bi_->m_, &i__1, &(Wrapper<K>::d__0), super::uc_, &i__1);
            super::co_->template callSolver<excluded>(super::uc_);
            Blas<K>::gemv("N", &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::dof_), super::uc_, &i__1, &(Wrapper<K>::d__0), super::structure_ + super::bi_->m_, &i__1);
          } else {
            Blas<K>::gemv(&(Wrapper<K>::transc), &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_ + super::bi_->m_, &(Subdomain<K>::a_->n_), super::structure_ + super::bi_->m_, &i__1, &(Wrapper<K>::d__0), super::uc_, &i__1);
            super::co_->template callSolver<excluded>(super::uc_);
            Blas<K>::gemv("N", &(Subdomain<K>::dof_), super::co_->getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_ + super::bi_->m_, &(Subdomain<K>::a_->n_), super::uc_, &i__1, &(Wrapper<K>::d__0), super::structure_ + super::bi_->m_, &i__1);
          }
        } else {
          super::co_->template callSolver<excluded>(super::uc_);
          std::fill_n(super::structure_ + super::bi_->m_, Subdomain<K>::dof_, K());
        }
        Wrapper<K>::diag(Subdomain<K>::dof_, m_, super::structure_ + super::bi_->m_);
        Subdomain<K>::exchange(super::structure_ + super::bi_->m_);
        if (trans == 'T') apply(super::structure_ + super::bi_->m_);
        if (out)
          for (unsigned int i = 0; i < Subdomain<K>::dof_; ++i) out[i] = in[i] - *(super::structure_ + super::bi_->m_ + i);
        else Blas<K>::axpy(&(Subdomain<K>::dof_), &(Wrapper<K>::d__2), super::structure_ + super::bi_->m_, &i__1, in, &i__1);
      } else super::co_->template callSolver<excluded>(super::uc_);
    } else if (!excluded && out) std::copy_n(in, Subdomain<K>::dof_, out);
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
         * See also: <Feti::buildTwo>, <Schwarz::buildTwo>.*/
  template <unsigned short excluded = 0>
  std::pair<MPI_Request, const K *> *buildTwo(const MPI_Comm &comm)
  {
    return super::template buildTwo<excluded, BddProjection<Bdd<Solver, CoarseSolver, S, K>, K>>(this, comm);
  }
  /* Function: computeSolution
         *
         *  Computes the solution after convergence of the Projected Conjugate Gradient.
         *
         * Template Parameter:
         *    excluded       - True if the main processes are excluded from the domain decomposition, false otherwise.
         *
         * Parameters:
         *    f              - Right-hand side.
         *    x              - Solution vector. */
  template <bool excluded>
  void computeSolution(const K *const f, K *const x) const
  {
    if (!excluded && super::bi_->m_) {
      std::copy_n(f, super::bi_->m_, x);
      Wrapper<K>::template csrmv<Wrapper<K>::I>(&(Wrapper<K>::transc), &(Subdomain<K>::dof_), &(super::bi_->m_), &(Wrapper<K>::d__2), false, super::bi_->a_, super::bi_->ia_, super::bi_->ja_, x + super::bi_->m_, &(Wrapper<K>::d__1), x);
      if (!super::schur_) super::s_.solve(x);
      else {
        std::copy_n(x, super::bi_->m_, super::structure_);
        super::s_.solve(super::structure_);
        std::copy_n(super::structure_, super::bi_->m_, x);
      }
    }
  }
  template <bool>
  void computeSolution(K *const *const, K *const) const
  {
  }
  /* Function: computeDot
         *
         *  Computes the dot product of two vectors.
         *
         * Template Parameter:
         *    excluded       - True if the main processes are excluded from the domain decomposition, false otherwise.
         *
         * Parameters:
         *    a              - Left-hand side.
         *    b              - Right-hand side. */
  template <bool excluded>
  void computeDot(underlying_type<K> *const val, const K *const a, const K *const b, const MPI_Comm &comm) const
  {
    if (!excluded) {
      Wrapper<K>::diag(Subdomain<K>::dof_, m_, a, super::work_);
      *val = std::real(Blas<K>::dot(&(Subdomain<K>::dof_), super::work_, &i__1, b, &i__1));
    } else *val = 0.0;
    MPI_Allreduce(MPI_IN_PLACE, val, 1, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
  }
  /* Function: getScaling
         *  Returns a constant pointer to <Bdd::m>. */
  const underlying_type<K> *getScaling() const { return m_; }
  /* Function: solveGEVP
         *
         *  Solves the GenEO problem.
         *
         * Template Parameter:
         *    L              - 'S'ymmetric or 'G'eneral transfer of the local Schur complements. */
  template <char L = 'S'>
  void solveGEVP()
  {
    super::template solveGEVP<L>(m_);
  }
};

template <template <class> class Solver, template <class> class CoarseSolver, char S, class K>
struct hpddm_method_id<Bdd<Solver, CoarseSolver, S, K>> {
  static constexpr char value = 3;
};
} // namespace HPDDM
