/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2013-03-12

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

#include <cstddef>
#include <queue>

namespace HPDDM
{
template <bool>
class Members { };
template <>
class Members<true> {
protected:
  std::unordered_map<unsigned short, unsigned int> offsets_;
  std::vector<std::vector<unsigned short>>         vecSparsity_;
  unsigned short                                   rank_;
  unsigned short                                   consolidate_;
};
template <char P, class Preconditioner, class K>
class OperatorBase : protected Members<P != 's' && P != 'u'> {
private:
  HPDDM_HAS_MEMBER(getLDR)
  template <class Q = Preconditioner>
  std::enable_if_t<has_getLDR<std::remove_reference_t<Q>>::value, bool> offsetDeflation()
  {
    const unsigned int offset = *p_.getLDR() - n_;
    if (deflation_ && offset) std::for_each(const_cast<K **>(deflation_), const_cast<K **>(deflation_) + local_, [&](K *&v) { v -= offset; });
    return true;
  }
  template <class Q = Preconditioner>
  std::enable_if_t<!has_getLDR<std::remove_reference_t<Q>>::value, bool> offsetDeflation()
  {
    return false;
  }

protected:
  const Preconditioner       &p_;
  const K *const *const       deflation_;
  const vectorNeighbor       &map_;
  std::vector<unsigned short> sparsity_;
  const int                   n_;
  const int                   local_;
  unsigned int                max_;
  unsigned short              signed_;
  unsigned short              connectivity_;
  template <char Q = P, std::enable_if_t<Q == 's' || Q == 'u'> * = nullptr>
  OperatorBase(const Preconditioner &p, const unsigned short &c, const unsigned int &max) :
    p_(p), deflation_(initDeflation(p)), map_(p.subdomain.getMap()), n_(p.subdomain.getDof()), local_(p.getLocal()), max_(max), connectivity_(c)
  {
    static_assert(Q == P, "Wrong sparsity pattern");
#ifdef PETSC_PCHPDDM_MAXLEVELS
    const vectorNeighbor &coarseMap = p.subdomain.getCoarseMap();
#else
    const vectorNeighbor &coarseMap = map_;
#endif
    sparsity_.reserve(coarseMap.size());
    std::transform(coarseMap.cbegin(), coarseMap.cend(), std::back_inserter(sparsity_), [](const pairNeighbor &n) { return n.first; });
  }
  template <char Q = P, std::enable_if_t<Q != 's' && Q != 'u'> * = nullptr>
  OperatorBase(const Preconditioner &p, const unsigned short &c, const unsigned int &max) :
    p_(p),
    deflation_(initDeflation(p)),
    map_(p.subdomain.getMap()),
    n_(p.subdomain.getDof()),
    local_(p.getLocal()),
    max_(max + std::max(1, (c - 1)) * (max & 4095)),
    connectivity_(c)
  {
#if HPDDM_BDD
    Members<true>::rank_      = p_.getRank();
    signed_                   = p_.getSigned();
    const unsigned int offset = *p_.getLDR() - n_;
#else
    int rank;
    MPI_Comm_rank(p_.subdomain.getCommunicator(), &rank);
    Members<true>::rank_ = rank;
    signed_ = static_cast<unsigned short>(std::count_if(map_.cbegin(), map_.cend(), [rank](const pairNeighbor &neighbor) { return neighbor.first < rank; }));
    constexpr unsigned int offset = 0;
#endif
    Members<true>::consolidate_ = 0;
    if (deflation_ && offset) std::for_each(const_cast<K **>(deflation_), const_cast<K **>(deflation_) + local_, [&](K *&v) { v += offset; });
    static_assert(Q == P, "Wrong sparsity pattern");
    if (!map_.empty()) {
      unsigned short **recvSparsity = new unsigned short *[map_.size() + 1];
      *recvSparsity                 = new unsigned short[(connectivity_ + 1) * map_.size()];
      unsigned short *sendSparsity  = *recvSparsity + connectivity_ * map_.size();
      MPI_Request    *rq            = p_.subdomain.getRq();
      for (unsigned short i = 0; i < map_.size(); ++i) {
        sendSparsity[i] = map_[i].first;
        recvSparsity[i] = *recvSparsity + static_cast<std::ptrdiff_t>(connectivity_) * i;
        MPI_Irecv(recvSparsity[i], connectivity_, MPI_UNSIGNED_SHORT, map_[i].first, 4, p_.subdomain.getCommunicator(), rq + i);
      }
      for (unsigned short i = 0; i < map_.size(); ++i)
        MPI_Isend(sendSparsity, map_.size(), MPI_UNSIGNED_SHORT, map_[i].first, 4, p_.subdomain.getCommunicator(), rq + map_.size() + i);
      Members<true>::vecSparsity_.resize(map_.size());
      for (unsigned short i = 0; i < map_.size(); ++i) {
        int        index, count;
        MPI_Status status;
        MPI_Waitany(map_.size(), rq, &index, &status);
        MPI_Get_count(&status, MPI_UNSIGNED_SHORT, &count);
        Members<true>::vecSparsity_[index].assign(recvSparsity[index], recvSparsity[index] + count);
      }
      MPI_Waitall(map_.size(), rq + map_.size(), MPI_STATUSES_IGNORE);

      delete[] *recvSparsity;
      delete[] recvSparsity;

      sparsity_.reserve(map_.size());
      if (P == 'c') {
        std::vector<unsigned short> neighbors;
        neighbors.reserve(map_.size());
        std::for_each(map_.cbegin(), map_.cend(), [&](const pairNeighbor &n) { neighbors.emplace_back(n.first); });
        using pairIt = std::pair<std::vector<unsigned short>::const_iterator, std::vector<unsigned short>::const_iterator>;
        auto                                                             comp = [](const pairIt &lhs, const pairIt &rhs) { return *lhs.first > *rhs.first; };
        std::priority_queue<pairIt, std::vector<pairIt>, decltype(comp)> pq(comp);
        pq.push({neighbors.cbegin(), neighbors.cend()});
        for (const std::vector<unsigned short> &v : Members<true>::vecSparsity_) pq.push({v.cbegin(), v.cend()});
        while (!pq.empty()) {
          pairIt p = pq.top();
          pq.pop();
          if (*p.first != Members<true>::rank_ && (sparsity_.empty() || (*p.first != sparsity_.back()))) sparsity_.emplace_back(*p.first);
          if (++p.first != p.second) pq.push(p);
        }
      } else {
        std::transform(map_.cbegin(), map_.cend(), std::back_inserter(sparsity_), [](const pairNeighbor &n) { return n.first; });
        for (std::vector<unsigned short> &v : Members<true>::vecSparsity_) {
          unsigned short i = 0, j = 0, k = 0;
          while (i < v.size() && j < sparsity_.size()) {
            if (v[i] == Members<true>::rank_) {
              v[k++] = Members<true>::rank_;
              ++i;
            } else if (v[i] < sparsity_[j]) ++i;
            else if (v[i] > sparsity_[j]) ++j;
            else {
              v[k++] = sparsity_[j++];
              ++i;
            }
          }
          v.resize(k);
        }
      }
    }
  }
  ~OperatorBase()
  {
    offsetDeflation();
    if (!has_getVectors<std::remove_reference_t<Preconditioner>>::value) delete[] deflation_;
  }
  HPDDM_HAS_MEMBER(getVectors)
  template <class Q = Preconditioner, std::enable_if_t<has_getVectors<std::remove_reference_t<Q>>::value> * = nullptr>
  static const K *const *initDeflation(const Preconditioner &p)
  {
    return p.getVectors();
  }
#if HPDDM_PETSC
  template <class Q = Preconditioner, std::enable_if_t<!has_getVectors<std::remove_reference_t<Q>>::value> * = nullptr>
  static const K *const *initDeflation(const Preconditioner &)
  {
    return nullptr;
  }
#endif
  template <char S, bool U, class T>
  void initialize(T &in, const unsigned short *info, T const &out, MPI_Request *const &rqRecv, unsigned short *&infoNeighbor)
  {
    static_assert(P == 'c' || P == 'f', "Unsupported constructor with such a sparsity pattern");
    if (!U) {
      if (P == 'c') {
        infoNeighbor                                      = new unsigned short[map_.size()];
        std::vector<unsigned short>::const_iterator begin = sparsity_.cbegin();
        for (unsigned short i = 0; i < map_.size(); ++i) {
          std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, sparsity_.cend(), map_[i].first);
          infoNeighbor[i]                                 = info[std::distance(sparsity_.cbegin(), idx)];
          begin                                           = idx + 1;
        }
      } else infoNeighbor = const_cast<unsigned short *>(info);
    }
    std::vector<unsigned int> displs;
    displs.reserve(2 * map_.size());
    if (S != 'S') {
      if (!U) {
        unsigned short size = std::accumulate(infoNeighbor, infoNeighbor + map_.size(), local_);
        if (!map_.empty()) displs.emplace_back(size * map_[0].second.size());
        for (unsigned short i = 1; i < map_.size(); ++i) displs.emplace_back(displs.back() + size * map_[i].second.size());
        for (unsigned short i = 0; i < map_.size(); ++i) {
          size                                              = infoNeighbor[i];
          std::vector<unsigned short>::const_iterator begin = sparsity_.cbegin();
          for (const unsigned short &rank : Members<true>::vecSparsity_[i]) {
            if (rank == Members<true>::rank_) size += local_;
            else {
              std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, sparsity_.cend(), rank);
              size += info[std::distance(sparsity_.cbegin(), idx)];
              begin = idx + 1;
            }
          }
          if (local_) displs.emplace_back(displs.back() + size * map_[i].second.size());
          else rqRecv[i] = MPI_REQUEST_NULL;
        }
      } else {
        if (!map_.empty()) displs.emplace_back(local_ * (map_.size() + 1) * map_[0].second.size());
        for (unsigned short i = 1; i < map_.size(); ++i) displs.emplace_back(displs.back() + local_ * (map_.size() + 1) * map_[i].second.size());
        for (unsigned short i = 0; i < map_.size(); ++i)
          displs.emplace_back(displs.back() + local_ * (Members<true>::vecSparsity_[i].size() + 1) * map_[i].second.size());
      }
    } else {
      if (!U) {
        unsigned short size = std::accumulate(infoNeighbor, infoNeighbor + map_.size(), 0);
        if (!map_.empty()) {
          displs.emplace_back((size + local_ * (0 < signed_)) * map_[0].second.size());
          size -= infoNeighbor[0];
        }
        for (unsigned short i = 1; i < map_.size(); ++i) {
          displs.emplace_back(displs.back() + (size + local_ * (i < signed_)) * map_[i].second.size());
          size -= infoNeighbor[i];
        }
        for (unsigned short i = 0; i < map_.size(); ++i) {
          size                                            = infoNeighbor[i] * !(i < signed_) + local_;
          std::vector<unsigned short>::const_iterator end = sparsity_.cend();
          for (std::vector<unsigned short>::const_reverse_iterator rit = Members<true>::vecSparsity_[i].rbegin(); *rit > Members<true>::rank_; ++rit) {
            std::vector<unsigned short>::const_iterator idx = std::lower_bound(sparsity_.cbegin(), end, *rit);
            size += info[std::distance(sparsity_.cbegin(), idx)];
            end = idx - 1;
          }
          if (local_) displs.emplace_back(displs.back() + size * map_[i].second.size());
          else rqRecv[i] = MPI_REQUEST_NULL;
        }
      } else {
        if (!map_.empty()) displs.emplace_back(local_ * (map_.size() + (0 < signed_)) * map_[0].second.size());
        for (unsigned short i = 1; i < map_.size(); ++i)
          displs.emplace_back(displs.back() + local_ * (map_.size() + (i < signed_) - i) * map_[i].second.size());
        for (unsigned short i = 0; i < map_.size(); ++i) {
          unsigned short size = std::distance(std::lower_bound(Members<true>::vecSparsity_[i].cbegin(), Members<true>::vecSparsity_[i].cend(),
                                                               Members<true>::rank_),
                                              Members<true>::vecSparsity_[i].cend()) +
                                !(i < signed_);
          displs.emplace_back(displs.back() + static_cast<std::size_t>(local_) * size * map_[i].second.size());
        }
      }
    }
    if (!displs.empty()) {
      *in = new K[displs.back()];
      for (unsigned short i = 1; i < map_.size(); ++i) in[i] = *in + displs[i - 1];
      if (U == 1 || local_)
        for (unsigned short i = 0; i < map_.size(); ++i) {
          if (displs[i + map_.size()] != displs[i - 1 + map_.size()]) {
            out[i] = *in + displs[i - 1 + map_.size()];
            MPI_Irecv(out[i], displs[i + map_.size()] - displs[i - 1 + map_.size()], Wrapper<K>::mpi_type(), map_[i].first, 2, p_.subdomain.getCommunicator(),
                      rqRecv + i);
          } else out[i] = nullptr;
        }
    } else if (out) *in = nullptr;
  }
  template <char S, char N, bool U, class T, char Q = P, std::enable_if_t<Q != 's'> * = nullptr>
  void assembleOperator(T *I, T *J, int coefficients, unsigned int offsetI, unsigned int *offsetJ, unsigned short *const &infoNeighbor)
  {
#ifdef HPDDM_CSR_CO
    ignore(I);
#endif
    if (Members<true>::consolidate_ == map_.size()) {
      unsigned short between = std::distance(sparsity_.cbegin(), std::lower_bound(sparsity_.cbegin(), sparsity_.cend(), Members<true>::rank_));
      unsigned int   offset  = 0;
      if (S != 'S' && S != 'B')
        for (unsigned short k = 0; k < between; ++k) {
          if (S != 'C') {
            for (unsigned short i = 0; i < local_; ++i) {
              unsigned int l = offset + coefficients * i;
              for (unsigned short j = 0; j < (U ? local_ : infoNeighbor[k]); ++j) {
#ifndef HPDDM_CSR_CO
                I[l + j] = offsetI + i;
#endif
                J[l + j] = (U ? sparsity_[k] * local_ + (N == 'F') : offsetJ[k]) + j;
              }
            }
            offset += U ? local_ : infoNeighbor[k];
          } else J[offset++] = sparsity_[k];
        }
      else if (S != 'C') coefficients += local_ - 1;
      if (S != 'B' && S != 'C') {
        for (unsigned short i = 0; i < local_; ++i) {
          unsigned int l = offset + coefficients * i - (S == 'S') * ((i * (i - 1)) / 2);
          for (unsigned short j = (S == 'S') * i; j < local_; ++j) {
#ifndef HPDDM_CSR_CO
            I[l + j] = offsetI + i;
#endif
            J[l + j] = offsetI + j;
          }
        }
        offset += local_;
      } else J[offset++] = offsetI / (U ? local_ : 1);
      for (unsigned short k = between; k < sparsity_.size(); ++k) {
        if (S != 'B' && S != 'C') {
          for (unsigned short i = 0; i < local_; ++i) {
            unsigned int l = offset + coefficients * i - (S == 'S') * ((i * (i - 1)) / 2);
            for (unsigned short j = 0; j < (U ? local_ : infoNeighbor[k]); ++j) {
#ifndef HPDDM_CSR_CO
              I[l + j] = offsetI + i;
#endif
              J[l + j] = (U ? sparsity_[k] * local_ + (N == 'F') : offsetJ[k - (S == 'S') * between]) + j;
            }
          }
          offset += U ? local_ : infoNeighbor[k];
        } else J[offset++] = sparsity_[k];
      }
    }
  }

protected:
  MPI_Request *sendRequestsEnd_ = nullptr;
  K           *workBase_        = nullptr;

public:
  static constexpr char pattern_   = (P != 's' && P != 'u') ? 'c' : (P == 's' ? 's' : 'u');
  static constexpr bool factorize_ = true;
  template <char, bool>
  void setPattern(int *, const int, const int, MPI_Comm, const unsigned short *const *const = nullptr, const unsigned short *const = nullptr) const
  {
  }
  void adjustConnectivity(const MPI_Comm &comm)
  {
    if (P == 'c') {
#if 0
                connectivity_ *= connectivity_ - 1;
#else
      connectivity_ = sparsity_.size();
      MPI_Allreduce(MPI_IN_PLACE, &connectivity_, 1, MPI_UNSIGNED_SHORT, MPI_MAX, comm);
#endif
    }
  }
  const std::vector<unsigned short> &getPattern() const { return sparsity_; }
  unsigned short                     getConnectivity() const { return connectivity_; }
  template <char Q = P, std::enable_if_t<Q != 's'> * = nullptr, class... Args>
  void initialize(unsigned int, Args &&...)
  {
  }
#ifdef PETSC_PCHPDDM_MAXLEVELS
  PetscErrorCode  computeBlocks(bool, bool) { return PETSC_SUCCESS; }
  unsigned short *getNeighborSizes() { return nullptr; }
  void            finishExchange()
  {
    if (sendRequestsEnd_) {
      MPI_Request *const base = p_.subdomain.getRq();
      MPI_Waitall(static_cast<int>(std::distance(base, sendRequestsEnd_)), base, MPI_STATUSES_IGNORE);
      sendRequestsEnd_ = nullptr;
    }
    if (!map_.empty()) {
      delete[] *p_.subdomain.getBuffer();
      *p_.subdomain.getBuffer() = nullptr;
    }
    delete[] workBase_;
    workBase_ = nullptr;
  }
#endif
};

template <class Preconditioner, class K>
class UserCoarseOperator : public OperatorBase<'u', Preconditioner, K> {
private:
  using super = OperatorBase<'u', Preconditioner, K>;
#if HPDDM_PETSC
protected:
  const Mat          A_;
  Mat                C_{};
  PC_HPDDM_Level    *level_;
  const std::string &prefix_;
#endif
public:
  HPDDM_CLASS_COARSE_OPERATOR(Solver, S, T) friend class CoarseOperator;
#if HPDDM_PETSC
  template <class T>
  friend class Schwarz;
  template <typename... Types>
  UserCoarseOperator(const Preconditioner &p, const unsigned short &c, const unsigned int &max, Mat A, PC_HPDDM_Level *level, std::string &prefix, Types...) :
    super(p, c, max), A_(A), level_(level), prefix_(prefix)
  {
    static_assert(sizeof...(Types) == 0, "Wrong constructor");
  }
#else
  UserCoarseOperator(const Preconditioner &p, const unsigned short &c, const unsigned int &max) : super(p, c, max) { }
#endif
};

#if HPDDM_SCHWARZ || HPDDM_SLEPC
template <class Preconditioner, class K>
class MatrixMultiplication : public OperatorBase<'s', Preconditioner, K> {
protected:
  #if HPDDM_SCHWARZ
  const MatrixCSR<K> *const A_;
  MatrixCSR<K>             *C_;
  #else
  const Mat          A_;
  Mat                C_{};
  PC_HPDDM_Level    *level_;
  const std::string &prefix_;
    #ifdef PETSC_PCHPDDM_MAXLEVELS
  Mat matP_         = nullptr;
  Mat matSelfBlock_ = nullptr;
  Mat matW_         = nullptr;
  Mat matCoupling_  = nullptr;
    #endif
  #endif
  K                              *work_;
  const underlying_type<K> *const D_;
  K                              *values_          = nullptr;
  unsigned int                    selfBlockOffset_ = 0;
  int                             rowStride_       = 0;

private:
  using super = OperatorBase<'s', Preconditioner, K>;
  template <bool U>
  void applyFromNeighbor(const K *in, unsigned short index, K *&work, unsigned short *infoNeighbor)
  {
    const int m = U ? super::local_ : *infoNeighbor;
    if (super::local_ && m) {
      const std::vector<int> &intersection = super::map_[index].second;
      const int               s            = intersection.size();
      if (s == 0) std::fill_n(work, super::local_ * m, K());
      else if (s == super::n_) {
        std::fill_n(work_, super::n_ * m, K());
        for (unsigned short i = 0; i < m; ++i)
          for (int j = 0; j < s; ++j) work_[i * s + intersection[j]] = D_[intersection[j]] * in[i * s + j];
        Blas<K>::gemm(&(Wrapper<K>::transc), "N", &(super::local_), &m, &(super::n_), &(Wrapper<K>::d_1), *super::deflation_, &(super::n_), work_, &(super::n_),
                      &(Wrapper<K>::d_0), work, &(super::local_));
      } else {
        for (unsigned short i = 0; i < super::local_; ++i)
          for (int j = 0; j < s; ++j) work_[i * s + j] = D_[intersection[j]] * super::deflation_[i][intersection[j]];
        Blas<K>::gemm(&(Wrapper<K>::transc), "N", &(super::local_), &m, &s, &(Wrapper<K>::d_1), work_, &s, in, &s, &(Wrapper<K>::d_0), work, &(super::local_));
      }
    }
  }
  #ifdef PETSC_PCHPDDM_MAXLEVELS
  template <bool U>
  void applyFromNeighbor(const K *in, unsigned short index, unsigned short neighborLocal)
  {
    if (!super::local_ || !neighborLocal) return;
    const std::vector<int> &overlap = super::map_[index].second;
    PetscScalar            *array;
    const PetscScalar      *scaling;
    PetscCallVoid(MatDenseGetArrayWrite(matW_, &array));
    PetscCallVoid(VecGetArrayRead(super::p_.getScalingVec(), &scaling));
    std::fill_n(array, static_cast<std::size_t>(neighborLocal) * super::n_, K());
    for (unsigned short i = 0; i < neighborLocal; ++i)
      for (unsigned int j = 0; j < overlap.size(); ++j) array[i * super::n_ + overlap[j]] = scaling[overlap[j]] * in[i * overlap.size() + j];
    PetscCallVoid(VecRestoreArrayRead(super::p_.getScalingVec(), &scaling));
    PetscCallVoid(MatDenseRestoreArrayWrite(matW_, &array));
    Mat W = matW_, X = matCoupling_;
    if (!U) {
      PetscCallVoid(MatDenseGetSubMatrix(matW_, 0, super::n_, 0, neighborLocal, &W));
      PetscCallVoid(MatDenseGetSubMatrix(matCoupling_, 0, super::local_, 0, neighborLocal, &X));
    }
    PetscCallVoid(MatConjugate(W));
    PetscCallVoid(MatTransposeMatMult(super::p_.getMat(), W, MAT_REUSE_MATRIX, PETSC_DEFAULT, &X));
    PetscCallVoid(MatConjugate(X));
    if (!U) {
      PetscCallVoid(MatDenseRestoreSubMatrix(matCoupling_, &X));
      PetscCallVoid(MatDenseRestoreSubMatrix(matW_, &W));
    }
  }
  #endif

public:
  HPDDM_CLASS_COARSE_OPERATOR(Solver, S, T) friend class CoarseOperator;
  template <typename... Types>
  #if HPDDM_PETSC
  MatrixMultiplication(const Preconditioner &p, const unsigned short &c, const unsigned int &max, Mat A, PC_HPDDM_Level *level, std::string &prefix, Types...) :
    super(p, c, max), A_(A), level_(level), prefix_(prefix), D_(p.getScaling())
  {
    static_assert(sizeof...(Types) == 0, "Wrong constructor");
  }
  PetscErrorCode setupScaledMatrix()
  {
    PetscBool flg;

    PetscFunctionBeginUser;
    PetscCall(PetscObjectTypeCompareAny((PetscObject)A_, &flg, MATSEQSBAIJ, MATMPISBAIJ, ""));
    PetscCall(MatConvert(A_, flg ? MATBAIJ : MATSAME, MAT_INITIAL_MATRIX, &C_));
    PetscCall(MatDiagonalScale(C_, nullptr, super::p_.getScalingVec()));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  void initialize(unsigned int k, K *&work, unsigned short s)
  {
    PetscCallVoid(setupScaledMatrix());
    work           = new K[2 * static_cast<std::size_t>(k)];
    work_          = work + k;
    super::signed_ = s;
  }
    #if defined(PETSC_PCHPDDM_MAXLEVELS)
  void createSelfBlockMats(PetscScalar *Pptr = nullptr, PetscScalar *SelfBlockPtr = nullptr)
  {
    PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, super::local_, Pptr, &matP_));
    PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::local_, super::local_, SelfBlockPtr, &matSelfBlock_));
  }
  void createCouplingMats(unsigned short maxColumns)
  {
    PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, maxColumns, nullptr, &matW_));
    PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::local_, maxColumns, nullptr, &matCoupling_));
  }
  void createWorkMats(unsigned short maxColumns)
  {
    createCouplingMats(maxColumns);
    PetscScalar *Pptr         = nullptr;
    PetscScalar *SelfBlockPtr = nullptr;
    PetscCallVoid(MatDenseGetArray(matW_, &Pptr));
    PetscCallVoid(MatDenseGetArray(matCoupling_, &SelfBlockPtr));
    createSelfBlockMats(Pptr, SelfBlockPtr);
    PetscCallVoid(MatDenseRestoreArray(matW_, &Pptr));
    PetscCallVoid(MatDenseRestoreArray(matCoupling_, &SelfBlockPtr));
  }
  template <char S, bool U, class T>
  void prepareExchange(T &in, T const &out, const unsigned short *info, MPI_Request *const &rqRecv)
  {
    if (super::map_.empty()) return;
    constexpr bool            isSymmetric               = (S == 'S');
    const unsigned short      skippedLowerNeighborCount = isSymmetric ? super::signed_ : 0;
    const unsigned short      neighborCount             = isSymmetric ? static_cast<unsigned short>(super::map_.size()) - super::signed_ : super::signed_;
    const unsigned short      sendNeighborCount         = super::signed_;
    std::vector<unsigned int> sendSizes(sendNeighborCount);
    std::vector<unsigned int> recvSizes(neighborCount);
    for (unsigned short i = 0; i < sendNeighborCount; ++i) sendSizes[i] = (U || info[i] ? super::local_ * super::map_[i].second.size() : 0);
    if (U || super::local_)
      for (unsigned short i = 0; i < neighborCount; ++i)
        recvSizes[i] = (U ? super::local_ : info[skippedLowerNeighborCount + i]) * super::map_[i + skippedLowerNeighborCount].second.size();
    const unsigned int sendCount = std::accumulate(sendSizes.begin(), sendSizes.end(), 0u);
    const unsigned int recvCount = std::accumulate(recvSizes.begin(), recvSizes.end(), 0u);
    *in                          = new K[sendCount + recvCount];

    unsigned int sendCursor = 0;
    for (unsigned short i = 0; i < sendNeighborCount; ++i) {
      in[i] = *in + sendCursor;
      sendCursor += sendSizes[i];
    }

    if (U || super::local_) {
      unsigned int recvCursor = sendCount;
      for (unsigned short i = 0; i < neighborCount; ++i) {
        out[i] = *in + recvCursor;
        MPI_Irecv(out[i], recvSizes[i], Wrapper<K>::mpi_type(), super::map_[i + skippedLowerNeighborCount].first, 2, super::p_.subdomain.getCommunicator(),
                  rqRecv + i);
        recvCursor += recvSizes[i];
      }
    } else std::fill_n(rqRecv, neighborCount, MPI_REQUEST_NULL);
  }
  template <char S, bool U>
  void assembleSelfBlock(Mat W)
  {
    if (!super::local_) return;
    PetscCallVoid(MatConjugate(W));
    PetscCallVoid(MatTransposeMatMult(super::p_.getMat(), W, MAT_REUSE_MATRIX, PETSC_DEFAULT, &matSelfBlock_));
    PetscCallVoid(MatConjugate(matSelfBlock_));
    const K *scratch;
    PetscCallVoid(MatDenseGetArrayRead(matSelfBlock_, &scratch));
    copySelfBlock<S, U>(scratch);
    PetscCallVoid(MatDenseRestoreArrayRead(matSelfBlock_, &scratch));
  }
  template <char S>
  unsigned short assemblyNeighborCount() const
  {
    return S == 'S' ? static_cast<unsigned short>(super::map_.size()) - super::signed_ : super::signed_;
  }
  template <char S, bool U, bool isMain, class T = unsigned int>
  void assembleCoarseOperator(K *C, const CoarseOffsets &offsets, const NeighborSizes &neighborSizes, T *J = nullptr)
  {
    K *const *const       in                        = super::p_.subdomain.getBuffer() + super::signed_;
    MPI_Request *const    rqRecv                    = super::p_.subdomain.getRq() + super::signed_;
    const unsigned short  skippedLowerNeighborCount = neighborSizes.skippedLowerNeighborCount();
    const unsigned short  neighborCount             = assemblyNeighborCount<S>();
    unsigned short *const infoUpper                 = neighborSizes.upper();
    for (unsigned short k = 0; k < neighborCount; ++k) {
      int neighborIndex;
      MPI_Waitany(neighborCount, rqRecv, &neighborIndex, MPI_STATUS_IGNORE);
      applyFromNeighbor<U>(in[neighborIndex], neighborIndex + skippedLowerNeighborCount, U ? super::local_ : infoUpper[neighborIndex]);
      const K *arrayX;
      PetscCallVoid(MatDenseGetArrayRead(matCoupling_, &arrayX));
      copyNeighborBlock<S, U, isMain>(arrayX, C, neighborIndex, super::map_[neighborIndex + skippedLowerNeighborCount].first, offsets, neighborSizes, J);
      PetscCallVoid(MatDenseRestoreArrayRead(matCoupling_, &arrayX));
    }
  }
  void destroyWorkMats()
  {
    PetscCallVoid(MatDestroy(&matCoupling_));
    PetscCallVoid(MatDestroy(&matW_));
    PetscCallVoid(MatDestroy(&matSelfBlock_));
    PetscCallVoid(MatDestroy(&matP_));
  }

private:
  template <char S, bool U>
  void copySelfBlock(const K *scratch)
  {
    if (!super::local_) return;
    if (S != 'S') {
      K *const  selfBlock = values_ + selfBlockOffset_ * (!U ? 1 : super::local_ * super::local_);
      const int ld        = !U ? rowStride_ : super::local_;
      Wrapper<K>::template omatcopy<'T'>(super::local_, super::local_, scratch, super::local_, selfBlock, ld);
    } else if (U) {
      std::copy_n(scratch, static_cast<std::size_t>(super::local_) * super::local_, values_);
    } else {
      for (unsigned short j = 0; j < super::local_; ++j)
        std::copy_n(scratch + j * super::local_ + j, super::local_ - j, values_ + j * (rowStride_ + super::local_) - (j * (j - 1)) / 2);
    }
  }
  template <char S, bool U, bool isMain, class T = unsigned int>
  void copyNeighborBlock(const K *arrayX, K *C, unsigned short neighborIndex, unsigned short neighbor, const CoarseOffsets &offsets,
                         const NeighborSizes &neighborSizes, T *J = nullptr)
  {
    const unsigned int valueOffset = offsets.neighborValue()[neighborIndex];
    const unsigned int offset      = valueOffset / (!U ? 1 : super::local_);

    if (!U && S != 'B') {
      const int      coefficients = offsets.rowStride();
      constexpr bool isSymmetric  = (S == 'S');
      for (unsigned short i = 0; i < super::local_; ++i) {
        unsigned int l = (coefficients + (isSymmetric ? super::local_ - 1 : 0)) * i - (isSymmetric ? (i * (i - 1)) / 2 : 0);
        for (unsigned short j = 0; j < neighborSizes.upper()[neighborIndex]; ++j) {
          if (isMain) J[offset + l + j] = offsets.neighborColumn()[neighborIndex] + j;
          C[valueOffset + l + j] = arrayX[j * super::local_ + i];
        }
      }
    } else {
      if (isMain) J[offset] = neighbor;
      if (super::local_ && (isMain || U))
        Wrapper<K>::template omatcopy<'T'>(super::local_, super::local_, arrayX, super::local_, C + valueOffset * (!U ? 1 : super::local_), super::local_);
    }
  }
  std::vector<K>              selfBlock_;
  std::unique_ptr<K[]>        blocks_;
  std::vector<unsigned int>   blockDispls_;
  std::vector<unsigned short> neighborSizes_;
  static PetscErrorCode       copyLocalBlock(Mat X, PetscInt m, PetscInt n, std::vector<K> &out)
  {
    const PetscScalar *array;
    PetscInt           ld;

    PetscFunctionBeginUser;
    PetscCall(MatDenseGetArrayRead(X, &array));
    PetscCall(MatDenseGetLDA(X, &ld));
    out.resize(static_cast<std::size_t>(m) * n);
    for (PetscInt j = 0; j < n; ++j) std::copy_n(array + static_cast<std::ptrdiff_t>(j) * ld, m, out.data() + static_cast<std::ptrdiff_t>(j) * m);
    PetscCall(MatDenseRestoreArrayRead(X, &array));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

public:
  PetscErrorCode computeBlocks(bool symmetric, bool uniform)
  {
    const Subdomain<K>       &subdomain = super::p_.subdomain;
    const vectorNeighbor     &map       = super::map_;
    const vectorNeighbor     &coarse    = subdomain.getCoarseMap();
    const MPI_Comm            comm      = subdomain.getCommunicator();
    const MPI_Comm            sub       = subdomain.getSubCommunicator();
    const Mat                 Z         = super::p_.getMat();
    const bool                leader    = subdomain.isLeader();
    const int                 n         = super::n_;
    int                       local     = super::local_, id;
    std::vector<int>          nuRank(uniform ? 0 : map.size());
    std::vector<MPI_Request>  rq(2 * map.size(), MPI_REQUEST_NULL);
    std::vector<unsigned int> sendDispls(map.size() + 1), recvDispls(map.size() + 1), neighborIndex(&coarse == &map ? 0 : map.size());
    K                        *send = nullptr, *recv = nullptr;
    Mat                       P = nullptr;

    PetscFunctionBeginUser;
    PetscCallMPI(MPI_Comm_rank(comm, &id));
    id = subdomain.getSubdomainOfRank(id);
    if (!uniform) {
      for (unsigned short i = 0; i < map.size(); ++i) {
        PetscCallMPI(MPI_Irecv(nuRank.data() + i, 1, MPI_INT, map[i].first, 7, comm, rq.data() + i));
        PetscCallMPI(MPI_Isend(&local, 1, MPI_INT, map[i].first, 7, comm, rq.data() + map.size() + i));
      }
      PetscCallMPI(MPI_Waitall(static_cast<int>(rq.size()), rq.data(), MPI_STATUSES_IGNORE));
      neighborSizes_.resize(coarse.size());
      if (&coarse == &map) std::copy(nuRank.cbegin(), nuRank.cend(), neighborSizes_.begin());
      else {
        std::vector<int> pairs, counts(coarse.size(), -1);
        int             *all, total;
        pairs.reserve(2 * map.size());
        for (unsigned short i = 0; i < map.size(); ++i) {
          pairs.emplace_back(subdomain.getSubdomainOfRank(map[i].first));
          pairs.emplace_back(nuRank[i]);
        }
        PetscCall(allgatherConcat(pairs.data(), static_cast<int>(pairs.size()), MPI_INT, sub, &total, &all));
        for (int i = 0; i < total; i += 2) {
          vectorNeighbor::const_iterator it = std::lower_bound(coarse.cbegin(), coarse.cend(), all[i],
                                                               [](const pairNeighbor &lhs, int rhs) { return static_cast<int>(lhs.first) < rhs; });
          PetscCheck(it != coarse.cend() && it->first == all[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Subdomain %d is not a neighbor of subdomain %d", all[i], id);
          counts[std::distance(coarse.cbegin(), it)] = all[i + 1];
        }
        PetscCall(PetscFree(all));
        for (std::size_t k = 0; k < coarse.size(); ++k) {
          PetscCheck(counts[k] >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing number of deflation vectors of subdomain %d", static_cast<int>(coarse[k].first));
          neighborSizes_[k] = counts[k];
        }
      }
    }
    if (local) {
      PetscBool host, sequential;

      PetscCall(PetscObjectTypeCompare((PetscObject)Z, MATSEQDENSE, &host));
      PetscCall(PetscObjectTypeCompareAny((PetscObject)A_, &sequential, MATSEQAIJ, MATSEQBAIJ, MATSEQSBAIJ, ""));
      if (host && sequential) {
        MatInfo info;

        PetscCall(MatGetInfo(A_, MAT_LOCAL, &info));
        // Scale the basis only when it requires no more scalar storage/work than the sparse copy.
        sequential = PetscBool(static_cast<PetscLogDouble>(n) * local <= info.nz_used);
      }
      if (host && sequential) {
        Mat scaled;

        PetscCall(MatDuplicate(Z, MAT_COPY_VALUES, &scaled));
        PetscCall(MatDiagonalScale(scaled, subdomain.getScalingVec(), nullptr));
        PetscCall(MatMatMult(A_, scaled, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &P));
        PetscCall(MatProductClear(P));
        PetscCall(MatDestroy(&scaled));
      } else {
        PetscCall(setupScaledMatrix());
        PetscCall(MatMatMult(C_, Z, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &P));
        PetscCall(MatProductClear(P));
        PetscCall(MatDestroy(&C_));
      }
    }
    sendDispls[0] = recvDispls[0] = 0;
    unsigned int scratchSize      = 0;
    int          receives         = 0;
    for (unsigned short i = 0; i < map.size(); ++i) {
      const int neighbor = subdomain.getSubdomainOfRank(map[i].first);
      const int nu       = uniform ? local : nuRank[i];
      const int overlap  = map[i].second.size();
      if (&coarse != &map) {
        vectorNeighbor::const_iterator it = std::lower_bound(coarse.cbegin(), coarse.cend(), neighbor,
                                                             [](const pairNeighbor &lhs, int rhs) { return static_cast<int>(lhs.first) < rhs; });
        PetscCheck(it != coarse.cend() && it->first == neighbor, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Subdomain %d is not a neighbor of subdomain %d", neighbor,
                   id);
        neighborIndex[i] = std::distance(coarse.cbegin(), it);
      }
      sendDispls[i + 1] = sendDispls[i] + ((!symmetric || neighbor < id) && nu ? overlap * local : 0);
      recvDispls[i + 1] = recvDispls[i] + ((!symmetric || neighbor > id) && local ? overlap * nu : 0);
      if (recvDispls[i + 1] != recvDispls[i]) {
        ++receives;
        scratchSize = std::max(scratchSize, static_cast<unsigned int>(overlap * (overlap == n ? std::min(local, nu) : local)));
      }
    }
    PetscCall(PetscMalloc2(sendDispls.back(), &send, recvDispls.back(), &recv));
    selfBlock_.clear();
    {
      const PetscScalar *array = nullptr;
      PetscInt           lda   = n;
      if (sendDispls.back()) {
        PetscCall(MatDenseGetArrayRead(P, &array));
        PetscCall(MatDenseGetLDA(P, &lda));
      }
      for (unsigned short i = 0; i < map.size(); ++i) {
        const int size = map[i].second.size();
        if (recvDispls[i + 1] != recvDispls[i])
          PetscCallMPI(MPI_Irecv(recv + recvDispls[i], recvDispls[i + 1] - recvDispls[i], Wrapper<K>::mpi_type(), map[i].first, 8, comm, rq.data() + i));
        if (sendDispls[i + 1] != sendDispls[i]) {
          for (int j = 0; j < local; ++j)
            Wrapper<K>::gthr(size, array + static_cast<std::ptrdiff_t>(j) * lda, send + sendDispls[i] + static_cast<std::ptrdiff_t>(j) * size,
                             map[i].second.data());
          PetscCallMPI(
            MPI_Isend(send + sendDispls[i], sendDispls[i + 1] - sendDispls[i], Wrapper<K>::mpi_type(), map[i].first, 8, comm, rq.data() + map.size() + i));
        }
      }
      if (local) {
        Mat X;

        if (sendDispls.back()) PetscCall(MatDenseRestoreArrayRead(P, &array));
        PetscCall(MatDiagonalScale(P, subdomain.getScalingVec(), nullptr));
        PetscCall(MatConjugate(P));
        PetscCall(MatTransposeMatMult(Z, P, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &X));
        PetscCall(MatConjugate(X));
        if (leader) PetscCall(copyLocalBlock(X, local, local, selfBlock_));
        PetscCall(MatDestroy(&X));
      }
    }
    PetscCall(MatDestroy(&P));
    if (local) {
      std::vector<unsigned char> assembled(coarse.size(), 0);
      const PetscScalar         *array = nullptr, *d = nullptr;
      PetscInt                   ldz     = n;
      K                         *scratch = nullptr;
      int                        size;

      blockDispls_.resize(coarse.size() + 1);
      blockDispls_[0] = 0;
      for (std::size_t k = 0; k < coarse.size(); ++k)
        blockDispls_[k + 1] = blockDispls_[k] + (!symmetric || coarse[k].first > id ? local * (uniform ? local : neighborSizes_[k]) : 0);
      blocks_.reset(new K[blockDispls_.back()]);
      if (receives) {
        PetscCall(MatDenseGetArrayRead(Z, &array));
        PetscCall(MatDenseGetLDA(Z, &ldz));
        PetscCall(VecGetArrayRead(subdomain.getScalingVec(), &d));
        PetscCall(PetscMalloc1(scratchSize, &scratch));
      }
      for (int received = 0; received < receives; ++received) {
        int index;

        PetscCallMPI(MPI_Waitany(static_cast<int>(map.size()), rq.data(), &index, MPI_STATUS_IGNORE));
        const unsigned int      k       = &coarse == &map ? static_cast<unsigned int>(index) : neighborIndex[index];
        const int               nu      = uniform ? local : nuRank[index];
        const std::vector<int> &overlap = map[index].second;
        const int               m       = overlap.size();
        const K *const          in      = recv + recvDispls[index];
        const K *const          beta    = assembled[k] ? &(Wrapper<K>::d_1) : &(Wrapper<K>::d_0);
        if (m == n && nu <= local) {
          for (int j = 0; j < nu; ++j)
            for (int r = 0; r < m; ++r) scratch[j * m + overlap[r]] = d[overlap[r]] * in[j * m + r];
          PetscBLASInt lda;
          PetscCall(PetscBLASIntCast(ldz, &lda));
          Blas<K>::gemm(&(Wrapper<K>::transc), "N", &local, &nu, &m, &(Wrapper<K>::d_1), array, &lda, scratch, &m, beta, blocks_.get() + blockDispls_[k],
                        &local);
        } else {
          for (int j = 0; j < local; ++j)
            for (int r = 0; r < m; ++r) scratch[j * m + r] = d[overlap[r]] * array[j * ldz + overlap[r]];
          Blas<K>::gemm(&(Wrapper<K>::transc), "N", &local, &nu, &m, &(Wrapper<K>::d_1), scratch, &m, in, &m, beta, blocks_.get() + blockDispls_[k], &local);
        }
        assembled[k] = 1;
      }
      if (receives) {
        PetscCall(PetscFree(scratch));
        PetscCall(VecRestoreArrayRead(subdomain.getScalingVec(), &d));
        PetscCall(MatDenseRestoreArrayRead(Z, &array));
      }
      for (std::size_t k = 0; k < coarse.size(); ++k)
        if (!assembled[k]) std::fill_n(blocks_.get() + blockDispls_[k], blockDispls_[k + 1] - blockDispls_[k], K());
      PetscCallMPI(MPI_Comm_size(sub, &size));
      if (size > 1 && blockDispls_.back())
        PetscCallMPI(MPI_Reduce(leader ? MPI_IN_PLACE : blocks_.get(), blocks_.get(), static_cast<int>(blockDispls_.back()), MPIU_SCALAR, MPIU_SUM, 0, sub));
      if (!leader) blocks_.reset();
    }
    if (!map.empty()) PetscCallMPI(MPI_Waitall(static_cast<int>(map.size()), rq.data() + map.size(), MPI_STATUSES_IGNORE));
    PetscCall(PetscFree2(send, recv));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  unsigned short *getNeighborSizes() { return neighborSizes_.empty() ? nullptr : neighborSizes_.data(); }
  void            initialize(unsigned short, unsigned short s, K *values = nullptr, unsigned int selfBlockOffset = 0, int rowStride = 0)
  {
    values_          = values;
    selfBlockOffset_ = selfBlockOffset;
    rowStride_       = rowStride;
    super::signed_   = s;
  }
  template <char S, bool U>
  void applyToNeighbor(const unsigned short *)
  {
    copySelfBlock<S, U>(selfBlock_.data());
  }
  template <char S, bool U>
  void assembleForMain(K *C, const CoarseOffsets &offsets, const NeighborSizes &neighborSizes)
  {
    copyNeighborBlocks<S, U, false>(C, offsets, neighborSizes);
  }
  template <char S, bool U, class T>
  void applyFromNeighborMain(T *J, K *C, const CoarseOffsets &offsets, const NeighborSizes &neighborSizes)
  {
    copyNeighborBlocks<S, U, true>(C, offsets, neighborSizes, J);
  }
  void finishExchange()
  {
    std::vector<K>().swap(selfBlock_);
    blocks_.reset();
    std::vector<unsigned int>().swap(blockDispls_);
    std::vector<unsigned short>().swap(neighborSizes_);
  }

private:
  template <char S, bool U, bool isMain, class T = unsigned int>
  void copyNeighborBlocks(K *C, const CoarseOffsets &offsets, const NeighborSizes &neighborSizes, T *J = nullptr)
  {
    const unsigned short skippedLowerNeighborCount = neighborSizes.skippedLowerNeighborCount();
    const unsigned short neighborCount             = static_cast<unsigned short>(super::sparsity_.size()) - skippedLowerNeighborCount;
    for (unsigned short neighborIndex = 0; neighborIndex < neighborCount; ++neighborIndex) {
      const K *block = super::local_ ? blocks_.get() + blockDispls_[neighborIndex + skippedLowerNeighborCount] : nullptr;
      copyNeighborBlock<S, U, isMain>(block, C, neighborIndex, super::sparsity_[neighborIndex + skippedLowerNeighborCount], offsets, neighborSizes, J);
    }
  }

public:
    #endif
  #else
  MatrixMultiplication(const Preconditioner &p, const unsigned short &c, const unsigned int &max, Types...) :
    super(p, c, max), A_(p.getMatrix()), C_(), D_(p.getScaling())
  {
    static_assert(sizeof...(Types) == 0, "Wrong constructor");
  }
  void initialize(unsigned int k, K *&work, unsigned short s)
  {
    if (A_->sym_) {
      std::vector<std::vector<std::pair<unsigned int, K>>> v(A_->n_);
      unsigned int                                         nnz = ((A_->nnz_ + A_->n_ - 1) / A_->n_) * 2;
      std::for_each(v.begin(), v.end(), [&](std::vector<std::pair<unsigned int, K>> &r) { r.reserve(nnz); });
      nnz = 0;
      for (unsigned int i = 0; i < A_->n_; ++i) {
        const underlying_type<K> scal = D_[i];
        unsigned int             j    = A_->ia_[i] - (HPDDM_NUMBERING == 'F');
        while (j < A_->ia_[i + 1] - (HPDDM_NUMBERING == 'F' ? 2 : 1)) {
          if (D_[A_->ja_[j] - (HPDDM_NUMBERING == 'F')] > HPDDM_EPS) {
            v[i].emplace_back(A_->ja_[j], A_->a_[j] * D_[A_->ja_[j] - (HPDDM_NUMBERING == 'F')]);
            ++nnz;
          }
          if (scal > HPDDM_EPS) {
            v[A_->ja_[j] - (HPDDM_NUMBERING == 'F')].emplace_back(i + (HPDDM_NUMBERING == 'F'), A_->a_[j] * scal);
            ++nnz;
          }
          ++j;
        }
        if (i != A_->ja_[j] - (HPDDM_NUMBERING == 'F')) {
          if (D_[A_->ja_[j] - (HPDDM_NUMBERING == 'F')] > HPDDM_EPS) {
            v[i].emplace_back(A_->ja_[j], A_->a_[j] * D_[A_->ja_[j] - (HPDDM_NUMBERING == 'F')]);
            ++nnz;
          }
        }
        if (scal > HPDDM_EPS) {
          v[A_->ja_[j] - (HPDDM_NUMBERING == 'F')].emplace_back(i + (HPDDM_NUMBERING == 'F'), A_->a_[j] * scal);
          ++nnz;
        }
      }
      C_         = new MatrixCSR<K>(A_->n_, A_->n_, nnz, false);
      C_->ia_[0] = (Wrapper<K>::I == 'F');
      nnz        = 0;
    #ifdef _OPENMP
      #pragma omp parallel for schedule(static, HPDDM_GRANULARITY)
    #endif
      for (unsigned int i = 0; i < A_->n_; ++i)
        std::sort(v[i].begin(), v[i].end(), [](const std::pair<unsigned int, K> &lhs, const std::pair<unsigned int, K> &rhs) { return lhs.first < rhs.first; });
      for (unsigned int i = 0; i < A_->n_; ++i) {
        for (const std::pair<unsigned int, K> &p : v[i]) {
          C_->ja_[nnz]  = p.first + (Wrapper<K>::I == 'F' && HPDDM_NUMBERING != Wrapper<K>::I);
          C_->a_[nnz++] = p.second;
        }
        C_->ia_[i + 1] = nnz + (Wrapper<K>::I == 'F');
      }
    } else {
      C_               = new MatrixCSR<K>(A_->n_, A_->n_, A_->nnz_, false);
      C_->ia_[0]       = (Wrapper<K>::I == 'F');
      unsigned int nnz = 0;
      for (unsigned int i = 0; i < A_->n_; ++i) {
        for (unsigned int j = A_->ia_[i] - (HPDDM_NUMBERING == 'F'); j < A_->ia_[i + 1] - (HPDDM_NUMBERING == 'F'); ++j)
          if (D_[A_->ja_[j] - (HPDDM_NUMBERING == 'F')] > HPDDM_EPS) {
            C_->ja_[nnz]  = A_->ja_[j] + (Wrapper<K>::I == 'F' && HPDDM_NUMBERING != Wrapper<K>::I);
            C_->a_[nnz++] = A_->a_[j] * D_[A_->ja_[j] - (HPDDM_NUMBERING == 'F')];
          }
        C_->ia_[i + 1] = nnz + (Wrapper<K>::I == 'F');
      }
      C_->nnz_ = nnz;
    }
    work           = new K[2 * static_cast<std::size_t>(k)];
    work_          = work + k;
    super::signed_ = s;
  }
  #endif
  template <char S, bool U, class T>
  void applyToNeighbor(T &in, K *&work, MPI_Request *&rq, const unsigned short *info, T = nullptr, MPI_Request * = nullptr)
  {
    if (super::local_)
  #if !HPDDM_PETSC
      Wrapper<K>::template csrmm<Wrapper<K>::I>(false, &(super::n_), &(super::local_), C_->a_, C_->ia_, C_->ja_, *super::deflation_, work_);
    delete C_;
  #else
    {
      Mat P;
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, super::local_, work_, &P));
      PetscCallVoid(MatMatMult(C_, super::p_.getMat(), MAT_REUSE_MATRIX, PETSC_DEFAULT, &P));
      PetscCallVoid(MatDestroy(&P));
    }
    PetscCallVoid(MatDestroy(&C_));
  #endif
    for (unsigned short i = 0; i < super::signed_; ++i) {
      if (U || info[i]) {
        for (unsigned short j = 0; j < super::local_; ++j)
          Wrapper<K>::gthr(super::map_[i].second.size(), work_ + j * super::n_, in[i] + j * super::map_[i].second.size(), super::map_[i].second.data());
        MPI_Isend(in[i], super::map_[i].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[i].first, 2, super::p_.subdomain.getCommunicator(),
                  rq++);
      }
    }
    Wrapper<K>::diag(super::n_, D_, work_, work, super::local_);
  }
  template <char S, bool U>
  void assembleForMain(K *C, const K *in, const int &coefficients, unsigned short index, K *arrayC, unsigned short *const &infoNeighbor = nullptr)
  {
    applyFromNeighbor<U>(in, index, arrayC, infoNeighbor);
    if (S != 'B')
      for (unsigned short j = 0; j < (U ? super::local_ : *infoNeighbor); ++j) {
        K *pt = C + j;
        for (unsigned short i = 0; i < super::local_; pt += coefficients - (S == 'S') * i++) *pt = arrayC[j * super::local_ + i];
      }
  }
  template <char S, char N, bool U, class T>
  void applyFromNeighborMain(const K *in, unsigned short index, T *I, T *J, K *C, int coefficients, unsigned int offsetI, unsigned int *offsetJ, K *arrayC,
                             unsigned short *const &infoNeighbor = nullptr)
  {
  #ifdef HPDDM_CSR_CO
    ignore(I, offsetI);
  #endif
    applyFromNeighbor<U>(in, index, S == 'B' && N == 'F' ? C : arrayC, infoNeighbor);
    unsigned int offset = (S == 'B' ? super::map_[index].first + (N == 'F') : (U ? super::map_[index].first * super::local_ + (N == 'F') : *offsetJ));
    if (S == 'B') {
      *J = offset;
      if (N == 'C') Wrapper<K>::template omatcopy<'T'>(super::local_, super::local_, arrayC, super::local_, C, super::local_);
    } else
      for (unsigned short i = 0; i < super::local_; ++i) {
        unsigned int l = coefficients * i - (S == 'S') * (i * (i - 1)) / 2;
        for (unsigned short j = 0; j < (U ? super::local_ : *infoNeighbor); ++j) {
  #ifndef HPDDM_CSR_CO
          I[l + j] = offsetI + i;
  #endif
          J[l + j] = offset + j;
          C[l + j] = arrayC[j * super::local_ + i];
        }
      }
  }
};
template <class Preconditioner, class K>
class MatrixAccumulation : public MatrixMultiplication<Preconditioner, K> {
private:
  std::vector<K>                                                                                            &overlap_;
  std::vector<std::vector<std::pair<unsigned short, unsigned short>>>                                       &reduction_;
  std::map<std::pair<unsigned short, unsigned short>, unsigned short>                                       &sizes_;
  std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>> &extra_;
  using super = MatrixMultiplication<Preconditioner, K>;
  int                                     *ldistribution_;
  int                                      size_;
  unsigned int                             sb_;
  std::vector<std::vector<unsigned short>> adjacency_;
  #if HPDDM_PETSC
    #ifdef PETSC_PCHPDDM_MAXLEVELS
  Mat matAP_ = nullptr;
    #endif
  #endif

  template <char S, bool U>
  void prepareAccumulation(const unsigned short *info, std::pair<int, int> **&block, std::vector<unsigned short> &interior,
                           std::vector<unsigned short> &overlap, std::vector<unsigned short> *&accumulate, MPI_Request *&rq)
  {
    std::vector<unsigned short> worker;
    int                         rank;
    MPI_Comm_rank(super::p_.subdomain.getCommunicator(), &rank);
    const unsigned short main = getMain(rank);
    worker.reserve(super::map_.size());
    for (unsigned short i = 0; i < super::map_.size(); ++i) worker.emplace_back(getMain(super::map_[i].first));
    if (super::map_.size()) {
      block  = new std::pair<int, int> *[super::map_.size()];
      *block = new std::pair<int, int>[super::map_.size() * super::map_.size()]();
      for (unsigned short i = 1; i < super::map_.size(); ++i) block[i] = *block + i * super::map_.size();
    }
    accumulate = new std::vector<unsigned short>[S == 'S' ? 2 * super::map_.size() : super::map_.size()];
    for (unsigned short i = 0; i < super::map_.size(); ++i) {
      accumulate[i].swap(adjacency_[i]);
      if (S == 'S') accumulate[super::map_.size() + i].reserve(super::connectivity_);
    }
    adjacency_.clear();
    rq = new MPI_Request[2 * super::map_.size()];
    std::vector<unsigned short> neighbors;
    neighbors.reserve(super::map_.size());
    for (const pairNeighbor &neighbor : super::map_) neighbors.emplace_back(neighbor.first);
    int m = 0;
    for (unsigned short i = 0; i < super::map_.size(); ++i) {
      if (worker[i] != main) {
        overlap.emplace_back(i);
        block[i][i] = std::make_pair(U == 1 ? super::local_ : info[i], U == 1 ? super::local_ : info[i]);
        if (block[i][i].first != 0) m += (U == 1 ? super::local_ * super::local_ : info[i] * info[i]);
        for (unsigned short j = (S != 'S' ? 0 : i + 1); j < super::map_.size(); ++j) {
          if (i != j && worker[j] != main && std::binary_search(accumulate[j].cbegin(), accumulate[j].cend(), super::map_[i].first)) {
            block[i][j] = std::make_pair(U == 1 ? super::local_ : info[i], U == 1 ? super::local_ : info[j]);
            if (block[i][j].first != 0 && block[i][j].second != 0) m += (U == 1 ? super::local_ * super::local_ : info[i] * info[j]);
          }
        }
        if (S == 'S' && i < super::signed_) m += (U == 1 ? super::local_ * super::local_ : super::local_ * info[i]);
      } else interior.emplace_back(i);
    }
    for (unsigned short &i : overlap) {
      unsigned short size = 0;
      for (unsigned short j = 0; j < accumulate[i].size(); ++j) {
        if (getMain(accumulate[i][j]) == worker[i]) {
          unsigned short *pt = std::lower_bound(neighbors.data(), neighbors.data() + neighbors.size(), accumulate[i][j]);
          if (pt != neighbors.data() + neighbors.size() && *pt == accumulate[i][j]) accumulate[i][size++] = std::distance(neighbors.data(), pt);
        }
        if (S == 'S' && getMain(accumulate[i][j]) == main) {
          unsigned short *pt = std::lower_bound(neighbors.data(), neighbors.data() + neighbors.size(), accumulate[i][j]);
          if (pt != neighbors.data() + neighbors.size() && *pt == accumulate[i][j])
            accumulate[super::map_.size() + i].emplace_back(std::distance(neighbors.data(), pt));
        }
      }
      accumulate[i].resize(size);
    }
    overlap_.resize(m);
  }

  template <bool U, bool transpose>
  unsigned short receiveAccumulation(const unsigned short *info, K *const *buff, MPI_Request *rq)
  {
    unsigned short count = 0;
    for (unsigned short i = 0; i < super::map_.size(); ++i) {
      const int entries = super::map_[i].second.size() * (transpose || U ? super::local_ : info[i]);
      if (entries) {
        MPI_Irecv(buff[i], entries, Wrapper<K>::mpi_type(), super::map_[i].first, transpose ? 21 : 20, super::p_.subdomain.getCommunicator(), rq + i);
        ++count;
      } else rq[i] = MPI_REQUEST_NULL;
    }
    return count;
  }
  template <bool U>
  unsigned short prepareAccumulationBuffers(const std::vector<typename Preconditioner::integer_type> &omap, const unsigned short *info,
                                            std::vector<std::vector<int>> &nmap, K **&buff, MPI_Request *rq)
  {
    std::size_t capacity = 0;
    nmap.resize(super::map_.size());
    for (unsigned short i = 0; i < super::map_.size(); ++i) {
      nmap[i].reserve(super::map_[i].second.size());
      for (const typename Preconditioner::integer_type &index : super::map_[i].second)
        nmap[i].emplace_back(std::distance(omap.cbegin(), std::lower_bound(omap.cbegin(), omap.cend(), index)));
      capacity += super::map_[i].second.size() * (U ? super::local_ : std::max(static_cast<unsigned short>(super::local_), info[i]));
    }
    buff               = new K *[std::max<std::size_t>(1, 2 * super::map_.size())];
    *buff              = new K[2 * capacity];
    std::size_t offset = 0;
    for (unsigned short i = 0; i < super::map_.size(); ++i) {
      buff[i]                      = *buff + offset;
      buff[super::map_.size() + i] = *buff + capacity + offset;
      offset += super::map_[i].second.size() * (U ? super::local_ : std::max(static_cast<unsigned short>(super::local_), info[i]));
    }
    return receiveAccumulation<U, false>(info, buff, rq);
  }
  void sendAccumulation(const K *values, const int lda, K *const *buff, MPI_Request *rq)
  {
    for (unsigned short i = 0; i < super::map_.size(); ++i) {
      const int entries = super::map_[i].second.size() * super::local_;
      if (entries) {
        for (unsigned short j = 0; j < super::local_; ++j)
          Wrapper<K>::gthr(super::map_[i].second.size(), values + j * lda, buff[super::map_.size() + i] + j * super::map_[i].second.size(),
                           super::map_[i].second.data());
        MPI_Isend(buff[super::map_.size() + i], entries, Wrapper<K>::mpi_type(), super::map_[i].first, 20, super::p_.subdomain.getCommunicator(),
                  rq + super::map_.size() + i);
      } else rq[super::map_.size() + i] = MPI_REQUEST_NULL;
    }
  }

public:
  #ifdef PETSC_PCHPDDM_MAXLEVELS
  using OperatorBase<'s', Preconditioner, K>::computeBlocks;
  using OperatorBase<'s', Preconditioner, K>::getNeighborSizes;
  using super::assembleForMain;
  using super::applyFromNeighborMain;
  #endif

  HPDDM_CLASS_COARSE_OPERATOR(Solver, S, T) friend class CoarseOperator;
  template <typename First, typename Second, typename Third, typename Fourth, typename Fifth, typename... Rest>
  #if !HPDDM_PETSC
  MatrixAccumulation(const Preconditioner &p, const unsigned short &c, const unsigned int &max, First &, Second &arg2, Third &arg3, Fourth &arg4, Fifth &arg5,
                     Rest &...args) :
    super(p, c, max, args...), overlap_(arg2), reduction_(arg3), sizes_(arg4), extra_(arg5)
  {
    static_assert(std::is_same<std::remove_pointer_t<First>, typename Preconditioner::super::co_type>::value, "Wrong constructor");
  }
  #else
  MatrixAccumulation(const Preconditioner &p, const unsigned short &c, const unsigned int &max, Mat A, PC_HPDDM_Level *level, std::string &prefix, First &,
                     Second &arg2, Third &arg3, Fourth &arg4, Fifth &arg5, Rest &...args) :
    super(p, c, max, A, level, prefix, args...), overlap_(arg2), reduction_(arg3), sizes_(arg4), extra_(arg5)
  {
    static_assert(std::is_same<std::remove_pointer_t<First>, typename Preconditioner::super::co_type>::value, "Wrong constructor");
  }
    #ifdef PETSC_PCHPDDM_MAXLEVELS
  void initialize(unsigned short maxColumns, unsigned short s, K *values = nullptr, unsigned int selfBlockOffset = 0, int rowStride = 0)
  {
    super::initialize(maxColumns, s, values, selfBlockOffset, rowStride);
    super::createWorkMats(maxColumns);
    PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, super::local_, nullptr, &matAP_));
  }
  template <char S, bool U>
  void assembleForMain(K *C, const CoarseOffsets &offsets, const NeighborSizes &neighborSizes)
  {
    super::template assembleCoarseOperator<S, U, false>(C, offsets, neighborSizes);
  }
  template <char S, bool U, class T>
  void applyFromNeighborMain(T *J, K *C, const CoarseOffsets &offsets, const NeighborSizes &neighborSizes)
  {
    super::template assembleCoarseOperator<S, U, true>(C, offsets, neighborSizes, J);
  }
  void finishExchange()
  {
    OperatorBase<'s', Preconditioner, K>::finishExchange();
    super::destroyWorkMats();
    PetscCallVoid(MatDestroy(&matAP_));
  }
  template <char S, bool U>
  void applyToNeighbor(const unsigned short *neighborSizes)
  {
    K          **in     = super::p_.subdomain.getBuffer();
    K          **out    = (U || super::local_) ? in + super::signed_ : nullptr;
    MPI_Request *rqRecv = super::p_.subdomain.getRq() + super::signed_;
    MPI_Request *rq     = super::p_.subdomain.getRq();
    super::template prepareExchange<S, U>(in, out, neighborSizes, rqRecv);
    std::pair<int, int>        **block = nullptr;
    std::vector<unsigned short>  interior, overlap;
    std::vector<unsigned short> *accumulate;
    MPI_Request                 *rqLocal;
    int                          m;
    prepareAccumulation<S, U>(neighborSizes, block, interior, overlap, accumulate, rqLocal);
    std::vector<typename Preconditioner::integer_type> omap;
    PetscInt                                           n;
    PetscCallVoid(MatGetLocalSize(super::A_, &n, nullptr));
    {
      std::set<typename Preconditioner::integer_type> o;
      for (unsigned short i = 0; i < super::map_.size(); ++i) o.insert(super::map_[i].second.cbegin(), super::map_[i].second.cend());
      omap.reserve(o.size());
      std::copy(o.cbegin(), o.cend(), std::back_inserter(omap));
      IS                    is;
      PetscInt             *data;
      std::vector<PetscInt> imap;
      if (n == super::n_) {
        if (std::is_same<typename Preconditioner::integer_type, PetscInt>::value) data = reinterpret_cast<PetscInt *>(omap.data());
        else {
          data = new PetscInt[omap.size()];
          std::copy_n(omap.data(), omap.size(), data);
        }
        PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, omap.size(), data, PETSC_USE_POINTER, &is));
      } else {
        imap.reserve(omap.size());
        const PetscScalar *scaling;
        PetscCallVoid(VecGetArrayRead(super::p_.getScalingVec(), &scaling));
        std::for_each(omap.cbegin(), omap.cend(), [&scaling, &imap](const typename Preconditioner::integer_type &i) {
          if (HPDDM::abs(scaling[i]) > underlying_type<K>(HPDDM_EPS)) imap.emplace_back(i);
        });
        PetscCallVoid(VecRestoreArrayRead(super::p_.getScalingVec(), &scaling));
        PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, imap.size(), imap.data(), PETSC_USE_POINTER, &is));
      }
      PetscCallVoid(MatCreateSubMatrix(super::A_, is, is, MAT_INITIAL_MATRIX, &(super::C_)));
      PetscCallVoid(ISDestroy(&is));
      if (n == super::n_ && !std::is_same<typename Preconditioner::integer_type, PetscInt>::value) delete[] data;
    }
    std::vector<unsigned int> colOffset(super::map_.size() + 1, 0);
    for (unsigned short i = 0; i < super::map_.size(); ++i) colOffset[i + 1] = colOffset[i] + (U == 1 ? super::local_ : neighborSizes[i]);
    const unsigned int colsTotal = colOffset[super::map_.size()];
    Mat                matZ, matCZ;
    PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, omap.size(), colsTotal, nullptr, &matZ));
    PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, omap.size(), colsTotal, nullptr, &matCZ));
    std::vector<std::vector<int>> nmap;
    K                           **buff;
    const unsigned short          receives = prepareAccumulationBuffers<U>(omap, neighborSizes, nmap, buff, rqLocal);
    if (super::local_) {
      PetscCallVoid(MatCopy(super::p_.getMat(), super::matP_, SAME_NONZERO_PATTERN));
      PetscCallVoid(MatDiagonalScale(super::matP_, super::p_.getScalingVec(), nullptr));
    }
    {
      const K *pArray = nullptr;
      if (super::local_) PetscCallVoid(MatDenseGetArrayRead(super::matP_, &pArray));
      sendAccumulation(pArray, super::n_, buff, rqLocal);
      if (super::local_) PetscCallVoid(MatDenseRestoreArrayRead(super::matP_, &pArray));
    }
    if (n == super::n_) {
      PetscCallVoid(MatMatMult(super::A_, super::matP_, MAT_REUSE_MATRIX, PETSC_DEFAULT, &matAP_));
    } else {
      Mat                matZc, matAPc;
      PetscScalar       *zcArray, *apArray;
      const PetscScalar *pArray, *apcArray, *scaling;
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, n, super::local_, nullptr, &matZc));
      PetscCallVoid(MatDenseGetArray(matZc, &zcArray));
      PetscCallVoid(MatDenseGetArrayRead(super::matP_, &pArray));
      PetscCallVoid(VecGetArrayRead(super::p_.getScalingVec(), &scaling));
      for (PetscInt i = 0, k = 0; i < super::n_; ++i) {
        if (HPDDM::abs(scaling[i]) > underlying_type<K>(HPDDM_EPS)) {
          for (unsigned short j = 0; j < super::local_; ++j) zcArray[k + j * n] = pArray[i + j * super::n_];
          ++k;
        }
      }
      PetscCallVoid(VecRestoreArrayRead(super::p_.getScalingVec(), &scaling));
      PetscCallVoid(MatDenseRestoreArrayRead(super::matP_, &pArray));
      PetscCallVoid(MatDenseRestoreArray(matZc, &zcArray));
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, n, super::local_, nullptr, &matAPc));
      PetscCallVoid(MatMatMult(super::A_, matZc, MAT_REUSE_MATRIX, PETSC_DEFAULT, &matAPc));
      PetscCallVoid(MatDestroy(&matZc));
      PetscCallVoid(MatDenseGetArrayRead(matAPc, &apcArray));
      PetscCallVoid(VecGetArrayRead(super::p_.getScalingVec(), &scaling));
      PetscCallVoid(MatDenseGetArray(matAP_, &apArray));
      std::fill_n(apArray, static_cast<std::size_t>(super::local_) * super::n_, K());
      for (PetscInt i = 0, k = 0; i < super::n_; ++i) {
        if (HPDDM::abs(scaling[i]) > underlying_type<K>(HPDDM_EPS)) {
          for (unsigned short j = 0; j < super::local_; ++j) apArray[i + j * super::n_] = apcArray[k + j * n];
          ++k;
        }
      }
      PetscCallVoid(VecRestoreArrayRead(super::p_.getScalingVec(), &scaling));
      PetscCallVoid(MatDenseRestoreArray(matAP_, &apArray));
      PetscCallVoid(MatDenseRestoreArrayRead(matAPc, &apcArray));
      PetscCallVoid(MatDestroy(&matAPc));
    }
    {
      PetscScalar *zArray;
      PetscCallVoid(MatDenseGetArray(matZ, &zArray));
      for (unsigned short i = 0; i < receives; ++i) {
        int index;
        MPI_Waitany(super::map_.size(), rqLocal, &index, MPI_STATUS_IGNORE);
        for (unsigned short k = 0; k < (U ? super::local_ : neighborSizes[index]); ++k)
          Wrapper<K>::sctr(nmap[index].size(), buff[index] + k * nmap[index].size(), nmap[index].data(), zArray + (colOffset[index] + k) * omap.size());
      }
      PetscCallVoid(MatDenseRestoreArray(matZ, &zArray));
    }
    receiveAccumulation<U, true>(neighborSizes, buff, rqLocal);
    if (n == super::n_) {
      PetscCallVoid(MatMatMult(super::C_, matZ, MAT_REUSE_MATRIX, PETSC_DEFAULT, &matCZ));
    } else {
      std::vector<PetscInt> imap;
      const PetscScalar    *scaling;
      PetscCallVoid(VecGetArrayRead(super::p_.getScalingVec(), &scaling));
      for (typename std::vector<typename Preconditioner::integer_type>::const_iterator it = omap.cbegin(); it != omap.cend(); ++it) {
        if (HPDDM::abs(scaling[*it]) > underlying_type<K>(HPDDM_EPS)) imap.emplace_back(std::distance(omap.cbegin(), it));
      }
      PetscCallVoid(VecRestoreArrayRead(super::p_.getScalingVec(), &scaling));
      Mat          matZc, matCZc;
      PetscScalar *zcArray, *czArray;
      const K     *zArray;
      const K     *czcArray;
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, imap.size(), colsTotal, nullptr, &matZc));
      PetscCallVoid(MatDenseGetArray(matZc, &zcArray));
      PetscCallVoid(MatDenseGetArrayRead(matZ, &zArray));
      for (std::size_t i = 0; i < imap.size(); ++i)
        for (unsigned int j = 0; j < colsTotal; ++j) zcArray[i + j * imap.size()] = zArray[imap[i] + j * omap.size()];
      PetscCallVoid(MatDenseRestoreArrayRead(matZ, &zArray));
      PetscCallVoid(MatDenseRestoreArray(matZc, &zcArray));
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, imap.size(), colsTotal, nullptr, &matCZc));
      PetscCallVoid(MatMatMult(super::C_, matZc, MAT_REUSE_MATRIX, PETSC_DEFAULT, &matCZc));
      PetscCallVoid(MatDestroy(&matZc));
      PetscCallVoid(MatDenseGetArray(matCZ, &czArray));
      std::fill_n(czArray, static_cast<std::size_t>(colsTotal) * omap.size(), K());
      PetscCallVoid(MatDenseGetArrayRead(matCZc, &czcArray));
      for (std::size_t i = 0; i < imap.size(); ++i)
        for (unsigned int j = 0; j < colsTotal; ++j) czArray[imap[i] + j * omap.size()] = czcArray[i + j * imap.size()];
      PetscCallVoid(MatDenseRestoreArrayRead(matCZc, &czcArray));
      PetscCallVoid(MatDenseRestoreArray(matCZ, &czArray));
      PetscCallVoid(MatDestroy(&matCZc));
    }
    PetscCallVoid(MatDestroy(&(super::C_)));
    PetscCallVoid(MatConjugate(matZ));
    MPI_Waitall(super::map_.size(), rqLocal + super::map_.size(), MPI_STATUSES_IGNORE);
    {
      const K *czArray;
      PetscCallVoid(MatDenseGetArrayRead(matCZ, &czArray));
      for (unsigned short i = 0; i < super::map_.size(); ++i) {
        m = (U == 1 ? super::local_ : neighborSizes[i]);
        for (unsigned short j = 0; j < m; ++j)
          Wrapper<K>::gthr(nmap[i].size(), czArray + (colOffset[i] + j) * omap.size(), buff[super::map_.size() + i] + j * nmap[i].size(), nmap[i].data());
        if (m && !super::map_[i].second.empty())
          MPI_Isend(buff[super::map_.size() + i], super::map_[i].second.size() * m, Wrapper<K>::mpi_type(), super::map_[i].first, 21,
                    super::p_.subdomain.getCommunicator(), rqLocal + super::map_.size() + i);
        else rqLocal[super::map_.size() + i] = MPI_REQUEST_NULL;
      }
      PetscCallVoid(MatDenseRestoreArrayRead(matCZ, &czArray));
    }
    K *pt = overlap_.data();
    for (unsigned short i = 0; i < overlap.size(); ++i) {
      for (unsigned short j = 0; j < overlap.size(); ++j) {
        if (block[overlap[i]][overlap[j]].first != 0 && block[overlap[i]][overlap[j]].second != 0) {
          const int rows = block[overlap[i]][overlap[j]].first;
          const int cols = block[overlap[i]][overlap[j]].second;
          Mat       Zi, CZj, G;
          PetscCallVoid(MatDenseGetSubMatrix(matZ, 0, omap.size(), colOffset[overlap[i]], colOffset[overlap[i]] + rows, &Zi));
          PetscCallVoid(MatDenseGetSubMatrix(matCZ, 0, omap.size(), colOffset[overlap[j]], colOffset[overlap[j]] + cols, &CZj));
          PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, rows, cols, pt, &G));
          PetscCallVoid(MatTransposeMatMult(Zi, CZj, MAT_REUSE_MATRIX, PETSC_DEFAULT, &G));
          PetscCallVoid(MatDestroy(&G));
          PetscCallVoid(MatDenseRestoreSubMatrix(matCZ, &CZj));
          PetscCallVoid(MatDenseRestoreSubMatrix(matZ, &Zi));
          pt += (U == 1 ? super::local_ * super::local_ : neighborSizes[overlap[i]] * neighborSizes[overlap[j]]);
        }
      }
    }
    if (block) {
      delete[] *block;
      delete[] block;
    }
    MPI_Waitall(super::map_.size(), rqLocal, MPI_STATUSES_IGNORE);
    if (S == 'S') {
      std::vector<K> scratch(omap.size() * super::local_);
      std::vector<K> received(omap.size());
      Mat            matScratch;
      const K       *apArray;
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, omap.size(), super::local_, scratch.data(), &matScratch));
      PetscCallVoid(MatDenseGetArrayRead(matAP_, &apArray));
      for (unsigned short i = 0; i < overlap.size() && overlap[i] < super::signed_; ++i) {
        m = (U == 1 ? super::local_ : neighborSizes[overlap[i]]);
        if (m) {
          for (unsigned short nu = 0; nu < super::local_; ++nu)
            Wrapper<K>::gthr(omap.size(), apArray + nu * super::n_, scratch.data() + nu * omap.size(), omap.data());
          const std::vector<unsigned short> &r = accumulate[super::map_.size() + overlap[i]];
          for (unsigned short j : r) {
            for (unsigned short nu = 0; nu < super::local_; ++nu) {
              std::fill(received.begin(), received.end(), K());
              Wrapper<K>::sctr(nmap[j].size(), buff[j] + nu * nmap[j].size(), nmap[j].data(), received.data());
              for (int k : nmap[overlap[i]]) scratch[k + nu * omap.size()] += received[k];
            }
          }
          Mat Zi, G;
          PetscCallVoid(MatDenseGetSubMatrix(matZ, 0, omap.size(), colOffset[overlap[i]], colOffset[overlap[i]] + m, &Zi));
          PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, m, super::local_, pt, &G));
          PetscCallVoid(MatTransposeMatMult(Zi, matScratch, MAT_REUSE_MATRIX, PETSC_DEFAULT, &G));
          PetscCallVoid(MatDestroy(&G));
          PetscCallVoid(MatDenseRestoreSubMatrix(matZ, &Zi));
          pt += super::local_ * m;
        }
      }
      PetscCallVoid(MatDenseRestoreArrayRead(matAP_, &apArray));
      PetscCallVoid(MatDestroy(&matScratch));
    }
    PetscCallVoid(MatDestroy(&matZ));
    PetscCallVoid(MatDestroy(&matCZ));
    {
      const K       *apArray;
      std::vector<K> scratchOmap(omap.size());
      PetscCallVoid(MatDenseGetArrayRead(matAP_, &apArray));
      for (unsigned short i = 0; i < overlap.size() && overlap[i] < super::signed_; ++i) {
        if (U || neighborSizes[overlap[i]]) {
          for (unsigned short nu = 0; nu < super::local_; ++nu)
            Wrapper<K>::gthr(super::map_[overlap[i]].second.size(), apArray + nu * super::n_, in[overlap[i]] + nu * super::map_[overlap[i]].second.size(),
                             super::map_[overlap[i]].second.data());
          const std::vector<unsigned short> &r = accumulate[overlap[i]];
          for (unsigned short k : r) {
            for (unsigned short nu = 0; nu < super::local_; ++nu) {
              std::fill(scratchOmap.begin(), scratchOmap.end(), K());
              Wrapper<K>::sctr(nmap[k].size(), buff[k] + nu * nmap[k].size(), nmap[k].data(), scratchOmap.data());
              for (unsigned int j = 0; j < super::map_[overlap[i]].second.size(); ++j)
                in[overlap[i]][j + nu * super::map_[overlap[i]].second.size()] += scratchOmap[nmap[overlap[i]][j]];
            }
          }
          MPI_Isend(in[overlap[i]], super::map_[overlap[i]].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[overlap[i]].first, 2,
                    super::p_.subdomain.getCommunicator(), rq + overlap[i]);
        } else rq[overlap[i]] = MPI_REQUEST_NULL;
      }
      PetscCallVoid(MatDenseRestoreArrayRead(matAP_, &apArray));
    }
    {
      K *apArray;
      PetscCallVoid(MatDenseGetArray(matAP_, &apArray));
      for (unsigned short i : interior) {
        for (unsigned short k = 0; k < super::local_; ++k)
          for (unsigned int j = 0; j < super::map_[i].second.size(); ++j)
            apArray[super::map_[i].second[j] + k * super::n_] += buff[i][j + k * super::map_[i].second.size()];
      }
      PetscCallVoid(MatDenseRestoreArray(matAP_, &apArray));
    }
    {
      const K *apArray;
      PetscCallVoid(MatDenseGetArrayRead(matAP_, &apArray));
      for (unsigned short i = 0; i < interior.size() && interior[i] < super::signed_; ++i) {
        if (U || neighborSizes[interior[i]]) {
          for (unsigned short nu = 0; nu < super::local_; ++nu)
            Wrapper<K>::gthr(super::map_[interior[i]].second.size(), apArray + nu * super::n_, in[interior[i]] + nu * super::map_[interior[i]].second.size(),
                             super::map_[interior[i]].second.data());
          MPI_Isend(in[interior[i]], super::map_[interior[i]].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[interior[i]].first, 2,
                    super::p_.subdomain.getCommunicator(), rq + interior[i]);
        } else rq[interior[i]] = MPI_REQUEST_NULL;
      }
      PetscCallVoid(MatDenseRestoreArrayRead(matAP_, &apArray));
    }
    rq += super::signed_;
    delete[] accumulate;
    if (super::local_) {
      PetscCallVoid(MatCopy(matAP_, super::matP_, SAME_NONZERO_PATTERN));
      PetscCallVoid(MatDiagonalScale(super::matP_, super::p_.getScalingVec(), nullptr));
    }
    MPI_Waitall(super::map_.size(), rqLocal + super::map_.size(), MPI_STATUSES_IGNORE);
    delete[] rqLocal;
    delete[] *buff;
    delete[] buff;
    super::template assembleSelfBlock<S, U>(super::matP_);
    super::sendRequestsEnd_ = rq;
  }
    #endif
  #endif
  static constexpr bool factorize_ = false;
  int                   getMain(const int rank) const
  {
    int *it = std::lower_bound(ldistribution_, ldistribution_ + size_, rank);
    if (it == ldistribution_ + size_) return size_ - 1;
    else {
      if (*it != rank && it != ldistribution_) --it;
      return std::distance(ldistribution_, it);
    }
  }
  template <char S, bool U>
  void setPattern(int *ldistribution, const int p, const int sizeSplit, MPI_Comm scatterComm, unsigned short *const *const split = nullptr,
                  const unsigned short *const world = nullptr)
  {
    ldistribution_ = ldistribution;
    size_          = p;
    int rank, size;
    MPI_Comm_rank(super::p_.subdomain.getCommunicator(), &rank);
    MPI_Comm_size(super::p_.subdomain.getCommunicator(), &size);
    std::vector<unsigned short> neighbors;
    if (p > 1) {
      neighbors.reserve(super::map_.size());
      for (const pairNeighbor &neighbor : super::map_) neighbors.emplace_back(neighbor.first);
    }
    adjacency_.resize(super::map_.size());
    std::vector<MPI_Request> requests(2 * neighbors.size());
    for (unsigned short i = 0; i < neighbors.size(); ++i) {
      adjacency_[i].resize(super::connectivity_);
      MPI_Irecv(adjacency_[i].data(), super::connectivity_, MPI_UNSIGNED_SHORT, neighbors[i], 123, super::p_.subdomain.getCommunicator(), requests.data() + i);
      MPI_Isend(neighbors.data(), neighbors.size(), MPI_UNSIGNED_SHORT, neighbors[i], 123, super::p_.subdomain.getCommunicator(),
                requests.data() + neighbors.size() + i);
    }
    for (unsigned short i = 0; i < neighbors.size(); ++i) {
      int        index, count;
      MPI_Status status;
      MPI_Waitany(neighbors.size(), requests.data(), &index, &status);
      MPI_Get_count(&status, MPI_UNSIGNED_SHORT, &count);
      adjacency_[index].resize(count);
    }
    if (!neighbors.empty()) MPI_Waitall(neighbors.size(), requests.data() + neighbors.size(), MPI_STATUSES_IGNORE);
    // Only the edges between a worker's external neighbors are needed by its coarse main.
    std::vector<unsigned int> edges;
    const int                 main = getMain(rank);
    for (unsigned short j = 0; j < neighbors.size(); ++j) {
      if (getMain(neighbors[j]) != main) {
        for (unsigned short k = j + 1; k < neighbors.size(); ++k)
          if (getMain(neighbors[k]) != main && std::binary_search(adjacency_[j].cbegin(), adjacency_[j].cend(), neighbors[k]))
            edges.emplace_back(static_cast<unsigned int>(neighbors[j]) * size + neighbors[k]);
      }
    }
    const int                 count = edges.size();
    std::vector<int>          counts(split ? sizeSplit : 0), offsets(split ? sizeSplit + 1 : 0);
    std::vector<unsigned int> pattern;
    if (p > 1 && sizeSplit > 1) {
      MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, scatterComm);
      if (split) {
        std::partial_sum(counts.cbegin(), counts.cend(), offsets.begin() + 1);
        pattern.resize(offsets.back());
      }
      MPI_Gatherv(edges.data(), count, MPI_UNSIGNED, pattern.data(), counts.data(), offsets.data(), MPI_UNSIGNED, 0, scatterComm);
    } else if (split && sizeSplit == 1) {
      counts[0]  = count;
      offsets[1] = count;
      pattern.swap(edges);
    }
    if (split) {
      int self = getMain(rank);
      reduction_.resize(sizeSplit);
      for (unsigned short i = 0; i < sizeSplit; ++i) {
        for (unsigned short j = 0; j < split[i][0]; ++j) {
          if (getMain(split[i][(U != 1 ? 3 : 1) + j]) != self) {
            sizes_[std::make_pair(split[i][(U != 1 ? 3 : 1) + j], split[i][(U != 1 ? 3 : 1) + j])] = (U != 1 ? world[split[i][(U != 1 ? 3 : 1) + j]]
                                                                                                             : super::local_);
            reduction_[i].emplace_back(split[i][(U != 1 ? 3 : 1) + j], split[i][(U != 1 ? 3 : 1) + j]);
            if (S == 'S' && split[i][(U != 1 ? 3 : 1) + j] < rank + i) {
              std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>::iterator it = extra_.find(i);
              if (it == extra_.end()) {
                std::pair<std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>::iterator, bool>
                  p                     = extra_.emplace(i, std::forward_as_tuple(0, 0, std::vector<unsigned short>()));
                it                      = p.first;
                std::get<0>(it->second) = (U != 1 ? world[rank + i] : 1);
                std::get<1>(it->second) = (U != 1 ? std::accumulate(world + rank, world + rank + i, 0) : i);
              }
              std::get<2>(it->second).emplace_back(split[i][(U != 1 ? 3 : 1) + j]);
            }
            for (unsigned short k = j + 1; k < split[i][0]; ++k) {
              if (std::binary_search(pattern.cbegin() + offsets[i], pattern.cbegin() + offsets[i + 1],
                                     static_cast<unsigned int>(split[i][(U != 1 ? 3 : 1) + j]) * size + split[i][(U != 1 ? 3 : 1) + k])) {
                sizes_[std::make_pair(split[i][(U != 1 ? 3 : 1) + j], split[i][(U != 1 ? 3 : 1) + k])] = (U != 1 ? world[split[i][(U != 1 ? 3 : 1) + k]]
                                                                                                                 : super::local_);
                if (S != 'S')
                  sizes_[std::make_pair(split[i][(U != 1 ? 3 : 1) + k], split[i][(U != 1 ? 3 : 1) + j])] = (U != 1 ? world[split[i][(U != 1 ? 3 : 1) + j]]
                                                                                                                   : super::local_);
                reduction_[i].emplace_back(split[i][(U != 1 ? 3 : 1) + j], split[i][(U != 1 ? 3 : 1) + k]);
                if (S != 'S') reduction_[i].emplace_back(split[i][(U != 1 ? 3 : 1) + k], split[i][(U != 1 ? 3 : 1) + j]);
              }
            }
          }
        }
        std::sort(reduction_[i].begin(), reduction_[i].end());
        if (S == 'S') {
          const unsigned short first = std::distance(split[i] + (U != 1 ? 3 : 1),
                                                     std::upper_bound(split[i] + (U != 1 ? 3 : 1), split[i] + (U != 1 ? 3 : 1) + split[i][0], rank + i));
          split[i][0] -= first;
          for (unsigned short j = 0; j < split[i][0]; ++j) split[i][(U != 1 ? 3 : 1) + j] = split[i][(U != 1 ? 3 : 1) + first + j];
        }
      }
    }
  }
  void initialize(unsigned int k, K *&work, unsigned short s)
  {
    sb_  = k;
    work = new K[2 * static_cast<std::size_t>(sb_)];
  #if !HPDDM_PETSC
    if (HPDDM_NUMBERING != Wrapper<K>::I) {
      if (HPDDM_NUMBERING == 'F') {
        std::for_each(super::A_->ja_, super::A_->ja_ + super::A_->nnz_, [](int &i) { --i; });
        std::for_each(super::A_->ia_, super::A_->ia_ + super::A_->n_ + 1, [](int &i) { --i; });
      } else {
        std::for_each(super::A_->ja_, super::A_->ja_ + super::A_->nnz_, [](int &i) { ++i; });
        std::for_each(super::A_->ia_, super::A_->ia_ + super::A_->n_ + 1, [](int &i) { ++i; });
      }
    }
  #endif
    super::work_ = work + sb_;
    std::fill_n(super::work_, sb_, K());
    super::signed_ = s;
  }
  template <char S, bool U, class T>
  void applyToNeighbor(T &in, K *&work, MPI_Request *&rs, const unsigned short *info, T = nullptr, MPI_Request * = nullptr)
  {
    std::pair<int, int>        **block = nullptr;
    std::vector<unsigned short>  interior, overlap;
    std::vector<unsigned short> *accumulate;
    MPI_Request                 *rq;
    int                          m;
    prepareAccumulation<S, U>(info, block, interior, overlap, accumulate, rq);
    std::vector<typename Preconditioner::integer_type> omap;
  #if HPDDM_PETSC
    PetscInt n;
    PetscCallVoid(MatGetLocalSize(super::A_, &n, nullptr));
  #endif
    {
      std::set<typename Preconditioner::integer_type> o;
      for (unsigned short i = 0; i < super::map_.size(); ++i) o.insert(super::map_[i].second.cbegin(), super::map_[i].second.cend());
      omap.reserve(o.size());
      std::copy(o.cbegin(), o.cend(), std::back_inserter(omap));
  #if !HPDDM_PETSC
      int *ia = new int[omap.size() + 1];
      ia[0]   = (Wrapper<K>::I == 'F');
      std::vector<std::pair<int, K>> restriction;
      restriction.reserve(super::A_->nnz_);
      int nnz = ia[0];
      for (int i = 0; i < omap.size(); ++i) {
        std::vector<int>::const_iterator it = omap.cbegin();
        for (int j = super::A_->ia_[omap[i]] - (Wrapper<K>::I == 'F'); j < super::A_->ia_[omap[i] + 1] - (Wrapper<K>::I == 'F'); ++j) {
          it = std::lower_bound(it, omap.cend(), super::A_->ja_[j] - (Wrapper<K>::I == 'F'));
          if (it != omap.cend() && *it == super::A_->ja_[j] - (Wrapper<K>::I == 'F') && std::abs(super::A_->a_[j]) > HPDDM_EPS) {
            restriction.emplace_back(std::distance(omap.cbegin(), it) + (Wrapper<K>::I == 'F'), super::A_->a_[j]);
            ++nnz;
          }
        }
        ia[i + 1] = nnz;
      }
      int *ja = new int[nnz - (Wrapper<K>::I == 'F')];
      K   *a  = new K[nnz - (Wrapper<K>::I == 'F')];
      for (int i = 0; i < nnz - (Wrapper<K>::I == 'F'); ++i) {
        ja[i] = restriction[i].first;
        a[i]  = restriction[i].second;
      }
      super::C_ = new MatrixCSR<K>(omap.size(), omap.size(), nnz - (Wrapper<K>::I == 'F'), a, ia, ja, super::A_->sym_, true);
  #else
      IS                    is;
      PetscInt             *data;
      std::vector<PetscInt> imap;
      if (n == super::n_) {
        if (std::is_same<typename Preconditioner::integer_type, PetscInt>::value) data = reinterpret_cast<PetscInt *>(omap.data());
        else {
          data = new PetscInt[omap.size()];
          std::copy_n(omap.data(), omap.size(), data);
        }
        PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, omap.size(), data, PETSC_USE_POINTER, &is));
      } else {
        imap.reserve(omap.size());
        const PetscScalar *scaling;
        PetscCallVoid(VecGetArrayRead(super::p_.getScalingVec(), &scaling));
        std::for_each(omap.cbegin(), omap.cend(), [&scaling, &imap](const typename Preconditioner::integer_type &i) {
          if (HPDDM::abs(scaling[i]) > underlying_type<K>(HPDDM_EPS)) imap.emplace_back(i);
        });
        PetscCallVoid(VecRestoreArrayRead(super::p_.getScalingVec(), &scaling));
        PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, imap.size(), imap.data(), PETSC_USE_POINTER, &is));
      }
      PetscCallVoid(MatCreateSubMatrix(super::A_, is, is, MAT_INITIAL_MATRIX, &(super::C_)));
      PetscCallVoid(ISDestroy(&is));
      if (n == super::n_ && !std::is_same<typename Preconditioner::integer_type, PetscInt>::value) delete[] data;
  #endif
    }
    K **tmp = new K *[2 * super::map_.size()];
    *tmp    = new K[omap.size() * (U == 1 ? super::map_.size() * super::local_ : std::max(super::local_, std::accumulate(info, info + super::map_.size(), 0))) +
                    std::max(static_cast<int>(omap.size() * (U == 1 ? super::map_.size() * super::local_
                                                                    : std::max(super::local_, std::accumulate(info, info + super::map_.size(), 0)))),
                             super::n_ * super::local_)]();
    for (unsigned short i = 1; i < super::map_.size(); ++i) tmp[i] = tmp[i - 1] + omap.size() * (U == 1 ? super::local_ : info[i - 1]);
    if (super::map_.size()) {
      tmp[super::map_.size()] = tmp[super::map_.size() - 1] + omap.size() * (U == 1 ? super::local_ : info[super::map_.size() - 1]);
      for (unsigned short i = 1; i < super::map_.size(); ++i) tmp[super::map_.size() + i] = tmp[super::map_.size() + i - 1] + std::distance(tmp[i - 1], tmp[i]);
    }
    std::vector<std::vector<int>> nmap;
    K                           **buff;
    const unsigned short          receives = prepareAccumulationBuffers<U>(omap, info, nmap, buff, rq);
  #if !HPDDM_PETSC
    if (super::local_) Wrapper<K>::diag(super::n_, super::D_, *super::deflation_, *tmp, super::local_);
  #else
    if (super::local_) {
      Mat scaled;
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, super::local_, *tmp, &scaled));
      PetscCallVoid(MatCopy(super::p_.getMat(), scaled, SAME_NONZERO_PATTERN));
      PetscCallVoid(MatDiagonalScale(scaled, super::p_.getScalingVec(), nullptr));
      PetscCallVoid(MatDestroy(&scaled));
    }
  #endif
    sendAccumulation(*tmp, super::n_, buff, rq);
  #if !HPDDM_PETSC
    Wrapper<K>::template csrmm<Wrapper<K>::I>(super::A_->sym_, &(super::n_), &(super::local_), super::A_->a_, super::A_->ia_, super::A_->ja_, *tmp,
                                              super::work_);
  #else
    Mat Z, P;
    if (n == super::n_) {
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, super::local_, *tmp, &Z));
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, super::local_, super::work_, &P));
    } else {
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, n, super::local_, nullptr, &Z));
      PetscScalar       *array;
      const PetscScalar *scaling;
      PetscCallVoid(MatDenseGetArrayWrite(Z, &array));
      PetscCallVoid(VecGetArrayRead(super::p_.getScalingVec(), &scaling));
      for (PetscInt i = 0, k = 0; i < super::n_; ++i) {
        if (HPDDM::abs(scaling[i]) > underlying_type<K>(HPDDM_EPS)) {
          for (unsigned short j = 0; j < super::local_; ++j) array[k + j * n] = tmp[0][i + j * super::n_];
          ++k;
        }
      }
      PetscCallVoid(VecRestoreArrayRead(super::p_.getScalingVec(), &scaling));
      PetscCallVoid(MatDenseRestoreArrayWrite(Z, &array));
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, n, super::local_, nullptr, &P));
    }
    PetscCallVoid(MatMatMult(super::A_, Z, MAT_REUSE_MATRIX, PETSC_DEFAULT, &P));
    PetscCallVoid(MatDestroy(&Z));
    if (n != super::n_) {
      const PetscScalar *array, *scaling;
      PetscCallVoid(MatDenseGetArrayRead(P, &array));
      PetscCallVoid(VecGetArrayRead(super::p_.getScalingVec(), &scaling));
      std::fill_n(super::work_, super::local_ * super::n_, K());
      for (PetscInt i = 0, k = 0; i < super::n_; ++i) {
        if (HPDDM::abs(scaling[i]) > underlying_type<K>(HPDDM_EPS)) {
          for (unsigned short j = 0; j < super::local_; ++j) super::work_[i + j * super::n_] = array[k + j * n];
          ++k;
        }
      }
      PetscCallVoid(VecRestoreArrayRead(super::p_.getScalingVec(), &scaling));
      PetscCallVoid(MatDenseRestoreArrayRead(P, &array));
    }
    PetscCallVoid(MatDestroy(&P));
  #endif
    std::fill_n(*tmp, super::local_ * super::n_, K());
    for (unsigned short i = 0; i < receives; ++i) {
      int index;
      MPI_Waitany(super::map_.size(), rq, &index, MPI_STATUS_IGNORE);
      for (unsigned short k = 0; k < (U ? super::local_ : info[index]); ++k)
        Wrapper<K>::sctr(nmap[index].size(), buff[index] + k * nmap[index].size(), nmap[index].data(), tmp[index] + k * omap.size());
    }
    receiveAccumulation<U, true>(info, buff, rq);
    m = std::distance(tmp[0], tmp[super::map_.size()]) / omap.size();
  #if !HPDDM_PETSC
    {
      std::vector<unsigned short> *compute = new std::vector<unsigned short>[super::C_->n_]();
      for (unsigned short i = 0; i < super::map_.size(); ++i)
        for (unsigned int j = 0; j < super::map_[i].second.size(); ++j) {
          std::vector<int>::const_iterator it = std::lower_bound(omap.cbegin(), omap.cend(), super::map_[i].second[j]);
          compute[std::distance(omap.cbegin(), it)].emplace_back(i);
        }
      std::fill_n(tmp[super::map_.size()], super::C_->n_ * m, K());
      for (int i = 0; i < super::C_->n_; ++i) {
        for (int j = super::C_->ia_[i] - (Wrapper<K>::I == 'F'); j < super::C_->ia_[i + 1] - (Wrapper<K>::I == 'F'); ++j) {
          for (unsigned short k = 0; k < compute[super::C_->ja_[j] - (Wrapper<K>::I == 'F')].size(); ++k) {
            const int m = (U == 1 ? super::local_ : info[compute[super::C_->ja_[j] - (Wrapper<K>::I == 'F')][k]]);
            Blas<K>::axpy(&m, super::C_->a_ + j, tmp[compute[super::C_->ja_[j] - (Wrapper<K>::I == 'F')][k]] + super::C_->ja_[j] - (Wrapper<K>::I == 'F'),
                          &(super::C_->n_), tmp[super::map_.size() + compute[super::C_->ja_[j] - (Wrapper<K>::I == 'F')][k]] + i, &(super::C_->n_));
            if (super::C_->sym_ && i != super::C_->ja_[j] - (Wrapper<K>::I == 'F')) {
              const int m = (U == 1 ? super::local_ : info[compute[super::C_->ja_[j] - (Wrapper<K>::I == 'F')][k]]);
              Blas<K>::axpy(&m, super::C_->a_ + j, tmp[compute[super::C_->ja_[j] - (Wrapper<K>::I == 'F')][k]] + i, &(super::C_->n_),
                            tmp[super::map_.size() + compute[super::C_->ja_[j] - (Wrapper<K>::I == 'F')][k]] + super::C_->ja_[j] - (Wrapper<K>::I == 'F'),
                            &(super::C_->n_));
            }
          }
        }
      }
      delete[] compute;
    }
    delete super::C_;
  #else
    {
      Mat                   Z, P;
      std::vector<PetscInt> imap;
      if (n == super::n_) {
        PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, omap.size(), m, *tmp, &Z));
        PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, omap.size(), m, tmp[super::map_.size()], &P));
      } else {
        imap.reserve(omap.size());
        const PetscScalar *scaling;
        PetscCallVoid(VecGetArrayRead(super::p_.getScalingVec(), &scaling));
        for (typename std::vector<typename Preconditioner::integer_type>::const_iterator it = omap.cbegin(); it != omap.cend(); ++it) {
          if (HPDDM::abs(scaling[*it]) > underlying_type<K>(HPDDM_EPS)) imap.emplace_back(std::distance(omap.cbegin(), it));
        }
        PetscCallVoid(VecRestoreArrayRead(super::p_.getScalingVec(), &scaling));
        PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, imap.size(), m, nullptr, &Z));
        PetscScalar *array;
        PetscCallVoid(MatDenseGetArray(Z, &array));
        for (int i = 0; i < imap.size(); ++i) {
          for (unsigned short j = 0; j < m; ++j) array[i + j * imap.size()] = tmp[0][imap[i] + j * omap.size()];
        }
        PetscCallVoid(MatDenseRestoreArray(Z, &array));
        PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, imap.size(), m, nullptr, &P));
      }
      PetscCallVoid(MatMatMult(super::C_, Z, MAT_REUSE_MATRIX, PETSC_DEFAULT, &P));
      PetscCallVoid(MatDestroy(&Z));
      if (n != super::n_) {
        PetscScalar *array;
        PetscCallVoid(MatDenseGetArray(P, &array));
        std::fill_n(tmp[super::map_.size()], m * omap.size(), K());
        for (int i = 0; i < imap.size(); ++i) {
          for (unsigned short j = 0; j < m; ++j) tmp[super::map_.size()][imap[i] + j * omap.size()] = array[i + j * imap.size()];
        }
        PetscCallVoid(MatDenseRestoreArray(P, &array));
      }
      PetscCallVoid(MatDestroy(&P));
    }
    PetscCallVoid(MatDestroy(&(super::C_)));
  #endif
    MPI_Waitall(super::map_.size(), rq + super::map_.size(), MPI_STATUSES_IGNORE);
    for (unsigned short i = 0; i < super::map_.size(); ++i) {
      m = (U == 1 ? super::local_ : info[i]);
      for (unsigned short j = 0; j < m; ++j)
        Wrapper<K>::gthr(nmap[i].size(), tmp[super::map_.size() + i] + j * omap.size(), buff[super::map_.size() + i] + j * nmap[i].size(), nmap[i].data());
      if (m && !super::map_[i].second.empty())
        MPI_Isend(buff[super::map_.size() + i], super::map_[i].second.size() * m, Wrapper<K>::mpi_type(), super::map_[i].first, 21,
                  super::p_.subdomain.getCommunicator(), rq + super::map_.size() + i);
      else rq[super::map_.size() + i] = MPI_REQUEST_NULL;
    }
    K *pt = overlap_.data();
    for (unsigned short i = 0; i < overlap.size(); ++i) {
      for (unsigned short j = 0; j < overlap.size(); ++j) {
        if (block[overlap[i]][overlap[j]].first != 0 && block[overlap[i]][overlap[j]].second != 0) {
          const int n = omap.size();
          Blas<K>::gemm(&(Wrapper<K>::transc), "N", &(block[overlap[i]][overlap[j]].first), &(block[overlap[i]][overlap[j]].second), &n, &(Wrapper<K>::d_1),
                        tmp[overlap[i]], &n, tmp[super::map_.size() + overlap[j]], &n, &(Wrapper<K>::d_0), pt, &(block[overlap[i]][overlap[j]].first));
          pt += (U == 1 ? super::local_ * super::local_ : info[overlap[i]] * info[overlap[j]]);
        }
      }
    }
    if (block) {
      delete[] *block;
      delete[] block;
    }
  #if !HPDDM_PETSC
    if (HPDDM_NUMBERING != Wrapper<K>::I) {
      if (Wrapper<K>::I == 'F') {
        std::for_each(super::A_->ja_, super::A_->ja_ + super::A_->nnz_, [](int &i) { --i; });
        std::for_each(super::A_->ia_, super::A_->ia_ + super::A_->n_ + 1, [](int &i) { --i; });
      } else {
        std::for_each(super::A_->ja_, super::A_->ja_ + super::A_->nnz_, [](int &i) { ++i; });
        std::for_each(super::A_->ia_, super::A_->ia_ + super::A_->n_ + 1, [](int &i) { ++i; });
      }
    }
  #endif
    MPI_Waitall(super::map_.size(), rq, MPI_STATUSES_IGNORE);
    if (S == 'S') {
      for (unsigned short i = 0; i < overlap.size() && overlap[i] < super::signed_; ++i) {
        m = (U == 1 ? super::local_ : info[overlap[i]]);
        if (m) {
          for (unsigned short nu = 0; nu < super::local_; ++nu)
            Wrapper<K>::gthr(omap.size(), super::work_ + nu * super::n_, work + nu * omap.size(), omap.data());
          const std::vector<unsigned short> &r = accumulate[super::map_.size() + overlap[i]];
          for (unsigned short j : r) {
            for (unsigned short nu = 0; nu < super::local_; ++nu) {
              std::fill_n(tmp[super::map_.size()], omap.size(), K());
              Wrapper<K>::sctr(nmap[j].size(), buff[j] + nu * nmap[j].size(), nmap[j].data(), tmp[super::map_.size()]);
              for (int &k : nmap[overlap[i]]) work[k + nu * omap.size()] += tmp[super::map_.size()][k];
            }
          }
          const int n = omap.size();
          Blas<K>::gemm(&(Wrapper<K>::transc), "N", &m, &(super::local_), &n, &(Wrapper<K>::d_1), tmp[overlap[i]], &n, work, &n, &(Wrapper<K>::d_0), pt, &m);
          pt += super::local_ * m;
        }
      }
    }
    delete[] *tmp;
    delete[] tmp;
    for (unsigned short i = 0; i < overlap.size() && overlap[i] < super::signed_; ++i) {
      if (U || info[overlap[i]]) {
        for (unsigned short nu = 0; nu < super::local_; ++nu)
          Wrapper<K>::gthr(super::map_[overlap[i]].second.size(), super::work_ + nu * super::n_, in[overlap[i]] + nu * super::map_[overlap[i]].second.size(),
                           super::map_[overlap[i]].second.data());
        const std::vector<unsigned short> &r = accumulate[overlap[i]];
        for (unsigned short k : r) {
          for (unsigned short nu = 0; nu < super::local_; ++nu) {
            std::fill_n(work, omap.size(), K());
            Wrapper<K>::sctr(nmap[k].size(), buff[k] + nu * nmap[k].size(), nmap[k].data(), work);
            for (unsigned int j = 0; j < super::map_[overlap[i]].second.size(); ++j)
              in[overlap[i]][j + nu * super::map_[overlap[i]].second.size()] += work[nmap[overlap[i]][j]];
          }
        }
        MPI_Isend(in[overlap[i]], super::map_[overlap[i]].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[overlap[i]].first, 2,
                  super::p_.subdomain.getCommunicator(), rs + overlap[i]);
      } else rs[overlap[i]] = MPI_REQUEST_NULL;
    }
    for (unsigned short &i : interior) {
      for (unsigned short k = 0; k < super::local_; ++k)
        for (unsigned int j = 0; j < super::map_[i].second.size(); ++j)
          super::work_[super::map_[i].second[j] + k * super::n_] += buff[i][j + k * super::map_[i].second.size()];
    }
    for (unsigned short i = 0; i < interior.size() && interior[i] < super::signed_; ++i) {
      if (U || info[interior[i]]) {
        for (unsigned short nu = 0; nu < super::local_; ++nu)
          Wrapper<K>::gthr(super::map_[interior[i]].second.size(), super::work_ + nu * super::n_, in[interior[i]] + nu * super::map_[interior[i]].second.size(),
                           super::map_[interior[i]].second.data());
        MPI_Isend(in[interior[i]], super::map_[interior[i]].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[interior[i]].first, 2,
                  super::p_.subdomain.getCommunicator(), rs + interior[i]);
      } else rs[interior[i]] = MPI_REQUEST_NULL;
    }
    rs += super::signed_;
    delete[] accumulate;
  #if !HPDDM_PETSC
    Wrapper<K>::diag(super::n_, super::D_, super::work_, work, super::local_);
  #else
    if (super::local_) {
      Mat W, Y;
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, super::local_, super::work_, &W));
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, super::local_, work, &Y));
      PetscCallVoid(MatCopy(W, Y, SAME_NONZERO_PATTERN));
      PetscCallVoid(MatDiagonalScale(Y, super::p_.getScalingVec(), nullptr));
      PetscCallVoid(MatDestroy(&Y));
      PetscCallVoid(MatDestroy(&W));
    }
  #endif
    MPI_Waitall(super::map_.size(), rq + super::map_.size(), MPI_STATUSES_IGNORE);
    delete[] rq;
    delete[] *buff;
    delete[] buff;
  }
};
#endif // HPDDM_SCHWARZ

#if HPDDM_FETI
template <class Preconditioner, FetiPrcndtnr Q, class K>
class FetiProjection : public OperatorBase<Q == FetiPrcndtnr::SUPERLUMPED ? 'f' : 'c', Preconditioner, K> {
private:
  using super = OperatorBase<Q == FetiPrcndtnr::SUPERLUMPED ? 'f' : 'c', Preconditioner, K>;
  template <char S, bool U>
  void applyFromNeighbor(const K *in, unsigned short index, K *&work, unsigned short *info)
  {
    std::vector<unsigned short>::const_iterator middle = std::lower_bound(super::vecSparsity_[index].cbegin(), super::vecSparsity_[index].cend(), super::rank_);
    unsigned int                                accumulate = 0;
    if (!(index < super::signed_)) {
      for (unsigned short k = 0;
           k <
           (U ? super::local_
              : info[std::distance(super::sparsity_.cbegin(), std::lower_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::map_[index].first))]);
           ++k)
        for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
          work[super::offsets_[super::map_[index].first] + super::map_[index].second[j] + k * super::n_] += in[k * super::map_[index].second.size() + j];
      accumulate += (U ? super::local_
                       : info[std::distance(super::sparsity_.cbegin(),
                                            std::lower_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::map_[index].first))]) *
                    super::map_[index].second.size();
    } else if (S != 'S') {
      for (unsigned short k = 0;
           k <
           (U ? super::local_
              : info[std::distance(super::sparsity_.cbegin(), std::lower_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::map_[index].first))]);
           ++k)
        for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
          work[super::offsets_[super::map_[index].first] + super::map_[index].second[j] + k * super::n_] -= in[k * super::map_[index].second.size() + j];
      accumulate += (U ? super::local_
                       : info[std::distance(super::sparsity_.cbegin(),
                                            std::lower_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::map_[index].first))]) *
                    super::map_[index].second.size();
    }
    std::vector<unsigned short>::const_iterator begin = super::sparsity_.cbegin();
    if (S != 'S')
      for (std::vector<unsigned short>::const_iterator it = super::vecSparsity_[index].cbegin(); it != middle; ++it) {
        if (!U) {
          std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::sparsity_.cend(), *it);
          if (*it > super::map_[index].first || super::signed_ > index)
            for (unsigned short k = 0; k < info[std::distance(super::sparsity_.cbegin(), idx)]; ++k)
              for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] -= in[accumulate + k * super::map_[index].second.size() + j];
          else
            for (unsigned short k = 0; k < info[std::distance(super::sparsity_.cbegin(), idx)]; ++k)
              for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
          accumulate += info[std::distance(super::sparsity_.cbegin(), idx)] * super::map_[index].second.size();
          begin = idx + 1;
        } else {
          if (*it > super::map_[index].first || super::signed_ > index)
            for (unsigned short k = 0; k < super::local_; ++k)
              for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] -= in[accumulate + k * super::map_[index].second.size() + j];
          else
            for (unsigned short k = 0; k < super::local_; ++k)
              for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
          accumulate += super::local_ * super::map_[index].second.size();
        }
      }
    if (index < super::signed_)
      for (unsigned short k = 0; k < super::local_; ++k)
        for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
          work[super::offsets_[super::rank_] + super::map_[index].second[j] + k * super::n_] -= in[accumulate + k * super::map_[index].second.size() + j];
    else
      for (unsigned short k = 0; k < super::local_; ++k)
        for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
          work[super::offsets_[super::rank_] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
    accumulate += super::local_ * super::map_[index].second.size();
    for (std::vector<unsigned short>::const_iterator it = middle + 1; it < super::vecSparsity_[index].cend(); ++it) {
      if (!U) {
        std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::sparsity_.cend(), *it);
        if (*it > super::map_[index].first && super::signed_ > index)
          for (unsigned short k = 0; k < info[std::distance(super::sparsity_.cbegin(), idx)]; ++k)
            for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
              work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] -= in[accumulate + k * super::map_[index].second.size() + j];
        else
          for (unsigned short k = 0; k < info[std::distance(super::sparsity_.cbegin(), idx)]; ++k)
            for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
              work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
        accumulate += info[std::distance(super::sparsity_.cbegin(), idx)] * super::map_[index].second.size();
        begin = idx + 1;
      } else {
        if (*it > super::map_[index].first && super::signed_ > index)
          for (unsigned short k = 0; k < super::local_; ++k)
            for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
              work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] -= in[accumulate + k * super::map_[index].second.size() + j];
        else
          for (unsigned short k = 0; k < super::local_; ++k)
            for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
              work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
        accumulate += super::local_ * super::map_[index].second.size();
      }
    }
  }

public:
  HPDDM_CLASS_COARSE_OPERATOR(Solver, S, T) friend class CoarseOperator;
  FetiProjection(const Preconditioner &p, const unsigned short &c, const unsigned int &max) : super(p, c, max) { }
  template <char S, bool U, class T>
  void applyToNeighbor(T &in, K *&work, MPI_Request *&rq, const unsigned short *info, T const &out = nullptr, MPI_Request *const &rqRecv = nullptr)
  {
    unsigned short *infoNeighbor;
    super::template initialize<S, U>(in, info, out, rqRecv, infoNeighbor);
    MPI_Request  *rqMult = new MPI_Request[2 * super::map_.size()];
    unsigned int *offset = new unsigned int[super::map_.size() + 2];
    offset[0]            = 0;
    offset[1]            = super::local_;
    for (unsigned short i = 2; i < super::map_.size() + 2; ++i) offset[i] = offset[i - 1] + (U ? super::local_ : infoNeighbor[i - 2]);
    const int       nbMult = super::p_.getMult();
    K              *mult   = new K[offset[super::map_.size() + 1] * nbMult];
    unsigned short *displs = new unsigned short[super::map_.size() + 1];
    displs[0]              = 0;
    for (unsigned short i = 0; i < super::map_.size(); ++i) {
      MPI_Irecv(mult + offset[i + 1] * nbMult + displs[i] * (U ? super::local_ : infoNeighbor[i]),
                super::map_[i].second.size() * (U ? super::local_ : infoNeighbor[i]), Wrapper<K>::mpi_type(), super::map_[i].first, 11,
                super::p_.getCommunicator(), rqMult + i);
      displs[i + 1] = displs[i] + super::map_[i].second.size();
    }

    K                                     *tmp = new K[offset[super::map_.size() + 1] * super::n_]();
    const underlying_type<K> *const *const m   = super::p_.getScaling();
    for (unsigned short i = 0; i < super::signed_; ++i) {
      for (unsigned short k = 0; k < super::local_; ++k)
        for (unsigned int j = 0; j < super::map_[i].second.size(); ++j)
          tmp[super::map_[i].second[j] + k * super::n_] -= m[i][j] * (mult[displs[i] * super::local_ + j +
                                                                           k * super::map_[i].second.size()] = -super::deflation_[k][super::map_[i].second[j]]);
      MPI_Isend(mult + displs[i] * super::local_, super::map_[i].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[i].first, 11,
                super::p_.getCommunicator(), rqMult + super::map_.size() + i);
    }
    for (unsigned short i = super::signed_; i < super::map_.size(); ++i) {
      for (unsigned short k = 0; k < super::local_; ++k)
        for (unsigned int j = 0; j < super::map_[i].second.size(); ++j)
          tmp[super::map_[i].second[j] + k * super::n_] += m[i][j] * (mult[displs[i] * super::local_ + j +
                                                                           k * super::map_[i].second.size()] = super::deflation_[k][super::map_[i].second[j]]);
      MPI_Isend(mult + displs[i] * super::local_, super::map_[i].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[i].first, 11,
                super::p_.getCommunicator(), rqMult + super::map_.size() + i);
    }

    for (unsigned short i = 0; i < super::map_.size(); ++i) {
      int index;
      MPI_Waitany(super::map_.size(), rqMult, &index, MPI_STATUS_IGNORE);
      if (index < super::signed_)
        for (unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[index]); ++k)
          for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
            tmp[super::map_[index].second[j] + (offset[index + 1] + k) * super::n_] = -m[index][j] *
                                                                                      mult[offset[index + 1] * nbMult +
                                                                                           displs[index] * (U ? super::local_ : infoNeighbor[index]) + j +
                                                                                           k * super::map_[index].second.size()];
      else
        for (unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[index]); ++k)
          for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
            tmp[super::map_[index].second[j] + (offset[index + 1] + k) * super::n_] = m[index][j] *
                                                                                      mult[offset[index + 1] * nbMult +
                                                                                           displs[index] * (U ? super::local_ : infoNeighbor[index]) + j +
                                                                                           k * super::map_[index].second.size()];
    }

    delete[] displs;

    if (offset[super::map_.size() + 1]) super::p_.applyLocalPreconditioner(tmp, offset[super::map_.size() + 1]);

    MPI_Waitall(super::map_.size(), rqMult + super::map_.size(), MPI_STATUSES_IGNORE);
    delete[] rqMult;
    delete[] mult;

    unsigned int   accumulate = 0;
    unsigned short stop       = std::distance(super::sparsity_.cbegin(), std::upper_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::rank_));
    if (S != 'S') {
      super::offsets_.reserve(super::sparsity_.size() + 1);
      for (unsigned short i = 0; i < stop; ++i) {
        super::offsets_.emplace(super::sparsity_[i], accumulate);
        accumulate += super::n_ * (U ? super::local_ : info[i]);
      }
    } else super::offsets_.reserve(super::sparsity_.size() + 1 - stop);
    super::offsets_.emplace(super::rank_, accumulate);
    accumulate += super::n_ * super::local_;
    for (unsigned short i = stop; i < super::sparsity_.size(); ++i) {
      super::offsets_.emplace(super::sparsity_[i], accumulate);
      accumulate += super::n_ * (U ? super::local_ : info[i]);
    }

    work = new K[accumulate]();

    for (unsigned short i = 0; i < super::signed_; ++i) {
      accumulate = super::local_;
      for (unsigned short k = 0; k < super::local_; ++k)
        for (unsigned int j = 0; j < super::map_[i].second.size(); ++j)
          work[super::offsets_[super::rank_] + super::map_[i].second[j] +
               k * super::n_] -= (in[i][k * super::map_[i].second.size() + j] = -m[i][j] * tmp[super::map_[i].second[j] + k * super::n_]);
      for (unsigned short l = (S != 'S' ? 0 : i); l < super::map_.size(); ++l) {
        if (Q == FetiPrcndtnr::SUPERLUMPED && l != i &&
            !std::binary_search(super::vecSparsity_[i].cbegin(), super::vecSparsity_[i].cend(), super::map_[l].first)) {
          if (S != 'S' || !(l < super::signed_))
            for (unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[l]); ++k)
              for (unsigned int j = 0; j < super::map_[i].second.size(); ++j)
                work[super::offsets_[super::map_[l].first] + super::map_[i].second[j] + k * super::n_] -= -m[i][j] * tmp[super::map_[i].second[j] +
                                                                                                                         (offset[l + 1] + k) * super::n_];
          continue;
        }
        for (unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[l]); ++k)
          for (unsigned int j = 0; j < super::map_[i].second.size(); ++j) {
            if (S != 'S' || !(l < super::signed_))
              work[super::offsets_[super::map_[l].first] + super::map_[i].second[j] +
                   k * super::n_] -= (in[i][(accumulate + k) * super::map_[i].second.size() + j] = -m[i][j] * tmp[super::map_[i].second[j] +
                                                                                                                  (offset[l + 1] + k) * super::n_]);
            else in[i][(accumulate + k) * super::map_[i].second.size() + j] = -m[i][j] * tmp[super::map_[i].second[j] + (offset[l + 1] + k) * super::n_];
          }
        accumulate += U ? super::local_ : infoNeighbor[l];
      }
      if (U || infoNeighbor[i])
        MPI_Isend(in[i], super::map_[i].second.size() * accumulate, Wrapper<K>::mpi_type(), super::map_[i].first, 2, super::p_.getCommunicator(), rq++);
    }
    for (unsigned short i = super::signed_; i < super::map_.size(); ++i) {
      if (S != 'S') {
        accumulate = super::local_;
        for (unsigned short k = 0; k < super::local_; ++k)
          for (unsigned int j = 0; j < super::map_[i].second.size(); ++j)
            work[super::offsets_[super::rank_] + super::map_[i].second[j] +
                 k * super::n_] += (in[i][k * super::map_[i].second.size() + j] = m[i][j] * tmp[super::map_[i].second[j] + k * super::n_]);
      } else {
        accumulate = 0;
        for (unsigned short k = 0; k < super::local_; ++k)
          for (unsigned int j = 0; j < super::map_[i].second.size(); ++j)
            work[super::offsets_[super::rank_] + super::map_[i].second[j] + k * super::n_] += m[i][j] * tmp[super::map_[i].second[j] + k * super::n_];
      }
      for (unsigned short l = S != 'S' ? 0 : super::signed_; l < super::map_.size(); ++l) {
        if (Q == FetiPrcndtnr::SUPERLUMPED && l != i &&
            !std::binary_search(super::vecSparsity_[i].cbegin(), super::vecSparsity_[i].cend(), super::map_[l].first)) {
          if (S != 'S' || !(l < i))
            for (unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[l]); ++k)
              for (unsigned int j = 0; j < super::map_[i].second.size(); ++j)
                work[super::offsets_[super::map_[l].first] + super::map_[i].second[j] + k * super::n_] += m[i][j] * tmp[super::map_[i].second[j] +
                                                                                                                        (offset[l + 1] + k) * super::n_];
          continue;
        }
        for (unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[l]); ++k)
          for (unsigned int j = 0; j < super::map_[i].second.size(); ++j) {
            if (S != 'S' || !(l < i))
              work[super::offsets_[super::map_[l].first] + super::map_[i].second[j] +
                   k * super::n_] += (in[i][(accumulate + k) * super::map_[i].second.size() + j] = m[i][j] * tmp[super::map_[i].second[j] +
                                                                                                                 (offset[l + 1] + k) * super::n_]);
            else
              work[super::offsets_[super::map_[l].first] + super::map_[i].second[j] + k * super::n_] += m[i][j] * tmp[super::map_[i].second[j] +
                                                                                                                      (offset[l + 1] + k) * super::n_];
          }
        if (S != 'S' || !(l < i)) accumulate += U ? super::local_ : infoNeighbor[l];
      }
      if (U || infoNeighbor[i])
        MPI_Isend(in[i], super::map_[i].second.size() * accumulate, Wrapper<K>::mpi_type(), super::map_[i].first, 2, super::p_.getCommunicator(), rq++);
    }
    delete[] tmp;
    delete[] offset;
    if (!U && Q != FetiPrcndtnr::SUPERLUMPED) delete[] infoNeighbor;
  }
  template <char S, bool U>
  void assembleForMain(K *C, const K *in, const int &coefficients, unsigned short index, K *arrayC, unsigned short *const &infoNeighbor = nullptr)
  {
    applyFromNeighbor<S, U>(in, index, arrayC, infoNeighbor);
    if (++super::consolidate_ == super::map_.size()) {
      if (S != 'S')
        Blas<K>::gemm(&(Wrapper<K>::transc), "N", &coefficients, &(super::local_), &(super::n_), &(Wrapper<K>::d_1), arrayC, &(super::n_), *super::deflation_,
                      super::p_.getLDR(), &(Wrapper<K>::d_0), C, &coefficients);
      else
        for (unsigned short j = 0; j < super::local_; ++j) {
          int local = coefficients + super::local_ - j;
          Blas<K>::gemv(&(Wrapper<K>::transc), &(super::n_), &local, &(Wrapper<K>::d_1), arrayC + super::n_ * j, &(super::n_), super::deflation_[j], &i_1,
                        &(Wrapper<K>::d_0), C - (j * (j - 1)) / 2 + j * (coefficients + super::local_), &i_1);
        }
    }
  }
  template <char S, char N, bool U>
  void applyFromNeighborMain(const K *in, unsigned short index, int *I, int *J, K *C, int coefficients, unsigned int offsetI, unsigned int *offsetJ, K *arrayC,
                             unsigned short *const &infoNeighbor = nullptr)
  {
    assembleForMain<S, U>(C, in, coefficients, index, arrayC, infoNeighbor);
    super::template assembleOperator<S, N, U>(I, J, coefficients, offsetI, offsetJ, infoNeighbor);
  }
};
#endif // HPDDM_FETI

#if HPDDM_BDD || HPDDM_PETSC
template <class Preconditioner, class K>
class BddProjection : public OperatorBase<'c', Preconditioner, K> {
private:
  #if HPDDM_PETSC
  const Mat          A_;
  Mat                C_{};
  PC_HPDDM_Level    *level_;
  const std::string &prefix_;
  #endif
  using super = OperatorBase<'c', Preconditioner, K>;
  template <char S, bool U>
  void applyFromNeighbor(const K *in, unsigned short index, K *&work, unsigned short *info)
  {
    std::vector<unsigned short>::const_iterator middle = std::lower_bound(super::vecSparsity_[index].cbegin(), super::vecSparsity_[index].cend(), super::rank_);
    unsigned int                                accumulate = 0;
    if (S != 'S' || !(index < super::signed_)) {
      for (unsigned short k = 0;
           k <
           (U ? super::local_
              : info[std::distance(super::sparsity_.cbegin(), std::lower_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::map_[index].first))]);
           ++k)
        for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
          work[super::offsets_[super::map_[index].first] + super::map_[index].second[j] + k * super::n_] += in[k * super::map_[index].second.size() + j];
      accumulate += (U ? super::local_
                       : info[std::distance(super::sparsity_.cbegin(),
                                            std::lower_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::map_[index].first))]) *
                    super::map_[index].second.size();
    }
    std::vector<unsigned short>::const_iterator begin = super::sparsity_.cbegin();
    if (S != 'S')
      for (std::vector<unsigned short>::const_iterator it = super::vecSparsity_[index].cbegin(); it != middle; ++it) {
        if (!U) {
          std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::sparsity_.cend(), *it);
          for (unsigned short k = 0; k < info[std::distance(super::sparsity_.cbegin(), idx)]; ++k)
            for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
              work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
          accumulate += info[std::distance(super::sparsity_.cbegin(), idx)] * super::map_[index].second.size();
          begin = idx + 1;
        } else {
          for (unsigned short k = 0; k < super::local_; ++k)
            for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
              work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
          accumulate += super::local_ * super::map_[index].second.size();
        }
      }
    for (unsigned short k = 0; k < super::local_; ++k) {
      for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
        work[super::offsets_[super::rank_] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
    }
    accumulate += super::local_ * super::map_[index].second.size();
    for (std::vector<unsigned short>::const_iterator it = middle + 1; it < super::vecSparsity_[index].cend(); ++it) {
      if (!U) {
        std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::sparsity_.cend(), *it);
        for (unsigned short k = 0; k < info[std::distance(super::sparsity_.cbegin(), idx)]; ++k)
          for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
            work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
        accumulate += info[std::distance(super::sparsity_.cbegin(), idx)] * super::map_[index].second.size();
        begin = idx + 1;
      } else {
        for (unsigned short k = 0; k < super::local_; ++k)
          for (unsigned int j = 0; j < super::map_[index].second.size(); ++j)
            work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
        accumulate += super::local_ * super::map_[index].second.size();
      }
    }
  }

public:
  HPDDM_CLASS_COARSE_OPERATOR(Solver, S, T) friend class CoarseOperator;
  template <typename... Types>
  #if HPDDM_PETSC
  BddProjection(const Preconditioner &p, const unsigned short &c, const int &max, Mat A, PC_HPDDM_Level *level, std::string &prefix, Types...) :
    super(p, c, max), A_(A), level_(level), prefix_(prefix)
  {
    static_assert(sizeof...(Types) == 0, "Wrong constructor");
  }
    #if PETSC_PCHPDDM_MAXLEVELS
  template <char S, bool U>
  void applyToNeighbor(const unsigned short *neighborSizes)
  {
    K          **in     = super::p_.subdomain.getBuffer();
    K          **out    = (U || super::local_) ? in + super::map_.size() : nullptr;
    MPI_Request *rqRecv = super::p_.subdomain.getRq() + super::map_.size();
    MPI_Request *rq     = super::p_.subdomain.getRq();
    applyToNeighbor<S, U>(in, super::workBase_, rq, neighborSizes, out, rqRecv);
    super::sendRequestsEnd_ = rq;
  }
  template <char S, bool U>
  void assembleForMain(K *C, const CoarseOffsets &offsets, const NeighborSizes &neighborSizes)
  {
    K *const *const       in           = super::p_.subdomain.getBuffer() + super::map_.size();
    MPI_Request          *rqRecv       = super::p_.subdomain.getRq() + super::map_.size();
    const int             coefficients = offsets.rowStride();
    unsigned short *const info         = neighborSizes.get();
    constexpr bool        isSymmetric  = (S == 'S');
    for (unsigned short k = 0; k < super::map_.size(); ++k) {
      int neighborIndex;
      MPI_Waitany(static_cast<int>(super::map_.size()), rqRecv, &neighborIndex, MPI_STATUS_IGNORE);
      assembleForMain<!U ? S : (isSymmetric ? 'B' : 'C'), U>(C, in[neighborIndex], coefficients, neighborIndex, super::workBase_, info);
    }
  }
  template <char S, bool U, class T>
  void applyFromNeighborMain(T *J, K *C, const CoarseOffsets &offsets, const NeighborSizes &neighborSizes)
  {
    K *const *const     in           = super::p_.subdomain.getBuffer() + super::map_.size();
    MPI_Request        *rqRecv       = super::p_.subdomain.getRq() + super::map_.size();
    const int           coefficients = offsets.rowStride();
    const unsigned int  offsetI      = super::max_;
    unsigned int *const offsetJ      = offsets.neighborColumn();
    constexpr bool      isSymmetric  = (S == 'S');
    for (unsigned short k = 0; k < super::map_.size(); ++k) {
      int neighborIndex;
      MPI_Waitany(static_cast<int>(super::map_.size()), rqRecv, &neighborIndex, MPI_STATUS_IGNORE);
      applyFromNeighborMain<!U ? S : (isSymmetric ? 'B' : 'C'), 'C', U>(in[neighborIndex], neighborIndex, static_cast<T *>(nullptr), J, C, coefficients,
                                                                        offsetI, U ? nullptr : offsetJ, super::workBase_, U ? nullptr : neighborSizes.get());
    }
  }
    #endif
  #else
  BddProjection(const Preconditioner &p, const unsigned short &c, const int &max, Types...) : super(p, c, max)
  {
    static_assert(sizeof...(Types) == 0, "Wrong constructor");
  }
  #endif
  template <char S, bool U, class T>
  void applyToNeighbor(T &in, K *&work, MPI_Request *&rq, const unsigned short *info, T const &out = nullptr, MPI_Request *const &rqRecv = nullptr)
  {
    unsigned short *infoNeighbor;
    super::template initialize<S, U>(in, info, out, rqRecv, infoNeighbor);
    MPI_Request  *rqMult = new MPI_Request[2 * super::map_.size()];
    unsigned int *displs[U ? 1 : 3];
    *displs      = new unsigned int[(U ? 1 : 3) * (super::map_.size() + 1) + 1];
    displs[0][0] = 0;
    if (!U) {
      displs[1]    = displs[0] + super::map_.size() + 1;
      displs[2]    = displs[1] + super::map_.size() + 1;
      displs[1][0] = displs[2][0] = 0;
      displs[2][1]                = super::local_;
    }
    for (unsigned short i = 0; i < super::map_.size(); ++i) {
      displs[0][i + 1] = displs[0][i] + super::map_[i].second.size();
      if (!U) {
        displs[1][i + 1] = displs[1][i] + super::map_[i].second.size() * infoNeighbor[i];
        displs[2][i + 2] = displs[2][i + 1] + infoNeighbor[i];
      }
    }
    K *mult = new K[U ? 2 * displs[0][super::map_.size()] * super::local_ : displs[0][super::map_.size()] * super::local_ + displs[1][super::map_.size()]];
    for (unsigned short i = 0; i < super::map_.size(); ++i)
      MPI_Irecv(mult + (U ? (displs[0][super::map_.size()] + displs[0][i]) * super::local_ : displs[0][super::map_.size()] * super::local_ + displs[1][i]),
                super::map_[i].second.size() * (U ? super::local_ : infoNeighbor[i]), Wrapper<K>::mpi_type(), super::map_[i].first, 11,
                super::p_.subdomain.getCommunicator(), rqMult + i);
    K *tmp = new K[(U ? (super::map_.size() + 1) * super::local_ : displs[2][super::map_.size() + 1]) * super::n_]();
  #if HPDDM_PETSC
    if (super::local_) {
      Mat scaled;
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, super::local_, tmp, &scaled));
      PetscCallVoid(MatCopy(super::p_.getMat(), scaled, SAME_NONZERO_PATTERN));
      PetscCallVoid(MatDiagonalScale(scaled, super::p_.getScalingVec(), nullptr));
      PetscCallVoid(MatDestroy(&scaled));
    }
  #else
    const underlying_type<K> *const m = super::p_.getScaling();
  #endif
    for (unsigned short i = 0; i < super::map_.size(); ++i) {
      for (unsigned short k = 0; k < super::local_; ++k)
  #if HPDDM_BDD
        for (unsigned int j = 0; j < super::map_[i].second.size(); ++j)
          tmp[super::map_[i].second[j] +
              k * super::n_] = (mult[displs[0][i] * super::local_ + j + k * super::map_[i].second.size()] = m[super::map_[i].second[j]] *
                                                                                                            super::deflation_[k][super::map_[i].second[j]]);
  #else
        Wrapper<K>::gthr(super::map_[i].second.size(), tmp + k * super::n_, mult + displs[0][i] * super::local_ + k * super::map_[i].second.size(),
                         super::map_[i].second.data());
  #endif
      MPI_Isend(mult + displs[0][i] * super::local_, super::map_[i].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[i].first, 11,
                super::p_.subdomain.getCommunicator(), rqMult + super::map_.size() + i);
    }

    for (unsigned short i = 0; i < super::map_.size(); ++i) {
      int index;
      MPI_Waitany(super::map_.size(), rqMult, &index, MPI_STATUS_IGNORE);
      for (unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[index]); ++k)
        Wrapper<K>::sctr(super::map_[index].second.size(),
                         mult +
                           (U ? (displs[0][super::map_.size()] + displs[0][index]) * super::local_
                              : displs[0][super::map_.size()] * super::local_ + displs[1][index]) +
                           k * super::map_[index].second.size(),
                         super::map_[index].second.data(), tmp + ((U ? (index + 1) * super::local_ : displs[2][index + 1]) + k) * super::n_);
    }
    if (U || displs[2][super::map_.size() + 1]) {
  #if HPDDM_BDD
      super::p_.applyLocalSchurComplement(tmp, U ? (super::map_.size() + 1) * super::local_ : displs[2][super::map_.size() + 1]);
  #else
      Mat Z, P;
      PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, U ? (super::map_.size() + 1) * super::local_ : displs[2][super::map_.size() + 1], tmp, &Z));
      PetscCallVoid(MatMatMult(A_, Z, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &P));
      PetscCallVoid(MatDestroy(&Z));
      const PetscScalar *array;
      PetscCallVoid(MatDenseGetArrayRead(P, &array));
      std::copy_n(array, (U ? (super::map_.size() + 1) * super::local_ : displs[2][super::map_.size() + 1]) * super::n_, tmp);
      PetscCallVoid(MatDenseRestoreArrayRead(P, &array));
      PetscCallVoid(MatDestroy(&P));
  #endif
    }

    MPI_Waitall(super::map_.size(), rqMult + super::map_.size(), MPI_STATUSES_IGNORE);
    delete[] rqMult;
    delete[] mult;

    unsigned int   accumulate = 0;
    unsigned short stop       = std::distance(super::sparsity_.cbegin(), std::upper_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::rank_));
    if (S != 'S') {
      super::offsets_.reserve(super::sparsity_.size() + 1);
      for (unsigned short i = 0; i < stop; ++i) {
        super::offsets_.emplace(super::sparsity_[i], accumulate);
        accumulate += super::n_ * (U ? super::local_ : info[i]);
      }
    } else super::offsets_.reserve(super::sparsity_.size() + 1 - stop);
    super::offsets_.emplace(super::rank_, accumulate);
    accumulate += super::n_ * super::local_;
    for (unsigned short i = stop; i < super::sparsity_.size(); ++i) {
      super::offsets_.emplace(super::sparsity_[i], accumulate);
      accumulate += super::n_ * (U ? super::local_ : info[i]);
    }

    work = new K[accumulate]();
  #if HPDDM_PETSC
    std::copy_n(tmp, super::local_ * super::n_, work + super::offsets_[super::rank_]);
    if (S != 'S') {
      for (unsigned short i = 0; i < super::signed_; ++i)
        std::copy(tmp + (U ? (i + 1) * super::local_ : displs[2][i + 1]) * super::n_, tmp + (U ? (i + 2) * super::local_ : displs[2][i + 2]) * super::n_,
                  work + super::offsets_[super::map_[i].first]);
    }
    for (unsigned short i = super::signed_; i < super::map_.size(); ++i)
      std::copy(tmp + (U ? (i + 1) * super::local_ : displs[2][i + 1]) * super::n_, tmp + (U ? (i + 2) * super::local_ : displs[2][i + 2]) * super::n_,
                work + super::offsets_[super::map_[i].first]);
  #endif

    for (unsigned short i = 0; i < super::map_.size(); ++i) {
      if (i < super::signed_ || S != 'S') {
        accumulate = super::local_;
        for (unsigned short k = 0; k < super::local_; ++k)
  #if HPDDM_BDD
          for (unsigned int j = 0; j < super::map_[i].second.size(); ++j)
            work[super::offsets_[super::rank_] + super::map_[i].second[j] + k * super::n_] = in[i][k * super::map_[i].second.size() +
                                                                                                   j] = tmp[super::map_[i].second[j] + k * super::n_];
  #else
          Wrapper<K>::gthr(super::map_[i].second.size(), tmp + k * super::n_, in[i] + k * super::map_[i].second.size(), super::map_[i].second.data());
  #endif
      } else {
        accumulate = 0;
  #if HPDDM_BDD
        for (unsigned short k = 0; k < super::local_; ++k)
          for (unsigned int j = 0; j < super::map_[i].second.size(); ++j)
            work[super::offsets_[super::rank_] + super::map_[i].second[j] + k * super::n_] = tmp[super::map_[i].second[j] + k * super::n_];
  #endif
      }
      for (unsigned short l = S != 'S' ? 0 : std::min(i, super::signed_); l < super::map_.size(); ++l) {
        for (unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[l]); ++k)
  #if HPDDM_BDD
          for (unsigned int j = 0; j < super::map_[i].second.size(); ++j) {
            if (S != 'S' || !(l < std::max(i, super::signed_)))
              work[super::offsets_[super::map_[l].first] + super::map_[i].second[j] +
                   k * super::n_] = in[i][(accumulate + k) * super::map_[i].second.size() +
                                          j] = tmp[super::map_[i].second[j] + ((U ? (l + 1) * super::local_ : displs[2][l + 1]) + k) * super::n_];
            else {
              if (i < super::signed_)
                in[i][(accumulate + k) * super::map_[i].second.size() + j] = tmp[super::map_[i].second[j] +
                                                                                 ((U ? (l + 1) * super::local_ : displs[2][l + 1]) + k) * super::n_];
              else
                work[super::offsets_[super::map_[l].first] + super::map_[i].second[j] +
                     k * super::n_] = tmp[super::map_[i].second[j] + ((U ? (l + 1) * super::local_ : displs[2][l + 1]) + k) * super::n_];
            }
          }
  #else
          if (S != 'S' || !(l < std::max(i, super::signed_)) || i < super::signed_)
            Wrapper<K>::gthr(super::map_[i].second.size(), tmp + ((U ? (l + 1) * super::local_ : displs[2][l + 1]) + k) * super::n_,
                             in[i] + (accumulate + k) * super::map_[i].second.size(), super::map_[i].second.data());
  #endif
        if (S != 'S' || !(l < i) || i < super::signed_) accumulate += U ? super::local_ : infoNeighbor[l];
      }
      if (U || infoNeighbor[i])
        MPI_Isend(in[i], super::map_[i].second.size() * accumulate, Wrapper<K>::mpi_type(), super::map_[i].first, 2, super::p_.subdomain.getCommunicator(),
                  rq++);
    }
    delete[] tmp;
    delete[] *displs;
    if (!U) delete[] infoNeighbor;
  }
  template <char S, bool U>
  void assembleForMain(K *C, const K *in, const int &coefficients, unsigned short index, K *arrayC, unsigned short *const &infoNeighbor = nullptr)
  {
    applyFromNeighbor<S == 'B' ? 'S' : (S == 'C' ? 'G' : S), U>(in, index, arrayC, infoNeighbor);
    if (++super::consolidate_ == super::map_.size()) {
  #if !HPDDM_PETSC
      const underlying_type<K> *const m = super::p_.getScaling();
      Wrapper<K>::diag(super::n_, m, arrayC, coefficients + (S == 'S') * super::local_);
      if (S == 'B' || S == 'C') {
        K *tmp = new K[coefficients * super::local_];
        Blas<K>::gemm(&(Wrapper<K>::transc), "N", &coefficients, &(super::local_), &(super::n_), &(Wrapper<K>::d_1), arrayC, &(super::n_), *super::deflation_,
                      &(super::n_), &(Wrapper<K>::d_0), tmp, &coefficients);
        for (int i = 0; i < coefficients; ++i) {
          for (unsigned short j = 0; j < super::local_; ++j)
            C[(i / super::local_) * super::local_ * super::local_ + j * super::local_ + (i % super::local_)] = tmp[i + j * coefficients];
        }
        delete[] tmp;
      } else if (S != 'S') {
        if (super::local_)
          Blas<K>::gemm(&(Wrapper<K>::transc), "N", &coefficients, &(super::local_), &(super::n_), &(Wrapper<K>::d_1), arrayC, &(super::n_), *super::deflation_,
                        super::p_.getLDR(), &(Wrapper<K>::d_0), C, &coefficients);
      } else
        for (unsigned short j = 0; j < super::local_; ++j) {
          int local = coefficients + super::local_ - j;
          Blas<K>::gemv(&(Wrapper<K>::transc), &(super::n_), &local, &(Wrapper<K>::d_1), arrayC + super::n_ * j, &(super::n_), super::deflation_[j], &i_1,
                        &(Wrapper<K>::d_0), C - (j * (j - 1)) / 2 + j * (coefficients + super::local_), &i_1);
        }
  #else
      const int      cols     = coefficients + (S == 'S') * super::local_;
      constexpr bool relayout = (S == 'B' || S == 'C' || S == 'S');
      if (cols && super::local_) {
        Mat block, product;
        PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, cols, arrayC, &block));
        PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, cols, super::local_, relayout ? nullptr : C, &product));
        PetscCallVoid(MatDiagonalScale(block, super::p_.getScalingVec(), nullptr));
        PetscCallVoid(MatConjugate(block));
        PetscCallVoid(MatTransposeMatMult(block, super::p_.getMat(), MAT_REUSE_MATRIX, PETSC_DEFAULT, &product));
        PetscCallVoid(MatConjugate(block));
        if (relayout) {
          const PetscScalar *array;
          PetscInt           lda;
          PetscCallVoid(MatDenseGetArrayRead(product, &array));
          PetscCallVoid(MatDenseGetLDA(product, &lda));
          if (S != 'S') {
            for (int i = 0; i < cols; ++i) {
              for (unsigned short j = 0; j < super::local_; ++j)
                C[(i / super::local_) * super::local_ * super::local_ + j * super::local_ + (i % super::local_)] = array[i + j * lda];
            }
          } else
            for (unsigned short j = 0; j < super::local_; ++j)
              std::copy_n(array + j + static_cast<std::ptrdiff_t>(j) * lda, cols - j, C - (j * (j - 1)) / 2 + static_cast<std::ptrdiff_t>(j) * cols);
          PetscCallVoid(MatDenseRestoreArrayRead(product, &array));
        }
        PetscCallVoid(MatDestroy(&block));
        PetscCallVoid(MatDestroy(&product));
      }
  #endif
    }
  }
  template <char S, char N, bool U, class T>
  void applyFromNeighborMain(const K *in, unsigned short index, T *I, T *J, K *C, int coefficients, unsigned int offsetI, unsigned int *offsetJ, K *arrayC,
                             unsigned short *const &infoNeighbor = nullptr)
  {
    assembleForMain<S, U>(C, in, coefficients, index, arrayC, infoNeighbor);
    super::template assembleOperator<S, N, U, T>(I, J, coefficients, offsetI, offsetJ, infoNeighbor);
  }
};
#endif // HPDDM_BDD || HPDDM_PETSC
} // namespace HPDDM
