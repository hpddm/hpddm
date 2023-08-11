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

#ifndef HPDDM_OPERATOR_HPP_
#define HPDDM_OPERATOR_HPP_

#include <queue>

namespace HPDDM {
template<bool> class Members { };
template<> class Members<true> {
    protected:
        std::unordered_map<unsigned short, unsigned int> offsets_;
        std::vector<std::vector<unsigned short>>     vecSparsity_;
        const unsigned short                                rank_;
        unsigned short                               consolidate_;
        explicit Members(unsigned short r) : rank_(r), consolidate_() { }
};
template<char P, class Preconditioner, class K>
class OperatorBase : protected Members<P != 's' && P != 'u'> {
    private:
        HPDDM_HAS_MEMBER(getLDR)
        template<class Q = Preconditioner> typename std::enable_if<has_getLDR<typename std::remove_reference<Q>::type>::value, bool>::type
        offsetDeflation() {
            const unsigned int offset = *p_.getLDR() - n_;
            if(deflation_ && offset)
                std::for_each(const_cast<K**>(deflation_), const_cast<K**>(deflation_) + local_, [&](K*& v) { v -= offset; });
            return true;
        }
        template<class Q = Preconditioner> typename std::enable_if<!has_getLDR<typename std::remove_reference<Q>::type>::value, bool>::type
        offsetDeflation() { return false; }
    protected:
        const Preconditioner&                 p_;
        const K* const* const         deflation_;
        const vectorNeighbor&               map_;
        std::vector<unsigned short>    sparsity_;
        const int                             n_;
        const int                         local_;
        unsigned int                        max_;
        unsigned short                   signed_;
        unsigned short             connectivity_;
        template<char Q = P, typename std::enable_if<Q == 's' || Q == 'u'>::type* = nullptr>
        OperatorBase(const Preconditioner& p, const unsigned short& c, const unsigned int& max) : p_(p), deflation_(p.getVectors()), map_(p.getMap()), n_(p.getDof()), local_(p.getLocal()), max_(max), connectivity_(c) {
            static_assert(Q == P, "Wrong sparsity pattern");
            sparsity_.reserve(map_.size());
            std::transform(map_.cbegin(), map_.cend(), std::back_inserter(sparsity_), [](const pairNeighbor& n) { return n.first; });
        }
        template<char Q = P, typename std::enable_if<Q != 's' && Q != 'u'>::type* = nullptr>
        OperatorBase(const Preconditioner& p, const unsigned short& c, const unsigned int& max) : Members<true>(p.getRank()), p_(p), deflation_(p.getVectors()), map_(p.getMap()), n_(p.getDof()), local_(p.getLocal()), max_(max + std::max(1, (c - 1)) * (max & 4095)), signed_(p_.getSigned()), connectivity_(c) {
            const unsigned int offset = *p_.getLDR() - n_;
            if(deflation_ && offset)
                std::for_each(const_cast<K**>(deflation_), const_cast<K**>(deflation_) + local_, [&](K*& v) { v += offset; });
            static_assert(Q == P, "Wrong sparsity pattern");
            if(!map_.empty()) {
                unsigned short** recvSparsity = new unsigned short*[map_.size() + 1];
                *recvSparsity = new unsigned short[(connectivity_ + 1) * map_.size()];
                unsigned short* sendSparsity = *recvSparsity + connectivity_ * map_.size();
                MPI_Request* rq = p_.getRq();
                for(unsigned short i = 0; i < map_.size(); ++i) {
                    sendSparsity[i] = map_[i].first;
                    recvSparsity[i] = *recvSparsity + connectivity_ * i;
                    MPI_Irecv(recvSparsity[i], connectivity_, MPI_UNSIGNED_SHORT, map_[i].first, 4, p_.getCommunicator(), rq + i);
                }
                for(unsigned short i = 0; i < map_.size(); ++i)
                    MPI_Isend(sendSparsity, map_.size(), MPI_UNSIGNED_SHORT, map_[i].first, 4, p_.getCommunicator(), rq + map_.size() + i);
                Members<true>::vecSparsity_.resize(map_.size());
                for(unsigned short i = 0; i < map_.size(); ++i) {
                    int index, count;
                    MPI_Status status;
                    MPI_Waitany(map_.size(), rq, &index, &status);
                    MPI_Get_count(&status, MPI_UNSIGNED_SHORT, &count);
                    Members<true>::vecSparsity_[index].assign(recvSparsity[index], recvSparsity[index] + count);
                }
                MPI_Waitall(map_.size(), rq + map_.size(), MPI_STATUSES_IGNORE);

                delete [] *recvSparsity;
                delete [] recvSparsity;

                sparsity_.reserve(map_.size());
                if(P == 'c') {
                    std::vector<unsigned short> neighbors;
                    neighbors.reserve(map_.size());
                    std::for_each(map_.cbegin(), map_.cend(), [&](const pairNeighbor& n) { neighbors.emplace_back(n.first); });
                    typedef std::pair<std::vector<unsigned short>::const_iterator, std::vector<unsigned short>::const_iterator> pairIt;
                    auto comp = [](const pairIt& lhs, const pairIt& rhs) { return *lhs.first > *rhs.first; };
                    std::priority_queue<pairIt, std::vector<pairIt>, decltype(comp)> pq(comp);
                    pq.push({ neighbors.cbegin(), neighbors.cend() });
                    for(const std::vector<unsigned short>& v : Members<true>::vecSparsity_)
                        pq.push({ v.cbegin(), v.cend() });
                    while(!pq.empty()) {
                        pairIt p = pq.top();
                        pq.pop();
                        if(*p.first != Members<true>::rank_ && (sparsity_.empty() || (*p.first != sparsity_.back())))
                            sparsity_.emplace_back(*p.first);
                        if(++p.first != p.second)
                            pq.push(p);
                    }
                }
                else {
                    std::transform(map_.cbegin(), map_.cend(), std::back_inserter(sparsity_), [](const pairNeighbor& n) { return n.first; });
                    for(std::vector<unsigned short>& v : Members<true>::vecSparsity_) {
                        unsigned short i = 0, j = 0, k = 0;
                        while(i < v.size() && j < sparsity_.size()) {
                            if(v[i] == Members<true>::rank_) {
                                v[k++] = Members<true>::rank_;
                                ++i;
                            }
                            else if(v[i] < sparsity_[j])
                                ++i;
                            else if(v[i] > sparsity_[j])
                                ++j;
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
        ~OperatorBase() { offsetDeflation(); }
        template<char S, bool U, class T>
        void initialize(T& in, const unsigned short* info, T const& out, MPI_Request* const& rqRecv, unsigned short*& infoNeighbor) {
            static_assert(P == 'c' || P == 'f', "Unsupported constructor with such a sparsity pattern");
            if(!U) {
                if(P == 'c') {
                    infoNeighbor = new unsigned short[map_.size()];
                    std::vector<unsigned short>::const_iterator begin = sparsity_.cbegin();
                    for(unsigned short i = 0; i < map_.size(); ++i) {
                        std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, sparsity_.cend(), map_[i].first);
                        infoNeighbor[i] = info[std::distance(sparsity_.cbegin(), idx)];
                        begin = idx + 1;
                    }
                }
                else
                    infoNeighbor = const_cast<unsigned short*>(info);
            }
            std::vector<unsigned int> displs;
            displs.reserve(2 * map_.size());
            if(S != 'S') {
                if(!U) {
                    unsigned short size = std::accumulate(infoNeighbor, infoNeighbor + map_.size(), local_);
                    if(!map_.empty())
                        displs.emplace_back(size * map_[0].second.size());
                    for(unsigned short i = 1; i < map_.size(); ++i)
                        displs.emplace_back(displs.back() + size * map_[i].second.size());
                    for(unsigned short i = 0; i < map_.size(); ++i) {
                        size = infoNeighbor[i];
                        std::vector<unsigned short>::const_iterator begin = sparsity_.cbegin();
                        for(const unsigned short& rank : Members<true>::vecSparsity_[i]) {
                            if(rank == Members<true>::rank_)
                                size += local_;
                            else {
                                std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, sparsity_.cend(), rank);
                                size += info[std::distance(sparsity_.cbegin(), idx)];
                                begin = idx + 1;
                            }
                        }
                        if(local_)
                            displs.emplace_back(displs.back() + size * map_[i].second.size());
                        else
                            rqRecv[i] = MPI_REQUEST_NULL;
                    }
                }
                else {
                    if(!map_.empty())
                        displs.emplace_back(local_ * (map_.size() + 1) * map_[0].second.size());
                    for(unsigned short i = 1; i < map_.size(); ++i)
                        displs.emplace_back(displs.back() + local_ * (map_.size() + 1) * map_[i].second.size());
                    for(unsigned short i = 0; i < map_.size(); ++i)
                        displs.emplace_back(displs.back() + local_ * (Members<true>::vecSparsity_[i].size() + 1) * map_[i].second.size());
                }
            }
            else {
                if(!U) {
                    unsigned short size = std::accumulate(infoNeighbor, infoNeighbor + map_.size(), 0);
                    if(!map_.empty()) {
                        displs.emplace_back((size + local_ * (0 < signed_)) * map_[0].second.size());
                        size -= infoNeighbor[0];
                    }
                    for(unsigned short i = 1; i < map_.size(); ++i) {
                        displs.emplace_back(displs.back() + (size + local_ * (i < signed_)) * map_[i].second.size());
                        size -= infoNeighbor[i];
                    }
                    for(unsigned short i = 0; i < map_.size(); ++i) {
                        size = infoNeighbor[i] * !(i < signed_) + local_;
                        std::vector<unsigned short>::const_iterator end = sparsity_.cend();
                        for(std::vector<unsigned short>::const_reverse_iterator rit = Members<true>::vecSparsity_[i].rbegin(); *rit > Members<true>::rank_; ++rit) {
                            std::vector<unsigned short>::const_iterator idx = std::lower_bound(sparsity_.cbegin(), end, *rit);
                            size += info[std::distance(sparsity_.cbegin(), idx)];
                            end = idx - 1;
                        }
                        if(local_)
                            displs.emplace_back(displs.back() + size * map_[i].second.size());
                        else
                            rqRecv[i] = MPI_REQUEST_NULL;
                    }
                }
                else {
                    if(!map_.empty())
                        displs.emplace_back(local_ * (map_.size() + (0 < signed_)) * map_[0].second.size());
                    for(unsigned short i = 1; i < map_.size(); ++i)
                        displs.emplace_back(displs.back() + local_ * (map_.size() + (i < signed_) - i) * map_[i].second.size());
                    for(unsigned short i = 0; i < map_.size(); ++i) {
                        unsigned short size = std::distance(std::lower_bound(Members<true>::vecSparsity_[i].cbegin(), Members<true>::vecSparsity_[i].cend(), Members<true>::rank_), Members<true>::vecSparsity_[i].cend()) + !(i < signed_);
                        displs.emplace_back(displs.back() + local_ * size * map_[i].second.size());
                    }
                }
            }
            if(!displs.empty()) {
                *in = new K[displs.back()];
                for(unsigned short i = 1; i < map_.size(); ++i)
                    in[i] = *in + displs[i - 1];
                if(U == 1 || local_)
                    for(unsigned short i = 0; i < map_.size(); ++i) {
                        if(displs[i + map_.size()] != displs[i - 1 + map_.size()]) {
                            out[i] = *in + displs[i - 1 + map_.size()];
                            MPI_Irecv(out[i], displs[i + map_.size()] - displs[i - 1 + map_.size()], Wrapper<K>::mpi_type(), map_[i].first, 2, p_.getCommunicator(), rqRecv + i);
                        }
                        else
                            out[i] = nullptr;
                    }
            }
            else
                *in = nullptr;
        }
        template<char S, char N, bool U, char Q = P, typename std::enable_if<Q != 's'>::type* = nullptr>
        void assembleOperator(int* I, int* J, int coefficients, unsigned int offsetI, unsigned int* offsetJ, unsigned short* const& infoNeighbor) {
#ifdef HPDDM_CSR_CO
            ignore(I);
#endif
            if(Members<true>::consolidate_ == map_.size()) {
                unsigned short between = std::distance(sparsity_.cbegin(), std::lower_bound(sparsity_.cbegin(), sparsity_.cend(), p_.getRank()));
                unsigned int offset = 0;
                if(S != 'S')
                    for(unsigned short k = 0; k < between; ++k)
                        for(unsigned short i = 0; i < local_; ++i) {
                            unsigned int l = offset + coefficients * i;
                            for(unsigned short j = 0; j < (U ? local_ : infoNeighbor[k]); ++j) {
#ifndef HPDDM_CSR_CO
                                I[l + j] = offsetI + i;
#endif
                                J[l + j] = (U ? sparsity_[k] * local_ + (N == 'F') : offsetJ[k]) + j;
                            }
                            offset += U ? local_ : infoNeighbor[k];
                        }
                else
                    coefficients += local_ - 1;
                for(unsigned short i = 0; i < local_; ++i) {
                    unsigned int l = offset + coefficients * i - (S == 'S') * ((i * (i - 1)) / 2);
                    for(unsigned short j = (S == 'S') * i; j < local_; ++j) {
#ifndef HPDDM_CSR_CO
                        I[l + j] = offsetI + i;
#endif
                        J[l + j] = offsetI + j;
                    }
                }
                offset += local_;
                for(unsigned short k = between; k < sparsity_.size(); ++k) {
                    for(unsigned short i = 0; i < local_; ++i) {
                        unsigned int l = offset + coefficients * i - (S == 'S') * ((i * (i - 1)) / 2);
                        for(unsigned short j = 0; j < (U ? local_ : infoNeighbor[k]); ++j) {
#ifndef HPDDM_CSR_CO
                            I[l + j] = offsetI + i;
#endif
                            J[l + j] = (U ? sparsity_[k] * local_ + (N == 'F') : offsetJ[k - (S == 'S') * between]) + j;
                        }
                    }
                    offset += U ? local_ : infoNeighbor[k];
                }
            }
        }
    public:
        static constexpr char pattern_ = (P != 's' && P != 'u') ? 'c' : (P == 's' ? 's' : 'u');
        static constexpr bool factorize_ = true;
        template<char, bool>
        void setPattern(int*, const int, const int, const unsigned short* const* const = nullptr, const unsigned short* const = nullptr) const { }
        void adjustConnectivity(const MPI_Comm& comm) {
            if(P == 'c') {
#if 0
                connectivity_ *= connectivity_ - 1;
#else
                connectivity_ = sparsity_.size();
                MPI_Allreduce(MPI_IN_PLACE, &connectivity_, 1, MPI_UNSIGNED_SHORT, MPI_MAX, comm);
#endif
            }
        }
        const std::vector<unsigned short>& getPattern() const { return sparsity_; }
        unsigned short getConnectivity() const { return connectivity_; }
        template<char Q = P, typename std::enable_if<Q != 's'>::type* = nullptr>
        void initialize(unsigned int, K*&, unsigned short) { }
};

template<class Preconditioner, class K>
class UserCoarseOperator : public OperatorBase<'u', Preconditioner, K> {
    private:
        typedef OperatorBase<'u', Preconditioner, K> super;
#if HPDDM_PETSC
    protected:
        const Mat                                       A_;
        Mat                                             C_;
        PC_HPDDM_Level*                             level_;
        const std::string&                         prefix_;
#endif
    public:
        HPDDM_CLASS_COARSE_OPERATOR(Solver, S, T) friend class CoarseOperator;
#if HPDDM_PETSC
        template<typename... Types>
        UserCoarseOperator(const Preconditioner& p, const unsigned short& c, const unsigned int& max, Mat A, PC_HPDDM_Level* level, std::string& prefix, Types...) : super(p, c, max), A_(A), C_(), level_(level), prefix_(prefix) { static_assert(sizeof...(Types) == 0, "Wrong constructor"); }
#else
        UserCoarseOperator(const Preconditioner& p, const unsigned short& c, const unsigned int& max) : super(p, c, max) { }
#endif
};

#if HPDDM_SCHWARZ || HPDDM_SLEPC
template<class Preconditioner, class K>
class MatrixMultiplication : public OperatorBase<'s', Preconditioner, K> {
    protected:
#if HPDDM_SCHWARZ
        const MatrixCSR<K>* const                       A_;
        MatrixCSR<K>*                                   C_;
#else
        const Mat                                       A_;
        Mat                                             C_;
        PC_HPDDM_Level*                             level_;
        const std::string&                         prefix_;
#endif
        K*                                           work_;
        const underlying_type<K>* const                 D_;
    private:
        typedef OperatorBase<'s', Preconditioner, K> super;
        template<bool U>
        void applyFromNeighbor(const K* in, unsigned short index, K*& work, unsigned short* infoNeighbor) {
            int m = U ? super::local_ : *infoNeighbor;
            std::fill_n(work_, m * super::n_, K());
            for(unsigned short i = 0; i < m; ++i)
                for(int j = 0; j < super::map_[index].second.size(); ++j)
                    work_[i * super::n_ + super::map_[index].second[j]] = D_[super::map_[index].second[j]] * in[i * super::map_[index].second.size() + j];
            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &(super::local_), &m, &(super::n_), &(Wrapper<K>::d__1), *super::deflation_, &(super::n_), work_, &(super::n_), &(Wrapper<K>::d__0), work, &(super::local_));
        }
    public:
        HPDDM_CLASS_COARSE_OPERATOR(Solver, S, T) friend class CoarseOperator;
        template<typename... Types>
#if HPDDM_PETSC
        MatrixMultiplication(const Preconditioner& p, const unsigned short& c, const unsigned int& max, Mat A, PC_HPDDM_Level* level, std::string& prefix, Types...) : super(p, c, max), A_(A), C_(), level_(level), prefix_(prefix), D_(p.getScaling()) { static_assert(sizeof...(Types) == 0, "Wrong constructor"); }
        void initialize(unsigned int k, K*& work, unsigned short s) {
            PetscBool sym;
            PetscCallVoid(PetscObjectTypeCompare((PetscObject)A_, MATSEQSBAIJ, &sym));
            PetscCallVoid(MatConvert(A_, sym ? MATSEQBAIJ : MATSAME, MAT_INITIAL_MATRIX, &C_));
            Vec D;
            PetscScalar* d;
            if(!std::is_same<PetscScalar, PetscReal>::value) {
                d = new PetscScalar[super::p_.getDof()];
                std::copy_n(D_, super::p_.getDof(), d);
                PetscCallVoid(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, super::p_.getDof(), d, &D));
            }
            else {
                d = nullptr;
                PetscCallVoid(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, super::p_.getDof(), reinterpret_cast<const PetscScalar*>(D_), &D));
            }
            PetscCallVoid(MatDiagonalScale(C_, nullptr, D));
            delete [] d;
            PetscCallVoid(VecDestroy(&D));
            work = new K[2 * k];
            work_ = work + k;
            super::signed_ = s;
        }
#else
        MatrixMultiplication(const Preconditioner& p, const unsigned short& c, const unsigned int& max, Types...) : super(p, c, max), A_(p.getMatrix()), C_(), D_(p.getScaling()) { static_assert(sizeof...(Types) == 0, "Wrong constructor"); }
        void initialize(unsigned int k, K*& work, unsigned short s) {
            if(A_->sym_) {
                std::vector<std::vector<std::pair<unsigned int, K>>> v(A_->n_);
                unsigned int nnz = ((A_->nnz_ + A_->n_ - 1) / A_->n_) * 2;
                std::for_each(v.begin(), v.end(), [&](std::vector<std::pair<unsigned int, K>>& r) { r.reserve(nnz); });
                nnz = 0;
                for(unsigned int i = 0; i < A_->n_; ++i) {
                    const underlying_type<K> scal = D_[i];
                    unsigned int j = A_->ia_[i] - (HPDDM_NUMBERING == 'F');
                    while(j < A_->ia_[i + 1] - (HPDDM_NUMBERING == 'F' ? 2 : 1)) {
                        if(D_[A_->ja_[j] - (HPDDM_NUMBERING == 'F')] > HPDDM_EPS) {
                            v[i].emplace_back(A_->ja_[j], A_->a_[j] * D_[A_->ja_[j] - (HPDDM_NUMBERING == 'F')]);
                            ++nnz;
                        }
                        if(scal > HPDDM_EPS) {
                            v[A_->ja_[j] - (HPDDM_NUMBERING == 'F')].emplace_back(i + (HPDDM_NUMBERING == 'F'), A_->a_[j] * scal);
                            ++nnz;
                        }
                        ++j;
                    }
                    if(i != A_->ja_[j] - (HPDDM_NUMBERING == 'F')) {
                        if(D_[A_->ja_[j] - (HPDDM_NUMBERING == 'F')] > HPDDM_EPS) {
                            v[i].emplace_back(A_->ja_[j], A_->a_[j] * D_[A_->ja_[j] - (HPDDM_NUMBERING == 'F')]);
                            ++nnz;
                        }
                    }
                    if(scal > HPDDM_EPS) {
                        v[A_->ja_[j] - (HPDDM_NUMBERING == 'F')].emplace_back(i + (HPDDM_NUMBERING == 'F'), A_->a_[j] * scal);
                        ++nnz;
                    }
                }
                C_ = new MatrixCSR<K>(A_->n_, A_->n_, nnz, false);
                C_->ia_[0] = (Wrapper<K>::I == 'F');
                nnz = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, HPDDM_GRANULARITY)
#endif
                for(unsigned int i = 0; i < A_->n_; ++i)
                    std::sort(v[i].begin(), v[i].end(), [](const std::pair<unsigned int, K>& lhs, const std::pair<unsigned int, K>& rhs) { return lhs.first < rhs.first; });
                for(unsigned int i = 0; i < A_->n_; ++i) {
                    for(const std::pair<unsigned int, K>& p : v[i]) {
                        C_->ja_[nnz] = p.first + (Wrapper<K>::I == 'F' && HPDDM_NUMBERING != Wrapper<K>::I);
                        C_->a_[nnz++] = p.second;
                    }
                    C_->ia_[i + 1] = nnz + (Wrapper<K>::I == 'F');
                }
            }
            else {
                C_ = new MatrixCSR<K>(A_->n_, A_->n_, A_->nnz_, false);
                C_->ia_[0] = (Wrapper<K>::I == 'F');
                unsigned int nnz = 0;
                for(unsigned int i = 0; i < A_->n_; ++i) {
                    for(unsigned int j = A_->ia_[i] - (HPDDM_NUMBERING == 'F'); j < A_->ia_[i + 1] - (HPDDM_NUMBERING == 'F'); ++j)
                        if(D_[A_->ja_[j] - (HPDDM_NUMBERING == 'F')] > HPDDM_EPS) {
                            C_->ja_[nnz] = A_->ja_[j] + (Wrapper<K>::I == 'F' && HPDDM_NUMBERING != Wrapper<K>::I);
                            C_->a_[nnz++] = A_->a_[j] * D_[A_->ja_[j] - (HPDDM_NUMBERING == 'F')];
                        }
                    C_->ia_[i + 1] = nnz + (Wrapper<K>::I == 'F');
                }
                C_->nnz_ = nnz;
            }
            work = new K[2 * k];
            work_ = work + k;
            super::signed_ = s;
        }
#endif
        template<char S, bool U, class T>
        void applyToNeighbor(T& in, K*& work, MPI_Request*& rq, const unsigned short* info, T = nullptr, MPI_Request* = nullptr) {
            if(super::local_)
#if !HPDDM_PETSC
                Wrapper<K>::template csrmm<Wrapper<K>::I>(false, &(super::n_), &(super::local_), C_->a_, C_->ia_, C_->ja_, *super::deflation_, work_);
            delete C_;
#else
            {
                Mat Z, P;
                PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, super::local_, const_cast<PetscScalar*>(*super::deflation_), &Z));
                PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, super::local_, work_, &P));
                PetscCallVoid(MatMatMult(C_, Z, MAT_REUSE_MATRIX, PETSC_DEFAULT, &P));
                PetscCallVoid(MatDestroy(&P));
                PetscCallVoid(MatDestroy(&Z));
            }
            PetscCallVoid(MatDestroy(&C_));
#endif
            for(unsigned short i = 0; i < super::signed_; ++i) {
                if(U || info[i]) {
                    for(unsigned short j = 0; j < super::local_; ++j)
                        Wrapper<K>::gthr(super::map_[i].second.size(), work_ + j * super::n_, in[i] + j * super::map_[i].second.size(), super::map_[i].second.data());
                    MPI_Isend(in[i], super::map_[i].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[i].first, 2, super::p_.getCommunicator(), rq++);
                }
            }
            Wrapper<K>::diag(super::n_, D_, work_, work, super::local_);
        }
        template<char S, bool U>
        void assembleForMain(K* C, const K* in, const int& coefficients, unsigned short index, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            applyFromNeighbor<U>(in, index, arrayC, infoNeighbor);
            if(S != 'B')
                for(unsigned short j = 0; j < (U ? super::local_ : *infoNeighbor); ++j) {
                    K* pt = C + j;
                    for(unsigned short i = 0; i < super::local_; pt += coefficients - (S == 'S') * i++)
                        *pt = arrayC[j * super::local_ + i];
                }
        }
        template<char S, char N, bool U, class T>
        void applyFromNeighborMain(const K* in, unsigned short index, T* I, T* J, K* C, int coefficients, unsigned int offsetI, unsigned int* offsetJ, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
#ifdef HPDDM_CSR_CO
            ignore(I, offsetI);
#endif
            applyFromNeighbor<U>(in, index, S == 'B' && N == 'F' ? C : arrayC, infoNeighbor);
            unsigned int offset = (S == 'B' ? super::map_[index].first + (N == 'F') : (U ? super::map_[index].first * super::local_ + (N == 'F') : *offsetJ));
            if(S == 'B') {
                *J = offset;
                if(N == 'C')
                    Wrapper<K>::template omatcopy<'T'>(super::local_, super::local_, arrayC, super::local_, C, super::local_);
            }
            else
                for(unsigned short i = 0; i < super::local_; ++i) {
                    unsigned int l = coefficients * i - (S == 'S') * (i * (i - 1)) / 2;
                    for(unsigned short j = 0; j < (U ? super::local_ : *infoNeighbor); ++j) {
#ifndef HPDDM_CSR_CO
                        I[l + j] = offsetI + i;
#endif
                        J[l + j] = offset + j;
                        C[l + j] = arrayC[j * super::local_ + i];
                    }
                }
        }
};
template<class Preconditioner, class K>
class MatrixAccumulation : public MatrixMultiplication<Preconditioner, K> {
    private:
        std::vector<K>&                                                                                          overlap_;
        std::vector<std::vector<std::pair<unsigned short, unsigned short>>>&                                   reduction_;
        std::map<std::pair<unsigned short, unsigned short>, unsigned short>&                                       sizes_;
        std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>& extra_;
        typedef MatrixMultiplication<Preconditioner, K> super;
        int*                                   ldistribution_;
        int                                             size_;
        unsigned int                                      sb_;
    public:
        HPDDM_CLASS_COARSE_OPERATOR(Solver, S, T) friend class CoarseOperator;
        template<typename First, typename Second, typename Third, typename Fourth, typename Fifth, typename... Rest>
#if !HPDDM_PETSC
        MatrixAccumulation(const Preconditioner& p, const unsigned short& c, const unsigned int& max, First&, Second& arg2, Third& arg3, Fourth& arg4, Fifth& arg5, Rest&... args) : super(p, c, max, args...), overlap_(arg2), reduction_(arg3), sizes_(arg4), extra_(arg5) { static_assert(std::is_same<typename std::remove_pointer<First>::type, typename Preconditioner::super::co_type>::value, "Wrong constructor"); }
#else
        MatrixAccumulation(const Preconditioner& p, const unsigned short& c, const unsigned int& max, Mat A, PC_HPDDM_Level* level, std::string& prefix, First&, Second& arg2, Third& arg3, Fourth& arg4, Fifth& arg5, Rest&... args) : super(p, c, max, A, level, prefix, args...), overlap_(arg2), reduction_(arg3), sizes_(arg4), extra_(arg5) { static_assert(std::is_same<typename std::remove_pointer<First>::type, typename Preconditioner::super::co_type>::value, "Wrong constructor"); }
#endif
        static constexpr bool factorize_ = false;
        int getMain(const int rank) const {
            int* it = std::lower_bound(ldistribution_, ldistribution_ + size_, rank);
            if(it == ldistribution_ + size_)
                return size_ - 1;
            else {
                if(*it != rank && it != ldistribution_)
                    --it;
                return std::distance(ldistribution_, it);
            }
        }
        template<char S, bool U>
        void setPattern(int* ldistribution, const int p, const int sizeSplit, unsigned short* const* const split = nullptr, const unsigned short* const world = nullptr) {
            ldistribution_ = ldistribution;
            size_ = p;
            int rank, size;
            MPI_Comm_rank(super::p_.getCommunicator(), &rank);
            MPI_Comm_size(super::p_.getCommunicator(), &size);
            char* pattern = new char[size * size]();
            for(unsigned short i = 0; i < super::map_.size(); ++i)
                pattern[rank * size + super::map_[i].first] = 1;
            MPI_Allreduce(MPI_IN_PLACE, pattern, size * size, MPI_CHAR, MPI_SUM, super::p_.getCommunicator());
            if(split) {
                int self = getMain(rank);
                reduction_.resize(sizeSplit);
                for(unsigned short i = 0; i < sizeSplit; ++i) {
                    for(unsigned short j = 0; j < split[i][0]; ++j) {
                        if(getMain(split[i][(U != 1 ? 3 : 1) + j]) != self) {
                            sizes_[std::make_pair(split[i][(U != 1 ? 3 : 1) + j], split[i][(U != 1 ? 3 : 1) + j])] = (U != 1 ? world[split[i][(U != 1 ? 3 : 1) + j]] : super::local_);
                            reduction_[i].emplace_back(split[i][(U != 1 ? 3 : 1) + j], split[i][(U != 1 ? 3 : 1) + j]);
                            if(S == 'S' && split[i][(U != 1 ? 3 : 1) + j] < rank + i) {
                                std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>::iterator it = extra_.find(i);
                                if(it == extra_.end()) {
                                    std::pair<std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>::iterator, bool> p = extra_.emplace(i, std::forward_as_tuple(0, 0, std::vector<unsigned short>()));
                                    it = p.first;
                                    std::get<0>(it->second) = (U != 1 ? world[rank + i] : 1);
                                    std::get<1>(it->second) = (U != 1 ? std::accumulate(world + rank, world + rank + i, 0) : i);
                                }
                                std::get<2>(it->second).emplace_back(split[i][(U != 1 ? 3 : 1) + j]);
                            }
                            for(unsigned short k = j + 1; k < split[i][0]; ++k) {
                                if(pattern[split[i][(U != 1 ? 3 : 1) + j] * size + split[i][(U != 1 ? 3 : 1) + k]] && getMain(split[i][(U != 1 ? 3 : 1) + k]) != self) {
                                    sizes_[std::make_pair(split[i][(U != 1 ? 3 : 1) + j], split[i][(U != 1 ? 3 : 1) + k])] = (U != 1 ? world[split[i][(U != 1 ? 3 : 1) + k]] : super::local_);
                                    if(S != 'S')
                                        sizes_[std::make_pair(split[i][(U != 1 ? 3 : 1) + k], split[i][(U != 1 ? 3 : 1) + j])] = (U != 1 ? world[split[i][(U != 1 ? 3 : 1) + j]] : super::local_);
                                    reduction_[i].emplace_back(split[i][(U != 1 ? 3 : 1) + j], split[i][(U != 1 ? 3 : 1) + k]);
                                    if(S != 'S')
                                        reduction_[i].emplace_back(split[i][(U != 1 ? 3 : 1) + k], split[i][(U != 1 ? 3 : 1) + j]);
                                }
                            }
                        }
                    }
                    std::sort(reduction_[i].begin(), reduction_[i].end());
                    if(S == 'S') {
                        const unsigned short first = std::distance(split[i] + (U != 1 ? 3 : 1), std::upper_bound(split[i] + (U != 1 ? 3 : 1), split[i] + (U != 1 ? 3 : 1) + split[i][0], rank + i));
                        split[i][0] -= first;
                        for(unsigned short j = 0; j < split[i][0]; ++j)
                            split[i][(U != 1 ? 3 : 1) + j] = split[i][(U != 1 ? 3 : 1) + first + j];
                    }
                }
            }
            delete [] pattern;
        }
        void initialize(unsigned int k, K*& work, unsigned short s) {
            sb_ = k;
            work = new K[2 * sb_];
#if !HPDDM_PETSC
            if(HPDDM_NUMBERING != Wrapper<K>::I) {
                if(HPDDM_NUMBERING == 'F') {
                    std::for_each(super::A_->ja_, super::A_->ja_ + super::A_->nnz_, [](int& i) { --i; });
                    std::for_each(super::A_->ia_, super::A_->ia_ + super::A_->n_ + 1, [](int& i) { --i; });
                }
                else {
                    std::for_each(super::A_->ja_, super::A_->ja_ + super::A_->nnz_, [](int& i) { ++i; });
                    std::for_each(super::A_->ia_, super::A_->ia_ + super::A_->n_ + 1, [](int& i) { ++i; });
                }
            }
#endif
            super::work_ = work + sb_;
            std::fill_n(super::work_, sb_, K());
            super::signed_ = s;
        }
        template<char S, bool U, class T>
        void applyToNeighbor(T& in, K*& work, MPI_Request*& rs, const unsigned short* info, T = nullptr, MPI_Request* = nullptr) {
            std::pair<int, int>** block = nullptr;
            std::vector<unsigned short> worker, interior, overlap, extraSend, extraRecv;
            int rank;
            MPI_Comm_rank(super::p_.getCommunicator(), &rank);
            const unsigned short main = getMain(rank);
            worker.reserve(super::map_.size());
            for(unsigned short i = 0; i < super::map_.size(); ++i)
                worker.emplace_back(getMain(super::map_[i].first));
            if(super::map_.size()) {
                block = new std::pair<int, int>*[super::map_.size()];
                *block = new std::pair<int, int>[super::map_.size() * super::map_.size()]();
                for(unsigned short i = 1; i < super::map_.size(); ++i)
                    block[i] = *block + i * super::map_.size();
            }
            std::vector<unsigned short>* accumulate = new std::vector<unsigned short>[S == 'S' ? 2 * super::map_.size() : super::map_.size()];
            for(unsigned short i = 0; i < super::map_.size(); ++i) {
                accumulate[i].resize(super::connectivity_);
                if(S == 'S')
                    accumulate[super::map_.size() + i].reserve(super::connectivity_);
            }
            MPI_Request* rq = new MPI_Request[2 * super::map_.size()];
            unsigned short* neighbors = new unsigned short[super::map_.size()];
            for(unsigned short i = 0; i < super::map_.size(); ++i)
                neighbors[i] = super::map_[i].first;
            for(unsigned short i = 0; i < super::map_.size(); ++i)
                MPI_Isend(neighbors, super::map_.size(), MPI_UNSIGNED_SHORT, super::map_[i].first, 123, super::p_.getCommunicator(), rq + super::map_.size() + i);
            for(unsigned short i = 0; i < super::map_.size(); ++i)
                MPI_Irecv(accumulate[i].data(), super::connectivity_, MPI_UNSIGNED_SHORT, super::map_[i].first, 123, super::p_.getCommunicator(), rq + i);
            for(unsigned short i = 0; i < super::map_.size(); ++i) {
                int index;
                MPI_Status st;
                MPI_Waitany(super::map_.size(), rq, &index, &st);
                int count;
                MPI_Get_count(&st, MPI_UNSIGNED_SHORT, &count);
                accumulate[index].resize(count);
            }
            int m = 0;
            for(unsigned short i = 0; i < super::map_.size(); ++i) {
                if(worker[i] != main) {
                    overlap.emplace_back(i);
                    block[i][i] = std::make_pair(U == 1 ? super::local_ : info[i], U == 1 ? super::local_ : info[i]);
                    if(block[i][i].first != 0)
                        m += (U == 1 ? super::local_ * super::local_ : info[i] * info[i]);
                    for(unsigned short j = (S != 'S' ? 0 : i + 1); j < super::map_.size(); ++j) {
                        if(i != j && worker[j] != main && std::binary_search(accumulate[j].cbegin(), accumulate[j].cend(), super::map_[i].first)) {
                            block[i][j] = std::make_pair(U == 1 ? super::local_ : info[i], U == 1 ? super::local_ : info[j]);
                            if(block[i][j].first != 0 && block[i][j].second != 0)
                                m += (U == 1 ? super::local_ * super::local_ : info[i] * info[j]);
                        }
                    }
                    if(S == 'S' && i < super::signed_)
                        m += (U == 1 ? super::local_ * super::local_ : super::local_ * info[i]);
                }
                else
                    interior.emplace_back(i);
            }
            for(unsigned short i = 0; i < overlap.size(); ++i) {
                unsigned short size = 0;
                for(unsigned short j = 0; j < accumulate[overlap[i]].size(); ++j) {
                    if(getMain(accumulate[overlap[i]][j]) == worker[overlap[i]]) {
                        unsigned short* pt = std::lower_bound(neighbors, neighbors + super::map_.size(), accumulate[overlap[i]][j]);
                        if(pt != neighbors + super::map_.size() && *pt == accumulate[overlap[i]][j])
                            accumulate[overlap[i]][size++] = std::distance(neighbors, pt);
                    }
                    if(S == 'S' && getMain(accumulate[overlap[i]][j]) == main) {
                        unsigned short* pt = std::lower_bound(neighbors, neighbors + super::map_.size(), accumulate[overlap[i]][j]);
                        if(pt != neighbors + super::map_.size() && *pt == accumulate[overlap[i]][j])
                            accumulate[super::map_.size() + overlap[i]].emplace_back(std::distance(neighbors, pt));
                    }
                }
                accumulate[overlap[i]].resize(size);
            }
            MPI_Waitall(super::map_.size(), rq + super::map_.size(), MPI_STATUSES_IGNORE);
            delete [] neighbors;
            overlap_.resize(m);
            std::vector<typename Preconditioner::integer_type> omap;
#if HPDDM_PETSC
            PetscInt n;
            PetscCallVoid(MatGetLocalSize(super::A_, &n, nullptr));
#endif
            {
                std::set<typename Preconditioner::integer_type> o;
                for(unsigned short i = 0; i < super::map_.size(); ++i)
                    o.insert(super::map_[i].second.cbegin(), super::map_[i].second.cend());
                omap.reserve(o.size());
                std::copy(o.cbegin(), o.cend(), std::back_inserter(omap));
#if !HPDDM_PETSC
                int* ia = new int[omap.size() + 1];
                ia[0] = (Wrapper<K>::I == 'F');
                std::vector<std::pair<int, K>> restriction;
                restriction.reserve(super::A_->nnz_);
                int nnz = ia[0];
                for(int i = 0; i < omap.size(); ++i) {
                    std::vector<int>::const_iterator it = omap.cbegin();
                    for(int j = super::A_->ia_[omap[i]] - (Wrapper<K>::I == 'F'); j < super::A_->ia_[omap[i] + 1] - (Wrapper<K>::I == 'F'); ++j) {
                        it = std::lower_bound(it, omap.cend(), super::A_->ja_[j] - (Wrapper<K>::I == 'F'));
                        if(it != omap.cend() && *it == super::A_->ja_[j] - (Wrapper<K>::I == 'F') && std::abs(super::A_->a_[j]) > HPDDM_EPS) {
                            restriction.emplace_back(std::distance(omap.cbegin(), it) + (Wrapper<K>::I == 'F'), super::A_->a_[j]);
                            ++nnz;
                        }
                    }
                    ia[i + 1] = nnz;
                }
                int* ja = new int[nnz - (Wrapper<K>::I == 'F')];
                K* a = new K[nnz - (Wrapper<K>::I == 'F')];
                for(int i = 0; i < nnz - (Wrapper<K>::I == 'F'); ++i) {
                    ja[i] = restriction[i].first;
                    a[i] = restriction[i].second;
                }
                super::C_ = new MatrixCSR<K>(omap.size(), omap.size(), nnz - (Wrapper<K>::I == 'F'), a, ia, ja, super::A_->sym_, true);
#else
                IS is;
                PetscInt* data;
                std::vector<PetscInt> imap;
                if(n == super::n_) {
                    if(std::is_same<typename Preconditioner::integer_type, PetscInt>::value)
                        data = reinterpret_cast<PetscInt*>(omap.data());
                    else {
                        data = new PetscInt[omap.size()];
                        std::copy_n(omap.data(), omap.size(), data);
                    }
                    PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, omap.size(), data, PETSC_USE_POINTER, &is));
                }
                else {
                    imap.reserve(omap.size());
                    const underlying_type<K>* const D = super::D_;
                    std::for_each(omap.cbegin(), omap.cend(), [&D, &imap](const typename Preconditioner::integer_type& i) { if(HPDDM::abs(D[i]) > HPDDM_EPS) imap.emplace_back(i); });
                    PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, imap.size(), imap.data(), PETSC_USE_POINTER, &is));
                }
                PetscCallVoid(MatCreateSubMatrix(super::A_, is, is, MAT_INITIAL_MATRIX, &(super::C_)));
                PetscCallVoid(ISDestroy(&is));
                if(n == super::n_ && !std::is_same<typename Preconditioner::integer_type, PetscInt>::value)
                    delete [] data;
#endif
            }
            K** tmp = new K*[2 * super::map_.size()];
            *tmp = new K[omap.size() * (U == 1 ? super::map_.size() * super::local_ : std::max(super::local_, std::accumulate(info, info + super::map_.size(), 0))) + std::max(static_cast<int>(omap.size() * (U == 1 ? super::map_.size() * super::local_ : std::max(super::local_, std::accumulate(info, info + super::map_.size(), 0)))), super::n_ * super::local_)]();
            for(unsigned short i = 1; i < super::map_.size(); ++i)
                tmp[i] = tmp[i - 1] + omap.size() * (U == 1 ? super::local_ : info[i - 1]);
            if(super::map_.size()) {
                tmp[super::map_.size()] = tmp[super::map_.size() - 1] + omap.size() * (U == 1 ? super::local_ : info[super::map_.size() - 1]);
                for(unsigned short i = 1; i < super::map_.size(); ++i)
                    tmp[super::map_.size() + i] = tmp[super::map_.size() + i - 1] + std::distance(tmp[i - 1], tmp[i]);
            }
            std::vector<std::vector<int>> nmap(super::map_.size());
            for(unsigned short i = 0; i < super::map_.size(); ++i) {
                nmap[i].reserve(super::map_[i].second.size());
                for(unsigned int j = 0; j < super::map_[i].second.size(); ++j) {
                    std::vector<int>::const_iterator it = std::lower_bound(omap.cbegin(), omap.cend(), super::map_[i].second[j]);
                    nmap[i].emplace_back(std::distance(omap.cbegin(), it));
                }
            }
            K** buff = new K*[2 * super::map_.size()];
            m = 0;
            for(unsigned short i = 0; i < super::map_.size(); ++i)
                m += super::map_[i].second.size() * 2 * (U == 1 ? super::local_ : std::max(static_cast<unsigned short>(super::local_), info[i]));
            *buff = new K[m];
            m = 0;
            for(unsigned short i = 0; i < super::map_.size(); ++i) {
                buff[i] = *buff + m;
                MPI_Irecv(buff[i], super::map_[i].second.size() * (U == 1 ? super::local_ : info[i]), Wrapper<K>::mpi_type(), super::map_[i].first, 20, super::p_.getCommunicator(), rq + i);
                m += super::map_[i].second.size() * (U == 1 ? super::local_ : std::max(static_cast<unsigned short>(super::local_), info[i]));
            }
            if(super::local_)
                Wrapper<K>::diag(super::n_, super::D_, *super::deflation_, *tmp, super::local_);
            for(unsigned short i = 0; i < super::map_.size(); ++i) {
                buff[super::map_.size() + i] = *buff + m;
                for(unsigned short j = 0; j < super::local_; ++j)
                    Wrapper<K>::gthr(super::map_[i].second.size(), *tmp + j * super::n_, buff[super::map_.size() + i] + j * super::map_[i].second.size(), super::map_[i].second.data());
                MPI_Isend(buff[super::map_.size() + i], super::map_[i].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[i].first, 20, super::p_.getCommunicator(), rq + super::map_.size() + i);
                m += super::map_[i].second.size() * (U == 1 ? super::local_ : std::max(static_cast<unsigned short>(super::local_), info[i]));
            }
#if !HPDDM_PETSC
            Wrapper<K>::template csrmm<Wrapper<K>::I>(super::A_->sym_, &(super::n_), &(super::local_), super::A_->a_, super::A_->ia_, super::A_->ja_, *tmp, super::work_);
#else
            Mat Z, P;
            if(n == super::n_) {
                PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, super::local_, *tmp, &Z));
                PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, super::n_, super::local_, super::work_, &P));
            }
            else {
                PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, n, super::local_, nullptr, &Z));
                PetscScalar* array;
                PetscCallVoid(MatDenseGetArrayWrite(Z, &array));
                for(PetscInt i = 0, k = 0; i < super::n_; ++i) {
                    if(HPDDM::abs(super::D_[i]) > HPDDM_EPS) {
                        for(unsigned short j = 0; j < super::local_; ++j)
                            array[k + j * n] = tmp[0][i + j * super::n_];
                        ++k;
                    }

                }
                PetscCallVoid(MatDenseRestoreArrayWrite(Z, &array));
                PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, n, super::local_, nullptr, &P));
            }
            PetscCallVoid(MatMatMult(super::A_, Z, MAT_REUSE_MATRIX, PETSC_DEFAULT, &P));
            PetscCallVoid(MatDestroy(&Z));
            if(n != super::n_) {
                const PetscScalar* array;
                PetscCallVoid(MatDenseGetArrayRead(P, &array));
                std::fill_n(super::work_, super::local_ * super::n_, K());
                for(PetscInt i = 0, k = 0; i < super::n_; ++i) {
                    if(HPDDM::abs(super::D_[i]) > HPDDM_EPS) {
                        for(unsigned short j = 0; j < super::local_; ++j)
                            super::work_[i + j * super::n_] = array[k + j * n];
                        ++k;
                    }

                }
                PetscCallVoid(MatDenseRestoreArrayRead(P, &array));
            }
            PetscCallVoid(MatDestroy(&P));
#endif
            std::fill_n(*tmp, super::local_ * super::n_, K());
            for(unsigned short i = 0; i < super::map_.size(); ++i) {
                int index;
                MPI_Waitany(super::map_.size(), rq, &index, MPI_STATUS_IGNORE);
                for(unsigned short k = 0; k < (U ? super::local_ : info[index]); ++k)
                    Wrapper<K>::sctr(nmap[index].size(), buff[index] + k * nmap[index].size(), nmap[index].data(), tmp[index] + k * omap.size());
            }
            for(unsigned short i = 0; i < super::map_.size(); ++i)
                MPI_Irecv(buff[i], super::map_[i].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[i].first, 21, super::p_.getCommunicator(), rq + i);
            m = std::distance(tmp[0], tmp[super::map_.size()]) / omap.size();
#if !HPDDM_PETSC
            {
                std::vector<unsigned short>* compute = new std::vector<unsigned short>[super::C_->n_]();
                for(unsigned short i = 0; i < super::map_.size(); ++i)
                    for(unsigned int j = 0; j < super::map_[i].second.size(); ++j) {
                        std::vector<int>::const_iterator it = std::lower_bound(omap.cbegin(), omap.cend(), super::map_[i].second[j]);
                        compute[std::distance(omap.cbegin(), it)].emplace_back(i);
                    }
                std::fill_n(tmp[super::map_.size()], super::C_->n_ * m, K());
                for(int i = 0; i < super::C_->n_; ++i) {
                    for(int j = super::C_->ia_[i] - (Wrapper<K>::I == 'F'); j < super::C_->ia_[i + 1] - (Wrapper<K>::I == 'F'); ++j) {
                        for(unsigned short k = 0; k < compute[super::C_->ja_[j] - (Wrapper<K>::I == 'F')].size(); ++k) {

                            const int m = (U == 1 ? super::local_ : info[compute[super::C_->ja_[j] - (Wrapper<K>::I == 'F')][k]]);
                            Blas<K>::axpy(&m, super::C_->a_ + j, tmp[compute[super::C_->ja_[j] - (Wrapper<K>::I == 'F')][k]] + super::C_->ja_[j] - (Wrapper<K>::I == 'F'), &(super::C_->n_), tmp[super::map_.size() + compute[super::C_->ja_[j] - (Wrapper<K>::I == 'F')][k]] + i, &(super::C_->n_));
                            if(super::C_->sym_ && i != super::C_->ja_[j] - (Wrapper<K>::I == 'F')) {

                                const int m = (U == 1 ? super::local_ : info[compute[super::C_->ja_[j] - (Wrapper<K>::I == 'F')][k]]);
                                Blas<K>::axpy(&m, super::C_->a_ + j, tmp[compute[super::C_->ja_[j] - (Wrapper<K>::I == 'F')][k]] + i, &(super::C_->n_), tmp[super::map_.size() + compute[super::C_->ja_[j] - (Wrapper<K>::I == 'F')][k]] + super::C_->ja_[j] - (Wrapper<K>::I == 'F'), &(super::C_->n_));
                            }
                        }
                    }
                }
                delete [] compute;
            }
            delete super::C_;
#else
            {
                Mat Z, P;
                std::vector<PetscInt> imap;
                if(n == super::n_) {
                    PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, omap.size(), m, *tmp, &Z));
                    PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, omap.size(), m, tmp[super::map_.size()], &P));
                }
                else {
                    imap.reserve(omap.size());
                    for(typename std::vector<typename Preconditioner::integer_type>::const_iterator it = omap.cbegin(); it != omap.cend(); ++it) {
                        if(HPDDM::abs(super::D_[*it]) > HPDDM_EPS)
                            imap.emplace_back(std::distance(omap.cbegin(), it));
                    }
                    PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, imap.size(), m, nullptr, &Z));
                    PetscScalar* array;
                    PetscCallVoid(MatDenseGetArray(Z, &array));
                    for(int i = 0; i < imap.size(); ++i) {
                        for(unsigned short j = 0; j < m; ++j)
                            array[i + j * imap.size()] = tmp[0][imap[i] + j * omap.size()];
                    }
                    PetscCallVoid(MatDenseRestoreArray(Z, &array));
                    PetscCallVoid(MatCreateSeqDense(PETSC_COMM_SELF, imap.size(), m, nullptr, &P));
                }
                PetscCallVoid(MatMatMult(super::C_, Z, MAT_REUSE_MATRIX, PETSC_DEFAULT, &P));
                PetscCallVoid(MatDestroy(&Z));
                if(n != super::n_) {
                    PetscScalar* array;
                    PetscCallVoid(MatDenseGetArray(P, &array));
                    std::fill_n(tmp[super::map_.size()], m * omap.size(), K());
                    for(int i = 0; i < imap.size(); ++i) {
                        for(unsigned short j = 0; j < m; ++j)
                            tmp[super::map_.size()][imap[i] + j * omap.size()] = array[i + j * imap.size()];
                    }
                    PetscCallVoid(MatDenseRestoreArray(P, &array));
                }
                PetscCallVoid(MatDestroy(&P));
            }
            PetscCallVoid(MatDestroy(&(super::C_)));
#endif
            MPI_Waitall(super::map_.size(), rq + super::map_.size(), MPI_STATUSES_IGNORE);
            for(unsigned short i = 0; i < super::map_.size(); ++i) {
                m = (U == 1 ? super::local_ : info[i]);
                for(unsigned short j = 0; j < m; ++j)
                    Wrapper<K>::gthr(nmap[i].size(), tmp[super::map_.size() + i] + j * omap.size(), buff[super::map_.size() + i] + j * nmap[i].size(), nmap[i].data());
                MPI_Isend(buff[super::map_.size() + i], super::map_[i].second.size() * m, Wrapper<K>::mpi_type(), super::map_[i].first, 21, super::p_.getCommunicator(), rq + super::map_.size() + i);
            }
            K* pt = overlap_.data();
            for(unsigned short i = 0; i < overlap.size(); ++i) {
                for(unsigned short j = 0; j < overlap.size(); ++j) {
                    if(block[overlap[i]][overlap[j]].first != 0 && block[overlap[i]][overlap[j]].second != 0) {
                        const int n = omap.size();
                        Blas<K>::gemm(&(Wrapper<K>::transc), "N", &(block[overlap[i]][overlap[j]].first), &(block[overlap[i]][overlap[j]].second), &n, &(Wrapper<K>::d__1), tmp[overlap[i]], &n, tmp[super::map_.size() + overlap[j]], &n, &(Wrapper<K>::d__0), pt, &(block[overlap[i]][overlap[j]].first));
                        pt += (U == 1 ? super::local_ * super::local_ : info[overlap[i]] * info[overlap[j]]);
                    }
                }
            }
            if(block) {
                delete [] *block;
                delete [] block;
            }
#if !HPDDM_PETSC
            if(HPDDM_NUMBERING != Wrapper<K>::I) {
                if(Wrapper<K>::I == 'F') {
                    std::for_each(super::A_->ja_, super::A_->ja_ + super::A_->nnz_, [](int& i) { --i; });
                    std::for_each(super::A_->ia_, super::A_->ia_ + super::A_->n_ + 1, [](int& i) { --i; });
                }
                else {
                    std::for_each(super::A_->ja_, super::A_->ja_ + super::A_->nnz_, [](int& i) { ++i; });
                    std::for_each(super::A_->ia_, super::A_->ia_ + super::A_->n_ + 1, [](int& i) { ++i; });
                }
            }
#endif
            MPI_Waitall(super::map_.size(), rq, MPI_STATUSES_IGNORE);
            if(S == 'S') {
                for(unsigned short i = 0; i < overlap.size() && overlap[i] < super::signed_; ++i) {
                    m = (U == 1 ? super::local_ : info[overlap[i]]);
                    if(m) {
                        for(unsigned short nu = 0; nu < super::local_; ++nu)
                            Wrapper<K>::gthr(omap.size(), super::work_ + nu * super::n_, work + nu * omap.size(), omap.data());
                        const std::vector<unsigned short>& r = accumulate[super::map_.size() + overlap[i]];
                        for(unsigned short j = 0; j < r.size(); ++j) {
                            for(unsigned short nu = 0; nu < super::local_; ++nu) {
                                std::fill_n(tmp[super::map_.size()], omap.size(), K());
                                Wrapper<K>::sctr(nmap[r[j]].size(), buff[r[j]] + nu * nmap[r[j]].size(), nmap[r[j]].data(), tmp[super::map_.size()]);
                                for(unsigned int k = 0; k < nmap[overlap[i]].size(); ++k)
                                    work[nmap[overlap[i]][k] + nu * omap.size()] += tmp[super::map_.size()][nmap[overlap[i]][k]];
                            }
                        }
                        const int n = omap.size();
                        Blas<K>::gemm(&(Wrapper<K>::transc), "N", &m, &(super::local_), &n, &(Wrapper<K>::d__1), tmp[overlap[i]], &n, work, &n, &(Wrapper<K>::d__0), pt, &m);
                        pt += super::local_ * m;
                    }
                }
            }
            delete [] *tmp;
            delete [] tmp;
            for(unsigned short i = 0; i < overlap.size() && overlap[i] < super::signed_; ++i) {
                if(U || info[overlap[i]]) {
                    for(unsigned short nu = 0; nu < super::local_; ++nu)
                        Wrapper<K>::gthr(super::map_[overlap[i]].second.size(), super::work_ + nu * super::n_, in[overlap[i]] + nu * super::map_[overlap[i]].second.size(), super::map_[overlap[i]].second.data());
                    const std::vector<unsigned short>& r = accumulate[overlap[i]];
                    for(unsigned short k = 0; k < r.size(); ++k) {
                        for(unsigned short nu = 0; nu < super::local_; ++nu) {
                            std::fill_n(work, omap.size(), K());
                            Wrapper<K>::sctr(nmap[r[k]].size(), buff[r[k]] + nu * nmap[r[k]].size(), nmap[r[k]].data(), work);
                            for(unsigned int j = 0; j < super::map_[overlap[i]].second.size(); ++j)
                                in[overlap[i]][j + nu * super::map_[overlap[i]].second.size()] += work[nmap[overlap[i]][j]];
                        }
                    }
                    MPI_Isend(in[overlap[i]], super::map_[overlap[i]].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[overlap[i]].first, 2, super::p_.getCommunicator(), rs + overlap[i]);
                }
            }
            for(unsigned short i = 0; i < interior.size(); ++i) {
                for(unsigned short k = 0; k < super::local_; ++k)
                    for(unsigned int j = 0; j < super::map_[interior[i]].second.size(); ++j)
                        super::work_[super::map_[interior[i]].second[j] + k * super::n_] += buff[interior[i]][j + k * super::map_[interior[i]].second.size()];
            }
            for(unsigned short i = 0; i < interior.size() && interior[i] < super::signed_; ++i) {
                if(U || info[interior[i]]) {
                    for(unsigned short nu = 0; nu < super::local_; ++nu)
                        Wrapper<K>::gthr(super::map_[interior[i]].second.size(), super::work_ + nu * super::n_, in[interior[i]] + nu * super::map_[interior[i]].second.size(), super::map_[interior[i]].second.data());
                    MPI_Isend(in[interior[i]], super::map_[interior[i]].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[interior[i]].first, 2, super::p_.getCommunicator(), rs + interior[i]);
                }
            }
            rs += super::signed_;
            delete [] accumulate;
            Wrapper<K>::diag(super::n_, super::D_, super::work_, work, super::local_);
            MPI_Waitall(super::map_.size(), rq + super::map_.size(), MPI_STATUSES_IGNORE);
            delete [] rq;
            delete [] *buff;
            delete [] buff;
        }
};
#endif // HPDDM_SCHWARZ

#if HPDDM_FETI
template<class Preconditioner, FetiPrcndtnr Q, class K>
class FetiProjection : public OperatorBase<Q == FetiPrcndtnr::SUPERLUMPED ? 'f' : 'c', Preconditioner, K> {
    private:
        typedef OperatorBase<Q == FetiPrcndtnr::SUPERLUMPED ? 'f' : 'c', Preconditioner, K> super;
        template<char S, bool U>
        void applyFromNeighbor(const K* in, unsigned short index, K*& work, unsigned short* info) {
            std::vector<unsigned short>::const_iterator middle = std::lower_bound(super::vecSparsity_[index].cbegin(), super::vecSparsity_[index].cend(), super::rank_);
            unsigned int accumulate = 0;
            if(!(index < super::signed_)) {
                for(unsigned short k = 0; k < (U ? super::local_ : info[std::distance(super::sparsity_.cbegin(), std::lower_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::map_[index].first))]); ++k)
                    for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                        work[super::offsets_[super::map_[index].first] + super::map_[index].second[j] + k * super::n_] += in[k * super::map_[index].second.size() + j];
                accumulate += (U ? super::local_ : info[std::distance(super::sparsity_.cbegin(), std::lower_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::map_[index].first))]) * super::map_[index].second.size();
            }
            else if(S != 'S') {
                for(unsigned short k = 0; k < (U ? super::local_ : info[std::distance(super::sparsity_.cbegin(), std::lower_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::map_[index].first))]); ++k)
                    for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                        work[super::offsets_[super::map_[index].first] + super::map_[index].second[j] + k * super::n_] -= in[k * super::map_[index].second.size() + j];
                accumulate += (U ? super::local_ : info[std::distance(super::sparsity_.cbegin(), std::lower_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::map_[index].first))]) * super::map_[index].second.size();
            }
            std::vector<unsigned short>::const_iterator begin = super::sparsity_.cbegin();
            if(S != 'S')
                for(std::vector<unsigned short>::const_iterator it = super::vecSparsity_[index].cbegin(); it != middle; ++it) {
                    if(!U) {
                        std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::sparsity_.cend(), *it);
                        if(*it > super::map_[index].first || super::signed_ > index)
                            for(unsigned short k = 0; k < info[std::distance(super::sparsity_.cbegin(), idx)]; ++k)
                                for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                                    work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] -= in[accumulate + k * super::map_[index].second.size() + j];
                        else
                            for(unsigned short k = 0; k < info[std::distance(super::sparsity_.cbegin(), idx)]; ++k)
                                for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                                    work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
                        accumulate += info[std::distance(super::sparsity_.cbegin(), idx)] * super::map_[index].second.size();
                        begin = idx + 1;
                    }
                    else {
                        if(*it > super::map_[index].first || super::signed_ > index)
                            for(unsigned short k = 0; k < super::local_; ++k)
                                for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                                    work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] -= in[accumulate + k * super::map_[index].second.size() + j];
                        else
                            for(unsigned short k = 0; k < super::local_; ++k)
                                for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                                    work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
                        accumulate += super::local_ * super::map_[index].second.size();
                    }
                }
            if(index < super::signed_)
                for(unsigned short k = 0; k < super::local_; ++k)
                    for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                        work[super::offsets_[super::rank_] + super::map_[index].second[j] + k * super::n_] -= in[accumulate + k * super::map_[index].second.size() + j];
            else
                for(unsigned short k = 0; k < super::local_; ++k)
                    for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                        work[super::offsets_[super::rank_] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
            accumulate += super::local_ * super::map_[index].second.size();
            for(std::vector<unsigned short>::const_iterator it = middle + 1; it < super::vecSparsity_[index].cend(); ++it) {
                if(!U) {
                    std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::sparsity_.cend(), *it);
                    if(*it > super::map_[index].first && super::signed_ > index)
                        for(unsigned short k = 0; k < info[std::distance(super::sparsity_.cbegin(), idx)]; ++k)
                            for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                                work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] -= in[accumulate + k * super::map_[index].second.size() + j];
                    else
                        for(unsigned short k = 0; k < info[std::distance(super::sparsity_.cbegin(), idx)]; ++k)
                            for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                                work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
                    accumulate += info[std::distance(super::sparsity_.cbegin(), idx)] * super::map_[index].second.size();
                    begin = idx + 1;
                }
                else {
                    if(*it > super::map_[index].first && super::signed_ > index)
                        for(unsigned short k = 0; k < super::local_; ++k)
                            for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                                work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] -= in[accumulate + k * super::map_[index].second.size() + j];
                    else
                        for(unsigned short k = 0; k < super::local_; ++k)
                            for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                                work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
                    accumulate += super::local_ * super::map_[index].second.size();
                }
            }
        }
    public:
        HPDDM_CLASS_COARSE_OPERATOR(Solver, S, T) friend class CoarseOperator;
        FetiProjection(const Preconditioner& p, const unsigned short& c, const unsigned int& max) : super(p, c, max) { }
        template<char S, bool U, class T>
        void applyToNeighbor(T& in, K*& work, MPI_Request*& rq, const unsigned short* info, T const& out = nullptr, MPI_Request* const& rqRecv = nullptr) {
            unsigned short* infoNeighbor;
            super::template initialize<S, U>(in, info, out, rqRecv, infoNeighbor);
            MPI_Request* rqMult = new MPI_Request[2 * super::map_.size()];
            unsigned int* offset = new unsigned int[super::map_.size() + 2];
            offset[0] = 0;
            offset[1] = super::local_;
            for(unsigned short i = 2; i < super::map_.size() + 2; ++i)
                offset[i] = offset[i - 1] + (U ? super::local_ : infoNeighbor[i - 2]);
            const int nbMult = super::p_.getMult();
            K* mult = new K[offset[super::map_.size() + 1] * nbMult];
            unsigned short* displs = new unsigned short[super::map_.size() + 1];
            displs[0] = 0;
            for(unsigned short i = 0; i < super::map_.size(); ++i) {
                MPI_Irecv(mult + offset[i + 1] * nbMult + displs[i] * (U ? super::local_ : infoNeighbor[i]), super::map_[i].second.size() * (U ? super::local_ : infoNeighbor[i]), Wrapper<K>::mpi_type(), super::map_[i].first, 11, super::p_.getCommunicator(), rqMult + i);
                displs[i + 1] = displs[i] + super::map_[i].second.size();
            }

            K* tmp = new K[offset[super::map_.size() + 1] * super::n_]();
            const underlying_type<K>* const* const m = super::p_.getScaling();
            for(unsigned short i = 0; i < super::signed_; ++i) {
                for(unsigned short k = 0; k < super::local_; ++k)
                    for(unsigned int j = 0; j < super::map_[i].second.size(); ++j)
                        tmp[super::map_[i].second[j] + k * super::n_] -= m[i][j] * (mult[displs[i] * super::local_ + j + k * super::map_[i].second.size()] = - super::deflation_[k][super::map_[i].second[j]]);
                MPI_Isend(mult + displs[i] * super::local_, super::map_[i].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[i].first, 11, super::p_.getCommunicator(), rqMult + super::map_.size() + i);
            }
            for(unsigned short i = super::signed_; i < super::map_.size(); ++i) {
                for(unsigned short k = 0; k < super::local_; ++k)
                    for(unsigned int j = 0; j < super::map_[i].second.size(); ++j)
                        tmp[super::map_[i].second[j] + k * super::n_] += m[i][j] * (mult[displs[i] * super::local_ + j + k * super::map_[i].second.size()] =   super::deflation_[k][super::map_[i].second[j]]);
                MPI_Isend(mult + displs[i] * super::local_, super::map_[i].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[i].first, 11, super::p_.getCommunicator(), rqMult + super::map_.size() + i);
            }

            for(unsigned short i = 0; i < super::map_.size(); ++i) {
                int index;
                MPI_Waitany(super::map_.size(), rqMult, &index, MPI_STATUS_IGNORE);
                if(index < super::signed_)
                    for(unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[index]); ++k)
                        for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                            tmp[super::map_[index].second[j] + (offset[index + 1] + k) * super::n_] = - m[index][j] * mult[offset[index + 1] * nbMult + displs[index] * (U ? super::local_ : infoNeighbor[index]) + j + k * super::map_[index].second.size()];
                else
                    for(unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[index]); ++k)
                        for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                            tmp[super::map_[index].second[j] + (offset[index + 1] + k) * super::n_] =   m[index][j] * mult[offset[index + 1] * nbMult + displs[index] * (U ? super::local_ : infoNeighbor[index]) + j + k * super::map_[index].second.size()];
            }

            delete [] displs;

            if(offset[super::map_.size() + 1])
                super::p_.applyLocalPreconditioner(tmp, offset[super::map_.size() + 1]);

            MPI_Waitall(super::map_.size(), rqMult + super::map_.size(), MPI_STATUSES_IGNORE);
            delete [] rqMult;
            delete [] mult;

            unsigned int accumulate = 0;
            unsigned short stop = std::distance(super::sparsity_.cbegin(), std::upper_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::rank_));
            if(S != 'S') {
                super::offsets_.reserve(super::sparsity_.size() + 1);
                for(unsigned short i = 0; i < stop; ++i) {
                    super::offsets_.emplace(super::sparsity_[i], accumulate);
                    accumulate += super::n_ * (U ? super::local_ : info[i]);
                }
            }
            else
                super::offsets_.reserve(super::sparsity_.size() + 1 - stop);
            super::offsets_.emplace(super::rank_, accumulate);
            accumulate += super::n_ * super::local_;
            for(unsigned short i = stop; i < super::sparsity_.size(); ++i) {
                super::offsets_.emplace(super::sparsity_[i], accumulate);
                accumulate += super::n_ * (U ? super::local_ : info[i]);
            }

            work = new K[accumulate]();

            for(unsigned short i = 0; i < super::signed_; ++i) {
                accumulate = super::local_;
                for(unsigned short k = 0; k < super::local_; ++k)
                    for(unsigned int j = 0; j < super::map_[i].second.size(); ++j)
                        work[super::offsets_[super::rank_] + super::map_[i].second[j] + k * super::n_] -= (in[i][k * super::map_[i].second.size() + j] = - m[i][j] * tmp[super::map_[i].second[j] + k * super::n_]);
                for(unsigned short l = (S != 'S' ? 0 : i); l < super::map_.size(); ++l) {
                    if(Q == FetiPrcndtnr::SUPERLUMPED && l != i && !std::binary_search(super::vecSparsity_[i].cbegin(), super::vecSparsity_[i].cend(), super::map_[l].first)) {
                        if(S != 'S' || !(l < super::signed_))
                            for(unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[l]); ++k)
                                for(unsigned int j = 0; j < super::map_[i].second.size(); ++j)
                                    work[super::offsets_[super::map_[l].first] + super::map_[i].second[j] + k * super::n_] -= - m[i][j] * tmp[super::map_[i].second[j] + (offset[l + 1] + k) * super::n_];
                        continue;
                    }
                    for(unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[l]); ++k)
                        for(unsigned int j = 0; j < super::map_[i].second.size(); ++j) {
                            if(S != 'S' || !(l < super::signed_))
                                work[super::offsets_[super::map_[l].first] + super::map_[i].second[j] + k * super::n_] -= (in[i][(accumulate + k) * super::map_[i].second.size() + j] = - m[i][j] * tmp[super::map_[i].second[j] + (offset[l + 1] + k) * super::n_]);
                            else
                                in[i][(accumulate + k) * super::map_[i].second.size() + j] = - m[i][j] * tmp[super::map_[i].second[j] + (offset[l + 1] + k) * super::n_];
                        }
                    accumulate += U ? super::local_ : infoNeighbor[l];
                }
                if(U || infoNeighbor[i])
                    MPI_Isend(in[i], super::map_[i].second.size() * accumulate, Wrapper<K>::mpi_type(), super::map_[i].first, 2, super::p_.getCommunicator(), rq++);
            }
            for(unsigned short i = super::signed_; i < super::map_.size(); ++i) {
                if(S != 'S') {
                    accumulate = super::local_;
                    for(unsigned short k = 0; k < super::local_; ++k)
                        for(unsigned int j = 0; j < super::map_[i].second.size(); ++j)
                            work[super::offsets_[super::rank_] + super::map_[i].second[j] + k * super::n_] += (in[i][k * super::map_[i].second.size() + j] =   m[i][j] * tmp[super::map_[i].second[j] + k * super::n_]);
                }
                else {
                    accumulate = 0;
                    for(unsigned short k = 0; k < super::local_; ++k)
                        for(unsigned int j = 0; j < super::map_[i].second.size(); ++j)
                            work[super::offsets_[super::rank_] + super::map_[i].second[j] + k * super::n_] += m[i][j] * tmp[super::map_[i].second[j] + k * super::n_];
                }
                for(unsigned short l = S != 'S' ? 0 : super::signed_; l < super::map_.size(); ++l) {
                    if(Q == FetiPrcndtnr::SUPERLUMPED && l != i && !std::binary_search(super::vecSparsity_[i].cbegin(), super::vecSparsity_[i].cend(), super::map_[l].first)) {
                        if(S != 'S' || !(l < i))
                            for(unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[l]); ++k)
                                for(unsigned int j = 0; j < super::map_[i].second.size(); ++j)
                                    work[super::offsets_[super::map_[l].first] + super::map_[i].second[j] + k * super::n_] +=   m[i][j] * tmp[super::map_[i].second[j] + (offset[l + 1] + k) * super::n_];
                        continue;
                    }
                    for(unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[l]); ++k)
                        for(unsigned int j = 0; j < super::map_[i].second.size(); ++j) {
                            if(S != 'S' || !(l < i))
                                work[super::offsets_[super::map_[l].first] + super::map_[i].second[j] + k * super::n_] += (in[i][(accumulate + k) * super::map_[i].second.size() + j] =   m[i][j] * tmp[super::map_[i].second[j] + (offset[l + 1] + k) * super::n_]);
                            else
                                work[super::offsets_[super::map_[l].first] + super::map_[i].second[j] + k * super::n_] += m[i][j] * tmp[super::map_[i].second[j] + (offset[l + 1] + k) * super::n_];
                        }
                    if(S != 'S' || !(l < i))
                        accumulate += U ? super::local_ : infoNeighbor[l];
                }
                if(U || infoNeighbor[i])
                    MPI_Isend(in[i], super::map_[i].second.size() * accumulate, Wrapper<K>::mpi_type(), super::map_[i].first, 2, super::p_.getCommunicator(), rq++);
            }
            delete [] tmp;
            delete [] offset;
            if(!U && Q != FetiPrcndtnr::SUPERLUMPED)
                delete [] infoNeighbor;
        }
        template<char S, bool U>
        void assembleForMain(K* C, const K* in, const int& coefficients, unsigned short index, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            applyFromNeighbor<S, U>(in, index, arrayC, infoNeighbor);
            if(++super::consolidate_ == super::map_.size()) {
                if(S != 'S')
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &coefficients, &(super::local_), &(super::n_), &(Wrapper<K>::d__1), arrayC, &(super::n_), *super::deflation_, super::p_.getLDR(), &(Wrapper<K>::d__0), C, &coefficients);
                else
                    for(unsigned short j = 0; j < super::local_; ++j) {
                        int local = coefficients + super::local_ - j;
                        Blas<K>::gemv(&(Wrapper<K>::transc), &(super::n_), &local, &(Wrapper<K>::d__1), arrayC + super::n_ * j, &(super::n_), super::deflation_[j], &i__1, &(Wrapper<K>::d__0), C - (j * (j - 1)) / 2 + j * (coefficients + super::local_), &i__1);
                    }
            }
        }
        template<char S, char N, bool U>
        void applyFromNeighborMain(const K* in, unsigned short index, int* I, int* J, K* C, int coefficients, unsigned int offsetI, unsigned int* offsetJ, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            assembleForMain<S, U>(C, in, coefficients, index, arrayC, infoNeighbor);
            super::template assembleOperator<S, N, U>(I, J, coefficients, offsetI, offsetJ, infoNeighbor);
        }
};
#endif // HPDDM_FETI

#if HPDDM_BDD
template<class Preconditioner, class K>
class BddProjection : public OperatorBase<'c', Preconditioner, K> {
    private:
        typedef OperatorBase<'c', Preconditioner, K> super;
        template<char S, bool U>
        void applyFromNeighbor(const K* in, unsigned short index, K*& work, unsigned short* info) {
            std::vector<unsigned short>::const_iterator middle = std::lower_bound(super::vecSparsity_[index].cbegin(), super::vecSparsity_[index].cend(), super::rank_);
            unsigned int accumulate = 0;
            if(S != 'S' || !(index < super::signed_)) {
                for(unsigned short k = 0; k < (U ? super::local_ : info[std::distance(super::sparsity_.cbegin(), std::lower_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::map_[index].first))]); ++k)
                    for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                        work[super::offsets_[super::map_[index].first] + super::map_[index].second[j] + k * super::n_] += in[k * super::map_[index].second.size() + j];
                accumulate += (U ? super::local_ : info[std::distance(super::sparsity_.cbegin(), std::lower_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::map_[index].first))]) * super::map_[index].second.size();
            }
            std::vector<unsigned short>::const_iterator begin = super::sparsity_.cbegin();
            if(S != 'S')
                for(std::vector<unsigned short>::const_iterator it = super::vecSparsity_[index].cbegin(); it != middle; ++it) {
                    if(!U) {
                        std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::sparsity_.cend(), *it);
                        for(unsigned short k = 0; k < info[std::distance(super::sparsity_.cbegin(), idx)]; ++k)
                            for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                                work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
                        accumulate += info[std::distance(super::sparsity_.cbegin(), idx)] * super::map_[index].second.size();
                        begin = idx + 1;
                    }
                    else {
                        for(unsigned short k = 0; k < super::local_; ++k)
                            for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                                work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
                        accumulate += super::local_ * super::map_[index].second.size();
                    }
                }
            for(unsigned short k = 0; k < super::local_; ++k) {
                for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                    work[super::offsets_[super::rank_] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
            }
            accumulate += super::local_ * super::map_[index].second.size();
            for(std::vector<unsigned short>::const_iterator it = middle + 1; it < super::vecSparsity_[index].cend(); ++it) {
                if(!U) {
                    std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::sparsity_.cend(), *it);
                    for(unsigned short k = 0; k < info[std::distance(super::sparsity_.cbegin(), idx)]; ++k)
                        for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                            work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
                    accumulate += info[std::distance(super::sparsity_.cbegin(), idx)] * super::map_[index].second.size();
                    begin = idx + 1;
                }
                else {
                    for(unsigned short k = 0; k < super::local_; ++k)
                        for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                            work[super::offsets_[*it] + super::map_[index].second[j] + k * super::n_] += in[accumulate + k * super::map_[index].second.size() + j];
                    accumulate += super::local_ * super::map_[index].second.size();
                }
            }
        }
    public:
        HPDDM_CLASS_COARSE_OPERATOR(Solver, S, T) friend class CoarseOperator;
        BddProjection(const Preconditioner& p, const unsigned short& c, const int& max) : super(p, c, max) { }
        template<char S, bool U, class T>
        void applyToNeighbor(T& in, K*& work, MPI_Request*& rq, const unsigned short* info, T const& out = nullptr, MPI_Request* const& rqRecv = nullptr) {
            unsigned short* infoNeighbor;
            super::template initialize<S, U>(in, info, out, rqRecv, infoNeighbor);
            MPI_Request* rqMult = new MPI_Request[2 * super::map_.size()];
            unsigned int* offset = new unsigned int[super::map_.size() + 2];
            offset[0] = 0;
            offset[1] = super::local_;
            for(unsigned short i = 2; i < super::map_.size() + 2; ++i)
                offset[i] = offset[i - 1] + (U ? super::local_ : infoNeighbor[i - 2]);
            const int nbMult = super::p_.getMult();
            K* mult = new K[offset[super::map_.size() + 1] * nbMult];
            unsigned short* displs = new unsigned short[super::map_.size() + 1];
            displs[0] = 0;
            for(unsigned short i = 0; i < super::map_.size(); ++i) {
                MPI_Irecv(mult + offset[i + 1] * nbMult + displs[i] * (U ? super::local_ : infoNeighbor[i]), super::map_[i].second.size() * (U ? super::local_ : infoNeighbor[i]), Wrapper<K>::mpi_type(), super::map_[i].first, 11, super::p_.getCommunicator(), rqMult + i);
                displs[i + 1] = displs[i] + super::map_[i].second.size();
            }

            K* tmp = new K[offset[super::map_.size() + 1] * super::n_]();
            const underlying_type<K>* const m = super::p_.getScaling();
            for(unsigned short i = 0; i < super::map_.size(); ++i) {
                for(unsigned short k = 0; k < super::local_; ++k)
                    for(unsigned int j = 0; j < super::map_[i].second.size(); ++j)
                        tmp[super::map_[i].second[j] + k * super::n_] = (mult[displs[i] * super::local_ + j + k * super::map_[i].second.size()] = m[super::map_[i].second[j]] * super::deflation_[k][super::map_[i].second[j]]);
                MPI_Isend(mult + displs[i] * super::local_, super::map_[i].second.size() * super::local_, Wrapper<K>::mpi_type(), super::map_[i].first, 11, super::p_.getCommunicator(), rqMult + super::map_.size() + i);
            }

            for(unsigned short i = 0; i < super::map_.size(); ++i) {
                int index;
                MPI_Waitany(super::map_.size(), rqMult, &index, MPI_STATUS_IGNORE);
                for(unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[index]); ++k)
                    for(unsigned int j = 0; j < super::map_[index].second.size(); ++j)
                        tmp[super::map_[index].second[j] + (offset[index + 1] + k) * super::n_] = mult[offset[index + 1] * nbMult + displs[index] * (U ? super::local_ : infoNeighbor[index]) + j + k * super::map_[index].second.size()];
            }

            delete [] displs;

            if(offset[super::map_.size() + 1])
                super::p_.applyLocalSchurComplement(tmp, offset[super::map_.size() + 1]);

            MPI_Waitall(super::map_.size(), rqMult + super::map_.size(), MPI_STATUSES_IGNORE);
            delete [] rqMult;
            delete [] mult;

            unsigned int accumulate = 0;
            unsigned short stop = std::distance(super::sparsity_.cbegin(), std::upper_bound(super::sparsity_.cbegin(), super::sparsity_.cend(), super::rank_));
            if(S != 'S') {
                super::offsets_.reserve(super::sparsity_.size() + 1);
                for(unsigned short i = 0; i < stop; ++i) {
                    super::offsets_.emplace(super::sparsity_[i], accumulate);
                    accumulate += super::n_ * (U ? super::local_ : info[i]);
                }
            }
            else
                super::offsets_.reserve(super::sparsity_.size() + 1 - stop);
            super::offsets_.emplace(super::rank_, accumulate);
            accumulate += super::n_ * super::local_;
            for(unsigned short i = stop; i < super::sparsity_.size(); ++i) {
                super::offsets_.emplace(super::sparsity_[i], accumulate);
                accumulate += super::n_ * (U ? super::local_ : info[i]);
            }

            work = new K[accumulate]();

            for(unsigned short i = 0; i < super::map_.size(); ++i) {
                if(i < super::signed_ || S != 'S') {
                    accumulate = super::local_;
                    for(unsigned short k = 0; k < super::local_; ++k)
                        for(unsigned int j = 0; j < super::map_[i].second.size(); ++j)
                            work[super::offsets_[super::rank_] + super::map_[i].second[j] + k * super::n_] = in[i][k * super::map_[i].second.size() + j] = tmp[super::map_[i].second[j] + k * super::n_];
                }
                else {
                    accumulate = 0;
                    for(unsigned short k = 0; k < super::local_; ++k)
                        for(unsigned int j = 0; j < super::map_[i].second.size(); ++j)
                            work[super::offsets_[super::rank_] + super::map_[i].second[j] + k * super::n_] = tmp[super::map_[i].second[j] + k * super::n_];
                }
                for(unsigned short l = S != 'S' ? 0 : std::min(i, super::signed_); l < super::map_.size(); ++l) {
                    for(unsigned short k = 0; k < (U ? super::local_ : infoNeighbor[l]); ++k)
                        for(unsigned int j = 0; j < super::map_[i].second.size(); ++j) {
                            if(S != 'S' || !(l < std::max(i, super::signed_)))
                                work[super::offsets_[super::map_[l].first] + super::map_[i].second[j] + k * super::n_] = in[i][(accumulate + k) * super::map_[i].second.size() + j] = tmp[super::map_[i].second[j] + (offset[l + 1] + k) * super::n_];
                            else {
                                if(i < super::signed_)
                                    in[i][(accumulate + k) * super::map_[i].second.size() + j] = tmp[super::map_[i].second[j] + (offset[l + 1] + k) * super::n_];
                                else
                                    work[super::offsets_[super::map_[l].first] + super::map_[i].second[j] + k * super::n_] = tmp[super::map_[i].second[j] + (offset[l + 1] + k) * super::n_];
                            }
                        }
                    if(S != 'S' || !(l < i) || i < super::signed_)
                        accumulate += U ? super::local_ : infoNeighbor[l];
                }
                if(U || infoNeighbor[i])
                    MPI_Isend(in[i], super::map_[i].second.size() * accumulate, Wrapper<K>::mpi_type(), super::map_[i].first, 2, super::p_.getCommunicator(), rq++);
            }
            delete [] tmp;
            delete [] offset;
            if(!U)
                delete [] infoNeighbor;
        }
        template<char S, bool U>
        void assembleForMain(K* C, const K* in, const int& coefficients, unsigned short index, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            applyFromNeighbor<S, U>(in, index, arrayC, infoNeighbor);
            if(++super::consolidate_ == super::map_.size()) {
                const underlying_type<K>* const m = super::p_.getScaling();
                for(unsigned short j = 0; j < coefficients + (S == 'S') * super::local_; ++j)
                    Wrapper<K>::diag(super::n_, m, arrayC + j * super::n_);
                if(S != 'S')
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &coefficients, &(super::local_), &(super::n_), &(Wrapper<K>::d__1), arrayC, &(super::n_), *super::deflation_, super::p_.getLDR(), &(Wrapper<K>::d__0), C, &coefficients);
                else
                    for(unsigned short j = 0; j < super::local_; ++j) {
                        int local = coefficients + super::local_ - j;
                        Blas<K>::gemv(&(Wrapper<K>::transc), &(super::n_), &local, &(Wrapper<K>::d__1), arrayC + super::n_ * j, &(super::n_), super::deflation_[j], &i__1, &(Wrapper<K>::d__0), C - (j * (j - 1)) / 2 + j * (coefficients + super::local_), &i__1);
                    }
            }
        }
        template<char S, char N, bool U>
        void applyFromNeighborMain(const K* in, unsigned short index, int* I, int* J, K* C, int coefficients, unsigned int offsetI, unsigned int* offsetJ, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            assembleForMain<S, U>(C, in, coefficients, index, arrayC, infoNeighbor);
            super::template assembleOperator<S, N, U>(I, J, coefficients, offsetI, offsetJ, infoNeighbor);
        }
};
#endif // HPDDM_BDD
} // HPDDM
#endif // HPDDM_OPERATOR_HPP_
