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

#ifndef HPDDM_COARSE_OPERATOR_IMPL_HPP_
#define HPDDM_COARSE_OPERATOR_IMPL_HPP_

#include "HPDDM_coarse_operator.hpp"

namespace HPDDM {
HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K)
template<bool exclude, class Operator>
inline void CoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>::constructionCommunicator(Operator&& v, const MPI_Comm& comm) {
    MPI_Comm_size(comm, &sizeWorld_);
    MPI_Comm_rank(comm, &rankWorld_);
#if !HPDDM_PETSC
    Option& opt = *Option::get();
#if !defined(DSUITESPARSE) && !defined(DLAPACK)
    unsigned short p = opt.val<unsigned short>("p", 1);
    if(p > sizeWorld_ / 2 && sizeWorld_ > 1) {
        p = opt["p"] = sizeWorld_ / 2;
        if(rankWorld_ == 0)
            std::cout << "WARNING -- the number of main processes was set to a value greater than MPI_Comm_size / 2, the value has been reset to " << p << std::endl;
    }
#else
    const unsigned short p = opt["p"] = 1;
#endif
    ignore(v);
#else
    unsigned short p;
    {
        PetscInt n = 1;
        PetscOptionsGetInt(nullptr, v.prefix_.c_str(), "-p", &n, nullptr);
        p = n;
    }
#endif
    DMatrix::ldistribution_ = new int[p]();
    if(p == 1) {
        MPI_Comm_dup(comm, &scatterComm_);
        gatherComm_ = scatterComm_;
        if(rankWorld_)
            DMatrix::communicator_ = MPI_COMM_NULL;
        else
            DMatrix::communicator_ = MPI_COMM_SELF;
        DMatrix::rank_ = 0;
    }
    else {
        MPI_Group main, split;
        MPI_Group world;
        MPI_Comm_group(comm, &world);
        int* ps;
        unsigned int tmp;
#if !HPDDM_PETSC
        const char T = opt.val<char>("topology", 0);
#else
        constexpr char T = 0;
#endif
        if(T == 2) {
            // Here, it is assumed that all subdomains have the same number of coarse degrees of freedom as the rank 0 ! (only true when the distribution is uniform)
            float area = sizeWorld_ *sizeWorld_ / (2.0 * p);
            *DMatrix::ldistribution_ = 0;
            for(unsigned short i = 1; i < p; ++i)
                DMatrix::ldistribution_[i] = static_cast<int>(sizeWorld_ - std::sqrt(std::max(sizeWorld_ * sizeWorld_ - 2 * sizeWorld_ * DMatrix::ldistribution_[i - 1] - 2 * area + DMatrix::ldistribution_[i - 1] * DMatrix::ldistribution_[i - 1], 1.0f)) + 0.5);
            int* idx = std::upper_bound(DMatrix::ldistribution_, DMatrix::ldistribution_ + p, rankWorld_);
            unsigned short i = idx - DMatrix::ldistribution_;
            tmp = (i == p) ? sizeWorld_ - DMatrix::ldistribution_[i - 1] : DMatrix::ldistribution_[i] - DMatrix::ldistribution_[i - 1];
            ps = new int[tmp];
            for(unsigned int j = 0; j < tmp; ++j)
                ps[j] = DMatrix::ldistribution_[i - 1] + j;
        }
#ifndef HPDDM_CONTIGUOUS
        else if(T == 1) {
            if(rankWorld_ == p - 1 || rankWorld_ > p - 1 + (p - 1) * ((sizeWorld_ - p) / p))
                tmp = sizeWorld_ - (p - 1) * (sizeWorld_ / p);
            else
                tmp = sizeWorld_ / p;
            ps = new int[tmp];
            if(rankWorld_ < p)
                ps[0] = rankWorld_;
            else {
                if(tmp == sizeWorld_ / p)
                    ps[0] = (rankWorld_ - p) / ((sizeWorld_ - p) / p);
                else
                    ps[0] = p - 1;
            }
            unsigned int offset = ps[0] * (sizeWorld_ / p - 1) + p - 1;
            std::iota(ps + 1, ps + tmp, offset + 1);
            std::iota(DMatrix::ldistribution_, DMatrix::ldistribution_ + p, 0);
        }
#endif
        else {
#if !HPDDM_PETSC
            if(T != 0)
                opt["topology"] = 0;
#endif
            if(rankWorld_ < (p - 1) * (sizeWorld_ / p))
                tmp = sizeWorld_ / p;
            else
                tmp = sizeWorld_ - (p - 1) * (sizeWorld_ / p);
            ps = new int[tmp];
            unsigned int offset;
            if(tmp != sizeWorld_ / p)
                offset = sizeWorld_ - tmp;
            else
                offset = (sizeWorld_ / p) * (rankWorld_ / (sizeWorld_ / p));
            std::iota(ps, ps + tmp, offset);
            for(unsigned short i = 0; i < p; ++i)
                DMatrix::ldistribution_[i] = i * (sizeWorld_ / p);
        }
        MPI_Group_incl(world, p, DMatrix::ldistribution_, &main);
        MPI_Group_incl(world, tmp, ps, &split);
        delete [] ps;

        MPI_Comm_create(comm, main, &(DMatrix::communicator_));
        if(DMatrix::communicator_ != MPI_COMM_NULL)
            MPI_Comm_rank(DMatrix::communicator_, &(DMatrix::rank_));
        MPI_Comm_create(comm, split, &scatterComm_);

        MPI_Group_free(&main);
        MPI_Group_free(&split);

        if(!exclude)
            MPI_Comm_dup(comm, &gatherComm_);
        else {
            MPI_Group global;
            MPI_Group_excl(world, p - 1, DMatrix::ldistribution_ + 1, &global);
            MPI_Comm_create(comm, global, &gatherComm_);
            MPI_Group_free(&global);
        }
        MPI_Group_free(&world);
    }
}

HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K)
template<bool U, typename DMatrix::Distribution D, bool excluded>
inline void CoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>::constructionCollective(const unsigned short* info, unsigned short p, const unsigned short* infoSplit) {
    if(!U) {
        if(excluded)
            sizeWorld_ -= p;
        DMatrix::gatherCounts_ = new int[2 * sizeWorld_];
        DMatrix::displs_ = DMatrix::gatherCounts_ + sizeWorld_;

        DMatrix::gatherCounts_[0] = info[0];
        DMatrix::displs_[0] = 0;
        for(unsigned int i = 1, j = 1; j < sizeWorld_; ++i)
            if(!excluded || info[i])
                DMatrix::gatherCounts_[j++] = info[i];
        std::partial_sum(DMatrix::gatherCounts_, DMatrix::gatherCounts_ + sizeWorld_ - 1, DMatrix::displs_ + 1);
        if(excluded)
            sizeWorld_ += p;
        if(D == DMatrix::DISTRIBUTED_SOL) {
            DMatrix::gatherSplitCounts_ = new int[2 * sizeSplit_];
            DMatrix::displsSplit_ = DMatrix::gatherSplitCounts_ + sizeSplit_;
            std::copy_n(infoSplit, sizeSplit_, DMatrix::gatherSplitCounts_);
            DMatrix::displsSplit_[0] = 0;
            std::partial_sum(DMatrix::gatherSplitCounts_, DMatrix::gatherSplitCounts_ + sizeSplit_ - 1, DMatrix::displsSplit_ + 1);
        }
    }
    else {
        DMatrix::gatherCounts_ = new int[1];
        *DMatrix::gatherCounts_ = local_;
    }
}

HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K)
template<char T, bool U, bool excluded>
inline void CoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>::constructionMap(unsigned short p, const unsigned short* info) {
    if(T == 0) {
        if(!U) {
            unsigned int accumulate = 0;
            for(unsigned short i = 0; i < p - 1; accumulate += DMatrix::ldistribution_[i++])
                DMatrix::ldistribution_[i] = std::accumulate(info + i * (sizeWorld_ / p), info + (i + 1) * (sizeWorld_ / p), 0);
            DMatrix::ldistribution_[p - 1] = DMatrix::n_ - accumulate;
        }
        else {
            if(p == 1)
                *DMatrix::ldistribution_ = DMatrix::n_;
            else {
                std::fill_n(DMatrix::ldistribution_, p - 1, local_ * (sizeWorld_ / p - excluded));
                DMatrix::ldistribution_[p - 1] = DMatrix::n_ - local_ * (sizeWorld_ / p - excluded) * (p - 1);
            }
        }
    }
#ifndef HPDDM_CONTIGUOUS
    else if(T == 1) {
        DMatrix::idistribution_ = new int[DMatrix::n_];
        unsigned int j = 0;
        if(!excluded)
            for(unsigned int i = 0; i < p * (sizeWorld_ / p); ++i) {
                unsigned int offset;
                if(i % (sizeWorld_ / p) == 0) {
                    j = i / (sizeWorld_ / p);
                    offset = U ? (sizeWorld_ / p) * local_ * j : (std::accumulate(info, info + j, 0) + std::accumulate(info + p, info + p + j * (sizeWorld_ / p - 1), 0));
                }
                else {
                    j = p - 1 + i - i / (sizeWorld_ / p);
                    offset  = U ? local_ * (1 + i  / (sizeWorld_ / p)) : std::accumulate(info, info + 1 + i / (sizeWorld_ / p), 0);
                    offset += U ? (j - p) * local_ : std::accumulate(info + p, info + j, 0);
                }
                std::iota(DMatrix::idistribution_ + offset, DMatrix::idistribution_ + offset + (U ? local_ : info[j]), U ? local_ * j : std::accumulate(info, info + j, 0));
                if(i % (sizeWorld_ / p) != 0)
                    j = offset + (U ? local_ : info[j]);
            }
        std::iota(DMatrix::idistribution_ + j, DMatrix::idistribution_ + DMatrix::n_, j);
        if(!U) {
            unsigned int accumulate = 0;
            for(unsigned short i = 0; i < p - 1; accumulate += DMatrix::ldistribution_[i++])
                DMatrix::ldistribution_[i] = std::accumulate(info + p + i * (sizeWorld_ / p - 1), info + p + (i + 1) * (sizeWorld_ / p - 1), info[i]);
            DMatrix::ldistribution_[p - 1] = DMatrix::n_ - accumulate;
        }
        else {
            std::fill_n(DMatrix::ldistribution_, p - 1, local_ * (sizeWorld_ / p - excluded));
            DMatrix::ldistribution_[p - 1] = DMatrix::n_ - local_ * (sizeWorld_ / p - excluded) * (p - 1);
        }
    }
#endif
    else if(T == 2) {
        if(!U) {
            unsigned int accumulate = 0;
            for(unsigned short i = 0; i < p - 1; accumulate += DMatrix::ldistribution_[i++])
                DMatrix::ldistribution_[i] = std::accumulate(info + DMatrix::ldistribution_[i], info + DMatrix::ldistribution_[i + 1], 0);
            DMatrix::ldistribution_[p - 1] = DMatrix::n_ - accumulate;
        }
        else {
            for(unsigned short i = 0; i < p - 1; ++i)
                DMatrix::ldistribution_[i] = (DMatrix::ldistribution_[i + 1] - DMatrix::ldistribution_[i] - excluded) * local_;
            DMatrix::ldistribution_[p - 1] = DMatrix::n_ - (DMatrix::ldistribution_[p - 1] - (excluded ? p - 1 : 0)) * local_;
        }
    }
}

HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K)
template<unsigned short U, unsigned short excluded, class Operator>
inline typename CoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>::return_type CoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>::construction(Operator&& v, const MPI_Comm& comm) {
#if HPDDM_PETSC
    PetscFunctionBeginUser;
#endif
    static_assert(super::numbering_ == 'C' || super::numbering_ == 'F', "Unknown numbering");
    static_assert(Operator::pattern_ == 's' || Operator::pattern_ == 'c' || Operator::pattern_ == 'u', "Unknown pattern");
    constructionCommunicator<excluded != 0>(v, comm);
    if(excluded > 0 && DMatrix::communicator_ != MPI_COMM_NULL) {
        int result;
        MPI_Comm_compare(v.p_.getCommunicator(), DMatrix::communicator_, &result);
        if(result != MPI_CONGRUENT)
            std::cerr << "The communicators for the coarse operator don't match those of the domain decomposition" << std::endl;
    }
    if(Operator::pattern_ == 'c')
        v.adjustConnectivity(scatterComm_);
    if(U == 2 && local_ == 0)
        offset_ = true;
    MPI_Comm_size(scatterComm_, &sizeSplit_);
#if !HPDDM_PETSC
    switch(Option::get()->val<char>("topology", 0)) {
#ifndef HPDDM_CONTIGUOUS
        case  1: return constructionMatrix<1, U, excluded, Operator>(v);
#endif
        case  2: return constructionMatrix<2, U, excluded, Operator>(v);
        default: return constructionMatrix<0, U, excluded, Operator>(v);
    }
#else
    const char *deft = MATSBAIJ;
    char type[256];
    char S;
    PetscBool flg;
    PetscErrorCode ierr;
    PetscOptionsBegin(v.p_.getCommunicator(), v.prefix_.c_str(), "", "");
    PetscCall(PetscOptionsFList("-mat_type", "Matrix type", "MatSetType", MatList, deft, type, 256, &flg));
    if(!flg)
        S = 'S';
    else {
        PetscCall(PetscStrcmp(type, MATSBAIJ, &flg));
        S = (flg ? 'S' : 'G');
    }
    PetscOptionsEnd();
    if(
#if !defined(PETSC_USE_REAL___FP16)
       !std::is_same<PetscScalar, PetscComplex>::value &&
#endif
                                                          S == 'S')
        ierr = constructionMatrix<'S', U, excluded, Operator>(v);
    else
        ierr = constructionMatrix<'G', U, excluded, Operator>(v);
    PetscFunctionReturn(ierr);
#endif
}

HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K)
template<char
#if !HPDDM_PETSC
              T
#else
              S
#endif
               , unsigned short U, unsigned short excluded, class Operator>
inline typename CoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>::return_type CoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>::constructionMatrix(typename std::enable_if<Operator::pattern_ != 'u', Operator>::type& v) {
#if HPDDM_PETSC
    PetscFunctionBeginUser;
#endif
    unsigned short* const info = new unsigned short[(U != 1 ? 3 : 1) + v.getConnectivity()]();
    const std::vector<unsigned short>& sparsity = v.getPattern();
    info[0] = sparsity.size(); // number of intersections
    int rank;
    MPI_Comm_rank(v.p_.getCommunicator(), &rank);
    const unsigned short first = (S == 'S' ? std::distance(sparsity.cbegin(), std::upper_bound(sparsity.cbegin(), sparsity.cend(), rank)) : 0);
    int rankSplit;
    MPI_Comm_rank(scatterComm_, &rankSplit);
    unsigned short* infoNeighbor;

    unsigned int size = 0;
#if !HPDDM_PETSC
    typedef int integer_type;
#else
    typedef PetscInt integer_type;
#endif
    integer_type* I = nullptr;
    integer_type* J = nullptr;
    K*            C = nullptr;

#if !HPDDM_PETSC
    const Option& opt = *Option::get();
    const unsigned short p = opt.val<unsigned short>("p", 1);
#else
    constexpr unsigned short T = 0;
    unsigned short p;
    const bool coarse = (v.prefix_.substr(v.prefix_.size() - 7).compare("coarse_") == 0);
    {
        PetscInt n = 1;
        PetscOptionsGetInt(nullptr, v.prefix_.c_str(), "-p", &n, nullptr);
        p = n;
    }
#if HPDDM_PETSC && defined(PETSC_HAVE_MUMPS)
    MPI_Comm extended = MPI_COMM_NULL;
    if(Operator::factorize_ && coarse) {
        PetscInt n = 1;
        PetscOptionsGetInt(nullptr, v.prefix_.c_str(), "-mat_mumps_use_omp_threads", &n, nullptr);
        if(n > 1) {
            int* group = new int[n * p];
            for(unsigned short i = 0; i < p; ++i) std::iota(group + n * i, group + n * (i + 1), DMatrix::ldistribution_[i]);
            MPI_Group world, main;
            MPI_Comm_group(v.p_.getCommunicator(), &world);
            MPI_Group_incl(world, n * p, group, &main);
            delete [] group;
            MPI_Comm_create(v.p_.getCommunicator(), main, &extended);
            MPI_Group_free(&world);
            MPI_Group_free(&main);
        }
    }
#endif
#endif
    constexpr bool blocked =
#if defined(DMKL_PARDISO) || defined(DELEMENTAL) || HPDDM_INEXACT_COARSE_OPERATOR
                             (U == 1 && Operator::pattern_ == 's');
#else
                             false;
#endif
#if !HPDDM_PETSC
    unsigned short treeDimension = opt.val<unsigned short>("assembly_hierarchy")
#else
    unsigned short treeDimension = 0
#endif
                                                                                 , currentHeight = 0;
    if(treeDimension <= 1 || treeDimension >= sizeSplit_)
        treeDimension = 0;
    unsigned short treeHeight = treeDimension ? std::ceil(std::log(sizeSplit_) / std::log(treeDimension)) : 0;
    std::vector<std::array<int, 3>>* msg = nullptr;
    if(rankSplit && treeDimension) {
        msg = new std::vector<std::array<int, 3>>();
        msg->reserve(treeHeight);
        int accumulate = 0, size;
        MPI_Comm_size(v.p_.getCommunicator(), &size);
        int full = v.max_;
        if(S != 'S')
            v.max_ = ((v.max_ & 4095) + 1) * pow(v.max_ >> 12, 2);
        for(unsigned short i = rankSplit; (i % treeDimension == 0) && currentHeight < treeHeight; i /= treeDimension) {
            const unsigned short bound = std::min(treeDimension, static_cast<unsigned short>(1 + ((sizeSplit_ - rankSplit - 1) / pow(treeDimension, currentHeight)))) - 1;
            if(S == 'S')
                v.max_ = std::min(size - (rank + pow(treeDimension, currentHeight)), full & 4095) * pow(full >> 12, 2);
            for(unsigned short k = 0; k < bound; ++k) {
                msg->emplace_back(std::array<int, 3>({{ static_cast<int>(std::min(pow(treeDimension, currentHeight), static_cast<unsigned short>(sizeSplit_ - (rankSplit + pow(treeDimension, currentHeight) * (k + 1)))) * v.max_ + (S == 'S' ? (!blocked ? ((full >> 12) * ((full >> 12) + 1)) / 2 : pow(full >> 12, 2)) : 0)), rankSplit + pow(treeDimension, currentHeight) * (k + 1), accumulate }}));
                accumulate += msg->back()[0];
            }
            ++currentHeight;
        }
    }
    if(U != 1) {
        infoNeighbor = new unsigned short[info[0]];
        info[1] = (excluded == 2 ? 0 : local_); // number of eigenvalues
        std::vector<MPI_Request> rqInfo;
        rqInfo.reserve(2 * info[0]);
        MPI_Request rq;
        if(excluded == 0) {
            if(T != 2) {
                for(unsigned short i = 0; i < info[0]; ++i)
                    if(!(T == 1 && sparsity[i] < p) &&
                       !(T == 0 && (sparsity[i] % (sizeWorld_ / p) == 0) && sparsity[i] < p * (sizeWorld_ / p))) {
                        MPI_Isend(info + 1, 1, MPI_UNSIGNED_SHORT, sparsity[i], 1, v.p_.getCommunicator(), &rq);
                        rqInfo.emplace_back(rq);
                    }
            }
            else {
                for(unsigned short i = 0; i < info[0]; ++i)
                    if(!std::binary_search(DMatrix::ldistribution_, DMatrix::ldistribution_ + p, sparsity[i])) {
                        MPI_Isend(info + 1, 1, MPI_UNSIGNED_SHORT, sparsity[i], 1, v.p_.getCommunicator(), &rq);
                        rqInfo.emplace_back(rq);
                    }
            }
        }
        else if(excluded < 2)
            for(unsigned short i = 0; i < info[0]; ++i) {
                MPI_Isend(info + 1, 1, MPI_UNSIGNED_SHORT, sparsity[i], 1, v.p_.getCommunicator(), &rq);
                rqInfo.emplace_back(rq);
            }
        if(rankSplit) {
            for(unsigned short i = 0; i < info[0]; ++i) {
                MPI_Irecv(infoNeighbor + i, 1, MPI_UNSIGNED_SHORT, sparsity[i], 1, v.p_.getCommunicator(), &rq);
                rqInfo.emplace_back(rq);
            }
            size = (S != 'S' ? local_ : 0);
            for(unsigned short i = 0; i < info[0]; ++i) {
                int index;
                MPI_Waitany(info[0], &rqInfo.back() - info[0] + 1, &index, MPI_STATUS_IGNORE);
                if(!(S == 'S' && sparsity[index] < rank))
                    size += infoNeighbor[index];
            }
            rqInfo.resize(rqInfo.size() - info[0]);
            info[2] = size;
            size *= local_;
            if(S == 'S') {
                if(Operator::factorize_)
                    info[0] -= first;
                size += local_ * (local_ + 1) / 2;
            }
        }
        if(local_ && (rankSplit || !Operator::factorize_)) {
            if(excluded == 0)
                std::copy_n(sparsity.cbegin() + (Operator::factorize_ ? first : 0), info[0], info + (U != 1 ? 3 : 1));
            else {
                if(T != 1) {
                    for(unsigned short i = 0; i < info[0]; ++i) {
                        info[(U != 1 ? 3 : 1) + i] = sparsity[i + (Operator::factorize_ ? first : 0)] + 1;
                        for(unsigned short j = 0; j < p - 1 && info[(U != 1 ? 3 : 1) + i] >= (T == 0 ? (sizeWorld_ / p) * (j + 1) : DMatrix::ldistribution_[j + 1]); ++j)
                            ++info[(U != 1 ? 3 : 1) + i];
                    }
                }
                else {
                    for(unsigned short i = 0; i < info[0]; ++i)
                        info[(U != 1 ? 3 : 1) + i] = p + sparsity[i + (Operator::factorize_ ? first : 0)];
                }
            }
        }
        MPI_Waitall(rqInfo.size(), rqInfo.data(), MPI_STATUSES_IGNORE);
    }
    else {
        infoNeighbor = nullptr;
        if(rankSplit) {
            if(S == 'S') {
                info[0] -= first;
                size = local_ * local_ * info[0] + (!blocked ? local_ * (local_ + 1) / 2 : local_ * local_);
            }
            else
                size = local_ * local_ * (1 + info[0]);
            if(Operator::factorize_)
                std::copy_n(sparsity.cbegin() + first, info[0], info + (U != 1 ? 3 : 1));
        }
        if(!Operator::factorize_) {
            if(S == 'S' && rankSplit)
                info[0] += first;
            std::copy_n(sparsity.cbegin(), info[0], info + (U != 1 ? 3 : 1));
        }
    }
    unsigned short** infoSplit = nullptr;
    unsigned int*    offsetIdx = nullptr;
    unsigned short*  infoWorld = nullptr;
#if HPDDM_INEXACT_COARSE_OPERATOR
    unsigned short*  neighbors;
#endif
#ifdef HPDDM_CSR_CO
    unsigned int nrow = 0;
    int* loc2glob = nullptr;
#endif
    if(rankSplit) {
        MPI_Gather(info, (U != 1 ? 3 : 1) + v.getConnectivity(), MPI_UNSIGNED_SHORT, NULL, 0, MPI_DATATYPE_NULL, 0, scatterComm_);
        if(!Operator::factorize_) {
            v.template setPattern<S, U == 1>(DMatrix::ldistribution_, p, sizeSplit_);
            if(S == 'S') {
                info[0] -= first;
                if(U)
                    std::copy_n(sparsity.cbegin() + first, info[0], info + (U != 1 ? 3 : 1));
                else {
                    for(unsigned short i = 0; i < info[0]; ++i)
                        info[(U != 1 ? 3 : 1) + i] = info[(U != 1 ? 3 : 1) + first + i];
                }
            }
        }
    }
    else {
        size = 0;
        infoSplit = new unsigned short*[sizeSplit_];
        *infoSplit = new unsigned short[sizeSplit_ * ((U != 1 ? 3 : 1) + v.getConnectivity()) + (U != 1) * sizeWorld_];
        MPI_Gather(info, (U != 1 ? 3 : 1) + v.getConnectivity(), MPI_UNSIGNED_SHORT, *infoSplit, (U != 1 ? 3 : 1) + v.getConnectivity(), MPI_UNSIGNED_SHORT, 0, scatterComm_);
        for(unsigned int i = 1; i < sizeSplit_; ++i)
            infoSplit[i] = *infoSplit + i * ((U != 1 ? 3 : 1) + v.getConnectivity());
        if(S == 'S' && Operator::pattern_ == 's' && Operator::factorize_)
            **infoSplit -= first;
        offsetIdx = new unsigned int[std::max(sizeSplit_ - 1, 2 * p)];
        if(U != 1) {
            infoWorld = *infoSplit + sizeSplit_ * (3 + v.getConnectivity());
            int* recvcounts = reinterpret_cast<int*>(offsetIdx);
            int* displs = recvcounts + p;
            displs[0] = 0;
            if(T == 2) {
                std::adjacent_difference(DMatrix::ldistribution_ + 1, DMatrix::ldistribution_ + p, recvcounts);
                recvcounts[p - 1] = sizeWorld_ - DMatrix::ldistribution_[p - 1];
            }
            else {
                std::fill_n(recvcounts, p - 1, sizeWorld_ / p);
                recvcounts[p - 1] = sizeWorld_ - (p - 1) * (sizeWorld_ / p);
            }
            std::partial_sum(recvcounts, recvcounts + p - 1, displs + 1);
            for(unsigned int i = 0; i < sizeSplit_; ++i)
                infoWorld[displs[DMatrix::rank_] + i] = infoSplit[i][1];
#ifdef HPDDM_CSR_CO
            nrow = std::accumulate(infoWorld + displs[DMatrix::rank_], infoWorld + displs[DMatrix::rank_] + sizeSplit_, 0);
#endif
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, infoWorld, recvcounts, displs, MPI_UNSIGNED_SHORT, DMatrix::communicator_);
            if(T == 1) {
                unsigned int i = (p - 1) * (sizeWorld_ / p);
                for(unsigned short k = p - 1, j = 1; k-- > 0; i -= sizeWorld_ / p, ++j) {
                    recvcounts[k] = infoWorld[i];
                    std::copy_backward(infoWorld + k * (sizeWorld_ / p), infoWorld + (k + 1) * (sizeWorld_ / p), infoWorld + (k + 1) * (sizeWorld_ / p) + j);
                }
                std::copy_n(recvcounts, p - 1, infoWorld + 1);
            }
            v.max_ = std::accumulate(infoWorld, infoWorld + rankWorld_, 0);
            DMatrix::n_ = std::accumulate(infoWorld + rankWorld_, infoWorld + sizeWorld_, v.max_);
            if(super::numbering_ == 'F')
                ++v.max_;
            unsigned short tmp = 0;
            for(unsigned short i = 0; i < info[0]; ++i) {
                infoNeighbor[i] = infoWorld[sparsity[i]];
                if(!(S == 'S' && i < first))
                    tmp += infoNeighbor[i];
            }
            for(unsigned short k = 1; k < sizeSplit_; ++k) {
                offsetIdx[k - 1] = size;
                size += infoSplit[k][2] * infoSplit[k][1] + (S == 'S' ? infoSplit[k][1] * (infoSplit[k][1] + 1) / 2 : 0);
            }
            if(excluded < 2)
                size += local_ * tmp + (S == 'S' ? local_ * (local_ + 1) / 2 : local_ * local_);
            if(S == 'S')
                info[0] -= first;
            if(!Operator::factorize_)
                v.template setPattern<S, U == 1>(DMatrix::ldistribution_, p, sizeSplit_, infoSplit, infoWorld);
        }
        else {
            if(!Operator::factorize_)
                v.template setPattern<S, U == 1>(DMatrix::ldistribution_, p, sizeSplit_, infoSplit, infoWorld);
            DMatrix::n_ = (sizeWorld_ - (excluded == 2 ? p : 0)) * local_;
            v.max_ = (rankWorld_ - (excluded == 2 ? rank : 0)) * local_ + (super::numbering_ == 'F');
#ifdef HPDDM_CSR_CO
            nrow = (sizeSplit_ - (excluded == 2)) * local_;
#endif
            if(S == 'S') {
                for(unsigned short i = 1; i < sizeSplit_; size += infoSplit[i++][0])
                    offsetIdx[i - 1] = size * local_ * local_ + (i - 1) * (!blocked ? local_ * (local_ + 1) / 2 : local_ * local_);
                info[0] -= first;
                size = (size + info[0]) * local_ * local_ + (sizeSplit_ - (excluded == 2)) * (!blocked ? local_ * (local_ + 1) / 2 : local_ * local_);
            }
            else {
                for(unsigned short i = 1; i < sizeSplit_; size += infoSplit[i++][0])
                    offsetIdx[i - 1] = (i - 1 + size) * local_ * local_;
                size = (size + info[0] + sizeSplit_ - (excluded == 2)) * local_ * local_;
            }
            if(sizeSplit_ == 1)
                offsetIdx[0] = size;
        }
#if HPDDM_INEXACT_COARSE_OPERATOR
        neighbors = new unsigned short[size / (!blocked ? 1 : local_ * local_)];
        if(T == 1)
            for(unsigned short i = 1; i < p; ++i)
                DMatrix::ldistribution_[i] = (excluded == 2 ? 0 : p) + i * ((sizeWorld_ / p) - 1);
        else if(U && excluded == 2)
            for(unsigned short i = 1; i < p; ++i)
                DMatrix::ldistribution_[i] -= i;
#endif
#ifdef HPDDM_CSR_CO
        I = new integer_type[(!blocked ? nrow + size : (nrow / local_ + size / (local_ * local_))) + 1];
        J = I + 1 + nrow / (!blocked ? 1 : local_);
        I[0] = (super::numbering_ == 'F');
#ifndef HPDDM_CONTIGUOUS
        loc2glob = new int[nrow];
#else
        loc2glob = new int[2];
#endif
#else
        I = new integer_type[2 * size];
        J = I + size;
#endif
        C = new K[!std::is_same<downscaled_type<K>, K>::value ? std::max((info[0] + 1) * local_ * local_, static_cast<int>(1 + ((size * sizeof(downscaled_type<K>) - 1) / sizeof(K)))) : size];
    }
    const vectorNeighbor& M = v.p_.getMap();

    MPI_Request* rqSend = v.p_.getRq();
    MPI_Request* rqRecv;
    MPI_Request* rqTree = treeDimension ? new MPI_Request[rankSplit ? msg->size() : (treeHeight * (treeDimension - 1))] : nullptr;

    K** sendNeighbor = v.p_.getBuffer();
    K** recvNeighbor;
    int coefficients = (U == 1 ? local_ * (info[0] + (S != 'S' || blocked)) : std::accumulate(infoNeighbor + first, infoNeighbor + sparsity.size(), S == 'S' ? 0 : local_));
    K* work = nullptr;
    if(Operator::pattern_ == 's') {
        rqRecv = (rankSplit == 0 && !treeDimension ? new MPI_Request[sizeSplit_ - 1 + info[0]] : rqSend + (S != 'S' ? info[0] : first));
        unsigned int accumulate = 0;
        for(unsigned short i = 0; i < (S != 'S' ? info[0] : first); ++i)
            if(U == 1 || infoNeighbor[i])
                accumulate += local_ * M[i].second.size();
        if(U == 1 || local_)
            for(unsigned short i = 0; i < info[0]; ++i)
                accumulate += (U == 1 ? local_ : infoNeighbor[i + first]) * M[i + first].second.size();
        if(excluded < 2 && !M.empty())
            *sendNeighbor = new K[accumulate];
        accumulate = 0;
        for(unsigned short i = 0; i < (S != 'S' ? info[0] : first); ++i) {
            sendNeighbor[i] = *sendNeighbor + accumulate;
            if(U == 1 || infoNeighbor[i])
                accumulate += local_ * M[i].second.size();
        }
        if(rankSplit)
            C = new K[treeDimension && !msg->empty() ? size + (!std::is_same<downscaled_type<K>, K>::value ? 1 + (((msg->back()[0] + msg->back()[2]) * sizeof(downscaled_type<K>) - 1) / sizeof(K)) : (msg->back()[0] + msg->back()[2])) : size];
        recvNeighbor = (U == 1 || local_ ? sendNeighbor + (S != 'S' ? info[0] : first) : nullptr);
        if(U == 1 || local_) {
            for(unsigned short i = 0; i < info[0]; ++i) {
                recvNeighbor[i] = *sendNeighbor + accumulate;
                MPI_Irecv(recvNeighbor[i], (U == 1 ? local_ : infoNeighbor[i + first]) * M[i + first].second.size(), Wrapper<K>::mpi_type(), M[i + first].first, 2, v.p_.getCommunicator(), rqRecv + i);
                accumulate += (U == 1 ? local_ : infoNeighbor[i + first]) * M[i + first].second.size();
            }
        }
        else
            std::fill_n(rqRecv, info[0], MPI_REQUEST_NULL);
        if(excluded < 2) {
            const K* const* const& EV = v.p_.getVectors();
            const int n = v.p_.getDof();
            v.initialize(n * (U == 1 || info[0] == 0 ? local_ : std::max(static_cast<unsigned short>(local_), *std::max_element(infoNeighbor + first, infoNeighbor + sparsity.size()))), work, S != 'S' ? info[0] : first);
            v.template applyToNeighbor<S, U == 1>(sendNeighbor, work, rqSend, infoNeighbor);
            if(S != 'S') {
                unsigned short before = 0;
                for(unsigned short j = 0; j < info[0] && sparsity[j] < rank; ++j)
                    before += (U == 1 ? (!blocked ? local_ : 1) : infoNeighbor[j]);
                if(local_) {
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &local_, &local_, &n, &(Wrapper<K>::d__1), work, &n, *EV, &n, &(Wrapper<K>::d__0), C + before * (!blocked ? 1 : local_ * local_), !blocked ? &coefficients : &local_);
                    Wrapper<K>::template imatcopy<super::numbering_ == 'F' && blocked ? 'C' : 'R'>(local_, local_, C + before * (!blocked ? 1 : local_ * local_), !blocked ? coefficients : local_, !blocked ? coefficients : local_);
                }
                if(rankSplit == 0) {
                    if(!blocked)
                        for(unsigned short j = 0; j < local_; ++j) {
#ifndef HPDDM_CSR_CO
                            std::fill_n(I + before + j * coefficients, local_, v.max_ + j);
#endif
                            std::iota(J + before + j * coefficients, J + before + j * coefficients + local_, v.max_);
#if HPDDM_INEXACT_COARSE_OPERATOR
                            std::fill_n(neighbors + before + j * coefficients, local_, DMatrix::rank_);
#endif
                        }
                    else {
                        J[before] = rankWorld_ - (excluded == 2 ? rank : 0) + (super::numbering_ == 'F');
#if HPDDM_INEXACT_COARSE_OPERATOR
                        neighbors[before] = DMatrix::rank_;
#endif
                    }
                }
            }
            else {
                if(blocked || (coefficients >= local_ && local_)) {
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &local_, &local_, &n, &(Wrapper<K>::d__1), *EV, &n, work, &n, &(Wrapper<K>::d__0), C, &local_);
                    if(!blocked)
                        for(unsigned short j = local_; j-- > 0; )
                            std::copy_backward(C + j * (local_ + 1), C + (j + 1) * local_, C - (j * (j + 1)) / 2 + j * coefficients + (j + 1) * local_);
                }
                else
                    for(unsigned short j = 0; j < local_; ++j) {
                        int local = local_ - j;
                        Blas<K>::gemv(&(Wrapper<K>::transc), &n, &local, &(Wrapper<K>::d__1), EV[j], &n, work + n * j, &i__1, &(Wrapper<K>::d__0), C - (j * (j - 1)) / 2 + j * (coefficients + local_), &i__1);
                    }
                if(rankSplit == 0) {
                    if(!blocked)
                        for(unsigned short j = local_; j-- > 0; ) {
#ifndef HPDDM_CSR_CO
                            std::fill_n(I + j * (coefficients + local_) - (j * (j - 1)) / 2, local_ - j, v.max_ + j);
#endif
                            std::iota(J + j * (coefficients + local_ - 1) - (j * (j - 1)) / 2 + j, J + j * (coefficients + local_ - 1) - (j * (j - 1)) / 2 + local_, v.max_ + j);
#if HPDDM_INEXACT_COARSE_OPERATOR
                            std::fill_n(neighbors + j * (coefficients + local_ - 1) - (j * (j - 1)) / 2 + j, local_ - j, DMatrix::rank_);
#endif
                        }
                    else {
                        *J = rankWorld_ - (excluded == 2 ? rank : 0) + (super::numbering_ == 'F');
#if HPDDM_INEXACT_COARSE_OPERATOR
                        *neighbors = DMatrix::rank_;
#endif
                    }
                }
            }
        }
    }
    else {
        rqRecv = (rankSplit == 0 && !treeDimension ? new MPI_Request[sizeSplit_ - 1 + M.size()] : (rqSend + M.size()));
        recvNeighbor = (U == 1 || local_) ? sendNeighbor + M.size() : nullptr;
        if(excluded < 2)
            v.template applyToNeighbor<S, U == 1>(sendNeighbor, work, rqSend, U == 1 ? nullptr : infoNeighbor, recvNeighbor, rqRecv);
        if(rankSplit)
            C = new K[treeDimension && !msg->empty() ? (size + msg->back()[0] + msg->back()[2]) : size];
    }
    std::pair<MPI_Request, const K*>* ret = nullptr;
    if(rankSplit) {
        if(treeDimension) {
            for(const std::array<int, 3>& m : *msg)
                MPI_Irecv(reinterpret_cast<downscaled_type<K>*>(C + size) + m[2], m[0], Wrapper<downscaled_type<K>>::mpi_type(), m[1], 3, scatterComm_, rqTree++);
            rqTree -= msg->size();
        }
        downscaled_type<K>* const pt = reinterpret_cast<downscaled_type<K>*>(C);
        if(U == 1 || local_) {
            if(Operator::pattern_ == 's') {
                if(info[0]) {
                    unsigned int* offsetArray = new unsigned int[info[0]];
                    if(S != 'S')
                        offsetArray[0] = M[0].first > rank ? local_ : 0;
                    else
                        offsetArray[0] = local_;
                    for(unsigned short k = 1; k < info[0]; ++k) {
                        offsetArray[k] = offsetArray[k - 1] + (U == 1 ? local_ : infoNeighbor[k - 1 + first]);
                        if(S != 'S' && sparsity[k - 1] < rank && sparsity[k] > rank)
                            offsetArray[k] += local_;
                    }
                    for(unsigned short k = 0; k < info[0]; ++k) {
                        int index;
                        MPI_Waitany(info[0], rqRecv, &index, MPI_STATUS_IGNORE);
                        v.template assembleForMain<!blocked ? S : 'B', U == 1>(C + offsetArray[index], recvNeighbor[index], coefficients + (S == 'S' && !blocked ? local_ - 1 : 0), index + first, blocked && super::numbering_ == 'F' ? C + offsetArray[index] * local_ : work, infoNeighbor + first + index);
                        if(blocked && super::numbering_ == 'C')
                            Wrapper<K>::template omatcopy<'T'>(local_, local_, work, local_, C + offsetArray[index] * local_, local_);
                    }
                    delete [] offsetArray;
                }
            }
            else {
                for(unsigned short k = 0; k < M.size(); ++k) {
                    int index;
                    MPI_Waitany(M.size(), rqRecv, &index, MPI_STATUS_IGNORE);
                    v.template assembleForMain<S, U == 1>(C, recvNeighbor[index], coefficients, index, work, infoNeighbor);
                }
            }
            if(excluded)
                ret = new std::pair<MPI_Request, const K*>(MPI_REQUEST_NULL, C);
            if(!std::is_same<downscaled_type<K>, K>::value)
                for(unsigned int i = 0; i < size; ++i)
                    pt[i] = C[i];
            if(!treeDimension) {
                if(excluded)
                    MPI_Isend(pt, size, Wrapper<downscaled_type<K>>::mpi_type(), 0, 3, scatterComm_, &ret->first);
                else
                    MPI_Send(pt, size, Wrapper<downscaled_type<K>>::mpi_type(), 0, 3, scatterComm_);
            }
        }
        if(treeDimension) {
            if(!msg->empty()) {
                for(unsigned short i = 0; i < msg->size(); ++i) {
                    MPI_Status st;
                    int idx;
                    MPI_Waitany(msg->size(), rqTree, &idx, &st);
                    MPI_Get_count(&st, Wrapper<downscaled_type<K>>::mpi_type(), &((*msg)[idx][1]));
                }
                (*msg)[0][0] = 0;
                for(unsigned short i = 0; i < msg->size(); ++i) {
                    if((*msg)[i][1])
                        std::copy_n(reinterpret_cast<downscaled_type<K>*>(C + size) + (*msg)[i][2], (*msg)[i][1], pt + size + (*msg)[i][0]);
                    if(i != msg->size() - 1)
                        (*msg)[i + 1][0] = (*msg)[i][0] + (*msg)[i][1];
                }
                size += msg->back()[0] + msg->back()[1];
            }
            delete [] rqTree;
            delete msg;
            if(size) {
                if(excluded)
                    MPI_Isend(pt, size, Wrapper<downscaled_type<K>>::mpi_type(), pow(treeDimension, currentHeight + 1) * (rankSplit / pow(treeDimension, currentHeight + 1)), 3, scatterComm_, &ret->first);
                else
                    MPI_Send(pt, size, Wrapper<downscaled_type<K>>::mpi_type(), pow(treeDimension, currentHeight + 1) * (rankSplit / pow(treeDimension, currentHeight + 1)), 3, scatterComm_);
            }
        }
        if(!excluded)
            delete [] C;
        delete [] info;
        sizeRHS_ = local_;
        if(U != 1)
            delete [] infoNeighbor;
        if(U == 0)
            DMatrix::displs_ = &rankWorld_;
        int nbRq = std::distance(v.p_.getRq(), rqSend);
        MPI_Waitall(nbRq, rqSend - nbRq, MPI_STATUSES_IGNORE);
        delete [] work;
#if HPDDM_PETSC && defined(PETSC_HAVE_MUMPS)
        if(extended != MPI_COMM_NULL) {
            PC pc;
            Mat E, A;
            PetscInt zero = 0;
            PetscCallMPI(MPI_Bcast(&rank, 1, MPI_INT, 0, extended));
            PetscCall(KSPCreate(extended, &v.level_->ksp));
            PetscCall(MatCreate(extended, &E));
            PetscCall(MatSetOptionsPrefix(E, v.prefix_.c_str()));
            PetscCall(MatSetFromOptions(E));
            PetscCall(MatSetBlockSize(E, !blocked ? 1 : local_));
            PetscCall(MatSetSizes(E, 0, 0, rank, rank));
            if(S == 'S') {
                PetscCall(MatSetType(E, MATSBAIJ));
                PetscCall(MatMPISBAIJSetPreallocationCSR(E, !blocked ? 1 : local_, &zero, nullptr, nullptr));
            }
            else {
                if(blocked && local_ > 1) {
                    PetscCall(MatSetType(E, MATBAIJ));
                    PetscCall(MatMPIBAIJSetPreallocationCSR(E, local_, &zero, nullptr, nullptr));
                }
                else {
                    PetscCall(MatSetType(E, MATAIJ));
                    PetscCall(MatMPIAIJSetPreallocationCSR(E, &zero, nullptr, nullptr));
                }
            }
            PetscCall(KSPGetOperators(v.level_->parent->levels[0]->ksp, nullptr, &A));
            PetscCall(MatPropagateSymmetryOptions(A, E));
            PetscReal chop = -1.0;
            PetscCall(PetscOptionsGetReal(nullptr, v.prefix_.c_str(), "-mat_chop", &chop, nullptr));
            if(chop >= 0.0) {
                PetscCall(MatConvert(E, MATAIJ, MAT_INPLACE_MATRIX, &E));
                PetscCall(MatChop(E, chop));
                PetscCall(MatEliminateZeros(E));
            }
            PetscCall(KSPSetOperators(v.level_->ksp, E, E));
            PetscCall(KSPSetOptionsPrefix(v.level_->ksp, v.prefix_.c_str()));
            PetscCall(KSPSetType(v.level_->ksp, KSPPREONLY));
            PetscCall(KSPGetPC(v.level_->ksp, &pc));
            if(blocked)
                PetscCall(PCSetType(pc, S == 'S' ? PCCHOLESKY : PCLU));
            PetscCall(KSPSetFromOptions(v.level_->ksp));
            PetscCall(MatDestroy(&E));
            super::s_ = v.level_;
        }
#endif
    }
    else {
        const unsigned short relative = (T == 1 ? p + rankWorld_ * ((sizeWorld_ / p) - 1) - 1 : rankWorld_);
        unsigned int* offsetPosition;
        if(excluded < 2)
            std::for_each(offsetIdx, offsetIdx + sizeSplit_ - 1, [&](unsigned int& i) { i += coefficients * local_ + (S == 'S' && !blocked) * (local_ * (local_ + 1)) / 2; });
        K* const backup = std::is_same<downscaled_type<K>, K>::value ? C : new K[offsetIdx[0]];
        if(!std::is_same<downscaled_type<K>, K>::value)
            std::copy_n(C, offsetIdx[0], backup);
        if(!treeDimension) {
            if(excluded < 2)
                treeHeight = Operator::pattern_ == 's' ? info[0] : M.size();
            else
                treeHeight = 0;
            for(unsigned short k = 1; k < sizeSplit_; ++k) {
                if(U != 1) {
                    if(infoSplit[k][1])
                        MPI_Irecv(reinterpret_cast<downscaled_type<K>*>(C) + offsetIdx[k - 1], infoSplit[k][2] * infoSplit[k][1] + (S == 'S' ? infoSplit[k][1] * (infoSplit[k][1] + 1) / 2 : 0), Wrapper<downscaled_type<K>>::mpi_type(), k, 3, scatterComm_, rqRecv + treeHeight + k - 1);
                    else
                        rqRecv[treeHeight + k - 1] = MPI_REQUEST_NULL;
                }
                else
                    MPI_Irecv(reinterpret_cast<downscaled_type<K>*>(C) + offsetIdx[k - 1], local_ * local_ * infoSplit[k][0] + (S == 'S' && !blocked ? local_ * (local_ + 1) / 2 : local_ * local_), Wrapper<downscaled_type<K>>::mpi_type(), k, 3, scatterComm_, rqRecv + treeHeight + k - 1);
            }
        }
        else {
            std::fill_n(rqTree, treeHeight * (treeDimension - 1), MPI_REQUEST_NULL);
            for(unsigned short i = 0; i < treeHeight; ++i) {
                const unsigned short leaf = pow(treeDimension, i);
                const unsigned short bound = std::min(treeDimension, static_cast<unsigned short>(1 + ((sizeSplit_ - 1) / leaf))) - 1;
                for(unsigned short k = 0; k < bound; ++k) {
                    const unsigned short nextLeaf = std::min(leaf * (k + 1) * treeDimension, sizeSplit_);
                    int nnz = 0;
                    if(U != 1) {
                        for(unsigned short j = leaf * (k + 1); j < nextLeaf; ++j)
                            nnz += infoSplit[j][2] * infoSplit[j][1] + (S == 'S' ? infoSplit[j][1] * (infoSplit[j][1] + 1) / 2 : 0);
                        if(nnz)
                            MPI_Irecv(reinterpret_cast<downscaled_type<K>*>(C) + offsetIdx[leaf * (k + 1) - 1], nnz, Wrapper<downscaled_type<K>>::mpi_type(), leaf * (k + 1), 3, scatterComm_, rqTree + i * (treeDimension - 1) + k);
                    }
                    else {
                        for(unsigned short j = leaf * (k + 1); j < nextLeaf; ++j)
                            nnz += infoSplit[j][0];
                        MPI_Irecv(reinterpret_cast<downscaled_type<K>*>(C) + offsetIdx[leaf * (k + 1) - 1], local_ * local_ * nnz + (S == 'S' && !blocked ? local_ * (local_ + 1) / 2 : local_ * local_) * (nextLeaf - leaf), Wrapper<downscaled_type<K>>::mpi_type(), leaf * (k + 1), 3, scatterComm_, rqTree + i * (treeDimension - 1) + k);
                    }
                }
            }
        }
        if(U != 1) {
            offsetPosition = new unsigned int[sizeSplit_];
            offsetPosition[0] = std::accumulate(infoWorld, infoWorld + relative, static_cast<unsigned int>(super::numbering_ == 'F'));
            if(T != 1)
                for(unsigned int k = 1; k < sizeSplit_; ++k)
                    offsetPosition[k] = offsetPosition[k - 1] + infoSplit[k - 1][1];
            else
                for(unsigned int k = 1; k < sizeSplit_; ++k)
                    offsetPosition[k] = offsetPosition[k - 1] + infoWorld[relative + k - 1];
        }
        if(blocked)
            std::for_each(offsetIdx, offsetIdx + sizeSplit_ - 1, [&](unsigned int& i) { i /= local_ * local_; });
#ifdef _OPENMP
#pragma omp parallel for shared(I, J, infoWorld, infoSplit, offsetIdx, offsetPosition) schedule(dynamic, 64)
#endif
        for(unsigned int k = 1; k < sizeSplit_; ++k) {
            if(U == 1 || infoSplit[k][1]) {
                unsigned int offsetSlave = static_cast<unsigned int>(super::numbering_ == 'F');
                if(U != 1 && infoSplit[k][0])
                    offsetSlave = std::accumulate(infoWorld, infoWorld + infoSplit[k][3], offsetSlave);
                unsigned short i = 0;
                integer_type* colIdx = J + offsetIdx[k - 1];
#if HPDDM_INEXACT_COARSE_OPERATOR
                unsigned short* nghbrs = neighbors + offsetIdx[k - 1];
#endif
                const unsigned short max = relative + k - (U == 1 && excluded == 2 ? (T == 1 ? p : 1 + rank) : 0);
                const unsigned int tmp = (U == 1 ? max * (!blocked ? local_ : 1) + (super::numbering_ == 'F') : offsetPosition[k]);
                if(S != 'S')
                    while(i < infoSplit[k][0] && infoSplit[k][(U != 1 ? 3 : 1) + i] < max) {
#if HPDDM_INEXACT_COARSE_OPERATOR
                        if(T == 1 && infoSplit[k][(U != 1 ? 3 : 1) + i] < p)
                            *nghbrs = infoSplit[k][(U != 1 ? 3 : 1) + i];
                        else
                            *nghbrs = std::distance(DMatrix::ldistribution_ + 1, std::upper_bound(DMatrix::ldistribution_ + 1, DMatrix::ldistribution_ + DMatrix::rank_ + 1, infoSplit[k][(U != 1 ? 3 : 1) + i]));
                        if(*nghbrs != DMatrix::rank_ && ((T != 1 && (i == 0 || *nghbrs != *(nghbrs - 1))) || (T == 1 && !std::binary_search(super::send_[*nghbrs].cbegin(), super::send_[*nghbrs].cend(), tmp - (super::numbering_ == 'F')))))
                            for(unsigned short row = 0; row < (U == 1 ? (!blocked ? local_ : 1) : infoSplit[k][1]); ++row)
                                super::send_[*nghbrs].emplace_back(tmp - (super::numbering_ == 'F') + row);
#endif
                        if(!blocked) {
                            if(U != 1) {
                                if(i > 0)
                                    offsetSlave = std::accumulate(infoWorld + infoSplit[k][2 + i], infoWorld + infoSplit[k][3 + i], offsetSlave);
                            }
                            else
                                offsetSlave = infoSplit[k][1 + i] * local_ + (super::numbering_ == 'F');
                            std::iota(colIdx, colIdx + (U == 1 ? local_ : infoWorld[infoSplit[k][3 + i]]), offsetSlave);
                            colIdx += (U == 1 ? local_ : infoWorld[infoSplit[k][3 + i]]);
#if HPDDM_INEXACT_COARSE_OPERATOR
                            std::fill_n(nghbrs + 1, (U == 1 ? local_ : infoWorld[infoSplit[k][3 + i]]) - 1, *nghbrs);
                            nghbrs += (U == 1 ? local_ : infoWorld[infoSplit[k][3 + i]]);
#endif
                        }
                        else {
                            *colIdx++ = infoSplit[k][1 + i] + (super::numbering_ == 'F');
#if HPDDM_INEXACT_COARSE_OPERATOR
                            ++nghbrs;
#endif
                        }
                        ++i;
                    }
                if(!blocked) {
                    std::iota(colIdx, colIdx + (U == 1 ? local_ : infoSplit[k][1]), tmp);
                    colIdx += (U == 1 ? local_ : infoSplit[k][1]);
#if HPDDM_INEXACT_COARSE_OPERATOR
                    std::fill_n(nghbrs, (U == 1 ? local_ : infoSplit[k][1]), DMatrix::rank_);
                    nghbrs += (U == 1 ? local_ : infoSplit[k][1]);
#endif
                }
                else {
                    *colIdx++ = tmp;
#if HPDDM_INEXACT_COARSE_OPERATOR
                    *nghbrs++ = DMatrix::rank_;
#endif
                }
                while(i < infoSplit[k][0]) {
#if HPDDM_INEXACT_COARSE_OPERATOR
                    if(U == 1 || infoWorld[infoSplit[k][(U != 1 ? 3 : 1) + i]]) {
                        *nghbrs = std::distance(DMatrix::ldistribution_ + 1, std::upper_bound(DMatrix::ldistribution_ + DMatrix::rank_ + 1, DMatrix::ldistribution_ + p, infoSplit[k][(U != 1 ? 3 : 1) + i]));
                        if(S != 'S' && *nghbrs != DMatrix::rank_ && ((T != 1 && (i == 0 || *nghbrs != *(nghbrs - 1))) || (T == 1 && !std::binary_search(super::send_[*nghbrs].cbegin(), super::send_[*nghbrs].cend(), tmp - (super::numbering_ == 'F')))))
                            for(unsigned short row = 0; row < (U == 1 ? (!blocked ? local_ : 1) : infoSplit[k][1]); ++row)
                                super::send_[*nghbrs].emplace_back(tmp - (super::numbering_ == 'F') + row);
                    }
#endif
                    if(!blocked) {
                        if(U != 1) {
                            if(i > 0)
                                offsetSlave = std::accumulate(infoWorld + infoSplit[k][2 + i], infoWorld + infoSplit[k][3 + i], offsetSlave);
                        }
                        else
                            offsetSlave = infoSplit[k][1 + i] * local_ + (super::numbering_ == 'F');
                        std::iota(colIdx, colIdx + (U == 1 ? local_ : infoWorld[infoSplit[k][3 + i]]), offsetSlave);
                        colIdx += (U == 1 ? local_ : infoWorld[infoSplit[k][3 + i]]);
#if HPDDM_INEXACT_COARSE_OPERATOR
                        std::fill_n(nghbrs + 1, (U == 1 ? local_ : infoWorld[infoSplit[k][3 + i]]) - 1, *nghbrs);
                        nghbrs += (U == 1 ? local_ : infoWorld[infoSplit[k][3 + i]]);
#endif
                    }
                    else {
                        *colIdx++ = infoSplit[k][1 + i] + (super::numbering_ == 'F');
#if HPDDM_INEXACT_COARSE_OPERATOR
                        ++nghbrs;
#endif
                    }
                    ++i;
                }
                unsigned int coefficientsSlave = colIdx - J - offsetIdx[k - 1];
#ifndef HPDDM_CSR_CO
                integer_type* rowIdx = I + std::distance(J, colIdx);
                std::fill(I + offsetIdx[k - 1], rowIdx, tmp);
#else
                offsetSlave = (U == 1 ? (k - (excluded == 2)) * (!blocked ? local_ : 1) : offsetPosition[k] - offsetPosition[1] + (excluded == 2 ? 0 : local_));
                I[offsetSlave + 1] = coefficientsSlave;
#ifndef HPDDM_CONTIGUOUS
                loc2glob[offsetSlave] = tmp;
#endif
#endif
                if(!blocked)
                    for(i = 1; i < (U == 1 ? local_ : infoSplit[k][1]); ++i) {
                        if(S == 'S')
                            --coefficientsSlave;
#ifndef HPDDM_CSR_CO
                        std::fill_n(rowIdx, coefficientsSlave, tmp + i);
                        rowIdx += coefficientsSlave;
#else
                        I[offsetSlave + 1 + i] = coefficientsSlave;
#ifndef HPDDM_CONTIGUOUS
                        loc2glob[offsetSlave + i] = tmp + i;
#endif
#endif
                        std::copy(colIdx - coefficientsSlave, colIdx, colIdx);
                        colIdx += coefficientsSlave;
#if HPDDM_INEXACT_COARSE_OPERATOR
                        std::copy(nghbrs - coefficientsSlave, nghbrs, nghbrs);
                        nghbrs += coefficientsSlave;
#endif
                    }
            }
        }
#ifdef HPDDM_CONTIGUOUS
        if(excluded == 2)
            loc2glob[0] = (U == 1 ? (relative + 1 - (T == 1 ? p : 1 + rank)) * (!blocked ? local_ : 1) + (super::numbering_ == 'F') : offsetPosition[1]);
        if(excluded == 2 || sizeSplit_ > 1)
            loc2glob[1] = (U == 1 ? (relative + sizeSplit_ - 1 - (U == 1 && excluded == 2 ? (T == 1 ? p : 1 + rank) : 0)) * (!blocked ? local_ : 1) + (super::numbering_ == 'F') : offsetPosition[sizeSplit_ - 1]) + (!blocked ? (U == 1 ? local_ : infoSplit[sizeSplit_ - 1][1]) - 1 : 0);
#endif
        if(std::is_same<downscaled_type<K>, K>::value)
            delete [] offsetIdx;
        if(excluded < 2) {
#ifdef HPDDM_CSR_CO
            if(!blocked) {
                I[1] = coefficients + (S == 'S' ? local_ : 0);
                for(unsigned short k = 1; k < local_; ++k) {
                    I[k + 1] = coefficients + (S == 'S' ? local_ - k : 0);
#ifndef HPDDM_CONTIGUOUS
                    loc2glob[k] = v.max_ + k;
#endif
                }
            }
            else
                I[1] = info[0] + 1;
            loc2glob[0] = ((!blocked || local_ == 1) ? v.max_ : v.max_ / local_ + (super::numbering_ == 'F'));
#ifdef HPDDM_CONTIGUOUS
            if(sizeSplit_ == 1)
                loc2glob[1] = ((!blocked || local_ == 1) ? v.max_ + local_ - 1 : v.max_ / local_ + (super::numbering_ == 'F'));
#endif
#endif
            unsigned int** offsetArray = new unsigned int*[info[0]];
            if(info[0]) {
                *offsetArray = new unsigned int[info[0] * ((Operator::pattern_ == 's') + (U != 1))];
                if(Operator::pattern_ == 's') {
                    if(S != 'S') {
                        offsetArray[0][0] = sparsity[0] > rankWorld_ ? local_ : 0;
                        if(U != 1)
                            offsetArray[0][1] = std::accumulate(infoWorld, infoWorld + sparsity[0], static_cast<unsigned int>(super::numbering_ == 'F'));
                    }
                    else if(info[0]) {
                        offsetArray[0][0] = local_;
                        if(U != 1)
                            offsetArray[0][1] = std::accumulate(infoWorld, infoWorld + sparsity[first], static_cast<unsigned int>(super::numbering_ == 'F'));
                    }
                    for(unsigned short k = 1; k < info[0]; ++k) {
                        if(U != 1) {
                            offsetArray[k] = *offsetArray + 2 * k;
                            offsetArray[k][1] = std::accumulate(infoWorld + sparsity[first + k - 1], infoWorld + sparsity[first + k], offsetArray[k - 1][1]);
                            offsetArray[k][0] = offsetArray[k - 1][0] + infoNeighbor[k - 1 + first];
                        }
                        else {
                            offsetArray[k] = *offsetArray + k;
                            offsetArray[k][0] = offsetArray[k - 1][0] + local_;
                        }
                        if(S != 'S' && sparsity[k - 1] < rankWorld_ && sparsity[k] > rankWorld_)
                            offsetArray[k][0] += local_;
                    }
                }
                else if(U != 1) {
                    if(S != 'S')
                        offsetArray[0][0] = std::accumulate(infoWorld, infoWorld + sparsity[0], static_cast<unsigned int>(super::numbering_ == 'F'));
                    else if(info[0])
                        offsetArray[0][0] = std::accumulate(infoWorld, infoWorld + sparsity[first], static_cast<unsigned int>(super::numbering_ == 'F'));
                    for(unsigned short k = 1; k < info[0]; ++k) {
                        offsetArray[k] = *offsetArray + k;
                        offsetArray[k][0] = std::accumulate(infoWorld + sparsity[first + k - 1], infoWorld + sparsity[first + k], offsetArray[k - 1][0]);
                    }
                }
            }
            if(U == 1 || local_) {
                for(unsigned int k = 0; k < (Operator::pattern_ == 's' ? info[0] : M.size()); ++k) {
                    int index;
                    MPI_Waitany(Operator::pattern_ == 's' ? info[0] : M.size(), rqRecv, &index, MPI_STATUS_IGNORE);
                    if(Operator::pattern_ == 's') {
                        const unsigned int offset = offsetArray[index][0] / (!blocked ? 1 : local_);
                        v.template applyFromNeighborMain<!blocked ? S : 'B', super::numbering_, U == 1>(recvNeighbor[index], index + first, I + offset, J + offset, backup + offsetArray[index][0] * (!blocked ? 1 : local_), coefficients + (S == 'S' && !blocked) * (local_ - 1), v.max_, U == 1 ? nullptr : (offsetArray[index] + 1), work, U == 1 ? nullptr : infoNeighbor + first + index);
#if HPDDM_INEXACT_COARSE_OPERATOR
                        if(T == 1 && M[first + index].first < p)
                            neighbors[offset] = M[first + index].first;
                        else if(blocked || offset < size)
                            neighbors[offset] = std::distance(DMatrix::ldistribution_ + 1, std::upper_bound(DMatrix::ldistribution_ + 1, DMatrix::ldistribution_ + p, M[first + index].first));
                        if(S != 'S' && (blocked || offset < size) && neighbors[offset] != DMatrix::rank_ && (super::send_[neighbors[offset]].empty() || super::send_[neighbors[offset]].back() != ((v.max_ - (super::numbering_ == 'F')) / (!blocked ? 1 : local_) + (!blocked ? local_ : 1) - 1)))
                            for(unsigned short i = 0; i < (!blocked ? local_ : 1); ++i)
                                super::send_[neighbors[offset]].emplace_back((v.max_ - (super::numbering_ == 'F')) / (!blocked ? 1 : local_) + i);
                        if(!blocked && offset < size)
                            for(unsigned short i = 0; i < local_; ++i)
                                std::fill_n(neighbors + offset + (coefficients + (S == 'S') * (local_ - 1)) * i - (S == 'S') * (i * (i - 1)) / 2, U == 1 ? local_ : infoNeighbor[first + index], neighbors[offset]);
#endif
                    }
                    else
                        v.template applyFromNeighborMain<S, super::numbering_, U == 1>(recvNeighbor[index], index, I, J, backup, coefficients, v.max_, U == 1 ? nullptr : *offsetArray, work, U == 1 ? nullptr : infoNeighbor);
                }
                downscaled_type<K>* pt = reinterpret_cast<downscaled_type<K>*>(C);
                if(!std::is_same<downscaled_type<K>, K>::value) {
                    if(blocked && sizeSplit_ > 1)
                        offsetIdx[0] *= local_ * local_;
                    for(unsigned int i = 0; i < offsetIdx[0]; ++i)
                        pt[i] = backup[i];
                    delete [] backup;
                }
            }
            if(info[0])
                delete [] *offsetArray;
            delete [] offsetArray;
        }
#ifdef HPDDM_CONTIGUOUS
        else if(sizeSplit_ == 1) {
            loc2glob[0] = 2;
            loc2glob[1] = 1;
        }
#endif
        if(!std::is_same<downscaled_type<K>, K>::value)
            delete [] offsetIdx;
        delete [] info;
        if(!treeDimension)
            MPI_Waitall(sizeSplit_ - 1, rqRecv + treeHeight, MPI_STATUSES_IGNORE);
        else {
            MPI_Waitall(treeHeight * (treeDimension - 1), rqTree, MPI_STATUSES_IGNORE);
            delete [] rqTree;
        }
        if(U != 1) {
            delete [] infoNeighbor;
            delete [] offsetPosition;
        }
        {
            int nbRq = std::distance(v.p_.getRq(), rqSend);
            MPI_Waitall(nbRq, rqSend - nbRq, MPI_STATUSES_IGNORE);
            delete [] work;
        }
        downscaled_type<K>* pt = reinterpret_cast<downscaled_type<K>*>(C);
#if !HPDDM_PETSC
        std::string filename = opt.prefix("dump_matrix", true);
        if(filename.size() > 0) {
            if(excluded == 2)
                filename += "_excluded";
            std::ofstream output { filename + "_" + S + "_" + super::numbering_ + "_" + to_string(T) + "_" + to_string(DMatrix::rank_) + ".txt" };
            output << std::scientific;
#ifndef HPDDM_CSR_CO
            for(unsigned int i = 0; i < size; ++i)
                output << std::setw(9) << I[i] + (super::numbering_ == 'C') << std::setw(9) << J[i] + (super::numbering_ == 'C') << " " << pt[i] << std::endl;
#else
            unsigned int accumulate = 0;
            for(unsigned int i = 0; i < nrow / (!blocked ? 1 : local_); ++i) {
                accumulate += I[i];
                for(unsigned int j = 0; j < I[i + 1]; ++j) {
                    output << std::setw(9) <<
#ifndef HPDDM_CONTIGUOUS
                    (loc2glob[i] - (super::numbering_ == 'F')) * (!blocked ? 1 : local_) + 1 <<
#else
                    (loc2glob[0] + i - (super::numbering_ == 'F')) * (!blocked ? 1 : local_) + 1 <<
#endif
                    std::setw(9) << (J[accumulate + j - (super::numbering_ == 'F')] - (super::numbering_ == 'F')) * (!blocked ? 1 : local_) + 1 << " ";
                    if(!blocked)
                        output << std::setw(13) << pt[accumulate + j - (super::numbering_ == 'F')] << "\n";
                    else {
                        for(unsigned short b = 0; b < local_; ++b) {
                            if(b)
                                output << "                   ";
                            for(unsigned short c = 0; c < local_; ++c) {
                                output << std::setw(13) << pt[(accumulate + j - (super::numbering_ == 'F')) * local_ * local_ + (super::numbering_ == 'C' ? b * local_ + c : b + c * local_)] << "  ";
                            }
                            output << "\n";
                        }
                    }
                }
            }
#endif
        }
#endif
#if HPDDM_INEXACT_COARSE_OPERATOR
        if(S != 'S' && (blocked || local_)) {
            int* backup = new int[!blocked ? local_ : 1];
            for(std::pair<const unsigned short, std::vector<int>>& i : super::send_) {
                if(i.second.size() > (!blocked ? local_ : 1) && *(i.second.end() - (!blocked ? local_ : 1) - 1) > *(i.second.end() - (!blocked ? local_ : 1))) {
                    std::vector<int>::iterator it = std::lower_bound(i.second.begin(), i.second.end(), *(i.second.end() - (!blocked ? local_ : 1)));
                    std::copy(i.second.end() - (!blocked ? local_ : 1), i.second.end(), backup);
                    std::copy_backward(it, i.second.end() - (!blocked ? local_ : 1), i.second.end());
                    std::copy_n(backup, !blocked ? local_ : 1, it);
                }
            }
            delete [] backup;
        }
        rank = DMatrix::n_;
#if !HPDDM_PETSC
        super::mu_ = std::min(p, opt.val<unsigned short>("aggregate_size", p));
        if(super::mu_ < p) {
            super::di_ = new int[T == 1 ? 3 : 1];
            unsigned int begin = (DMatrix::rank_ / super::mu_) * super::mu_;
            unsigned int end = std::min(p, static_cast<unsigned short>(begin + super::mu_));
#else
        if(1 < p) {
            super::di_ = new integer_type[T == 1 ? 3 : 1];
            unsigned int begin = DMatrix::rank_;
            unsigned int end = std::min(p, static_cast<unsigned short>(begin + 1));
#endif
            if(T == 1) {
                super::di_[0] = begin;
                super::di_[1] = end;
                super::di_[2] = p + begin * ((sizeWorld_ / p) - 1);
                if(end == p)
                    end = sizeWorld_;
                else
                    end = p + end * ((sizeWorld_ / p) - 1);
            }
            else {
                super::di_[0] = DMatrix::ldistribution_[begin];
                if(end == p)
                    end = sizeWorld_ - (U && excluded == 2 ? p : 0);
                else
                    end = DMatrix::ldistribution_[end];
            }
            if(!U) {
                DMatrix::n_ = std::accumulate(infoWorld + super::di_[0], infoWorld + (T == 1 ? super::di_[1] : end), 0);
                super::di_[0] = std::accumulate(infoWorld, infoWorld + super::di_[0], 0);
                if(T == 1) {
                    begin = super::di_[1];
                    super::di_[1] = super::di_[0] + DMatrix::n_;
                    DMatrix::n_ = std::accumulate(infoWorld + super::di_[2], infoWorld + end, DMatrix::n_);
                    super::di_[2] = std::accumulate(infoWorld + begin, infoWorld + super::di_[2], super::di_[1]);
                }
            }
            else {
                DMatrix::n_ = ((T == 1 ? super::di_[1] : end) - super::di_[0]) * local_;
                if(T == 1)
                    DMatrix::n_ += (end - super::di_[2]) * local_;
            }
        } // }
        super::bs_ = (!blocked ? 1 : local_);
#if !HPDDM_PETSC
        super::template numfact<T, Operator::factorize_>(nrow / (!blocked ? 1 : local_), I, loc2glob, J, pt, neighbors);
#else
        std::partial_sum(I, I + 1 + nrow / (!blocked ? 1 : local_), I);
        if(Operator::factorize_) {
            Mat E, A;
#if !defined(PETSC_HAVE_MUMPS)
            const MPI_Comm extended = MPI_COMM_NULL;
#endif
            if(extended != MPI_COMM_NULL)
                PetscCallMPI(MPI_Bcast(&rank, 1, MPI_INT, 0, extended));
            PetscCall(KSPCreate(extended != MPI_COMM_NULL ? extended : DMatrix::communicator_, &v.level_->ksp));
            PetscCall(MatCreate(extended != MPI_COMM_NULL ? extended : DMatrix::communicator_, &E));
            PetscCall(MatSetOptionsPrefix(E, v.prefix_.c_str()));
            PetscCall(MatSetFromOptions(E));
            PetscCall(MatSetBlockSize(E, !blocked ? 1 : local_));
            PetscCall(MatSetSizes(E, nrow, nrow, rank, rank));
            if(S == 'S') {
                PetscCall(MatSetType(E, MATSBAIJ));
                if(p == 1 && extended == MPI_COMM_NULL)
                    PetscCall(MatSeqSBAIJSetPreallocationCSR(E, super::bs_, I, J, pt));
                else
                    PetscCall(MatMPISBAIJSetPreallocationCSR(E, super::bs_, I, J, pt));
            }
            else {
                if(super::bs_ > 1) {
                    PetscCall(MatSetType(E, MATBAIJ));
                    if(p == 1 && extended == MPI_COMM_NULL)
                        PetscCall(MatSeqBAIJSetPreallocationCSR(E, super::bs_, I, J, pt));
                    else
                        PetscCall(MatMPIBAIJSetPreallocationCSR(E, super::bs_, I, J, pt));
                }
                else {
                    PetscCall(MatSetType(E, MATAIJ));
                    if(p == 1 && extended == MPI_COMM_NULL)
                        PetscCall(MatSeqAIJSetPreallocationCSR(E, I, J, pt));
                    else
                        PetscCall(MatMPIAIJSetPreallocationCSR(E, I, J, pt));
                }
            }
            PetscCall(KSPGetOperators(v.level_->parent->levels[0]->ksp, nullptr, &A));
            PetscCall(MatPropagateSymmetryOptions(A, E));
            PetscReal chop = -1.0;
            PetscCall(PetscOptionsGetReal(nullptr, v.prefix_.c_str(), "-mat_chop", &chop, nullptr));
            if(chop >= 0.0) {
                PetscCall(MatConvert(E, MATAIJ, MAT_INPLACE_MATRIX, &E));
                PetscCall(MatChop(E, chop));
                PetscCall(MatEliminateZeros(E));
            }
            PetscCall(KSPSetOperators(v.level_->ksp, E, E));
            PetscCall(KSPSetOptionsPrefix(v.level_->ksp, v.prefix_.c_str()));
            if(coarse) {
                PC pc;
                PetscCall(KSPSetType(v.level_->ksp, KSPPREONLY));
                PetscCall(KSPGetPC(v.level_->ksp, &pc));
                if(blocked) {
#if !(defined(PETSC_HAVE_MUMPS) || defined(PETSC_HAVE_MKL_CPARDISO))
                    if(p == 1)
#endif
                        PetscCall(PCSetType(pc, S == 'S' ? PCCHOLESKY : PCLU));
                }
            }
            PetscCall(KSPSetFromOptions(v.level_->ksp));
            PetscCall(MatDestroy(&E));
            super::s_ = v.level_;
        }
        if(!coarse)
            super::template numfact<S, Operator::factorize_>(nrow / (!blocked ? 1 : local_), I, loc2glob, J, pt, neighbors);
        else {
            delete [] I;
            delete [] loc2glob;
            delete [] pt;
            delete [] neighbors;
            delete [] super::di_;
            super::di_ = nullptr;
        }
#endif
        std::swap(DMatrix::n_, rank);
        if(T == 1)
            std::iota(DMatrix::ldistribution_ + 1, DMatrix::ldistribution_ + p, 1);
        else if(U && excluded == 2)
            for(unsigned short i = 1; i < p; ++i)
                DMatrix::ldistribution_[i] += i;
#else
# ifdef HPDDM_CSR_CO
#  ifndef DHYPRE
         std::partial_sum(I, I + 1 + nrow / (!blocked ? 1 : local_), I);
#  endif
#  if defined(DSUITESPARSE) || defined(DLAPACK)
        super::template numfact<S>(nrow, I, J, pt);
        delete [] loc2glob;
#  elif defined(DMKL_PARDISO) || defined(DELEMENTAL)
        super::template numfact<S>(!blocked ? 1 : local_, I, loc2glob, J, pt);
        C = reinterpret_cast<K*>(pt);
#  else
        super::template numfact<S>(nrow, I, loc2glob, J, pt);
#  endif
# else
        super::template numfact<S>(size, I, J, pt);
# endif
        delete [] C;
#endif
        if(!treeDimension)
            delete [] rqRecv;
    }
    if(excluded < 2 && !M.empty()) {
        delete [] *sendNeighbor;
        *sendNeighbor = nullptr;
    }
    finishSetup<T, U, excluded, blocked>(infoWorld, rankSplit, p, infoSplit, rank);
#if !HPDDM_PETSC
    return ret;
#else
#if HPDDM_PETSC && defined(PETSC_HAVE_MUMPS)
    if(extended != MPI_COMM_NULL)
        MPI_Comm_free(&extended);
#endif
    PetscFunctionReturn(0);
#endif
}

HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K)
template<char
#if !HPDDM_PETSC
              T
#else
              S
#endif
               , unsigned short U, unsigned short excluded, class Operator>
inline typename CoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>::return_type CoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>::constructionMatrix(typename std::enable_if<Operator::pattern_ == 'u', Operator>::type& v) {
#if HPDDM_PETSC
    PetscFunctionBeginUser;
#endif
    unsigned short* const info = new unsigned short[(U != 1 ? 3 : 1) + v.getConnectivity()]();
    const std::vector<unsigned short>& sparsity = v.getPattern();
    info[0] = sparsity.size(); // number of intersections
    int rank;
    MPI_Comm_rank(v.p_.getCommunicator(), &rank);
    const unsigned short first = (S == 'S' ? std::distance(sparsity.cbegin(), std::upper_bound(sparsity.cbegin(), sparsity.cend(), rank)) : 0);
    int rankSplit;
    MPI_Comm_rank(scatterComm_, &rankSplit);
    unsigned short* infoNeighbor;

    unsigned int size = 0;

#if !HPDDM_PETSC
    const Option& opt = *Option::get();
    const unsigned short p = opt.val<unsigned short>("p", 1);
#else
    constexpr unsigned short T = 0;
    unsigned short p;
    {
        PetscInt n = 1;
        PetscOptionsGetInt(nullptr, v.prefix_.c_str(), "-p", &n, nullptr);
        p = n;
    }
#endif
    constexpr bool blocked = false;
    if(U != 1) {
        infoNeighbor = new unsigned short[info[0]];
        info[1] = (excluded == 2 ? 0 : local_); // number of eigenvalues
        std::vector<MPI_Request> rqInfo;
        rqInfo.reserve(2 * info[0]);
        MPI_Request rq;
        if(excluded == 0) {
            if(T != 2) {
                for(unsigned short i = 0; i < info[0]; ++i)
                    if(!(T == 1 && sparsity[i] < p) &&
                       !(T == 0 && (sparsity[i] % (sizeWorld_ / p) == 0) && sparsity[i] < p * (sizeWorld_ / p))) {
                        MPI_Isend(info + 1, 1, MPI_UNSIGNED_SHORT, sparsity[i], 1, v.p_.getCommunicator(), &rq);
                        rqInfo.emplace_back(rq);
                    }
            }
            else {
                for(unsigned short i = 0; i < info[0]; ++i)
                    if(!std::binary_search(DMatrix::ldistribution_, DMatrix::ldistribution_ + p, sparsity[i])) {
                        MPI_Isend(info + 1, 1, MPI_UNSIGNED_SHORT, sparsity[i], 1, v.p_.getCommunicator(), &rq);
                        rqInfo.emplace_back(rq);
                    }
            }
        }
        else if(excluded < 2)
            for(unsigned short i = 0; i < info[0]; ++i) {
                MPI_Isend(info + 1, 1, MPI_UNSIGNED_SHORT, sparsity[i], 1, v.p_.getCommunicator(), &rq);
                rqInfo.emplace_back(rq);
            }
        if(rankSplit) {
            for(unsigned short i = 0; i < info[0]; ++i) {
                MPI_Irecv(infoNeighbor + i, 1, MPI_UNSIGNED_SHORT, sparsity[i], 1, v.p_.getCommunicator(), &rq);
                rqInfo.emplace_back(rq);
            }
            size = (S != 'S' ? local_ : 0);
            for(unsigned short i = 0; i < info[0]; ++i) {
                int index;
                MPI_Waitany(info[0], &rqInfo.back() - info[0] + 1, &index, MPI_STATUS_IGNORE);
                if(!(S == 'S' && sparsity[index] < rank))
                    size += infoNeighbor[index];
            }
            rqInfo.resize(rqInfo.size() - info[0]);
            info[2] = size;
            size *= local_;
            if(S == 'S') {
                info[0] -= first;
                size += local_ * (local_ + 1) / 2;
            }
            if(local_) {
                if(excluded == 0)
                    std::copy_n(sparsity.cbegin() + first, info[0], info + (U != 1 ? 3 : 1));
                else {
                    if(T != 1) {
                        for(unsigned short i = 0; i < info[0]; ++i) {
                            info[(U != 1 ? 3 : 1) + i] = sparsity[i + first] + 1;
                            for(unsigned short j = 0; j < p - 1 && info[(U != 1 ? 3 : 1) + i] >= (T == 0 ? (sizeWorld_ / p) * (j + 1) : DMatrix::ldistribution_[j + 1]); ++j)
                                ++info[(U != 1 ? 3 : 1) + i];
                        }
                    }
                    else {
                        for(unsigned short i = 0; i < info[0]; ++i)
                            info[(U != 1 ? 3 : 1) + i] = p + sparsity[i + first];
                    }
                }
            }
        }
        MPI_Waitall(rqInfo.size(), rqInfo.data(), MPI_STATUSES_IGNORE);
    }
    else {
        infoNeighbor = nullptr;
        if(rankSplit) {
            if(S == 'S') {
                info[0] -= first;
                size = local_ * local_ * info[0] + (!blocked ? local_ * (local_ + 1) / 2 : local_ * local_);
            }
            else
                size = local_ * local_ * (1 + info[0]);
            std::copy_n(sparsity.cbegin() + first, info[0], info + (U != 1 ? 3 : 1));
        }
    }
    unsigned short** infoSplit;
    unsigned short*  infoWorld = nullptr;
#ifdef HPDDM_CSR_CO
    int* loc2glob;
#endif
    if(rankSplit)
        MPI_Gather(info, (U != 1 ? 3 : 1) + v.getConnectivity(), MPI_UNSIGNED_SHORT, NULL, 0, MPI_DATATYPE_NULL, 0, scatterComm_);
    else {
        size = 0;
        infoSplit = new unsigned short*[sizeSplit_];
        *infoSplit = new unsigned short[sizeSplit_ * ((U != 1 ? 3 : 1) + v.getConnectivity()) + (U != 1) * sizeWorld_];
        MPI_Gather(info, (U != 1 ? 3 : 1) + v.getConnectivity(), MPI_UNSIGNED_SHORT, *infoSplit, (U != 1 ? 3 : 1) + v.getConnectivity(), MPI_UNSIGNED_SHORT, 0, scatterComm_);
        for(unsigned int i = 1; i < sizeSplit_; ++i)
            infoSplit[i] = *infoSplit + i * ((U != 1 ? 3 : 1) + v.getConnectivity());
        if(S == 'S' && Operator::pattern_ == 's')
            **infoSplit -= first;
        unsigned int* offsetIdx = new unsigned int[std::max(sizeSplit_ - 1, 2 * p)];
        if(U != 1) {
            infoWorld = *infoSplit + sizeSplit_ * (3 + v.getConnectivity());
            int* recvcounts = reinterpret_cast<int*>(offsetIdx);
            int* displs = recvcounts + p;
            displs[0] = 0;
            if(T == 2) {
                std::adjacent_difference(DMatrix::ldistribution_ + 1, DMatrix::ldistribution_ + p, recvcounts);
                recvcounts[p - 1] = sizeWorld_ - DMatrix::ldistribution_[p - 1];
            }
            else {
                std::fill_n(recvcounts, p - 1, sizeWorld_ / p);
                recvcounts[p - 1] = sizeWorld_ - (p - 1) * (sizeWorld_ / p);
            }
            std::partial_sum(recvcounts, recvcounts + p - 1, displs + 1);
            for(unsigned int i = 0; i < sizeSplit_; ++i)
                infoWorld[displs[DMatrix::rank_] + i] = infoSplit[i][1];
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, infoWorld, recvcounts, displs, MPI_UNSIGNED_SHORT, DMatrix::communicator_);
            if(T == 1) {
                unsigned int i = (p - 1) * (sizeWorld_ / p);
                for(unsigned short k = p - 1, j = 1; k-- > 0; i -= sizeWorld_ / p, ++j) {
                    recvcounts[k] = infoWorld[i];
                    std::copy_backward(infoWorld + k * (sizeWorld_ / p), infoWorld + (k + 1) * (sizeWorld_ / p), infoWorld + (k + 1) * (sizeWorld_ / p) + j);
                }
                std::copy_n(recvcounts, p - 1, infoWorld + 1);
            }
            v.max_ = std::accumulate(infoWorld, infoWorld + rankWorld_, 0);
            DMatrix::n_ = std::accumulate(infoWorld + rankWorld_, infoWorld + sizeWorld_, v.max_);
            if(super::numbering_ == 'F')
                ++v.max_;
            unsigned short tmp = 0;
            for(unsigned short i = 0; i < info[0]; ++i) {
                infoNeighbor[i] = infoWorld[sparsity[i]];
                if(!(S == 'S' && i < first))
                    tmp += infoNeighbor[i];
            }
            for(unsigned short k = 1; k < sizeSplit_; ++k) {
                offsetIdx[k - 1] = size;
                size += infoSplit[k][2] * infoSplit[k][1] + (S == 'S' ? infoSplit[k][1] * (infoSplit[k][1] + 1) / 2 : 0);
            }
            if(excluded < 2)
                size += local_ * tmp + (S == 'S' ? local_ * (local_ + 1) / 2 : local_ * local_);
            if(S == 'S')
                info[0] -= first;
        }
        else {
            DMatrix::n_ = (sizeWorld_ - (excluded == 2 ? p : 0)) * local_;
            v.max_ = (rankWorld_ - (excluded == 2 ? rank : 0)) * local_ + (super::numbering_ == 'F');
            if(S == 'S') {
                for(unsigned short i = 1; i < sizeSplit_; size += infoSplit[i++][0])
                    offsetIdx[i - 1] = size * local_ * local_ + (i - 1) * (!blocked ? local_ * (local_ + 1) / 2 : local_ * local_);
                info[0] -= first;
                size = (size + info[0]) * local_ * local_ + (sizeSplit_ - (excluded == 2)) * (!blocked ? local_ * (local_ + 1) / 2 : local_ * local_);
            }
            else {
                for(unsigned short i = 1; i < sizeSplit_; size += infoSplit[i++][0])
                    offsetIdx[i - 1] = (i - 1 + size) * local_ * local_;
                size = (size + info[0] + sizeSplit_ - (excluded == 2)) * local_ * local_;
            }
            if(sizeSplit_ == 1)
                offsetIdx[0] = size;
        }
        delete [] offsetIdx;
    }
    if(rankSplit) {
        delete [] info;
        sizeRHS_ = local_;
        if(U != 1)
            delete [] infoNeighbor;
        if(U == 0)
            DMatrix::displs_ = &rankWorld_;
    }
    else {
#ifdef HPDDM_CONTIGUOUS
        loc2glob = new int[2];
        const unsigned short relative = (T == 1 ? p + rankWorld_ * ((sizeWorld_ / p) - 1) - 1 : rankWorld_);
        if(excluded == 2 || sizeSplit_ > 1) {
            unsigned int* offsetPosition = nullptr;
            if(U != 1) {
                offsetPosition = new unsigned int[sizeSplit_];
                offsetPosition[0] = std::accumulate(infoWorld, infoWorld + relative, static_cast<unsigned int>(super::numbering_ == 'F'));
                if(T != 1)
                    for(unsigned int k = 1; k < sizeSplit_; ++k)
                        offsetPosition[k] = offsetPosition[k - 1] + infoSplit[k - 1][1];
                else
                    for(unsigned int k = 1; k < sizeSplit_; ++k)
                        offsetPosition[k] = offsetPosition[k - 1] + infoWorld[relative + k - 1];
            }
            if(excluded == 2)
                loc2glob[0] = (U == 1 ? (relative + 1 - (T == 1 ? p : 1 + rank)) * (!blocked ? local_ : 1) + (super::numbering_ == 'F') : offsetPosition[1]);
            if(sizeSplit_ > 1)
                loc2glob[1] = (U == 1 ? (relative + sizeSplit_ - 1 - (U == 1 && excluded == 2 ? (T == 1 ? p : 1 + rank) : 0)) * (!blocked ? local_ : 1) + (super::numbering_ == 'F') : offsetPosition[sizeSplit_ - 1]) + (!blocked ? (U == 1 ? local_ : infoSplit[sizeSplit_ - 1][1]) - 1 : 0);
            delete [] offsetPosition;
        }
        if(excluded < 2) {
            loc2glob[0] = ((!blocked || local_ == 1) ? v.max_ : v.max_ / local_ + (super::numbering_ == 'F'));
            if(sizeSplit_ == 1)
                loc2glob[1] = ((!blocked || local_ == 1) ? v.max_ + local_ - 1 : v.max_ / local_ + (super::numbering_ == 'F'));
        }
        else if(sizeSplit_ == 1) {
            loc2glob[0] = 2;
            loc2glob[1] = 1;
        }
#endif
        delete [] info;
        if(U != 1)
            delete [] infoNeighbor;
        const K* const E = v.p_.getOperator();
#if !HPDDM_PETSC
#if defined(DSUITESPARSE) || defined(DLAPACK)
        super::template numfact<S>(DMatrix::n_, nullptr, nullptr, const_cast<K*&>(E));
        delete [] loc2glob;
#elif defined(HPDDM_CONTIGUOUS)
        super::template numfact<S>(!blocked ? 1 : local_, nullptr, loc2glob, nullptr, const_cast<K*&>(E));
#endif
#else
#ifdef HPDDM_CONTIGUOUS
        delete [] loc2glob;
#endif
        Mat P, A;
        PetscScalar* ptr;
        PetscCall(MatCreateDense(DMatrix::communicator_, DMatrix::n_, DMatrix::n_, DMatrix::n_, DMatrix::n_, nullptr, &P));
        PetscCall(MatDenseGetArrayWrite(P, &ptr));
        std::copy_n(E, DMatrix::n_ * DMatrix::n_, ptr);
        PetscCall(MatDenseRestoreArrayWrite(P, &ptr));
        PetscCall(MatSetOptionsPrefix(P, v.prefix_.c_str()));
        PetscCall(MatSetFromOptions(P));
        PetscCall(KSPGetOperators(v.level_->parent->levels[0]->ksp, nullptr, &A));
        PetscCall(MatPropagateSymmetryOptions(A, P));
        PetscCall(KSPCreate(DMatrix::communicator_, &v.level_->ksp));
        PetscCall(KSPSetOperators(v.level_->ksp, P, P));
        PetscCall(KSPSetOptionsPrefix(v.level_->ksp, v.prefix_.c_str()));
        PetscCall(KSPSetFromOptions(v.level_->ksp));
        PetscCall(MatDestroy(&P));
        super::s_ = v.level_;
#endif
    }
    finishSetup<T, U, excluded, blocked>(infoWorld, rankSplit, p, infoSplit, rank);
#if !HPDDM_PETSC
    return nullptr;
#else
    PetscFunctionReturn(0);
#endif
}

HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K)
template<char T, unsigned short U, unsigned short excluded, bool blocked>
inline void CoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>::finishSetup(unsigned short*& infoWorld, const int rankSplit, const unsigned short p, unsigned short**& infoSplit, const int rank) {
#if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
    DMatrix::distribution_ = static_cast<DMatrix::Distribution>(Option::get()->val<char>("distribution", HPDDM_DISTRIBUTION_CENTRALIZED));
#endif
    if(U != 2) {
#if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
        if(DMatrix::distribution_ == DMatrix::CENTRALIZED) {
            if(gatherComm_ != scatterComm_) {
                MPI_Comm_free(&scatterComm_);
                scatterComm_ = gatherComm_;
            }
        }
#else
        if(gatherComm_ != scatterComm_) {
            MPI_Comm_free(&gatherComm_);
            gatherComm_ = scatterComm_;
        }
#endif
    }
    else {
        unsigned int size;
        unsigned short* pt;
#if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
        if(DMatrix::distribution_ == DMatrix::CENTRALIZED) {
            if(rankSplit)
                infoWorld = new unsigned short[sizeWorld_];
            pt = infoWorld;
            size = sizeWorld_;
        }
        else {
            size = sizeWorld_ + sizeSplit_;
            pt = new unsigned short[size];
            if(rankSplit == 0) {
                std::copy_n(infoWorld, sizeWorld_, pt);
                for(unsigned int i = 0; i < sizeSplit_; ++i)
                    pt[sizeWorld_ + i] = infoSplit[i][1];
            }
        }
#else
        unsigned short* infoMain;
        if(rankSplit == 0) {
            infoMain = infoSplit[0];
            for(unsigned int i = 0; i < sizeSplit_; ++i)
                infoMain[i] = infoSplit[i][1];
        }
        else
            infoMain = new unsigned short[sizeSplit_];
        pt = infoMain;
        size = sizeSplit_;
#endif
        MPI_Bcast(pt, size, MPI_UNSIGNED_SHORT, 0, scatterComm_);
#if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
        if(DMatrix::distribution_ == DMatrix::CENTRALIZED) {
            constructionCommunicatorCollective<(excluded > 0)>(pt, size, gatherComm_, &scatterComm_);
#else
            constructionCommunicatorCollective<false>(pt, size, scatterComm_);
#endif
            if(gatherComm_ != scatterComm_) {
                if(scatterComm_ != MPI_COMM_NULL)
                    MPI_Comm_free(&gatherComm_);
                gatherComm_ = scatterComm_;
            }
#if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
        }
        else {
            constructionCommunicatorCollective<(excluded > 0)>(pt, sizeWorld_, gatherComm_);
            constructionCommunicatorCollective<false>(pt + sizeWorld_, sizeSplit_, scatterComm_);
        }
#endif
        if(rankSplit
#if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
                || DMatrix::distribution_ == DMatrix::DISTRIBUTED_SOL
#endif
                )
            delete [] pt;
    }
    if(rankSplit == 0) {
#if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
        if(DMatrix::distribution_ == DMatrix::CENTRALIZED) {
            if(rankWorld_ == 0) {
                sizeRHS_ = DMatrix::n_;
                if(U == 1)
                    constructionCollective<true, DMatrix::CENTRALIZED, excluded == 2>();
                else if(U == 2) {
                    DMatrix::gatherCounts_ = new int[1];
                    if(local_ == 0) {
                        local_ = *DMatrix::gatherCounts_ = *std::find_if(infoWorld, infoWorld + sizeWorld_, [](const unsigned short& nu) { return nu != 0; });
                        sizeRHS_ += local_;
                    }
                    else
                        *DMatrix::gatherCounts_ = local_;
                }
                else
                    constructionCollective<false, DMatrix::CENTRALIZED, excluded == 2>(infoWorld, p - 1);
            }
            else {
                if(U == 0)
                    DMatrix::displs_ = &rankWorld_;
                sizeRHS_ = local_;
            }
        }
        else {
#endif
            constructionMap<T, U == 1, excluded == 2>(p, U == 1 ? nullptr : infoWorld);
#if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
            if(rankWorld_ == 0)
                sizeRHS_ = DMatrix::n_;
            else
#endif
                sizeRHS_ = DMatrix::ldistribution_[DMatrix::rank_];
            if(U == 1)
                constructionCollective<true, DMatrix::DISTRIBUTED_SOL, excluded == 2>();
            else if(U == 2) {
                DMatrix::gatherCounts_ = new int[1];
                if(local_ == 0) {
                    local_ = *DMatrix::gatherCounts_ = *std::find_if(infoWorld, infoWorld + sizeWorld_, [](const unsigned short& nu) { return nu != 0; });
                    sizeRHS_ += local_;
                }
                else
                    *DMatrix::gatherCounts_ = local_;
            }
            else {
                unsigned short* infoMain = infoSplit[0];
                for(unsigned int i = 0; i < sizeSplit_; ++i)
                    infoMain[i] = infoSplit[i][1];
                constructionCollective<false, DMatrix::DISTRIBUTED_SOL, excluded == 2>(infoWorld, p - 1, infoMain);
            }
#if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
        }
#endif
        delete [] *infoSplit;
        delete [] infoSplit;
        if(excluded == 2) {
#if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
            if(DMatrix::distribution_ == DMatrix::CENTRALIZED && rankWorld_ == 0)
                sizeRHS_ += local_;
            else if(DMatrix::distribution_ == DMatrix::DISTRIBUTED_SOL)
#endif
                sizeRHS_ += local_;
        }
        DMatrix::n_
#if HPDDM_INEXACT_COARSE_OPERATOR
            = rank /
#else
            /=
#endif
                    (!blocked ? 1 : local_);
    }
#if !HPDDM_INEXACT_COARSE_OPERATOR
    ignore(rank);
#endif
}

HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K)
template<bool excluded>
inline void CoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>::callSolver(K* const pt, const unsigned short& mu) {
    downscaled_type<K>* rhs = reinterpret_cast<downscaled_type<K>*>(pt);
    if(!std::is_same<downscaled_type<K>, K>::value)
        for(int i = 0; i < mu * local_; ++i)
            rhs[i] = pt[i];
    if(scatterComm_ != MPI_COMM_NULL) {
#if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
        if(DMatrix::distribution_ == DMatrix::DISTRIBUTED_SOL) {
            if(DMatrix::displs_) {
                if(rankWorld_ == 0) {
                    int p = 0;
                    if(excluded) {
                        MPI_Comm_size(DMatrix::communicator_, &p);
                        --p;
                    }
                    transfer<false>(DMatrix::gatherCounts_, sizeWorld_ - p, mu, rhs);
                    std::for_each(DMatrix::gatherCounts_, DMatrix::displs_ + sizeWorld_ - 2 * p, [&](int& i) { i /= mu; });
                }
                else if(gatherComm_ != MPI_COMM_NULL)
                    MPI_Gatherv(rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_);
                if(DMatrix::communicator_ != MPI_COMM_NULL) {
                    super::template solve<DMatrix::DISTRIBUTED_SOL>(rhs, mu);
                    std::for_each(DMatrix::gatherSplitCounts_, DMatrix::displsSplit_ + sizeSplit_, [&](int& i) { i *= mu; });
                    transfer<true>(DMatrix::gatherSplitCounts_, mu, sizeSplit_, rhs);
                }
                else
                    MPI_Scatterv(NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), 0, scatterComm_);
            }
            else {
                if(rankWorld_ == 0) {
                    MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rhs, mu * *DMatrix::gatherCounts_, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_);
                    int p = 0;
                    if(offset_ || excluded)
                        MPI_Comm_size(DMatrix::communicator_, &p);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(sizeWorld_ - p, mu, rhs + (p ? mu * *DMatrix::gatherCounts_ : 0), *DMatrix::gatherCounts_);
                }
                else if(gatherComm_ != MPI_COMM_NULL)
                    MPI_Gather(rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, MPI_DATATYPE_NULL, 0, gatherComm_);
                if(DMatrix::communicator_ != MPI_COMM_NULL) {
                    super::template solve<DMatrix::DISTRIBUTED_SOL>(rhs + (offset_ || excluded ? mu * *DMatrix::gatherCounts_ : 0), mu);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(mu, sizeSplit_ - (offset_ || excluded), rhs + (offset_ || excluded ? mu * *DMatrix::gatherCounts_ : 0), *DMatrix::gatherCounts_);
                    MPI_Scatter(rhs, mu * *DMatrix::gatherCounts_, Wrapper<downscaled_type<K>>::mpi_type(), MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 0, scatterComm_);
                }
                else
                    MPI_Scatter(NULL, 0, Wrapper<downscaled_type<K>>::mpi_type(), rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), 0, scatterComm_);
            }
        }
        else {
            int p = 0;
            if(DMatrix::displs_) {
                if(rankWorld_ == 0) {
                    if(excluded) {
                        MPI_Comm_size(DMatrix::communicator_, &p);
                        --p;
                    }
                    transfer<false>(DMatrix::gatherCounts_, sizeWorld_ - p, mu, rhs);
                }
                else if(gatherComm_ != MPI_COMM_NULL)
                    MPI_Gatherv(rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_);
                if(DMatrix::communicator_ != MPI_COMM_NULL)
                    super::template solve<DMatrix::CENTRALIZED>(rhs, mu);
                if(rankWorld_ == 0)
                    transfer<true>(DMatrix::gatherCounts_, mu, sizeWorld_ - p, rhs);
                else if(gatherComm_ != MPI_COMM_NULL)
                    MPI_Scatterv(NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_);
            }
            else {
                if(rankWorld_ == 0) {
                    if(offset_ || excluded)
                        MPI_Comm_size(DMatrix::communicator_, &p);
                    MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rhs, mu * *DMatrix::gatherCounts_, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(sizeWorld_ - p, mu, rhs + (p ? mu * *DMatrix::gatherCounts_ : 0), *DMatrix::gatherCounts_);
                }
                else
                    MPI_Gather(rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, MPI_DATATYPE_NULL, 0, gatherComm_);
                if(DMatrix::communicator_ != MPI_COMM_NULL)
                    super::template solve<DMatrix::CENTRALIZED>(rhs + (offset_ || excluded ? mu * local_ : 0), mu);
                if(rankWorld_ == 0) {
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(mu, sizeWorld_ - p, rhs + (p ? mu * *DMatrix::gatherCounts_ : 0), *DMatrix::gatherCounts_);
                    MPI_Scatter(rhs, mu * *DMatrix::gatherCounts_, Wrapper<downscaled_type<K>>::mpi_type(), MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 0, scatterComm_);
                }
                else
                    MPI_Scatter(NULL, 0, Wrapper<downscaled_type<K>>::mpi_type(), rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), 0, scatterComm_);
            }
        }
#else
            if(DMatrix::displs_) {
                if(DMatrix::communicator_ != MPI_COMM_NULL) {
                    transfer<false>(DMatrix::gatherSplitCounts_, sizeSplit_, mu, rhs);
                    super::solve(rhs, mu);
                    transfer<true>(DMatrix::gatherSplitCounts_, mu, sizeSplit_, rhs);
                }
                else {
                    MPI_Gatherv(rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_);
#if HPDDM_PETSC && defined(PETSC_HAVE_MUMPS)
                    if(super::s_)
                        super::solve(nullptr, mu);
#endif
                    MPI_Scatterv(NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), 0, scatterComm_);
                }
            }
            else {
                if(DMatrix::communicator_ != MPI_COMM_NULL) {
                    MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rhs, mu * *DMatrix::gatherCounts_, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(sizeSplit_ - (offset_ || excluded), mu, rhs + (offset_ || excluded ? mu * *DMatrix::gatherCounts_ : 0), *DMatrix::gatherCounts_);
                    super::solve(rhs + (offset_ || excluded ? mu * *DMatrix::gatherCounts_ : 0), mu);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(mu, sizeSplit_ - (offset_ || excluded), rhs + (offset_ || excluded ? mu * *DMatrix::gatherCounts_ : 0), *DMatrix::gatherCounts_);
                    MPI_Scatter(rhs, mu * *DMatrix::gatherCounts_, Wrapper<downscaled_type<K>>::mpi_type(), MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 0, scatterComm_);
                }
                else {
                    MPI_Gather(rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, MPI_DATATYPE_NULL, 0, gatherComm_);
#if HPDDM_PETSC && defined(PETSC_HAVE_MUMPS)
                    if(super::s_)
                        super::solve(nullptr, mu);
#endif
                    MPI_Scatter(NULL, 0, Wrapper<downscaled_type<K>>::mpi_type(), rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), 0, scatterComm_);
                }
            }
#endif
    }
    else if(DMatrix::communicator_ != MPI_COMM_NULL) {
#if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
        if(DMatrix::distribution_ == DMatrix::DISTRIBUTED_SOL)
            super::template solve<DMatrix::DISTRIBUTED_SOL>(rhs, mu);
        else
            super::template solve<DMatrix::CENTRALIZED>(rhs, mu);
#else
            super::solve(rhs, mu);
#endif
    }
    if(!std::is_same<downscaled_type<K>, K>::value)
        for(unsigned int i = mu * local_; i-- > 0; )
            pt[i] = rhs[i];
}

#if HPDDM_ICOLLECTIVE
template<template<class> class Solver, char S, class K>
template<bool excluded>
inline void CoarseOperator<Solver, S, K>::IcallSolver(K* const pt, const unsigned short& mu, MPI_Request* rq) {
    downscaled_type<K>* rhs = reinterpret_cast<downscaled_type<K>*>(pt);
    if(!std::is_same<downscaled_type<K>, K>::value)
        for(unsigned int i = 0; i < mu * local_; ++i)
            rhs[i] = pt[i];
    if(scatterComm_ != MPI_COMM_NULL) {
#if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
        if(DMatrix::distribution_ == DMatrix::DISTRIBUTED_SOL) {
            if(DMatrix::displs_) {
                if(rankWorld_ == 0) {
                    Itransfer<false>(DMatrix::gatherCounts_, sizeWorld_, mu, rhs, rq);
                    std::for_each(DMatrix::gatherCounts_, DMatrix::displs_ + sizeWorld_, [&](int& i) { i /= mu; });
                }
                else if(gatherComm_ != MPI_COMM_NULL)
                    MPI_Igatherv(rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_, rq);
                if(DMatrix::communicator_ != MPI_COMM_NULL) {
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    super::template solve<DMatrix::DISTRIBUTED_SOL>(rhs, mu);
                    std::for_each(DMatrix::gatherSplitCounts_, DMatrix::displsSplit_ + sizeSplit_, [&](int& i) { i *= mu; });
                    Itransfer<true>(DMatrix::gatherSplitCounts_, mu, sizeSplit_, rhs, rq + 1);
                }
                else
                    MPI_Iscatterv(NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), 0, scatterComm_, rq + 1);
            }
            else {
                if(rankWorld_ == 0) {
                    MPI_Igather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rhs, mu * *DMatrix::gatherCounts_, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_, rq);
                    int p = 0;
                    if(offset_ || excluded)
                        MPI_Comm_size(DMatrix::communicator_, &p);
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    *rq = MPI_REQUEST_NULL;
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(sizeWorld_ - p, mu, rhs + (p ? mu * *DMatrix::gatherCounts_ : 0), *DMatrix::gatherCounts_);
                }
                else if(gatherComm_ != MPI_COMM_NULL)
                    MPI_Igather(rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_, rq);
                if(DMatrix::communicator_ != MPI_COMM_NULL) {
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    super::template solve<DMatrix::DISTRIBUTED_SOL>(rhs + (offset_ || excluded ? mu * *DMatrix::gatherCounts_ : 0), mu);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(mu, sizeSplit_ - (offset_ || excluded), rhs + (offset_ || excluded ? mu * *DMatrix::gatherCounts_ : 0), *DMatrix::gatherCounts_);
                    MPI_Iscatter(rhs, mu * *DMatrix::gatherCounts_, Wrapper<downscaled_type<K>>::mpi_type(), MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 0, scatterComm_, rq + 1);
                }
                else
                    MPI_Iscatter(NULL, 0, MPI_DATATYPE_NULL, rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), 0, scatterComm_, rq + 1);
            }
        }
        else {
            if(DMatrix::displs_) {
                if(rankWorld_ == 0)
                    Itransfer<false>(DMatrix::gatherCounts_, sizeWorld_, mu, rhs, rq);
                else if(gatherComm_ != MPI_COMM_NULL)
                    MPI_Igatherv(rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_, rq);
                if(DMatrix::communicator_ != MPI_COMM_NULL) {
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    super::template solve<DMatrix::CENTRALIZED>(rhs, mu);
                }
                if(rankWorld_ == 0)
                    Itransfer<true>(DMatrix::gatherCounts_, mu, sizeWorld_, rhs, rq + 1);
                else if(gatherComm_ != MPI_COMM_NULL)
                    MPI_Iscatterv(NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_, rq + 1);
            }
            else {
                int p = 0;
                if(rankWorld_ == 0) {
                    if(offset_ || excluded)
                        MPI_Comm_size(DMatrix::communicator_, &p);
                    MPI_Igather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rhs, mu * *DMatrix::gatherCounts_, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_, rq);
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    *rq = MPI_REQUEST_NULL;
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(sizeWorld_ - p, mu, rhs + (p ? mu * *DMatrix::gatherCounts_ : 0), *DMatrix::gatherCounts_);
                }
                else
                    MPI_Igather(rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_, rq);
                if(DMatrix::communicator_ != MPI_COMM_NULL) {
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    super::template solve<DMatrix::CENTRALIZED>(rhs + (offset_ || excluded ? mu * local_ : 0), mu);
                }
                if(rankWorld_ == 0) {
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(mu, sizeWorld_ - p, rhs + (p ? mu * *DMatrix::gatherCounts_ : 0), *DMatrix::gatherCounts_);
                    MPI_Iscatter(rhs, mu * *DMatrix::gatherCounts_, Wrapper<downscaled_type<K>>::mpi_type(), MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 0, scatterComm_, rq + 1);
                }
                else
                    MPI_Iscatter(NULL, 0, MPI_DATATYPE_NULL, rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), 0, scatterComm_, rq + 1);
            }
        }
#else
            if(DMatrix::displs_) {
                if(DMatrix::communicator_ != MPI_COMM_NULL) {
                    Itransfer<false>(DMatrix::gatherSplitCounts_, sizeSplit_, mu, rhs, rq);
                    super::solve(rhs, mu);
                    Itransfer<true>(DMatrix::gatherSplitCounts_, mu, sizeSplit_, rhs, rq + 1);
                }
                else {
                    MPI_Igatherv(rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_, rq);
                    MPI_Iscatterv(NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), 0, scatterComm_, rq + 1);
                }
            }
            else {
                if(DMatrix::communicator_ != MPI_COMM_NULL) {
                    MPI_Igather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rhs, mu * *DMatrix::gatherCounts_, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_, rq);
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(sizeSplit_ - (offset_ || excluded), mu, rhs + (offset_ || excluded ? mu * *DMatrix::gatherCounts_ : 0), *DMatrix::gatherCounts_);
                    super::solve(rhs + (offset_ || excluded ? mu * *DMatrix::gatherCounts_ : 0), mu);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(mu, sizeSplit_ - (offset_ || excluded), rhs + (offset_ || excluded ? mu * *DMatrix::gatherCounts_ : 0), *DMatrix::gatherCounts_);
                    MPI_Iscatter(rhs, mu * *DMatrix::gatherCounts_, Wrapper<downscaled_type<K>>::mpi_type(), MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 0, scatterComm_, rq + 1);
                }
                else {
                    MPI_Igather(rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_, rq);
                    MPI_Iscatter(NULL, 0, MPI_DATATYPE_NULL, rhs, mu * local_, Wrapper<downscaled_type<K>>::mpi_type(), 0, scatterComm_, rq + 1);
                }
            }
#endif
    }
    else if(DMatrix::communicator_ != MPI_COMM_NULL) {
#if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
        if(DMatrix::distribution_ == DMatrix::DISTRIBUTED_SOL)
            super::template solve<DMatrix::DISTRIBUTED_SOL>(rhs, mu);
        else
            super::template solve<DMatrix::CENTRALIZED>(rhs, mu);
#else
            super::solve(rhs, mu);
#endif
        return;
    }
    if(!std::is_same<downscaled_type<K>, K>::value)
        for(unsigned int i = mu * local_; i-- > 0; )
            pt[i] = rhs[i];
}
#endif // HPDDM_ICOLLECTIVE
} // HPDDM
#endif // HPDDM_COARSE_OPERATOR_IMPL_HPP_
