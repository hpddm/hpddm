/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
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

#ifndef _HPDDM_COARSE_OPERATOR_IMPL_
#define _HPDDM_COARSE_OPERATOR_IMPL_

#include <array>
#include "coarse_operator.hpp"

namespace HPDDM {
template<template<class> class Solver, char S, class K>
template<bool exclude>
inline void CoarseOperator<Solver, S, K>::constructionCommunicator(const MPI_Comm& comm) {
    MPI_Comm_size(comm, &_sizeWorld);
    MPI_Comm_rank(comm, &_rankWorld);
    Option& opt = *Option::get();
    unsigned short p = opt.val<unsigned short>("master_p", 1);
#ifndef DSUITESPARSE
    if(p > _sizeWorld / 2 && _sizeWorld > 1) {
        p = opt["master_p"] = _sizeWorld / 2;
        if(_rankWorld == 0)
            std::cout << "WARNING -- the number of master processes was set to a value greater than MPI_Comm_size / 2, the value has been reset to " << p << std::endl;
    }
#else
    p = opt["master_p"] = 1;
#endif
    if(p == 1) {
        MPI_Comm_dup(comm, &_scatterComm);
        _gatherComm = _scatterComm;
        if(_rankWorld)
            DMatrix::_communicator = MPI_COMM_NULL;
        else
            DMatrix::_communicator = MPI_COMM_SELF;
        DMatrix::_rank = 0;
        DMatrix::_ldistribution = new int[1]();
    }
    else {
        MPI_Group master, split;
        MPI_Group world;
        MPI_Comm_group(comm, &world);
        int* ps;
        unsigned int tmp;
        DMatrix::_ldistribution = new int[p];
        const char T = opt.val<char>("master_topology", 0);
        if(T == 2) {
            // Here, it is assumed that all subdomains have the same number of coarse degrees of freedom as the rank 0 ! (only true when the distribution is uniform)
            float area = _sizeWorld *_sizeWorld / (2.0 * p);
            *DMatrix::_ldistribution = 0;
            for(unsigned short i = 1; i < p; ++i)
                DMatrix::_ldistribution[i] = static_cast<int>(_sizeWorld - std::sqrt(std::max(_sizeWorld * _sizeWorld - 2 * _sizeWorld * DMatrix::_ldistribution[i - 1] - 2 * area + DMatrix::_ldistribution[i - 1] * DMatrix::_ldistribution[i - 1], 1.0f)) + 0.5);
            int* idx = std::upper_bound(DMatrix::_ldistribution, DMatrix::_ldistribution + p, _rankWorld);
            unsigned short i = idx - DMatrix::_ldistribution;
            tmp = (i == p) ? _sizeWorld - DMatrix::_ldistribution[i - 1] : DMatrix::_ldistribution[i] - DMatrix::_ldistribution[i - 1];
            ps = new int[tmp];
            for(unsigned int j = 0; j < tmp; ++j)
                ps[j] = DMatrix::_ldistribution[i - 1] + j;
        }
#ifndef HPDDM_CONTIGUOUS
        else if(T == 1) {
            if(_rankWorld == p - 1 || _rankWorld > p - 1 + (p - 1) * ((_sizeWorld - p) / p))
                tmp = _sizeWorld - (p - 1) * (_sizeWorld / p);
            else
                tmp = _sizeWorld / p;
            ps = new int[tmp];
            if(_rankWorld < p)
                ps[0] = _rankWorld;
            else {
                if(tmp == _sizeWorld / p)
                    ps[0] = (_rankWorld - p) / ((_sizeWorld - p) / p);
                else
                    ps[0] = p - 1;
            }
            unsigned int offset = ps[0] * (_sizeWorld / p - 1) + p - 1;
            std::iota(ps + 1, ps + tmp, offset + 1);
            std::iota(DMatrix::_ldistribution, DMatrix::_ldistribution + p, 0);
        }
#endif
        else {
            if(T != 0)
                opt["master_topology"] = 0;
            if(_rankWorld < (p - 1) * (_sizeWorld / p))
                tmp = _sizeWorld / p;
            else
                tmp = _sizeWorld - (p - 1) * (_sizeWorld / p);
            ps = new int[tmp];
            unsigned int offset;
            if(tmp != _sizeWorld / p)
                offset = _sizeWorld - tmp;
            else
                offset = (_sizeWorld / p) * (_rankWorld / (_sizeWorld / p));
            std::iota(ps, ps + tmp, offset);
            for(unsigned short i = 0; i < p; ++i)
                DMatrix::_ldistribution[i] = i * (_sizeWorld / p);
        }
        MPI_Group_incl(world, p, DMatrix::_ldistribution, &master);
        MPI_Group_incl(world, tmp, ps, &split);
        delete [] ps;

        MPI_Comm_create(comm, master, &(DMatrix::_communicator));
        if(DMatrix::_communicator != MPI_COMM_NULL)
            MPI_Comm_rank(DMatrix::_communicator, &(DMatrix::_rank));
        MPI_Comm_create(comm, split, &_scatterComm);

        MPI_Group_free(&master);
        MPI_Group_free(&split);

        if(!exclude)
            MPI_Comm_dup(comm, &_gatherComm);
        else {
            MPI_Group global;
            MPI_Group_excl(world, p - 1, DMatrix::_ldistribution + 1, &global);
            MPI_Comm_create(comm, global, &_gatherComm);
            MPI_Group_free(&global);
        }
        MPI_Group_free(&world);
    }
}

template<template<class> class Solver, char S, class K>
template<bool U, typename DMatrix::Distribution D, bool excluded>
inline void CoarseOperator<Solver, S, K>::constructionCollective(const unsigned short* info, unsigned short p, const unsigned short* infoSplit) {
    if(!U) {
        if(excluded)
            _sizeWorld -= p;
        DMatrix::_gatherCounts = new int[2 * _sizeWorld];
        DMatrix::_displs = DMatrix::_gatherCounts + _sizeWorld;

        DMatrix::_gatherCounts[0] = info[0];
        DMatrix::_displs[0] = 0;
        for(unsigned int i = 1, j = 1; j < _sizeWorld; ++i)
            if(!excluded || info[i])
                DMatrix::_gatherCounts[j++] = info[i];
        std::partial_sum(DMatrix::_gatherCounts, DMatrix::_gatherCounts + _sizeWorld - 1, DMatrix::_displs + 1);
        if(excluded)
            _sizeWorld += p;
        if(D == DMatrix::DISTRIBUTED_SOL) {
            DMatrix::_gatherSplitCounts = new int[2 * _sizeSplit];
            DMatrix::_displsSplit = DMatrix::_gatherSplitCounts + _sizeSplit;
            std::copy_n(infoSplit, _sizeSplit, DMatrix::_gatherSplitCounts);
            DMatrix::_displsSplit[0] = 0;
            std::partial_sum(DMatrix::_gatherSplitCounts, DMatrix::_gatherSplitCounts + _sizeSplit - 1, DMatrix::_displsSplit + 1);
        }
    }
    else {
        DMatrix::_gatherCounts = new int[1];
        *DMatrix::_gatherCounts = _local;
    }
}

template<template<class> class Solver, char S, class K>
template<char T, bool U, bool excluded>
inline void CoarseOperator<Solver, S, K>::constructionMap(unsigned short p, const unsigned short* info) {
    if(T == 0) {
        if(!U) {
            unsigned int accumulate = 0;
            for(unsigned short i = 0; i < p - 1; accumulate += DMatrix::_ldistribution[i++])
                DMatrix::_ldistribution[i] = std::accumulate(info + i * (_sizeWorld / p), info + (i + 1) * (_sizeWorld / p), 0);
            DMatrix::_ldistribution[p - 1] = DMatrix::_n - accumulate;
        }
        else {
            if(p == 1)
                *DMatrix::_ldistribution = DMatrix::_n;
            else {
                std::fill_n(DMatrix::_ldistribution, p - 1, _local * (_sizeWorld / p - excluded));
                DMatrix::_ldistribution[p - 1] = DMatrix::_n - _local * (_sizeWorld / p - excluded) * (p - 1);
            }
        }
    }
#ifndef HPDDM_CONTIGUOUS
    else if(T == 1) {
        DMatrix::_idistribution = new int[DMatrix::_n];
        unsigned int j = 0;
        if(!excluded)
            for(unsigned int i = 0; i < p * (_sizeWorld / p); ++i) {
                unsigned int offset;
                if(i % (_sizeWorld / p) == 0) {
                    j = i / (_sizeWorld / p);
                    offset = U ? (_sizeWorld / p) * _local * j : (std::accumulate(info, info + j, 0) + std::accumulate(info + p, info + p + j * (_sizeWorld / p - 1), 0));
                }
                else {
                    j = p - 1 + i - i / (_sizeWorld / p);
                    offset  = U ? _local * (1 + i  / (_sizeWorld / p)) : std::accumulate(info, info + 1 + i / (_sizeWorld / p), 0);
                    offset += U ? (j - p) * _local : std::accumulate(info + p, info + j, 0);
                }
                std::iota(DMatrix::_idistribution + offset, DMatrix::_idistribution + offset + (U ? _local : info[j]), U ? _local * j : std::accumulate(info, info + j, 0));
                if(i % (_sizeWorld / p) != 0)
                    j = offset + (U ? _local : info[j]);
            }
        std::iota(DMatrix::_idistribution + j, DMatrix::_idistribution + DMatrix::_n, j);
        if(!U) {
            unsigned int accumulate = 0;
            for(unsigned short i = 0; i < p - 1; accumulate += DMatrix::_ldistribution[i++])
                DMatrix::_ldistribution[i] = std::accumulate(info + p + i * (_sizeWorld / p - 1), info + p + (i + 1) * (_sizeWorld / p - 1), info[i]);
            DMatrix::_ldistribution[p - 1] = DMatrix::_n - accumulate;
        }
        else {
            std::fill_n(DMatrix::_ldistribution, p - 1, _local * (_sizeWorld / p - excluded));
            DMatrix::_ldistribution[p - 1] = DMatrix::_n - _local * (_sizeWorld / p - excluded) * (p - 1);
        }
    }
#endif
    else if(T == 2) {
        if(!U) {
            unsigned int accumulate = 0;
            for(unsigned short i = 0; i < p - 1; accumulate += DMatrix::_ldistribution[i++])
                DMatrix::_ldistribution[i] = std::accumulate(info + DMatrix::_ldistribution[i], info + DMatrix::_ldistribution[i + 1], 0);
            DMatrix::_ldistribution[p - 1] = DMatrix::_n - accumulate;
        }
        else {
            for(unsigned short i = 0; i < p - 1; ++i)
                DMatrix::_ldistribution[i] = (DMatrix::_ldistribution[i + 1] - DMatrix::_ldistribution[i] - excluded) * _local;
            DMatrix::_ldistribution[p - 1] = DMatrix::_n - (DMatrix::_ldistribution[p - 1] - (excluded ? p - 1 : 0)) * _local;
        }
    }
}

template<template<class> class Solver, char S, class K>
template<unsigned short U, unsigned short excluded, class Operator>
inline std::pair<MPI_Request, const K*>* CoarseOperator<Solver, S, K>::construction(Operator&& v, const MPI_Comm& comm) {
    static_assert(super::_numbering == 'C' || super::_numbering == 'F', "Unknown numbering");
    static_assert(Operator::_pattern == 's' || Operator::_pattern == 'c', "Unknown pattern");
    constructionCommunicator<excluded != 0>(comm);
    if(excluded > 0 && DMatrix::_communicator != MPI_COMM_NULL) {
        int result;
        MPI_Comm_compare(v._p.getCommunicator(), DMatrix::_communicator, &result);
        if(result != MPI_CONGRUENT)
            std::cerr << "The communicators for the coarse operator don't match those of the domain decomposition" << std::endl;
    }
    if(Operator::_pattern == 'c')
        v.adjustConnectivity(_scatterComm);
    if(U == 2 && _local == 0)
        _offset = true;
    switch(Option::get()->val<char>("master_topology", 0)) {
#ifndef HPDDM_CONTIGUOUS
        case  1: return constructionMatrix<1, U, excluded>(v);
#endif
        case  2: return constructionMatrix<2, U, excluded>(v);
        default: return constructionMatrix<0, U, excluded>(v);
    }
}

template<template<class> class Solver, char S, class K>
template<char T, unsigned short U, unsigned short excluded, class Operator>
inline std::pair<MPI_Request, const K*>* CoarseOperator<Solver, S, K>::constructionMatrix(Operator& v) {
    unsigned short* const info = new unsigned short[(U != 1 ? 3 : 1) + v.getConnectivity()];
    const std::vector<unsigned short>& sparsity = v.getPattern();
    info[0] = sparsity.size(); // number of intersections
    int rank;
    MPI_Comm_rank(v._p.getCommunicator(), &rank);
    const unsigned short first = (S == 'S' ? std::distance(sparsity.cbegin(), std::upper_bound(sparsity.cbegin(), sparsity.cend(), rank)) : 0);
    int rankSplit;
    MPI_Comm_size(_scatterComm, &_sizeSplit);
    MPI_Comm_rank(_scatterComm, &rankSplit);
    unsigned short* infoNeighbor;

    unsigned int size = 0;
    int* I;
    int* J;
    K*   C;

    const Option& opt = *Option::get();
    const unsigned short p = opt.val<unsigned short>("master_p", 1);
    constexpr bool blocked =
#if defined(DMKL_PARDISO) || HPDDM_INEXACT_COARSE_OPERATOR
                             (U == 1 && Operator::_pattern == 's');
#else
                             false;
#endif
    unsigned short treeDimension = opt.val<unsigned short>("master_assembly_hierarchy"), currentHeight = 0;
    if(treeDimension <= 1 || treeDimension >= _sizeSplit)
        treeDimension = 0;
    unsigned short treeHeight = treeDimension ? std::ceil(std::log(_sizeSplit) / std::log(treeDimension)) : 0;
    std::vector<std::array<int, 3>>* msg = nullptr;
    if(rankSplit && treeDimension) {
        msg = new std::vector<std::array<int, 3>>();
        msg->reserve(treeHeight);
        int accumulate = 0, size;
        MPI_Comm_size(v._p.getCommunicator(), &size);
        int full = v._max;
        if(S != 'S')
            v._max = ((v._max & 4095) + 1) * pow(v._max >> 12, 2);
        for(unsigned short i = rankSplit; (i % treeDimension == 0) && currentHeight < treeHeight; i /= treeDimension) {
            const unsigned short bound = std::min(treeDimension, static_cast<unsigned short>(1 + ((_sizeSplit - rankSplit - 1) / pow(treeDimension, currentHeight)))) - 1;
            if(S == 'S')
                v._max = std::min(size - (rank + pow(treeDimension, currentHeight)), full & 4095) * pow(full >> 12, 2);
            for(unsigned short k = 0; k < bound; ++k) {
                msg->emplace_back(std::array<int, 3>({{ static_cast<int>(std::min(pow(treeDimension, currentHeight), static_cast<unsigned short>(_sizeSplit - (rankSplit + pow(treeDimension, currentHeight) * (k + 1)))) * v._max + (S == 'S' ? (!blocked ? ((full >> 12) * ((full >> 12) + 1)) / 2 : pow(full >> 12, 2)) : 0)), rankSplit + pow(treeDimension, currentHeight) * (k + 1), accumulate }}));
                accumulate += msg->back()[0];
            }
            ++currentHeight;
        }
    }
    if(U != 1) {
        infoNeighbor = new unsigned short[info[0]];
        info[1] = (excluded == 2 ? 0 : _local); // number of eigenvalues
        std::vector<MPI_Request> rqInfo;
        rqInfo.reserve(2 * info[0]);
        MPI_Request rq;
        if(excluded == 0) {
            if(T != 2) {
                for(unsigned short i = 0; i < info[0]; ++i)
                    if(!(T == 1 && sparsity[i] < p) &&
                       !(T == 0 && (sparsity[i] % (_sizeWorld / p) == 0) && sparsity[i] < p * (_sizeWorld / p))) {
                        MPI_Isend(info + 1, 1, MPI_UNSIGNED_SHORT, sparsity[i], 1, v._p.getCommunicator(), &rq);
                        rqInfo.emplace_back(rq);
                    }
            }
            else {
                for(unsigned short i = 0; i < info[0]; ++i)
                    if(!std::binary_search(DMatrix::_ldistribution, DMatrix::_ldistribution + p, sparsity[i])) {
                        MPI_Isend(info + 1, 1, MPI_UNSIGNED_SHORT, sparsity[i], 1, v._p.getCommunicator(), &rq);
                        rqInfo.emplace_back(rq);
                    }
            }
        }
        else if(excluded < 2)
            for(unsigned short i = 0; i < info[0]; ++i) {
                MPI_Isend(info + 1, 1, MPI_UNSIGNED_SHORT, sparsity[i], 1, v._p.getCommunicator(), &rq);
                rqInfo.emplace_back(rq);
            }
        if(rankSplit) {
            for(unsigned short i = 0; i < info[0]; ++i) {
                MPI_Irecv(infoNeighbor + i, 1, MPI_UNSIGNED_SHORT, sparsity[i], 1, v._p.getCommunicator(), &rq);
                rqInfo.emplace_back(rq);
            }
            size = (S != 'S' ? _local : 0);
            for(unsigned short i = 0; i < info[0]; ++i) {
                int index;
                MPI_Waitany(info[0], &rqInfo.back() - info[0] + 1, &index, MPI_STATUS_IGNORE);
                if(!(S == 'S' && sparsity[index] < rank))
                    size += infoNeighbor[index];
            }
            rqInfo.resize(rqInfo.size() - info[0]);
            size *= _local;
            if(S == 'S') {
                info[0] -= first;
                size += _local * (_local + 1) / 2;
            }
            info[2] = size;
            if(_local) {
                if(excluded == 0)
                    std::copy_n(sparsity.cbegin() + first, info[0], info + (U != 1 ? 3 : 1));
                else {
                    if(T != 1) {
                        for(unsigned short i = 0; i < info[0]; ++i) {
                            info[(U != 1 ? 3 : 1) + i] = sparsity[i + first] + 1;
                            for(unsigned short j = 0; j < p - 1 && info[(U != 1 ? 3 : 1) + i] >= (T == 0 ? (_sizeWorld / p) * (j + 1) : DMatrix::_ldistribution[j + 1]); ++j)
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
                size = _local * _local * info[0] + (!blocked ? _local * (_local + 1) / 2 : _local * _local);
            }
            else
                size = _local * _local * (1 + info[0]);
            std::copy_n(sparsity.cbegin() + first, info[0], info + (U != 1 ? 3 : 1));
        }
    }
    unsigned short** infoSplit;
    unsigned int*    offsetIdx;
    unsigned short*  infoWorld;
#if HPDDM_INEXACT_COARSE_OPERATOR
    unsigned short*  neighbors;
#endif
#ifdef HPDDM_CSR_CO
    unsigned int nrow;
    int* loc2glob;
#endif
    if(rankSplit)
        MPI_Gather(info, (U != 1 ? 3 : 1) + v.getConnectivity(), MPI_UNSIGNED_SHORT, NULL, 0, MPI_DATATYPE_NULL, 0, _scatterComm);
    else {
        size = 0;
        infoSplit = new unsigned short*[_sizeSplit];
        *infoSplit = new unsigned short[_sizeSplit * ((U != 1 ? 3 : 1) + v.getConnectivity()) + (U != 1) * _sizeWorld];
        MPI_Gather(info, (U != 1 ? 3 : 1) + v.getConnectivity(), MPI_UNSIGNED_SHORT, *infoSplit, (U != 1 ? 3 : 1) + v.getConnectivity(), MPI_UNSIGNED_SHORT, 0, _scatterComm);
        for(unsigned int i = 1; i < _sizeSplit; ++i)
            infoSplit[i] = *infoSplit + i * ((U != 1 ? 3 : 1) + v.getConnectivity());
        if(S == 'S' && Operator::_pattern == 's')
            **infoSplit -= first;
        offsetIdx = new unsigned int[std::max(_sizeSplit - 1, 2 * p)];
        if(U != 1) {
            infoWorld = *infoSplit + _sizeSplit * (3 + v.getConnectivity());
            int* recvcounts = reinterpret_cast<int*>(offsetIdx);
            int* displs = recvcounts + p;
            displs[0] = 0;
            if(T == 2) {
                std::adjacent_difference(DMatrix::_ldistribution + 1, DMatrix::_ldistribution + p, recvcounts);
                recvcounts[p - 1] = _sizeWorld - DMatrix::_ldistribution[p - 1];
            }
            else {
                std::fill_n(recvcounts, p - 1, _sizeWorld / p);
                recvcounts[p - 1] = _sizeWorld - (p - 1) * (_sizeWorld / p);
            }
            std::partial_sum(recvcounts, recvcounts + p - 1, displs + 1);
            for(unsigned int i = 0; i < _sizeSplit; ++i)
                infoWorld[displs[DMatrix::_rank] + i] = infoSplit[i][1];
#ifdef HPDDM_CSR_CO
            nrow = std::accumulate(infoWorld + displs[DMatrix::_rank], infoWorld + displs[DMatrix::_rank] + _sizeSplit, 0);
#endif
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, infoWorld, recvcounts, displs, MPI_UNSIGNED_SHORT, DMatrix::_communicator);
            if(T == 1) {
                unsigned int i = (p - 1) * (_sizeWorld / p);
                for(unsigned short k = p - 1, j = 1; k-- > 0; i -= _sizeWorld / p, ++j) {
                    recvcounts[k] = infoWorld[i];
                    std::copy_backward(infoWorld + k * (_sizeWorld / p), infoWorld + (k + 1) * (_sizeWorld / p), infoWorld + (k + 1) * (_sizeWorld / p) + j);
                }
                std::copy_n(recvcounts, p - 1, infoWorld + 1);
            }
            v._max = std::accumulate(infoWorld, infoWorld + _rankWorld, 0);
            DMatrix::_n = std::accumulate(infoWorld + _rankWorld, infoWorld + _sizeWorld, v._max);
            if(super::_numbering == 'F')
                ++v._max;
            unsigned short tmp = 0;
            for(unsigned short i = 0; i < info[0]; ++i) {
                infoNeighbor[i] = infoWorld[sparsity[i]];
                if(!(S == 'S' && i < first))
                    tmp += infoNeighbor[i];
            }
            for(unsigned short k = 1; k < _sizeSplit; size += infoSplit[k++][2])
                offsetIdx[k - 1] = size;
            if(excluded < 2)
                size += _local * tmp + (S == 'S' ? _local * (_local + 1) / 2 : _local * _local);
            if(S == 'S')
                info[0] -= first;
        }
        else {
            DMatrix::_n = (_sizeWorld - (excluded == 2 ? p : 0)) * _local;
            v._max = (_rankWorld - (excluded == 2 ? rank : 0)) * _local + (super::_numbering == 'F');
#ifdef HPDDM_CSR_CO
            nrow = (_sizeSplit - (excluded == 2)) * _local;
#endif
            if(S == 'S') {
                for(unsigned short i = 1; i < _sizeSplit; size += infoSplit[i++][0])
                    offsetIdx[i - 1] = size * _local * _local + (i - 1) * (!blocked ? _local * (_local + 1) / 2 : _local * _local);
                info[0] -= first;
                size = (size + info[0]) * _local * _local + (_sizeSplit - (excluded == 2)) * (!blocked ? _local * (_local + 1) / 2 : _local * _local);
            }
            else {
                for(unsigned short i = 1; i < _sizeSplit; size += infoSplit[i++][0])
                    offsetIdx[i - 1] = (i - 1 + size) * _local * _local;
                size = (size + info[0] + _sizeSplit - (excluded == 2)) * _local * _local;
            }
            if(_sizeSplit == 1)
                offsetIdx[0] = size;
        }
#if HPDDM_INEXACT_COARSE_OPERATOR
        neighbors = new unsigned short[size / (!blocked ? 1 : _local * _local)];
        if(T == 1)
            for(unsigned short i = 1; i < p; ++i)
                DMatrix::_ldistribution[i] = (excluded == 2 ? 0 : p) + i * ((_sizeWorld / p) - 1);
        else if(excluded == 2)
            for(unsigned short i = 1; i < p; ++i)
                DMatrix::_ldistribution[i] -= i;
#endif
#ifdef HPDDM_CSR_CO
        I = new int[(!blocked ? nrow + size : (nrow / _local + size / (_local * _local))) + 1];
        J = I + 1 + nrow / (!blocked ? 1 : _local);
        I[0] = (super::_numbering == 'F');
#ifndef HPDDM_CONTIGUOUS
        loc2glob = new int[nrow];
#else
        loc2glob = new int[2];
#endif
#else
        I = new int[2 * size];
        J = I + size;
#endif
        C = new K[!std::is_same<downscaled_type<K>, K>::value ? std::max((info[0] + 1) * _local * _local, static_cast<int>(1 + ((size * sizeof(downscaled_type<K>) - 1) / sizeof(K)))) : size];
    }
    const vectorNeighbor& M = v._p.getMap();

    MPI_Request* rqSend = v._p.getRq();
    MPI_Request* rqRecv;
    MPI_Request* rqTree = treeDimension ? new MPI_Request[rankSplit ? msg->size() : (treeHeight * (treeDimension - 1))] : nullptr;

    K** sendNeighbor = v._p.getBuffer();
    K** recvNeighbor;
    int coefficients = (U == 1 ? _local * (info[0] + (S != 'S' || blocked)) : std::accumulate(infoNeighbor + first, infoNeighbor + sparsity.size(), S == 'S' ? 0 : _local));
    K* work = nullptr;
    if(Operator::_pattern == 's') {
        rqRecv = (rankSplit == 0 && !treeDimension ? new MPI_Request[_sizeSplit - 1 + info[0]] : rqSend + (S != 'S' ? info[0] : first));
        unsigned int accumulate = 0;
        for(unsigned short i = 0; i < (S != 'S' ? info[0] : first); ++i)
            if(U == 1 || infoNeighbor[i])
                accumulate += _local * M[i].second.size();
        if(U == 1 || _local)
            for(unsigned short i = 0; i < info[0]; ++i)
                accumulate += (U == 1 ? _local : infoNeighbor[i + first]) * M[i + first].second.size();
        if(excluded < 2)
            *sendNeighbor = new K[accumulate];
        accumulate = 0;
        for(unsigned short i = 0; i < (S != 'S' ? info[0] : first); ++i) {
            sendNeighbor[i] = *sendNeighbor + accumulate;
            if(U == 1 || infoNeighbor[i])
                accumulate += _local * M[i].second.size();
        }
        if(rankSplit) {
            const unsigned int tmp = treeDimension && !msg->empty() ? size + (!std::is_same<downscaled_type<K>, K>::value ? 1 + (((msg->back()[0] + msg->back()[2]) * sizeof(downscaled_type<K>) - 1) / sizeof(K)) : (msg->back()[0] + msg->back()[2])) : size;
            if(!excluded && tmp <= accumulate)
                C = *sendNeighbor;
            else
                C = new K[tmp];
        }
        recvNeighbor = (U == 1 || _local ? sendNeighbor + (S != 'S' ? info[0] : first) : nullptr);
        if(U == 1 || _local) {
            for(unsigned short i = 0; i < info[0]; ++i) {
                recvNeighbor[i] = *sendNeighbor + accumulate;
                MPI_Irecv(recvNeighbor[i], (U == 1 ? _local : infoNeighbor[i + first]) * M[i + first].second.size(), Wrapper<K>::mpi_type(), M[i + first].first, 2, v._p.getCommunicator(), rqRecv + i);
                accumulate += (U == 1 ? _local : infoNeighbor[i + first]) * M[i + first].second.size();
            }
        }
        else
            std::fill_n(rqRecv, info[0], MPI_REQUEST_NULL);
        if(excluded < 2) {
            const K* const* const& EV = v._p.getVectors();
            const int n = v._p.getDof();
            v.initialize(n * (U == 1 || info[0] == 0 ? _local : std::max(static_cast<unsigned short>(_local), *std::max_element(infoNeighbor + first, infoNeighbor + sparsity.size()))), work, S != 'S' ? info[0] : first);
            v.template applyToNeighbor<S, U == 1>(sendNeighbor, work, rqSend, infoNeighbor);
            if(S != 'S') {
                unsigned short before = 0;
                for(unsigned short j = 0; j < info[0] && sparsity[j] < rank; ++j)
                    before += (U == 1 ? (!blocked ? _local : 1) : infoNeighbor[j]);
                Blas<K>::gemm(&(Wrapper<K>::transc), "N", &_local, &_local, &n, &(Wrapper<K>::d__1), work, &n, *EV, &n, &(Wrapper<K>::d__0), C + before * (!blocked ? 1 : _local * _local), !blocked ? &coefficients : &_local);
                Wrapper<K>::template imatcopy<'R'>(_local, _local, C + before * (!blocked ? 1 : _local * _local), !blocked ? coefficients : _local, !blocked ? coefficients : _local);
                if(rankSplit == 0) {
                    if(!blocked)
                        for(unsigned short j = 0; j < _local; ++j) {
#ifndef HPDDM_CSR_CO
                            std::fill_n(I + before + j * coefficients, _local, v._max + j);
#endif
                            std::iota(J + before + j * coefficients, J + before + j * coefficients + _local, v._max);
#if HPDDM_INEXACT_COARSE_OPERATOR
                            std::fill_n(neighbors + before + j * coefficients, _local, DMatrix::_rank);
#endif
                        }
                    else {
                        J[before] = _rankWorld - (excluded == 2 ? rank : 0) + (super::_numbering == 'F');
#if HPDDM_INEXACT_COARSE_OPERATOR
                        neighbors[before] = DMatrix::_rank;
#endif
                    }
                }
            }
            else {
                if(blocked || coefficients >= _local) {
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &_local, &_local, &n, &(Wrapper<K>::d__1), *EV, &n, work, &n, &(Wrapper<K>::d__0), C, &_local);
                    if(!blocked)
                        for(unsigned short j = _local; j-- > 0; )
                            std::copy_backward(C + j * (_local + 1), C + (j + 1) * _local, C - (j * (j + 1)) / 2 + j * coefficients + (j + 1) * _local);
                }
                else
                    for(unsigned short j = 0; j < _local; ++j) {
                        int local = _local - j;
                        Blas<K>::gemv(&(Wrapper<K>::transc), &n, &local, &(Wrapper<K>::d__1), EV[j], &n, work + n * j, &i__1, &(Wrapper<K>::d__0), C - (j * (j - 1)) / 2 + j * (coefficients + _local), &i__1);
                    }
                if(rankSplit == 0) {
                    if(!blocked)
                        for(unsigned short j = _local; j-- > 0; ) {
#ifndef HPDDM_CSR_CO
                            std::fill_n(I + j * (coefficients + _local) - (j * (j - 1)) / 2, _local - j, v._max + j);
#endif
                            std::iota(J + j * (coefficients + _local - 1) - (j * (j - 1)) / 2 + j, J + j * (coefficients + _local - 1) - (j * (j - 1)) / 2 + _local, v._max + j);
#if HPDDM_INEXACT_COARSE_OPERATOR
                            std::fill(neighbors + j * (coefficients + _local - 1) - (j * (j - 1)) / 2 + j, neighbors + j * (coefficients + _local - 1) - (j * (j - 1)) / 2 + _local, DMatrix::_rank);
#endif
                        }
                    else {
                        *J = _rankWorld - (excluded == 2 ? rank : 0) + (super::_numbering == 'F');
#if HPDDM_INEXACT_COARSE_OPERATOR
                        *neighbors = DMatrix::_rank;
#endif
                    }
                }
            }
        }
    }
    else {
        rqRecv = (rankSplit == 0 && !treeDimension ? new MPI_Request[_sizeSplit - 1 + M.size()] : (rqSend + M.size()));
        recvNeighbor = (U == 1 || _local) ? sendNeighbor + M.size() : nullptr;
        if(excluded < 2)
            v.template applyToNeighbor<S, U == 1>(sendNeighbor, work, rqSend, U == 1 ? nullptr : infoNeighbor, recvNeighbor, rqRecv);
        if(rankSplit)
            C = new K[treeDimension && !msg->empty() ? (size + msg->back()[0] + msg->back()[2]) : size];
    }
    std::pair<MPI_Request, const K*>* ret = nullptr;
    if(rankSplit) {
        downscaled_type<K>* const pt = reinterpret_cast<downscaled_type<K>*>(C);
        if(treeDimension) {
            for(const std::array<int, 3>& m : *msg)
                MPI_Irecv(pt + size + m[2], m[0], Wrapper<downscaled_type<K>>::mpi_type(), m[1], 3, _scatterComm, rqTree++);
            rqTree -= msg->size();
        }
        if(U == 1 || _local) {
            if(Operator::_pattern == 's') {
                unsigned int* offsetArray = new unsigned int[info[0]];
                if(S != 'S')
                    offsetArray[0] = M[0].first > rank ? _local : 0;
                else if(info[0])
                    offsetArray[0] = _local;
                for(unsigned short k = 1; k < info[0]; ++k) {
                    offsetArray[k] = offsetArray[k - 1] + (U == 1 ? _local : infoNeighbor[k - 1 + first]);
                    if(S != 'S' && sparsity[k - 1] < rank && sparsity[k] > rank)
                        offsetArray[k] += _local;
                }
                for(unsigned short k = 0; k < info[0]; ++k) {
                    int index;
                    MPI_Waitany(info[0], rqRecv, &index, MPI_STATUS_IGNORE);
                    v.template assembleForMaster<!blocked ? S : 'B', U == 1>(C + offsetArray[index], recvNeighbor[index], coefficients + (S == 'S' && !blocked ? _local - 1 : 0), index + first, blocked && super::_numbering == 'F' ? C + offsetArray[index] * _local : work, infoNeighbor + first + index);
                    if(blocked && super::_numbering == 'C')
                        Wrapper<K>::template omatcopy<'T'>(_local, _local, work, _local, C + offsetArray[index] * _local, _local);
                }
                delete [] offsetArray;
            }
            else {
                for(unsigned short k = 0; k < M.size(); ++k) {
                    int index;
                    MPI_Waitany(M.size(), rqRecv, &index, MPI_STATUS_IGNORE);
                    v.template assembleForMaster<S, U == 1>(C, recvNeighbor[index], coefficients, index, work, infoNeighbor);
                }
            }
            if(excluded)
                ret = new std::pair<MPI_Request, const K*>(MPI_REQUEST_NULL, C);
            if(!std::is_same<downscaled_type<K>, K>::value)
                for(unsigned int i = 0; i < size; ++i)
                    pt[i] = C[i];
            if(!treeDimension) {
                if(excluded)
                    MPI_Isend(pt, size, Wrapper<downscaled_type<K>>::mpi_type(), 0, 3, _scatterComm, &ret->first);
                else
                    MPI_Send(pt, size, Wrapper<downscaled_type<K>>::mpi_type(), 0, 3, _scatterComm);
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
                (*msg)[0][0] = (*msg)[0][1];
                for(unsigned short i = 1; i < msg->size(); ++i) {
                    if((*msg)[i][2] != (*msg)[i - 1][0])
                        std::copy(pt + size + (*msg)[i][2], pt + size + (*msg)[i][2] + (*msg)[i][1], pt + size + (*msg)[i - 1][0]);
                    (*msg)[i][0] = (*msg)[i - 1][0] + (*msg)[i][1];
                }
                size += msg->back()[0];
            }
            delete [] rqTree;
            if(size) {
                if(excluded)
                    MPI_Isend(pt, size, Wrapper<downscaled_type<K>>::mpi_type(), pow(treeDimension, currentHeight + 1) * (rankSplit / pow(treeDimension, currentHeight + 1)), 3, _scatterComm, &ret->first);
                else
                    MPI_Send(pt, size, Wrapper<downscaled_type<K>>::mpi_type(), pow(treeDimension, currentHeight + 1) * (rankSplit / pow(treeDimension, currentHeight + 1)), 3, _scatterComm);
            }
            delete msg;
        }
        if(!excluded && C != *sendNeighbor)
            delete [] C;
        delete [] info;
        _sizeRHS = _local;
        if(U != 1)
            delete [] infoNeighbor;
        if(U == 0)
            DMatrix::_displs = &_rankWorld;
        int nbRq = std::distance(v._p.getRq(), rqSend);
        MPI_Waitall(nbRq, rqSend - nbRq, MPI_STATUSES_IGNORE);
        delete [] work;
    }
    else {
        const unsigned short relative = (T == 1 ? p + _rankWorld * ((_sizeWorld / p) - 1) - 1 : _rankWorld);
        unsigned int* offsetPosition;
        if(excluded < 2)
            std::for_each(offsetIdx, offsetIdx + _sizeSplit - 1, [&](unsigned int& i) { i += coefficients * _local + (S == 'S' && !blocked) * (_local * (_local + 1)) / 2; });
        K* const backup = std::is_same<downscaled_type<K>, K>::value ? C : new K[offsetIdx[0]];
        if(!std::is_same<downscaled_type<K>, K>::value)
            std::copy_n(C, offsetIdx[0], backup);
        if(!treeDimension) {
            if(excluded < 2)
                treeHeight = Operator::_pattern == 's' ? info[0] : M.size();
            else
                treeHeight = 0;
            for(unsigned short k = 1; k < _sizeSplit; ++k) {
                if(U != 1) {
                    if(infoSplit[k][2])
                        MPI_Irecv(reinterpret_cast<downscaled_type<K>*>(C) + offsetIdx[k - 1], infoSplit[k][2], Wrapper<downscaled_type<K>>::mpi_type(), k, 3, _scatterComm, rqRecv + treeHeight + k - 1);
                    else
                        rqRecv[treeHeight + k - 1] = MPI_REQUEST_NULL;
                }
                else
                    MPI_Irecv(reinterpret_cast<downscaled_type<K>*>(C) + offsetIdx[k - 1], _local * _local * infoSplit[k][0] + (S == 'S' && !blocked ? _local * (_local + 1) / 2 : _local * _local), Wrapper<downscaled_type<K>>::mpi_type(), k, 3, _scatterComm, rqRecv + treeHeight + k - 1);
            }
        }
        else {
            std::fill_n(rqTree, treeHeight * (treeDimension - 1), MPI_REQUEST_NULL);
            for(unsigned short i = 0; i < treeHeight; ++i) {
                const unsigned short leaf = pow(treeDimension, i);
                const unsigned short bound = std::min(treeDimension, static_cast<unsigned short>(1 + ((_sizeSplit - 1) / leaf))) - 1;
                for(unsigned short k = 0; k < bound; ++k) {
                    const unsigned short nextLeaf = std::min(leaf * (k + 1) * treeDimension, _sizeSplit);
                    int nnz = 0;
                    for(unsigned short j = leaf * (k + 1); j < nextLeaf; ++j)
                        nnz += infoSplit[j][U != 1 ? 2 : 0];
                    if(U != 1) {
                        if(nnz)
                            MPI_Irecv(reinterpret_cast<downscaled_type<K>*>(C) + offsetIdx[leaf * (k + 1) - 1], infoSplit[leaf * (k + 1)][2], Wrapper<downscaled_type<K>>::mpi_type(), leaf * (k + 1), 3, _scatterComm, rqTree + i * (treeDimension - 1) + k);
                    }
                    else
                        MPI_Irecv(reinterpret_cast<downscaled_type<K>*>(C) + offsetIdx[leaf * (k + 1) - 1], _local * _local * nnz + (S == 'S' && !blocked ? _local * (_local + 1) / 2 : _local * _local) * (nextLeaf - leaf), Wrapper<downscaled_type<K>>::mpi_type(), leaf * (k + 1), 3, _scatterComm, rqTree + i * (treeDimension - 1) + k);
                }
            }
        }
        if(U != 1) {
            offsetPosition = new unsigned int[_sizeSplit];
            offsetPosition[0] = std::accumulate(infoWorld, infoWorld + relative, static_cast<unsigned int>(super::_numbering == 'F'));
            if(T != 1)
                for(unsigned int k = 1; k < _sizeSplit; ++k)
                    offsetPosition[k] = offsetPosition[k - 1] + infoSplit[k - 1][1];
            else
                for(unsigned int k = 1; k < _sizeSplit; ++k)
                    offsetPosition[k] = offsetPosition[k - 1] + infoWorld[relative + k - 1];
        }
        if(blocked)
            std::for_each(offsetIdx, offsetIdx + _sizeSplit - 1, [&](unsigned int& i) { i /= _local * _local; });
#ifdef _OPENMP
#pragma omp parallel for shared(I, J, infoWorld, infoSplit, relative, offsetIdx, offsetPosition) schedule(dynamic, 64)
#endif
        for(unsigned int k = 1; k < _sizeSplit; ++k) {
            if(U == 1 || infoSplit[k][2]) {
                unsigned int offsetSlave = static_cast<unsigned int>(super::_numbering == 'F');
                if(U != 1 && infoSplit[k][0])
                    offsetSlave = std::accumulate(infoWorld, infoWorld + infoSplit[k][3], offsetSlave);
                unsigned short i = 0;
                int* colIdx = J + offsetIdx[k - 1];
#if HPDDM_INEXACT_COARSE_OPERATOR
                unsigned short* nghbrs = neighbors + offsetIdx[k - 1];
#endif
                const unsigned short max = relative + k - (U == 1 && excluded == 2 ? (T == 1 ? p : 1 + rank) : 0);
                const unsigned int tmp = (U == 1 ? max * (!blocked ? _local : 1) + (super::_numbering == 'F') : offsetPosition[k]);
                if(S != 'S')
                    while(i < infoSplit[k][0] && infoSplit[k][(U != 1 ? 3 : 1) + i] < max) {
#if HPDDM_INEXACT_COARSE_OPERATOR
                        if(T == 1 && infoSplit[k][(U != 1 ? 3 : 1) + i] < p)
                            *nghbrs = infoSplit[k][(U != 1 ? 3 : 1) + i];
                        else
                            *nghbrs = std::distance(DMatrix::_ldistribution + 1, std::upper_bound(DMatrix::_ldistribution + 1, DMatrix::_ldistribution + DMatrix::_rank + 1, infoSplit[k][(U != 1 ? 3 : 1) + i]));
                        if(*nghbrs != DMatrix::_rank && ((T != 1 && (i == 0 || *nghbrs != *(nghbrs - 1))) || (T == 1 && !std::binary_search(super::_send[*nghbrs].cbegin(), super::_send[*nghbrs].cend(), tmp - (super::_numbering == 'F')))))
                            for(unsigned short row = 0; row < (U == 1 ? (!blocked ? _local : 1) : infoSplit[k][1]); ++row)
                                super::_send[*nghbrs].emplace_back(tmp - (super::_numbering == 'F') + row);
#endif
                        if(!blocked) {
                            if(U != 1) {
                                if(i > 0)
                                    offsetSlave = std::accumulate(infoWorld + infoSplit[k][2 + i], infoWorld + infoSplit[k][3 + i], offsetSlave);
                            }
                            else
                                offsetSlave = infoSplit[k][1 + i] * _local + (super::_numbering == 'F');
                            std::iota(colIdx, colIdx + (U == 1 ? _local : infoWorld[infoSplit[k][3 + i]]), offsetSlave);
                            colIdx += (U == 1 ? _local : infoWorld[infoSplit[k][3 + i]]);
#if HPDDM_INEXACT_COARSE_OPERATOR
                            std::fill_n(nghbrs + 1, (U == 1 ? _local : infoWorld[infoSplit[k][3 + i]]) - 1, *nghbrs);
                            nghbrs += (U == 1 ? _local : infoWorld[infoSplit[k][3 + i]]);
#endif
                        }
                        else {
                            *colIdx++ = infoSplit[k][1 + i] + (super::_numbering == 'F');
#if HPDDM_INEXACT_COARSE_OPERATOR
                            ++nghbrs;
#endif
                        }
                        ++i;
                    }
                if(!blocked) {
                    std::iota(colIdx, colIdx + (U == 1 ? _local : infoSplit[k][1]), tmp);
                    colIdx += (U == 1 ? _local : infoSplit[k][1]);
#if HPDDM_INEXACT_COARSE_OPERATOR
                    std::fill_n(nghbrs, (U == 1 ? _local : infoSplit[k][1]), DMatrix::_rank);
                    nghbrs += (U == 1 ? _local : infoSplit[k][1]);
#endif
                }
                else {
                    *colIdx++ = tmp;
#if HPDDM_INEXACT_COARSE_OPERATOR
                    *nghbrs++ = DMatrix::_rank;
#endif
                }
                while(i < infoSplit[k][0]) {
#if HPDDM_INEXACT_COARSE_OPERATOR
                    *nghbrs = std::distance(DMatrix::_ldistribution + 1, std::upper_bound(DMatrix::_ldistribution + DMatrix::_rank + 1, DMatrix::_ldistribution + p, infoSplit[k][(U != 1 ? 3 : 1) + i]));
                    if(S != 'S' && *nghbrs != DMatrix::_rank && ((T != 1 && (i == 0 || *nghbrs != *(nghbrs - 1))) || (T == 1 && !std::binary_search(super::_send[*nghbrs].cbegin(), super::_send[*nghbrs].cend(), tmp - (super::_numbering == 'F')))))
                        for(unsigned short row = 0; row < (U == 1 ? (!blocked ? _local : 1) : infoSplit[k][1]); ++row)
                            super::_send[*nghbrs].emplace_back(tmp - (super::_numbering == 'F') + row);
#endif
                    if(!blocked) {
                        if(U != 1) {
                            if(i > 0)
                                offsetSlave = std::accumulate(infoWorld + infoSplit[k][2 + i], infoWorld + infoSplit[k][3 + i], offsetSlave);
                        }
                        else
                            offsetSlave = infoSplit[k][1 + i] * _local + (super::_numbering == 'F');
                        std::iota(colIdx, colIdx + (U == 1 ? _local : infoWorld[infoSplit[k][3 + i]]), offsetSlave);
                        colIdx += (U == 1 ? _local : infoWorld[infoSplit[k][3 + i]]);
#if HPDDM_INEXACT_COARSE_OPERATOR
                        std::fill_n(nghbrs + 1, (U == 1 ? _local : infoWorld[infoSplit[k][3 + i]]) - 1, *nghbrs);
                        nghbrs += (U == 1 ? _local : infoWorld[infoSplit[k][3 + i]]);
#endif
                    }
                    else {
                        *colIdx++ = infoSplit[k][1 + i] + (super::_numbering == 'F');
#if HPDDM_INEXACT_COARSE_OPERATOR
                        ++nghbrs;
#endif
                    }
                    ++i;
                }
                unsigned int coefficientsSlave = colIdx - J - offsetIdx[k - 1];
#ifndef HPDDM_CSR_CO
                int* rowIdx = I + std::distance(J, colIdx);
                std::fill(I + offsetIdx[k - 1], rowIdx, tmp);
#else
                offsetSlave = (U == 1 ? (k - (excluded == 2)) * (!blocked ? _local : 1) : offsetPosition[k] - offsetPosition[1] + (excluded == 2 ? 0 : _local));
                I[offsetSlave + 1] = coefficientsSlave;
#ifndef HPDDM_CONTIGUOUS
                loc2glob[offsetSlave] = tmp;
#endif
#endif
                if(!blocked)
                    for(i = 1; i < (U == 1 ? _local : infoSplit[k][1]); ++i) {
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
#ifdef HPDDM_CONTIGUOUS
                if(excluded == 2 && k == 1)
                    loc2glob[0] = tmp;
                if(k == _sizeSplit - 1)
                    loc2glob[1] = tmp + (!blocked ? (U == 1 ? _local : infoSplit[k][1]) - 1 : 0);
#endif
            }
        }
        if(std::is_same<downscaled_type<K>, K>::value)
            delete [] offsetIdx;
        if(excluded < 2) {
#ifdef HPDDM_CSR_CO
            if(!blocked) {
                I[1] = coefficients + (S == 'S' ? _local : 0);
                for(unsigned short k = 1; k < _local; ++k) {
                    I[k + 1] = coefficients + (S == 'S' ? _local - k : 0);
#ifndef HPDDM_CONTIGUOUS
                    loc2glob[k] = v._max + k;
#endif
                }
            }
            else
                I[1] = info[0] + 1;
            loc2glob[0] = ((!blocked || _local == 1) ? v._max : v._max / _local + (super::_numbering == 'F'));
#ifdef HPDDM_CONTIGUOUS
            if(_sizeSplit == 1)
                loc2glob[1] = ((!blocked || _local == 1) ? v._max + _local - 1 : v._max / _local + (super::_numbering == 'F'));
#endif
#endif
            unsigned int** offsetArray = new unsigned int*[info[0]];
            *offsetArray = new unsigned int[info[0] * ((Operator::_pattern == 's') + (U != 1))];
            if(Operator::_pattern == 's') {
                if(S != 'S') {
                    offsetArray[0][0] = sparsity[0] > _rankWorld ? _local : 0;
                    if(U != 1)
                        offsetArray[0][1] = std::accumulate(infoWorld, infoWorld + sparsity[0], static_cast<unsigned int>(super::_numbering == 'F'));
                }
                else if(info[0]) {
                    offsetArray[0][0] = _local;
                    if(U != 1)
                        offsetArray[0][1] = std::accumulate(infoWorld, infoWorld + sparsity[first], static_cast<unsigned int>(super::_numbering == 'F'));
                }
                for(unsigned short k = 1; k < info[0]; ++k) {
                    if(U != 1) {
                        offsetArray[k] = *offsetArray + 2 * k;
                        offsetArray[k][1] = std::accumulate(infoWorld + sparsity[first + k - 1], infoWorld + sparsity[first + k], offsetArray[k - 1][1]);
                        offsetArray[k][0] = offsetArray[k - 1][0] + infoNeighbor[k - 1 + first];
                    }
                    else {
                        offsetArray[k] = *offsetArray + k;
                        offsetArray[k][0] = offsetArray[k - 1][0] + _local;
                    }
                    if(S != 'S' && sparsity[k - 1] < _rankWorld && sparsity[k] > _rankWorld)
                        offsetArray[k][0] += _local;
                }
            }
            else {
                if(U != 1) {
                    if(S != 'S')
                        offsetArray[0][0] = std::accumulate(infoWorld, infoWorld + sparsity[0], static_cast<unsigned int>(super::_numbering == 'F'));
                    else if(info[0])
                        offsetArray[0][0] = std::accumulate(infoWorld, infoWorld + sparsity[first], static_cast<unsigned int>(super::_numbering == 'F'));
                    for(unsigned short k = 1; k < info[0]; ++k) {
                        offsetArray[k] = *offsetArray + k;
                        offsetArray[k][0] = std::accumulate(infoWorld + sparsity[first + k - 1], infoWorld + sparsity[first + k], offsetArray[k - 1][0]);
                    }
                }
                info[0] = M.size();
            }
            if(U == 1 || _local) {
                for(unsigned int k = 0; k < info[0]; ++k) {
                    int index;
                    MPI_Waitany(info[0], rqRecv, &index, MPI_STATUS_IGNORE);
                    if(Operator::_pattern == 's') {
                        const unsigned int offset = offsetArray[index][0] / (!blocked ? 1 : _local);
                        v.template applyFromNeighborMaster<!blocked ? S : 'B', super::_numbering, U == 1>(recvNeighbor[index], index + first, I + offset, J + offset, backup + offsetArray[index][0] * (!blocked ? 1 : _local), coefficients + (S == 'S' && !blocked) * (_local - 1), v._max, U == 1 ? nullptr : (offsetArray[index] + 1), work, U == 1 ? nullptr : infoNeighbor + first + index);
#if HPDDM_INEXACT_COARSE_OPERATOR
                        if(T == 1 && M[first + index].first < p)
                            neighbors[offset] = M[first + index].first;
                        else
                            neighbors[offset] = std::distance(DMatrix::_ldistribution + 1, std::upper_bound(DMatrix::_ldistribution + 1, DMatrix::_ldistribution + p, M[first + index].first));
                        if(S != 'S' && neighbors[offset] != DMatrix::_rank && (super::_send[neighbors[offset]].empty() || super::_send[neighbors[offset]].back() != ((v._max - (super::_numbering == 'F')) / (!blocked ? 1 : _local) + (!blocked ? _local : 1) - 1)))
                            for(unsigned short i = 0; i < (!blocked ? _local : 1); ++i)
                                super::_send[neighbors[offset]].emplace_back((v._max - (super::_numbering == 'F')) / (!blocked ? 1 : _local) + i);
                        if(!blocked)
                            for(unsigned short i = 0; i < _local; ++i)
                                std::fill_n(neighbors + offset + (coefficients + (S == 'S') * (_local - 1)) * i - (S == 'S') * (i * (i - 1)) / 2, U == 1 ? _local : infoNeighbor[first + index], neighbors[offset]);
#endif
                    }
                    else
                        v.template applyFromNeighborMaster<S, super::_numbering, U == 1>(recvNeighbor[index], index, I, J, backup, coefficients, v._max, U == 1 ? nullptr : *offsetArray, work, U == 1 ? nullptr : infoNeighbor);
                }
                downscaled_type<K>* pt = reinterpret_cast<downscaled_type<K>*>(C);
                if(!std::is_same<downscaled_type<K>, K>::value) {
                    if(blocked && _sizeSplit > 1)
                        offsetIdx[0] *= _local * _local;
                    for(unsigned int i = 0; i < offsetIdx[0]; ++i)
                        pt[i] = backup[i];
                    delete [] backup;
                }
            }
            delete [] *offsetArray;
            delete [] offsetArray;
        }
#ifdef HPDDM_CONTIGUOUS
        else if(_sizeSplit == 1) {
            loc2glob[0] = 2;
            loc2glob[1] = 1;
        }
#endif
        if(!std::is_same<downscaled_type<K>, K>::value)
            delete [] offsetIdx;
        delete [] info;
        if(!treeDimension)
            MPI_Waitall(_sizeSplit - 1, rqRecv + treeHeight, MPI_STATUSES_IGNORE);
        else {
            MPI_Waitall(treeHeight * (treeDimension - 1), rqTree, MPI_STATUSES_IGNORE);
            delete [] rqTree;
        }
        if(U != 1) {
            delete [] infoNeighbor;
            delete [] offsetPosition;
        }
        delete [] work;
        downscaled_type<K>* pt = reinterpret_cast<downscaled_type<K>*>(C);
        std::string filename = opt.prefix("master_dump_matrix", true);
        if(filename.size() > 0) {
            if(excluded == 2)
                filename += "_excluded";
            std::ofstream output { filename + "_" + S + "_" + super::_numbering + "_" + to_string(T) + "_" + to_string(DMatrix::_rank) + ".txt" };
            output << std::scientific;
#ifndef HPDDM_CSR_CO
            for(unsigned int i = 0; i < size; ++i)
                output << std::setw(9) << I[i] + (super::_numbering == 'C') << std::setw(9) << J[i] + (super::_numbering == 'C') << " " << pt[i] << std::endl;
#else
            unsigned int accumulate = 0;
            for(unsigned int i = 0; i < nrow / (!blocked ? 1 : _local); ++i) {
                accumulate += I[i];
                for(unsigned int j = 0; j < I[i + 1]; ++j) {
                    output << std::setw(9) <<
#ifndef HPDDM_CONTIGUOUS
                    (loc2glob[i] - (super::_numbering == 'F')) * (!blocked ? 1 : _local) + 1 <<
#else
                    (loc2glob[0] + i - (super::_numbering == 'F')) * (!blocked ? 1 : _local) + 1 <<
#endif
                    std::setw(9) << (J[accumulate + j - (super::_numbering == 'F')] - (super::_numbering == 'F')) * (!blocked ? 1 : _local) + 1 << " ";
                    if(!blocked)
                        output << std::setw(13) << pt[accumulate + j - (super::_numbering == 'F')] << "\n";
                    else {
                        for(unsigned short b = 0; b < _local; ++b) {
                            if(b)
                                output << "                   ";
                            for(unsigned short c = 0; c < _local; ++c) {
                                output << std::setw(13) << pt[(accumulate + j - (super::_numbering == 'F')) * _local * _local + (super::_numbering == 'C' ? b * _local + c : b + c * _local)] << "  ";
                            }
                            output << "\n";
                        }
                    }
                }
            }
#endif
        }
#if HPDDM_INEXACT_COARSE_OPERATOR
        if(S != 'S') {
            int* backup = new int[!blocked ? _local : 1];
            for(std::pair<const unsigned short, std::vector<int>>& i : super::_send) {
                if(i.second.size() > (!blocked ? _local : 1) && *(i.second.end() - (!blocked ? _local : 1) - 1) > *(i.second.end() - (!blocked ? _local : 1))) {
                    std::vector<int>::iterator it = std::lower_bound(i.second.begin(), i.second.end(), *(i.second.end() - (!blocked ? _local : 1)));
                    std::copy(i.second.end() - (!blocked ? _local : 1), i.second.end(), backup);
                    std::copy_backward(it, i.second.end() - (!blocked ? _local : 1), i.second.end());
                    std::copy_n(backup, !blocked ? _local : 1, it);
                }
            }
            delete [] backup;
        }
        super::_mu = std::min(p, opt.val<unsigned short>("master_aggregate_sizes", p));
        rank = DMatrix::_n;
        if(super::_mu < p) {
            super::_di = new int[T == 1 ? 3 : 1];
            unsigned int begin = (DMatrix::_rank / super::_mu) * super::_mu;
            unsigned int end = std::min(p, static_cast<unsigned short>(begin + super::_mu));
            if(T == 1) {
                super::_di[0] = begin;
                super::_di[1] = end;
                super::_di[2] = p + begin * ((_sizeWorld / p) - 1);
                if(end == p)
                    end = _sizeWorld;
                else
                    end = p + end * ((_sizeWorld / p) - 1);
            }
            else {
                super::_di[0] = DMatrix::_ldistribution[begin];
                if(end == p)
                    end = _sizeWorld - (excluded == 2 ? p : 0);
                else
                    end = DMatrix::_ldistribution[end];
            }
            if(!U) {
                DMatrix::_n = std::accumulate(infoWorld + super::_di[0], infoWorld + (T == 1 ? super::_di[1] : end), 0);
                super::_di[0] = std::accumulate(infoWorld, infoWorld + super::_di[0], 0);
                if(T == 1) {
                    begin = super::_di[1];
                    super::_di[1] = super::_di[0] + DMatrix::_n;
                    DMatrix::_n = std::accumulate(infoWorld + super::_di[2], infoWorld + end, DMatrix::_n);
                    super::_di[2] = std::accumulate(infoWorld + begin, infoWorld + super::_di[2], super::_di[1]);
                }
            }
            else {
                DMatrix::_n = ((T == 1 ? super::_di[1] : end) - super::_di[0]) * _local;
                if(T == 1)
                    DMatrix::_n += (end - super::_di[2]) * _local;
            }
        }
        super::_bs = (!blocked ? 1 : _local);
        super::template numfact<T>(nrow / (!blocked ? 1 : _local), I, loc2glob, J, pt, neighbors);
        std::swap(DMatrix::_n, rank);
        if(T == 1)
            std::iota(DMatrix::_ldistribution + 1, DMatrix::_ldistribution + p, 1);
        else if(excluded == 2)
            for(unsigned short i = 1; i < p; ++i)
                DMatrix::_ldistribution[i] += i;
#else
# ifdef HPDDM_CSR_CO
#  ifndef DHYPRE
         std::partial_sum(I, I + 1 + nrow / (!blocked ? 1 : _local), I);
#  endif
#  if defined(DSUITESPARSE)
        super::template numfact<S>(nrow, I, J, pt);
        delete [] loc2glob;
#  elif defined(DMKL_PARDISO)
        super::template numfact<S>(!blocked ? 1 : _local, I, loc2glob, J, pt);
#  else
        super::template numfact<S>(nrow, I, loc2glob, J, pt);
#  endif
# else
         super::template numfact<S>(size, I, J, pt);
# endif
# ifdef DMKL_PARDISO
         if(S == 'S' || p != 1)
             delete [] C;
# else
         delete [] C;
# endif
#endif
        if(!treeDimension)
            delete [] rqRecv;
    }
    if(excluded < 2)
        delete [] *sendNeighbor;
#ifdef DMUMPS
    DMatrix::_distribution = static_cast<DMatrix::Distribution>(opt.val<char>("master_distribution", 0));
#endif
    if(U != 2) {
#ifdef DMUMPS
        if(DMatrix::_distribution == DMatrix::CENTRALIZED)
            _scatterComm = _gatherComm;
#else
        _gatherComm = _scatterComm;
#endif
    }
    else {
        unsigned short* pt;
#ifdef DMUMPS
        if(DMatrix::_distribution == DMatrix::CENTRALIZED) {
            if(rankSplit)
                infoWorld = new unsigned short[_sizeWorld];
            pt = infoWorld;
            v._max = _sizeWorld;
        }
        else {
            v._max = _sizeWorld + _sizeSplit;
            pt = new unsigned short[v._max];
            if(rankSplit == 0) {
                std::copy_n(infoWorld, _sizeWorld, pt);
                for(unsigned int i = 0; i < _sizeSplit; ++i)
                    pt[_sizeWorld + i] = infoSplit[i][1];
            }
        }
#else
        unsigned short* infoMaster;
        if(rankSplit == 0) {
            infoMaster = infoSplit[0];
            for(unsigned int i = 0; i < _sizeSplit; ++i)
                infoMaster[i] = infoSplit[i][1];
        }
        else
            infoMaster = new unsigned short[_sizeSplit];
        pt = infoMaster;
        v._max = _sizeSplit;
#endif
        MPI_Bcast(pt, v._max, MPI_UNSIGNED_SHORT, 0, _scatterComm);
#ifdef DMUMPS
        if(DMatrix::_distribution == DMatrix::CENTRALIZED) {
            constructionCommunicatorCollective<(excluded > 0)>(pt, v._max, _gatherComm, &_scatterComm);
#else
            constructionCommunicatorCollective<false>(pt, v._max, _scatterComm);
#endif
            _gatherComm = _scatterComm;
#ifdef DMUMPS
        }
        else {
            constructionCommunicatorCollective<(excluded > 0)>(pt, _sizeWorld, _gatherComm);
            constructionCommunicatorCollective<false>(pt + _sizeWorld, _sizeSplit, _scatterComm);
        }
#endif
        if(rankSplit
#ifdef DMUMPS
                || DMatrix::_distribution == DMatrix::DISTRIBUTED_SOL
#endif
                )
            delete [] pt;
    }
    if(rankSplit == 0) {
#ifdef DMUMPS
        if(DMatrix::_distribution == DMatrix::CENTRALIZED) {
            if(_rankWorld == 0) {
                _sizeRHS = DMatrix::_n;
                if(U == 1)
                    constructionCollective<true, DMatrix::CENTRALIZED, excluded == 2>();
                else if(U == 2) {
                    DMatrix::_gatherCounts = new int[1];
                    if(_local == 0) {
                        _local = *DMatrix::_gatherCounts = *std::find_if(infoWorld, infoWorld + _sizeWorld, [](const unsigned short& nu) { return nu != 0; });
                        _sizeRHS += _local;
                    }
                    else
                        *DMatrix::_gatherCounts = _local;
                }
                else
                    constructionCollective<false, DMatrix::CENTRALIZED, excluded == 2>(infoWorld, p - 1);
            }
            else {
                if(U == 0)
                    DMatrix::_displs = &_rankWorld;
                _sizeRHS = _local;
            }
        }
        else {
#endif
            constructionMap<T, U == 1, excluded == 2>(p, U == 1 ? nullptr : infoWorld);
#ifdef DMUMPS
            if(_rankWorld == 0)
                _sizeRHS = DMatrix::_n;
            else
#endif
                _sizeRHS = DMatrix::_ldistribution[DMatrix::_rank];
            if(U == 1)
                constructionCollective<true, DMatrix::DISTRIBUTED_SOL, excluded == 2>();
            else if(U == 2) {
                DMatrix::_gatherCounts = new int[1];
                if(_local == 0) {
                    _local = *DMatrix::_gatherCounts = *std::find_if(infoWorld, infoWorld + _sizeWorld, [](const unsigned short& nu) { return nu != 0; });
                    _sizeRHS += _local;
                }
                else
                    *DMatrix::_gatherCounts = _local;
            }
            else {
                unsigned short* infoMaster = infoSplit[0];
                for(unsigned int i = 0; i < _sizeSplit; ++i)
                    infoMaster[i] = infoSplit[i][1];
                constructionCollective<false, DMatrix::DISTRIBUTED_SOL, excluded == 2>(infoWorld, p - 1, infoMaster);
            }
#ifdef DMUMPS
        }
#endif
        delete [] *infoSplit;
        delete [] infoSplit;
        if(excluded == 2) {
#ifdef DMUMPS
            if(DMatrix::_distribution == DMatrix::CENTRALIZED && _rankWorld == 0)
                _sizeRHS += _local;
            else if(DMatrix::_distribution == DMatrix::DISTRIBUTED_SOL)
#endif
                _sizeRHS += _local;
        }
        DMatrix::_n
#if HPDDM_INEXACT_COARSE_OPERATOR
            = rank /
#else
            /=
#endif
                    (!blocked ? 1 : _local);
    }
    return ret;
}

template<template<class> class Solver, char S, class K>
template<bool excluded>
inline void CoarseOperator<Solver, S, K>::callSolver(K* const pt, const unsigned short& mu) {
    downscaled_type<K>* rhs = reinterpret_cast<downscaled_type<K>*>(pt);
    if(!std::is_same<downscaled_type<K>, K>::value)
        for(unsigned int i = 0; i < mu * _local; ++i)
            rhs[i] = pt[i];
    if(_scatterComm != MPI_COMM_NULL) {
#ifdef DMUMPS
        if(DMatrix::_distribution == DMatrix::DISTRIBUTED_SOL) {
            if(DMatrix::_displs) {
                if(_rankWorld == 0) {
                    transfer<false>(DMatrix::_gatherCounts, _sizeWorld, mu, rhs);
                    std::for_each(DMatrix::_gatherCounts, DMatrix::_displs + _sizeWorld, [&](int& i) { i /= mu; });
                }
                else if(_gatherComm != MPI_COMM_NULL)
                    MPI_Gatherv(rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, _gatherComm);
                if(DMatrix::_communicator != MPI_COMM_NULL) {
                    super::template solve<DMatrix::DISTRIBUTED_SOL>(rhs, mu);
                    std::for_each(DMatrix::_gatherSplitCounts, DMatrix::_displsSplit + _sizeSplit, [&](int& i) { i *= mu; });
                    transfer<true>(DMatrix::_gatherSplitCounts, mu, _sizeSplit, rhs);
                }
                else
                    MPI_Scatterv(NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), 0, _scatterComm);
            }
            else {
                if(_rankWorld == 0) {
                    MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rhs, mu * *DMatrix::_gatherCounts, Wrapper<downscaled_type<K>>::mpi_type(), 0, _gatherComm);
                    int p = 0;
                    if(_offset || excluded)
                        MPI_Comm_size(DMatrix::_communicator, &p);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(_sizeWorld - p, mu, rhs + (p ? mu * *DMatrix::_gatherCounts : 0), *DMatrix::_gatherCounts);
                }
                else if(_gatherComm != MPI_COMM_NULL)
                    MPI_Gather(rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, MPI_DATATYPE_NULL, 0, _gatherComm);
                if(DMatrix::_communicator != MPI_COMM_NULL) {
                    super::template solve<DMatrix::DISTRIBUTED_SOL>(rhs + (_offset || excluded ? mu * *DMatrix::_gatherCounts : 0), mu);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(mu, _sizeSplit - (_offset || excluded), rhs + (_offset || excluded ? mu * *DMatrix::_gatherCounts : 0), *DMatrix::_gatherCounts);
                    MPI_Scatter(rhs, mu * *DMatrix::_gatherCounts, Wrapper<downscaled_type<K>>::mpi_type(), MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 0, _scatterComm);
                }
                else
                    MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), 0, _scatterComm);
            }
        }
        else {
            if(DMatrix::_displs) {
                if(_rankWorld == 0)
                    transfer<false>(DMatrix::_gatherCounts, _sizeWorld, mu, rhs);
                else if(_gatherComm != MPI_COMM_NULL)
                    MPI_Gatherv(rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, _gatherComm);
                if(DMatrix::_communicator != MPI_COMM_NULL)
                    super::template solve<DMatrix::CENTRALIZED>(rhs, mu);
                if(_rankWorld == 0)
                    transfer<true>(DMatrix::_gatherCounts, mu, _sizeWorld, rhs);
                else if(_gatherComm != MPI_COMM_NULL)
                    MPI_Scatterv(NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), 0, _gatherComm);
            }
            else {
                int p = 0;
                if(_rankWorld == 0) {
                    if(_offset || excluded)
                        MPI_Comm_size(DMatrix::_communicator, &p);
                    MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rhs, mu * *DMatrix::_gatherCounts, Wrapper<downscaled_type<K>>::mpi_type(), 0, _gatherComm);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(_sizeWorld - p, mu, rhs + (p ? mu * *DMatrix::_gatherCounts : 0), *DMatrix::_gatherCounts);
                }
                else
                    MPI_Gather(rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, MPI_DATATYPE_NULL, 0, _gatherComm);
                if(DMatrix::_communicator != MPI_COMM_NULL)
                    super::template solve<DMatrix::CENTRALIZED>(rhs + (_offset || excluded ? mu * _local : 0), mu);
                if(_rankWorld == 0) {
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(mu, _sizeWorld - p, rhs + (p ? mu * *DMatrix::_gatherCounts : 0), *DMatrix::_gatherCounts);
                    MPI_Scatter(rhs, mu * *DMatrix::_gatherCounts, Wrapper<downscaled_type<K>>::mpi_type(), MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 0, _scatterComm);
                }
                else
                    MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), 0, _scatterComm);
            }
        }
#else
            if(DMatrix::_displs) {
                if(DMatrix::_communicator != MPI_COMM_NULL) {
                    transfer<false>(DMatrix::_gatherSplitCounts, _sizeSplit, mu, rhs);
                    super::solve(rhs, mu);
                    transfer<true>(DMatrix::_gatherSplitCounts, mu, _sizeSplit, rhs);
                }
                else {
                    MPI_Gatherv(rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, _gatherComm);
                    MPI_Scatterv(NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), 0, _scatterComm);
                }
            }
            else {
                if(DMatrix::_communicator != MPI_COMM_NULL) {
                    MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rhs, mu * *DMatrix::_gatherCounts, Wrapper<downscaled_type<K>>::mpi_type(), 0, _gatherComm);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(_sizeSplit - (_offset || excluded), mu, rhs + (_offset || excluded ? mu * *DMatrix::_gatherCounts : 0), *DMatrix::_gatherCounts);
                    super::solve(rhs + (_offset || excluded ? mu * *DMatrix::_gatherCounts : 0), mu);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(mu, _sizeSplit - (_offset || excluded), rhs + (_offset || excluded ? mu * *DMatrix::_gatherCounts : 0), *DMatrix::_gatherCounts);
                    MPI_Scatter(rhs, mu * *DMatrix::_gatherCounts, Wrapper<downscaled_type<K>>::mpi_type(), MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 0, _scatterComm);
                }
                else {
                    MPI_Gather(rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, MPI_DATATYPE_NULL, 0, _gatherComm);
                    MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), 0, _scatterComm);
                }
            }
#endif
    }
    else if(DMatrix::_communicator != MPI_COMM_NULL) {
#ifdef DMUMPS
        if(DMatrix::_distribution == DMatrix::DISTRIBUTED_SOL)
            super::template solve<DMatrix::DISTRIBUTED_SOL>(rhs, mu);
        else
            super::template solve<DMatrix::CENTRALIZED>(rhs, mu);
#else
            super::solve(rhs, mu);
#endif
    }
    if(!std::is_same<downscaled_type<K>, K>::value)
        for(unsigned int i = mu * _local; i-- > 0; )
            pt[i] = rhs[i];
}

#if HPDDM_ICOLLECTIVE
template<template<class> class Solver, char S, class K>
template<bool excluded>
inline void CoarseOperator<Solver, S, K>::IcallSolver(K* const pt, const unsigned short& mu, MPI_Request* rq) {
    downscaled_type<K>* rhs = reinterpret_cast<downscaled_type<K>*>(pt);
    if(!std::is_same<downscaled_type<K>, K>::value)
        for(unsigned int i = 0; i < mu * _local; ++i)
            rhs[i] = pt[i];
    if(_scatterComm != MPI_COMM_NULL) {
#ifdef DMUMPS
        if(DMatrix::_distribution == DMatrix::DISTRIBUTED_SOL) {
            if(DMatrix::_displs) {
                if(_rankWorld == 0) {
                    Itransfer<false>(DMatrix::_gatherCounts, _sizeWorld, mu, rhs, rq);
                    std::for_each(DMatrix::_gatherCounts, DMatrix::_displs + _sizeWorld, [&](int& i) { i /= mu; });
                }
                else if(_gatherComm != MPI_COMM_NULL)
                    MPI_Igatherv(rhs, _local, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, _gatherComm, rq);
                if(DMatrix::_communicator != MPI_COMM_NULL) {
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    super::template solve<DMatrix::DISTRIBUTED_SOL>(rhs, mu);
                    std::for_each(DMatrix::_gatherSplitCounts, DMatrix::_displsSplit + _sizeSplit, [&](int& i) { i *= mu; });
                    Itransfer<true>(DMatrix::_gatherSplitCounts, mu, _sizeSplit, rhs, rq + 1);
                }
                else
                    MPI_Iscatterv(NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), 0, _scatterComm, rq + 1);
            }
            else {
                if(_rankWorld == 0) {
                    MPI_Igather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rhs, mu * *DMatrix::_gatherCounts, Wrapper<downscaled_type<K>>::mpi_type(), 0, _gatherComm, rq);
                    int p = 0;
                    if(_offset || excluded)
                        MPI_Comm_size(DMatrix::_communicator, &p);
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    *rq = MPI_REQUEST_NULL;
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(_sizeWorld - p, mu, rhs + (p ? mu * *DMatrix::_gatherCounts : 0), *DMatrix::_gatherCounts);
                }
                else if(_gatherComm != MPI_COMM_NULL)
                    MPI_Igather(rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, MPI_DATATYPE_NULL, 0, _gatherComm, rq);
                if(DMatrix::_communicator != MPI_COMM_NULL) {
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    super::template solve<DMatrix::DISTRIBUTED_SOL>(rhs + (_offset || excluded ? *DMatrix::_gatherCounts : 0), mu);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(mu, _sizeSplit - (_offset || excluded), rhs + (_offset || excluded ? mu * *DMatrix::_gatherCounts : 0), *DMatrix::_gatherCounts);
                    MPI_Iscatter(rhs, mu * *DMatrix::_gatherCounts, Wrapper<downscaled_type<K>>::mpi_type(), MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 0, _scatterComm, rq + 1);
                }
                else
                    MPI_Iscatter(NULL, 0, MPI_DATATYPE_NULL, rhs, _local, Wrapper<downscaled_type<K>>::mpi_type(), 0, _scatterComm, rq + 1);
            }
        }
        else {
            if(DMatrix::_displs) {
                if(_rankWorld == 0)
                    Itransfer<false>(DMatrix::_gatherCounts, _sizeWorld, mu, rhs, rq);
                else if(_gatherComm != MPI_COMM_NULL)
                    MPI_Igatherv(rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, _gatherComm, rq);
                if(DMatrix::_communicator != MPI_COMM_NULL) {
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    super::template solve<DMatrix::CENTRALIZED>(rhs, mu);
                }
                if(_rankWorld == 0)
                    Itransfer<true>(DMatrix::_gatherCounts, mu, _sizeWorld, rhs, rq + 1);
                else if(_gatherComm != MPI_COMM_NULL)
                    MPI_Iscatterv(NULL, 0, 0, MPI_DATATYPE_NULL, rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), 0, _gatherComm, rq + 1);
            }
            else {
                int p = 0;
                if(_rankWorld == 0) {
                    if(_offset || excluded)
                        MPI_Comm_size(DMatrix::_communicator, &p);
                    MPI_Igather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rhs, mu * *DMatrix::_gatherCounts, Wrapper<downscaled_type<K>>::mpi_type(), 0, _gatherComm, rq);
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    *rq = MPI_REQUEST_NULL;
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(_sizeWorld - p, mu, rhs + (p ? mu * *DMatrix::_gatherCounts : 0), *DMatrix::_gatherCounts);
                }
                else
                    MPI_Igather(rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, MPI_DATATYPE_NULL, 0, _gatherComm, rq);
                if(DMatrix::_communicator != MPI_COMM_NULL) {
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    super::template solve<DMatrix::CENTRALIZED>(rhs + (_offset || excluded ? _local : 0), mu);
                }
                if(_rankWorld == 0) {
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(mu, _sizeWorld - p, rhs + (p ? mu * *DMatrix::_gatherCounts : 0), *DMatrix::_gatherCounts);
                    MPI_Iscatter(rhs, mu * *DMatrix::_gatherCounts, Wrapper<downscaled_type<K>>::mpi_type(), MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 0, _scatterComm, rq + 1);
                }
                else
                    MPI_Iscatter(NULL, 0, MPI_DATATYPE_NULL, rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), 0, _scatterComm, rq + 1);
            }
        }
#else
            if(DMatrix::_displs) {
                if(DMatrix::_communicator != MPI_COMM_NULL) {
                    Itransfer<false>(DMatrix::_gatherSplitCounts, _sizeSplit, mu, rhs, rq);
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    super::solve(rhs, mu);
                    Itransfer<true>(DMatrix::_gatherSplitCounts, mu, _sizeSplit, rhs, rq + 1);
                }
                else {
                    MPI_Igatherv(rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, _gatherComm, rq);
                    MPI_Iscatterv(NULL, 0, 0, Wrapper<downscaled_type<K>>::mpi_type(), rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), 0, _scatterComm, rq + 1);
                }
            }
            else {
                if(DMatrix::_communicator != MPI_COMM_NULL) {
                    MPI_Igather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rhs, mu * *DMatrix::_gatherCounts, Wrapper<downscaled_type<K>>::mpi_type(), 0, _gatherComm, rq);
                    MPI_Wait(rq, MPI_STATUS_IGNORE);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(_sizeSplit - (_offset || excluded), mu, rhs + (_offset || excluded ? mu * *DMatrix::_gatherCounts : 0), *DMatrix::_gatherCounts);
                    super::solve(rhs + (_offset || excluded ? *DMatrix::_gatherCounts : 0), mu);
                    Wrapper<downscaled_type<K>>::template cycle<'T'>(mu, _sizeSplit - (_offset || excluded), rhs + (_offset || excluded ? mu * *DMatrix::_gatherCounts : 0), *DMatrix::_gatherCounts);
                    MPI_Iscatter(rhs, mu * *DMatrix::_gatherCounts, Wrapper<downscaled_type<K>>::mpi_type(), MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 0, _scatterComm, rq + 1);
                }
                else {
                    MPI_Igather(rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), NULL, 0, MPI_DATATYPE_NULL, 0, _gatherComm, rq);
                    MPI_Iscatter(NULL, 0, MPI_DATATYPE_NULL, rhs, mu * _local, Wrapper<downscaled_type<K>>::mpi_type(), 0, _scatterComm, rq + 1);
                }
            }
#endif
    }
    else if(DMatrix::_communicator != MPI_COMM_NULL) {
#ifdef DMUMPS
        if(DMatrix::_distribution == DMatrix::DISTRIBUTED_SOL)
            super::template solve<DMatrix::DISTRIBUTED_SOL>(rhs, mu); break;
        else
            super::template solve<DMatrix::CENTRALIZED>(rhs, mu); break;
#else
            super::solve(rhs, mu); break;
#endif
    }
    if(!std::is_same<downscaled_type<K>, K>::value)
        for(unsigned int i = mu * _local; i-- > 0; )
            pt[i] = rhs[i];
}
#endif // HPDDM_ICOLLECTIVE
} // HPDDM
#endif // _HPDDM_COARSE_OPERATOR_IMPL_
