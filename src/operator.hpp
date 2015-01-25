/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <jolivet@ann.jussieu.fr>
        Date: 2013-03-12

   Copyright (C) 2011-2014 Université de Grenoble

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

#ifndef _OPERATOR_
#define _OPERATOR_

#include <queue>
#if HPDDM_BDD || HPDDM_FETI
#include <unordered_map>
#endif

namespace HPDDM {
template<char P, class Preconditioner, class K>
class OperatorBase {
    protected:
        const Preconditioner&                               _p;
        K** const                                   _deflation;
        const vectorNeighbor&                             _map;
        std::vector<unsigned short>                  _sparsity;
        std::vector<std::vector<unsigned short>>  _vecSparsity;
        const int                                           _n;
        const int                                       _local;
    public:
        static constexpr char                     _pattern = P;
        inline const std::vector<unsigned short>& getPattern() const { return _sparsity; }
        OperatorBase(const Preconditioner& p, const unsigned short& nu) : _p(p), _deflation(p.getVectors()), _map(p.getMap()), _sparsity(), _n(p.getDof()), _local(nu) {
            static_assert(P == 's', "Unsupported constructor with such a sparsity pattern");
            _sparsity.reserve(_map.size());
            for(const pairNeighbor& neighbor : _map)
                _sparsity.emplace_back(neighbor.first);
        }
        OperatorBase(const Preconditioner& p, const unsigned short& nu, const unsigned short& relative) : _p(p), _deflation(p.getVectors()), _map(p.getMap()), _sparsity(), _n(p.getDof()), _local(nu) {
            static_assert(P == 'c', "Unsupported constructor with such a sparsity pattern");

            if(!_map.empty()) {
                _vecSparsity.resize(_map.size());
                unsigned short** recvSparsity = new unsigned short*[_map.size() + 1];
                *recvSparsity = new unsigned short[(HPDDM_MAXCO + 1) * _map.size()];
                unsigned short* sendSparsity = *recvSparsity + HPDDM_MAXCO * _map.size();
                MPI_Request* rq = new MPI_Request[2 * _map.size()];
                for(unsigned short i = 0; i < _map.size(); ++i) {
                    sendSparsity[i] = _map[i].first;
                    recvSparsity[i] = *recvSparsity + HPDDM_MAXCO * i;
                    MPI_Irecv(recvSparsity[i], HPDDM_MAXCO, MPI_UNSIGNED_SHORT, _map[i].first, 4, _p.getCommunicator(), rq + i);
                }
                for(unsigned short i = 0; i < _map.size(); ++i)
                    MPI_Isend(sendSparsity, _map.size(), MPI_UNSIGNED_SHORT, _map[i].first, 4, _p.getCommunicator(), rq + _map.size() + i);
                for(unsigned short i = 0; i < _map.size(); ++i) {
                    int index, count;
                    MPI_Status status;
                    MPI_Waitany(_map.size(), rq, &index, &status);
                    MPI_Get_count(&status, MPI_UNSIGNED_SHORT, &count);
                    _vecSparsity[index].resize(count);
                    std::copy(recvSparsity[index], recvSparsity[index] + count, _vecSparsity[index].begin());
                }
                MPI_Waitall(_map.size(), rq + _map.size(), MPI_STATUSES_IGNORE);

                delete [] *recvSparsity;
                delete [] recvSparsity;
                delete [] rq;

                _sparsity.reserve(_map.size());
                std::vector<unsigned short> neighbors;
                neighbors.reserve(_map.size());
                for(const pairNeighbor& neighbor : _map)
                    neighbors.emplace_back(neighbor.first);
                typedef std::pair<std::vector<unsigned short>::const_iterator, std::vector<unsigned short>::const_iterator> pairIt;
                auto comp = [](const pairIt& lhs, const pairIt& rhs) { return *lhs.first > *rhs.first; };
                std::priority_queue<pairIt, std::vector<pairIt>, decltype(comp)> pq(comp);
                pq.push(std::make_pair(neighbors.cbegin(), neighbors.cend()));
                for(const std::vector<unsigned short>& v : _vecSparsity)
                    pq.push(std::make_pair(v.cbegin(), v.cend()));
                while(!pq.empty()) {
                    pairIt p = pq.top();
                    pq.pop();
                    if(*p.first != relative && (_sparsity.empty() || (*p.first != _sparsity.back())))
                        _sparsity.emplace_back(*p.first);
                    if(++p.first != p.second)
                        pq.push(p);
                }
            }
        }
};

#if HPDDM_SCHWARZ
template<class Preconditioner, class K>
class MatrixMultiplication : public OperatorBase<'s', Preconditioner, K> {
    private:
        typedef OperatorBase<'s', Preconditioner, K> super;
        const MatrixCSR<K>* const                       _A;
        MatrixCSR<K, Wrapper<K>::I>*                    _C;
        const typename Wrapper<K>::ul_type* const       _D;
        K*                                           _work;
        unsigned short                             _signed;
        template<char S, bool U>
        inline void applyFromNeighbor(const K* in, unsigned short index, K*& work, unsigned short* infoNeighbor) {
            int m = U ? super::_local : *infoNeighbor;
            std::fill(work, work + m * super::_n, 0.0);
            for(unsigned short i = 0; i < m; ++i)
                Wrapper<K>::sctr(super::_map[index].second.size(), in + i * super::_map[index].second.size(), super::_map[index].second.data(), work + i * super::_n);
            Wrapper<K>::diagm(super::_n, m, _D, work, _work);
            Wrapper<K>::gemm(&(Wrapper<K>::transc), &transa, &(super::_local), &m, &(super::_n), &(Wrapper<K>::d__1), *super::_deflation, &(super::_n), _work, &(super::_n), &(Wrapper<K>::d__0), work, &(super::_local));
        }
    public:
        template<template<class> class Solver, char S, class T> friend class CoarseOperator;
        MatrixMultiplication(const Preconditioner& p, const unsigned short& nu) : OperatorBase<'s', Preconditioner, K>(p, nu), _A(p.getMatrix()), _C(), _D(p.getScaling()) { }
        ~MatrixMultiplication() { }
        inline void initialize(unsigned int k, K*& work, unsigned short s) {
            if(_A->_sym) {
                std::vector<std::vector<std::pair<unsigned int, K>>> v(_A->_n);
                unsigned int nnz = std::floor((_A->_nnz + _A->_n - 1) / _A->_n) * 2;
                for(unsigned int i = 0; i < _A->_n; ++i)
                    v[i].reserve(nnz);
                nnz = 0;
                for(unsigned int i = 0; i < _A->_n; ++i) {
                    const typename Wrapper<K>::ul_type scal = _D[i];
                    for(unsigned int j = _A->_ia[i]; j < _A->_ia[i + 1] - 1; ++j) {
                        if(_D[_A->_ja[j]] > HPDDM_EPS) {
                            v[i].emplace_back(_A->_ja[j], _A->_a[j] * _D[_A->_ja[j]]);
                            ++nnz;
                        }
                        if(scal > HPDDM_EPS) {
                            v[_A->_ja[j]].emplace_back(i, _A->_a[j] * scal);
                            ++nnz;
                        }
                    }
                    if(scal > HPDDM_EPS) {
                        v[i].emplace_back(i, _A->_a[_A->_ia[i + 1] - 1] * scal);
                        ++nnz;
                    }
                }
                _C = new MatrixCSR<K, Wrapper<K>::I>(_A->_n, _A->_n, nnz, false);
                _C->_ia[0] = (Wrapper<K>::I == 'F');
                nnz = 0;
                unsigned int i;
#pragma omp parallel for schedule(static, HPDDM_GRANULARITY)
                for(i = 0; i < _A->_n; ++i)
                    std::sort(v[i].begin(), v[i].end(), [](const std::pair<unsigned int, K>& lhs, const std::pair<unsigned int, K>& rhs) { return lhs.first < rhs.first; });
                for(i = 0; i < _A->_n; ++i) {
                    for(const std::pair<unsigned int, K>& p : v[i]) {
                        _C->_ja[nnz] = p.first + (Wrapper<K>::I == 'F');
                        _C->_a[nnz++] = p.second;
                    }
                    _C->_ia[i + 1] = nnz + (Wrapper<K>::I == 'F');
                }
            }
            else {
                _C = new MatrixCSR<K, Wrapper<K>::I>(_A->_n, _A->_n, _A->_nnz, false);
                _C->_ia[0] = (Wrapper<K>::I == 'F');
                unsigned int nnz = 0;
                for(unsigned int i = 0; i < _A->_n; ++i) {
                    for(unsigned int j = _A->_ia[i]; j < _A->_ia[i + 1]; ++j) {
                        if(_D[_A->_ja[j]] > HPDDM_EPS) {
                            _C->_ja[nnz] = _A->_ja[j] + (Wrapper<K>::I == 'F');
                            _C->_a[nnz++] = _A->_a[j] * _D[_A->_ja[j]];
                        }
                    }
                    _C->_ia[i + 1] = nnz + (Wrapper<K>::I == 'F');
                }
            }
            work = new K[2 * k];
            _work = work + k;
            _signed = s;
        }
        template<char S, bool U, class T>
        inline void applyToNeighbor(T& in, K*& work, std::vector<MPI_Request>& rqSend, const unsigned short*, T = nullptr, MPI_Request* = nullptr) {
            Wrapper<K>::template csrmm<Wrapper<K>::I>(&transa, &(super::_n), &(super::_local), &(super::_n), &(Wrapper<K>::d__1), false, _C->_a, _C->_ia, _C->_ja, *super::_deflation, &(super::_n), &(Wrapper<K>::d__0), _work, &(super::_n));
            delete _C;
            MPI_Request rq;
            for(unsigned short i = 0; i < _signed; ++i) {
                if(U || in[i]) {
                    for(unsigned short j = 0; j < super::_local; ++j)
                        Wrapper<K>::gthr(super::_map[i].second.size(), _work + j * super::_n, in[i]+ j * super::_map[i].second.size(), super::_map[i].second.data());
                    MPI_Isend(in[i], super::_map[i].second.size() * super::_local, Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), &rq);
                    rqSend.emplace_back(rq);
                }
            }
            Wrapper<K>::diagm(super::_n, super::_local, _D, _work, work);
        }
        template<char S, bool U>
        inline void assembleForMaster(K* C, const K* in, const int& coefficients, unsigned short index, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            applyFromNeighbor<S, U>(in, index, arrayC, infoNeighbor);
            for(unsigned short j = 0; j < (U ? super::_local : *infoNeighbor); ++j) {
                K* pt = C + j;
                for(unsigned short i = 0; i < super::_local; pt += coefficients - (S == 'S') * i++)
                    *pt = arrayC[j * super::_local + i];
            }
        }
        template<char S, char N, bool U>
        inline void applyFromNeighborMaster(const K* in, unsigned short index, int* I, int* J, K* C, int coefficients, unsigned int offsetI, unsigned int* offsetJ, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            applyFromNeighbor<S, U>(in, index, arrayC, infoNeighbor);
            unsigned int offset = U ? super::_map[index].first * super::_local + (N == 'F') : *offsetJ;
            for(unsigned short i = 0; i < super::_local; ++i) {
                unsigned int l = coefficients * i - (S == 'S') * (i * (i - 1)) / 2;
                for(unsigned short j = 0; j < (U ? super::_local : *infoNeighbor); ++j) {
#ifndef HPDDM_CSR_CO
                    I[l + j] = offsetI + i;
#endif
                    J[l + j] = offset + j;
                    C[l + j] = arrayC[j * super::_local + i];
                }
            }
        }
};
#endif // HPDDM_SCHWARZ

#if HPDDM_FETI
template<class Preconditioner, class K>
class FetiProjection : public OperatorBase<'c', Preconditioner, K> {
    private:
        typedef OperatorBase<'c', Preconditioner, K>                super;
        std::unordered_map<unsigned short, unsigned short>       _offsets;
        unsigned short                                       _consolidate;
        template<char S, bool U>
        inline void applyFromNeighbor(const K* in, unsigned short index, K*& work, unsigned short* info) {
            unsigned short rankWorld = super::_p.getRank();
            unsigned short between   = super::_p.getSigned();
            std::vector<unsigned short>::const_iterator middle = std::lower_bound(super::_vecSparsity[index].cbegin(), super::_vecSparsity[index].cend(), rankWorld);
            unsigned int accumulate = 0;
            if(!(index < between)) {
                for(unsigned short k = 0; k < (U ? super::_local : info[std::distance(super::_sparsity.cbegin(), std::lower_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_map[index].first))]); ++k) {
                    for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                        work[_offsets[super::_map[index].first] + super::_map[index].second[j] + k * super::_n] += in[k * super::_map[index].second.size() + j];
                }
                accumulate += (U ? super::_local : info[std::distance(super::_sparsity.cbegin(), std::lower_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_map[index].first))]) * super::_map[index].second.size();
            }
            else if(S != 'S') {
                for(unsigned short k = 0; k < (U ? super::_local : info[std::distance(super::_sparsity.cbegin(), std::lower_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_map[index].first))]); ++k) {
                    for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                        work[_offsets[super::_map[index].first] + super::_map[index].second[j] + k * super::_n] -= in[k * super::_map[index].second.size() + j];
                }
                accumulate += (U ? super::_local : info[std::distance(super::_sparsity.cbegin(), std::lower_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_map[index].first))]) * super::_map[index].second.size();
            }
            std::vector<unsigned short>::const_iterator begin = super::_sparsity.cbegin();
            if(S != 'S')
                for(std::vector<unsigned short>::const_iterator it = super::_vecSparsity[index].cbegin(); it != middle; ++it) {
                    if(!U) {
                        std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::_sparsity.cend(), *it);
                        if(*it > super::_map[index].first || between > index)
                            for(unsigned short k = 0; k < info[std::distance(super::_sparsity.cbegin(), idx)]; ++k) {
                                for(unsigned int j = 0; j < super::_map[index].second.size(); ++j) {
                                    work[_offsets[*it] + super::_map[index].second[j] + k * super::_n] -= in[accumulate + k * super::_map[index].second.size() + j];
                                }
                            }
                        else
                            for(unsigned short k = 0; k < info[std::distance(super::_sparsity.cbegin(), idx)]; ++k) {
                                for(unsigned int j = 0; j < super::_map[index].second.size(); ++j) {
                                    work[_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                                }
                            }
                        accumulate += info[std::distance(super::_sparsity.cbegin(), idx)] * super::_map[index].second.size();
                        begin = idx + 1;
                    }
                    else {
                        if(*it > super::_map[index].first || between > index)
                            for(unsigned short k = 0; k < super::_local; ++k) {
                                for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                    work[_offsets[*it] + super::_map[index].second[j] + k * super::_n] -= in[accumulate + k * super::_map[index].second.size() + j];
                            }
                        else
                            for(unsigned short k = 0; k < super::_local; ++k) {
                                for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                    work[_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                            }
                        accumulate += super::_local * super::_map[index].second.size();
                    }
                }
            if(index < between)
                for(unsigned short k = 0; k < super::_local; ++k) {
                    for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                        work[_offsets[rankWorld] + super::_map[index].second[j] + k * super::_n] -= in[accumulate + k * super::_map[index].second.size() + j];
                }
            else
                for(unsigned short k = 0; k < super::_local; ++k) {
                    for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                        work[_offsets[rankWorld] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                }
            accumulate += super::_local * super::_map[index].second.size();
            for(std::vector<unsigned short>::const_iterator it = middle + 1; it != super::_vecSparsity[index].cend(); ++it) {
                if(!U) {
                    std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::_sparsity.cend(), *it);
                    if(*it > super::_map[index].first && between > index)
                        for(unsigned short k = 0; k < info[std::distance(super::_sparsity.cbegin(), idx)]; ++k) {
                            for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                work[_offsets[*it] + super::_map[index].second[j] + k * super::_n] -= in[accumulate + k * super::_map[index].second.size() + j];
                        }
                    else
                        for(unsigned short k = 0; k < info[std::distance(super::_sparsity.cbegin(), idx)]; ++k) {
                            for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                work[_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                        }
                    accumulate += info[std::distance(super::_sparsity.cbegin(), idx)] * super::_map[index].second.size();
                    begin = idx + 1;
                }
                else {
                    if(*it > super::_map[index].first && between > index)
                        for(unsigned short k = 0; k < super::_local; ++k) {
                            for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                work[_offsets[*it] + super::_map[index].second[j] + k * super::_n] -= in[accumulate + k * super::_map[index].second.size() + j];
                        }
                    else
                        for(unsigned short k = 0; k < super::_local; ++k) {
                            for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                work[_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                        }
                    accumulate += super::_local * super::_map[index].second.size();
                }
            }
        }
    public:
        template<template<class> class Solver, char S, class T> friend class CoarseOperator;
        FetiProjection(const Preconditioner& p, const unsigned short& nu) : OperatorBase<'c', Preconditioner, K>(p, nu, p.getRank()), _consolidate() {
            if(super::_deflation)
                for(unsigned short i = 0; i < super::_local; ++i)
                    super::_deflation[i] += *super::_p.getLDR() - super::_n;
        }
        ~FetiProjection() {
            if(super::_deflation)
                for(unsigned short i = 0; i < super::_local; ++i)
                    super::_deflation[i] -= *super::_p.getLDR() - super::_n;
        }
        inline void initialize(unsigned int, K*&, unsigned short) { }
        template<char S, bool U, class T>
        inline void applyToNeighbor(T& in, K*& work, std::vector<MPI_Request>& rqSend, const unsigned short* info, T const& out = nullptr, MPI_Request* const& rqRecv = nullptr) {
            unsigned short rankWorld = super::_p.getRank();
            unsigned short between = super::_p.getSigned();
            unsigned short* infoNeighbor;
            if(!U) {
                infoNeighbor = new unsigned short[super::_map.size()];
                std::vector<unsigned short>::const_iterator begin = super::_sparsity.cbegin();
                for(unsigned short i = 0; i < super::_map.size(); ++i) {
                    std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::_sparsity.cend(), super::_map[i].first);
                    infoNeighbor[i] = info[std::distance(super::_sparsity.cbegin(), idx)];
                    begin = idx + 1;
                }
            }
            if(S != 'S') {
                if(!U) {
                    unsigned short size = std::accumulate(infoNeighbor, infoNeighbor + super::_map.size(), super::_local);
                    for(unsigned short i = 0; i < super::_map.size(); ++i)
                        in[i] = new K[size * super::_map[i].second.size()];
                    for(unsigned short i = 0; i < super::_map.size(); ++i) {
                        size = infoNeighbor[i];
                        std::vector<unsigned short>::const_iterator begin = super::_sparsity.cbegin();
                        for(const unsigned short& rank : super::_vecSparsity[i]) {
                            if(rank == rankWorld)
                                size += super::_local;
                            else {
                                std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::_sparsity.cend(), rank);
                                size += info[std::distance(super::_sparsity.cbegin(), idx)];
                                begin = idx + 1;
                            }
                        }
                        if(super::_local) {
                            out[i] = new K[size * super::_map[i].second.size()];
                            MPI_Irecv(out[i], size * super::_map[i].second.size(), Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), rqRecv + i);
                        }
                        else
                            rqRecv[i] = MPI_REQUEST_NULL;
                    }
                }
                else {
                    for(unsigned short i = 0; i < super::_map.size(); ++i) {
                        in[i] = new K[super::_local * (super::_map.size() + 1) * super::_map[i].second.size()];
                        out[i] = new K[super::_local * (super::_vecSparsity[i].size() + 1) * super::_map[i].second.size()];
                        MPI_Irecv(out[i], super::_local * (super::_vecSparsity[i].size() + 1) * super::_map[i].second.size(), Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), rqRecv + i);
                    }
                }
            }
            else {
                if(!U) {
                    unsigned short size = std::accumulate(infoNeighbor, infoNeighbor + super::_map.size(), 0);
                    for(unsigned short i = 0; i < super::_map.size(); ++i) {
                        in[i] = new K[(size + super::_local * (i < between)) * super::_map[i].second.size()];
                        size -= infoNeighbor[i];
                    }
                    for(unsigned short i = 0; i < super::_map.size(); ++i) {
                        size = infoNeighbor[i] * !(i < between) + super::_local;
                        std::vector<unsigned short>::const_iterator end = super::_sparsity.cend();
                        for(std::vector<unsigned short>::const_reverse_iterator rit = super::_vecSparsity[i].rbegin(); *rit > rankWorld; ++rit) {
                            std::vector<unsigned short>::const_iterator idx = std::lower_bound(super::_sparsity.cbegin(), end, *rit);
                            size += info[std::distance(super::_sparsity.cbegin(), idx)];
                            end = idx - 1;
                        }
                        if(super::_local) {
                            out[i] = new K[size * super::_map[i].second.size()];
                            MPI_Irecv(out[i], size * super::_map[i].second.size(), Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), rqRecv + i);
                        }
                        else
                            rqRecv[i] = MPI_REQUEST_NULL;
                    }
                }
                else {
                    for(unsigned short i = 0; i < super::_map.size(); ++i)
                        in[i] = new K[super::_local * (super::_map.size() + (i < between) - i) * super::_map[i].second.size()];
                    for(unsigned short i = 0; i < super::_map.size(); ++i) {
                        unsigned short size = std::distance(std::lower_bound(super::_vecSparsity[i].cbegin(), super::_vecSparsity[i].cend(), rankWorld), super::_vecSparsity[i].cend()) + !(i < between);
                        out[i] = new K[super::_local * size * super::_map[i].second.size()];
                        MPI_Irecv(out[i], super::_local * size * super::_map[i].second.size(), Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), rqRecv + i);
                    }
                }
            }
            MPI_Request* rqMult = new MPI_Request[2 * super::_map.size()];
            unsigned int* offset = new unsigned int[super::_map.size() + 2];
            offset[0] = 0;
            offset[1] = super::_local;
            for(unsigned short i = 2; i < super::_map.size() + 2; ++i)
                offset[i] = offset[i - 1] + (U ? super::_local : infoNeighbor[i - 2]);
            const int nbMult = super::_p.getMult();
            K* mult = new K[offset[super::_map.size() + 1] * nbMult];
            unsigned short* accumulator = new unsigned short[super::_map.size() + 1];
            accumulator[0] = 0;
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                MPI_Irecv(mult + offset[i + 1] * nbMult + accumulator[i] * (U ? super::_local : infoNeighbor[i]), super::_map[i].second.size() * (U ? super::_local : infoNeighbor[i]), Wrapper<K>::mpi_type(), super::_map[i].first, 11, super::_p.Subdomain<K>::getCommunicator(), rqMult + i);
                accumulator[i + 1] = accumulator[i] + super::_map[i].second.size();
            }

            K* tmp = new K[offset[super::_map.size() + 1] * super::_n]();
            const typename Wrapper<K>::ul_type* const* const m = super::_p.getScaling();
            for(unsigned short i = 0; i < between; ++i) {
                for(unsigned short k = 0; k < super::_local; ++k)
                    for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                        tmp[super::_map[i].second[j] + k * super::_n] -= m[i][j] * (mult[accumulator[i] * super::_local + j + k * super::_map[i].second.size()] = - super::_deflation[k][super::_map[i].second[j]]);
                MPI_Isend(mult + accumulator[i] * super::_local, super::_map[i].second.size() * super::_local, Wrapper<K>::mpi_type(), super::_map[i].first, 11, super::_p.Subdomain<K>::getCommunicator(), rqMult + super::_map.size() + i);
            }
            for(unsigned short i = between; i < super::_map.size(); ++i) {
                for(unsigned short k = 0; k < super::_local; ++k)
                    for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                        tmp[super::_map[i].second[j] + k * super::_n] += m[i][j] * (mult[accumulator[i] * super::_local + j + k * super::_map[i].second.size()] =   super::_deflation[k][super::_map[i].second[j]]);
                MPI_Isend(mult + accumulator[i] * super::_local, super::_map[i].second.size() * super::_local, Wrapper<K>::mpi_type(), super::_map[i].first, 11, super::_p.Subdomain<K>::getCommunicator(), rqMult + super::_map.size() + i);
            }

            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                int index;
                MPI_Waitany(super::_map.size(), rqMult, &index, MPI_STATUS_IGNORE);
                if(index < between)
                    for(unsigned short k = 0; k < (U ? super::_local : infoNeighbor[index]); ++k)
                        for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                            tmp[super::_map[index].second[j] + (offset[index + 1] + k) * super::_n] = - m[index][j] * mult[offset[index + 1] * nbMult + accumulator[index] * (U ? super::_local : infoNeighbor[index]) + j + k * super::_map[index].second.size()];
                else
                    for(unsigned short k = 0; k < (U ? super::_local : infoNeighbor[index]); ++k)
                        for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                            tmp[super::_map[index].second[j] + (offset[index + 1] + k) * super::_n] =   m[index][j] * mult[offset[index + 1] * nbMult + accumulator[index] * (U ? super::_local : infoNeighbor[index]) + j + k * super::_map[index].second.size()];
            }

            delete [] accumulator;

            if(offset[super::_map.size() + 1])
                super::_p.applyLocalPreconditioner(tmp, offset[super::_map.size() + 1]);

            MPI_Waitall(super::_map.size(), rqMult + super::_map.size(), MPI_STATUSES_IGNORE);
            delete [] rqMult;
            delete [] mult;

            unsigned int accumulate = 0;
            unsigned short stop = std::distance(super::_sparsity.cbegin(), std::upper_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), rankWorld));
            if(S != 'S') {
                _offsets.reserve(super::_sparsity.size() + 1);
                for(unsigned short i = 0; i < stop; ++i) {
                    _offsets.emplace(super::_sparsity[i], accumulate);
                    accumulate += super::_n * (U ? super::_local : info[i]);
                }
            }
            else
                _offsets.reserve(super::_sparsity.size() + 1 - stop);
            _offsets.emplace(rankWorld, accumulate);
            accumulate += super::_n * super::_local;
            for(unsigned short i = stop; i < super::_sparsity.size(); ++i) {
                _offsets.emplace(super::_sparsity[i], accumulate);
                accumulate += super::_n * (U ? super::_local : info[i]);
            }

            work = new K[accumulate]();

            MPI_Request rq;
            for(unsigned short i = 0; i < between; ++i) {
                accumulate = super::_local;
                for(unsigned short k = 0; k < super::_local; ++k)
                    for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                        work[_offsets[rankWorld] + super::_map[i].second[j] + k * super::_n] -= (in[i][k * super::_map[i].second.size() + j] = - m[i][j] * tmp[super::_map[i].second[j] + k * super::_n]);
                for(unsigned short l = (S != 'S' ? 0 : i); l < super::_map.size(); ++l) {
                    for(unsigned short k = 0; k < (U ? super::_local : infoNeighbor[l]); ++k)
                        for(unsigned int j = 0; j < super::_map[i].second.size(); ++j) {
                            if(S != 'S' || !(l < between))
                                work[_offsets[super::_map[l].first] + super::_map[i].second[j] + k * super::_n] -= (in[i][(accumulate + k) * super::_map[i].second.size() + j] = - m[i][j] * tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n]);
                            else
                                in[i][(accumulate + k) * super::_map[i].second.size() + j] = - m[i][j] * tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n];
                        }
                    accumulate += U ? super::_local : infoNeighbor[l];
                }
                if(U || infoNeighbor[i]) {
                    MPI_Isend(in[i], super::_map[i].second.size() * accumulate, Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), &rq);
                    rqSend.emplace_back(rq);
                }
            }
            for(unsigned short i = between; i < super::_map.size(); ++i) {
                if(S != 'S') {
                    accumulate = super::_local;
                    for(unsigned short k = 0; k < super::_local; ++k)
                        for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                            work[_offsets[rankWorld] + super::_map[i].second[j] + k * super::_n] += (in[i][k * super::_map[i].second.size() + j] =   m[i][j] * tmp[super::_map[i].second[j] + k * super::_n]);
                }
                else {
                    accumulate = 0;
                    for(unsigned short k = 0; k < super::_local; ++k)
                        for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                            work[_offsets[rankWorld] + super::_map[i].second[j] + k * super::_n] += m[i][j] * tmp[super::_map[i].second[j] + k * super::_n];
                }
                for(unsigned short l = S != 'S' ? 0 : between; l < super::_map.size(); ++l) {
                    for(unsigned short k = 0; k < (U ? super::_local : infoNeighbor[l]); ++k)
                        for(unsigned int j = 0; j < super::_map[i].second.size(); ++j) {
                            if(S != 'S' || !(l < i))
                                work[_offsets[super::_map[l].first] + super::_map[i].second[j] + k * super::_n] += (in[i][(accumulate + k) * super::_map[i].second.size() + j] =   m[i][j] * tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n]);
                            else
                                work[_offsets[super::_map[l].first] + super::_map[i].second[j] + k * super::_n] += m[i][j] * tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n];
                        }
                    if(S != 'S' || !(l < i))
                        accumulate += U ? super::_local : infoNeighbor[l];
                }
                if(U || infoNeighbor[i]) {
                    MPI_Isend(in[i], super::_map[i].second.size() * accumulate, Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), &rq);
                    rqSend.emplace_back(rq);
                }
            }
            delete [] tmp;
            delete [] offset;
            if(!U)
                delete [] infoNeighbor;
        }
        template<char S, bool U>
        inline void assembleForMaster(K* C, const K* in, const int& coefficients, unsigned short index, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            applyFromNeighbor<S, U>(in, index, arrayC, infoNeighbor);
            if(++_consolidate == super::_map.size()) {
                if(S != 'S')
                    Wrapper<K>::gemm(&transb, &transa, &coefficients, &(super::_local), &(super::_n), &(Wrapper<K>::d__1), arrayC, &(super::_n), *super::_deflation, super::_p.getLDR(), &(Wrapper<K>::d__0), C, &coefficients);
                else
                    for(unsigned short j = 0; j < super::_local; ++j) {
                        int local = coefficients + super::_local - j;
                        Wrapper<K>::gemv(&transb, &(super::_n), &local, &(Wrapper<K>::d__1), arrayC + super::_n * j, &(super::_n), super::_deflation[j], &i__1, &(Wrapper<K>::d__0), C - (j * (j - 1)) / 2 + j * (coefficients + super::_local), &i__1);
                    }
            }
        }
        template<char S, char N, bool U>
        inline void applyFromNeighborMaster(const K* in, unsigned short index, int* I, int* J, K* C, int coefficients, unsigned int offsetI, unsigned int* offsetJ, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            assembleForMaster<S, U>(C, in, coefficients, index, arrayC, infoNeighbor);
            if(_consolidate == super::_map.size()) {
                unsigned short between = std::distance(super::_sparsity.cbegin(), std::lower_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_p.getRank()));
                unsigned int offset = 0;
                if(S != 'S')
                    for(unsigned short k = 0; k < between; ++k)
                        for(unsigned short i = 0; i < super::_local; ++i) {
                            unsigned int l = offset + coefficients * i;
                            for(unsigned short j = 0; j < (U ? super::_local : infoNeighbor[k]); ++j) {
#ifndef HPDDM_CSR_CO
                                I[l + j] = offsetI + i;
#endif
                                J[l + j] = (U ? super::_sparsity[k] * super::_local + (N == 'F') : offsetJ[k]) + j;
                            }
                        offset += U ? super::_local : infoNeighbor[k];
                    }
                else
                    coefficients += super::_local - 1;
                for(unsigned short i = 0; i < super::_local; ++i) {
                    unsigned int l = offset + coefficients * i - (S == 'S') * ((i * (i - 1)) / 2);
                    for(unsigned short j = (S == 'S') * i; j < super::_local; ++j) {
#ifndef HPDDM_CSR_CO
                        I[l + j] = offsetI + i;
#endif
                        J[l + j] = offsetI + j;
                    }
                }
                offset += super::_local;
                for(unsigned short k = between; k < super::_sparsity.size(); ++k) {
                    for(unsigned short i = 0; i < super::_local; ++i) {
                        unsigned int l = offset + coefficients * i - (S == 'S') * ((i * (i - 1)) / 2);
                        for(unsigned short j = 0; j < (U ? super::_local : infoNeighbor[k]); ++j) {
#ifndef HPDDM_CSR_CO
                            I[l + j] = offsetI + i;
#endif
                            J[l + j] = (U ? super::_sparsity[k] * super::_local + (N == 'F') : offsetJ[k - (S == 'S') * between]) + j;
                        }
                    }
                    offset += U ? super::_local : infoNeighbor[k];
                }
            }
        }
};
#endif // HPDDM_FETI

#if HPDDM_BDD
template<class Preconditioner, class K>
class BddProjection : public OperatorBase<'c', Preconditioner, K> {
    private:
        typedef OperatorBase<'c', Preconditioner, K>                super;
        std::unordered_map<unsigned short, unsigned short>       _offsets;
        unsigned short                                       _consolidate;
        template<char S, bool U>
        inline void applyFromNeighbor(const K* in, unsigned short index, K*& work, unsigned short* info) {
            unsigned short rankWorld = super::_p.getRank();
            unsigned short between   = super::_p.getSigned();
            std::vector<unsigned short>::const_iterator middle = std::lower_bound(super::_vecSparsity[index].cbegin(), super::_vecSparsity[index].cend(), rankWorld);
            unsigned int accumulate = 0;
            if(S != 'S' || !(index < between)) {
                for(unsigned short k = 0; k < (U ? super::_local : info[std::distance(super::_sparsity.cbegin(), std::lower_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_map[index].first))]); ++k)
                    for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                        work[_offsets[super::_map[index].first] + super::_map[index].second[j] + k * super::_n] += in[k * super::_map[index].second.size() + j];
                accumulate += (U ? super::_local : info[std::distance(super::_sparsity.cbegin(), std::lower_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_map[index].first))]) * super::_map[index].second.size();
            }
            std::vector<unsigned short>::const_iterator begin = super::_sparsity.cbegin();
            if(S != 'S')
                for(std::vector<unsigned short>::const_iterator it = super::_vecSparsity[index].cbegin(); it != middle; ++it) {
                    if(!U) {
                        std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::_sparsity.cend(), *it);
                        for(unsigned short k = 0; k < info[std::distance(super::_sparsity.cbegin(), idx)]; ++k) {
                            for(unsigned int j = 0; j < super::_map[index].second.size(); ++j) {
                                work[_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                            }
                        }
                        accumulate += info[std::distance(super::_sparsity.cbegin(), idx)] * super::_map[index].second.size();
                        begin = idx + 1;
                    }
                    else {
                        for(unsigned short k = 0; k < super::_local; ++k) {
                            for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                work[_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                        }
                        accumulate += super::_local * super::_map[index].second.size();
                    }
                }
            for(unsigned short k = 0; k < super::_local; ++k) {
                for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                    work[_offsets[rankWorld] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
            }
            accumulate += super::_local * super::_map[index].second.size();
            for(std::vector<unsigned short>::const_iterator it = middle + 1; it != super::_vecSparsity[index].cend(); ++it) {
                if(!U) {
                    std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::_sparsity.cend(), *it);
                    for(unsigned short k = 0; k < info[std::distance(super::_sparsity.cbegin(), idx)]; ++k) {
                        for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                            work[_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                    }
                    accumulate += info[std::distance(super::_sparsity.cbegin(), idx)] * super::_map[index].second.size();
                    begin = idx + 1;
                }
                else {
                    for(unsigned short k = 0; k < super::_local; ++k) {
                        for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                            work[_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                    }
                    accumulate += super::_local * super::_map[index].second.size();
                }
            }
        }
    public:
        template<template<class> class Solver, char S, class T> friend class CoarseOperator;
        BddProjection(const Preconditioner& p, const unsigned short& nu) : OperatorBase<'c', Preconditioner, K>(p, nu, p.getRank()), _consolidate() {
            if(super::_deflation)
                for(unsigned short i = 0; i < super::_local; ++i)
                    super::_deflation[i] += *super::_p.getLDR() - super::_n;
        }
        ~BddProjection() {
            if(super::_deflation)
                for(unsigned short i = 0; i < super::_local; ++i)
                    super::_deflation[i] -= *super::_p.getLDR() - super::_n;
        }
        inline void initialize(unsigned int, K*&, unsigned short) { }
        template<char S, bool U, class T>
        inline void applyToNeighbor(T& in, K*& work, std::vector<MPI_Request>& rqSend, const unsigned short* info, T const& out = nullptr, MPI_Request* const& rqRecv = nullptr) {
            unsigned short rankWorld = super::_p.getRank();
            unsigned short between = super::_p.getSigned();
            unsigned short* infoNeighbor;
            if(!U) {
                infoNeighbor = new unsigned short[super::_map.size()];
                std::vector<unsigned short>::const_iterator begin = super::_sparsity.cbegin();
                for(unsigned short i = 0; i < super::_map.size(); ++i) {
                    std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::_sparsity.cend(), super::_map[i].first);
                    infoNeighbor[i] = info[std::distance(super::_sparsity.cbegin(), idx)];
                    begin = idx + 1;
                }
            }
            if(S != 'S') {
                if(!U) {
                    unsigned short size = std::accumulate(infoNeighbor, infoNeighbor + super::_map.size(), super::_local);
                    for(unsigned short i = 0; i < super::_map.size(); ++i)
                        in[i] = new K[size * super::_map[i].second.size()];
                    for(unsigned short i = 0; i < super::_map.size(); ++i) {
                        size = infoNeighbor[i];
                        std::vector<unsigned short>::const_iterator begin = super::_sparsity.cbegin();
                        for(const unsigned short& rank : super::_vecSparsity[i]) {
                            if(rank == rankWorld)
                                size += super::_local;
                            else {
                                std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::_sparsity.cend(), rank);
                                size += info[std::distance(super::_sparsity.cbegin(), idx)];
                                begin = idx + 1;
                            }
                        }
                        if(super::_local) {
                            out[i] = new K[size * super::_map[i].second.size()];
                            MPI_Irecv(out[i], size * super::_map[i].second.size(), Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), rqRecv + i);
                        }
                        else
                            rqRecv[i] = MPI_REQUEST_NULL;
                    }
                }
                else {
                    for(unsigned short i = 0; i < super::_map.size(); ++i) {
                        in[i] = new K[super::_local * (super::_map.size() + 1) * super::_map[i].second.size()];
                        out[i] = new K[super::_local * (super::_vecSparsity[i].size() + 1) * super::_map[i].second.size()];
                        MPI_Irecv(out[i], super::_local * (super::_vecSparsity[i].size() + 1) * super::_map[i].second.size(), Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), rqRecv + i);
                    }
                }
            }
            else {
                if(!U) {
                    unsigned short size = std::accumulate(infoNeighbor, infoNeighbor + super::_map.size(), 0);
                    for(unsigned short i = 0; i < super::_map.size(); ++i) {
                        in[i] = new K[(size + super::_local * (i < between)) * super::_map[i].second.size()];
                        size -= infoNeighbor[i];
                    }
                    for(unsigned short i = 0; i < super::_map.size(); ++i) {
                        size = infoNeighbor[i] * !(i < between) + super::_local;
                        std::vector<unsigned short>::const_iterator end = super::_sparsity.cend();
                        for(std::vector<unsigned short>::const_reverse_iterator rit = super::_vecSparsity[i].rbegin(); *rit > rankWorld; ++rit) {
                            std::vector<unsigned short>::const_iterator idx = std::lower_bound(super::_sparsity.cbegin(), end, *rit);
                            size += info[std::distance(super::_sparsity.cbegin(), idx)];
                            end = idx - 1;
                        }
                        if(super::_local) {
                            out[i] = new K[size * super::_map[i].second.size()];
                            MPI_Irecv(out[i], size * super::_map[i].second.size(), Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), rqRecv + i);
                        }
                        else
                            rqRecv[i] = MPI_REQUEST_NULL;
                    }
                }
                else {
                    for(unsigned short i = 0; i < super::_map.size(); ++i)
                        in[i] = new K[super::_local * (super::_map.size() + (i < between) - i) * super::_map[i].second.size()];
                    for(unsigned short i = 0; i < super::_map.size(); ++i) {
                        unsigned short size = std::distance(std::lower_bound(super::_vecSparsity[i].cbegin(), super::_vecSparsity[i].cend(), rankWorld), super::_vecSparsity[i].cend()) + !(i < between);
                        out[i] = new K[super::_local * size * super::_map[i].second.size()];
                        MPI_Irecv(out[i], super::_local * size * super::_map[i].second.size(), Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), rqRecv + i);
                    }
                }
            }
            MPI_Request* rqMult = new MPI_Request[2 * super::_map.size()];
            unsigned int* offset = new unsigned int[super::_map.size() + 2];
            offset[0] = 0;
            offset[1] = super::_local;
            for(unsigned short i = 2; i < super::_map.size() + 2; ++i)
                offset[i] = offset[i - 1] + (U ? super::_local : infoNeighbor[i - 2]);
            const int nbMult = super::_p.getMult();
            K* mult = new K[offset[super::_map.size() + 1] * nbMult];
            unsigned short* accumulator = new unsigned short[super::_map.size() + 1];
            accumulator[0] = 0;
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                MPI_Irecv(mult + offset[i + 1] * nbMult + accumulator[i] * (U ? super::_local : infoNeighbor[i]), super::_map[i].second.size() * (U ? super::_local : infoNeighbor[i]), Wrapper<K>::mpi_type(), super::_map[i].first, 11, super::_p.Subdomain<K>::getCommunicator(), rqMult + i);
                accumulator[i + 1] = accumulator[i] + super::_map[i].second.size();
            }

            K* tmp = new K[offset[super::_map.size() + 1] * super::_n]();
            const typename Wrapper<K>::ul_type* const m = super::_p.getScaling();
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                for(unsigned short k = 0; k < super::_local; ++k)
                    for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                        tmp[super::_map[i].second[j] + k * super::_n] = (mult[accumulator[i] * super::_local + j + k * super::_map[i].second.size()] = m[super::_map[i].second[j]] * super::_deflation[k][super::_map[i].second[j]]);
                MPI_Isend(mult + accumulator[i] * super::_local, super::_map[i].second.size() * super::_local, Wrapper<K>::mpi_type(), super::_map[i].first, 11, super::_p.Subdomain<K>::getCommunicator(), rqMult + super::_map.size() + i);
            }

            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                int index;
                MPI_Waitany(super::_map.size(), rqMult, &index, MPI_STATUS_IGNORE);
                for(unsigned short k = 0; k < (U ? super::_local : infoNeighbor[index]); ++k)
                    for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                        tmp[super::_map[index].second[j] + (offset[index + 1] + k) * super::_n] = mult[offset[index + 1] * nbMult + accumulator[index] * (U ? super::_local : infoNeighbor[index]) + j + k * super::_map[index].second.size()];
            }

            delete [] accumulator;

            if(offset[super::_map.size() + 1])
                super::_p.applyLocalSchurComplement(tmp, offset[super::_map.size() + 1]);

            MPI_Waitall(super::_map.size(), rqMult + super::_map.size(), MPI_STATUSES_IGNORE);
            delete [] rqMult;
            delete [] mult;

            unsigned int accumulate = 0;
            unsigned short stop = std::distance(super::_sparsity.cbegin(), std::upper_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), rankWorld));
            if(S != 'S') {
                _offsets.reserve(super::_sparsity.size() + 1);
                for(unsigned short i = 0; i < stop; ++i) {
                    _offsets.emplace(super::_sparsity[i], accumulate);
                    accumulate += super::_n * (U ? super::_local : info[i]);
                }
            }
            else
                _offsets.reserve(super::_sparsity.size() + 1 - stop);
            _offsets.emplace(rankWorld, accumulate);
            accumulate += super::_n * super::_local;
            for(unsigned short i = stop; i < super::_sparsity.size(); ++i) {
                _offsets.emplace(super::_sparsity[i], accumulate);
                accumulate += super::_n * (U ? super::_local : info[i]);
            }

            work = new K[accumulate]();

            MPI_Request rq;
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                if(i < between || S != 'S') {
                    accumulate = super::_local;
                    for(unsigned short k = 0; k < super::_local; ++k)
                        for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                            work[_offsets[rankWorld] + super::_map[i].second[j] + k * super::_n] = in[i][k * super::_map[i].second.size() + j] = tmp[super::_map[i].second[j] + k * super::_n];
                }
                else {
                    accumulate = 0;
                    for(unsigned short k = 0; k < super::_local; ++k)
                        for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                            work[_offsets[rankWorld] + super::_map[i].second[j] + k * super::_n] = tmp[super::_map[i].second[j] + k * super::_n];
                }
                for(unsigned short l = S != 'S' ? 0 : std::min(i, between); l < super::_map.size(); ++l) {
                    for(unsigned short k = 0; k < (U ? super::_local : infoNeighbor[l]); ++k)
                        for(unsigned int j = 0; j < super::_map[i].second.size(); ++j) {
                            if(S != 'S' || !(l < std::max(i, between)))
                                work[_offsets[super::_map[l].first] + super::_map[i].second[j] + k * super::_n] = in[i][(accumulate + k) * super::_map[i].second.size() + j] = tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n];
                            else {
                                if(i < between)
                                    in[i][(accumulate + k) * super::_map[i].second.size() + j] = tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n];
                                else
                                    work[_offsets[super::_map[l].first] + super::_map[i].second[j] + k * super::_n] = tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n];
                            }
                        }
                    if(S != 'S' || !(l < i) || i < between)
                        accumulate += U ? super::_local : infoNeighbor[l];
                }
                if(U || infoNeighbor[i]) {
                    MPI_Isend(in[i], super::_map[i].second.size() * accumulate, Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), &rq);
                    rqSend.emplace_back(rq);
                }
            }
            delete [] tmp;
            delete [] offset;
            if(!U)
                delete [] infoNeighbor;
        }
        template<char S, bool U>
        inline void assembleForMaster(K* C, const K* in, const int& coefficients, unsigned short index, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            applyFromNeighbor<S, U>(in, index, arrayC, infoNeighbor);
            if(++_consolidate == super::_map.size()) {
                const typename Wrapper<K>::ul_type* const m = super::_p.getScaling();
                for(unsigned short j = 0; j < coefficients + (S == 'S') * super::_local; ++j)
                    Wrapper<K>::diagv(super::_n, m, arrayC + j * super::_n);
                if(S != 'S')
                    Wrapper<K>::gemm(&transb, &transa, &coefficients, &(super::_local), &(super::_n), &(Wrapper<K>::d__1), arrayC, &(super::_n), *super::_deflation, super::_p.getLDR(), &(Wrapper<K>::d__0), C, &coefficients);
                else
                    for(unsigned short j = 0; j < super::_local; ++j) {
                        int local = coefficients + super::_local - j;
                        Wrapper<K>::gemv(&transb, &(super::_n), &local, &(Wrapper<K>::d__1), arrayC + super::_n * j, &(super::_n), super::_deflation[j], &i__1, &(Wrapper<K>::d__0), C - (j * (j - 1)) / 2 + j * (coefficients + super::_local), &i__1);
                    }
            }
        }
        template<char S, char N, bool U>
        inline void applyFromNeighborMaster(const K* in, unsigned short index, int* I, int* J, K* C, int coefficients, unsigned int offsetI, unsigned int* offsetJ, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            assembleForMaster<S, U>(C, in, coefficients, index, arrayC, infoNeighbor);
            if(_consolidate == super::_map.size()) {
                unsigned short between = std::distance(super::_sparsity.cbegin(), std::lower_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_p.getRank()));
                unsigned int offset = 0;
                if(S != 'S')
                    for(unsigned short k = 0; k < between; ++k)
                        for(unsigned short i = 0; i < super::_local; ++i) {
                            unsigned int l = offset + coefficients * i;
                            for(unsigned short j = 0; j < (U ? super::_local : infoNeighbor[k]); ++j) {
#ifndef HPDDM_CSR_CO
                                I[l + j] = offsetI + i;
#endif
                                J[l + j] = (U ? super::_sparsity[k] * super::_local + (N == 'F') : offsetJ[k]) + j;
                            }
                        offset += U ? super::_local : infoNeighbor[k];
                    }
                else
                    coefficients += super::_local - 1;
                for(unsigned short i = 0; i < super::_local; ++i) {
                    unsigned int l = offset + coefficients * i - (S == 'S') * ((i * (i - 1)) / 2);
                    for(unsigned short j = (S == 'S') * i; j < super::_local; ++j) {
#ifndef HPDDM_CSR_CO
                        I[l + j] = offsetI + i;
#endif
                        J[l + j] = offsetI + j;
                    }
                }
                offset += super::_local;
                for(unsigned short k = between; k < super::_sparsity.size(); ++k) {
                    for(unsigned short i = 0; i < super::_local; ++i) {
                        unsigned int l = offset + coefficients * i - (S == 'S') * ((i * (i - 1)) / 2);
                        for(unsigned short j = 0; j < (U ? super::_local : infoNeighbor[k]); ++j) {
#ifndef HPDDM_CSR_CO
                            I[l + j] = offsetI + i;
#endif
                            J[l + j] = (U ? super::_sparsity[k] * super::_local + (N == 'F') : offsetJ[k - (S == 'S') * between]) + j;
                        }
                    }
                    offset += U ? super::_local : infoNeighbor[k];
                }
            }
        }
};
#endif // HPDDM_BDD
} // HPDDM
#endif // _OPERATOR_
