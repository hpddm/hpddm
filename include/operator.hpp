/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
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

#ifndef _HPDDM_OPERATOR_
#define _HPDDM_OPERATOR_

#include <queue>

namespace HPDDM {
template<bool> class Members { };
template<> class Members<true> {
    protected:
        std::unordered_map<unsigned short, unsigned int> _offsets;
        std::vector<std::vector<unsigned short>>     _vecSparsity;
        const unsigned short                                _rank;
        unsigned short                               _consolidate;
        explicit Members(unsigned short r) : _rank(r), _consolidate() { }
};
template<char P, class Preconditioner, class K>
class OperatorBase : protected Members<P != 's' && P != 'u'> {
    private:
        template<class T>
        class has_LDR {
            private:
                typedef char one;
                typedef one (&two)[2];
                template<class C> static one test(decltype(&C::getLDR));
                template<class C> static two test(...);
            public:
                static constexpr bool value = (sizeof(test<T>(0)) == sizeof(one));
        };
        template<class Q = Preconditioner> typename std::enable_if<has_LDR<typename std::remove_reference<Q>::type>::value, bool>::type
        offsetDeflation() {
            const unsigned int offset = *_p.getLDR() - _n;
            if(_deflation && offset)
                std::for_each(const_cast<K**>(_deflation), const_cast<K**>(_deflation) + _local, [&](K*& v) { v -= offset; });
            return true;
        }
        template<class Q = Preconditioner> typename std::enable_if<!has_LDR<typename std::remove_reference<Q>::type>::value, bool>::type
        offsetDeflation() { return false; }
    protected:
        const Preconditioner&                 _p;
        const K* const* const         _deflation;
        const vectorNeighbor&               _map;
        std::vector<unsigned short>    _sparsity;
        const int                             _n;
        const int                         _local;
        unsigned int                        _max;
        unsigned short                   _signed;
        unsigned short             _connectivity;
        template<char Q = P, typename std::enable_if<Q == 's' || Q == 'u'>::type* = nullptr>
        OperatorBase(const Preconditioner& p, const unsigned short& c, const unsigned int& max) : _p(p), _deflation(p.getVectors()), _map(p.getMap()), _n(p.getDof()), _local(p.getLocal()), _max(max), _connectivity(c) {
            static_assert(Q == P, "Wrong sparsity pattern");
            _sparsity.reserve(_map.size());
            std::transform(_map.cbegin(), _map.cend(), std::back_inserter(_sparsity), [](const pairNeighbor& n) { return n.first; });
        }
        template<char Q = P, typename std::enable_if<Q != 's' && Q != 'u'>::type* = nullptr>
        OperatorBase(const Preconditioner& p, const unsigned short& c, const unsigned int& max) : Members<true>(p.getRank()), _p(p), _deflation(p.getVectors()), _map(p.getMap()), _n(p.getDof()), _local(p.getLocal()), _max(max + std::max(1, (c - 1)) * (max & 4095)), _signed(_p.getSigned()), _connectivity(c) {
            const unsigned int offset = *_p.getLDR() - _n;
            if(_deflation && offset)
                std::for_each(const_cast<K**>(_deflation), const_cast<K**>(_deflation) + _local, [&](K*& v) { v += offset; });
            static_assert(Q == P, "Wrong sparsity pattern");
            if(!_map.empty()) {
                unsigned short** recvSparsity = new unsigned short*[_map.size() + 1];
                *recvSparsity = new unsigned short[(_connectivity + 1) * _map.size()];
                unsigned short* sendSparsity = *recvSparsity + _connectivity * _map.size();
                MPI_Request* rq = _p.getRq();
                for(unsigned short i = 0; i < _map.size(); ++i) {
                    sendSparsity[i] = _map[i].first;
                    recvSparsity[i] = *recvSparsity + _connectivity * i;
                    MPI_Irecv(recvSparsity[i], _connectivity, MPI_UNSIGNED_SHORT, _map[i].first, 4, _p.getCommunicator(), rq + i);
                }
                for(unsigned short i = 0; i < _map.size(); ++i)
                    MPI_Isend(sendSparsity, _map.size(), MPI_UNSIGNED_SHORT, _map[i].first, 4, _p.getCommunicator(), rq + _map.size() + i);
                Members<true>::_vecSparsity.resize(_map.size());
                for(unsigned short i = 0; i < _map.size(); ++i) {
                    int index, count;
                    MPI_Status status;
                    MPI_Waitany(_map.size(), rq, &index, &status);
                    MPI_Get_count(&status, MPI_UNSIGNED_SHORT, &count);
                    Members<true>::_vecSparsity[index].assign(recvSparsity[index], recvSparsity[index] + count);
                }
                MPI_Waitall(_map.size(), rq + _map.size(), MPI_STATUSES_IGNORE);

                delete [] *recvSparsity;
                delete [] recvSparsity;

                _sparsity.reserve(_map.size());
                if(P == 'c') {
                    std::vector<unsigned short> neighbors;
                    neighbors.reserve(_map.size());
                    std::for_each(_map.cbegin(), _map.cend(), [&](const pairNeighbor& n) { neighbors.emplace_back(n.first); });
                    typedef std::pair<std::vector<unsigned short>::const_iterator, std::vector<unsigned short>::const_iterator> pairIt;
                    auto comp = [](const pairIt& lhs, const pairIt& rhs) { return *lhs.first > *rhs.first; };
                    std::priority_queue<pairIt, std::vector<pairIt>, decltype(comp)> pq(comp);
                    pq.push({ neighbors.cbegin(), neighbors.cend() });
                    for(const std::vector<unsigned short>& v : Members<true>::_vecSparsity)
                        pq.push({ v.cbegin(), v.cend() });
                    while(!pq.empty()) {
                        pairIt p = pq.top();
                        pq.pop();
                        if(*p.first != Members<true>::_rank && (_sparsity.empty() || (*p.first != _sparsity.back())))
                            _sparsity.emplace_back(*p.first);
                        if(++p.first != p.second)
                            pq.push(p);
                    }
                }
                else {
                    std::transform(_map.cbegin(), _map.cend(), std::back_inserter(_sparsity), [](const pairNeighbor& n) { return n.first; });
                    for(std::vector<unsigned short>& v : Members<true>::_vecSparsity) {
                        unsigned short i = 0, j = 0, k = 0;
                        while(i < v.size() && j < _sparsity.size()) {
                            if(v[i] == Members<true>::_rank) {
                                v[k++] = Members<true>::_rank;
                                ++i;
                            }
                            else if(v[i] < _sparsity[j])
                                ++i;
                            else if(v[i] > _sparsity[j])
                                ++j;
                            else {
                                v[k++] = _sparsity[j++];
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
                    infoNeighbor = new unsigned short[_map.size()];
                    std::vector<unsigned short>::const_iterator begin = _sparsity.cbegin();
                    for(unsigned short i = 0; i < _map.size(); ++i) {
                        std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, _sparsity.cend(), _map[i].first);
                        infoNeighbor[i] = info[std::distance(_sparsity.cbegin(), idx)];
                        begin = idx + 1;
                    }
                }
                else
                    infoNeighbor = const_cast<unsigned short*>(info);
            }
            std::vector<unsigned int> displs;
            displs.reserve(2 * _map.size());
            if(S != 'S') {
                if(!U) {
                    unsigned short size = std::accumulate(infoNeighbor, infoNeighbor + _map.size(), _local);
                    if(!_map.empty())
                        displs.emplace_back(size * _map[0].second.size());
                    for(unsigned short i = 1; i < _map.size(); ++i)
                        displs.emplace_back(displs.back() + size * _map[i].second.size());
                    for(unsigned short i = 0; i < _map.size(); ++i) {
                        size = infoNeighbor[i];
                        std::vector<unsigned short>::const_iterator begin = _sparsity.cbegin();
                        for(const unsigned short& rank : Members<true>::_vecSparsity[i]) {
                            if(rank == Members<true>::_rank)
                                size += _local;
                            else {
                                std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, _sparsity.cend(), rank);
                                size += info[std::distance(_sparsity.cbegin(), idx)];
                                begin = idx + 1;
                            }
                        }
                        if(_local)
                            displs.emplace_back(displs.back() + size * _map[i].second.size());
                        else
                            rqRecv[i] = MPI_REQUEST_NULL;
                    }
                }
                else {
                    if(!_map.empty())
                        displs.emplace_back(_local * (_map.size() + 1) * _map[0].second.size());
                    for(unsigned short i = 1; i < _map.size(); ++i)
                        displs.emplace_back(displs.back() + _local * (_map.size() + 1) * _map[i].second.size());
                    for(unsigned short i = 0; i < _map.size(); ++i)
                        displs.emplace_back(displs.back() + _local * (Members<true>::_vecSparsity[i].size() + 1) * _map[i].second.size());
                }
            }
            else {
                if(!U) {
                    unsigned short size = std::accumulate(infoNeighbor, infoNeighbor + _map.size(), 0);
                    if(!_map.empty()) {
                        displs.emplace_back((size + _local * (0 < _signed)) * _map[0].second.size());
                        size -= infoNeighbor[0];
                    }
                    for(unsigned short i = 1; i < _map.size(); ++i) {
                        displs.emplace_back(displs.back() + (size + _local * (i < _signed)) * _map[i].second.size());
                        size -= infoNeighbor[i];
                    }
                    for(unsigned short i = 0; i < _map.size(); ++i) {
                        size = infoNeighbor[i] * !(i < _signed) + _local;
                        std::vector<unsigned short>::const_iterator end = _sparsity.cend();
                        for(std::vector<unsigned short>::const_reverse_iterator rit = Members<true>::_vecSparsity[i].rbegin(); *rit > Members<true>::_rank; ++rit) {
                            std::vector<unsigned short>::const_iterator idx = std::lower_bound(_sparsity.cbegin(), end, *rit);
                            size += info[std::distance(_sparsity.cbegin(), idx)];
                            end = idx - 1;
                        }
                        if(_local)
                            displs.emplace_back(displs.back() + size * _map[i].second.size());
                        else
                            rqRecv[i] = MPI_REQUEST_NULL;
                    }
                }
                else {
                    if(!_map.empty())
                        displs.emplace_back(_local * (_map.size() + (0 < _signed)) * _map[0].second.size());
                    for(unsigned short i = 1; i < _map.size(); ++i)
                        displs.emplace_back(displs.back() + _local * (_map.size() + (i < _signed) - i) * _map[i].second.size());
                    for(unsigned short i = 0; i < _map.size(); ++i) {
                        unsigned short size = std::distance(std::lower_bound(Members<true>::_vecSparsity[i].cbegin(), Members<true>::_vecSparsity[i].cend(), Members<true>::_rank), Members<true>::_vecSparsity[i].cend()) + !(i < _signed);
                        displs.emplace_back(displs.back() + _local * size * _map[i].second.size());
                    }
                }
            }
            if(!displs.empty()) {
                *in = new K[displs.back()];
                for(unsigned short i = 1; i < _map.size(); ++i)
                    in[i] = *in + displs[i - 1];
                if(U == 1 || _local)
                    for(unsigned short i = 0; i < _map.size(); ++i) {
                        if(displs[i + _map.size()] != displs[i - 1 + _map.size()]) {
                            out[i] = *in + displs[i - 1 + _map.size()];
                            MPI_Irecv(out[i], displs[i + _map.size()] - displs[i - 1 + _map.size()], Wrapper<K>::mpi_type(), _map[i].first, 2, _p.getCommunicator(), rqRecv + i);
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
            if(Members<true>::_consolidate == _map.size()) {
                unsigned short between = std::distance(_sparsity.cbegin(), std::lower_bound(_sparsity.cbegin(), _sparsity.cend(), _p.getRank()));
                unsigned int offset = 0;
                if(S != 'S')
                    for(unsigned short k = 0; k < between; ++k)
                        for(unsigned short i = 0; i < _local; ++i) {
                            unsigned int l = offset + coefficients * i;
                            for(unsigned short j = 0; j < (U ? _local : infoNeighbor[k]); ++j) {
#ifndef HPDDM_CSR_CO
                                I[l + j] = offsetI + i;
#endif
                                J[l + j] = (U ? _sparsity[k] * _local + (N == 'F') : offsetJ[k]) + j;
                            }
                            offset += U ? _local : infoNeighbor[k];
                        }
                else
                    coefficients += _local - 1;
                for(unsigned short i = 0; i < _local; ++i) {
                    unsigned int l = offset + coefficients * i - (S == 'S') * ((i * (i - 1)) / 2);
                    for(unsigned short j = (S == 'S') * i; j < _local; ++j) {
#ifndef HPDDM_CSR_CO
                        I[l + j] = offsetI + i;
#endif
                        J[l + j] = offsetI + j;
                    }
                }
                offset += _local;
                for(unsigned short k = between; k < _sparsity.size(); ++k) {
                    for(unsigned short i = 0; i < _local; ++i) {
                        unsigned int l = offset + coefficients * i - (S == 'S') * ((i * (i - 1)) / 2);
                        for(unsigned short j = 0; j < (U ? _local : infoNeighbor[k]); ++j) {
#ifndef HPDDM_CSR_CO
                            I[l + j] = offsetI + i;
#endif
                            J[l + j] = (U ? _sparsity[k] * _local + (N == 'F') : offsetJ[k - (S == 'S') * between]) + j;
                        }
                    }
                    offset += U ? _local : infoNeighbor[k];
                }
            }
        }
    public:
        static constexpr char _pattern = (P != 's' && P != 'u') ? 'c' : (P == 's' ? 's' : 'u');
        static constexpr bool _factorize = true;
        template<char, bool>
        void setPattern(int*, const int, const int, const unsigned short* const* const = nullptr, const unsigned short* const = nullptr) const { }
        void adjustConnectivity(const MPI_Comm& comm) {
            if(P == 'c') {
#if 0
                _connectivity *= _connectivity - 1;
#else
                _connectivity = _sparsity.size();
                MPI_Allreduce(MPI_IN_PLACE, &_connectivity, 1, MPI_UNSIGNED_SHORT, MPI_MAX, comm);
#endif
            }
        }
        const std::vector<unsigned short>& getPattern() const { return _sparsity; }
        unsigned short getConnectivity() const { return _connectivity; }
        template<char Q = P, typename std::enable_if<Q != 's'>::type* = nullptr>
        void initialize(unsigned int, K*&, unsigned short) { }
};

template<class Preconditioner, class K>
class UserCoarseOperator : public OperatorBase<'u', Preconditioner, K> {
    private:
        typedef OperatorBase<'u', Preconditioner, K> super;
    public:
        template<template<class> class Solver, char S, class T> friend class CoarseOperator;
        UserCoarseOperator(const Preconditioner& p, const unsigned short& c, const unsigned int& max) : super(p, c, max)  { }
};

#if HPDDM_SCHWARZ
template<class Preconditioner, class K>
class MatrixMultiplication : public OperatorBase<'s', Preconditioner, K> {
    protected:
        const MatrixCSR<K>* const                       _A;
        MatrixCSR<K>*                                   _C;
        K*                                           _work;
        const underlying_type<K>* const                 _D;
    private:
        typedef OperatorBase<'s', Preconditioner, K> super;
        template<bool U>
        void applyFromNeighbor(const K* in, unsigned short index, K*& work, unsigned short* infoNeighbor) {
            int m = U ? super::_local : *infoNeighbor;
            std::fill_n(_work, m * super::_n, K());
            for(unsigned short i = 0; i < m; ++i)
                for(int j = 0; j < super::_map[index].second.size(); ++j)
                    _work[i * super::_n + super::_map[index].second[j]] = _D[super::_map[index].second[j]] * in[i * super::_map[index].second.size() + j];
            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &(super::_local), &m, &(super::_n), &(Wrapper<K>::d__1), *super::_deflation, &(super::_n), _work, &(super::_n), &(Wrapper<K>::d__0), work, &(super::_local));
        }
    public:
        template<template<class> class Solver, char S, class T> friend class CoarseOperator;
        template<typename... Types>
        MatrixMultiplication(const Preconditioner& p, const unsigned short& c, const unsigned int& max, Types... args) : super(p, c, max), _A(p.getMatrix()), _C(), _D(p.getScaling()) { static_assert(sizeof...(Types) == 0, "Wrong constructor"); }
        void initialize(unsigned int k, K*& work, unsigned short s) {
            if(_A->_sym) {
                std::vector<std::vector<std::pair<unsigned int, K>>> v(_A->_n);
                unsigned int nnz = ((_A->_nnz + _A->_n - 1) / _A->_n) * 2;
                std::for_each(v.begin(), v.end(), [&](std::vector<std::pair<unsigned int, K>>& r) { r.reserve(nnz); });
                nnz = 0;
                for(unsigned int i = 0; i < _A->_n; ++i) {
                    const underlying_type<K> scal = _D[i];
                    unsigned int j = _A->_ia[i] - (HPDDM_NUMBERING == 'F');
                    while(j < _A->_ia[i + 1] - (HPDDM_NUMBERING == 'F' ? 2 : 1)) {
                        if(_D[_A->_ja[j] - (HPDDM_NUMBERING == 'F')] > HPDDM_EPS) {
                            v[i].emplace_back(_A->_ja[j], _A->_a[j] * _D[_A->_ja[j] - (HPDDM_NUMBERING == 'F')]);
                            ++nnz;
                        }
                        if(scal > HPDDM_EPS) {
                            v[_A->_ja[j] - (HPDDM_NUMBERING == 'F')].emplace_back(i + (HPDDM_NUMBERING == 'F'), _A->_a[j] * scal);
                            ++nnz;
                        }
                        ++j;
                    }
                    if(i != _A->_ja[j] - (HPDDM_NUMBERING == 'F')) {
                        if(_D[_A->_ja[j] - (HPDDM_NUMBERING == 'F')] > HPDDM_EPS) {
                            v[i].emplace_back(_A->_ja[j], _A->_a[j] * _D[_A->_ja[j] - (HPDDM_NUMBERING == 'F')]);
                            ++nnz;
                        }
                    }
                    if(scal > HPDDM_EPS) {
                        v[_A->_ja[j] - (HPDDM_NUMBERING == 'F')].emplace_back(i + (HPDDM_NUMBERING == 'F'), _A->_a[j] * scal);
                        ++nnz;
                    }
                }
                _C = new MatrixCSR<K>(_A->_n, _A->_n, nnz, false);
                _C->_ia[0] = (Wrapper<K>::I == 'F');
                nnz = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, HPDDM_GRANULARITY)
#endif
                for(unsigned int i = 0; i < _A->_n; ++i)
                    std::sort(v[i].begin(), v[i].end(), [](const std::pair<unsigned int, K>& lhs, const std::pair<unsigned int, K>& rhs) { return lhs.first < rhs.first; });
                for(unsigned int i = 0; i < _A->_n; ++i) {
                    for(const std::pair<unsigned int, K>& p : v[i]) {
                        _C->_ja[nnz] = p.first + (Wrapper<K>::I == 'F' && HPDDM_NUMBERING != Wrapper<K>::I);
                        _C->_a[nnz++] = p.second;
                    }
                    _C->_ia[i + 1] = nnz + (Wrapper<K>::I == 'F');
                }
            }
            else {
                _C = new MatrixCSR<K>(_A->_n, _A->_n, _A->_nnz, false);
                _C->_ia[0] = (Wrapper<K>::I == 'F');
                unsigned int nnz = 0;
                for(unsigned int i = 0; i < _A->_n; ++i) {
                    for(unsigned int j = _A->_ia[i] - (HPDDM_NUMBERING == 'F'); j < _A->_ia[i + 1] - (HPDDM_NUMBERING == 'F'); ++j)
                        if(_D[_A->_ja[j] - (HPDDM_NUMBERING == 'F')] > HPDDM_EPS) {
                            _C->_ja[nnz] = _A->_ja[j] + (Wrapper<K>::I == 'F' && HPDDM_NUMBERING != Wrapper<K>::I);
                            _C->_a[nnz++] = _A->_a[j] * _D[_A->_ja[j] - (HPDDM_NUMBERING == 'F')];
                        }
                    _C->_ia[i + 1] = nnz + (Wrapper<K>::I == 'F');
                }
                _C->_nnz = nnz;
            }
            work = new K[2 * k];
            _work = work + k;
            super::_signed = s;
        }
        template<char S, bool U, class T>
        void applyToNeighbor(T& in, K*& work, MPI_Request*& rq, const unsigned short* info, T = nullptr, MPI_Request* = nullptr) {
            Wrapper<K>::template csrmm<Wrapper<K>::I>(false, &(super::_n), &(super::_local), _C->_a, _C->_ia, _C->_ja, *super::_deflation, _work);
            delete _C;
            for(unsigned short i = 0; i < super::_signed; ++i) {
                if(U || info[i]) {
                    for(unsigned short j = 0; j < super::_local; ++j)
                        Wrapper<K>::gthr(super::_map[i].second.size(), _work + j * super::_n, in[i] + j * super::_map[i].second.size(), super::_map[i].second.data());
                    MPI_Isend(in[i], super::_map[i].second.size() * super::_local, Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), rq++);
                }
            }
            Wrapper<K>::diag(super::_n, _D, _work, work, super::_local);
        }
        template<char S, bool U>
        void assembleForMaster(K* C, const K* in, const int& coefficients, unsigned short index, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            applyFromNeighbor<U>(in, index, arrayC, infoNeighbor);
            if(S != 'B')
                for(unsigned short j = 0; j < (U ? super::_local : *infoNeighbor); ++j) {
                    K* pt = C + j;
                    for(unsigned short i = 0; i < super::_local; pt += coefficients - (S == 'S') * i++)
                        *pt = arrayC[j * super::_local + i];
                }
        }
        template<char S, char N, bool U>
        void applyFromNeighborMaster(const K* in, unsigned short index, int* I, int* J, K* C, int coefficients, unsigned int offsetI, unsigned int* offsetJ, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
#ifdef HPDDM_CSR_CO
            (void)I; (void)offsetI;
#endif
            applyFromNeighbor<U>(in, index, S == 'B' && N == 'F' ? C : arrayC, infoNeighbor);
            unsigned int offset = (S == 'B' ? super::_map[index].first + (N == 'F') : (U ? super::_map[index].first * super::_local + (N == 'F') : *offsetJ));
            if(S == 'B') {
                *J = offset;
                if(N == 'C')
                    Wrapper<K>::template omatcopy<'T'>(super::_local, super::_local, arrayC, super::_local, C, super::_local);
            }
            else
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
template<class Preconditioner, class K>
class MatrixAccumulation : public MatrixMultiplication<Preconditioner, K> {
    private:
        std::vector<K>&                                                                                          _overlap;
        std::vector<std::vector<std::pair<unsigned short, unsigned short>>>&                                   _reduction;
        std::map<std::pair<unsigned short, unsigned short>, unsigned short>&                                       _sizes;
        std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>& _extra;
        typedef MatrixMultiplication<Preconditioner, K> super;
        int*                                   _ldistribution;
        int                                             _size;
        unsigned int                                      _sb;
    public:
        template<template<class> class Solver, char S, class T> friend class CoarseOperator;
        template<typename First, typename Second, typename Third, typename Fourth, typename Fifth, typename... Rest>
        MatrixAccumulation(const Preconditioner& p, const unsigned short& c, const unsigned int& max, First& arg1, Second& arg2, Third& arg3, Fourth& arg4, Fifth& arg5, Rest&... args) : super(p, c, max, args...), _overlap(arg2), _reduction(arg3), _sizes(arg4), _extra(arg5) { static_assert(std::is_same<typename std::remove_pointer<First>::type, typename Preconditioner::super::co_type>::value, "Wrong constructor"); }
        static constexpr bool _factorize = false;
        int getMaster(const int rank) const {
            int* it = std::lower_bound(_ldistribution, _ldistribution + _size, rank);
            if(it == _ldistribution + _size)
                return _size - 1;
            else {
                if(*it != rank && it != _ldistribution)
                    --it;
                return std::distance(_ldistribution, it);
            }
        }
        template<char S, bool U>
        void setPattern(int* ldistribution, const int p, const int sizeSplit, unsigned short* const* const split = nullptr, const unsigned short* const world = nullptr) {
            _ldistribution = ldistribution;
            _size = p;
            int rank, size;
            MPI_Comm_rank(super::_p.getCommunicator(), &rank);
            MPI_Comm_size(super::_p.getCommunicator(), &size);
            char* pattern = new char[size * size]();
            for(unsigned short i = 0; i < super::_map.size(); ++i)
                pattern[rank * size + super::_map[i].first] = 1;
            MPI_Allreduce(MPI_IN_PLACE, pattern, size * size, MPI_CHAR, MPI_SUM, super::_p.getCommunicator());
            if(split) {
                int self = getMaster(rank);
                _reduction.resize(sizeSplit);
                for(unsigned short i = 0; i < sizeSplit; ++i) {
                    for(unsigned short j = 0; j < split[i][0]; ++j) {
                        if(getMaster(split[i][(U != 1 ? 3 : 1) + j]) != self) {
                            _sizes[std::make_pair(split[i][(U != 1 ? 3 : 1) + j], split[i][(U != 1 ? 3 : 1) + j])] = (U != 1 ? world[split[i][(U != 1 ? 3 : 1) + j]] : super::_local);
                            _reduction[i].emplace_back(split[i][(U != 1 ? 3 : 1) + j], split[i][(U != 1 ? 3 : 1) + j]);
                            if(S == 'S' && split[i][(U != 1 ? 3 : 1) + j] < rank + i) {
                                std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>::iterator it = _extra.find(i);
                                if(it == _extra.end()) {
                                    std::pair<std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>::iterator, bool> p = _extra.emplace(i, std::forward_as_tuple(0, 0, std::vector<unsigned short>()));
                                    it = p.first;
                                    std::get<0>(it->second) = (U != 1 ? world[rank + i] : 1);
                                    std::get<1>(it->second) = (U != 1 ? std::accumulate(world + rank, world + rank + i, 0) : i);
                                }
                                std::get<2>(it->second).emplace_back(split[i][(U != 1 ? 3 : 1) + j]);
                            }
                            for(unsigned short k = j + 1; k < split[i][0]; ++k) {
                                if(pattern[split[i][(U != 1 ? 3 : 1) + j] * size + split[i][(U != 1 ? 3 : 1) + k]] && getMaster(split[i][(U != 1 ? 3 : 1) + k]) != self) {
                                    _sizes[std::make_pair(split[i][(U != 1 ? 3 : 1) + j], split[i][(U != 1 ? 3 : 1) + k])] = (U != 1 ? world[split[i][(U != 1 ? 3 : 1) + k]] : super::_local);
                                    if(S != 'S')
                                        _sizes[std::make_pair(split[i][(U != 1 ? 3 : 1) + k], split[i][(U != 1 ? 3 : 1) + j])] = (U != 1 ? world[split[i][(U != 1 ? 3 : 1) + j]] : super::_local);
                                    _reduction[i].emplace_back(split[i][(U != 1 ? 3 : 1) + j], split[i][(U != 1 ? 3 : 1) + k]);
                                    if(S != 'S')
                                        _reduction[i].emplace_back(split[i][(U != 1 ? 3 : 1) + k], split[i][(U != 1 ? 3 : 1) + j]);
                                }
                            }
                        }
                    }
                    std::sort(_reduction[i].begin(), _reduction[i].end());
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
            _sb = k;
            work = new K[2 * _sb];
            if(HPDDM_NUMBERING != Wrapper<K>::I) {
                if(HPDDM_NUMBERING == 'F') {
                    std::for_each(super::_A->_ja, super::_A->_ja + super::_A->_nnz, [](int& i) { --i; });
                    std::for_each(super::_A->_ia, super::_A->_ia + super::_A->_n + 1, [](int& i) { --i; });
                }
                else {
                    std::for_each(super::_A->_ja, super::_A->_ja + super::_A->_nnz, [](int& i) { ++i; });
                    std::for_each(super::_A->_ia, super::_A->_ia + super::_A->_n + 1, [](int& i) { ++i; });
                }
            }
            super::_work = work + _sb;
            std::fill_n(super::_work, _sb, K());
            super::_signed = s;
        }
        template<char S, bool U, class T>
        void applyToNeighbor(T& in, K*& work, MPI_Request*& rs, const unsigned short* info, T = nullptr, MPI_Request* = nullptr) {
            std::pair<int, int>** block = nullptr;
            std::vector<unsigned short> masters, interior, overlap, extraSend, extraRecv;
            int rank;
            MPI_Comm_rank(super::_p.getCommunicator(), &rank);
            const unsigned short master = getMaster(rank);
            masters.reserve(super::_map.size());
            for(unsigned short i = 0; i < super::_map.size(); ++i)
                masters.emplace_back(getMaster(super::_map[i].first));
            if(super::_map.size()) {
                block = new std::pair<int, int>*[super::_map.size()];
                *block = new std::pair<int, int>[super::_map.size() * super::_map.size()]();
                for(unsigned short i = 1; i < super::_map.size(); ++i)
                    block[i] = *block + i * super::_map.size();
            }
            std::vector<unsigned short>* accumulate = new std::vector<unsigned short>[S == 'S' ? 2 * super::_map.size() : super::_map.size()];
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                accumulate[i].resize(super::_connectivity);
                if(S == 'S')
                    accumulate[super::_map.size() + i].reserve(super::_connectivity);
            }
            MPI_Request* rq = new MPI_Request[2 * super::_map.size()];
            unsigned short* neighbors = new unsigned short[super::_map.size()];
            for(unsigned short i = 0; i < super::_map.size(); ++i)
                neighbors[i] = super::_map[i].first;
            for(unsigned short i = 0; i < super::_map.size(); ++i)
                MPI_Isend(neighbors, super::_map.size(), MPI_UNSIGNED_SHORT, super::_map[i].first, 123, super::_p.getCommunicator(), rq + super::_map.size() + i);
            for(unsigned short i = 0; i < super::_map.size(); ++i)
                MPI_Irecv(accumulate[i].data(), super::_connectivity, MPI_UNSIGNED_SHORT, super::_map[i].first, 123, super::_p.getCommunicator(), rq + i);
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                int index;
                MPI_Status st;
                MPI_Waitany(super::_map.size(), rq, &index, &st);
                int count;
                MPI_Get_count(&st, MPI_UNSIGNED_SHORT, &count);
                accumulate[index].resize(count);
            }
            int m = 0;
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                if(masters[i] != master) {
                    overlap.emplace_back(i);
                    block[i][i] = std::make_pair(U == 1 ? super::_local : info[i], U == 1 ? super::_local : info[i]);
                    if(block[i][i].first != 0)
                        m += (U == 1 ? super::_local * super::_local : info[i] * info[i]);
                    for(unsigned short j = (S != 'S' ? 0 : i + 1); j < super::_map.size(); ++j) {
                        if(i != j && masters[j] != master && std::binary_search(accumulate[j].cbegin(), accumulate[j].cend(), super::_map[i].first)) {
                            block[i][j] = std::make_pair(U == 1 ? super::_local : info[i], U == 1 ? super::_local : info[j]);
                            if(block[i][j].first != 0 && block[i][j].second != 0)
                                m += (U == 1 ? super::_local * super::_local : info[i] * info[j]);
                        }
                    }
                    if(S == 'S' && i < super::_signed)
                        m += (U == 1 ? super::_local * super::_local : super::_local * info[i]);
                }
                else
                    interior.emplace_back(i);
            }
            for(unsigned short i = 0; i < overlap.size(); ++i) {
                unsigned short size = 0;
                for(unsigned short j = 0; j < accumulate[overlap[i]].size(); ++j) {
                    if(getMaster(accumulate[overlap[i]][j]) == masters[overlap[i]]) {
                        unsigned short* pt = std::lower_bound(neighbors, neighbors + super::_map.size(), accumulate[overlap[i]][j]);
                        if(pt != neighbors + super::_map.size() && *pt == accumulate[overlap[i]][j])
                            accumulate[overlap[i]][size++] = std::distance(neighbors, pt);
                    }
                    if(S == 'S' && getMaster(accumulate[overlap[i]][j]) == master) {
                        unsigned short* pt = std::lower_bound(neighbors, neighbors + super::_map.size(), accumulate[overlap[i]][j]);
                        if(pt != neighbors + super::_map.size() && *pt == accumulate[overlap[i]][j])
                            accumulate[super::_map.size() + overlap[i]].emplace_back(std::distance(neighbors, pt));
                    }
                }
                accumulate[overlap[i]].resize(size);
            }
            MPI_Waitall(super::_map.size(), rq + super::_map.size(), MPI_STATUSES_IGNORE);
            delete [] neighbors;
            _overlap.resize(m);
            std::vector<int> omap;
            {
                std::set<int> o;
                for(unsigned short i = 0; i < super::_map.size(); ++i)
                    o.insert(super::_map[i].second.cbegin(), super::_map[i].second.cend());
                omap.reserve(o.size());
                std::copy(o.cbegin(), o.cend(), std::back_inserter(omap));
                int* ia = new int[omap.size() + 1];
                ia[0] = (Wrapper<K>::I == 'F');
                std::vector<std::pair<int, K>> restriction;
                restriction.reserve(super::_A->_nnz);
                int nnz = ia[0];
                for(int i = 0; i < omap.size(); ++i) {
                    std::vector<int>::const_iterator it = omap.cbegin();
                    for(int j = super::_A->_ia[omap[i]] - (Wrapper<K>::I == 'F'); j < super::_A->_ia[omap[i] + 1] - (Wrapper<K>::I == 'F'); ++j) {
                        it = std::lower_bound(it, omap.cend(), super::_A->_ja[j] - (Wrapper<K>::I == 'F'));
                        if(it != omap.cend() && *it == super::_A->_ja[j] - (Wrapper<K>::I == 'F') && std::abs(super::_A->_a[j]) > HPDDM_EPS) {
                            restriction.emplace_back(std::distance(omap.cbegin(), it) + (Wrapper<K>::I == 'F'), super::_A->_a[j]);
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
                super::_C = new MatrixCSR<K>(omap.size(), omap.size(), nnz - (Wrapper<K>::I == 'F'), a, ia, ja, super::_A->_sym, true);
            }
            K** tmp = new K*[2 * super::_map.size()];
            *tmp = new K[omap.size() * (U == 1 ? super::_map.size() * super::_local : std::max(super::_local, std::accumulate(info, info + super::_map.size(), 0))) + std::max(static_cast<int>(omap.size() * (U == 1 ? super::_map.size() * super::_local : std::max(super::_local, std::accumulate(info, info + super::_map.size(), 0)))), super::_n * super::_local)]();
            for(unsigned short i = 1; i < super::_map.size(); ++i)
                tmp[i] = tmp[i - 1] + omap.size() * (U == 1 ? super::_local : info[i - 1]);
            if(super::_map.size()) {
                tmp[super::_map.size()] = tmp[super::_map.size() - 1] + omap.size() * (U == 1 ? super::_local : info[super::_map.size() - 1]);
                for(unsigned short i = 1; i < super::_map.size(); ++i)
                    tmp[super::_map.size() + i] = tmp[super::_map.size() + i - 1] + std::distance(tmp[i - 1], tmp[i]);
            }
            std::vector<std::vector<int>> nmap(super::_map.size());
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                nmap[i].reserve(super::_map[i].second.size());
                for(unsigned int j = 0; j < super::_map[i].second.size(); ++j) {
                    std::vector<int>::const_iterator it = std::lower_bound(omap.cbegin(), omap.cend(), super::_map[i].second[j]);
                    nmap[i].emplace_back(std::distance(omap.cbegin(), it));
                }
            }
            K** buff = new K*[2 * super::_map.size()];
            m = 0;
            for(unsigned short i = 0; i < super::_map.size(); ++i)
                m += super::_map[i].second.size() * 2 * (U == 1 ? super::_local : std::max(static_cast<unsigned short>(super::_local), info[i]));
            *buff = new K[m];
            m = 0;
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                buff[i] = *buff + m;
                MPI_Irecv(buff[i], super::_map[i].second.size() * (U == 1 ? super::_local : info[i]), Wrapper<K>::mpi_type(), super::_map[i].first, 20, super::_p.getCommunicator(), rq + i);
                m += super::_map[i].second.size() * (U == 1 ? super::_local : std::max(static_cast<unsigned short>(super::_local), info[i]));
            }
            Wrapper<K>::diag(super::_n, super::_D, *super::_deflation, *tmp, super::_local);
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                buff[super::_map.size() + i] = *buff + m;
                for(unsigned short j = 0; j < super::_local; ++j)
                    Wrapper<K>::gthr(super::_map[i].second.size(), *tmp + j * super::_n, buff[super::_map.size() + i] + j * super::_map[i].second.size(), super::_map[i].second.data());
                MPI_Isend(buff[super::_map.size() + i], super::_map[i].second.size() * super::_local, Wrapper<K>::mpi_type(), super::_map[i].first, 20, super::_p.getCommunicator(), rq + super::_map.size() + i);
                m += super::_map[i].second.size() * (U == 1 ? super::_local : std::max(static_cast<unsigned short>(super::_local), info[i]));
            }
            Wrapper<K>::template csrmm<Wrapper<K>::I>(super::_A->_sym, &(super::_n), &(super::_local), super::_A->_a, super::_A->_ia, super::_A->_ja, *tmp, super::_work);
            std::fill_n(*tmp, super::_local * super::_n, K());
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                int index;
                MPI_Waitany(super::_map.size(), rq, &index, MPI_STATUS_IGNORE);
                for(unsigned short k = 0; k < (U ? super::_local : info[index]); ++k)
                    Wrapper<K>::sctr(nmap[index].size(), buff[index] + k * nmap[index].size(), nmap[index].data(), tmp[index] + k * omap.size());
            }
            for(unsigned short i = 0; i < super::_map.size(); ++i)
                MPI_Irecv(buff[i], super::_map[i].second.size() * super::_local, Wrapper<K>::mpi_type(), super::_map[i].first, 21, super::_p.getCommunicator(), rq + i);
            m = std::distance(tmp[0], tmp[super::_map.size()]) / omap.size();
            {
                std::vector<unsigned short>* compute = new std::vector<unsigned short>[super::_C->_n]();
                for(unsigned short i = 0; i < super::_map.size(); ++i)
                    for(unsigned int j = 0; j < super::_map[i].second.size(); ++j) {
                        std::vector<int>::const_iterator it = std::lower_bound(omap.cbegin(), omap.cend(), super::_map[i].second[j]);
                        compute[std::distance(omap.cbegin(), it)].emplace_back(i);
                    }
                std::fill_n(tmp[super::_map.size()], super::_C->_n * m, K());
                for(int i = 0; i < super::_C->_n; ++i) {
                    for(int j = super::_C->_ia[i] - (Wrapper<K>::I == 'F'); j < super::_C->_ia[i + 1] - (Wrapper<K>::I == 'F'); ++j) {
                        for(unsigned short k = 0; k < compute[super::_C->_ja[j] - (Wrapper<K>::I == 'F')].size(); ++k) {

                            const int m = (U == 1 ? super::_local : info[compute[super::_C->_ja[j] - (Wrapper<K>::I == 'F')][k]]);
                            Blas<K>::axpy(&m, super::_C->_a + j, tmp[compute[super::_C->_ja[j] - (Wrapper<K>::I == 'F')][k]] + super::_C->_ja[j] - (Wrapper<K>::I == 'F'), &(super::_C->_n), tmp[super::_map.size() + compute[super::_C->_ja[j] - (Wrapper<K>::I == 'F')][k]] + i, &(super::_C->_n));
                            if(super::_C->_sym && i != super::_C->_ja[j] - (Wrapper<K>::I == 'F')) {

                                const int m = (U == 1 ? super::_local : info[compute[super::_C->_ja[j] - (Wrapper<K>::I == 'F')][k]]);
                                Blas<K>::axpy(&m, super::_C->_a + j, tmp[compute[super::_C->_ja[j] - (Wrapper<K>::I == 'F')][k]] + i, &(super::_C->_n), tmp[super::_map.size() + compute[super::_C->_ja[j] - (Wrapper<K>::I == 'F')][k]] + super::_C->_ja[j] - (Wrapper<K>::I == 'F'), &(super::_C->_n));
                            }
                        }
                    }
                }
                delete [] compute;
            }
            delete super::_C;
            MPI_Waitall(super::_map.size(), rq + super::_map.size(), MPI_STATUSES_IGNORE);
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                m = (U == 1 ? super::_local : info[i]);
                for(unsigned short j = 0; j < m; ++j)
                    Wrapper<K>::gthr(nmap[i].size(), tmp[super::_map.size() + i] + j * omap.size(), buff[super::_map.size() + i] + j * nmap[i].size(), nmap[i].data());
                MPI_Isend(buff[super::_map.size() + i], super::_map[i].second.size() * m, Wrapper<K>::mpi_type(), super::_map[i].first, 21, super::_p.getCommunicator(), rq + super::_map.size() + i);
            }
            K* pt = _overlap.data();
            for(unsigned short i = 0; i < overlap.size(); ++i) {
                for(unsigned short j = 0; j < overlap.size(); ++j) {
                    if(block[overlap[i]][overlap[j]].first != 0 && block[overlap[i]][overlap[j]].second != 0) {
                        const int n = omap.size();
                        Blas<K>::gemm(&(Wrapper<K>::transc), "N", &(block[overlap[i]][overlap[j]].first), &(block[overlap[i]][overlap[j]].second), &n, &(Wrapper<K>::d__1), tmp[overlap[i]], &n, tmp[super::_map.size() + overlap[j]], &n, &(Wrapper<K>::d__0), pt, &(block[overlap[i]][overlap[j]].first));
                        pt += (U == 1 ? super::_local * super::_local : info[overlap[i]] * info[overlap[j]]);
                    }
                }
            }
            if(block) {
                delete [] *block;
                delete [] block;
            }
            if(HPDDM_NUMBERING != Wrapper<K>::I) {
                if(Wrapper<K>::I == 'F') {
                    std::for_each(super::_A->_ja, super::_A->_ja + super::_A->_nnz, [](int& i) { --i; });
                    std::for_each(super::_A->_ia, super::_A->_ia + super::_A->_n + 1, [](int& i) { --i; });
                }
                else {
                    std::for_each(super::_A->_ja, super::_A->_ja + super::_A->_nnz, [](int& i) { ++i; });
                    std::for_each(super::_A->_ia, super::_A->_ia + super::_A->_n + 1, [](int& i) { ++i; });
                }
            }
            MPI_Waitall(super::_map.size(), rq, MPI_STATUSES_IGNORE);
            if(S == 'S') {
                for(unsigned short i = 0; i < overlap.size() && overlap[i] < super::_signed; ++i) {
                    m = (U == 1 ? super::_local : info[overlap[i]]);
                    if(m) {
                        for(unsigned short nu = 0; nu < super::_local; ++nu)
                            Wrapper<K>::gthr(omap.size(), super::_work + nu * super::_n, work + nu * omap.size(), omap.data());
                        const std::vector<unsigned short>& r = accumulate[super::_map.size() + overlap[i]];
                        for(unsigned short j = 0; j < r.size(); ++j) {
                            for(unsigned short nu = 0; nu < super::_local; ++nu) {
                                std::fill_n(tmp[super::_map.size()], omap.size(), K());
                                Wrapper<K>::sctr(nmap[r[j]].size(), buff[r[j]] + nu * nmap[r[j]].size(), nmap[r[j]].data(), tmp[super::_map.size()]);
                                for(unsigned int k = 0; k < nmap[overlap[i]].size(); ++k)
                                    work[nmap[overlap[i]][k] + nu * omap.size()] += tmp[super::_map.size()][nmap[overlap[i]][k]];
                            }
                        }
                        const int n = omap.size();
                        Blas<K>::gemm(&(Wrapper<K>::transc), "N", &(super::_local), &m, &n, &(Wrapper<K>::d__1), work, &n, tmp[overlap[i]], &n, &(Wrapper<K>::d__0), pt, &(super::_local));
                        pt += super::_local * m;
                    }
                }
            }
            delete [] *tmp;
            delete [] tmp;
            for(unsigned short i = 0; i < overlap.size() && overlap[i] < super::_signed; ++i) {
                if(U || info[overlap[i]]) {
                    for(unsigned short nu = 0; nu < super::_local; ++nu)
                        Wrapper<K>::gthr(super::_map[overlap[i]].second.size(), super::_work + nu * super::_n, in[overlap[i]] + nu * super::_map[overlap[i]].second.size(), super::_map[overlap[i]].second.data());
                    const std::vector<unsigned short>& r = accumulate[overlap[i]];
                    for(unsigned short k = 0; k < r.size(); ++k) {
                        for(unsigned short nu = 0; nu < super::_local; ++nu) {
                            std::fill_n(work, omap.size(), K());
                            Wrapper<K>::sctr(nmap[r[k]].size(), buff[r[k]] + nu * nmap[r[k]].size(), nmap[r[k]].data(), work);
                            for(unsigned int j = 0; j < super::_map[overlap[i]].second.size(); ++j)
                                in[overlap[i]][j + nu * super::_map[overlap[i]].second.size()] += work[nmap[overlap[i]][j]];
                        }
                    }
                    MPI_Isend(in[overlap[i]], super::_map[overlap[i]].second.size() * super::_local, Wrapper<K>::mpi_type(), super::_map[overlap[i]].first, 2, super::_p.getCommunicator(), rs + overlap[i]);
                }
            }
            for(unsigned short i = 0; i < interior.size(); ++i) {
                for(unsigned short k = 0; k < super::_local; ++k)
                    for(unsigned int j = 0; j < super::_map[interior[i]].second.size(); ++j)
                        super::_work[super::_map[interior[i]].second[j] + k * super::_n] += buff[interior[i]][j + k * super::_map[interior[i]].second.size()];
            }
            for(unsigned short i = 0; i < interior.size() && interior[i] < super::_signed; ++i) {
                if(U || info[interior[i]]) {
                    for(unsigned short nu = 0; nu < super::_local; ++nu)
                        Wrapper<K>::gthr(super::_map[interior[i]].second.size(), super::_work + nu * super::_n, in[interior[i]] + nu * super::_map[interior[i]].second.size(), super::_map[interior[i]].second.data());
                    MPI_Isend(in[interior[i]], super::_map[interior[i]].second.size() * super::_local, Wrapper<K>::mpi_type(), super::_map[interior[i]].first, 2, super::_p.getCommunicator(), rs + interior[i]);
                }
            }
            rs += super::_signed;
            delete [] accumulate;
            Wrapper<K>::diag(super::_n, super::_D, super::_work, work, super::_local);
            MPI_Waitall(super::_map.size(), rq + super::_map.size(), MPI_STATUSES_IGNORE);
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
            std::vector<unsigned short>::const_iterator middle = std::lower_bound(super::_vecSparsity[index].cbegin(), super::_vecSparsity[index].cend(), super::_rank);
            unsigned int accumulate = 0;
            if(!(index < super::_signed)) {
                for(unsigned short k = 0; k < (U ? super::_local : info[std::distance(super::_sparsity.cbegin(), std::lower_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_map[index].first))]); ++k)
                    for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                        work[super::_offsets[super::_map[index].first] + super::_map[index].second[j] + k * super::_n] += in[k * super::_map[index].second.size() + j];
                accumulate += (U ? super::_local : info[std::distance(super::_sparsity.cbegin(), std::lower_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_map[index].first))]) * super::_map[index].second.size();
            }
            else if(S != 'S') {
                for(unsigned short k = 0; k < (U ? super::_local : info[std::distance(super::_sparsity.cbegin(), std::lower_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_map[index].first))]); ++k)
                    for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                        work[super::_offsets[super::_map[index].first] + super::_map[index].second[j] + k * super::_n] -= in[k * super::_map[index].second.size() + j];
                accumulate += (U ? super::_local : info[std::distance(super::_sparsity.cbegin(), std::lower_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_map[index].first))]) * super::_map[index].second.size();
            }
            std::vector<unsigned short>::const_iterator begin = super::_sparsity.cbegin();
            if(S != 'S')
                for(std::vector<unsigned short>::const_iterator it = super::_vecSparsity[index].cbegin(); it != middle; ++it) {
                    if(!U) {
                        std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::_sparsity.cend(), *it);
                        if(*it > super::_map[index].first || super::_signed > index)
                            for(unsigned short k = 0; k < info[std::distance(super::_sparsity.cbegin(), idx)]; ++k)
                                for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                    work[super::_offsets[*it] + super::_map[index].second[j] + k * super::_n] -= in[accumulate + k * super::_map[index].second.size() + j];
                        else
                            for(unsigned short k = 0; k < info[std::distance(super::_sparsity.cbegin(), idx)]; ++k)
                                for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                    work[super::_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                        accumulate += info[std::distance(super::_sparsity.cbegin(), idx)] * super::_map[index].second.size();
                        begin = idx + 1;
                    }
                    else {
                        if(*it > super::_map[index].first || super::_signed > index)
                            for(unsigned short k = 0; k < super::_local; ++k)
                                for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                    work[super::_offsets[*it] + super::_map[index].second[j] + k * super::_n] -= in[accumulate + k * super::_map[index].second.size() + j];
                        else
                            for(unsigned short k = 0; k < super::_local; ++k)
                                for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                    work[super::_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                        accumulate += super::_local * super::_map[index].second.size();
                    }
                }
            if(index < super::_signed)
                for(unsigned short k = 0; k < super::_local; ++k)
                    for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                        work[super::_offsets[super::_rank] + super::_map[index].second[j] + k * super::_n] -= in[accumulate + k * super::_map[index].second.size() + j];
            else
                for(unsigned short k = 0; k < super::_local; ++k)
                    for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                        work[super::_offsets[super::_rank] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
            accumulate += super::_local * super::_map[index].second.size();
            for(std::vector<unsigned short>::const_iterator it = middle + 1; it < super::_vecSparsity[index].cend(); ++it) {
                if(!U) {
                    std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::_sparsity.cend(), *it);
                    if(*it > super::_map[index].first && super::_signed > index)
                        for(unsigned short k = 0; k < info[std::distance(super::_sparsity.cbegin(), idx)]; ++k)
                            for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                work[super::_offsets[*it] + super::_map[index].second[j] + k * super::_n] -= in[accumulate + k * super::_map[index].second.size() + j];
                    else
                        for(unsigned short k = 0; k < info[std::distance(super::_sparsity.cbegin(), idx)]; ++k)
                            for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                work[super::_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                    accumulate += info[std::distance(super::_sparsity.cbegin(), idx)] * super::_map[index].second.size();
                    begin = idx + 1;
                }
                else {
                    if(*it > super::_map[index].first && super::_signed > index)
                        for(unsigned short k = 0; k < super::_local; ++k)
                            for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                work[super::_offsets[*it] + super::_map[index].second[j] + k * super::_n] -= in[accumulate + k * super::_map[index].second.size() + j];
                    else
                        for(unsigned short k = 0; k < super::_local; ++k)
                            for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                work[super::_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                    accumulate += super::_local * super::_map[index].second.size();
                }
            }
        }
    public:
        template<template<class> class Solver, char S, class T> friend class CoarseOperator;
        FetiProjection(const Preconditioner& p, const unsigned short& c, const unsigned int& max) : super(p, c, max) { }
        template<char S, bool U, class T>
        void applyToNeighbor(T& in, K*& work, MPI_Request*& rq, const unsigned short* info, T const& out = nullptr, MPI_Request* const& rqRecv = nullptr) {
            unsigned short* infoNeighbor;
            super::template initialize<S, U>(in, info, out, rqRecv, infoNeighbor);
            MPI_Request* rqMult = new MPI_Request[2 * super::_map.size()];
            unsigned int* offset = new unsigned int[super::_map.size() + 2];
            offset[0] = 0;
            offset[1] = super::_local;
            for(unsigned short i = 2; i < super::_map.size() + 2; ++i)
                offset[i] = offset[i - 1] + (U ? super::_local : infoNeighbor[i - 2]);
            const int nbMult = super::_p.getMult();
            K* mult = new K[offset[super::_map.size() + 1] * nbMult];
            unsigned short* displs = new unsigned short[super::_map.size() + 1];
            displs[0] = 0;
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                MPI_Irecv(mult + offset[i + 1] * nbMult + displs[i] * (U ? super::_local : infoNeighbor[i]), super::_map[i].second.size() * (U ? super::_local : infoNeighbor[i]), Wrapper<K>::mpi_type(), super::_map[i].first, 11, super::_p.getCommunicator(), rqMult + i);
                displs[i + 1] = displs[i] + super::_map[i].second.size();
            }

            K* tmp = new K[offset[super::_map.size() + 1] * super::_n]();
            const underlying_type<K>* const* const m = super::_p.getScaling();
            for(unsigned short i = 0; i < super::_signed; ++i) {
                for(unsigned short k = 0; k < super::_local; ++k)
                    for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                        tmp[super::_map[i].second[j] + k * super::_n] -= m[i][j] * (mult[displs[i] * super::_local + j + k * super::_map[i].second.size()] = - super::_deflation[k][super::_map[i].second[j]]);
                MPI_Isend(mult + displs[i] * super::_local, super::_map[i].second.size() * super::_local, Wrapper<K>::mpi_type(), super::_map[i].first, 11, super::_p.getCommunicator(), rqMult + super::_map.size() + i);
            }
            for(unsigned short i = super::_signed; i < super::_map.size(); ++i) {
                for(unsigned short k = 0; k < super::_local; ++k)
                    for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                        tmp[super::_map[i].second[j] + k * super::_n] += m[i][j] * (mult[displs[i] * super::_local + j + k * super::_map[i].second.size()] =   super::_deflation[k][super::_map[i].second[j]]);
                MPI_Isend(mult + displs[i] * super::_local, super::_map[i].second.size() * super::_local, Wrapper<K>::mpi_type(), super::_map[i].first, 11, super::_p.getCommunicator(), rqMult + super::_map.size() + i);
            }

            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                int index;
                MPI_Waitany(super::_map.size(), rqMult, &index, MPI_STATUS_IGNORE);
                if(index < super::_signed)
                    for(unsigned short k = 0; k < (U ? super::_local : infoNeighbor[index]); ++k)
                        for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                            tmp[super::_map[index].second[j] + (offset[index + 1] + k) * super::_n] = - m[index][j] * mult[offset[index + 1] * nbMult + displs[index] * (U ? super::_local : infoNeighbor[index]) + j + k * super::_map[index].second.size()];
                else
                    for(unsigned short k = 0; k < (U ? super::_local : infoNeighbor[index]); ++k)
                        for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                            tmp[super::_map[index].second[j] + (offset[index + 1] + k) * super::_n] =   m[index][j] * mult[offset[index + 1] * nbMult + displs[index] * (U ? super::_local : infoNeighbor[index]) + j + k * super::_map[index].second.size()];
            }

            delete [] displs;

            if(offset[super::_map.size() + 1])
                super::_p.applyLocalPreconditioner(tmp, offset[super::_map.size() + 1]);

            MPI_Waitall(super::_map.size(), rqMult + super::_map.size(), MPI_STATUSES_IGNORE);
            delete [] rqMult;
            delete [] mult;

            unsigned int accumulate = 0;
            unsigned short stop = std::distance(super::_sparsity.cbegin(), std::upper_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_rank));
            if(S != 'S') {
                super::_offsets.reserve(super::_sparsity.size() + 1);
                for(unsigned short i = 0; i < stop; ++i) {
                    super::_offsets.emplace(super::_sparsity[i], accumulate);
                    accumulate += super::_n * (U ? super::_local : info[i]);
                }
            }
            else
                super::_offsets.reserve(super::_sparsity.size() + 1 - stop);
            super::_offsets.emplace(super::_rank, accumulate);
            accumulate += super::_n * super::_local;
            for(unsigned short i = stop; i < super::_sparsity.size(); ++i) {
                super::_offsets.emplace(super::_sparsity[i], accumulate);
                accumulate += super::_n * (U ? super::_local : info[i]);
            }

            work = new K[accumulate]();

            for(unsigned short i = 0; i < super::_signed; ++i) {
                accumulate = super::_local;
                for(unsigned short k = 0; k < super::_local; ++k)
                    for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                        work[super::_offsets[super::_rank] + super::_map[i].second[j] + k * super::_n] -= (in[i][k * super::_map[i].second.size() + j] = - m[i][j] * tmp[super::_map[i].second[j] + k * super::_n]);
                for(unsigned short l = (S != 'S' ? 0 : i); l < super::_map.size(); ++l) {
                    if(Q == FetiPrcndtnr::SUPERLUMPED && l != i && !std::binary_search(super::_vecSparsity[i].cbegin(), super::_vecSparsity[i].cend(), super::_map[l].first)) {
                        if(S != 'S' || !(l < super::_signed))
                            for(unsigned short k = 0; k < (U ? super::_local : infoNeighbor[l]); ++k)
                                for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                                    work[super::_offsets[super::_map[l].first] + super::_map[i].second[j] + k * super::_n] -= - m[i][j] * tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n];
                        continue;
                    }
                    for(unsigned short k = 0; k < (U ? super::_local : infoNeighbor[l]); ++k)
                        for(unsigned int j = 0; j < super::_map[i].second.size(); ++j) {
                            if(S != 'S' || !(l < super::_signed))
                                work[super::_offsets[super::_map[l].first] + super::_map[i].second[j] + k * super::_n] -= (in[i][(accumulate + k) * super::_map[i].second.size() + j] = - m[i][j] * tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n]);
                            else
                                in[i][(accumulate + k) * super::_map[i].second.size() + j] = - m[i][j] * tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n];
                        }
                    accumulate += U ? super::_local : infoNeighbor[l];
                }
                if(U || infoNeighbor[i])
                    MPI_Isend(in[i], super::_map[i].second.size() * accumulate, Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), rq++);
            }
            for(unsigned short i = super::_signed; i < super::_map.size(); ++i) {
                if(S != 'S') {
                    accumulate = super::_local;
                    for(unsigned short k = 0; k < super::_local; ++k)
                        for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                            work[super::_offsets[super::_rank] + super::_map[i].second[j] + k * super::_n] += (in[i][k * super::_map[i].second.size() + j] =   m[i][j] * tmp[super::_map[i].second[j] + k * super::_n]);
                }
                else {
                    accumulate = 0;
                    for(unsigned short k = 0; k < super::_local; ++k)
                        for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                            work[super::_offsets[super::_rank] + super::_map[i].second[j] + k * super::_n] += m[i][j] * tmp[super::_map[i].second[j] + k * super::_n];
                }
                for(unsigned short l = S != 'S' ? 0 : super::_signed; l < super::_map.size(); ++l) {
                    if(Q == FetiPrcndtnr::SUPERLUMPED && l != i && !std::binary_search(super::_vecSparsity[i].cbegin(), super::_vecSparsity[i].cend(), super::_map[l].first)) {
                        if(S != 'S' || !(l < i))
                            for(unsigned short k = 0; k < (U ? super::_local : infoNeighbor[l]); ++k)
                                for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                                    work[super::_offsets[super::_map[l].first] + super::_map[i].second[j] + k * super::_n] +=   m[i][j] * tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n];
                        continue;
                    }
                    for(unsigned short k = 0; k < (U ? super::_local : infoNeighbor[l]); ++k)
                        for(unsigned int j = 0; j < super::_map[i].second.size(); ++j) {
                            if(S != 'S' || !(l < i))
                                work[super::_offsets[super::_map[l].first] + super::_map[i].second[j] + k * super::_n] += (in[i][(accumulate + k) * super::_map[i].second.size() + j] =   m[i][j] * tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n]);
                            else
                                work[super::_offsets[super::_map[l].first] + super::_map[i].second[j] + k * super::_n] += m[i][j] * tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n];
                        }
                    if(S != 'S' || !(l < i))
                        accumulate += U ? super::_local : infoNeighbor[l];
                }
                if(U || infoNeighbor[i])
                    MPI_Isend(in[i], super::_map[i].second.size() * accumulate, Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), rq++);
            }
            delete [] tmp;
            delete [] offset;
            if(!U && Q != FetiPrcndtnr::SUPERLUMPED)
                delete [] infoNeighbor;
        }
        template<char S, bool U>
        void assembleForMaster(K* C, const K* in, const int& coefficients, unsigned short index, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            applyFromNeighbor<S, U>(in, index, arrayC, infoNeighbor);
            if(++super::_consolidate == super::_map.size()) {
                if(S != 'S')
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &coefficients, &(super::_local), &(super::_n), &(Wrapper<K>::d__1), arrayC, &(super::_n), *super::_deflation, super::_p.getLDR(), &(Wrapper<K>::d__0), C, &coefficients);
                else
                    for(unsigned short j = 0; j < super::_local; ++j) {
                        int local = coefficients + super::_local - j;
                        Blas<K>::gemv(&(Wrapper<K>::transc), &(super::_n), &local, &(Wrapper<K>::d__1), arrayC + super::_n * j, &(super::_n), super::_deflation[j], &i__1, &(Wrapper<K>::d__0), C - (j * (j - 1)) / 2 + j * (coefficients + super::_local), &i__1);
                    }
            }
        }
        template<char S, char N, bool U>
        void applyFromNeighborMaster(const K* in, unsigned short index, int* I, int* J, K* C, int coefficients, unsigned int offsetI, unsigned int* offsetJ, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            assembleForMaster<S, U>(C, in, coefficients, index, arrayC, infoNeighbor);
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
            std::vector<unsigned short>::const_iterator middle = std::lower_bound(super::_vecSparsity[index].cbegin(), super::_vecSparsity[index].cend(), super::_rank);
            unsigned int accumulate = 0;
            if(S != 'S' || !(index < super::_signed)) {
                for(unsigned short k = 0; k < (U ? super::_local : info[std::distance(super::_sparsity.cbegin(), std::lower_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_map[index].first))]); ++k)
                    for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                        work[super::_offsets[super::_map[index].first] + super::_map[index].second[j] + k * super::_n] += in[k * super::_map[index].second.size() + j];
                accumulate += (U ? super::_local : info[std::distance(super::_sparsity.cbegin(), std::lower_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_map[index].first))]) * super::_map[index].second.size();
            }
            std::vector<unsigned short>::const_iterator begin = super::_sparsity.cbegin();
            if(S != 'S')
                for(std::vector<unsigned short>::const_iterator it = super::_vecSparsity[index].cbegin(); it != middle; ++it) {
                    if(!U) {
                        std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::_sparsity.cend(), *it);
                        for(unsigned short k = 0; k < info[std::distance(super::_sparsity.cbegin(), idx)]; ++k)
                            for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                work[super::_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                        accumulate += info[std::distance(super::_sparsity.cbegin(), idx)] * super::_map[index].second.size();
                        begin = idx + 1;
                    }
                    else {
                        for(unsigned short k = 0; k < super::_local; ++k)
                            for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                                work[super::_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                        accumulate += super::_local * super::_map[index].second.size();
                    }
                }
            for(unsigned short k = 0; k < super::_local; ++k) {
                for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                    work[super::_offsets[super::_rank] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
            }
            accumulate += super::_local * super::_map[index].second.size();
            for(std::vector<unsigned short>::const_iterator it = middle + 1; it < super::_vecSparsity[index].cend(); ++it) {
                if(!U) {
                    std::vector<unsigned short>::const_iterator idx = std::lower_bound(begin, super::_sparsity.cend(), *it);
                    for(unsigned short k = 0; k < info[std::distance(super::_sparsity.cbegin(), idx)]; ++k)
                        for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                            work[super::_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                    accumulate += info[std::distance(super::_sparsity.cbegin(), idx)] * super::_map[index].second.size();
                    begin = idx + 1;
                }
                else {
                    for(unsigned short k = 0; k < super::_local; ++k)
                        for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                            work[super::_offsets[*it] + super::_map[index].second[j] + k * super::_n] += in[accumulate + k * super::_map[index].second.size() + j];
                    accumulate += super::_local * super::_map[index].second.size();
                }
            }
        }
    public:
        template<template<class> class Solver, char S, class T> friend class CoarseOperator;
        BddProjection(const Preconditioner& p, const unsigned short& c, const int& max) : super(p, c, max) { }
        template<char S, bool U, class T>
        void applyToNeighbor(T& in, K*& work, MPI_Request*& rq, const unsigned short* info, T const& out = nullptr, MPI_Request* const& rqRecv = nullptr) {
            unsigned short* infoNeighbor;
            super::template initialize<S, U>(in, info, out, rqRecv, infoNeighbor);
            MPI_Request* rqMult = new MPI_Request[2 * super::_map.size()];
            unsigned int* offset = new unsigned int[super::_map.size() + 2];
            offset[0] = 0;
            offset[1] = super::_local;
            for(unsigned short i = 2; i < super::_map.size() + 2; ++i)
                offset[i] = offset[i - 1] + (U ? super::_local : infoNeighbor[i - 2]);
            const int nbMult = super::_p.getMult();
            K* mult = new K[offset[super::_map.size() + 1] * nbMult];
            unsigned short* displs = new unsigned short[super::_map.size() + 1];
            displs[0] = 0;
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                MPI_Irecv(mult + offset[i + 1] * nbMult + displs[i] * (U ? super::_local : infoNeighbor[i]), super::_map[i].second.size() * (U ? super::_local : infoNeighbor[i]), Wrapper<K>::mpi_type(), super::_map[i].first, 11, super::_p.getCommunicator(), rqMult + i);
                displs[i + 1] = displs[i] + super::_map[i].second.size();
            }

            K* tmp = new K[offset[super::_map.size() + 1] * super::_n]();
            const underlying_type<K>* const m = super::_p.getScaling();
            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                for(unsigned short k = 0; k < super::_local; ++k)
                    for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                        tmp[super::_map[i].second[j] + k * super::_n] = (mult[displs[i] * super::_local + j + k * super::_map[i].second.size()] = m[super::_map[i].second[j]] * super::_deflation[k][super::_map[i].second[j]]);
                MPI_Isend(mult + displs[i] * super::_local, super::_map[i].second.size() * super::_local, Wrapper<K>::mpi_type(), super::_map[i].first, 11, super::_p.getCommunicator(), rqMult + super::_map.size() + i);
            }

            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                int index;
                MPI_Waitany(super::_map.size(), rqMult, &index, MPI_STATUS_IGNORE);
                for(unsigned short k = 0; k < (U ? super::_local : infoNeighbor[index]); ++k)
                    for(unsigned int j = 0; j < super::_map[index].second.size(); ++j)
                        tmp[super::_map[index].second[j] + (offset[index + 1] + k) * super::_n] = mult[offset[index + 1] * nbMult + displs[index] * (U ? super::_local : infoNeighbor[index]) + j + k * super::_map[index].second.size()];
            }

            delete [] displs;

            if(offset[super::_map.size() + 1])
                super::_p.applyLocalSchurComplement(tmp, offset[super::_map.size() + 1]);

            MPI_Waitall(super::_map.size(), rqMult + super::_map.size(), MPI_STATUSES_IGNORE);
            delete [] rqMult;
            delete [] mult;

            unsigned int accumulate = 0;
            unsigned short stop = std::distance(super::_sparsity.cbegin(), std::upper_bound(super::_sparsity.cbegin(), super::_sparsity.cend(), super::_rank));
            if(S != 'S') {
                super::_offsets.reserve(super::_sparsity.size() + 1);
                for(unsigned short i = 0; i < stop; ++i) {
                    super::_offsets.emplace(super::_sparsity[i], accumulate);
                    accumulate += super::_n * (U ? super::_local : info[i]);
                }
            }
            else
                super::_offsets.reserve(super::_sparsity.size() + 1 - stop);
            super::_offsets.emplace(super::_rank, accumulate);
            accumulate += super::_n * super::_local;
            for(unsigned short i = stop; i < super::_sparsity.size(); ++i) {
                super::_offsets.emplace(super::_sparsity[i], accumulate);
                accumulate += super::_n * (U ? super::_local : info[i]);
            }

            work = new K[accumulate]();

            for(unsigned short i = 0; i < super::_map.size(); ++i) {
                if(i < super::_signed || S != 'S') {
                    accumulate = super::_local;
                    for(unsigned short k = 0; k < super::_local; ++k)
                        for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                            work[super::_offsets[super::_rank] + super::_map[i].second[j] + k * super::_n] = in[i][k * super::_map[i].second.size() + j] = tmp[super::_map[i].second[j] + k * super::_n];
                }
                else {
                    accumulate = 0;
                    for(unsigned short k = 0; k < super::_local; ++k)
                        for(unsigned int j = 0; j < super::_map[i].second.size(); ++j)
                            work[super::_offsets[super::_rank] + super::_map[i].second[j] + k * super::_n] = tmp[super::_map[i].second[j] + k * super::_n];
                }
                for(unsigned short l = S != 'S' ? 0 : std::min(i, super::_signed); l < super::_map.size(); ++l) {
                    for(unsigned short k = 0; k < (U ? super::_local : infoNeighbor[l]); ++k)
                        for(unsigned int j = 0; j < super::_map[i].second.size(); ++j) {
                            if(S != 'S' || !(l < std::max(i, super::_signed)))
                                work[super::_offsets[super::_map[l].first] + super::_map[i].second[j] + k * super::_n] = in[i][(accumulate + k) * super::_map[i].second.size() + j] = tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n];
                            else {
                                if(i < super::_signed)
                                    in[i][(accumulate + k) * super::_map[i].second.size() + j] = tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n];
                                else
                                    work[super::_offsets[super::_map[l].first] + super::_map[i].second[j] + k * super::_n] = tmp[super::_map[i].second[j] + (offset[l + 1] + k) * super::_n];
                            }
                        }
                    if(S != 'S' || !(l < i) || i < super::_signed)
                        accumulate += U ? super::_local : infoNeighbor[l];
                }
                if(U || infoNeighbor[i])
                    MPI_Isend(in[i], super::_map[i].second.size() * accumulate, Wrapper<K>::mpi_type(), super::_map[i].first, 2, super::_p.getCommunicator(), rq++);
            }
            delete [] tmp;
            delete [] offset;
            if(!U)
                delete [] infoNeighbor;
        }
        template<char S, bool U>
        void assembleForMaster(K* C, const K* in, const int& coefficients, unsigned short index, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            applyFromNeighbor<S, U>(in, index, arrayC, infoNeighbor);
            if(++super::_consolidate == super::_map.size()) {
                const underlying_type<K>* const m = super::_p.getScaling();
                for(unsigned short j = 0; j < coefficients + (S == 'S') * super::_local; ++j)
                    Wrapper<K>::diag(super::_n, m, arrayC + j * super::_n);
                if(S != 'S')
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &coefficients, &(super::_local), &(super::_n), &(Wrapper<K>::d__1), arrayC, &(super::_n), *super::_deflation, super::_p.getLDR(), &(Wrapper<K>::d__0), C, &coefficients);
                else
                    for(unsigned short j = 0; j < super::_local; ++j) {
                        int local = coefficients + super::_local - j;
                        Blas<K>::gemv(&(Wrapper<K>::transc), &(super::_n), &local, &(Wrapper<K>::d__1), arrayC + super::_n * j, &(super::_n), super::_deflation[j], &i__1, &(Wrapper<K>::d__0), C - (j * (j - 1)) / 2 + j * (coefficients + super::_local), &i__1);
                    }
            }
        }
        template<char S, char N, bool U>
        void applyFromNeighborMaster(const K* in, unsigned short index, int* I, int* J, K* C, int coefficients, unsigned int offsetI, unsigned int* offsetJ, K* arrayC, unsigned short* const& infoNeighbor = nullptr) {
            assembleForMaster<S, U>(C, in, coefficients, index, arrayC, infoNeighbor);
            super::template assembleOperator<S, N, U>(I, J, coefficients, offsetI, offsetJ, infoNeighbor);
        }
};
#endif // HPDDM_BDD
} // HPDDM
#endif // _HPDDM_OPERATOR_
