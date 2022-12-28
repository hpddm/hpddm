/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2016-09-13

   Copyright (C) 2016-     Centre National de la Recherche Scientifique

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

#ifndef HPDDM_INEXACT_COARSE_OPERATOR_HPP_
#define HPDDM_INEXACT_COARSE_OPERATOR_HPP_

#if !HPDDM_PETSC
# if !defined(PETSCSUB)
class Mat;
# endif
#else
# include "HPDDM_subdomain.hpp"
# if HPDDM_SLEPC
#  include <slepc.h>
# endif
#endif

namespace HPDDM {
HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K) class CoarseOperator;
template<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
    template<class> class Solver, template<class> class CoarseSolver, char S,
#endif
    class K>
class Schwarz;

HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K)
class InexactCoarseOperator : public OptionsPrefix<K>, public Solver
#if !HPDDM_PETSC
                                                                    <K>
#endif
    {
    protected:
#if !HPDDM_PETSC
        typedef typename Subdomain<K>::integer_type integer_type;
        typedef void                                 return_type;
        Schwarz<SUBDOMAIN, COARSEOPERATOR, S, K>*       _s;
        const Schwarz<SUBDOMAIN, COARSEOPERATOR, S, K>* _p;
        MPI_Comm _communicator;
        K**              _buff;
        MPI_Request*       _rq;
        K*                  _x;
        K*                  _o;
        unsigned short     _mu;
#else
        typedef PetscErrorCode return_type;
        typedef PetscInt      integer_type;
        PC_HPDDM_Level*                 _s;
        PetscInt*                     _idx;
#endif
        vectorNeighbor   _recv;
        std::map<unsigned short, std::vector<int>> _send;
        K*                 _da;
        K*                 _oa;
        integer_type*      _di;
        integer_type*      _oi;
        integer_type*      _dj;
        integer_type*      _oj;
        integer_type*     _ogj;
        unsigned int*   _range;
        int               _dof;
        int               _off;
        int                _bs;
        template<char
#if !HPDDM_PETSC
                      T
#else
                      S
#endif
                       , bool factorize>
        void numfact(unsigned int nrow, integer_type* I, int* loc2glob, integer_type* J, K* C, unsigned short* neighbors) {
            _da = C;
            _dj = J;
#if !HPDDM_PETSC
            Option& opt = *Option::get();
#ifdef DMKL_PARDISO
            if(factorize) {
                _range = new unsigned int[3];
                std::copy_n(loc2glob, 2, _range);
                _range[2] = 0;
            }
#else
            if(factorize && S == 'S') {
                _range = new unsigned int[2];
                std::copy_n(loc2glob, 2, _range);
            }
#endif
            MPI_Comm_dup(DMatrix::_communicator, &_communicator);
            MPI_Comm_size(_communicator, &_off);
            if(_off > 1) {
                unsigned int accumulate = 0;
                {
                    int* ia = nullptr;
                    K* a;
                    if(_mu < _off) {
                        MPI_Group world, aggregate;
                        MPI_Comm_group(_communicator, &world);
                        int ranges[1][3];
                        ranges[0][0] = (DMatrix::_rank / _mu) * _mu;
                        ranges[0][1] = std::min(_off, ranges[0][0] + _mu) - 1;
                        ranges[0][2] = 1;
                        MPI_Group_range_incl(world, 1, ranges, &aggregate);
                        MPI_Comm_free(&(DMatrix::_communicator));
                        MPI_Comm_create(_communicator, aggregate, &(DMatrix::_communicator));
                        MPI_Group_free(&aggregate);
                        MPI_Group_free(&world);
                        bool r = false;
                        std::vector<int*> range;
                        range.reserve((S == 'S' ? 1 : (T == 1 ? 8 : 4)) * nrow);
                        range.emplace_back(J);
                        int* R = new int[nrow + 1];
                        R[0] = (super::_numbering == 'F');
                        for(unsigned int i = 0; i < nrow; ++i) {
                            R[i + 1] = R[i];
                            for(unsigned int j = 0; j < I[i + 1]; ++j) {
                                const int k = accumulate + j;
                                if(ranges[0][0] <= neighbors[k] && neighbors[k] <= ranges[0][1]) {
                                    R[i + 1]++;
                                    if(r) {
                                        r = false;
                                        range.emplace_back(J + k);
                                    }
                                }
                                else if(!r) {
                                    r = true;
                                    range.emplace_back(J + k);
                                }
                            }
                            accumulate += I[i + 1];
                        }
                        if(range.size() > 1) {
                            range.emplace_back(J + accumulate);
                            std::vector<int*>::const_iterator it;
                            a = new K[(R[nrow] - (super::_numbering == 'F')) * _bs * _bs];
                            ia = new int[nrow + 1 + R[nrow] - (super::_numbering == 'F')];
                            accumulate = 0;
                            ia += nrow + 1;
                            for(it = range.cbegin(); it < range.cend() - 1; it += 2) {
                                unsigned int size = *(it + 1) - *it;
                                for(unsigned int i = 0; i < size; ++i) {
                                    ia[accumulate + i] = (*it)[i] - _di[0];
                                    if(T == 1 && (*it)[i] > _di[1])
                                        ia[accumulate + i] -= _di[2];
                                }
                                std::copy_n(C + std::distance(J, *it) * _bs * _bs, size * _bs * _bs, a + accumulate * _bs * _bs);
                                accumulate += size;
                            }
                            ia -= nrow + 1;
                            std::copy_n(R, nrow + 1, ia);
                        }
                        delete [] R;
                    }
                    else {
                        MPI_Comm_free(&_communicator);
                        _communicator = DMatrix::_communicator;
                        accumulate = std::accumulate(I + 1, I + 1 + nrow, 0);
                    }
                    int* ja;
                    if(!ia) {
                        ia = new int[nrow + 1 + (_mu < _off ? accumulate : 0)];
                        std::partial_sum(I, I + nrow + 1, ia);
                        if(_mu < _off) {
                            ja = ia + nrow + 1;
                            for(unsigned int i = 0; i < nrow; ++i) {
                                for(unsigned int j = ia[i]; j < ia[i + 1]; ++j) {
                                    const unsigned int idx = J[j - (super::_numbering == 'F')];
                                    ja[j - (super::_numbering == 'F')] = idx - _di[0];
                                    if(T == 1 && idx > _di[1])
                                        ja[j - (super::_numbering == 'F')] -= _di[2];
                                }
                            }
                        }
                        else
                            ja = I + nrow + 1;
                        a = C;
                    }
                    const std::string prefix = OptionsPrefix<K>::prefix();
#ifdef DMKL_PARDISO
                    if(factorize && (a != C || Option::get()->set(prefix + "schwarz_method")))
                        _range[2] = 1;
#endif
                    if(_mu < _off) {
                        loc2glob[0] -= _di[0];
                        loc2glob[1] -= _di[0];
                        delete [] _di;
                    }
                    if(factorize && !Option::get()->set(prefix + "schwarz_method")) {
#ifdef DMKL_PARDISO
                        Solver<K>::template numfact<S>(_bs, ia, loc2glob, ja, a);
                        loc2glob = nullptr;
#else
                        int* irn;
                        int* jcn;
                        K* b;
                        Wrapper<K>::template bsrcoo<S, Solver<K>::_numbering, 'F'>(nrow, _bs, a, ia, ja, b, irn, jcn, loc2glob[0] - (Solver<K>::_numbering == 'F'));
                        if(a != b && a != C)
                            delete [] a;
                        if(DMatrix::_n)
                            Solver<K>::template numfact<S>(std::distance(irn, jcn), irn, jcn, a = b);
                        Solver<K>::_range = { (loc2glob[0] - (Solver<K>::_numbering == 'F')) * _bs, (loc2glob[1] + (Solver<K>::_numbering == 'C')) * _bs };
#endif
                    }
#ifdef DMKL_PARDISO
                    else
#endif
                        delete [] ia;
                    if(a != C)
                        delete [] a;
                }
                _di = new int[nrow + 1];
                _di[0] = (super::_numbering == 'F');
                std::map<int, unsigned short> off;
                std::set<int> on;
                std::map<unsigned short, unsigned int> allocation;
                bool r = false;
                std::vector<int*> range;
                range.reserve((S == 'S' ? 1 : (T == 1 ? 8 : 4)) * nrow);
                range.emplace_back(J);
                for(unsigned int i = 0; i < nrow; ++i) {
                    _di[i + 1] = _di[i];
                    for(unsigned int j = 0; j < I[i + 1]; ++j) {
                        const int k = I[i] + _di[i] + j - (super::_numbering == 'F' ? 2 : 0);
                        if(neighbors[k] == DMatrix::_rank) {
                            _di[i + 1]++;
                            on.insert(J[k]);
                            if(r) {
                                r = false;
                                range.emplace_back(J + k);
                            }
                        }
                        else {
                            off[J[k]] = neighbors[k];
                            allocation[neighbors[k]] += 1;
                            if(!r) {
                                r = true;
                                range.emplace_back(J + k);
                            }
                        }
                    }
                    I[i + 1] += I[i] - (_di[i + 1] - _di[i]);
                }
                delete [] neighbors;
                _dof = on.size();
                if(opt.val<char>("krylov_method", HPDDM_KRYLOV_METHOD_GMRES) != HPDDM_KRYLOV_METHOD_NONE) {
                    accumulate = 0;
                    if(range.size() > 1) {
                        range.emplace_back(J + I[nrow] + _di[nrow] - (super::_numbering == 'F' ? 2 : 0));
                        K* D = new K[(_di[nrow] - (super::_numbering == 'F') - (range[1] - range[0])) * _bs * _bs];
                        int* L = new int[_di[nrow] - (super::_numbering == 'F') - (range[1] - range[0])];
                        std::vector<int*>::const_iterator it;
                        for(it = range.cbegin() + 2; it < range.cend() - 1; it += 2) {
                            unsigned int size = *(it + 1) - *it;
                            std::copy_n(*it, size, L + accumulate);
                            std::copy_n(C + std::distance(J, *it) * _bs * _bs, size * _bs * _bs, D + accumulate * _bs * _bs);
                            accumulate += size;
                        }
                        accumulate = std::distance(J, range.back());
                        if(it != range.cend())
                            accumulate -= std::distance(*(it - 1), *it);
                        while(it > range.cbegin() + 3) {
                            it -= 2;
                            std::copy_backward(*(it - 1), *it, J + accumulate);
                            std::copy_backward(C + std::distance(J, *(it - 1)) * _bs * _bs, C + std::distance(J, *it) * _bs * _bs, C + (accumulate + 0) * _bs * _bs);
                            accumulate -= *it - *(it - 1);
                        }
                        std::copy_n(D, (_di[nrow] - (super::_numbering == 'F') - (range[1] - range[0])) * _bs * _bs, C + (range[1] - range[0]) * _bs * _bs);
                        std::copy_n(L, _di[nrow] - (super::_numbering == 'F') - (range[1] - range[0]), J + (range[1] - range[0]));
                        delete [] L;
                        delete [] D;
                    }
                    _recv.reserve(allocation.size());
                    for(const std::pair<unsigned short, unsigned int>& p : allocation) {
                        _recv.emplace_back(p.first, std::vector<int>());
                        _recv.back().second.reserve(p.second);
                    }
                    std::unordered_map<int, int> g2l;
                    g2l.reserve(_dof + off.size());
                    accumulate = 0;
                    for(const int& i : on)
                        g2l.emplace(i - (super::_numbering == 'F'), accumulate++);
                    std::set<int>().swap(on);
                    unsigned short search[2] { 0, std::numeric_limits<unsigned short>::max() };
                    for(std::pair<const int, unsigned short>& i : off) {
                        if(search[1] != i.second) {
                            search[0] = std::distance(allocation.begin(), allocation.find(i.second));
                            search[1] = i.second;
                        }
                        _recv[search[0]].second.emplace_back(accumulate++);
                    }
                    if(S == 'S') {
                        char* table = new char[((T == 1 ? _off * _off : (_off * (_off - 1)) / 2) >> 3) + 1]();
                        std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator begin = (T == 1 ? _recv.cbegin() : std::upper_bound(_recv.cbegin(), _recv.cend(), std::make_pair(static_cast<unsigned short>(DMatrix::_rank), std::vector<int>()), [](const std::pair<unsigned short, std::vector<int>>& lhs, const std::pair<unsigned short, std::vector<int>>& rhs) { return lhs.first < rhs.first; }));
                        for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != _recv.cend(); ++it) {
                            const unsigned int idx = (T == 1 ? it->first * _off : (it->first * (it->first - 1)) / 2) + DMatrix::_rank;
                            table[idx >> 3] |= 1 << (idx & 7);
                        }
                        MPI_Allreduce(MPI_IN_PLACE, table, ((T == 1 ? _off * _off : (_off * (_off - 1)) / 2) >> 3) + 1, MPI_CHAR, MPI_BOR, _communicator);
                        std::vector<unsigned short> infoRecv;
                        infoRecv.reserve(T == 1 ? _off : DMatrix::_rank);
                        for(unsigned short i = 0; i < (T == 1 ? _off : DMatrix::_rank); ++i) {
                            const unsigned int idx = (T == 1 ? DMatrix::_rank * _off : (DMatrix::_rank * (DMatrix::_rank - 1)) / 2) + i;
                            if(table[idx >> 3] & (1 << (idx & 7)))
                                infoRecv.emplace_back(i);
                        }
                        delete [] table;
                        const unsigned short size = infoRecv.size() + std::distance(begin, _recv.cend());
                        unsigned int* lengths = new unsigned int[size];
                        unsigned short distance = 0;
                        MPI_Request* rq = new MPI_Request[size];
                        for(const unsigned short& i : infoRecv) {
                            MPI_Irecv(lengths + distance, 1, MPI_UNSIGNED, i, 11, _communicator, rq + distance);
                            ++distance;
                        }
                        for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != _recv.cend(); ++it) {
                            lengths[distance] = it->second.size();
                            MPI_Isend(lengths + distance, 1, MPI_UNSIGNED, it->first, 11, _communicator, rq + distance);
                            ++distance;
                        }
                        MPI_Waitall(size, rq, MPI_STATUSES_IGNORE);
                        distance = 0;
                        for(const unsigned short& i : infoRecv) {
                            std::map<unsigned short, std::vector<int>>::iterator it = _send.emplace_hint(_send.end(), i, std::vector<int>(lengths[distance]));
                            MPI_Irecv(it->second.data(), it->second.size(), MPI_INT, i, 12, _communicator, rq + distance++);
                        }
                        accumulate = std::accumulate(lengths + infoRecv.size(), lengths + size, 0);
                        delete [] lengths;
                        int* sendIdx = new int[accumulate];
                        accumulate = 0;
                        for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != _recv.cend(); ++it) {
                            std::map<int, unsigned short>::const_iterator global = off.begin();
                            for(unsigned int k = 0; k < it->second.size(); ++k) {
                                std::advance(global, it->second[k] - (k == 0 ? _dof : it->second[k - 1]));
                                sendIdx[accumulate + k] = global->first - (super::_numbering == 'F');
                            }
                            MPI_Isend(sendIdx + accumulate, it->second.size(), MPI_INT, it->first, 12, _communicator, rq + distance++);
                            accumulate += it->second.size();
                        }
                        for(unsigned int i = 0; i < infoRecv.size(); ++i) {
                            int index;
                            MPI_Waitany(infoRecv.size(), rq, &index, MPI_STATUS_IGNORE);
                            std::for_each(_send[infoRecv[index]].begin(), _send[infoRecv[index]].end(), [&g2l](int& j) { j = g2l.at(j); });
                        }
                        MPI_Waitall(size - infoRecv.size(), rq + infoRecv.size(), MPI_STATUSES_IGNORE);
                        delete [] sendIdx;
                        delete [] rq;
                    }
                    else
                        for(std::pair<const unsigned short, std::vector<int>>& i : _send)
                            std::for_each(i.second.begin(), i.second.end(), [&g2l](int& j) { j = g2l.at(j); });
                    accumulate = 0;
                    for(std::pair<const int, unsigned short>& i : off)
                        g2l.emplace(i.first - (super::_numbering == 'F'), accumulate++);
                    for(std::pair<unsigned short, std::vector<int>>& i : _recv)
                        std::for_each(i.second.begin(), i.second.end(), [&](int& j) { j -= _dof; });
                    _ogj = new int[I[nrow] - (super::_numbering == 'F' ? 1 : 0)];
                    std::copy_n(J + _di[nrow] - (super::_numbering == 'F' ? 1 : 0), I[nrow] - (super::_numbering == 'F' ? 1 : 0), _ogj);
                    std::for_each(J, J + I[nrow] + _di[nrow] - (super::_numbering == 'F' ? 2 : 0), [&](int& i) { i = g2l[i - (this->_numbering == 'F')] + (this->_numbering == 'F'); });
                    _buff = new K*[_send.size() + _recv.size()];
                    accumulate = std::accumulate(_recv.cbegin(), _recv.cend(), 0, [](unsigned int init, const std::pair<unsigned short, std::vector<int>>& i) { return init + i.second.size(); });
                    accumulate = std::accumulate(_send.cbegin(), _send.cend(), accumulate, [](unsigned int init, const std::pair<unsigned short, std::vector<int>>& i) { return init + i.second.size(); });
                    *_buff = new K[accumulate * _bs];
                    accumulate = 0;
                    _off = 0;
                    for(const std::pair<unsigned short, std::vector<int>>& i : _recv) {
                        _buff[_off++] = *_buff + accumulate * _bs;
                        accumulate += i.second.size();
                    }
                    for(const std::pair<unsigned short, std::vector<int>>& i : _send) {
                        _buff[_off++] = *_buff + accumulate * _bs;
                        accumulate += i.second.size();
                    }
                    _rq = new MPI_Request[_send.size() + _recv.size()];
                    _oi = I;
                    _oa = C + (_di[nrow] - (super::_numbering == 'F')) * _bs * _bs;
                    _oj = J + _di[nrow] - (super::_numbering == 'F');
                    _off = off.size();
                    if(DMatrix::_rank != 0)
                        opt.remove("verbosity");
                }
                else {
                    delete [] _di;
                    _di = nullptr;
                    _off = 0;
                }
            }
            else {
                _dof = nrow;
                _off = 0;
                std::partial_sum(I, I + _dof + 1, I);
                _di = I;
                delete [] neighbors;
                if(factorize) {
#ifdef DMKL_PARDISO
                    Solver<K>::template numfact<S>(_bs, I, loc2glob, J, C);
                    loc2glob = nullptr;
#else
                    int* irn;
                    int* jcn;
                    K* b;
                    Wrapper<K>::template bsrcoo<S, Solver<K>::_numbering, 'F'>(nrow, _bs, C, I, J, b, irn, jcn, loc2glob[0] - (Solver<K>::_numbering == 'F'));
                    if(DMatrix::_n)
                        Solver<K>::template numfact<S>(std::distance(irn, jcn), irn, jcn, b);
                    if(b != C)
                        delete [] b;
                    Solver<K>::_range = { (loc2glob[0] - (Solver<K>::_numbering == 'F')) * _bs, (loc2glob[1] + (Solver<K>::_numbering == 'C')) * _bs };
#endif
                }
            }
            OptionsPrefix<K>::setPrefix(opt.getPrefix());
            _mu = 0;
#else
            constexpr char T = 0;
            if(S == 'S') {
                _range = new unsigned int[2];
                std::copy_n(loc2glob, 2, _range);
            }
            {
                int* ia = nullptr;
                K* a;
                bool r = false;
                std::vector<integer_type*> range;
                range.reserve((S == 'S' ? 1 : (T == 1 ? 8 : 4)) * nrow);
                range.emplace_back(J);
                int* R = new int[nrow + 1];
                R[0] = (super::_numbering == 'F');
                for(unsigned int i = 0; i < nrow; ++i) {
                    R[i + 1] = R[i];
                    for(unsigned int j = 0; j < I[i + 1] - I[i]; ++j) {
                        const int k = I[i] + j - (super::_numbering == 'F');
                        if(DMatrix::_rank <= neighbors[k] && neighbors[k] <= DMatrix::_rank + 1) {
                            R[i + 1]++;
                            if(r) {
                                r = false;
                                range.emplace_back(J + k);
                            }
                        }
                        else if(!r) {
                            r = true;
                            range.emplace_back(J + k);
                        }
                    }
                }
                if(range.size() > 1) {
                    range.emplace_back(J + I[nrow] - (super::_numbering == 'F'));
                    std::vector<integer_type*>::const_iterator it;
                    a = new K[(R[nrow] - (super::_numbering == 'F')) * _bs * _bs];
                    ia = new int[nrow + 1 + R[nrow] - (super::_numbering == 'F')];
                    unsigned int accumulate = 0;
                    ia += nrow + 1;
                    for(it = range.cbegin(); it < range.cend() - 1; it += 2) {
                        unsigned int size = *(it + 1) - *it;
                        for(unsigned int i = 0; i < size; ++i) {
                            ia[accumulate + i] = (*it)[i] - _di[0];
                            if(T == 1 && (*it)[i] > _di[1])
                                ia[accumulate + i] -= _di[2];
                        }
                        std::copy_n(C + std::distance(J, *it) * _bs * _bs, size * _bs * _bs, a + accumulate * _bs * _bs);
                        accumulate += size;
                    }
                    ia -= nrow + 1;
                    std::copy_n(R, nrow + 1, ia);
                }
                delete [] R;
                int* ja;
                if(!ia) {
                    ia = new int[nrow + 1 + I[nrow] - (super::_numbering == 'F')];
                    std::copy_n(I, nrow + 1, ia);
                    ja = ia + nrow + 1;
                    for(unsigned int i = 0; i < nrow; ++i) {
                        for(unsigned int j = ia[i]; j < ia[i + 1]; ++j) {
                            const unsigned int idx = J[j - (super::_numbering == 'F')];
                            ja[j - (super::_numbering == 'F')] = idx - _di[0];
                            if(T == 1 && idx > _di[1])
                                ja[j - (super::_numbering == 'F')] -= _di[2];
                        }
                    }
                    a = C;
                }
                loc2glob[0] -= _di[0];
                loc2glob[1] -= _di[0];
                delete [] _di;
                delete [] ia;
                if(a != C)
                    delete [] a;
            }
            _di = new integer_type[nrow + 1];
            _di[0] = (super::_numbering == 'F');
            std::map<int, unsigned short> off;
            std::set<int> on;
            std::map<unsigned short, unsigned int> allocation;
            std::unordered_map<unsigned short, std::set<int>> exchange;
            bool r = false;
            std::vector<integer_type*> range;
            range.reserve((S == 'S' ? 1 : (T == 1 ? 8 : 4)) * nrow);
            range.emplace_back(J);
            for(unsigned int i = 0; i < nrow; ++i) {
                _di[i + 1] = _di[i];
                for(unsigned int j = 0; j < I[i + 1] - I[i] - _di[i]; ++j) {
                    const int k = I[i] + _di[i] + j - (super::_numbering == 'F' ? 2 : 0);
                    if(neighbors[k] == DMatrix::_rank) {
                        _di[i + 1]++;
                        on.insert(J[k]);
                        if(r) {
                            r = false;
                            range.emplace_back(J + k);
                        }
                    }
                    else {
                        off[J[k]] = neighbors[k];
                        allocation[neighbors[k]] += 1;
                        if(S == 'S' && factorize && neighbors[k] > DMatrix::_rank)
                            exchange[neighbors[k]].insert(_range[0] + i);
                        if(!r) {
                            r = true;
                            range.emplace_back(J + k);
                        }
                    }
                }
                I[i + 1] -= _di[i + 1];
            }
            delete [] neighbors;
            _dof = on.size();
            unsigned int accumulate = 0;
            if(range.size() > 1) {
                range.emplace_back(J + I[nrow] + _di[nrow] - (super::_numbering == 'F' ? 2 : 0));
                K* D = new K[(_di[nrow] - (super::_numbering == 'F') - (range[1] - range[0])) * _bs * _bs];
                int* L = new int[_di[nrow] - (super::_numbering == 'F') - (range[1] - range[0])];
                std::vector<integer_type*>::const_iterator it;
                for(it = range.cbegin() + 2; it < range.cend() - 1; it += 2) {
                    unsigned int size = *(it + 1) - *it;
                    std::copy_n(*it, size, L + accumulate);
                    std::copy_n(C + std::distance(J, *it) * _bs * _bs, size * _bs * _bs, D + accumulate * _bs * _bs);
                    accumulate += size;
                }
                accumulate = std::distance(J, range.back());
                if(it != range.cend())
                    accumulate -= std::distance(*(it - 1), *it);
                while(it > range.cbegin() + 3) {
                    it -= 2;
                    std::copy_backward(*(it - 1), *it, J + accumulate);
                    std::copy_backward(C + std::distance(J, *(it - 1)) * _bs * _bs, C + std::distance(J, *it) * _bs * _bs, C + (accumulate + 0) * _bs * _bs);
                    accumulate -= *it - *(it - 1);
                }
                std::copy_n(D, (_di[nrow] - (super::_numbering == 'F') - (range[1] - range[0])) * _bs * _bs, C + (range[1] - range[0]) * _bs * _bs);
                std::copy_n(L, _di[nrow] - (super::_numbering == 'F') - (range[1] - range[0]), J + (range[1] - range[0]));
                delete [] L;
                delete [] D;
            }
            _recv.reserve(allocation.size());
            for(const std::pair<const unsigned short, unsigned int>& p : allocation) {
                _recv.emplace_back(p.first, std::vector<int>());
                _recv.back().second.reserve(p.second);
            }
            std::unordered_map<int, int> g2l;
            g2l.reserve(_dof + off.size());
            accumulate = 0;
            for(const int& i : on)
                g2l.emplace(i - (super::_numbering == 'F'), accumulate++);
            unsigned short search[2] { 0, std::numeric_limits<unsigned short>::max() };
            for(std::pair<const int, unsigned short>& i : off) {
                if(search[1] != i.second) {
                    search[0] = std::distance(allocation.begin(), allocation.find(i.second));
                    search[1] = i.second;
                }
                _recv[search[0]].second.emplace_back(accumulate++);
            }
            std::set<int> overlap;
            if(S == 'S') {
                MPI_Comm_size(DMatrix::_communicator, &_off);
                char* table = new char[((T == 1 ? _off * _off : (_off * (_off - 1)) / 2) >> 3) + 1]();
                std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator begin = (T == 1 ? _recv.cbegin() : std::upper_bound(_recv.cbegin(), _recv.cend(), std::make_pair(static_cast<unsigned short>(DMatrix::_rank), std::vector<int>()), [](const std::pair<unsigned short, std::vector<int>>& lhs, const std::pair<unsigned short, std::vector<int>>& rhs) { return lhs.first < rhs.first; }));
                for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != _recv.cend(); ++it) {
                    const unsigned int idx = (T == 1 ? it->first * _off : (it->first * (it->first - 1)) / 2) + DMatrix::_rank;
                    table[idx >> 3] |= 1 << (idx & 7);
                }
                MPI_Allreduce(MPI_IN_PLACE, table, ((T == 1 ? _off * _off : (_off * (_off - 1)) / 2) >> 3) + 1, MPI_CHAR, MPI_BOR, DMatrix::_communicator);
                std::vector<unsigned short> infoRecv;
                infoRecv.reserve(T == 1 ? _off : DMatrix::_rank);
                for(unsigned short i = 0; i < (T == 1 ? _off : DMatrix::_rank); ++i) {
                    const unsigned int idx = (T == 1 ? DMatrix::_rank * _off : (DMatrix::_rank * (DMatrix::_rank - 1)) / 2) + i;
                    if(table[idx >> 3] & (1 << (idx & 7)))
                        infoRecv.emplace_back(i);
                }
                delete [] table;
                const unsigned short size = infoRecv.size() + std::distance(begin, _recv.cend());
                unsigned int* lengths = new unsigned int[size + 1];
                unsigned short distance = 0;
                MPI_Request* rq = new MPI_Request[size];
                for(const unsigned short& i : infoRecv) {
                    MPI_Irecv(lengths + distance, 1, MPI_UNSIGNED, i, 11, DMatrix::_communicator, rq + distance);
                    ++distance;
                }
                for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != _recv.cend(); ++it, ++distance) {
                    lengths[distance] = it->second.size();
                    MPI_Isend(lengths + distance, 1, MPI_UNSIGNED, it->first, 11, DMatrix::_communicator, rq + distance);
                }
                MPI_Waitall(size, rq, MPI_STATUSES_IGNORE);
                distance = 0;
                for(const unsigned short& i : infoRecv) {
                    std::map<unsigned short, std::vector<int>>::iterator it = _send.emplace_hint(_send.end(), i, std::vector<int>(lengths[distance]));
                    MPI_Irecv(it->second.data(), it->second.size(), MPI_INT, i, 12, DMatrix::_communicator, rq + distance++);
                }
                accumulate = std::accumulate(lengths + infoRecv.size(), lengths + size, 0);
                int* sendIdx = new int[accumulate];
                accumulate = 0;
                for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != _recv.cend(); ++it) {
                    std::map<int, unsigned short>::const_iterator global = off.begin();
                    for(unsigned int k = 0; k < it->second.size(); ++k) {
                        std::advance(global, it->second[k] - (k == 0 ? _dof : it->second[k - 1]));
                        sendIdx[accumulate + k] = global->first - (super::_numbering == 'F');
                    }
                    MPI_Isend(sendIdx + accumulate, it->second.size(), MPI_INT, it->first, 12, DMatrix::_communicator, rq + distance++);
                    accumulate += it->second.size();
                }
                for(unsigned int i = 0; i < infoRecv.size(); ++i) {
                    int index;
                    MPI_Waitany(infoRecv.size(), rq, &index, MPI_STATUS_IGNORE);
                    std::for_each(_send[infoRecv[index]].begin(), _send[infoRecv[index]].end(), [&g2l, &on, &overlap](int& j) {
                        if(factorize && on.find(j) == on.cend())
                            overlap.insert(j);
                        j = g2l.at(j);
                    });
                }
                MPI_Waitall(size - infoRecv.size(), rq + infoRecv.size(), MPI_STATUSES_IGNORE);
                if(factorize) {
                    distance = 0;
                    for(const unsigned short& i : infoRecv) {
                        MPI_Irecv(lengths + distance + 1, 1, MPI_UNSIGNED, i, 121, DMatrix::_communicator, rq + distance);
                        ++distance;
                    }
                    for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != _recv.cend(); ++it, ++distance) {
                        lengths[distance + 1] = exchange[it->first].size();
                        MPI_Isend(lengths + distance + 1, 1, MPI_UNSIGNED, it->first, 121, DMatrix::_communicator, rq + distance);
                    }
                    MPI_Waitall(size, rq, MPI_STATUSES_IGNORE);
                    delete [] sendIdx;
                    lengths[0] = 0;
                    std::partial_sum(lengths + 1, lengths + size + 1, lengths + 1);
                    sendIdx = new int[lengths[size]];
                    distance = 0;
                    for(const unsigned short& i : infoRecv) {
                        MPI_Irecv(sendIdx + lengths[distance], lengths[distance + 1] - lengths[distance], MPI_INT, i, 122, DMatrix::_communicator, rq + distance);
                        ++distance;
                    }
                    for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != _recv.cend(); ++it, ++distance) {
                        std::copy(exchange[it->first].begin(), exchange[it->first].end(), sendIdx + lengths[distance]);
                        MPI_Isend(sendIdx + lengths[distance], exchange[it->first].size(), MPI_INT, it->first, 122, DMatrix::_communicator, rq + distance);
                    }
                    for(unsigned int i = 0; i < infoRecv.size(); ++i) {
                        int index;
                        MPI_Waitany(infoRecv.size(), rq, &index, MPI_STATUS_IGNORE);
                        std::for_each(sendIdx + lengths[index], sendIdx + lengths[index + 1], [&overlap](int& j) { overlap.insert(j); });
                    }
                    MPI_Waitall(size - infoRecv.size(), rq + infoRecv.size(), MPI_STATUSES_IGNORE);
                }
                delete [] lengths;
                delete [] sendIdx;
                delete [] rq;
            }
            else
                for(std::pair<const unsigned short, std::vector<int>>& i : _send)
                    std::for_each(i.second.begin(), i.second.end(), [&g2l](int& j) { j = g2l.at(j); });
            accumulate = 0;
            for(std::pair<const int, unsigned short>& i : off)
                g2l.emplace(i.first - (super::_numbering == 'F'), accumulate++);
            for(std::pair<unsigned short, std::vector<int>>& i : _recv)
                std::for_each(i.second.begin(), i.second.end(), [&](int& j) { j -= _dof; });
            _ogj = new integer_type[I[nrow] - (super::_numbering == 'F' ? 1 : 0)];
            std::copy_n(J + _di[nrow] - (super::_numbering == 'F' ? 1 : 0), I[nrow] - (super::_numbering == 'F' ? 1 : 0), _ogj);
            std::for_each(J, J + I[nrow] + _di[nrow] - (super::_numbering == 'F' ? 2 : 0), [&](integer_type& i) { i = g2l[i - (this->_numbering == 'F')] + (this->_numbering == 'F'); });
            _oi = I;
            _oa = C + (_di[nrow] - (super::_numbering == 'F')) * _bs * _bs;
            _oj = J + _di[nrow] - (super::_numbering == 'F');
            _off = off.size();
            if(factorize && !_idx) {
                PetscMalloc1(on.size() + overlap.size() + off.size(), &_idx);
                for(const auto& range : { on, overlap })
                    for(const int& i : range)
                        *_idx++ = i - (super::_numbering == 'F');
                for(std::pair<const int, unsigned short>& i : off)
                    *_idx++ = i.first - (super::_numbering == 'F');
                _idx -= on.size() + overlap.size() + off.size();
            }
#endif
            delete [] loc2glob;
        }
    public:
        InexactCoarseOperator() : OptionsPrefix<K>(), super(), _s(),
#if !HPDDM_PETSC
                                                                    _p(), _communicator(MPI_COMM_NULL), _buff(), _rq(), _x(), _mu()
#else
                                                                    _idx(), _da()
#endif
                                                                          , _di(), _oi(), _ogj(), _range(), _off() { }
        ~InexactCoarseOperator() {
#if !HPDDM_PETSC
            if(_s)
                delete [] _s->getScaling();
            delete _s;
            if(_buff) {
                delete [] *_buff;
                delete [] _buff;
            }
            delete [] _x;
            if(_communicator != MPI_COMM_NULL) {
                int size;
                MPI_Comm_size(_communicator, &size);
#ifdef DMKL_PARDISO
                if(size > 1)
#endif
                    delete [] _di;
#ifdef DMKL_PARDISO
                if(_range && _range[2] == 1)
#endif
                    delete [] _da;
                if(_communicator != DMatrix::_communicator)
                    MPI_Comm_free(&_communicator);
            }
            delete [] _rq;
#else
            PetscFree(_idx);
            delete [] _di;
            delete [] _da;
#endif
            delete [] _ogj;
            delete [] _range;
            delete [] _oi;
            _di = _ogj = _oi = nullptr;
            _range = nullptr;
            _da = nullptr;
            std::map<unsigned short, std::vector<int>>().swap(_send);
            vectorNeighbor().swap(_recv);
        }
        constexpr int getDof() const { return _dof * _bs; }
        return_type solve(K* rhs, const unsigned short& mu) {
#if !HPDDM_PETSC
            if(_s) {
                K* in, *out;
                const unsigned int n = _s->getDof();
                in = new K[mu * n];
                out = new K[mu * n]();
                for(unsigned short i = 0; i < mu; ++i) {
                    std::copy_n(rhs + i * _dof * _bs, _dof * _bs, in + i * n);
                    std::fill(in + i * n + _dof * _bs, in + (i + 1) * n, K());
                }
                bool allocate = _s->setBuffer();
                _s->exchange(in, mu);
                _s->clearBuffer(allocate);
                IterativeMethod::solve(*_s, in, out, mu, _communicator);
                for(unsigned short i = 0; i < mu; ++i)
                    std::copy_n(out + i * n, _dof * _bs, rhs + i * n);
                delete [] out;
                delete [] in;
            }
            else {
                if(_mu != mu) {
                    delete [] _x;
                    _x = new K[mu * _dof * _bs]();
                    _mu = mu;
                }
                const std::string prefix = OptionsPrefix<K>::prefix();
                if(Option::get()->val<char>(prefix + "krylov_method", HPDDM_KRYLOV_METHOD_GMRES) != HPDDM_KRYLOV_METHOD_NONE)
                    IterativeMethod::solve(*this, rhs, _x, mu, _communicator);
                else
                    Solver<K>::solve(rhs, _x, mu);
                std::copy_n(_x, mu * _dof * _bs, rhs);
            }
#else
            PetscFunctionBeginUser;
            PetscCall(PCHPDDMSolve_Private(_s, rhs, mu));
            PetscFunctionReturn(0);
#endif
        }
        decltype(_s) getSubdomain() const {
            return _s;
        }
#if !HPDDM_PETSC
        void setParent(decltype(_p) const p) {
            _p = p;
        }
        int GMV(const K* const in, K* const out, const int& mu = 1) const {
            exchange<'N'>(in, nullptr, mu);
            Wrapper<K>::template bsrmm<Solver<K>::_numbering>(S == 'S', &_dof, &mu, &_bs, _da, _di, _dj, in, out);
            wait<'N'>(_o + (mu - 1) * _off * _bs);
            Wrapper<K>::template bsrmm<Solver<K>::_numbering>("N", &_dof, &mu, &_off, &_bs, &(Wrapper<K>::d__1), false, _oa, _oi, _oj, _o, &(Wrapper<K>::d__1), out);
            if(S == 'S') {
                Wrapper<K>::template bsrmm<Solver<K>::_numbering>(&(Wrapper<K>::transc), &_dof, &mu, &_off, &_bs, &(Wrapper<K>::d__1), false, _oa, _oi, _oj, in, &(Wrapper<K>::d__0), _o);
                exchange<'T'>(nullptr, out, mu);
                wait<'T'>(out + (mu - 1) * _dof * _bs);
            }
            return 0;
        }
        template<bool>
        int apply(const K* const in, K* const out, const unsigned short& mu = 1, K* = nullptr) const {
#ifdef DMUMPS
            if(DMatrix::_n)
#endif
                Solver<K>::solve(in, out, mu);
            return 0;
        }
        template<bool = false>
        bool start(const K* const, K* const, const unsigned short& mu = 1) const {
            if(_off) {
                unsigned short k = 1;
                const std::string prefix = OptionsPrefix<K>::prefix();
                const Option& opt = *Option::get();
                if(opt.any_of(prefix + "krylov_method", { HPDDM_KRYLOV_METHOD_GCRODR, HPDDM_KRYLOV_METHOD_BGCRODR }) && !opt.val<unsigned short>(prefix + "recycle_same_system"))
                    k = std::max(opt.val<int>(prefix + "recycle", 1), 1);
                K** ptr = const_cast<K**>(&_o);
                *ptr = new K[mu * k * _off * _bs]();
                return true;
            }
            else
                return false;
        }
        void end(const bool free) const {
            if(free)
                delete [] _o;
        }
#endif
        static constexpr underlying_type<K>* getScaling() { return nullptr; }
        static constexpr std::unordered_map<unsigned int, K> boundaryConditions() { return std::unordered_map<unsigned int, K>(); }
        typename std::conditional<HPDDM_PETSC, Mat, MatrixCSR<K>*>::type buildMatrix(const InexactCoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>* in, const std::vector<unsigned int>& displs, const unsigned int* const ranges, const std::vector<std::vector<unsigned int>>& off, const std::vector<std::vector<std::pair<unsigned short, unsigned short>>>& reduction = std::vector<std::vector<std::pair<unsigned short, unsigned short>>>(), const std::map<std::pair<unsigned short, unsigned short>, unsigned short>& sizes = std::map<std::pair<unsigned short, unsigned short>, unsigned short>(), const std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>& extra = std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>(), const std::tuple<int*, K*, MPI_Request*>& transfer = std::tuple<int*, K*, MPI_Request*>()) const {
#if HPDDM_PETSC
            char S;
            {
                Mat A;
                KSPGetOperators(_s->ksp, &A, nullptr);
                PetscBool symmetric;
                PetscObjectTypeCompare((PetscObject)A, MATMPISBAIJ, &symmetric);
                S = (symmetric ? 'S' : 'G');
            }
#endif
            int rank, size;
            MPI_Comm_size(in->_communicator, &size);
            MPI_Comm_rank(in->_communicator, &rank);
            integer_type* di = new integer_type[in->_dof + in->_off + displs.back() + 1];
            const int bss = in->_bs * in->_bs;
            di[0] = 0;
            for(unsigned int i = 0; i < in->_dof; ++i)
                di[i + 1] = di[i] + in->_oi[i + 1] - in->_oi[i] + in->_di[i + 1] - in->_di[i];
            std::vector<std::vector<std::pair<int, K*>>> to;
            if(S != 'S') {
                to.resize(in->_off);
                for(unsigned int i = 0; i < in->_dof + 1; ++i)
                    for(unsigned int j = in->_oi[i]; j < in->_oi[i + 1]; ++j)
                        to[in->_oj[j - (super::_numbering == 'F')] - (super::_numbering == 'F')].emplace_back(i, in->_oa + (j - (super::_numbering == 'F')) * bss);
                for(unsigned int i = 0; i < in->_off; ++i) {
                    std::sort(to[i].begin(), to[i].end(), [](const std::pair<int, K*>& lhs, const std::pair<int, K*>& rhs) { return lhs.first < rhs.first; });
                    di[in->_dof + i + 1] = di[in->_dof + i] + to[i].size();
                }
            }
            else {
                if(in != this) {
                    for(std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>::const_iterator it = extra.cbegin(); it != extra.cend(); ++it) {
                        if(bss > 1)
                            std::for_each(di + std::get<1>(it->second) + 1, di + in->_dof + 1, [&](integer_type& k) { k += std::get<2>(it->second).size(); });
                        else {
                            unsigned int row = 0;
                            for(const unsigned short& p : std::get<2>(it->second))
                                row += sizes.at(std::make_pair(p, p));
                            for(unsigned short i = 0; i < std::get<0>(it->second); ++i)
                                di[std::get<1>(it->second) + i + 1] += (i + 1) * row;
                            std::for_each(di + std::get<1>(it->second) + std::get<0>(it->second) + 1, di + in->_dof + 1, [&](integer_type& k) { k += std::get<0>(it->second) * row; });
                        }
                    }
                }
                std::fill(di + in->_dof + 1, di + in->_dof + in->_off + displs.back() + 1, di[in->_dof]);
            }
            std::vector<std::pair<std::map<std::pair<unsigned short, unsigned short>, unsigned short>::const_iterator, unsigned short>> rows;
            std::unordered_map<unsigned short, unsigned int> map;
            std::vector<std::pair<int*, K*>>* recv;
            std::vector<int> nnz;
            std::vector<std::pair<unsigned int, K*>>* oa = nullptr;
            if(in == this) {
                recv = new std::vector<std::pair<int*, K*>>[S == 'S' ? 2 : 1]();
                MPI_Request* rqRecv = new MPI_Request[2 * (in->_send.size() + in->_recv.size())];
                MPI_Request* rqSend;
                std::vector<std::pair<int*, K*>> send;
                {
                    std::set<int> unique;
                    for(int i = 0; i < in->_dof; ++i) {
                        for(int j = in->_oi[i] - (super::_numbering == 'F'); j < in->_oi[i + 1] - (super::_numbering == 'F'); ++j)
                            unique.insert(in->_ogj[j] - (super::_numbering == 'F'));
                    }
                    nnz.reserve(unique.size());
                    std::copy(unique.cbegin(), unique.cend(), std::back_inserter(nnz));
                }
                if(S == 'S') {
                    rqSend = rqRecv + 2 * in->_send.size();
                    recv[1].reserve(in->_send.size());
                    send.reserve(in->_recv.size());
                    std::map<unsigned short, std::vector<int>>::const_iterator it = in->_send.cbegin();
                    for(unsigned short i = 0; i < in->_send.size(); ++i) {
                        recv[1].emplace_back(std::make_pair(new int[2 * (displs[i + 1] - displs[i])], nullptr));
                        MPI_Irecv(recv[1].back().first, 2 * (displs[i + 1] - displs[i]), MPI_INT, it->first, 11, in->_communicator, rqRecv + i);
                        ++it;
                    }
                    for(unsigned short j = 0; j < in->_recv.size(); ++j) {
                        send.emplace_back(std::make_pair(new int[2 * off[j].size()], nullptr));
                        unsigned int nnz = 0;
                        for(unsigned int i = 0; i < off[j].size(); ++i) {
                            send.back().first[i] = in->_oi[off[j][i] + 1] - in->_oi[off[j][i]];
                            nnz += send.back().first[i];
                        }
                        std::copy(off[j].cbegin(), off[j].cend(), send.back().first + off[j].size());
                        MPI_Isend(send.back().first, 2 * off[j].size(), MPI_INT, in->_recv[j].first, 11, in->_communicator, rqSend + j);
                        send.back().second = new K[nnz + (nnz + (off[j].size() * (off[j].size() + 1)) / 2) * bss]();
                        int* ja = reinterpret_cast<int*>(send.back().second + nnz * bss);
                        nnz = 0;
                        for(unsigned int i = 0; i < off[j].size(); ++i) {
                            for(unsigned k = in->_oi[off[j][i]] - (super::_numbering == 'F'); k < in->_oi[off[j][i] + 1] - (super::_numbering == 'F'); ++k) {
                                std::copy_n(in->_oa + k * bss, bss, send.back().second + nnz * bss);
                                ja[nnz++] = in->_ogj[k];
                            }
                        }
                        K* const diag = send.back().second + nnz * (1 + bss);
                        std::fill_n(diag, ((off[j].size() * (off[j].size() + 1)) / 2) * bss, K());
                        for(unsigned int i = 0; i < off[j].size(); ++i)
                            for(unsigned int k = i; k < off[j].size(); ++k) {
                                integer_type* const pt = std::lower_bound(in->_dj + in->_di[off[j][i]] - (super::_numbering == 'F'), in->_dj + in->_di[off[j][i] + 1] - (super::_numbering == 'F'), off[j][k] + (super::_numbering == 'F'));
                                if(pt != in->_dj + in->_di[off[j][i] + 1] - (super::_numbering == 'F') && *pt == off[j][k] + (super::_numbering == 'F'))
                                    std::copy_n(in->_da + std::distance(in->_dj, pt) * bss, bss, diag + (i * off[j].size() + k - ((i * (i - 1)) / 2 + i)) * bss);
                            }
                        MPI_Isend(send.back().second, nnz + (nnz + (off[j].size() * (off[j].size() + 1)) / 2) * bss, Wrapper<K>::mpi_type(), in->_recv[j].first, 12, in->_communicator, rqSend + in->_recv.size() + j);
                    }
                    for(unsigned short i = 0; i < in->_send.size(); ++i) {
                        it = in->_send.cbegin();
                        int index;
                        MPI_Waitany(in->_send.size(), rqRecv, &index, MPI_STATUS_IGNORE);
                        std::advance(it, index);
                        unsigned int nnz = std::accumulate(recv[1][index].first, recv[1][index].first + displs[index + 1] - displs[index], 0);
                        recv[1][index].second = new K[nnz + (nnz + ((displs[index + 1] - displs[index]) * (displs[index + 1] - displs[index] + 1)) / 2) * bss];
                        MPI_Irecv(recv[1][index].second, nnz + (nnz + ((displs[index + 1] - displs[index]) * (displs[index + 1] - displs[index] + 1)) / 2) * bss, Wrapper<K>::mpi_type(), it->first, 12, in->_communicator, rqRecv + in->_send.size() + index);
                    }
                    oa = new std::vector<std::pair<unsigned int, K*>>[in->_dof + in->_off + displs.back()]();
                    for(unsigned short i = 0; i < in->_send.size(); ++i) {
                        it = in->_send.cbegin();
                        int index;
                        MPI_Waitany(in->_send.size(), rqRecv + in->_send.size(), &index, MPI_STATUS_IGNORE);
                        std::advance(it, index);
                        unsigned int accumulate = std::accumulate(recv[1][index].first, recv[1][index].first + displs[index + 1] - displs[index], 0);
                        int* ja = reinterpret_cast<int*>(recv[1][index].second + accumulate * bss);
                        accumulate = 0;
                        for(unsigned int j = 0; j < displs[index + 1] - displs[index]; ++j) {
                            for(unsigned int k = 0; k < recv[1][index].first[j]; ++k) {
                                if(ja[accumulate] >= _range[0] && ja[accumulate] <= _range[1])
                                    oa[ja[accumulate] - _range[0]].emplace_back(in->_dof + displs[index] + j, recv[1][index].second + accumulate * bss);
                                else {
                                    bool kept = false;
                                    for(unsigned int i = index + 1; i < in->_send.size() && !kept; ++i) {
                                        if(ja[accumulate] >= ranges[2 * i] && ja[accumulate] <= ranges[2 * i + 1]) {
                                            const int* pt = std::lower_bound(recv[1][i].first + displs[i + 1] - displs[i], recv[1][i].first + 2 * (displs[i + 1] - displs[i]), ja[accumulate] - ranges[2 * i]);
                                            if(pt != recv[1][i].first + 2 * (displs[i + 1] - displs[i]) && *pt == ja[accumulate] - ranges[2 * i])
                                                oa[in->_dof + displs[index] + j].emplace_back(in->_dof + displs[i] + std::distance(recv[1][i].first + displs[i + 1] - displs[i], std::lower_bound(recv[1][i].first + displs[i + 1] - displs[i], recv[1][i].first + 2 * (displs[i + 1] - displs[i]), ja[accumulate] - ranges[2 * i])), recv[1][index].second + accumulate * bss);
                                            kept = true;
                                        }
                                    }
                                    for(unsigned int i = in->_send.size(); i < in->_send.size() + in->_recv.size() && !kept; ++i) {
                                        if(ja[accumulate] >= ranges[2 * i] && ja[accumulate] <= ranges[2 * i + 1]) {
                                            std::vector<int>::const_iterator pt = std::lower_bound(nnz.cbegin(), nnz.cend(), ja[accumulate] - (super::_numbering == 'F'));
                                            if(pt != nnz.cend() && *pt == ja[accumulate] - (super::_numbering == 'F'))
                                                oa[in->_dof + displs[index] + j].emplace_back(in->_dof + displs.back() + std::distance(nnz.cbegin(), pt), recv[1][index].second + accumulate * bss);
                                            kept = true;
                                        }
                                    }
                                }
                                ++accumulate;
                            }
                        }
                        for(unsigned int j = 0; j < displs[index + 1] - displs[index]; ++j) {
                            for(unsigned int k = j; k < displs[index + 1] - displs[index]; ++k)
                                oa[in->_dof + displs[index] + j].emplace_back(in->_dof + displs[index] + k, recv[1][index].second + accumulate + (accumulate + j * (displs[index + 1] - displs[index]) - (j * (j - 1)) / 2 + k - j) * bss);
                        }
                    }
                    unsigned int accumulate = 0;
                    for(unsigned int i = 0; i < in->_dof + in->_off + displs.back(); ++i) {
                        accumulate += oa[i].size();
                        di[i + 1] += accumulate;
                        std::sort(oa[i].begin(), oa[i].end());
                    }
                    for(unsigned int i = 0; i < 2 * in->_recv.size(); ++i) {
                        int index;
                        MPI_Waitany(2 * in->_recv.size(), rqSend, &index, MPI_STATUS_IGNORE);
                        if(index < in->_recv.size())
                            delete [] send[index].first;
                        else
                            delete [] send[index - in->_recv.size()].second;
                    }
                    send.clear();
                }
                rqSend = rqRecv + 2 * in->_recv.size();
                recv[0].reserve(in->_recv.size());
                send.reserve(in->_send.size());
                for(unsigned short j = 0; j < in->_recv.size(); ++j) {
                    recv[0].emplace_back(std::make_pair(new int[2 * in->_recv[j].second.size()], nullptr));
                    MPI_Irecv(recv[0].back().first, 2 * in->_recv[j].second.size(), MPI_INT, in->_recv[j].first, 322, in->_communicator, rqRecv + j);
                }
                for(const std::pair<const unsigned short, std::vector<int>>& p : in->_send) {
                    send.emplace_back(std::make_pair(new int[2 * p.second.size()](), nullptr));
                    std::vector<int> col;
                    std::vector<K> val;
                    for(unsigned int i = 0; i < p.second.size(); ++i) {
                        for(unsigned int k = (S != 'S' ? 0 : i); k < p.second.size(); ++k) {
                            integer_type* const pt = std::lower_bound(in->_dj + in->_di[p.second[i]] - (super::_numbering == 'F'), in->_dj + in->_di[p.second[i] + 1] - (super::_numbering == 'F'), p.second[k] + (super::_numbering == 'F'));
                            if(pt != in->_dj + in->_di[p.second[i] + 1] - (super::_numbering == 'F') && *pt == p.second[k] + (super::_numbering == 'F')) {
                                send.back().first[2 * i]++;
                                col.emplace_back(k);
                                val.insert(val.end(), in->_da + std::distance(in->_dj, pt) * bss, in->_da + (std::distance(in->_dj, pt) + 1) * bss);
                            }
                        }
                        send.back().first[2 * i + 1] = in->_oi[p.second[i] + 1] - in->_oi[p.second[i]];
                        for(unsigned int k = in->_oi[p.second[i]] - (super::_numbering == 'F'); k < in->_oi[p.second[i] + 1] - (super::_numbering == 'F'); ++k) {
                            col.emplace_back(in->_ogj[k] - (super::_numbering == 'F'));
                            val.insert(val.end(), in->_oa + k * bss, in->_oa + (k + 1) * bss);
                        }
                    }
                    MPI_Isend(send.back().first, 2 * p.second.size(), MPI_INT, p.first, 322, in->_communicator, rqSend);
                    unsigned int accumulate = 0;
                    for(unsigned int i = 0; i < p.second.size(); ++i)
                        accumulate += send.back().first[2 * i] + send.back().first[2 * i + 1];
                    send.back().second = new K[accumulate * (1 + bss)];
                    accumulate = 0;
                    for(unsigned int i = 0; i < p.second.size(); ++i) {
                        std::copy_n(col.cbegin() + accumulate, send.back().first[2 * i], send.back().second + accumulate * (1 + bss));
                        std::copy_n(val.cbegin() + accumulate * bss, send.back().first[2 * i] * bss, send.back().second + accumulate * (1 + bss) + send.back().first[2 * i]);
                        std::copy_n(col.cbegin() + accumulate + send.back().first[2 * i], send.back().first[2 * i + 1], send.back().second + (accumulate + send.back().first[2 * i]) * (1 + bss));
                        std::copy_n(val.cbegin() + (accumulate + send.back().first[2 * i]) * bss, send.back().first[2 * i + 1] * bss, send.back().second + (accumulate + send.back().first[2 * i]) * (1 + bss) + send.back().first[2 * i + 1]);
                        accumulate += send.back().first[2 * i] + send.back().first[2 * i + 1];
                    }
                    MPI_Isend(send.back().second, accumulate * (1 + bss), Wrapper<K>::mpi_type(), p.first, 323, in->_communicator, rqSend + in->_send.size());
                    ++rqSend;
                }
                rqSend -= in->_send.size();
                for(unsigned short j = 0; j < in->_recv.size(); ++j) {
                    int index;
                    MPI_Waitany(in->_recv.size(), rqRecv, &index, MPI_STATUS_IGNORE);
                    unsigned int nnz = std::accumulate(recv[0][index].first, recv[0][index].first + 2 * in->_recv[index].second.size(), 0);
                    recv[0][index].second = new K[nnz * (1 + bss)];
                    MPI_Irecv(recv[0][index].second, nnz * (1 + bss), Wrapper<K>::mpi_type(), in->_recv[index].first, 323, in->_communicator, rqRecv + in->_recv.size() + index);
                }
                for(unsigned short j = 0; j < in->_recv.size(); ++j) {
                    int index;
                    MPI_Waitany(in->_recv.size(), rqRecv + in->_recv.size(), &index, MPI_STATUS_IGNORE);
                    unsigned int accumulate = 0;
                    for(unsigned int j = 0; j < in->_recv[index].second.size(); ++j) {
                        unsigned int onnz = 0;
                        std::vector<int>::const_iterator it = nnz.cbegin();
                        for(unsigned int k = 0; k < recv[0][index].first[2 * j + 1]; ++k) {
                            it = std::lower_bound(it, nnz.cend(), HPDDM::lround(HPDDM::abs(recv[0][index].second[accumulate + recv[0][index].first[2 * j] * (1 + bss) + k])));
                            if(it != nnz.cend() && *it == HPDDM::lround(HPDDM::abs(recv[0][index].second[accumulate + recv[0][index].first[2 * j] * (1 + bss) + k])))
                                ++onnz;
                        }
                        for(unsigned int k = in->_recv[index].second[j]; k < in->_off; ++k)
                            di[in->_dof + displs.back() + k + 1] += recv[0][index].first[2 * j] + onnz;
                        accumulate += (recv[0][index].first[2 * j] + recv[0][index].first[2 * j + 1]) * (1 + bss);
                    }
                }
                for(unsigned int i = 0; i < 2 * in->_send.size(); ++i) {
                    int index;
                    MPI_Waitany(2 * in->_send.size(), rqSend, &index, MPI_STATUS_IGNORE);
                    if(index < in->_send.size())
                        delete [] send[index].first;
                    else
                        delete [] send[index - in->_send.size()].second;
                }
                delete [] rqRecv;
            }
            else if(!sizes.empty()) {
                std::map<unsigned short, unsigned short> nnz;
                rows.reserve(sizes.size());
                rows.emplace_back(std::make_pair(sizes.cbegin(), in->_bs > 1 ? in->_bs : sizes.at(std::make_pair(sizes.cbegin()->first.first, sizes.cbegin()->first.first))));
                for(std::map<std::pair<unsigned short, unsigned short>, unsigned short>::const_iterator it = sizes.cbegin(); it != sizes.cend(); ++it) {
                    if(it->first.first != rows.back().first->first.first)
                        rows.emplace_back(std::make_pair(it, in->_bs > 1 ? in->_bs : sizes.at(std::make_pair(it->first.first, it->first.first))));
                }
                rows.emplace_back(std::make_pair(sizes.cend(), 0));
                unsigned int accumulate = 0;
                for(unsigned short k = 0; k < rows.size() - 1; ++k) {
                    unsigned int row = 0;
                    if(in->_bs == 1) {
                        for(std::map<std::pair<unsigned short, unsigned short>, unsigned short>::const_iterator it = rows[k].first; it != rows[k + 1].first; ++it)
                            row += sizes.at(std::make_pair(it->first.second, it->first.second));
                        map[rows[k].first->first.first] = accumulate;
                        for(unsigned int j = 0; j < rows[k].second; ++j)
                            di[in->_dof + accumulate + j + 1] += row * (j + 1) - (S == 'S' ? (j * (j + 1)) / 2 : 0);
                        accumulate += rows[k].second;
                        for(unsigned int j = accumulate; j < in->_off + displs.back(); ++j)
                            di[in->_dof + j + 1] += row * rows[k].second - (S == 'S' ? (rows[k].second * (rows[k].second - 1)) / 2 : 0);
                    }
                    else {
                        row += std::distance(rows[k].first, rows[k + 1].first);
                        map[rows[k].first->first.first] = accumulate;
                        di[in->_dof + accumulate + 1] += row;
                        accumulate += 1;
                        for(unsigned int j = accumulate; j < in->_off + displs.back(); ++j)
                            di[in->_dof + j + 1] += row;
                    }
                }
            }
            integer_type* dj = new integer_type[di[in->_dof + in->_off + displs.back()]];
            K* da = new K[di[in->_dof + in->_off + displs.back()] * bss]();
            for(unsigned int i = 0; i < in->_dof; ++i) {
                for(unsigned int j = in->_di[i]; j < in->_di[i + 1]; ++j)
                    dj[di[i] + j - in->_di[i]] = in->_dj[j - (super::_numbering == 'F')] - (super::_numbering == 'F');
                for(unsigned int j = in->_di[i]; j < in->_di[i + 1]; ++j) {
#if HPDDM_PETSC
                    if(in == this && S == 'S' && i == in->_dj[j])
                        Wrapper<K>::template omatcopy<'T'>(in->_bs, in->_bs, in->_da + (j - (super::_numbering == 'F')) * bss, in->_bs, da + (di[i] + j - in->_di[i]) * bss, in->_bs);
                    else
#endif
                        std::copy_n(in->_da + (j - (super::_numbering == 'F')) * bss, bss, da + (di[i] + j - in->_di[i]) * bss);
                }
                unsigned int shift[2] = { 0, 0 };
                if(S == 'S') {
                    if(in == this) {
                        for(unsigned int j = 0; j < oa[i].size(); ++j) {
                            dj[di[i] + in->_di[i + 1] - in->_di[i] + j] = oa[i][j].first;
                            Wrapper<K>::template omatcopy<'T'>(in->_bs, in->_bs, oa[i][j].second, in->_bs, da + (di[i] + in->_di[i + 1] - in->_di[i] + j) * bss, in->_bs);
                        }
                        shift[0] = oa[i].size();
                        shift[1] = displs.back();
                    }
                    else if(!extra.empty()) {
                        std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>::const_iterator it = extra.cbegin();
                        while(it != extra.cend()) {
                            if(std::get<1>(it->second) <= i && i < (std::get<1>(it->second) + std::get<0>(it->second)))
                                break;
                            ++it;
                        }
                        if(it != extra.cend()) {
                            for(const unsigned short& p : std::get<2>(it->second)) {
                                for(unsigned int j = 0; j < (in->_bs > 1 ? 1 : sizes.at(std::make_pair(p, p))); ++j)
                                    dj[di[i] + shift[0] + j + in->_di[i + 1] - in->_di[i]] = in->_dof + map[p] + j;
                                shift[0] += (in->_bs > 1 ? 1 : sizes.at(std::make_pair(p, p)));
                            }
                        }
                        shift[1] = displs.back();
                    }
                }
                for(unsigned int j = in->_oi[i]; j < in->_oi[i + 1]; ++j)
                    dj[di[i] + shift[0] + j - in->_oi[i] + in->_di[i + 1] - in->_di[i]] = in->_dof + shift[1] + in->_oj[j - (super::_numbering == 'F')] - (super::_numbering == 'F');
                for(unsigned int j = in->_oi[i]; j < in->_oi[i + 1]; ++j)
                    std::copy_n(in->_oa + (j - (super::_numbering == 'F')) * bss, bss, da + (di[i] + shift[0] + j - in->_oi[i] + in->_di[i + 1] - in->_di[i]) * bss);
            }
            if(S != 'S') {
                for(unsigned int i = 0; i < in->_off; ++i)
                    for(unsigned int j = 0; j < to[i].size(); ++j) {
                        dj[di[in->_dof + i] + j] = to[i][j].first;
                        Wrapper<K>::template omatcopy<'T'>(in->_bs, in->_bs, to[i][j].second, in->_bs, da + (di[in->_dof + i] + j) * bss, in->_bs);
                    }
            }
            if(in == this) {
                for(unsigned short i = 0; i < in->_recv.size(); ++i) {
                    unsigned int accumulate = 0;
                    for(unsigned int j = 0; j < in->_recv[i].second.size(); ++j) {
                        std::vector<std::pair<int, K*>> row;
                        row.reserve(di[in->_dof + displs.back() + in->_recv[i].second[j] + 1] - di[in->_dof + displs.back() + in->_recv[i].second[j]] - (S != 'S' ? to[in->_recv[i].second[j]].size() : 0));
                        for(unsigned int k = 0; k < recv[0][i].first[2 * j]; ++k) {
                            row.emplace_back(in->_recv[i].second[HPDDM::lround(HPDDM::abs(recv[0][i].second[accumulate * (1 + bss) + k]))], recv[0][i].second + accumulate * (1 + bss) + recv[0][i].first[2 * j] + k * bss);
                        }
                        for(unsigned int k = 0; k < recv[0][i].first[2 * j + 1]; ++k) {
                            std::vector<int>::const_iterator pt = std::lower_bound(nnz.cbegin(), nnz.cend(), HPDDM::lround(HPDDM::abs(recv[0][i].second[(accumulate + recv[0][i].first[2 * j]) * (1 + bss) + k])));
                            if(pt != nnz.cend() && *pt == HPDDM::lround(HPDDM::abs(recv[0][i].second[(accumulate + recv[0][i].first[2 * j]) * (1 + bss) + k]))) {
                                row.emplace_back(std::distance(nnz.cbegin(), pt), recv[0][i].second + (accumulate + recv[0][i].first[2 * j]) * (1 + bss) + recv[0][i].first[2 * j + 1] + k * bss);
                            }
                        }
                        std::sort(row.begin(), row.end());
                        for(unsigned int k = 0; k < row.size(); ++k) {
                            dj[di[in->_dof + displs.back() + in->_recv[i].second[j]] + (S != 'S' ? to[in->_recv[i].second[j]].size() : 0) + k] = in->_dof + displs.back() + row[k].first;
                            std::copy_n(row[k].second, bss, da + (di[in->_dof + displs.back() + in->_recv[i].second[j]] + (S != 'S' ? to[in->_recv[i].second[j]].size() : 0) + k) * bss);
                        }
                        accumulate += recv[0][i].first[2 * j] + recv[0][i].first[2 * j + 1];
                    }
                    delete [] recv[0][i].first;
                    delete [] recv[0][i].second;
                }
                if(S == 'S') {
                    for(unsigned int i = in->_dof; i < in->_dof + in->_off + displs.back(); ++i) {
                        for(unsigned int j = 0; j < oa[i].size(); ++j) {
                            dj[di[i] + j] = oa[i][j].first;
                            std::copy_n(oa[i][j].second, bss, da + (di[i] + j) * bss);
                        }
                    }
                    delete [] oa;
                    for(unsigned short i = 0; i < in->_send.size(); ++i) {
                        delete [] recv[1][i].first;
                        delete [] recv[1][i].second;
                    }
                }
                delete [] recv;
            }
            else {
                unsigned int row = 0;
                const std::function<size_t(unsigned short, unsigned short)> key = [](unsigned short i, unsigned short j) { return static_cast<size_t>(i) << 32 | static_cast<size_t>(j); };
                std::unordered_map<size_t, unsigned int> loc;
                for(unsigned short k = 0; k < rows.size() - 1; ++k) {
                    unsigned int col = 0;
                    for(std::map<std::pair<unsigned short, unsigned short>, unsigned short>::const_iterator it = rows[k].first; it != rows[k + 1].first; ++it) {
                        loc[key(rows[k].first->first.first, it->first.second)] = di[in->_dof + map[rows[k].first->first.first]] + (S != 'S' ? to[row].size() : 0) + col;
                        if(in->_bs == 1) {
                            for(unsigned short j = 0; j < sizes.at(std::make_pair(it->first.second, it->first.second)); ++j) {
                                dj[di[in->_dof + map[rows[k].first->first.first]] + (S != 'S' ? to[row].size() : 0) + col] = in->_dof + map[it->first.second] + j;
                                ++col;
                            }
                        }
                        else {
                            dj[di[in->_dof + map[rows[k].first->first.first]] + (S != 'S' ? to[row].size() : 0) + col] = in->_dof + map[it->first.second];
                            ++col;
                        }
                    }
                    if(in->_bs == 1) {
                        for(unsigned short j = 1; j < rows[k].second; ++j)
                            std::copy_n(dj + di[in->_dof + map[rows[k].first->first.first]] + (S != 'S' ? to[row].size() : j), col - (S == 'S' ? j : 0), dj + di[in->_dof + map[rows[k].first->first.first] + j] + (S != 'S' ? to[row + j].size() : 0));
                        row += rows[k].second;
                    }
                    else
                        row += 1;
                }
                for(unsigned int j = 0; j < reduction.size(); ++j) {
                    int index;
                    MPI_Status st;
                    MPI_Waitany(reduction.size(), std::get<2>(transfer), &index, &st);
                    if(st.MPI_TAG != MPI_ANY_TAG && st.MPI_SOURCE != MPI_ANY_SOURCE) {
                        unsigned int accumulate = std::get<0>(transfer)[index];
                        for(unsigned short j = 0; j < reduction[index].size(); ++j) {
                            const unsigned short r1 = reduction[index][j].first;
                            const unsigned short r2 = reduction[index][j].second;
                            const unsigned int row = std::distance(di, std::lower_bound(di + in->_dof, di + in->_dof + in->_off + displs.back(), loc[key(r1, r2)])) - (S == 'S' ? 0 : 1);
                            if(in->_bs == 1) {
                                const unsigned short mu1 = sizes.at(std::make_pair(r1, r1));
                                const unsigned short mu2 = sizes.at(std::make_pair(r2, r2));
                                for(unsigned short nu1 = 0; nu1 < mu1; ++nu1)
                                    for(unsigned short nu2 = (S == 'S' && r1 == r2 ? nu1 : 0); nu2 < mu2; ++nu2) {
                                        if(S != 'S' || r1 != r2)
                                            da[loc[key(r1, r2)] + nu2 + nu1 * (di[row + 1] - di[row]) - (S == 'S' ? (nu1 * (nu1 - 1)) / 2 : 0)] += std::get<1>(transfer)[nu2 * mu1 + nu1 + accumulate];
                                        else
                                            da[loc[key(r1, r2)] + nu2 + nu1 * (di[row + 1] - di[row]) - (nu1 * (nu1 + 1)) / 2] += std::get<1>(transfer)[nu2 * mu1 + nu1 + accumulate];
                                    }
                                accumulate += mu1 * mu2;
                            }
                            else {
                                if(super::_numbering == 'F')
                                    Blas<K>::axpy(&bss, &(Wrapper<K>::d__1), std::get<1>(transfer) + accumulate, &i__1, da + loc[key(r1, r2)] * bss, &i__1);
                                else
                                    for(unsigned short nu1 = 0; nu1 < in->_bs; ++nu1)
                                        for(unsigned short nu2 = 0; nu2 < in->_bs; ++nu2)
                                            da[loc[key(r1, r2)] * bss + nu2 * in->_bs + nu1] += std::get<1>(transfer)[nu2 + nu1 * in->_bs + accumulate];
                                accumulate += bss;
                            }
                        }
                        if(S == 'S') {
                            std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>::const_iterator it = extra.find(index);
                            if(it != extra.cend()) {
                                unsigned int shift = 0;
                                for(const unsigned short& p : std::get<2>(it->second)) {
                                    const unsigned int row = std::get<1>(it->second);
                                    if(in->_bs == 1) {
                                        const unsigned short mu1 = std::get<0>(it->second);
                                        const unsigned short mu2 = sizes.at(std::make_pair(p, p));
                                        for(unsigned short nu1 = 0; nu1 < mu1; ++nu1)
                                            for(unsigned short nu2 = 0; nu2 < mu2; ++nu2)
                                                da[di[row] + in->_di[row + 1] - in->_di[row] + shift + nu2 + nu1 * (di[row + 1] - di[row]) - (nu1 * (nu1 + 1)) / 2] += std::get<1>(transfer)[nu2 + nu1 * mu2 + accumulate];
                                        accumulate += mu1 * mu2;
                                        shift += mu2;
                                    }
                                    else {
                                        if(super::_numbering == 'C')
                                            Blas<K>::axpy(&bss, &(Wrapper<K>::d__1), std::get<1>(transfer) + accumulate, &i__1, da + (di[row] + in->_di[row + 1] - in->_di[row]) * bss + shift, &i__1);
                                        else
                                            for(unsigned short nu1 = 0; nu1 < in->_bs; ++nu1)
                                                for(unsigned short nu2 = 0; nu2 < in->_bs; ++nu2)
                                                    da[(di[row] + in->_di[row + 1] - in->_di[row]) * bss + shift + nu2 * in->_bs + nu1] += std::get<1>(transfer)[nu2 + nu1 * in->_bs + accumulate];
                                        accumulate += bss;
                                        shift += bss;
                                    }
                                }
                            }
                        }
                    }
                }
                delete [] std::get<0>(transfer);
                delete [] std::get<1>(transfer);
                delete [] std::get<2>(transfer);
            }
#if !HPDDM_PETSC
            MatrixCSR<K>* ret = new MatrixCSR<K>(in->_dof + in->_off + displs.back(), in->_dof + in->_off + displs.back(), di[in->_dof + in->_off + displs.back()], da, di, dj, S == 'S', true);
            if(bss > 1) {
                int* di = new int[ret->_n * in->_bs + 1];
                for(int i = 0; i < ret->_n; ++i)
                    for(int k = 0; k < in->_bs; ++k)
                        di[i * in->_bs + k] = ret->_ia[i] * bss + (ret->_ia[i + 1] - ret->_ia[i]) * in->_bs * k - (S == 'S' ? i * (in->_bs * (in->_bs - 1)) / 2 + (k * (k - 1)) / 2 : 0);
                ret->_nnz *= bss;
                if(S == 'S')
                    ret->_nnz -= ret->_n * (in->_bs * (in->_bs - 1)) / 2;
                di[ret->_n * in->_bs] = ret->_nnz;
                int* dj = new int[ret->_nnz];
                for(int i = 0; i < ret->_n; ++i) {
                    for(int j = ret->_ia[i]; j < ret->_ia[i + 1]; ++j)
                        for(int k = 0; k < in->_bs; ++k)
                            dj[di[i * in->_bs] + (j - ret->_ia[i]) * in->_bs + k] = ret->_ja[j] * in->_bs + k;
                    for(int k = 1; k < in->_bs; ++k)
                        std::copy_n(dj + di[i * in->_bs] + (S == 'S' ? k : 0), (ret->_ia[i + 1] - ret->_ia[i]) * in->_bs - (S == 'S' ? k : 0), dj + di[i * in->_bs + k]);
                }
                K* da = new K[ret->_nnz];
                if(S != 'S') {
                    if(super::_numbering == 'F') {
                        for(int i = 0; i < ret->_n; ++i)
                            Wrapper<K>::template omatcopy<'T'>((ret->_ia[i + 1] - ret->_ia[i]) * in->_bs, in->_bs, ret->_a + ret->_ia[i] * bss, in->_bs, da + ret->_ia[i] * bss, (ret->_ia[i + 1] - ret->_ia[i]) * in->_bs);
                    }
                    else {
                        for(int i = 0; i < ret->_n; ++i)
                            for(int j = ret->_ia[i]; j < ret->_ia[i + 1]; ++j)
                                Wrapper<K>::template omatcopy<'N'>(in->_bs, in->_bs, ret->_a + j * bss, in->_bs, da + ret->_ia[i] * bss + (j - ret->_ia[i]) * in->_bs, (ret->_ia[i + 1] - ret->_ia[i]) * in->_bs);
                    }
                }
                else {
                    for(int i = 0; i < ret->_n; ++i)
                        for(int j = ret->_ia[i]; j < ret->_ia[i + 1]; ++j) {
                            for(int nu = 0; nu < in->_bs; ++nu) {
                                for(int mu = (j == ret->_ia[i] ? nu : 0); mu < in->_bs; ++mu) {
                                    da[di[i * in->_bs + nu] + (j - ret->_ia[i]) * in->_bs + mu - nu] = ret->_a[j * bss + (super::_numbering == 'F' ? mu * in->_bs + nu : mu + nu * in->_bs)];
                                }
                            }
                        }
                }
                ret->_n *= in->_bs;
                ret->_m *= in->_bs;
                delete [] ret->_ia;
                ret->_ia = di;
                delete [] ret->_ja;
                ret->_ja = dj;
                delete [] ret->_a;
                ret->_a = da;
            }
            if(S == 'S') {
                MatrixCSR<K>* t = new MatrixCSR<K>(ret->_n, ret->_n, ret->_ia[ret->_n], true);
                Wrapper<K>::template csrcsc<'C', HPDDM_NUMBERING>(&ret->_n, ret->_a, ret->_ja, ret->_ia, t->_a, t->_ja, t->_ia);
                delete ret;
                ret = t;
            }
            else if(HPDDM_NUMBERING == 'F') {
                std::for_each(ret->_ja, ret->_ja + ret->_ia[ret->_n], [](int& i) { ++i; });
                std::for_each(ret->_ia, ret->_ia + ret->_n, [](int& i) { ++i; });
            }
#else
            Mat ret;
            MatCreate(PETSC_COMM_SELF, &ret);
            {
                const char* prefix;
                KSPGetOptionsPrefix(_s->ksp, &prefix);
                MatSetOptionsPrefix(ret, prefix);
            }
            MatSetFromOptions(ret);
            MatSetBlockSize(ret, in->_bs);
            MatSetSizes(ret, (in->_dof + in->_off + displs.back()) * in->_bs, (in->_dof + in->_off + displs.back()) * in->_bs, (in->_dof + in->_off + displs.back()) * in->_bs, (in->_dof + in->_off + displs.back()) * in->_bs);
            if(S == 'S') {
                MatSetType(ret, MATSEQSBAIJ);
                MatSeqSBAIJSetPreallocationCSR(ret, in->_bs, di, dj, da);
            }
            else {
                if(in->_bs > 1) {
                    MatSetType(ret, MATSEQBAIJ);
                    MatSeqBAIJSetPreallocationCSR(ret, in->_bs, di, dj, da);
                }
                else {
                    MatSetType(ret, MATSEQAIJ);
                    MatSeqAIJSetPreallocationCSR(ret, di, dj, da);
                }
            }
            delete [] da;
            delete [] dj;
            delete [] di;
#endif
            return ret;
        }
        typedef Solver
#if !HPDDM_PETSC
                      <K>
#endif
                          super;

        return_type buildThree(CoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>* const& A, const std::vector<std::vector<std::pair<unsigned short, unsigned short>>>& reduction, const std::map<std::pair<unsigned short, unsigned short>, unsigned short>& sizes, const std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>& extra
#if HPDDM_PETSC
                , Mat* D, Mat* N, PC_HPDDM_Level* const level
#endif
                ) {
#if HPDDM_PETSC
            PetscFunctionBeginUser;
#endif
            if(DMatrix::_communicator != MPI_COMM_NULL) {
#if HPDDM_PETSC
                char S;
                {
                    Mat A;
                    KSPGetOperators(level->ksp, &A, nullptr);
                    PetscBool symmetric;
                    PetscObjectTypeCompare((PetscObject)A, MATMPISBAIJ, &symmetric);
                    S = (symmetric ? 'S' : 'G');
                }
#endif
                std::tuple<int*, K*, MPI_Request*> transfer;
                std::get<0>(transfer) = new int[reduction.size() + 1]();
                for(unsigned short i = 0; i < reduction.size(); ++i) {
                    std::get<0>(transfer)[i + 1] = std::get<0>(transfer)[i];
                    for(unsigned short j = 0; j < reduction[i].size(); ++j)
                        std::get<0>(transfer)[i + 1] += (_bs > 1 ? _bs * _bs : sizes.at(std::make_pair(reduction[i][j].first, reduction[i][j].first)) * sizes.at(std::make_pair(reduction[i][j].first, reduction[i][j].second)));
                    if(S == 'S') {
                        std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>::const_iterator it = extra.find(i);
                        if(it != extra.cend()) {
                            if(_bs > 1)
                                std::get<0>(transfer)[i + 1] += _bs * _bs * std::get<2>(it->second).size();
                            else {
                                for(std::vector<unsigned short>::const_iterator p = std::get<2>(it->second).cbegin(); p != std::get<2>(it->second).cend(); ++p)
                                    std::get<0>(transfer)[i + 1] += std::get<0>(it->second) * sizes.at(std::make_pair(*p, *p));
                            }
                        }
                    }
                }
                std::get<1>(transfer) = new K[std::get<0>(transfer)[reduction.size()]];
                std::get<2>(transfer) = new MPI_Request[reduction.size()];
                for(unsigned short i = 0; i < reduction.size(); ++i) {
                    if(std::get<0>(transfer)[i + 1] - std::get<0>(transfer)[i])
                        MPI_Irecv(std::get<1>(transfer) + std::get<0>(transfer)[i], std::get<0>(transfer)[i + 1] - std::get<0>(transfer)[i], Wrapper<K>::mpi_type(), i, 300, A->getCommunicator(), std::get<2>(transfer) + i);
                    else
                        std::get<2>(transfer)[i] = MPI_REQUEST_NULL;
                }
                std::vector<std::vector<unsigned int>> off;
                std::vector<unsigned int> displs;
                unsigned int* ranges = nullptr;
                if(S == 'S') {
                    unsigned int* s = new unsigned int[3 * (_recv.size() + _send.size())];
                    ranges = new unsigned int[2 * (_recv.size() + _send.size())];
                    MPI_Request* rq = new MPI_Request[2 * (_recv.size() + _send.size())];
                    displs.resize(_send.size() + 1);
                    off.resize(_recv.size());
                    s += 3 * _recv.size();
                    for(const std::pair<const unsigned short, std::vector<int>>& p : _send) {
#if !HPDDM_PETSC
                        MPI_Irecv(s, 3, MPI_UNSIGNED, p.first, 13, _communicator, rq++);
#else
                        MPI_Irecv(s, 3, MPI_UNSIGNED, p.first, 13, DMatrix::_communicator, rq++);
#endif
                        s += 3;
                    }
                    s -= 3 * (_recv.size() + _send.size());
                    for(unsigned short i = 0; i < _recv.size(); ++i)
#if !HPDDM_PETSC
                        MPI_Irecv(ranges + 2 * (_send.size() + i), 2, MPI_UNSIGNED, _recv[i].first, 13, _communicator, rq++);
#else
                        MPI_Irecv(ranges + 2 * (_send.size() + i), 2, MPI_UNSIGNED, _recv[i].first, 13, DMatrix::_communicator, rq++);
#endif
                    for(const std::pair<const unsigned short, std::vector<int>>& p : _send)
#if !HPDDM_PETSC
                        MPI_Isend(_range, 2, MPI_UNSIGNED, p.first, 13, _communicator, rq++);
#else
                        MPI_Isend(_range, 2, MPI_UNSIGNED, p.first, 13, DMatrix::_communicator, rq++);
#endif
                    for(unsigned short i = 0; i < _recv.size(); ++i) {
                        off[i].reserve(_dof);
                        for(unsigned int j = 0; j < _dof; ++j) {
                            if(_oi[j + 1] - _oi[j]) {
                                integer_type* start = std::lower_bound(_oj + _oi[j] - (super::_numbering == 'F'), _oj + _oi[j + 1] - (super::_numbering == 'F'), _recv[i].second[0] + (super::_numbering == 'F'));
                                integer_type* end = std::lower_bound(start, _oj + _oi[j + 1] - (super::_numbering == 'F'), _recv[i].second.back() + (super::_numbering == 'F'));
                                if(start == _oj + _oi[j + 1] - (super::_numbering == 'F'))
                                    --start;
                                if(end == _oj + _oi[j + 1] - (super::_numbering == 'F'))
                                    --end;
                                if((*start - (super::_numbering == 'F')) <= _recv[i].second.back() && _recv[i].second[0] <= (*end - (super::_numbering == 'F')))
                                    off[i].emplace_back(j);
                            }
                        }
                        s[3 * i] = off[i].size();
                        s[3 * i + 1] = _range[0];
                        s[3 * i + 2] = _range[1];
#if !HPDDM_PETSC
                        MPI_Isend(s + 3 * i, 3, MPI_UNSIGNED, _recv[i].first, 13, _communicator, rq++);
#else
                        MPI_Isend(s + 3 * i, 3, MPI_UNSIGNED, _recv[i].first, 13, DMatrix::_communicator, rq++);
#endif
                    }
                    rq -= 2 * (_recv.size() + _send.size());
                    MPI_Waitall(_send.size(), rq, MPI_STATUSES_IGNORE);
                    displs[0] = 0;
                    for(unsigned short i = 0; i < _send.size(); ++i) {
                        displs[i + 1] = displs[i] + s[3 * (_recv.size() + i)];
                        ranges[2 * i] = s[3 * (_recv.size() + i) + 1];
                        ranges[2 * i + 1] = s[3 * (_recv.size() + i) + 2];
                    }
                    MPI_Waitall(_send.size() + 2 * _recv.size(), rq + _send.size(), MPI_STATUSES_IGNORE);
                    delete [] rq;
                    delete [] s;
                }
                else {
                    displs.resize(1);
                    displs[0] = 0;
                }
                auto overlapDirichlet = buildMatrix(this, displs, ranges, off);
                delete [] ranges;
                ranges = nullptr;
#if !HPDDM_PETSC
                _s = new Schwarz<SUBDOMAIN, COARSEOPERATOR, S, K>;
                Option& opt = *Option::get();
                _s->setPrefix(opt.getPrefix());
#else
                Schwarz<K>* s = new Schwarz<K>;
                _s->P = s;
#endif
                std::vector<unsigned short> o;
                o.reserve(_recv.size() + (S == 'S' ? _send.size() : 0));
                std::vector<std::vector<int>> r(_recv.size() + (S == 'S' ? _send.size() : 0));
                unsigned short j = 0;
                if(S == 'S') {
                    for(const std::pair<const unsigned short, std::vector<int>>& p : _send) {
                        o.emplace_back(p.first);
                        r[j].resize((p.second.size() + displs[j + 1] - displs[j]) * _bs);
                        std::vector<int>::iterator it = r[j].begin();
                        for(unsigned int i = 0; i < displs[j + 1] - displs[j]; ++i)
                            for(unsigned int k = 0; k < _bs; ++k)
                                *it++ = (_dof + displs[j] + i) * _bs + k;
                        for(unsigned int i = 0; i < p.second.size(); ++i)
                            for(unsigned int k = 0; k < _bs; ++k)
                                *it++ = p.second[i] * _bs + k;
                        ++j;
                    }
                }
                for(const pairNeighbor& p : _recv) {
                    o.emplace_back(p.first);
                    r[j].resize((p.second.size() + (S != 'S' ? _send[p.first].size() : off[j - _send.size()].size())) * _bs);
                    if(S != 'S' && p.first < DMatrix::_rank) {
                        for(unsigned int i = 0; i < p.second.size(); ++i)
                            for(unsigned int k = 0; k < _bs; ++k)
                                r[j][i * _bs + k] = (_dof + p.second[i]) * _bs + k;
                        for(unsigned int i = 0; i < _send[p.first].size(); ++i)
                            for(unsigned int k = 0; k < _bs; ++k)
                                r[j][(p.second.size() + i) * _bs + k] = _send[p.first][i] * _bs + k;
                    }
                    else {
                        std::vector<int>::iterator it = r[j].begin();
                        if(S != 'S')
                            for(unsigned int i = 0; i < _send[p.first].size(); ++i)
                                for(unsigned int k = 0; k < _bs; ++k)
                                    *it++ = _send[p.first][i] * _bs + k;
                        else {
                            for(unsigned int i = 0; i < off[j - _send.size()].size(); ++i) {
                                for(unsigned int k = 0; k < _bs; ++k)
                                    *it++ = off[j - _send.size()][i] * _bs + k;
                            }
                        }
                        for(unsigned int i = 0; i < p.second.size(); ++i)
                            for(unsigned int k = 0; k < _bs; ++k)
                                *it++ = (_dof + displs.back() + p.second[i]) * _bs + k;
                    }
                    ++j;
                }
#if !HPDDM_PETSC
                _s->Subdomain<K>::initialize(overlapDirichlet, o, r, &_communicator);
                int m = overlapDirichlet->_n;
                underlying_type<K>* d = new underlying_type<K>[m]();
                _s->initialize(d);
#else
                const char* prefix;
                PetscInt m;
                IS is;
                PetscCall(KSPGetOptionsPrefix(_s->ksp, &prefix));
                PetscCall(MatGetLocalSize(overlapDirichlet, &m, nullptr));
                static_cast<Subdomain<K>*>(s)->initialize(nullptr, o, r, &(DMatrix::_communicator));
                s->setDof(m);
                PetscCall(VecCreateMPI(PETSC_COMM_SELF, m, PETSC_DETERMINE, &_s->D));
                {
                    Mat P;
                    Vec v;
                    PetscCall(KSPGetOperators(_s->ksp, &P, nullptr));
                    {
                        PetscInt n, N;
                        PetscCall(MatGetLocalSize(P, &n, nullptr));
                        PetscCall(MatGetSize(P, &N, nullptr));
                        PetscCall(VecCreateMPI(DMatrix::_communicator, n, N, &v));
                    }
                    PetscCall(ISCreateBlock(PETSC_COMM_SELF, _bs, m / _bs, _idx, PETSC_OWN_POINTER, &is));
                    PetscCall(VecScatterCreate(v, is, _s->D, nullptr, &_s->scatter));
                    PetscCall(VecDestroy(&v));
                }
                _idx = nullptr;
                PetscCall(ISDestroy(&is));
                PetscReal* d;
                if(!std::is_same<PetscScalar, PetscReal>::value)
                    d = new PetscReal[m]();
                else {
                    PetscCall(VecSet(_s->D, 0.0));
                    PetscCall(VecGetArray(_s->D, reinterpret_cast<PetscScalar**>(&d)));
                }
#endif
                std::fill_n(d, _dof * _bs, Wrapper<underlying_type<K>>::d__1);
#if HPDDM_PETSC
                s->initialize(d);
                if(!std::is_same<PetscScalar, PetscReal>::value) {
                    PetscScalar* c;
                    VecGetArray(_s->D, &c);
                    std::copy_n(d, m, c);
                }
                VecRestoreArray(_s->D, nullptr);
#endif
                if(A) {
                    auto overlapNeumann = buildMatrix(A, displs, ranges, off, reduction, sizes, extra, transfer);
#ifdef DMKL_PARDISO
                    delete [] A->_da;
#endif
                    {
#if !HPDDM_PETSC
                        std::vector<int> ia, ja;
                        ia.reserve(overlapDirichlet->_n + 1);
                        ia.emplace_back(HPDDM_NUMBERING == 'F');
                        std::vector<K> a;
                        ja.reserve(overlapDirichlet->_nnz);
                        a.reserve(overlapDirichlet->_nnz);
                        for(unsigned int i = 0; i < overlapDirichlet->_n; ++i) {
                            if(std::abs(d[i]) > HPDDM_EPS)
                                for(unsigned int j = overlapDirichlet->_ia[i] - (HPDDM_NUMBERING == 'F'); j < overlapDirichlet->_ia[i + 1] - (HPDDM_NUMBERING == 'F'); ++j) {
                                    if(std::abs(d[overlapDirichlet->_ja[j]]) > HPDDM_EPS) {
                                        ja.emplace_back(overlapDirichlet->_ja[j]);
                                        a.emplace_back(overlapDirichlet->_a[j] * d[i] * d[ja.back() - (HPDDM_NUMBERING == 'F')]);
                                    }
                                }
                            ia.emplace_back(ja.size() + (HPDDM_NUMBERING == 'F'));
                        }
                        MatrixCSR<K> weighted(overlapDirichlet->_n, overlapDirichlet->_m, a.size(), a.data(), ia.data(), ja.data(), overlapDirichlet->_sym);
                        _s->template solveGEVP<EIGENSOLVER>(overlapNeumann, &weighted);
#elif HPDDM_SLEPC
                        delete [] _di;
                        delete [] _da;
                        delete [] _ogj;
                        delete [] _range;
                        delete [] _oi;
                        _di = _ogj = _oi = nullptr;
                        _range = nullptr;
                        _da = nullptr;
                        std::map<unsigned short, std::vector<int>>().swap(_send);
                        vectorNeighbor().swap(_recv);
                        Mat weighted;
                        PetscCall(MatConvert(overlapDirichlet, MATSAME, MAT_INITIAL_MATRIX, &weighted));
                        PetscCall(MatDiagonalScale(weighted, _s->D, _s->D));
                        EPS eps;
                        ST st;
                        PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
                        PetscCall(EPSSetOptionsPrefix(eps, prefix));
                        PetscCall(EPSSetOperators(eps, overlapNeumann, weighted));
                        PetscCall(EPSSetProblemType(eps, EPS_GHEP));
                        PetscCall(EPSSetTarget(eps, 0.0));
                        PetscCall(EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE));
                        PetscCall(EPSGetST(eps, &st));
                        PetscCall(STSetType(st, STSINVERT));
                        PetscCall(STSetMatStructure(st, SAME_NONZERO_PATTERN));
                        PetscCall(EPSSetFromOptions(eps));
                        PetscInt nev, nconv;
                        PetscCall(EPSGetDimensions(eps, &nev, nullptr, nullptr));
                        PetscCall(EPSSolve(eps));
                        PetscCall(EPSGetConverged(eps, &nconv));
                        level->nu = std::min(nconv, nev);
                        if(level->threshold >= 0.0) {
                            PetscInt i = 0;
                            while(i < level->nu) {
                                PetscScalar eigr;
                                PetscCall(EPSGetEigenvalue(eps, i, &eigr, nullptr));
                                if(HPDDM::abs(eigr) > level->threshold)
                                    break;
                                ++i;
                            }
                            level->nu = i;
                        }
                        if(level->nu) {
                            K** ev = new K*[level->nu];
                            *ev = new K[m * level->nu];
                            for(unsigned short i = 1; i < level->nu; ++i)
                                ev[i] = *ev + i * m;
                            Vec vr;
                            PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, m, ev[0], &vr));
                            for(int i = 0; i < level->nu; ++i) {
                                VecPlaceArray(vr, ev[i]);
                                PetscCall(EPSGetEigenvector(eps, i, vr, nullptr));
                                PetscCall(VecResetArray(vr));
                            }
                            PetscCall(VecDestroy(&vr));
                            s->setVectors(ev);
                        }
                        PetscCall(EPSDestroy(&eps));
                        PetscCall(MatDestroy(&weighted));
#endif
                    }
#if !HPDDM_PETSC
                    delete overlapNeumann;
                    _s->template buildTwo<0>(_communicator, nullptr);
#else
                    *D = overlapDirichlet;
                    *N = overlapNeumann;
#endif
                }
                else {
                    delete [] std::get<0>(transfer);
                    delete [] std::get<1>(transfer);
                    delete [] std::get<2>(transfer);
#if HPDDM_PETSC
                    *N = nullptr;
#endif
                }
#if !HPDDM_PETSC
                _s->callNumfact();
#endif
            }
#if HPDDM_PETSC
            else
                *D = *N = nullptr;
            PetscFunctionReturn(0);
#endif
        }
    private:
#if !HPDDM_PETSC
        template<char T>
        void exchange(const K* const in, K* const out, const unsigned short& mu = 1) const {
            for(unsigned short nu = 0; nu < mu; ++nu) {
                unsigned short i = (T == 'N' ? 0 : _recv.size());
                if(T == 'N')
                    while(i < _recv.size()) {
                        MPI_Irecv(_buff[i], _recv[i].second.size() * _bs, Wrapper<K>::mpi_type(), _recv[i].first, 10, _communicator, _rq + i);
                        ++i;
                    }
                else
                    for(const std::pair<unsigned short, std::vector<int>>& p : _send) {
                        MPI_Irecv(_buff[i], p.second.size() * _bs, Wrapper<K>::mpi_type(), p.first, 20, _communicator, _rq + i);
                        ++i;
                    }
                if(T == 'N')
                    for(const std::pair<unsigned short, std::vector<int>>& p : _send) {
                        for(unsigned int j = 0; j < p.second.size(); ++j)
                            std::copy_n(in + (nu * _dof + p.second[j]) * _bs, _bs, _buff[i] + j * _bs);
                        MPI_Isend(_buff[i], p.second.size() * _bs, Wrapper<K>::mpi_type(), p.first, 10, _communicator, _rq + i);
                        ++i;
                    }
                else {
                    i = 0;
                    while(i < _recv.size()) {
                        for(unsigned int j = 0; j < _recv[i].second.size(); ++j)
                            std::copy_n(_o + (nu * _off + _recv[i].second[j]) * _bs, _bs, _buff[i] + j * _bs);
                        MPI_Isend(_buff[i], _recv[i].second.size() * _bs, Wrapper<K>::mpi_type(), _recv[i].first, 20, _communicator, _rq + i);
                        ++i;
                    }
                }
                if(nu != mu - 1)
                    wait<T>(T == 'N' ? _o + nu * _off * _bs : out + nu * _dof * _bs);
            }
        }
        template<char T>
        void wait(K* const in) const {
            if(T == 'N') {
                for(unsigned short i = 0; i < _recv.size(); ++i) {
                    int index;
                    MPI_Waitany(_recv.size(), _rq, &index, MPI_STATUS_IGNORE);
                    for(unsigned int j = 0; j < _recv[index].second.size(); ++j)
                        std::copy_n(_buff[index] + j * _bs, _bs, in + _recv[index].second[j] * _bs);
                }
                MPI_Waitall(_send.size(), _rq + _recv.size(), MPI_STATUSES_IGNORE);
            }
            else {
                for(unsigned short i = 0; i < _send.size(); ++i) {
                    int index;
                    MPI_Status st;
                    MPI_Waitany(_send.size(), _rq + _recv.size(), &index, &st);
                    const std::vector<int>& v = _send.at(st.MPI_SOURCE);
                    for(unsigned int j = 0; j < v.size(); ++j)
                        Blas<K>::axpy(&_bs, &(Wrapper<K>::d__1), _buff[_recv.size() + index] + j * _bs, &i__1, in + v[j] * _bs, &i__1);
                }
                MPI_Waitall(_recv.size(), _rq, MPI_STATUSES_IGNORE);
            }
        }
#endif
};
} // HPDDM
#endif // HPDDM_INEXACT_COARSE_OPERATOR_HPP_
