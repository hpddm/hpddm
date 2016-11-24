/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
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

#ifndef _HPDDM_INEXACT_COARSE_OPERATOR_
#define _HPDDM_INEXACT_COARSE_OPERATOR_

namespace HPDDM {
template<template<class> class Solver, char S, class K>
class InexactCoarseOperator : public OptionsPrefix, public Solver<K> {
    protected:
        vectorNeighbor   _recv;
        std::map<unsigned short, std::vector<int>> _send;
        K**              _buff;
        K*                  _x;
        K*                 _da;
        K*                 _oa;
        int*               _di;
        int*               _oi;
        int*               _dj;
        int*               _oj;
        mutable K*          _o;
        MPI_Request*       _rq;
        int               _dof;
        int               _off;
        int                _bs;
        MPI_Comm _communicator;
        unsigned short     _mu;
        template<char T>
        void numfact(unsigned int nrow, int* I, int* loc2glob, int* J, K* C, unsigned short* neighbors) {
            _da = C;
            _dj = J;
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
                        MPI_Comm_create(_communicator, aggregate, &(DMatrix::_communicator));
                        MPI_Group_free(&world);
                        bool r = false;
                        std::vector<int*> range;
                        range.reserve((S == 'S' ? 1 : (T == 1 ? 8 : 4)) * nrow);
                        range.emplace_back(J);
                        int* R = new int[nrow + 1];
                        R[0] = (Solver<K>::_numbering == 'F');
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
                            a = new K[(R[nrow] - (Solver<K>::_numbering == 'F')) * _bs * _bs];
                            ia = new int[nrow + 1 + R[nrow] - (Solver<K>::_numbering == 'F')];
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
                    if(!ia) {
                        ia = new int[nrow + 1 + accumulate];
                        ia[0] = I[0];
                        for(unsigned int i = 0; i < nrow; ++i)
                            ia[i + 1] = I[i + 1] + ia[i];
                        if(_mu < _off) {
                            int* const ja = ia + nrow + 1;
                            for(unsigned int i = 0; i < nrow; ++i) {
                                for(unsigned int j = ia[i]; j < ia[i + 1]; ++j) {
                                    const unsigned int idx = J[j - (Solver<K>::_numbering == 'F')];
                                    ja[j - (Solver<K>::_numbering == 'F')] = idx - _di[0];
                                    if(T == 1 && idx > _di[1])
                                        ja[j - (Solver<K>::_numbering == 'F')] -= _di[2];
                                }
                            }
                        }
                        else
                            std::copy_n(I + nrow + 1, accumulate, ia + nrow + 1);
                        a = new K[accumulate * _bs * _bs];
                        std::copy_n(C, accumulate * _bs * _bs, a);
                    }
                    if(_mu < _off) {
#ifdef DMKL_PARDISO
                        loc2glob[0] -= _di[0];
                        loc2glob[1] -= _di[0];
#endif
                        delete [] _di;
                    }
#ifdef DMKL_PARDISO
                    Solver<K>::template numfact<S>(_bs, ia, loc2glob, ia + nrow + 1, a);
                    delete [] a;
#endif
                }
                _di = new int[nrow + 1];
                _di[0] = (Solver<K>::_numbering == 'F');
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
                        const int k = I[i] + _di[i] + j - (Solver<K>::_numbering == 'F' ? 2 : 0);
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
                accumulate = 0;
                if(range.size() > 1) {
                    range.emplace_back(J + I[nrow] + _di[nrow] - (Solver<K>::_numbering == 'F' ? 2 : 0));
                    K* D = new K[(_di[nrow] - (Solver<K>::_numbering == 'F') - (range[1] - range[0])) * _bs * _bs];
                    int* L = new int[_di[nrow] - (Solver<K>::_numbering == 'F') - (range[1] - range[0])];
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
                    std::copy_n(D, (_di[nrow] - (Solver<K>::_numbering == 'F') - (range[1] - range[0])) * _bs * _bs, C + (range[1] - range[0]) * _bs * _bs);
                    std::copy_n(L, _di[nrow] - (Solver<K>::_numbering == 'F') - (range[1] - range[0]), J + (range[1] - range[0]));
                    delete [] L;
                    delete [] D;
                }
                _recv.reserve(allocation.size());
                for(const std::pair<unsigned short, unsigned int>& p : allocation) {
                    _recv.emplace_back(p.first, std::vector<int>());
                    _recv.back().second.reserve(p.second);
                }
                _dof = on.size();
                std::unordered_map<int, int> g2l;
                g2l.reserve(_dof + off.size());
                accumulate = 0;
                for(const int& i : on)
                    g2l.emplace(i - (Solver<K>::_numbering == 'F'), accumulate++);
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
                            sendIdx[accumulate + k] = global->first - (Solver<K>::_numbering == 'F');
                        }
                        MPI_Isend(sendIdx + accumulate, it->second.size(), MPI_INT, it->first, 12, _communicator, rq + distance++);
                        accumulate += it->second.size();
                    }
                    for(unsigned int i = 0; i < infoRecv.size(); ++i) {
                        int index;
                        MPI_Waitany(infoRecv.size(), rq, &index, MPI_STATUS_IGNORE);
                        for(int& j : _send[infoRecv[index]])
                            j = g2l[j];
                    }
                    MPI_Waitall(size - infoRecv.size(), rq + infoRecv.size(), MPI_STATUSES_IGNORE);
                    delete [] sendIdx;
                    delete [] rq;
                }
                else
                    for(std::pair<const unsigned short, std::vector<int>>& i : _send)
                        for(int& j : i.second)
                            j = g2l[j];
                accumulate = 0;
                for(std::pair<const int, unsigned short>& i : off)
                    g2l.emplace(i.first - (Solver<K>::_numbering == 'F'), accumulate++);
                for(std::pair<unsigned short, std::vector<int>>& i : _recv)
                    for(int& j : i.second)
                        j -= _dof;
                std::for_each(J, J + I[nrow] + _di[nrow] - (Solver<K>::_numbering == 'F' ? 2 : 0), [&](int& i) { i = g2l[i - (Solver<K>::_numbering == 'F')] + (Solver<K>::_numbering == 'F'); });
                _buff = new K*[_send.size() + _recv.size()];
                accumulate = 0;
                for(const std::pair<unsigned short, std::vector<int>>& i : _recv)
                    accumulate += i.second.size();
                for(const std::pair<unsigned short, std::vector<int>>& i : _send)
                    accumulate += i.second.size();
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
                _oa = C + (_di[nrow] - (Solver<K>::_numbering == 'F')) * _bs * _bs;
                _oj = J + _di[nrow] - (Solver<K>::_numbering == 'F');
                _off = off.size();
                Option& opt = *Option::get();
                if(DMatrix::_rank != 0)
                    opt.remove("master_verbosity");
            }
            else {
                _dof = nrow;
                _off = 0;
                std::partial_sum(I, I + _dof + 1, I);
                _di = I;
                delete [] neighbors;
#ifdef DMKL_PARDISO
                Solver<K>::template numfact<S>(_bs, I, loc2glob, J, C);
#endif
            }
            _mu = 0;
            OptionsPrefix::setPrefix("master_");
        }
    public:
        InexactCoarseOperator() : OptionsPrefix(), Solver<K>(), _buff(), _x(), _di(), _oi(), _rq(), _off(), _communicator(MPI_COMM_NULL), _mu() { }
        ~InexactCoarseOperator() {
            if(_buff) {
                delete [] *_buff;
                delete [] _buff;
            }
            delete [] _x;
            if(_communicator != MPI_COMM_NULL) {
                MPI_Comm_size(_communicator, &_off);
                if(_off > 1) {
                    delete [] _di;
                    delete [] _da;
                }
#ifdef DMKL_PARDISO
                else {
                    MPI_Comm_size(DMatrix::_communicator, &_off);
                    if((S == 'S' && Option::get()->val<char>("master_not_spd", 0) != 1) || _off > 1)
                        delete [] _da;
                }
#endif
                if(_communicator != DMatrix::_communicator)
                    MPI_Comm_free(&_communicator);
            }
            delete [] _oi;
            delete [] _rq;
        }
        int getDof() const { return _dof * _bs; }
        void solve(K* rhs, const unsigned short& n) {
            if(_mu != n) {
                delete [] _x;
                _x = new K[n * _dof * _bs]();
                _mu = n;
            }
            IterativeMethod::template solve<false>(*this, rhs, _x, n, _communicator);
            std::copy_n(_x, n * _dof * _bs, rhs);
        }
        void GMV(const K* const in, K* const out, const int& mu = 1) const {
            exchange<'N'>(in, nullptr, mu);
            Wrapper<K>::template bsrmm<Solver<K>::_numbering>(S == 'S', &_dof, &mu, &_bs, _da, _di, _dj, in, out);
            wait<'N'>(_o + (mu - 1) * _off * _bs);
            Wrapper<K>::template bsrmm<Solver<K>::_numbering>("N", &_dof, &mu, &_off, &_bs, &(Wrapper<K>::d__1), false, _oa, _oi, _oj, _o, &(Wrapper<K>::d__1), out);
            if(S == 'S') {
                Wrapper<K>::template bsrmm<Solver<K>::_numbering>(&(Wrapper<K>::transc), &_dof, &mu, &_off, &_bs, &(Wrapper<K>::d__1), false, _oa, _oi, _oj, in, &(Wrapper<K>::d__0), _o);
                exchange<'T'>(nullptr, out, mu);
                wait<'T'>(out + (mu - 1) * _dof * _bs);
            }
        }
        template<bool>
        void apply(const K* const in, K* const out, const unsigned short& mu = 1, K* = nullptr) const {
            Solver<K>::solve(in, out, mu);
        }
        template<bool = false>
        bool start(const K* const, K* const, const unsigned short& mu = 1) const {
            if(_off) {
                unsigned short k = 1;
                const std::string prefix = OptionsPrefix::prefix();
                const Option& opt = *Option::get();
                if(opt.any_of(prefix + "krylov_method", { 4, 5 }) && !opt.val<unsigned short>(prefix + "recycle_same_system"))
                    k = std::max(opt.val<int>(prefix + "recycle", 1), 1);
                _o = new K[mu * k * _off * _bs]();
                return true;
            }
            else
                return false;
        }
        void end(const bool free) const {
            if(free)
                delete [] _o;
        }
        static constexpr underlying_type<K>* getScaling() { return nullptr; }
    private:
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
                        MPI_Irecv(_buff[i], p.second.size() * _bs, Wrapper<K>::mpi_type(), p.first, 10, _communicator, _rq + i);
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
                        MPI_Isend(_buff[i], _recv[i].second.size() * _bs, Wrapper<K>::mpi_type(), _recv[i].first, 10, _communicator, _rq + i);
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
};
} // HPDDM
#endif // _HPDDM_INEXACT_COARSE_OPERATOR_
