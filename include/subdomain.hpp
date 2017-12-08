/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
              Frédéric Nataf <nataf@ann.jussieu.fr>
        Date: 2012-12-15

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

#ifndef _HPDDM_SUBDOMAIN_
#define _HPDDM_SUBDOMAIN_

#include <unordered_set>
#include <map>

namespace HPDDM {
/* Class: Subdomain
 *
 *  A class for handling all communications and computations between subdomains.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Subdomain : public OptionsPrefix {
    protected:
        /* Variable: a
         *  Local matrix. */
        MatrixCSR<K>*                _a;
        /* Variable : buff
         *  Array used as the receiving and receiving buffer for point-to-point communications with neighboring subdomains. */
        K**                       _buff;
        /* Variable: map */
        vectorNeighbor             _map;
        /* Variable: rq
         *  Array of MPI requests to check completion of the MPI transfers with neighboring subdomains. */
        MPI_Request*                _rq;
        /* Variable: communicator
         *  MPI communicator of the subdomain. */
        MPI_Comm          _communicator;
        /* Variable: dof
         *  Number of degrees of freedom in the current subdomain. */
        int                        _dof;
    public:
        Subdomain() : OptionsPrefix(), _a(), _buff(), _map(), _rq(), _dof() { }
        Subdomain(const Subdomain<K>& s) {
            _a = nullptr;
            _map = s._map;
            _communicator = s._communicator;
            _dof = s._dof;
            _rq = new MPI_Request[2 * _map.size()];
            _buff = new K*[2 * _map.size()];
        }
        ~Subdomain() {
            delete [] _rq;
            vectorNeighbor().swap(_map);
            delete [] _buff;
            destroyMatrix(nullptr);
        }
        /* Function: getCommunicator
         *  Returns a reference to <Subdomain::communicator>. */
        const MPI_Comm& getCommunicator() const { return _communicator; }
        /* Function: getMap
         *  Returns a reference to <Subdomain::map>. */
        const vectorNeighbor& getMap() const { return _map; }
        /* Function: exchange
         *
         *  Exchanges and reduces values of duplicated unknowns.
         *
         * Parameter:
         *    in             - Input vector. */
        void exchange(K* const in, const unsigned short& mu = 1) const {
            for(unsigned short nu = 0; nu < mu; ++nu) {
                for(unsigned short i = 0, size = _map.size(); i < size; ++i) {
                    MPI_Irecv(_buff[i], _map[i].second.size(), Wrapper<K>::mpi_type(), _map[i].first, 0, _communicator, _rq + i);
                    Wrapper<K>::gthr(_map[i].second.size(), in + nu * _dof, _buff[size + i], _map[i].second.data());
                    MPI_Isend(_buff[size + i], _map[i].second.size(), Wrapper<K>::mpi_type(), _map[i].first, 0, _communicator, _rq + size + i);
                }
                for(unsigned short i = 0; i < _map.size(); ++i) {
                    int index;
                    MPI_Waitany(_map.size(), _rq, &index, MPI_STATUS_IGNORE);
                    for(unsigned int j = 0; j < _map[index].second.size(); ++j)
                        in[_map[index].second[j] + nu * _dof] += _buff[index][j];
                }
                MPI_Waitall(_map.size(), _rq + _map.size(), MPI_STATUSES_IGNORE);
            }
        }
        template<class T, typename std::enable_if<!HPDDM::Wrapper<K>::is_complex && HPDDM::Wrapper<T>::is_complex && std::is_same<K, underlying_type<T>>::value>::type* = nullptr>
        void exchange(T* const in, const unsigned short& mu = 1) const {
            for(unsigned short nu = 0; nu < mu; ++nu) {
                K* transpose = reinterpret_cast<K*>(in + nu * _dof);
                Wrapper<K>::template cycle<'T'>(_dof, 2, transpose, 1);
                exchange(transpose, 2);
                Wrapper<K>::template cycle<'T'>(2, _dof, transpose, 1);
            }
        }
        /* Function: recvBuffer
         *
         *  Exchanges values of duplicated unknowns.
         *
         * Parameter:
         *    in             - Input vector. */
        void recvBuffer(const K* const in) const {
            for(unsigned short i = 0, size = _map.size(); i < size; ++i) {
                MPI_Irecv(_buff[i], _map[i].second.size(), Wrapper<K>::mpi_type(), _map[i].first, 0, _communicator, _rq + i);
                Wrapper<K>::gthr(_map[i].second.size(), in, _buff[size + i], _map[i].second.data());
                MPI_Isend(_buff[size + i], _map[i].second.size(), Wrapper<K>::mpi_type(), _map[i].first, 0, _communicator, _rq + size + i);
            }
            MPI_Waitall(2 * _map.size(), _rq, MPI_STATUSES_IGNORE);
        }
        /* Function: initialize
         *
         *  Initializes all buffers for point-to-point communications and set internal pointers to user-defined values.
         *
         * Parameters:
         *    a              - Local matrix.
         *    o              - Indices of neighboring subdomains.
         *    r              - Local-to-neighbor mappings.
         *    comm           - MPI communicator of the domain decomposition. */
        template<class Neighbor, class Mapping>
        void initialize(MatrixCSR<K>* const& a, const Neighbor& o, const Mapping& r, MPI_Comm* const& comm = nullptr) {
            if(comm)
                _communicator = *comm;
            else
                _communicator = MPI_COMM_WORLD;
            _a = a;
            if(_a)
                _dof = _a->_n;
            std::vector<unsigned short> sortable;
            for(const auto& i : o)
                sortable.emplace_back(i);
            _map.reserve(sortable.size());
            std::vector<unsigned short> idx(sortable.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](const unsigned short& lhs, const unsigned short& rhs) { return sortable[lhs] < sortable[rhs]; });
            unsigned short j = 0;
            for(const unsigned short& i : idx) {
                if(r[idx[j]].size() > 0) {
                    _map.emplace_back(sortable[i], typename decltype(_map)::value_type::second_type());
                    _map.back().second.reserve(r[idx[j]].size());
                    for(int k = 0; k < r[idx[j]].size(); ++k)
                        _map.back().second.emplace_back(r[idx[j]][k]);
                }
                ++j;
            }
            _rq = new MPI_Request[2 * _map.size()];
            _buff = new K*[2 * _map.size()]();
        }
        void initialize(MatrixCSR<K>* const& a, const int neighbors, const int* const list, const int* const sizes, const int* const* const connectivity, MPI_Comm* const& comm = nullptr) {
            if(comm)
                _communicator = *comm;
            else
                _communicator = MPI_COMM_WORLD;
            _a = a;
            if(_a)
                _dof = _a->_n;
            _map.reserve(neighbors);
            std::vector<unsigned short> idx(neighbors);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](const unsigned short& lhs, const unsigned short& rhs) { return list[lhs] < list[rhs]; });
            unsigned short j = 0;
            while(j < neighbors) {
                if(sizes[idx[j]] > 0) {
                    _map.emplace_back(list[idx[j]], typename decltype(_map)::value_type::second_type());
                    _map.back().second.reserve(sizes[idx[j]]);
                    for(int k = 0; k < sizes[idx[j]]; ++k)
                        _map.back().second.emplace_back(connectivity[idx[j]][k]);
                }
                ++j;
            }
            _rq = new MPI_Request[2 * _map.size()];
            _buff = new K*[2 * _map.size()];
        }
        bool setBuffer(K* wk = nullptr, const int& space = 0) const {
            unsigned int n = 0;
            for(const auto& i : _map)
                n += i.second.size();
            if(n == 0)
                return false;
            bool allocate;
            if(2 * n <= space && wk) {
                *_buff = wk;
                allocate = false;
            }
            else {
                *_buff = new K[2 * n];
                allocate = true;
            }
            _buff[_map.size()] = *_buff + n;
            n = 0;
            for(unsigned short i = 1, size = _map.size(); i < size; ++i) {
                n += _map[i - 1].second.size();
                _buff[i] = *_buff + n;
                _buff[size + i] = _buff[size] + n;
            }
            return allocate;
        }
        void clearBuffer(const bool free = true) const {
            if(free && !_map.empty())
                delete [] *_buff;
        }
        void end(const bool free = true) const { clearBuffer(free); }
        /* Function: initialize(dummy)
         *  Dummy function for masters excluded from the domain decomposition. */
        void initialize(MPI_Comm* const& comm = nullptr) {
            if(comm)
                _communicator = *comm;
            else
                _communicator = MPI_COMM_WORLD;
        }
        /* Function: exclusion
         *
         *  Checks whether <Subdomain::communicator> has been built by excluding some processes.
         *
         * Parameter:
         *    comm          - Reference MPI communicator. */
        bool exclusion(const MPI_Comm& comm) const {
            int result;
            MPI_Comm_compare(_communicator, comm, &result);
            return result != MPI_CONGRUENT && result != MPI_IDENT;
        }
        K boundaryCond(const unsigned int i) const {
            const int shift = _a->_ia[0];
            unsigned int stop;
            if(_a->_ia[i] != _a->_ia[i + 1]) {
                if(!_a->_sym) {
                    int* const bound = std::upper_bound(_a->_ja + _a->_ia[i] - shift, _a->_ja + _a->_ia[i + 1] - shift, i + shift);
                    stop = std::distance(_a->_ja, bound);
                }
                else
                    stop = _a->_ia[i + 1] - shift;
                if((_a->_sym || stop < _a->_ia[i + 1] - shift) && _a->_ja[std::max(1U, stop) - 1] == i + shift && std::abs(_a->_a[stop - 1]) < HPDDM_EPS * HPDDM_PEN)
                    for(unsigned int j = _a->_ia[i] - shift; j < stop; ++j) {
                        if(i != _a->_ja[j] - shift && std::abs(_a->_a[j]) > HPDDM_EPS)
                            return K();
                        else if(i == _a->_ja[j] - shift && std::abs(_a->_a[j] - K(1.0)) > HPDDM_EPS)
                            return K();
                    }
            }
            else
                return K();
            return _a->_a[stop - 1];
        }
        std::unordered_map<unsigned int, K> boundaryConditions() const {
            std::unordered_map<unsigned int, K> map;
            map.reserve(_dof / 1000);
            for(unsigned int i = 0; i < Subdomain<K>::_dof; ++i) {
                const K boundary = boundaryCond(i);
                if(std::abs(boundary) > HPDDM_EPS)
                    map[i] = boundary;
            }
            return map;
        }
        /* Function: getDof
         *  Returns the value of <Subdomain::dof>. */
        int getDof() const { return _dof; }
        /* Function: getMatrix
         *  Returns a pointer to <Subdomain::a>. */
        const MatrixCSR<K>* getMatrix() const { return _a; }
        /* Function: setMatrix
         *  Sets the pointer <Subdomain::a>. */
        bool setMatrix(MatrixCSR<K>* const& a) {
            bool ret = !(_a && a && _a->_n == a->_n && _a->_m == a->_m && _a->_nnz == a->_nnz);
            if(!_dof && a)
                _dof = a->_n;
            delete _a;
            _a = a;
            return ret;
        }
        /* Function: destroyMatrix
         *  Destroys the pointer <Subdomain::a> using a custom deallocator. */
        void destroyMatrix(void (*dtor)(void*)) {
            if(_a) {
                int rankWorld;
                MPI_Finalized(&rankWorld);
                if(!rankWorld) {
                    MPI_Comm_rank(_communicator, &rankWorld);
                    const std::string prefix = OptionsPrefix::prefix();
                    const Option& opt = *Option::get();
                    std::string filename = opt.prefix(prefix + "dump_matrices", true);
                    if(filename.size() == 0)
                        filename = opt.prefix(prefix + "dump_matrix_" + to_string(rankWorld), true);
                    if(filename.size() != 0) {
                        int sizeWorld;
                        MPI_Comm_size(_communicator, &sizeWorld);
                        std::ofstream output { filename + "_" + to_string(rankWorld) + "_" + to_string(sizeWorld) + ".txt" };
                        output << *_a;
                    }
                }
                if(dtor)
                    _a->destroy(dtor);
                delete _a;
                _a = nullptr;
            }
        }
        /* Function: getRq
         *  Returns a pointer to <Subdomain::rq>. */
        MPI_Request* getRq() const { return _rq; }
        /* Function: getBuffer
         *  Returns a pointer to <Subdomain::buff>. */
        K** getBuffer() const { return _buff; }
        template<bool excluded>
        void scatter(const K* const, K*&, const unsigned short, unsigned short&, const MPI_Comm&) const { }
        void statistics() const {
            unsigned long long local[4], global[4];
            unsigned short* const table = new unsigned short[_dof];
            int n;
            MPI_Comm_rank(_communicator, &n);
            std::fill_n(table, _dof, n);
            for(const auto& i : _map)
                for(const int& j : i.second)
                    table[j] = i.first;
            local[0] = local[2] = 0;
            for(unsigned int i = 0; i < _dof; ++i)
                if(table[i] <= n) {
                    ++local[0];
                    local[2] += _a->_ia[i + 1] - _a->_ia[i];
                }
            if(_a->_sym) {
                local[2] *= 2;
                local[2] -= local[0];
            }
            if(_dof == _a->_n)
                local[1] = _dof - local[0];
            else {
                local[1] = local[0];
                local[0] = _a->_n - local[1];
            }
            delete [] table;
            local[3] = _map.size();
            MPI_Allreduce(local, global, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, _communicator);
            if(n == 0) {
                std::vector<std::string> v;
                v.reserve(7);
                const std::string& prefix = OptionsPrefix::prefix();
                v.emplace_back(" ┌");
                v.emplace_back(" │ HPDDM statistics" + std::string(prefix.size() ? " for operator \""  + prefix + "\"": "") + ":");
                v.emplace_back(" │  " + to_string(global[0]) + " unknown" + (global[0] > 1 ? "s" : ""));
                v.emplace_back(" │  " + to_string(global[1]) + " interprocess unknown" + (global[1] > 1 ? "s" : ""));
                std::stringstream ss;
                ss << std::fixed << std::setprecision(1) << global[2] / static_cast<float>(global[0]) << " nonzero entr" << (global[2] / static_cast<float>(global[0]) > 1 ? "ies" : "y") << " per unknown";
                v.emplace_back(" │  " + ss.str());
                ss.clear();
                ss.str(std::string());
                MPI_Comm_size(_communicator, &n);
                ss << std::fixed << std::setprecision(1) << global[3] / static_cast<float>(n) << " neighboring process" << (global[3] / static_cast<float>(n) > 1.0 ? "es" : "") << " (average)";
                v.emplace_back(" │  " + ss.str());
                v.emplace_back(" └");
                std::vector<std::string>::const_iterator max = std::max_element(v.cbegin(), v.cend(), [](const std::string& lhs, const std::string& rhs) { return lhs.size() < rhs.size(); });
                Option::output(v, max->size());
            }
        }
        /* Function: interaction
         *
         *  Builds a vector of matrices to store interactions with neighboring subdomains.
         *
         * Template Parameters:
         *    N              - 0- or 1-based indexing of the input matrix.
         *    sorted         - True if the column indices of each matrix in the vector must be sorted.
         *    scale          - True if the matrices must be scaled by the neighboring partition of unity.
         *
         * Parameters:
         *    v              - Output vector.
         *    scaling        - Local partition of unity.
         *    pt             - Pointer to a <MatrixCSR>. */
        template<char N, bool sorted = true, bool scale = false>
        void interaction(std::vector<const MatrixCSR<K>*>& v, const underlying_type<K>* const scaling = nullptr, const MatrixCSR<K>* const pt = nullptr) const {
            const MatrixCSR<K>& ref = pt ? *pt : *_a;
            if(ref._n != _dof || ref._m != _dof)
                std::cerr << "Problem with the input matrix" << std::endl;
            std::vector<std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>> send(_map.size());
            unsigned int* sendSize = new unsigned int[4 * _map.size()];
            unsigned int* recvSize = sendSize + 2 * _map.size();
            for(unsigned short k = 0; k < _map.size(); ++k)
                MPI_Irecv(recvSize + 2 * k, 2, MPI_UNSIGNED, _map[k].first, 10, _communicator, _rq + k);
            for(unsigned short k = 0; k < _map.size(); ++k) {
                std::vector<std::pair<unsigned int, unsigned int>> fast;
                fast.reserve(_map[k].second.size());
                for(unsigned int i = 0; i < _map[k].second.size(); ++i)
                    fast.emplace_back(_map[k].second[i], i);
                std::sort(fast.begin(), fast.end());
                std::vector<std::pair<unsigned int, unsigned int>>::const_iterator itRow = fast.cbegin();
                for(unsigned int i = 0; i < _dof; ++i) {
                    std::vector<std::pair<unsigned int, unsigned int>>::const_iterator begin = fast.cbegin();
                    if(itRow != fast.cend() && itRow->first == i) {
                        if(ref._sym) {
                            for(unsigned int j = ref._ia[i]; j < ref._ia[i + 1]; ++j) {
                                std::vector<std::pair<unsigned int, unsigned int>>::const_iterator it = std::lower_bound(begin, fast.cend(), std::make_pair(ref._ja[j], 0), [](const std::pair<unsigned int, unsigned int>& lhs, const std::pair<unsigned int, unsigned int>& rhs) { return lhs.first < rhs.first; });
                                if(it == fast.cend() || ref._ja[j] < it->first)
                                    send[k].emplace_back(itRow->second, ref._ja[j], j - (N == 'F'));
                                else
                                    begin = it;
                            }
                        }
                        ++itRow;
                    }
                    else {
                        for(unsigned int j = ref._ia[i]; j < ref._ia[i + 1]; ++j) {
                            std::vector<std::pair<unsigned int, unsigned int>>::const_iterator it = std::lower_bound(begin, fast.cend(), std::make_pair(ref._ja[j], 0), [](const std::pair<unsigned int, unsigned int>& lhs, const std::pair<unsigned int, unsigned int>& rhs) { return lhs.first < rhs.first; });
                            if(it != fast.cend() && !(ref._ja[j] < it->first)) {
                                send[k].emplace_back(it->second, i, j - (N == 'F'));
                                begin = it;
                            }
                        }
                    }
                }
                std::sort(send[k].begin(), send[k].end());
                sendSize[2 * k] = send[k].empty() ? 0 : 1;
                for(unsigned int i = 1; i < send[k].size(); ++i)
                    if(std::get<0>(send[k][i]) != std::get<0>(send[k][i - 1]))
                        ++sendSize[2 * k];
                sendSize[2 * k + 1] = 1 + ((sendSize[2 * k] * sizeof(unsigned short)
                                    + (sendSize[2 * k] + 1 + send[k].size()) * sizeof(unsigned int) - 1) / sizeof(K))
                                    + send[k].size();
                MPI_Isend(sendSize + 2 * k, 2, MPI_UNSIGNED, _map[k].first, 10, _communicator, _rq + _map.size() + k);
            }
            MPI_Waitall(2 * _map.size(), _rq, MPI_STATUSES_IGNORE);
            unsigned short maxRecv = 0;
            unsigned int accumulate = 0;
            while(maxRecv < _map.size()) {
                unsigned int next = accumulate + recvSize[2 * maxRecv + 1];
                if(next < std::distance(_buff[0], _buff[2 * _map.size() - 1]) + _map.back().second.size())
                    accumulate = next;
                else
                    break;
                ++maxRecv;
            }
            unsigned short maxSend = 0;
            if(maxRecv == _map.size())
                while(maxSend < _map.size()) {
                    unsigned int next = accumulate + sendSize[2 * maxSend + 1];
                    if(next < std::distance(_buff[0], _buff[2 * _map.size() - 1]) + _map.back().second.size())
                        accumulate = next;
                    else
                        break;
                    ++maxSend;
                }
            std::vector<K*> rbuff;
            rbuff.reserve(_map.size());
            accumulate = 0;
            for(unsigned int k = 0; k < _map.size(); ++k) {
                if(k < maxRecv) {
                    rbuff.emplace_back(_buff[0] + accumulate);
                    accumulate += recvSize[2 * k + 1];
                }
                else if(k == maxRecv) {
                    unsigned int accumulateSend = 0;
                    for(unsigned short j = k; j < _map.size(); ++j)
                        accumulateSend += recvSize[2 * j + 1];
                    accumulate += accumulateSend;
                    for(unsigned short j = 0; j < _map.size(); ++j)
                        accumulateSend += sendSize[2 * j + 1];
                    rbuff.emplace_back(new K[accumulateSend]);
                }
                else
                    rbuff.emplace_back(rbuff.back() + recvSize[2 * k - 1]);
                MPI_Irecv(rbuff[k], recvSize[2 * k + 1], Wrapper<K>::mpi_type(), _map[k].first, 100, _communicator, _rq + k);
            }
            std::vector<K*> sbuff;
            sbuff.reserve(_map.size());
            for(unsigned short k = 0; k < _map.size(); ++k) {
                if(maxRecv < _map.size()) {
                    if(k == 0)
                        sbuff.emplace_back(rbuff.back() + recvSize[2 * _map.size() - 1]);
                    else
                        sbuff.emplace_back(sbuff.back() + sendSize[2 * k - 1]);
                }
                else if(k < maxSend) {
                    sbuff.emplace_back(rbuff[0] + accumulate);
                    accumulate += sendSize[2 * k + 1];
                }
                else if(k == maxSend) {
                    unsigned int accumulateTotal = accumulate;
                    for(unsigned int j = k; j < _map.size(); ++j)
                        accumulateTotal += sendSize[2 * j + 1];
                    sbuff.emplace_back(new K[accumulateTotal]);
                }
                else
                    sbuff.emplace_back(sbuff.back() + sendSize[2 * k - 1]);
                unsigned short* ia = reinterpret_cast<unsigned short*>(sbuff[k]);
                unsigned int* mapRow = reinterpret_cast<unsigned int*>(ia + sendSize[2 * k]);
                unsigned int* ja = mapRow + sendSize[2 * k] + 1;
                K* a = sbuff[k] + sendSize[2 * k + 1] - send[k].size();
                *mapRow++ = send[k].size();
                if(!send[k].empty()) {
                    *mapRow++ = std::get<0>(send[k][0]);
                    unsigned int prev = 0;
                    for(unsigned int i = 0; i < send[k].size(); ++i) {
                        if(i > 0 && std::get<0>(send[k][i]) != std::get<0>(send[k][i - 1])) {
                            *ia++ = i - prev;
                            prev = i;
                            *mapRow++ = std::get<0>(send[k][i]);
                        }
                        *ja++ = std::get<1>(send[k][i]);
                        *a = ref._a[std::get<2>(send[k][i])];
                        if(scale && scaling)
                            *a *= scaling[ref._ja[std::get<2>(send[k][i])]];
                        ++a;
                    }
                    *ia++ = send[k].size() - prev;
                }
                MPI_Isend(sbuff[k], sendSize[2 * k + 1], Wrapper<K>::mpi_type(), _map[k].first, 100, _communicator, _rq + _map.size() + k);
            }
            decltype(send)().swap(send);
            if(!v.empty())
                v.clear();
            v.reserve(_map.size());
            for(unsigned short k = 0; k < _map.size(); ++k) {
                int index;
                MPI_Waitany(_map.size(), _rq, &index, MPI_STATUS_IGNORE);
                unsigned short* ia = reinterpret_cast<unsigned short*>(rbuff[index]);
                unsigned int* mapRow = reinterpret_cast<unsigned int*>(ia + recvSize[2 * index]);
                unsigned int* ja = mapRow + recvSize[2 * index] + 1;
                const unsigned int nnz = *mapRow++;
                K* a = rbuff[index] + recvSize[2 * index + 1] - nnz;
                std::unordered_map<unsigned int, unsigned int> mapCol;
                mapCol.reserve(nnz);
                for(unsigned int i = 0, j = 0; i < nnz; ++i)
                    if(mapCol.count(ja[i]) == 0)
                        mapCol[ja[i]] = j++;
                MatrixCSR<K>* AIJ = new MatrixCSR<K>(_dof, mapCol.size(), nnz, false);
                v.emplace_back(AIJ);
                std::fill_n(AIJ->_ia, AIJ->_n + 1, 0);
                for(unsigned int i = 0; i < recvSize[2 * index]; ++i) {
#if 0
                    if(std::abs(scaling[_map[index].second[mapRow[i]]]) > HPDDM_EPS)
                        std::cerr << "Problem with the partition of unity: (std::abs(d[" << _map[index].second[mapRow[i]] << "]) = " << std::abs(scaling[_map[index].second[mapRow[i]]]) << ") > HPDDM_EPS" << std::endl;
#endif
                    AIJ->_ia[_map[index].second[mapRow[i]] + 1] = ia[i];
                }
                std::partial_sum(AIJ->_ia, AIJ->_ia + AIJ->_n + 1, AIJ->_ia);
                if(AIJ->_ia[AIJ->_n] != nnz)
                    std::cerr << "Problem with the received CSR: (AIJ->_ia[" << AIJ->_n << "] = " << AIJ->_ia[AIJ->_n] << ") != " << nnz << std::endl;
                for(unsigned int i = 0, m = 0; i < recvSize[2 * index]; ++i) {
                    unsigned int pos = AIJ->_ia[_map[index].second[mapRow[i]]];
                    for(unsigned short j = 0; j < ia[i]; ++j, ++m) {
                        AIJ->_ja[pos + j] = mapCol[ja[m]];
                        AIJ->_a[pos + j] = a[m];
                    }
                    if(sorted) {
                        std::vector<unsigned short> idx;
                        idx.reserve(ia[i]);
                        for(unsigned short j = 0; j < ia[i]; ++j)
                            idx.emplace_back(j);
                        std::sort(idx.begin(), idx.end(), [&](const unsigned short& lhs, const unsigned short& rhs) { return AIJ->_ja[pos + lhs] < AIJ->_ja[pos + rhs]; });
                        reorder(idx, AIJ->_ja + pos, AIJ->_a + pos);
                    }
                }
            }
            MPI_Waitall(_map.size(), _rq + _map.size(), MPI_STATUSES_IGNORE);
            delete [] sendSize;
            if(maxRecv < _map.size())
                delete [] rbuff[maxRecv];
            else if(maxSend < _map.size())
                delete [] sbuff[maxSend];
        }
        /* Function: globalMapping
         *
         *  Computes a global numbering of all unknowns.
         *
         * Template Parameters:
         *    N              - 0- or 1-based indexing.
         *    It             - Random iterator.
         *
         * Parameters:
         *    first         - First element of the list of local unknowns with the global numbering.
         *    last          - Last element of the list of local unknowns with the global numbering.
         *    start         - Lowest global number of the local unknowns.
         *    end           - Highest global number of the local unknowns.
         *    global        - Global number of unknowns.
         *    d             - Local partition of unity (optional). */
        template<char N, class It>
        void globalMapping(It first, It last, unsigned int& start, unsigned int& end, unsigned int& global, const underlying_type<K>* const d = nullptr, const unsigned int* const list = nullptr) const {
            unsigned int between = 0;
            int rankWorld, sizeWorld;
            MPI_Comm_rank(_communicator, &rankWorld);
            MPI_Comm_size(_communicator, &sizeWorld);
            std::map<unsigned int, unsigned int> r;
            if(list) {
                for(unsigned int i = 0; i < Subdomain<K>::_dof; ++i)
                    if(list[i] > 0)
                        r[list[i]] = i;
            }
            if(sizeWorld > 1) {
                setBuffer();
                for(unsigned short i = 0; i < _map.size() && _map[i].first < rankWorld; ++i)
                    ++between;
                unsigned int size = 1 + ((2 * (std::distance(_buff[0], _buff[_map.size()]) + 1) * sizeof(unsigned int) - 1) / sizeof(K));
                unsigned int* rbuff = (size < std::distance(_buff[0], _buff[2 * _map.size() - 1]) + _map.back().second.size() ? reinterpret_cast<unsigned int*>(_buff[0]) : new unsigned int[2 * (std::distance(_buff[0], _buff[_map.size()]) + 1)]);
                unsigned int* sbuff = rbuff + std::distance(_buff[0], _buff[_map.size()]) + 1;
                size = 0;
                MPI_Request* rq = new MPI_Request[2];

                for(unsigned short i = 0; i < between; ++i) {
                    MPI_Irecv(rbuff + size, _map[i].second.size() + (_map[i].first == rankWorld - 1), MPI_UNSIGNED, _map[i].first, 10, _communicator, _rq + i);
                    size += _map[i].second.size();
                }

                if(rankWorld && ((between && _map[between - 1].first != rankWorld - 1) || !between))
                    MPI_Irecv(rbuff + size, 1, MPI_UNSIGNED, rankWorld - 1, 10, _communicator, rq);
                else
                    rq[0] = MPI_REQUEST_NULL;

                ++size;
                for(unsigned short i = between; i < _map.size(); ++i) {
                    MPI_Irecv(rbuff + size, _map[i].second.size(), MPI_UNSIGNED, _map[i].first, 10, _communicator, _rq + _map.size() + i);
                    size += _map[i].second.size();
                }

                unsigned int begining;
                std::fill(first, last, std::numeric_limits<unsigned int>::max());
                if(rankWorld == 0) {
                    begining = (N == 'F');
                    start = begining;
                    if(!list) {
                        for(unsigned int i = 0; i < std::distance(first, last); ++i)
                            if(!d || d[i] > 0.1)
                                *(first + i) = begining++;
                    }
                    else {
                        for(const std::pair<unsigned int, unsigned int>& p : r) {
                            if(!d || d[p.second] > 0.1) {
                                *(first + p.second) = begining++;
                            }
                        }
                    }
                    end = begining;
                }
                size = 0;
                for(unsigned short i = 0; i < between; ++i) {
                    MPI_Wait(_rq + i, MPI_STATUS_IGNORE);
                    for(unsigned int j = 0; j < _map[i].second.size(); ++j)
                        first[_map[i].second[j]] = rbuff[size + j];
                    size += _map[i].second.size();
                }
                if(rankWorld) {
                    if((between && _map[between - 1].first != rankWorld - 1) || !between)
                        MPI_Wait(rq, MPI_STATUS_IGNORE);
                    begining = rbuff[size];
                    start = begining;
                    if(!list) {
                        for(unsigned int i = 0; i < std::distance(first, last); ++i)
                            if((!d || d[i] > 0.1) && *(first + i) == std::numeric_limits<unsigned int>::max())
                                *(first + i) = begining++;
                    }
                    else {
                        for(const std::pair<unsigned int, unsigned int>& p : r) {
                            if((!d || d[p.second] > 0.1) && *(first + p.second) == std::numeric_limits<unsigned int>::max()) {
                                *(first + p.second) = begining++;
                            }
                        }
                    }
                    end = begining;
                }
                size = 0;
                if(rankWorld != sizeWorld - 1) {
                    if(between < _map.size()) {
                        if(_map[between].first == rankWorld + 1) {
                            sbuff[_map[between].second.size()] = begining;
                            rq[1] = MPI_REQUEST_NULL;
                        }
                        else
                            MPI_Isend(&begining, 1, MPI_UNSIGNED, rankWorld + 1, 10, _communicator, rq + 1);
                        for(unsigned short i = between; i < _map.size(); ++i) {
                            for(unsigned short j = 0; j < _map[i].second.size(); ++j)
                                sbuff[size + j] = *(first + _map[i].second[j]);
                            MPI_Isend(sbuff + size, _map[i].second.size() + (_map[i].first == rankWorld + 1), MPI_UNSIGNED, _map[i].first, 10, _communicator, _rq + i);
                            size += _map[i].second.size() + (_map[i].first == rankWorld + 1);
                        }
                    }
                    else
                        MPI_Isend(&begining, 1, MPI_UNSIGNED, rankWorld + 1, 10, _communicator, rq + 1);
                }
                else
                    rq[1] = MPI_REQUEST_NULL;
                unsigned int stop = 0;
                for(unsigned short i = 0; i < between; ++i) {
                    for(unsigned short j = 0; j < _map[i].second.size(); ++j)
                        sbuff[size + j] = *(first + _map[i].second[j]);
                    MPI_Isend(sbuff + size, _map[i].second.size(), MPI_UNSIGNED, _map[i].first, 10, _communicator, _rq + _map.size() + i);
                    size += _map[i].second.size();
                    stop += _map[i].second.size();
                }
                ++stop;
                for(unsigned short i = between; i < _map.size(); ++i) {
                    MPI_Wait(_rq + _map.size() + i, MPI_STATUS_IGNORE);
                    for(unsigned int j = 0; j < _map[i].second.size(); ++j)
                        first[_map[i].second[j]] = rbuff[stop + j];
                    stop += _map[i].second.size();
                }
                MPI_Waitall(_map.size(), _rq + between, MPI_STATUSES_IGNORE);
                MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
                delete [] rq;
                if(rbuff != reinterpret_cast<unsigned int*>(_buff[0]))
                    delete [] rbuff;
                global = end - (N == 'F');
                MPI_Bcast(&global, 1, MPI_UNSIGNED, sizeWorld - 1, _communicator);
                clearBuffer();
            }
            else {
                if(!list) {
                    std::iota(first, last, static_cast<unsigned int>(N == 'F'));
                    end = std::distance(first, last);
                }
                else {
                    unsigned int j = (N == 'F');
                    for(const std::pair<unsigned int, unsigned int>& p : r) {
                        *(first + p.second) = j++;
                    }
                    end = r.size();
                }
                start = (N == 'F');
                global = end - start;
            }
        }
        /* Function: distributedCSR
         *  Assembles a distributed matrix that can be used by a backend such as PETSc.
         *
         * See also: <Subdomain::globalMapping>. */
        template<class T = K>
        static bool distributedCSR(unsigned int* num, unsigned int first, unsigned int last, int*& ia, int*& ja, T*& c, const MatrixCSR<K>* const& A) {
            if(first != 0 || last != A->_n) {
                std::vector<std::pair<unsigned int, unsigned int>> s;
                s.reserve(A->_n);
                for(unsigned int i = 0; i < A->_n; ++i)
                    s.emplace_back(num[i], i);
                std::sort(s.begin(), s.end());
                std::vector<std::pair<unsigned int, unsigned int>>::iterator begin = std::lower_bound(s.begin(), s.end(), std::make_pair(first, static_cast<unsigned int>(0)));
                std::vector<std::pair<unsigned int, unsigned int>>::iterator end = std::upper_bound(begin, s.end(), std::make_pair(last, static_cast<unsigned int>(0)));
                unsigned int dof = std::distance(begin, end);
                std::vector<std::pair<unsigned int, T>> tmp;
                tmp.reserve(A->_nnz);
                if(!ia)
                    ia = new int[dof + 1];
                ia[0] = 0;
                for(std::vector<std::pair<unsigned int, unsigned int>>::iterator it = begin; it != end; it++) {
                    for(unsigned int j = A->_ia[it->second]; j < A->_ia[it->second + 1]; ++j)
                        tmp.emplace_back(num[A->_ja[j]], std::is_same<K, T>::value ? A->_a[j] : j);
                    std::sort(tmp.begin() + ia[std::distance(begin, it)], tmp.end(), [](const std::pair<unsigned int, T>& lhs, const std::pair<unsigned int, T>& rhs) { return lhs.first < rhs.first; });
                    ia[std::distance(begin, it) + 1] = tmp.size();
                }
                unsigned int nnz = tmp.size();
                if(!c)
                    c  = reinterpret_cast<T*>(new K[nnz * (1 + (sizeof(K) - 1) / sizeof(T))]);
                if(!ja)
                    ja = new int[nnz];
                for(unsigned int i = 0; i < tmp.size(); ++i) {
                    ja[i] = tmp[i].first;
                    c[i] = tmp[i].second;
                }
                return true;
            }
            else {
                if(std::is_same<K, T>::value)
                    c  = reinterpret_cast<T*>(A->_a);
                else
                    c = nullptr;
                ia = A->_ia;
                ja = A->_ja;
                return false;
            }
        }
        /* Function: distributedVec
         *  Assembles a distributed vector that can by used by a backend such as PETSc.
         *
         * See also: <Subdomain::globalMapping>. */
        template<bool V, class T = K>
        static void distributedVec(unsigned int* num, unsigned int first, unsigned int last, T* const& in, T*& out, const unsigned int n, const unsigned short bs = 1) {
            if(first != 0 || last != n) {
                if(!out) {
                    unsigned int dof = 0;
                    for(unsigned int i = 0; i < n; ++i) {
                        if(num[i] >= first && num[i] < last)
                            ++dof;
                    }
                    out = new T[dof];
                }
                for(unsigned int i = 0; i < n; ++i) {
                    if(num[i] >= first && num[i] < last) {
                        if(!V)
                            std::copy_n(in + bs * i, bs, out + bs * (num[i] - first));
                        else
                            std::copy_n(out + bs * (num[i] - first), bs, in + bs * i);
                    }
                }
            }
            else {
                if(!V)
                    std::copy_n(in, bs * n, out);
                else
                    std::copy_n(out, bs * n, in);
            }
        }
};

template<bool excluded, class Operator, class K, typename std::enable_if<hpddm_method_id<Operator>::value>::type*>
inline void IterativeMethod::preprocess(const Operator& A, const K* const b, K*& sb, K* const x, K*& sx, const int& mu, unsigned short& k, const MPI_Comm& comm) {
    int size;
    if(excluded) {
        MPI_Comm_size(comm, &size);
        int master;
        MPI_Comm_size(A.getCommunicator(), &master);
        size -= master;
    }
    else
        MPI_Comm_size(A.getCommunicator(), &size);
    if(k < 2 || size == 1 || mu > 1) {
        sx = x;
        sb = const_cast<K*>(b);
        k = 1;
    }
    else {
        int rank;
        MPI_Comm_rank(A.getCommunicator(), &rank);
        k = std::min(k, static_cast<unsigned short>(size));
        unsigned int* local = new unsigned int[2 * k];
        unsigned int* global = local + k;
        const int n = excluded ? 0 : A.getDof();
        const vectorNeighbor& map = A.getMap();
        int displ = 0;
        std::unordered_set<int> redundant;
        unsigned short j = std::min(k - 1, rank / (size / k));
        std::function<void ()> check_size = [&] {
            std::fill_n(local, k, 0U);
            if(!excluded)
                for(unsigned int i = 0; i < n; ++i) {
                    if(std::abs(b[i]) > HPDDM_EPS && redundant.count(i) == 0) {
                        if(++local[j] > k)
                            break;
                    }
                }
            MPI_Allreduce(local, global, k, MPI_UNSIGNED, MPI_SUM, comm);
        };
        if(!excluded)
            for(const auto& i : map) {
                displ += i.second.size();
                for(const int& k : i.second)
                    redundant.emplace(k);
            }
        check_size();
        {
            unsigned int max = 0;
            for(unsigned short nu = 0; nu < k && max < k; ++nu)
                max += global[nu];
            if(max < k) {
                k = std::max(1U, max);
                global = local + k;
                j = std::min(k - 1, rank / (size / k));
                check_size();
            }
        }
        if(k > 1) {
            unsigned short* idx = new unsigned short[n + displ];
            unsigned short* buff = idx + n;
            sx = new K[k * n]();
            sb = new K[k * n]();
            const int div = size / k;
            if(!excluded) {
                std::fill_n(idx, n + displ, std::min(rank / div, k - 1) + 1);
                displ = 0;
                for(unsigned short i = 0; i < map.size(); ++i) {
                    if(rank < map[i].first)
                        std::fill_n(buff + displ, map[i].second.size(), std::min(map[i].first / div, k - 1) + 1);
                    displ += map[i].second.size();
                }
                displ = 0;
                for(unsigned short i = 0; i < map.size(); ++i) {
                    Wrapper<unsigned short>::sctr(map[i].second.size(), buff + displ, map[i].second.data(), idx);
                    displ += map[i].second.size();
                }
                for(unsigned int i = 0; i < n; ++i) {
                    sx[i + (idx[i] - 1) * n] = x[i];
                    sb[i + (idx[i] - 1) * n] = b[i];
                }
            }
            std::function<K* (K*, unsigned int*, unsigned int*, int)> lambda = [&](K* sb, unsigned int* local, unsigned int* swap, int n) {
                for(unsigned int i = 0; i < n; ++i)
                    if(std::abs(sb[std::distance(local, swap) * n + i]) > HPDDM_EPS && redundant.count(i) == 0)
                        return sb + std::distance(local, swap) * n + i;
                return sb + (std::distance(local, swap) + 1) * n;
            };
            IterativeMethod::equilibrate<excluded>(n, sb, sx, lambda, local, k, rank, size / k, comm);
            delete [] idx;
        }
        else {
            sx = x;
            sb = const_cast<K*>(b);
        }
        delete [] local;
    }
    checkEnlargedMethod(A.prefix(), k);
}

template<class K>
struct hpddm_method_id<Subdomain<K>> { static constexpr char value = 10; };
} // HPDDM
#endif // _HPDDM_SUBDOMAIN_
