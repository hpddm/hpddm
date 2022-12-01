/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
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
class Subdomain
#if !HPDDM_PETSC
                : public OptionsPrefix<K>
#endif
                                          {
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
        void dtor() {
            clearBuffer();
            delete [] _rq;
            _rq = nullptr;
            vectorNeighbor().swap(_map);
            delete [] _buff;
            _buff = nullptr;
#ifndef PETSCHPDDM_H
            destroyMatrix(nullptr);
#endif
        }
    public:
        Subdomain() :
#if !HPDDM_PETSC
                      OptionsPrefix<K>(),
#endif
                                          _a(), _buff(), _map(), _rq(), _dof() { }
        Subdomain(const Subdomain<K>& s) :
#if !HPDDM_PETSC
                                           OptionsPrefix<K>(),
#endif
                                                               _a(), _buff(new K*[2 * s._map.size()]), _map(s._map), _rq(new MPI_Request[2 * s._map.size()]), _communicator(s._communicator), _dof(s._dof) { }
        ~Subdomain() {
            dtor();
        }
        typedef int integer_type;
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
                    ignore(MPI_Waitany(_map.size(), _rq, &index, MPI_STATUS_IGNORE));
                    for(unsigned int j = 0; j < _map[index].second.size(); ++j)
                        in[_map[index].second[j] + nu * _dof] += _buff[index][j];
                }
                ignore(MPI_Waitall(_map.size(), _rq + _map.size(), MPI_STATUSES_IGNORE));
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
        void initialize(MatrixCSR<K>* const& a, const Neighbor& o, const Mapping& r, MPI_Comm* const& comm = nullptr, const MatrixCSR<void>* const& restriction = nullptr) {
            if(comm)
                _communicator = *comm;
            else
                _communicator = MPI_COMM_WORLD;
            unsigned int* perm = nullptr;
#ifndef PETSCHPDDM_H
            if(a && restriction) {
                perm = new unsigned int[a->_n]();
                for(unsigned int i = 0; i < restriction->_n; ++i)
                    perm[restriction->_ja[i]] = i + 1;
                _a = new MatrixCSR<K>(a, restriction, perm);
            }
            else
                _a = a;
            if(_a)
                _dof = _a->_n;
#else
            ignore(restriction);
#endif
            std::vector<unsigned short> sortable;
            std::copy(o.begin(), o.end(), std::back_inserter(sortable));
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
            if(perm) {
                const int size = _map.size();
                MPI_Request* rq = new MPI_Request[2 * size];
                unsigned int space = 0;
                for(unsigned short i = 0; i < size; ++i)
                    space += _map[i].second.size();
                unsigned char* send = new unsigned char[2 * space];
                unsigned char* recv = send + space;
                space = 0;
                for(unsigned short i = 0; i < size; ++i) {
                    MPI_Irecv(recv, _map[i].second.size(), MPI_UNSIGNED_CHAR, _map[i].first, 100, _communicator, rq + i);
                    if(a->_n)
                        for(unsigned int j = 0; j < _map[i].second.size(); ++j)
                            send[j] = (perm[_map[i].second[j]] > 0 ? 'a' : 'b');
                    else
                        std::fill_n(send, _map[i].second.size(), 'b');
                    MPI_Isend(send, _map[i].second.size(), MPI_UNSIGNED_CHAR, _map[i].first, 100, _communicator, rq + size + i);
                    send += _map[i].second.size();
                    recv += _map[i].second.size();
                    space += _map[i].second.size();
                }
                MPI_Waitall(2 * size, rq, MPI_STATUSES_IGNORE);
                vectorNeighbor map;
                map.reserve(size);
                send -= space;
                recv -= space;
                for(unsigned short i = 0; i < size; ++i) {
                    std::pair<unsigned short, typename decltype(_map)::value_type::second_type> c(_map[i].first, typename decltype(_map)::value_type::second_type());
                    for(unsigned int j = 0; j < _map[i].second.size(); ++j) {
                        if(recv[j] == 'a' && send[j] == 'a')
                            c.second.emplace_back(perm[_map[i].second[j]] - 1);
                    }
                    if(!c.second.empty())
                        map.emplace_back(c);
                    send += _map[i].second.size();
                    recv += _map[i].second.size();
                }
                send -= space;
                delete [] send;
                delete [] rq;
                _map = map;
            }
            delete [] perm;
            _rq = new MPI_Request[2 * _map.size()];
            _buff = new K*[2 * _map.size()]();
        }
#ifndef PETSCHPDDM_H
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
            _buff = new K*[2 * _map.size()]();
        }
#endif
        bool setBuffer(K* wk = nullptr, const int& space = 0) const {
            int n = std::accumulate(_map.cbegin(), _map.cend(), 0, [](unsigned int init, const pairNeighbor& i) { return init + i.second.size(); });
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
            if(free && !_map.empty() && _buff) {
                delete [] *_buff;
                *_buff = nullptr;
            }
        }
        void end(const bool free = true) const { clearBuffer(free); }
        /* Function: initialize(dummy)
         *  Dummy function for main processes excluded from the domain decomposition. */
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
#ifndef PETSCHPDDM_H
        K boundaryCond(const unsigned int i) const {
            if(_a->_ia) {
                const int shift = _a->_ia[0];
                unsigned int stop;
                if(_a->_ia[i] != _a->_ia[i + 1]) {
                    if(!_a->_sym)
                        stop = std::distance(_a->_ja, std::upper_bound(_a->_ja + _a->_ia[i] - shift, _a->_ja + _a->_ia[i + 1] - shift, i + shift));
                    else
                        stop = _a->_ia[i + 1] - shift;
                    if((_a->_sym || stop < _a->_ia[i + 1] - shift || _a->_ja[_a->_ia[i + 1] - shift - 1] == i + shift) && _a->_ja[std::max(1U, stop) - 1] == i + shift && std::abs(_a->_a[stop - 1]) < HPDDM_EPS * HPDDM_PEN)
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
            else
                return K();
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
#endif
        /* Function: getDof
         *  Returns the value of <Subdomain::dof>. */
        constexpr int getDof() const { return _dof; }
        /* Function: setDof
         *  Sets the value of <Subdomain::dof>. */
        void setDof(int dof) {
            if(!_dof
#ifndef PETSCHPDDM_H
                    && !_a
#endif
                          )
                _dof = dof;
        }
#ifndef PETSCHPDDM_H
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
                int isFinalized;
                MPI_Finalized(&isFinalized);
                if(!isFinalized) {
                    int rankWorld;
                    MPI_Comm_rank(_communicator, &rankWorld);
#if !HPDDM_PETSC
                    const std::string prefix = OptionsPrefix<K>::prefix();
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
#endif
                }
                if(dtor)
                    _a->destroy(dtor);
                delete _a;
                _a = nullptr;
            }
        }
#endif
        /* Function: getRq
         *  Returns a pointer to <Subdomain::rq>. */
        MPI_Request* getRq() const { return _rq; }
        /* Function: getBuffer
         *  Returns a pointer to <Subdomain::buff>. */
        K** getBuffer() const { return _buff; }
        template<bool excluded>
        void scatter(const K* const, K*&, const unsigned short, unsigned short&, const MPI_Comm&) const { }
        void statistics() const {
#if !HPDDM_PETSC
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
                const std::string& prefix = OptionsPrefix<K>::prefix();
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
#endif
        }
#ifndef PETSCHPDDM_H
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
        template<char N, class It, class T>
        void globalMapping(It first, It last, T& start, T& end, long long& global, const underlying_type<K>* const d = nullptr, const T* const list = nullptr) const {
            static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Unsupported input type");
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
                T between = 0;
                for(unsigned short i = 0; i < _map.size() && _map[i].first < rankWorld; ++i)
                    ++between;
                T* local = new T[sizeWorld];
                local[rankWorld] = (list ? r.size() : std::distance(first, last));
                std::unordered_set<unsigned int> removed;
                removed.reserve(local[rankWorld]);
                for(unsigned short i = 0; i < _map.size(); ++i)
                    for(const int& j : _map[i].second) {
                        if(d && d[j] < HPDDM_EPS && removed.find(j) == removed.cend() && (!list || list[j] > 0)) {
                            --local[rankWorld];
                            removed.insert(j);
                        }
                    }
                MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, local, 1, Wrapper<T>::mpi_type(), _communicator);
                start = std::accumulate(local, local + rankWorld, static_cast<long long>(N == 'F'));
                end = start + local[rankWorld];
                if(start > end)
                    std::cerr << "Probable integer overflow on process #" << rankWorld << ": " << start << " > " << end << std::endl;
                global = std::accumulate(local + rankWorld + 1, local + sizeWorld, static_cast<long long>(end));
                delete [] local;
                T beginning = start;
                std::fill(first, last, std::numeric_limits<T>::max());
                if(!list) {
                    for(unsigned int i = 0; i < std::distance(first, last); ++i)
                        if(removed.find(i) == removed.cend())
                            *(first + i) = beginning++;
                }
                else {
                    for(const std::pair<const unsigned int, unsigned int>& p : r)
                        if(removed.find(p.second) == removed.cend())
                            *(first + p.second) = beginning++;
                }
                if(!_map.empty()) {
                    for(unsigned short i = 0; i < _map.size(); ++i)
                        MPI_Irecv(static_cast<void*>(_buff[i]), _map[i].second.size(), Wrapper<T>::mpi_type(), _map[i].first, 10, _communicator, _rq + i);
                    for(unsigned short i = 0; i < _map.size(); ++i) {
                        T* sbuff = reinterpret_cast<T*>(_buff[_map.size() + i]);
                        for(unsigned int j = 0; j < _map[i].second.size(); ++j)
                            sbuff[j] = *(first + _map[i].second[j]);
                        MPI_Isend(static_cast<void*>(sbuff), _map[i].second.size(), Wrapper<T>::mpi_type(), _map[i].first, 10, _communicator, _rq + _map.size() + i);
                    }
                    for(unsigned short i = 0; i < _map.size(); ++i) {
                        int index;
                        MPI_Waitany(_map.size(), _rq, &index, MPI_STATUS_IGNORE);
                        T* rbuff = reinterpret_cast<T*>(_buff[index]);
                        for(const int& j : _map[index].second) {
                            if(first[j] == std::numeric_limits<T>::max())
                                first[j] = *rbuff;
                            ++rbuff;
                        }
                    }
                }
                MPI_Waitall(_map.size(), _rq + _map.size(), MPI_STATUSES_IGNORE);
                clearBuffer();
            }
            else {
                if(!list) {
                    std::iota(first, last, static_cast<T>(N == 'F'));
                    end = std::distance(first, last);
                }
                else {
                    T j = (N == 'F');
                    for(const std::pair<const unsigned int, unsigned int>& p : r) {
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
        template<class I, class T = K>
        static bool distributedCSR(const I* const row, I first, I last, I*& ia, I*& ja, T*& c, const MatrixCSR<K>* const& A, const I* col = nullptr) {
            std::vector<std::pair<int, int>>* transpose = nullptr;
            if(A->_sym) {
                if(col || !std::is_same<K, T>::value)
                    std::cerr << "Not implemented" << std::endl;
                transpose = new std::vector<std::pair<int, int>>[A->_n]();
                for(int i = 0; i < A->_n; ++i)
                    for(int j = A->_ia[i] - (HPDDM_NUMBERING == 'F'); j < A->_ia[i + 1] - (HPDDM_NUMBERING == 'F'); ++j)
                        transpose[A->_ja[j] - (HPDDM_NUMBERING == 'F')].emplace_back(i, j);
                for(int i = 0; i < A->_n; ++i)
                    std::sort(transpose[i].begin(), transpose[i].end());
            }
            if(first != 0 || last != A->_n || col) {
                if(!col)
                    col = row;
                std::vector<std::pair<I, I>> s;
                s.reserve(A->_n);
                for(unsigned int i = 0; i < A->_n; ++i)
                    s.emplace_back(row[i], i);
                std::sort(s.begin(), s.end());
                typename std::vector<std::pair<I, I>>::iterator begin = std::lower_bound(s.begin(), s.end(), std::make_pair(first, static_cast<I>(0)));
                typename std::vector<std::pair<I, I>>::iterator end = std::upper_bound(begin, s.end(), std::make_pair(last, static_cast<I>(0)));
                unsigned int dof = std::distance(begin, end);
                std::vector<std::pair<I, T>> tmp;
                tmp.reserve(A->_nnz);
                if(!ia)
                    ia = new I[dof + 1];
                ia[0] = 0;
                for(typename std::vector<std::pair<I, I>>::iterator it = begin; it != end; ++it) {
                    for(unsigned int j = A->_ia[it->second]; j < A->_ia[it->second + 1]; ++j)
                        tmp.emplace_back(col[A->_ja[j]], std::is_same<K, T>::value ? A->_a[j] : j);
                    if(A->_sym) {
                        for(unsigned int j = 0; j < transpose[it->second].size(); ++j) {
                            if(transpose[it->second][j].first != it->second)
                                tmp.emplace_back(col[transpose[it->second][j].first], A->_a[transpose[it->second][j].second]);
                        }
                    }
                    std::sort(tmp.begin() + ia[std::distance(begin, it)], tmp.end(), [](const std::pair<I, T>& lhs, const std::pair<I, T>& rhs) { return lhs.first < rhs.first; });
                    if(A->_sym) {
                        const unsigned int row = std::distance(begin, it);
                        tmp.erase(std::remove_if(tmp.begin() + ia[row], tmp.end(), [&row](const std::pair<I, T>& x) { return x.first < row; }), tmp.end());
                    }
                    ia[std::distance(begin, it) + 1] = tmp.size();
                }
                unsigned int nnz = tmp.size();
                if(!c)
                    c = reinterpret_cast<T*>(new K[nnz * (1 + (sizeof(K) - 1) / sizeof(T))]);
                if(!ja)
                    ja = new I[nnz];
                for(unsigned int i = 0; i < tmp.size(); ++i) {
                    ja[i] = tmp[i].first;
                    c[i] = tmp[i].second;
                }
            }
            else {
                if(!A->_sym) {
                    if(std::is_same<decltype(A->_ia), I>::value) {
                        ia = reinterpret_cast<I*>(A->_ia);
                        ja = reinterpret_cast<I*>(A->_ja);
                        if(std::is_same<K, T>::value)
                            c = reinterpret_cast<T*>(A->_a);
                        else
                            c = nullptr;
                        return false;
                    }
                    else {
                        if(!std::is_same<K, T>::value)
                            std::cerr << "Not implemented" << std::endl;
                        if(!ia)
                            ia = new I[A->_n + 1];
                        std::copy_n(A->_ia, A->_n + 1, ia);
                        if(!ja)
                            ja = new I[A->_nnz];
                        std::copy_n(A->_ja, A->_nnz, ja);
                        if(!c)
                            c = new T[A->_nnz];
                        std::copy_n(A->_a, A->_nnz, c);
                        return true;
                    }
                }
                else {
                    if(!ia)
                        ia = new I[A->_n + 1];
                    if(!ja)
                        ja = new I[A->_nnz];
                    if(!c)
                        c = new T[A->_nnz];
                    ia[0] = 0;
                    for(int i = 0; i < A->_n; ++i) {
                        for(int j = 0; j < transpose[i].size(); ++j) {
                            c[ia[i] + j] = A->_a[transpose[i][j].second];
                            ja[ia[i] + j] = transpose[i][j].first;
                        }
                        ia[i + 1] = ia[i] + transpose[i].size();
                    }
                }
            }
            delete [] transpose;
            return true;
        }
        /* Function: distributedVec
         *  Assembles a distributed vector that can by used by a backend such as PETSc.
         *
         * See also: <Subdomain::globalMapping>. */
        template<bool V, class I, class T = K>
        static void distributedVec(const I* const num, I first, I last, T* const& in, T*& out, const I n, const unsigned short bs = 1) {
            if(first != 0 || last != n) {
                if(first != last) {
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
            }
            else {
                if(!V)
                    std::copy_n(in, bs * n, out);
                else
                    std::copy_n(out, bs * n, in);
            }
        }
#endif
};

#if !HPDDM_PETSC || defined(_KSPIMPL_H)
template<bool excluded, class Operator, class K, typename std::enable_if<hpddm_method_id<Operator>::value != 0>::type*>
inline void IterativeMethod::preprocess(const Operator& A, const K* const b, K*& sb, K* const x, K*& sx, const int& mu, unsigned short& k, const MPI_Comm& comm) {
    int size;
    if(excluded) {
        MPI_Comm_size(comm, &size);
        int main;
        MPI_Comm_size(A.getCommunicator(), &main);
        size -= main;
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
        int accumulate = 0;
        std::unordered_set<int> redundant;
        unsigned short j = std::min(k - 1, rank / (size / k));
        std::function<void ()> check_size = [&] {
            std::fill_n(local, k, 0U);
            if(!excluded)
                for(unsigned int i = 0; i < n; ++i) {
                    if(HPDDM::abs(b[i]) > HPDDM_EPS && redundant.count(i) == 0) {
                        if(++local[j] > k)
                            break;
                    }
                }
            MPI_Allreduce(local, global, k, MPI_UNSIGNED, MPI_SUM, comm);
        };
        if(!excluded)
            for(const auto& i : map) {
                accumulate += i.second.size();
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
            unsigned short* idx = new unsigned short[n + accumulate];
            unsigned short* buff = idx + n;
            sx = new K[k * n]();
            sb = new K[k * n]();
            const int div = size / k;
            if(!excluded) {
                std::fill_n(idx, n + accumulate, std::min(rank / div, k - 1) + 1);
                accumulate = 0;
                for(unsigned short i = 0; i < map.size(); ++i) {
                    if(rank < map[i].first)
                        std::fill_n(buff + accumulate, map[i].second.size(), std::min(static_cast<int>(map[i].first / div), static_cast<int>(k - 1)) + 1);
                    accumulate += map[i].second.size();
                }
                accumulate = 0;
                for(unsigned short i = 0; i < map.size(); ++i) {
                    Wrapper<unsigned short>::sctr(map[i].second.size(), buff + accumulate, map[i].second.data(), idx);
                    accumulate += map[i].second.size();
                }
                for(unsigned int i = 0; i < n; ++i) {
                    sx[i + (idx[i] - 1) * n] = x[i];
                    sb[i + (idx[i] - 1) * n] = b[i];
                }
            }
            std::function<K* (K*, unsigned int*, unsigned int*, int)> lambda = [&](K* sb, unsigned int* local, unsigned int* swap, int n) {
                for(unsigned int i = 0; i < n; ++i)
                    if(HPDDM::abs(sb[std::distance(local, swap) * n + i]) > HPDDM_EPS && redundant.count(i) == 0)
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
    checkEnlargedMethod(A, k);
}
#endif

template<class K>
struct hpddm_method_id<Subdomain<K>> { static constexpr char value = 10; };
} // HPDDM
#endif // _HPDDM_SUBDOMAIN_
