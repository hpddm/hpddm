/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <jolivet@ann.jussieu.fr>
        Date: 2012-12-15

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

#ifndef _SUBDOMAIN_
#define _SUBDOMAIN_

namespace HPDDM {
/* Class: Subdomain
 *
 *  A class for handling all communications and computations between subdomains.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Subdomain {
    protected:
        /* Variable: rbuff
         *  Vector used as the receiving buffer for point-to-point communications with neighboring subdomains. */
        std::vector<K*>          _rbuff;
        /* Variable: sbuff
         *  Vector used as the sending buffer for point-to-point communications with neighboring subdomains. */
        std::vector<K*>          _sbuff;
        /* Variable: rq
         *  Array of MPI requests to check completion of the MPI transfers with neighboring subdomains. */
        MPI_Request*                _rq;
        /* Variable: communicator
         *  MPI communicator of the subdomain. */
        MPI_Comm          _communicator;
        /* Variable: dof
         *  Number of degrees of freedom in the current subdomain. */
        int                        _dof;
        /* Variable: map */
        vectorNeighbor             _map;
        /* Variable: a
         *  Local matrix. */
        MatrixCSR<K>*                _a;
    public:
        Subdomain() : _rq(), _map(), _a() { }
        ~Subdomain() {
            delete _a;
            delete [] _rq;
            if(!_sbuff.empty())
                delete [] _sbuff[0];
        }
        /* Function: getCommunicator
         *  Returns a reference to <Subdomain::communicator>. */
        inline const MPI_Comm& getCommunicator() const { return _communicator; }
        /* Function: getMap
         *  Returns a reference to <Subdomain::map>. */
        inline const vectorNeighbor& getMap() const { return _map; }
        /* Function: exchange
         *
         *  Exchanges and reduces values of duplicated unknowns.
         *
         * Parameter:
         *    in             - Input vector. */
        inline void exchange(K* const in) const {
            for(unsigned short i = 0; i < _map.size(); ++i) {
                MPI_Irecv(_rbuff[i], _map[i].second.size(), Wrapper<K>::mpi_type(), _map[i].first, 0, _communicator, _rq + i);
                Wrapper<K>::gthr(_map[i].second.size(), in, _sbuff[i], _map[i].second.data());
                MPI_Isend(_sbuff[i], _map[i].second.size(), Wrapper<K>::mpi_type(), _map[i].first, 0, _communicator, _rq + _map.size() + i);
            }
            for(unsigned short i = 0; i < _map.size(); ++i) {
                int index;
                MPI_Waitany(_map.size(), _rq, &index, MPI_STATUS_IGNORE);
                for(unsigned int j = 0; j < _map[index].second.size(); ++j)
                    in[_map[index].second[j]] += _rbuff[index][j];
            }
            MPI_Waitall(_map.size(), _rq + _map.size(), MPI_STATUSES_IGNORE);
        }
        /* Function: recvBuffer
         *
         *  Exchanges values of duplicated unknowns.
         *
         * Parameter:
         *    in             - Input vector. */
        inline void recvBuffer(const K* const in) const {
            for(unsigned short i = 0; i < _map.size(); ++i) {
                MPI_Irecv(_rbuff[i], _map[i].second.size(), Wrapper<K>::mpi_type(), _map[i].first, 0, _communicator, _rq + i);
                Wrapper<K>::gthr(_map[i].second.size(), in, _sbuff[i], _map[i].second.data());
                MPI_Isend(_sbuff[i], _map[i].second.size(), Wrapper<K>::mpi_type(), _map[i].first, 0, _communicator, _rq + _map.size() + i);
            }
            MPI_Waitall(2 * _map.size(), _rq, MPI_STATUSES_IGNORE);
}
        /* Function: initialize
         *
         *  Initializes all buffers for point-to-point communications and set internal pointers to user-defined values.
         *
         * Template Parameters:
         *    It             - Forward iterator.
         *    Container      - Class of the local-to-neighbor mappings.
         *
         * Parameters:
         *    a              - Local matrix.
         *    begin          - Iterator pointing to the first element of the container of indices of neighboring subdomains.
         *    end            - Iterator pointing to the past-the-end element of the container of indices of neighboring subdomains.
         *    comm           - MPI communicator of the domain decomposition. */
        template<class It, class Container>
        inline void initialize(MatrixCSR<K>* const& a, const It& begin, const It& end, std::vector<Container*> const& r, MPI_Comm* const& comm = nullptr) {
            if(comm)
                _communicator = *comm;
            else
                _communicator = MPI_COMM_WORLD;
            _a = a;
            _dof = _a->_n;
            if(begin != end) {
                _map.resize(std::distance(begin, end));
                unsigned int size = 0;
                It it = begin;
                do {
                    pairNeighbor& ref = _map[std::distance(begin, it)];
                    ref.first = *it;
                    const Container& in = *r[std::distance(begin, it)];
                    ref.second.reserve(in.size());
                    for(int j = 0; j < in.size(); ++j)
                        ref.second.emplace_back(in[j]);
                    size += ref.second.size();
                } while(++it != end);
                _rq = new MPI_Request[2 * _map.size()];
                _rbuff.reserve(_map.size());
                _sbuff.reserve(_map.size());
                if(size) {
                    K* sbuff = new K[2 * size];
                    K* rbuff = sbuff + size;
                    size = 0;
                    for(unsigned short i = 0; i < _map.size(); ++i) {
                        _sbuff.emplace_back(sbuff + size);
                        _rbuff.emplace_back(rbuff + size);
                        size +=_map[i].second.size();
                    }
                }
            }
        /* Function: initialize(dummy)
         *  Dummy function for masters excluded from the domain decomposition. */
        }
        inline void initialize(MPI_Comm* const& comm = nullptr) {
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
        inline bool exclusion(const MPI_Comm& comm) const {
            int result;
            MPI_Comm_compare(_communicator, comm, &result);
            return result != MPI_CONGRUENT;
        }
        /* Function: getDof
         *  Returns the value of <Subdomain::dof>. */
        inline int getDof() const { return _dof; }
        /* Function: getMatrix
         *  Returns a constant pointer to <Subdomain::a>. */
        inline const MatrixCSR<K>* getMatrix() const { return _a; }
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
        inline void globalMapping(It first, It last, unsigned int& start, unsigned int& end, unsigned int& global, const double* const d = nullptr) const {
            unsigned int between = 0;
            int rankWorld, sizeWorld;
            MPI_Comm_rank(_communicator, &rankWorld);
            MPI_Comm_size(_communicator, &sizeWorld);
            if(sizeWorld > 1) {
                for(unsigned short i = 0; i < _map.size() && _map[i].first < rankWorld; ++i)
                    ++between;
                unsigned int* sbuff = new unsigned int[2 * (std::distance(_sbuff.front(), _sbuff.back()) + _map.back().second.size() + 1)];
                unsigned int* rbuff = sbuff + std::distance(_sbuff.front(), _sbuff.back()) + _map.back().second.size() + 1;
                unsigned int size = 0;
                MPI_Request* rq = new MPI_Request[2 * _map.size() + 2];

                for(unsigned short i = 0; i < between; ++i) {
                    MPI_Irecv(rbuff + size, _map[i].second.size() + (_map[i].first == rankWorld - 1), MPI_UNSIGNED, _map[i].first, 10, _communicator, rq + i);
                    size += _map[i].second.size();
                }

                if(rankWorld && ((between && _map[between - 1].first != rankWorld - 1) || !between))
                    MPI_Irecv(rbuff + size, 1, MPI_UNSIGNED, rankWorld - 1, 10, _communicator, rq + _map.size());
                else
                    rq[_map.size()] = MPI_REQUEST_NULL;

                ++size;
                for(unsigned short i = between; i < _map.size(); ++i) {
                    MPI_Irecv(rbuff + size, _map[i].second.size(), MPI_UNSIGNED, _map[i].first, 10, _communicator, rq + _map.size() + 2 + i);
                    size += _map[i].second.size();
                }

                unsigned int begining;
                std::fill(first, last, std::numeric_limits<unsigned int>::max());
                if(rankWorld == 0) {
                    begining = static_cast<unsigned int>(N == 'F');
                    start = begining;
                    for(unsigned int i = 0; i < std::distance(first, last); ++i)
                        if(!d || d[i] > 0.1)
                            *(first + i) = begining++;
                    end = begining;
                }
                size = 0;
                for(unsigned short i = 0; i < between; ++i) {
                    MPI_Wait(rq + i, MPI_STATUS_IGNORE);
                    for(unsigned int j = 0; j < _map[i].second.size(); ++j)
                        first[_map[i].second[j]] = rbuff[size + j];
                    size += _map[i].second.size();
                }
                if(rankWorld) {
                    if((between && _map[between - 1].first != rankWorld - 1) || !between)
                        MPI_Wait(rq + _map.size(), MPI_STATUS_IGNORE);
                    begining = rbuff[size];
                    start = begining;
                    for(unsigned int i = 0; i < std::distance(first, last); ++i)
                        if((!d || d[i] > 0.1) && *(first + i) == std::numeric_limits<unsigned int>::max())
                            *(first + i) = begining++;
                    end = begining;
                }
                size = 0;
                if(rankWorld != sizeWorld - 1) {
                    if(_map[between].first == rankWorld + 1)
                        sbuff[_map[between].second.size()] = begining;
                    for(unsigned short i = between; i < _map.size(); ++i) {
                        for(unsigned short j = 0; j < _map[i].second.size(); ++j)
                            sbuff[size + j] = *(first + _map[i].second[j]);
                        MPI_Isend(sbuff + size, _map[i].second.size() + (_map[i].first == rankWorld + 1), MPI_UNSIGNED, _map[i].first, 10, _communicator, rq + i);
                        size += _map[i].second.size() + (_map[i].first == rankWorld + 1);
                    }
                    if(_map[between].first != rankWorld + 1)
                        MPI_Isend(&begining, 1, MPI_UNSIGNED, rankWorld + 1, 10, _communicator, rq + _map.size() + 1);
                    else
                        rq[_map.size() + 1] = MPI_REQUEST_NULL;
                }
                else
                    rq[_map.size() + 1] = MPI_REQUEST_NULL;
                unsigned int stop = 0;
                for(unsigned short i = 0; i < between; ++i) {
                    for(unsigned short j = 0; j < _map[i].second.size(); ++j)
                        sbuff[size + j] = *(first + _map[i].second[j]);
                    MPI_Isend(sbuff + size, _map[i].second.size(), MPI_UNSIGNED, _map[i].first, 10, _communicator, rq + _map.size() + 2 + i);
                    size += _map[i].second.size();
                    stop += _map[i].second.size();
                }
                ++stop;
                for(unsigned short i = between; i < _map.size(); ++i) {
                    MPI_Wait(rq + _map.size() + 2 + i, MPI_STATUS_IGNORE);
                    for(unsigned int j = 0; j < _map[i].second.size(); ++j)
                        first[_map[i].second[j]] = rbuff[stop + j];
                    stop += _map[i].second.size();
                }
                MPI_Waitall(2 * _map.size() + 2, rq, MPI_STATUSES_IGNORE);
                delete [] sbuff;
                delete [] rq;
                global = end - (N == 'F');
                MPI_Bcast(&global, 1, MPI_UNSIGNED, sizeWorld - 1, _communicator);
            }
            else {
                std::iota(first, last, static_cast<unsigned int>(N == 'F'));
                start = (N == 'F');
                end = std::distance(first, last);
                global = end - start;
            }
        }
        /* Function: distributedCSR
         *  Assembles a distributed matrix that can by used by a backend such as PETSc.
         *
         * See also: <Subdomain::globalMapping>. */
        inline bool distributedCSR(unsigned int* num, unsigned int first, unsigned int last, int*& ia, int*& ja, K*& c, const MatrixCSR<K>* const& A) const {
            if(first != 0 || last != A->_n) {
                unsigned int nnz = 0;
                unsigned int dof = 0;
                for(unsigned int i = 0; i < A->_n; ++i) {
                    if(num[i] >= first && num[i] < last)
                        ++dof;
                }
                std::vector<std::vector<std::pair<unsigned int, K>>> tmp(dof);
                for(unsigned int i = 0; i < A->_n; ++i) {
                    if(num[i] >= first && num[i] < last)
                            tmp[num[i] - first].reserve(A->_ia[i + 1] - A->_ia[i]);
                }
                nnz = 0;
                for(unsigned int i = 0; i < A->_n; ++i) {
                    if(num[i] >= first && num[i] < last) {
                        for(unsigned int j = A->_ia[i]; j < A->_ia[i + 1]; ++j)
                            tmp[num[i] - first].emplace_back(num[A->_ja[j]], A->_a[j]);
                    }
                }
                for(const std::vector<std::pair<unsigned int, K>>& v : tmp)
                    nnz += v.size();
                if(!c)
                    c  = new K[nnz];
                if(!ia)
                    ia = new int[dof + 1];
                if(!ja)
                    ja = new int[nnz];
                ia[0] = 0;
                nnz = 0;
                for(unsigned int i = 0; i < dof; ++i) {
                    std::sort(tmp[i].begin(), tmp[i].end());
                    for(std::pair<unsigned int, K>& p : tmp[i]) {
                        ja[nnz] = p.first;
                        c[nnz++] = p.second;
                    }
                    ia[i + 1] = nnz;
                }
                return true;
            }
            else {
                c  = A->_a;
                ia = A->_ia;
                ja = A->_ja;
                return false;
            }
        }
        /* Function: distributedVec
         *  Assembles a distributed vector that can by used by a backend such as PETSc.
         *
         * See also: <Subdomain::globalMapping>. */
        template<bool T>
        inline void distributedVec(unsigned int* num, unsigned int first, unsigned int last, K* const& in, K*& out, unsigned int n) const {
            if(first != 0 || last != n) {
                unsigned int dof = 0;
                for(unsigned int i = 0; i < n; ++i) {
                    if(num[i] >= first && num[i] < last)
                        ++dof;
                }
                if(!out)
                    out = new K[dof];
                for(unsigned int i = 0; i < n; ++i) {
                    if(num[i] >= first && num[i] < last) {
                        if(!T)
                            out[num[i] - first] = in[i];
                        else
                            in[i] = out[num[i] - first];
                    }
                }
            }
            else {
                if(!T)
                    std::copy(in, in + n, out);
                else
                    std::copy(out, out + n, in);
            }
        }
};
} // HPDDM
#endif // _SUBDOMAIN_
