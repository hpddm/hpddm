/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <jolivet@ann.jussieu.fr>
        Date: 2012-10-04

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

#ifndef _DISTRIBUTED_MATRIX_
#define _DISTRIBUTED_MATRIX_

#include <map>

namespace HPDDM {
/* Class: DMatrix
 *  A class for handling all communications and computations involving a distributed matrix. */
class DMatrix {
    public:
        /* Enum: Distribution
         *
         *  Defines the distribution of both right-hand sides and solution vectors.
         *
         * NON_DISTRIBUTED         - Neither are distributed, both are centralized on the root of <DMatrix::communicator>.
         * DISTRIBUTED_SOL         - Right-hand sides are centralized, while solution vectors are distributed on <DMatrix::communicator>.
         * DISTRIBUTED_SOL_AND_RHS - Both are distributed on <DMatrix::communicator>. */
        enum Distribution : char {
            NON_DISTRIBUTED, DISTRIBUTED_SOL, DISTRIBUTED_SOL_AND_RHS = 3
        };
    protected:
        /* Typedef: pair_type
         *  std::pair of unsigned integers. */
        typedef std::pair<unsigned int, unsigned int>   pair_type;
        /* Typedef: map_type
         *
         *  std::map of std::vector<T> indexed by unsigned short integers.
         *
         * Template Parameter:
         *    T              - Class. */
        template<class T>
        using map_type = std::map<unsigned short, std::vector<T>>;
        /* Variable: mapRecv
         *  Values that have to be received to match the distribution of the direct solver and of the user. */
        map_type<pair_type>*        _mapRecv;
        /* Variable: mapSend
         *  Values that have to be sent to match the distribution of the direct solver and of the user. */
        map_type<pair_type>*        _mapSend;
        /* Variable: mapOwn
         *  Values that have to remain on this process to match the distribution of the direct solver and of the user. */
        std::vector<pair_type>*      _mapOwn;
        /* Variable: ldistribution
         *  User distribution. */
        int*                  _ldistribution;
        /* Variable: idistribution */
        int*                  _idistribution;
        /* Variable: gatherCounts */
        int*                   _gatherCounts;
        /* Variable: gatherSplitCounts */
        int*              _gatherSplitCounts;
        /* Variable: displs */
        int*                         _displs;
        /* Variable: displsSplit */
        int*                    _displsSplit;
        /* Variable: communicator
         *  MPI communicator on which the matrix is distributed. */
        MPI_Comm               _communicator;
        /* Variable: n
         *  Size of the coarse operator. */
        int                               _n;
        /* Variable: rank
         *  Rank of the current master process in <Coarse operator::communicator>. */
        int                            _rank;
        /* Variable: distribution
         *  <Distribution> used for right-hand sides and solution vectors. */
        Distribution           _distribution;
        /* Function: initializeMap
         *
         *  Initializes <Coarse operator::mapRecv>, <Coarse operator::mapSend>, and <Coarse operator::mapOwn>.
         *
         * Template Parameter:
         *    isRHS          - True if this function is first called to redistribute a right-hand side, false otherwise.
         *
         * Parameters:
         *    info           - Local dimension returned by the direct solver.
         *    isol_loc       - Numbering of the direct solver.
         *    sol_loc        - Vector following the numbering of the direct solver.
         *    sol            - Vector following the numbering of the user. */
        template<bool isRHS, class K>
        inline void initializeMap(const int& info, const int* const isol_loc, K* const sol_loc, K* const sol) {
            int size;
            MPI_Comm_size(_communicator, &size);
            int* lsol_loc_glob = new int[size];
            lsol_loc_glob[_rank] = info;
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, lsol_loc_glob, 1, MPI_INT, _communicator);
            int* isol_loc_glob = new int[_n];
            int* disp_lsol_loc_glob = new int[size];
            disp_lsol_loc_glob[0] = 0;
            std::partial_sum(lsol_loc_glob, lsol_loc_glob + size - 1, disp_lsol_loc_glob + 1);
            MPI_Allgatherv(const_cast<int*>(isol_loc), info, MPI_INT, isol_loc_glob, lsol_loc_glob, disp_lsol_loc_glob, MPI_INT, _communicator);
            delete [] disp_lsol_loc_glob;
            std::vector<std::pair<unsigned short, unsigned int>> mapping(_n);
            std::vector<std::pair<unsigned short, unsigned int>> mapping_user(_n);
            _mapRecv = new map_type<pair_type>;
            _mapSend = new map_type<pair_type>;
            _mapOwn = new std::vector<pair_type>;
            unsigned int offset = 0;
            for(unsigned short z = 0; z < size; ++z)
                for(unsigned int w = 0; w < lsol_loc_glob[z]; ++w)
                    mapping[isol_loc_glob[offset++] - 1] = std::make_pair(z, w);
            offset = 0;
            if(_idistribution) {
                for(unsigned short z = 0; z < size; ++z)
                    for(unsigned int w = 0; w < _ldistribution[z]; ++w)
                        mapping_user[_idistribution[offset++]] = std::make_pair(z, w);
            }
            else {
                for(unsigned short z = 0; z < size; ++z)
                    for(unsigned int w = 0; w < _ldistribution[z]; ++w)
                        mapping_user[offset++] = std::make_pair(z, w);
            }
            map_type<K> map_recv;
            map_type<K> map_send;
            offset = std::accumulate(_ldistribution, _ldistribution + _rank, 0);
            if(!isRHS) {
                for(unsigned int i = 0; i < info; ++i) {
                    std::pair<unsigned short, unsigned int> tmp = mapping_user[isol_loc[i] - 1];
                    if(tmp.first != _rank) {
                        map_send[tmp.first].emplace_back(sol_loc[i]);
                        (*_mapSend)[tmp.first].emplace_back(i, tmp.second);
                    }
                }
                if(_idistribution)
                    for(unsigned int x = offset; x < offset + _ldistribution[_rank]; ++x) {
                        std::pair<unsigned short, unsigned int> tmp = mapping[_idistribution[x]];
                        if(tmp.first != _rank)
                            map_recv[tmp.first].resize(map_recv[tmp.first].size() + 1);
                        else {
                            _mapOwn->emplace_back(mapping_user[_idistribution[x]].second, tmp.second);
                            sol[mapping_user[_idistribution[x]].second] = sol_loc[tmp.second];
                        }
                    }
                else
                    for(unsigned int x = offset; x < offset + _ldistribution[_rank]; ++x) {
                        std::pair<unsigned short, unsigned int> tmp = mapping[x];
                        if(tmp.first != _rank)
                            map_recv[tmp.first].resize(map_recv[tmp.first].size() + 1);
                        else {
                            _mapOwn->emplace_back(mapping_user[x].second, tmp.second);
                            sol[mapping_user[x].second] = sol_loc[tmp.second];
                        }
                    }
            }
            else {
                for(unsigned int i = 0; i < info; ++i) {
                    unsigned short tmp = mapping_user[isol_loc[i] - 1].first;
                    if(tmp != _rank)
                        map_recv[tmp].resize(map_recv[tmp].size() + 1);
                }
                if(_idistribution)
                    for(unsigned int x = offset; x < offset + _ldistribution[_rank]; ++x) {
                        std::pair<unsigned short, unsigned int> tmp = mapping[_idistribution[x]];
                        if(tmp.first != _rank) {
                            map_send[tmp.first].emplace_back(sol[x - offset]);
                            (*_mapSend)[tmp.first].emplace_back(x - offset, tmp.second);
                        }
                        else {
                            _mapOwn->emplace_back(mapping[_idistribution[x]].second, mapping_user[_idistribution[x]].second);
                            sol_loc[mapping[_idistribution[x]].second] = sol[mapping_user[_idistribution[x]].second];
                        }
                    }
                else
                    for(unsigned int x = offset; x < offset + _ldistribution[_rank]; ++x) {
                        std::pair<unsigned short, unsigned int> tmp = mapping[x];
                        if(tmp.first != _rank) {
                            map_send[tmp.first].emplace_back(sol[x - offset]);
                            (*_mapSend)[tmp.first].emplace_back(x - offset, tmp.second);
                        }
                        else {
                            _mapOwn->emplace_back(mapping[x].second, mapping_user[x].second);
                            sol_loc[mapping[x].second] = sol[mapping_user[x].second];
                        }
                    }
            }
            MPI_Request* rqSend = new MPI_Request[map_send.size() + map_recv.size()];
            MPI_Request* rqRecv = rqSend + map_send.size();
            unsigned int i = 0;
            for(typename map_type<K>::iterator it = map_send.begin(); it != map_send.end(); ++it)
                MPI_Isend(it->second.data(), it->second.size(), Wrapper<K>::mpi_type(), it->first, 4, _communicator, rqSend + i++);
            for(typename map_type<K>::iterator it = map_recv.begin(); it != map_recv.end(); ++it)
                MPI_Irecv(it->second.data(), it->second.size(), Wrapper<K>::mpi_type(), it->first, 4, _communicator, rqSend + i++);
            for(unsigned int i = 0; i < map_recv.size(); ++i) {
                int index;
                typename map_type<K>::const_iterator it_index = map_recv.cbegin();
                MPI_Waitany(map_recv.size(), rqRecv, &index, MPI_STATUS_IGNORE);
                std::advance(it_index, index);
                if(!isRHS) {
                    unsigned int offset = std::accumulate(lsol_loc_glob, lsol_loc_glob + it_index->first, 0);
                    for(unsigned int x = offset, accumulate = 0; x < offset + lsol_loc_glob[it_index->first]; ++x)
                        if(mapping_user[isol_loc_glob[x] - 1].first == _rank) {
                            (*_mapRecv)[it_index->first].emplace_back(mapping_user[isol_loc_glob[x] - 1].second, accumulate);
                            sol[mapping_user[isol_loc_glob[x] - 1].second] = (it_index->second)[accumulate++];
                        }
                }
                else {
                    unsigned int offset = std::accumulate(_ldistribution, _ldistribution + it_index->first, 0);
                    if(_idistribution) {
                        for(unsigned int x = offset, accumulate = 0; x < offset + _ldistribution[it_index->first]; ++x)
                            if(mapping[_idistribution[x]].first == _rank) {
                                (*_mapRecv)[it_index->first].emplace_back(mapping[_idistribution[x]].second, accumulate);
                                sol_loc[mapping[_idistribution[x]].second] = (it_index->second)[accumulate++];
                            }
                    }
                    else {
                        for(unsigned int x = offset, accumulate = 0; x < offset + _ldistribution[it_index->first]; ++x)
                            if(mapping[x].first == _rank) {
                                (*_mapRecv)[it_index->first].emplace_back(mapping[x].second, accumulate);
                                sol_loc[mapping[x].second] = (it_index->second)[accumulate++];
                            }
                    }
                }
            }
            MPI_Waitall(map_send.size(), rqSend, MPI_STATUSES_IGNORE);
            delete [] rqSend;
            delete [] isol_loc_glob;
            delete [] lsol_loc_glob;
            delete [] _ldistribution;
            delete [] _idistribution;
        }
        /* Function: redistribute
         *
         *  Transfers a vector numbered by the user to match the numbering of the direct solver, and vice versa.
         *
         * Template Parameter:
         *    P              - Renumbering identifier.
         *
         * Parameters:
         *    vec            - Vector following the numbering of the direct solver.
         *    res            - Vector following the numbering of the user.
         *    fuse           - Number of fused reductions (optional). */
        template<char P, class K>
        inline void redistribute(K* const vec, K* const res, const unsigned short& fuse = 0) {
            map_type<pair_type>* map_recv_index;
            map_type<pair_type>* map_send_index;
            if(P == 0 || P == 1) {
                map_recv_index = _mapRecv;
                map_send_index = _mapSend;
            }
            else {
                map_recv_index = _mapSend;
                map_send_index = _mapRecv;
            }
            map_type<K> map_recv;
            map_type<K> map_send;
            int gatherCount = fuse > 0 ? *_gatherCounts - fuse : 1;
            MPI_Request* rqSend = new MPI_Request[map_send_index->size() + map_recv_index->size()];
            MPI_Request* rqRecv = rqSend + map_send_index->size();
            unsigned short i = 0;
            for(map_type<pair_type>::const_reference q : *map_send_index) {
                map_send[q.first].reserve(q.second.size());
                for(std::vector<pair_type>::const_reference p : q.second) {
                    if(P == 0)
                        map_send[q.first].emplace_back(vec[p.first]);
                    else if(P == 1)
                        map_send[q.first].emplace_back(res[p.first + fuse * (p.first / gatherCount)]);
                    else if(P == 2)
                        map_send[q.first].emplace_back(res[p.first]);
                }
                MPI_Isend(map_send[q.first].data(), q.second.size(), Wrapper<K>::mpi_type(), q.first, 5, _communicator, rqSend + i++);
            }
            for(map_type<pair_type>::const_reference q : *map_recv_index) {
                map_recv[q.first].reserve(q.second.size());
                MPI_Irecv(map_recv[q.first].data(), q.second.size(), Wrapper<K>::mpi_type(), q.first, 5, _communicator, rqSend + i++);
            }
            for(std::vector<pair_type>::const_reference p : *_mapOwn) {
                if(P == 0)
                    res[p.first] = vec[p.second];
                else if(P == 1)
                    vec[p.first] = res[p.second + fuse * (p.second / gatherCount)];
                else if(P == 2)
                    vec[p.second + fuse * (p.second / gatherCount)] = res[p.first];
            }
            for(i = 0; i < map_recv.size(); ++i) {
                int index;
                typename map_type<K>::const_iterator it_index = map_recv.cbegin();
                MPI_Waitany(map_recv.size(), rqRecv, &index, MPI_STATUS_IGNORE);
                std::advance(it_index, index);
                K* pt = map_recv[it_index->first].data();
                for(std::vector<pair_type>::const_reference p : (*map_recv_index)[it_index->first]) {
                    if(P == 0)
                        res[p.first] = pt[p.second];
                    else if(P == 1)
                        vec[p.first] = pt[p.second];
                    else if(P == 2)
                        vec[p.first + fuse * (p.first / gatherCount)] = *pt++;
                }
            }
            MPI_Waitall(map_send.size(), rqSend, MPI_STATUSES_IGNORE);
            delete [] rqSend;
        }
    public:
        DMatrix() : _mapRecv(), _mapSend(), _mapOwn(), _ldistribution(), _idistribution(), _gatherCounts(), _gatherSplitCounts(), _displs(), _displsSplit(), _communicator(MPI_COMM_NULL), _n(), _rank(), _distribution() { }
        DMatrix(const DMatrix&) = delete;
        ~DMatrix() {
            if(!_mapRecv) {
                delete [] _ldistribution;
                delete [] _idistribution;
            }
            delete _mapRecv;
            delete _mapSend;
            delete _mapOwn;
            delete [] _gatherCounts;
            delete [] _gatherSplitCounts;
            delete [] _displs;
            delete [] _displsSplit;
        }
        /* Function: getDistribution
         *  Returns <Coarse operator::distribution>. */
        inline Distribution getDistribution() const {
            return _distribution;
        }
};
} // HPDDM
#endif // _DISTRIBUTED_MATRIX_
