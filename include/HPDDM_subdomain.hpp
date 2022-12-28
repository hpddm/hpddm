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

#ifndef HPDDM_SUBDOMAIN_HPP_
#define HPDDM_SUBDOMAIN_HPP_

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
        MatrixCSR<K>*                a_;
        /* Variable : buff
         *  Array used as the receiving and receiving buffer for point-to-point communications with neighboring subdomains. */
        K**                       buff_;
        /* Variable: map */
        vectorNeighbor             map_;
        /* Variable: rq
         *  Array of MPI requests to check completion of the MPI transfers with neighboring subdomains. */
        MPI_Request*                rq_;
        /* Variable: communicator
         *  MPI communicator of the subdomain. */
        MPI_Comm          communicator_;
        /* Variable: dof
         *  Number of degrees of freedom in the current subdomain. */
        int                        dof_;
        void dtor() {
            clearBuffer();
            delete [] rq_;
            rq_ = nullptr;
            vectorNeighbor().swap(map_);
            delete [] buff_;
            buff_ = nullptr;
#ifndef PETSCHPDDM_H
            destroyMatrix(nullptr);
#endif
        }
    public:
        Subdomain() :
#if !HPDDM_PETSC
                      OptionsPrefix<K>(),
#endif
                                          a_(), buff_(), map_(), rq_(), dof_() { }
        Subdomain(const Subdomain<K>& s) :
#if !HPDDM_PETSC
                                           OptionsPrefix<K>(),
#endif
                                                               a_(), buff_(new K*[2 * s.map_.size()]), map_(s.map_), rq_(new MPI_Request[2 * s.map_.size()]), communicator_(s.communicator_), dof_(s.dof_) { }
        ~Subdomain() {
            dtor();
        }
        typedef int integer_type;
        /* Function: getCommunicator
         *  Returns a reference to <Subdomain::communicator>. */
        const MPI_Comm& getCommunicator() const { return communicator_; }
        /* Function: getMap
         *  Returns a reference to <Subdomain::map>. */
        const vectorNeighbor& getMap() const { return map_; }
        /* Function: exchange
         *
         *  Exchanges and reduces values of duplicated unknowns.
         *
         * Parameter:
         *    in             - Input vector. */
        void exchange(K* const in, const unsigned short& mu = 1) const {
            for(unsigned short nu = 0; nu < mu; ++nu) {
                for(unsigned short i = 0, size = map_.size(); i < size; ++i) {
                    MPI_Irecv(buff_[i], map_[i].second.size(), Wrapper<K>::mpi_type(), map_[i].first, 0, communicator_, rq_ + i);
                    Wrapper<K>::gthr(map_[i].second.size(), in + nu * dof_, buff_[size + i], map_[i].second.data());
                    MPI_Isend(buff_[size + i], map_[i].second.size(), Wrapper<K>::mpi_type(), map_[i].first, 0, communicator_, rq_ + size + i);
                }
                for(unsigned short i = 0; i < map_.size(); ++i) {
                    int index;
                    ignore(MPI_Waitany(map_.size(), rq_, &index, MPI_STATUS_IGNORE));
                    for(unsigned int j = 0; j < map_[index].second.size(); ++j)
                        in[map_[index].second[j] + nu * dof_] += buff_[index][j];
                }
                ignore(MPI_Waitall(map_.size(), rq_ + map_.size(), MPI_STATUSES_IGNORE));
            }
        }
        template<class T, typename std::enable_if<!HPDDM::Wrapper<K>::is_complex && HPDDM::Wrapper<T>::is_complex && std::is_same<K, underlying_type<T>>::value>::type* = nullptr>
        void exchange(T* const in, const unsigned short& mu = 1) const {
            for(unsigned short nu = 0; nu < mu; ++nu) {
                K* transpose = reinterpret_cast<K*>(in + nu * dof_);
                Wrapper<K>::template cycle<'T'>(dof_, 2, transpose, 1);
                exchange(transpose, 2);
                Wrapper<K>::template cycle<'T'>(2, dof_, transpose, 1);
            }
        }
        /* Function: recvBuffer
         *
         *  Exchanges values of duplicated unknowns.
         *
         * Parameter:
         *    in             - Input vector. */
        void recvBuffer(const K* const in) const {
            for(unsigned short i = 0, size = map_.size(); i < size; ++i) {
                MPI_Irecv(buff_[i], map_[i].second.size(), Wrapper<K>::mpi_type(), map_[i].first, 0, communicator_, rq_ + i);
                Wrapper<K>::gthr(map_[i].second.size(), in, buff_[size + i], map_[i].second.data());
                MPI_Isend(buff_[size + i], map_[i].second.size(), Wrapper<K>::mpi_type(), map_[i].first, 0, communicator_, rq_ + size + i);
            }
            MPI_Waitall(2 * map_.size(), rq_, MPI_STATUSES_IGNORE);
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
                communicator_ = *comm;
            else
                communicator_ = MPI_COMM_WORLD;
            unsigned int* perm = nullptr;
#ifndef PETSCHPDDM_H
            if(a && restriction) {
                perm = new unsigned int[a->n_]();
                for(unsigned int i = 0; i < restriction->n_; ++i)
                    perm[restriction->ja_[i]] = i + 1;
                a_ = new MatrixCSR<K>(a, restriction, perm);
            }
            else
                a_ = a;
            if(a_)
                dof_ = a_->n_;
#else
            ignore(restriction);
#endif
            std::vector<unsigned short> sortable;
            std::copy(o.begin(), o.end(), std::back_inserter(sortable));
            map_.reserve(sortable.size());
            std::vector<unsigned short> idx(sortable.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](const unsigned short& lhs, const unsigned short& rhs) { return sortable[lhs] < sortable[rhs]; });
            unsigned short j = 0;
            for(const unsigned short& i : idx) {
                if(r[idx[j]].size() > 0) {
                    map_.emplace_back(sortable[i], typename decltype(map_)::value_type::second_type());
                    map_.back().second.reserve(r[idx[j]].size());
                    for(int k = 0; k < r[idx[j]].size(); ++k)
                        map_.back().second.emplace_back(r[idx[j]][k]);
                }
                ++j;
            }
            if(perm) {
                const int size = map_.size();
                MPI_Request* rq = new MPI_Request[2 * size];
                unsigned int space = 0;
                for(unsigned short i = 0; i < size; ++i)
                    space += map_[i].second.size();
                unsigned char* send = new unsigned char[2 * space];
                unsigned char* recv = send + space;
                space = 0;
                for(unsigned short i = 0; i < size; ++i) {
                    MPI_Irecv(recv, map_[i].second.size(), MPI_UNSIGNED_CHAR, map_[i].first, 100, communicator_, rq + i);
                    if(a->n_)
                        for(unsigned int j = 0; j < map_[i].second.size(); ++j)
                            send[j] = (perm[map_[i].second[j]] > 0 ? 'a' : 'b');
                    else
                        std::fill_n(send, map_[i].second.size(), 'b');
                    MPI_Isend(send, map_[i].second.size(), MPI_UNSIGNED_CHAR, map_[i].first, 100, communicator_, rq + size + i);
                    send += map_[i].second.size();
                    recv += map_[i].second.size();
                    space += map_[i].second.size();
                }
                MPI_Waitall(2 * size, rq, MPI_STATUSES_IGNORE);
                vectorNeighbor map;
                map.reserve(size);
                send -= space;
                recv -= space;
                for(unsigned short i = 0; i < size; ++i) {
                    std::pair<unsigned short, typename decltype(map_)::value_type::second_type> c(map_[i].first, typename decltype(map_)::value_type::second_type());
                    for(unsigned int j = 0; j < map_[i].second.size(); ++j) {
                        if(recv[j] == 'a' && send[j] == 'a')
                            c.second.emplace_back(perm[map_[i].second[j]] - 1);
                    }
                    if(!c.second.empty())
                        map.emplace_back(c);
                    send += map_[i].second.size();
                    recv += map_[i].second.size();
                }
                send -= space;
                delete [] send;
                delete [] rq;
                map_ = map;
            }
            delete [] perm;
            rq_ = new MPI_Request[2 * map_.size()];
            buff_ = new K*[2 * map_.size()]();
        }
#ifndef PETSCHPDDM_H
        void initialize(MatrixCSR<K>* const& a, const int neighbors, const int* const list, const int* const sizes, const int* const* const connectivity, MPI_Comm* const& comm = nullptr) {
            if(comm)
                communicator_ = *comm;
            else
                communicator_ = MPI_COMM_WORLD;
            a_ = a;
            if(a_)
                dof_ = a_->n_;
            map_.reserve(neighbors);
            std::vector<unsigned short> idx(neighbors);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](const unsigned short& lhs, const unsigned short& rhs) { return list[lhs] < list[rhs]; });
            unsigned short j = 0;
            while(j < neighbors) {
                if(sizes[idx[j]] > 0) {
                    map_.emplace_back(list[idx[j]], typename decltype(map_)::value_type::second_type());
                    map_.back().second.reserve(sizes[idx[j]]);
                    for(int k = 0; k < sizes[idx[j]]; ++k)
                        map_.back().second.emplace_back(connectivity[idx[j]][k]);
                }
                ++j;
            }
            rq_ = new MPI_Request[2 * map_.size()];
            buff_ = new K*[2 * map_.size()]();
        }
#endif
        bool setBuffer(K* wk = nullptr, const int& space = 0) const {
            int n = std::accumulate(map_.cbegin(), map_.cend(), 0, [](unsigned int init, const pairNeighbor& i) { return init + i.second.size(); });
            if(n == 0)
                return false;
            bool allocate;
            if(2 * n <= space && wk) {
                *buff_ = wk;
                allocate = false;
            }
            else {
                *buff_ = new K[2 * n];
                allocate = true;
            }
            buff_[map_.size()] = *buff_ + n;
            n = 0;
            for(unsigned short i = 1, size = map_.size(); i < size; ++i) {
                n += map_[i - 1].second.size();
                buff_[i] = *buff_ + n;
                buff_[size + i] = buff_[size] + n;
            }
            return allocate;
        }
        void clearBuffer(const bool free = true) const {
            if(free && !map_.empty() && buff_) {
                delete [] *buff_;
                *buff_ = nullptr;
            }
        }
        void end(const bool free = true) const { clearBuffer(free); }
        /* Function: initialize(dummy)
         *  Dummy function for main processes excluded from the domain decomposition. */
        void initialize(MPI_Comm* const& comm = nullptr) {
            if(comm)
                communicator_ = *comm;
            else
                communicator_ = MPI_COMM_WORLD;
        }
        /* Function: exclusion
         *
         *  Checks whether <Subdomain::communicator> has been built by excluding some processes.
         *
         * Parameter:
         *    comm          - Reference MPI communicator. */
        bool exclusion(const MPI_Comm& comm) const {
            int result;
            MPI_Comm_compare(communicator_, comm, &result);
            return result != MPI_CONGRUENT && result != MPI_IDENT;
        }
#ifndef PETSCHPDDM_H
        K boundaryCond(const unsigned int i) const {
            if(a_->ia_) {
                const int shift = a_->ia_[0];
                unsigned int stop;
                if(a_->ia_[i] != a_->ia_[i + 1]) {
                    if(!a_->sym_)
                        stop = std::distance(a_->ja_, std::upper_bound(a_->ja_ + a_->ia_[i] - shift, a_->ja_ + a_->ia_[i + 1] - shift, i + shift));
                    else
                        stop = a_->ia_[i + 1] - shift;
                    if((a_->sym_ || stop < a_->ia_[i + 1] - shift || a_->ja_[a_->ia_[i + 1] - shift - 1] == i + shift) && a_->ja_[std::max(1U, stop) - 1] == i + shift && std::abs(a_->a_[stop - 1]) < HPDDM_EPS * HPDDM_PEN)
                        for(unsigned int j = a_->ia_[i] - shift; j < stop; ++j) {
                            if(i != a_->ja_[j] - shift && std::abs(a_->a_[j]) > HPDDM_EPS)
                                return K();
                            else if(i == a_->ja_[j] - shift && std::abs(a_->a_[j] - K(1.0)) > HPDDM_EPS)
                                return K();
                        }
                }
                else
                    return K();
                return a_->a_[stop - 1];
            }
            else
                return K();
        }
        std::unordered_map<unsigned int, K> boundaryConditions() const {
            std::unordered_map<unsigned int, K> map;
            map.reserve(dof_ / 1000);
            for(unsigned int i = 0; i < Subdomain<K>::dof_; ++i) {
                const K boundary = boundaryCond(i);
                if(std::abs(boundary) > HPDDM_EPS)
                    map[i] = boundary;
            }
            return map;
        }
#endif
        /* Function: getDof
         *  Returns the value of <Subdomain::dof>. */
        constexpr int getDof() const { return dof_; }
        /* Function: setDof
         *  Sets the value of <Subdomain::dof>. */
        void setDof(int dof) {
            if(!dof_
#ifndef PETSCHPDDM_H
                    && !a_
#endif
                          )
                dof_ = dof;
        }
#ifndef PETSCHPDDM_H
        /* Function: getMatrix
         *  Returns a pointer to <Subdomain::a>. */
        const MatrixCSR<K>* getMatrix() const { return a_; }
        /* Function: setMatrix
         *  Sets the pointer <Subdomain::a>. */
        bool setMatrix(MatrixCSR<K>* const& a) {
            bool ret = !(a_ && a && a_->n_ == a->n_ && a_->m_ == a->m_ && a_->nnz_ == a->nnz_);
            if(!dof_ && a)
                dof_ = a->n_;
            delete a_;
            a_ = a;
            return ret;
        }
        /* Function: destroyMatrix
         *  Destroys the pointer <Subdomain::a> using a custom deallocator. */
        void destroyMatrix(void (*dtor)(void*)) {
            if(a_) {
                int isFinalized;
                MPI_Finalized(&isFinalized);
                if(!isFinalized) {
                    int rankWorld;
                    MPI_Comm_rank(communicator_, &rankWorld);
#if !HPDDM_PETSC
                    const std::string prefix = OptionsPrefix<K>::prefix();
                    const Option& opt = *Option::get();
                    std::string filename = opt.prefix(prefix + "dump_matrices", true);
                    if(filename.size() == 0)
                        filename = opt.prefix(prefix + "dump_matrix_" + to_string(rankWorld), true);
                    if(filename.size() != 0) {
                        int sizeWorld;
                        MPI_Comm_size(communicator_, &sizeWorld);
                        std::ofstream output { filename + "_" + to_string(rankWorld) + "_" + to_string(sizeWorld) + ".txt" };
                        output << *a_;
                    }
#endif
                }
                if(dtor)
                    a_->destroy(dtor);
                delete a_;
                a_ = nullptr;
            }
        }
#endif
        /* Function: getRq
         *  Returns a pointer to <Subdomain::rq>. */
        MPI_Request* getRq() const { return rq_; }
        /* Function: getBuffer
         *  Returns a pointer to <Subdomain::buff>. */
        K** getBuffer() const { return buff_; }
        template<bool excluded>
        void scatter(const K* const, K*&, const unsigned short, unsigned short&, const MPI_Comm&) const { }
        void statistics() const {
#if !HPDDM_PETSC
            unsigned long long local[4], global[4];
            unsigned short* const table = new unsigned short[dof_];
            int n;
            MPI_Comm_rank(communicator_, &n);
            std::fill_n(table, dof_, n);
            for(const auto& i : map_)
                for(const int& j : i.second)
                    table[j] = i.first;
            local[0] = local[2] = 0;
            for(unsigned int i = 0; i < dof_; ++i)
                if(table[i] <= n) {
                    ++local[0];
                    local[2] += a_->ia_[i + 1] - a_->ia_[i];
                }
            if(a_->sym_) {
                local[2] *= 2;
                local[2] -= local[0];
            }
            if(dof_ == a_->n_)
                local[1] = dof_ - local[0];
            else {
                local[1] = local[0];
                local[0] = a_->n_ - local[1];
            }
            delete [] table;
            local[3] = map_.size();
            MPI_Allreduce(local, global, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator_);
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
                MPI_Comm_size(communicator_, &n);
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
            MPI_Comm_rank(communicator_, &rankWorld);
            MPI_Comm_size(communicator_, &sizeWorld);
            std::map<unsigned int, unsigned int> r;
            if(list) {
                for(unsigned int i = 0; i < Subdomain<K>::dof_; ++i)
                    if(list[i] > 0)
                        r[list[i]] = i;
            }
            if(sizeWorld > 1) {
                setBuffer();
                T between = 0;
                for(unsigned short i = 0; i < map_.size() && map_[i].first < rankWorld; ++i)
                    ++between;
                T* local = new T[sizeWorld];
                local[rankWorld] = (list ? r.size() : std::distance(first, last));
                std::unordered_set<unsigned int> removed;
                removed.reserve(local[rankWorld]);
                for(unsigned short i = 0; i < map_.size(); ++i)
                    for(const int& j : map_[i].second) {
                        if(d && d[j] < HPDDM_EPS && removed.find(j) == removed.cend() && (!list || list[j] > 0)) {
                            --local[rankWorld];
                            removed.insert(j);
                        }
                    }
                MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, local, 1, Wrapper<T>::mpi_type(), communicator_);
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
                if(!map_.empty()) {
                    for(unsigned short i = 0; i < map_.size(); ++i)
                        MPI_Irecv(static_cast<void*>(buff_[i]), map_[i].second.size(), Wrapper<T>::mpi_type(), map_[i].first, 10, communicator_, rq_ + i);
                    for(unsigned short i = 0; i < map_.size(); ++i) {
                        T* sbuff = reinterpret_cast<T*>(buff_[map_.size() + i]);
                        for(unsigned int j = 0; j < map_[i].second.size(); ++j)
                            sbuff[j] = *(first + map_[i].second[j]);
                        MPI_Isend(static_cast<void*>(sbuff), map_[i].second.size(), Wrapper<T>::mpi_type(), map_[i].first, 10, communicator_, rq_ + map_.size() + i);
                    }
                    for(unsigned short i = 0; i < map_.size(); ++i) {
                        int index;
                        MPI_Waitany(map_.size(), rq_, &index, MPI_STATUS_IGNORE);
                        T* rbuff = reinterpret_cast<T*>(buff_[index]);
                        for(const int& j : map_[index].second) {
                            if(first[j] == std::numeric_limits<T>::max())
                                first[j] = *rbuff;
                            ++rbuff;
                        }
                    }
                }
                MPI_Waitall(map_.size(), rq_ + map_.size(), MPI_STATUSES_IGNORE);
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
            if(A->sym_) {
                if(col || !std::is_same<K, T>::value)
                    std::cerr << "Not implemented" << std::endl;
                transpose = new std::vector<std::pair<int, int>>[A->n_]();
                for(int i = 0; i < A->n_; ++i)
                    for(int j = A->ia_[i] - (HPDDM_NUMBERING == 'F'); j < A->ia_[i + 1] - (HPDDM_NUMBERING == 'F'); ++j)
                        transpose[A->ja_[j] - (HPDDM_NUMBERING == 'F')].emplace_back(i, j);
                for(int i = 0; i < A->n_; ++i)
                    std::sort(transpose[i].begin(), transpose[i].end());
            }
            if(first != 0 || last != A->n_ || col) {
                if(!col)
                    col = row;
                std::vector<std::pair<I, I>> s;
                s.reserve(A->n_);
                for(unsigned int i = 0; i < A->n_; ++i)
                    s.emplace_back(row[i], i);
                std::sort(s.begin(), s.end());
                typename std::vector<std::pair<I, I>>::iterator begin = std::lower_bound(s.begin(), s.end(), std::make_pair(first, static_cast<I>(0)));
                typename std::vector<std::pair<I, I>>::iterator end = std::upper_bound(begin, s.end(), std::make_pair(last, static_cast<I>(0)));
                unsigned int dof = std::distance(begin, end);
                std::vector<std::pair<I, T>> tmp;
                tmp.reserve(A->nnz_);
                if(!ia)
                    ia = new I[dof + 1];
                ia[0] = 0;
                for(typename std::vector<std::pair<I, I>>::iterator it = begin; it != end; ++it) {
                    for(unsigned int j = A->ia_[it->second]; j < A->ia_[it->second + 1]; ++j)
                        tmp.emplace_back(col[A->ja_[j]], std::is_same<K, T>::value ? A->a_[j] : j);
                    if(A->sym_) {
                        for(unsigned int j = 0; j < transpose[it->second].size(); ++j) {
                            if(transpose[it->second][j].first != it->second)
                                tmp.emplace_back(col[transpose[it->second][j].first], A->a_[transpose[it->second][j].second]);
                        }
                    }
                    std::sort(tmp.begin() + ia[std::distance(begin, it)], tmp.end(), [](const std::pair<I, T>& lhs, const std::pair<I, T>& rhs) { return lhs.first < rhs.first; });
                    if(A->sym_) {
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
                if(!A->sym_) {
                    if(std::is_same<decltype(A->ia_), I>::value) {
                        ia = reinterpret_cast<I*>(A->ia_);
                        ja = reinterpret_cast<I*>(A->ja_);
                        if(std::is_same<K, T>::value)
                            c = reinterpret_cast<T*>(A->a_);
                        else
                            c = nullptr;
                        return false;
                    }
                    else {
                        if(!std::is_same<K, T>::value)
                            std::cerr << "Not implemented" << std::endl;
                        if(!ia)
                            ia = new I[A->n_ + 1];
                        std::copy_n(A->ia_, A->n_ + 1, ia);
                        if(!ja)
                            ja = new I[A->nnz_];
                        std::copy_n(A->ja_, A->nnz_, ja);
                        if(!c)
                            c = new T[A->nnz_];
                        std::copy_n(A->a_, A->nnz_, c);
                        return true;
                    }
                }
                else {
                    if(!ia)
                        ia = new I[A->n_ + 1];
                    if(!ja)
                        ja = new I[A->nnz_];
                    if(!c)
                        c = new T[A->nnz_];
                    ia[0] = 0;
                    for(int i = 0; i < A->n_; ++i) {
                        for(int j = 0; j < transpose[i].size(); ++j) {
                            c[ia[i] + j] = A->a_[transpose[i][j].second];
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
#endif // HPDDM_SUBDOMAIN_HPP_
