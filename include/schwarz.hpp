/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
              Frédéric Nataf <nataf@ann.jussieu.fr>
        Date: 2013-03-10

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

#ifndef _HPDDM_SCHWARZ_
#define _HPDDM_SCHWARZ_

#include <set>
#include "preconditioner.hpp"

namespace HPDDM {
#if HPDDM_DENSE
template<template<class> class, template<class> class, char, class> class Dense;
#endif
/* Class: Schwarz
 *
 *  A class for solving problems using Schwarz methods that inherits from <Preconditioner>.
 *
 * Template Parameters:
 *    Solver         - Solver used for the factorization of local matrices.
 *    CoarseOperator - Class of the coarse operator.
 *    S              - 'S'ymmetric or 'G'eneral coarse operator.
 *    K              - Scalar type. */
template<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
    template<class> class Solver, template<class> class CoarseSolver,
#endif
    char S, class K>
class Schwarz : public Preconditioner<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
                Solver, CoarseOperator<CoarseSolver, S, K>,
#endif
                K> {
    public:
        /* Enum: Prcndtnr
         *
         *  Defines the Schwarz method used as a preconditioner.
         *
         * NO           - No preconditioner.
         * SY           - Symmetric preconditioner, e.g. Additive Schwarz method.
         * GE           - Nonsymmetric preconditioner, e.g. Restricted Additive Schwarz method.
         * OS           - Optimized symmetric preconditioner, e.g. Optimized Schwarz method.
         * OG           - Optimized nonsymmetric preconditioner, e.g. Optimized Restricted Additive Schwarz method. */
        enum class Prcndtnr : char {
            NO, SY, GE, OS, OG
        };
    protected:
        /* Variable: d
         *  Local partition of unity. */
        const underlying_type<K>* _d;
        std::size_t            _hash;
        /* Variable: type
         *  Type of <Prcndtnr> used in <Schwarz::apply> and <Schwarz::deflation>. */
        Prcndtnr               _type;
    public:
        Schwarz() : _d(), _hash(), _type(Prcndtnr::NO) { }
        explicit Schwarz(const Subdomain<K>& s) : super(s), _d(), _hash(), _type(Prcndtnr::NO) { }
        ~Schwarz() { _d = nullptr; }
        /* Typedef: super
         *  Type of the immediate parent class <Preconditioner>. */
        typedef Preconditioner<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
            Solver, CoarseOperator<CoarseSolver, S, K>,
#endif
            K> super;
        /* Function: initialize
         *  Sets <Schwarz::d>. */
        void initialize(underlying_type<K>* const& d) {
            _d = d;
        }
        /* Function: exchange */
        template<bool allocate = false>
        void exchange(K* const x, const unsigned short& mu = 1) const {
            bool free = false;
            if(allocate)
                free = Subdomain<K>::setBuffer();
            Wrapper<K>::diag(Subdomain<K>::_dof, _d, x, mu);
            Subdomain<K>::exchange(x, mu);
            if(allocate)
                Subdomain<K>::clearBuffer(free);
        }
        void exchange() const {
            std::vector<K>* send = new std::vector<K>[Subdomain<K>::_map.size()];
            unsigned int* sizes = new unsigned int[Subdomain<K>::_map.size()]();
            std::vector<std::pair<int, int>>* transpose = nullptr;
            if(Subdomain<K>::_a->_sym) {
                transpose = new std::vector<std::pair<int, int>>[Subdomain<K>::_dof]();
                for(int i = 0; i < Subdomain<K>::_dof; ++i)
                    for(int j = Subdomain<K>::_a->_ia[i] - (HPDDM_NUMBERING == 'F'); j < Subdomain<K>::_a->_ia[i + 1] - (HPDDM_NUMBERING == 'F'); ++j)
                        transpose[Subdomain<K>::_a->_ja[j] - (HPDDM_NUMBERING == 'F')].emplace_back(i, j);
            }
            for(unsigned short i = 0, size = Subdomain<K>::_map.size(); i < size; ++i) {
                const pairNeighbor& pair = Subdomain<K>::_map[i];
                std::vector<int> idx(pair.second.size());
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&pair](int lhs, int rhs) { return pair.second[lhs] < pair.second[rhs]; });
                for(unsigned int j = 0; j < pair.second.size(); ++j) {
                    unsigned int nnz = 0, n = send[i].size() + 1;
                    if(_d[pair.second[j]] > HPDDM_EPS) {
                        send[i].emplace_back(j);
                        send[i].emplace_back(0);
                    }
                    sizes[i] += 2;
                    std::vector<int>::const_iterator it = idx.cbegin();
                    for(int k = Subdomain<K>::_a->_ia[pair.second[j]] - (HPDDM_NUMBERING == 'F'); k < Subdomain<K>::_a->_ia[pair.second[j] + 1] - (HPDDM_NUMBERING == 'F'); ++k) {
                        it = std::lower_bound(it, idx.cend(), Subdomain<K>::_a->_ja[k] - (HPDDM_NUMBERING == 'F'), [&pair](int lhs, int rhs) { return pair.second[lhs] < rhs; });
                        if(it != idx.cend() && pair.second[*it] == Subdomain<K>::_a->_ja[k] - (HPDDM_NUMBERING == 'F')) {
                            if(_d[pair.second[j]] > HPDDM_EPS) {
                                send[i].emplace_back(*it);
                                send[i].emplace_back(Subdomain<K>::_a->_a[k]);
                                ++nnz;
                            }
                            sizes[i] += 2;
                        }
                    }
                    if(Subdomain<K>::_a->_sym) {
                        for(int k = 0; k < transpose[pair.second[j]].size(); ++k) {
                            std::vector<int>::const_iterator first = std::lower_bound(it, idx.cend(), transpose[pair.second[j]][k].first, [&pair](int lhs, int rhs) { return pair.second[lhs] < rhs; });
                            if(first != idx.cend() && pair.second[*first] == transpose[pair.second[j]][k].first) {
                                if(_d[pair.second[j]] > HPDDM_EPS) {
                                    send[i].emplace_back(*first);
                                    send[i].emplace_back(Subdomain<K>::_a->_a[transpose[pair.second[j]][k].second]);
                                    ++nnz;
                                }
                                sizes[i] += 2;
                            }
                        }
                    }
                    if(_d[pair.second[j]] > HPDDM_EPS)
                        send[i][n] = nnz;
                }
                MPI_Isend(send[i].data(), send[i].size(), Wrapper<K>::mpi_type(), Subdomain<K>::_map[i].first, 13, Subdomain<K>::_communicator, Subdomain<K>::_rq + size + i);
            }
            delete [] transpose;
            K** recv = new K*[Subdomain<K>::_map.size()]();
            for(unsigned short i = 0, size = Subdomain<K>::_map.size(); i < size; ++i) {
                if(sizes[i]) {
                    recv[i] = new K[sizes[i]];
                    MPI_Irecv(recv[i], sizes[i], Wrapper<K>::mpi_type(), Subdomain<K>::_map[i].first, 13, Subdomain<K>::_communicator, Subdomain<K>::_rq + i);
                }
                else
                    Subdomain<K>::_rq[i] = MPI_REQUEST_NULL;
            }
            for(unsigned short i = 0, size = Subdomain<K>::_map.size(); i < size; ++i) {
                int index;
                MPI_Status st;
                MPI_Waitany(size, Subdomain<K>::_rq, &index, &st);
                if(st.MPI_SOURCE != MPI_ANY_SOURCE && st.MPI_TAG == 13) {
                    int size;
                    MPI_Get_count(&st, Wrapper<K>::mpi_type(), &size);
                    for(unsigned int j = 0; j < size; ) {
                        const unsigned int row = Subdomain<K>::_map[index].second[std::lround(std::abs(recv[index][j]))];
                        const unsigned int nnz = std::lround(std::abs(recv[index][j + 1]));
                        j += 2;
                        for(unsigned int k = 0; k < nnz; ++k, j += 2) {
                            int* const pt = std::lower_bound(Subdomain<K>::_a->_ja + Subdomain<K>::_a->_ia[row] - (HPDDM_NUMBERING == 'F'), Subdomain<K>::_a->_ja + Subdomain<K>::_a->_ia[row + 1] - (HPDDM_NUMBERING == 'F'), Subdomain<K>::_map[index].second[std::lround(std::abs(recv[index][j]))] + (HPDDM_NUMBERING == 'F'));
                            if(pt != Subdomain<K>::_a->_ja + Subdomain<K>::_a->_ia[row + 1] - (HPDDM_NUMBERING == 'F') && *pt == Subdomain<K>::_map[index].second[std::lround(std::abs(recv[index][j]))] + (HPDDM_NUMBERING == 'F'))
                                Subdomain<K>::_a->_a[std::distance(Subdomain<K>::_a->_ja, pt)] = recv[index][j + 1];
                        }
                    }
                }
            }
            std::for_each(recv, recv + Subdomain<K>::_map.size(), std::default_delete<K[]>());
            delete [] recv;
            MPI_Waitall(Subdomain<K>::_map.size(), Subdomain<K>::_rq + Subdomain<K>::_map.size(), MPI_STATUSES_IGNORE);
            delete [] send;
            delete [] sizes;
        }
        bool restriction(underlying_type<K>* const D) const {
            unsigned int n = 0;
            for(const auto& i : Subdomain<K>::_map)
                n += i.second.size();
            std::vector<int> p;
            if(n && !Subdomain<K>::_map.empty()) {
                underlying_type<K>** const buff = new underlying_type<K>*[2 * Subdomain<K>::_map.size()];
                *buff = new underlying_type<K>[2 * n];
                buff[Subdomain<K>::_map.size()] = *buff + n;
                n = 0;
                for(unsigned short i = 0, size = Subdomain<K>::_map.size(); i < size; ++i) {
                    buff[i] = *buff + n;
                    buff[size + i] = buff[size] + n;
                    MPI_Irecv(buff[i], Subdomain<K>::_map[i].second.size(), Wrapper<K>::mpi_underlying_type(), Subdomain<K>::_map[i].first, 0, Subdomain<K>::_communicator, Subdomain<K>::_rq + i);
                    Wrapper<underlying_type<K>>::gthr(Subdomain<K>::_map[i].second.size(), D, buff[size + i], Subdomain<K>::_map[i].second.data());
                    MPI_Isend(buff[size + i], Subdomain<K>::_map[i].second.size(), Wrapper<K>::mpi_underlying_type(), Subdomain<K>::_map[i].first, 0, Subdomain<K>::_communicator, Subdomain<K>::_rq + size + i);
                    n += Subdomain<K>::_map[i].second.size();
                }
                underlying_type<K>* const d = new underlying_type<K>[Subdomain<K>::_dof];
                std::copy_n(D, Subdomain<K>::_dof, d);
                for(unsigned short i = 0, size = Subdomain<K>::_map.size(); i < size; ++i) {
                    int index;
                    MPI_Waitany(size, Subdomain<K>::_rq, &index, MPI_STATUS_IGNORE);
                    for(int j = 0; j < Subdomain<K>::_map[index].second.size(); ++j)
                        d[Subdomain<K>::_map[index].second[j]] += buff[index][j];
                }
                p.reserve(Subdomain<K>::_dof);
                for(int i = 0; i < Subdomain<K>::_dof; ++i)
                    if((std::abs(D[i] - 1.0) > HPDDM_EPS && std::abs(D[i]) > HPDDM_EPS) || std::abs(d[i] - 1.0) > HPDDM_EPS)
                        p.emplace_back(i);
                delete [] d;
                int rank;
                MPI_Comm_rank(Subdomain<K>::_communicator, &rank);
                for(int k = 0; k < p.size(); ++k) {
                    bool largest = true;
                    for(unsigned short i = 0, size = Subdomain<K>::_map.size(); i < size && largest; ++i) {
                        std::vector<int>::const_iterator it = std::find(Subdomain<K>::_map[i].second.cbegin(), Subdomain<K>::_map[i].second.cend(), p[k]);
                        if(it != Subdomain<K>::_map[i].second.cend()) {
                            const underlying_type<K> v = D[p[k]] - buff[i][std::distance(Subdomain<K>::_map[i].second.cbegin(), it)];
                            largest = (v > HPDDM_EPS || (std::abs(v) < HPDDM_EPS && rank > Subdomain<K>::_map[i].first));
                        }
                    }
                    D[p[k]] = (largest ? 1.0 : 0.0);
                }
                MPI_Waitall(Subdomain<K>::_map.size(), Subdomain<K>::_rq + Subdomain<K>::_map.size(), MPI_STATUSES_IGNORE);
                delete [] *buff;
                delete [] buff;
            }
            return p.size() > 0;
        }
#if HPDDM_SCHWARZ
        /* Function: callNumfact
         *  Factorizes <Subdomain::a> or another user-supplied matrix, useful for <Prcndtnr::OS> and <Prcndtnr::OG>. */
        template<char N = HPDDM_NUMBERING>
        void callNumfact(MatrixCSR<K>* const& A = nullptr) {
            Option& opt = *Option::get();
            const bool resetPrefix = (opt.getPrefix().size() == 0 && super::prefix().size() != 0);
            if(resetPrefix)
                opt.setPrefix(super::prefix());
            unsigned short m = opt.val<unsigned short>("schwarz_method");
            if(A) {
                std::size_t hash = A->hashIndices();
                if(_hash != hash) {
                    _hash = hash;
                    super::destroySolver();
                }
            }
            switch(m) {
                case HPDDM_SCHWARZ_METHOD_SORAS: _type = (A ? Prcndtnr::OS : Prcndtnr::SY); break;
                case HPDDM_SCHWARZ_METHOD_ASM:   _type = Prcndtnr::SY; break;
                case HPDDM_SCHWARZ_METHOD_NONE:  _type = Prcndtnr::NO; return;
                default:                         _type = (A && (m == HPDDM_SCHWARZ_METHOD_ORAS || m == HPDDM_SCHWARZ_METHOD_OSM) ? Prcndtnr::OG : Prcndtnr::GE);
            }
            m = opt.val<unsigned short>("reuse_preconditioner");
            if(m <= 1)
                super::_s.template numfact<N>(_type == Prcndtnr::OS || _type == Prcndtnr::OG ? A : Subdomain<K>::_a);
            if(m >= 1)
                opt["reuse_preconditioner"] += 1;
            if(resetPrefix)
                opt.setPrefix("");
        }
        void setMatrix(MatrixCSR<K>* const& a) {
            bool fact = super::setMatrix(a) && _type != Prcndtnr::OS && _type != Prcndtnr::OG;
            if(fact) {
                Option& opt = *Option::get();
                const bool resetPrefix = (opt.getPrefix().size() == 0 && super::prefix().size() != 0);
                if(resetPrefix)
                    opt.setPrefix(super::prefix());
                super::destroySolver();
                super::_s.numfact(a);
                _hash = a->hashIndices();
                if(resetPrefix)
                    opt.setPrefix("");
            }
        }
        /* Function: multiplicityScaling
         *
         *  Builds the multiplicity scaling.
         *
         * Parameter:
         *    d              - Array of values. */
        void multiplicityScaling(underlying_type<K>* const d) const {
            bool allocate = Subdomain<K>::setBuffer();
            for(unsigned short i = 0, size = Subdomain<K>::_map.size(); i < size; ++i) {
                underlying_type<K>* const recv = reinterpret_cast<underlying_type<K>*>(Subdomain<K>::_buff[i]);
                underlying_type<K>* const send = reinterpret_cast<underlying_type<K>*>(Subdomain<K>::_buff[size + i]);
                MPI_Irecv(recv, Subdomain<K>::_map[i].second.size(), Wrapper<K>::mpi_underlying_type(), Subdomain<K>::_map[i].first, 0, Subdomain<K>::_communicator, Subdomain<K>::_rq + i);
                Wrapper<underlying_type<K>>::gthr(Subdomain<K>::_map[i].second.size(), d, send, Subdomain<K>::_map[i].second.data());
                MPI_Isend(send, Subdomain<K>::_map[i].second.size(), Wrapper<K>::mpi_underlying_type(), Subdomain<K>::_map[i].first, 0, Subdomain<K>::_communicator, Subdomain<K>::_rq + size + i);
            }
            std::fill_n(d, Subdomain<K>::_dof, 1.0);
            for(unsigned short i = 0, size = Subdomain<K>::_map.size(); i < size; ++i) {
                int index;
                MPI_Waitany(size, Subdomain<K>::_rq, &index, MPI_STATUS_IGNORE);
                underlying_type<K>* const recv = reinterpret_cast<underlying_type<K>*>(Subdomain<K>::_buff[index]);
                underlying_type<K>* const send = reinterpret_cast<underlying_type<K>*>(Subdomain<K>::_buff[size + index]);
                for(unsigned int j = 0; j < Subdomain<K>::_map[index].second.size(); ++j) {
                    if(std::abs(send[j]) < HPDDM_EPS)
                        d[Subdomain<K>::_map[index].second[j]] = 0.0;
                    else
                        d[Subdomain<K>::_map[index].second[j]] /= 1.0 + d[Subdomain<K>::_map[index].second[j]] * recv[j] / send[j];
                }
            }
            MPI_Waitall(Subdomain<K>::_map.size(), Subdomain<K>::_rq + Subdomain<K>::_map.size(), MPI_STATUSES_IGNORE);
            Subdomain<K>::clearBuffer(allocate);
        }
        /* Function: deflation
         *
         *  Computes a coarse correction.
         *
         * Template parameter:
         *    excluded       - True if the master processes are excluded from the domain decomposition, false otherwise. 
         *
         * Parameters:
         *    in             - Input vectors.
         *    out            - Output vectors.
         *    mu             - Number of vectors. */
        template<bool excluded>
        void deflation(const K* const in, K* const out, const unsigned short& mu) const {
            if(excluded)
                super::_co->template callSolver<excluded>(super::_uc, mu);
            else {
                Wrapper<K>::diag(Subdomain<K>::_dof, _d, in, out, mu);                                                                                                                                                            // out = D in
                int tmp = mu;
                int local = super::getLocal();
                if(local)
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &local, &tmp, &(Subdomain<K>::_dof), &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), out, &(Subdomain<K>::_dof), &(Wrapper<K>::d__0), super::_uc, &local); // _uc = _ev^T D in
                super::_co->template callSolver<excluded>(super::_uc, mu);                                                                                                                                                        // _uc = E \ _ev^T D in
                if(local)
                    Blas<K>::gemm("N", "N", &(Subdomain<K>::_dof), &tmp, &local, &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), super::_uc, &local, &(Wrapper<K>::d__0), out, &(Subdomain<K>::_dof));                   // out = _ev E \ _ev^T D in
                exchange(out, mu);
            }
        }
#if HPDDM_ICOLLECTIVE
        /* Function: Ideflation
         *
         *  Computes the first part of a coarse correction asynchronously.
         *
         * Template parameter:
         *    excluded       - True if the master processes are excluded from the domain decomposition, false otherwise. 
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector.
         *    rq             - MPI request to check completion of the MPI transfers. */
        template<bool excluded>
        void Ideflation(const K* const in, K* const out, const unsigned short& mu, MPI_Request* rq) const {
            if(excluded)
                super::_co->template IcallSolver<excluded>(super::_uc, mu, rq);
            else {
                Wrapper<K>::diag(Subdomain<K>::_dof, _d, in, out, mu);
                int tmp = mu;
                Blas<K>::gemm(&(Wrapper<K>::transc), "N", super::getAddrLocal(), &tmp, &(Subdomain<K>::_dof), &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), out, &(Subdomain<K>::_dof), &(Wrapper<K>::d__0), super::_uc, super::getAddrLocal());
                super::_co->template IcallSolver<excluded>(super::_uc, mu, rq);
            }
        }
#endif // HPDDM_ICOLLECTIVE
        /* Function: buildTwo
         *
         *  Assembles and factorizes the coarse operator by calling <Preconditioner::buildTwo>.
         *
         * Template Parameter:
         *    excluded       - Greater than 0 if the master processes are excluded from the domain decomposition, equal to 0 otherwise.
         *
         * Parameter:
         *    comm           - Global MPI communicator.
         *
         * See also: <Bdd::buildTwo>, <Feti::buildTwo>. */
        template<unsigned short excluded = 0>
        std::pair<MPI_Request, const K*>* buildTwo(const MPI_Comm& comm, MatrixCSR<K>* const& A = nullptr) {
#if HPDDM_INEXACT_COARSE_OPERATOR
            Option& opt = *Option::get();
            const std::string prefix = super::prefix();
            if(opt.val<unsigned short>(prefix + "level_2_aggregate_size") != 1)
                opt.remove(prefix + "level_2_schwarz_method");
#endif
            std::pair<MPI_Request, const K*>* ret = super::template buildTwo<excluded, MatrixMultiplication<Schwarz<Solver, CoarseSolver, S, K>, K>>(this, comm);
#if HPDDM_INEXACT_COARSE_OPERATOR
            if(super::_co) {
                super::_co->setParent(this);
                const unsigned short p = opt.val<unsigned short>(prefix + "level_2_p", 1);
                const unsigned short method = opt.val<unsigned short>(prefix + "level_2_schwarz_method", HPDDM_SCHWARZ_METHOD_NONE);
                if(super::_co && p > 1 && opt.val<unsigned short>(prefix + "level_2_aggregate_size", p) == 1 && (method == HPDDM_SCHWARZ_METHOD_RAS || method == HPDDM_SCHWARZ_METHOD_ASM) && opt.val<char>(prefix + "level_2_krylov_method", HPDDM_KRYLOV_METHOD_GMRES) != HPDDM_KRYLOV_METHOD_NONE) {
                    CoarseOperator<CoarseSolver, S, K>* coNeumann  = nullptr;
                    std::vector<K> overlap;
                    std::vector<std::vector<std::pair<unsigned short, unsigned short>>> reduction;
                    std::map<std::pair<unsigned short, unsigned short>, unsigned short> sizes;
                    std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>> extra;
                    MPI_Request rs = MPI_REQUEST_NULL;
                    if(opt.set(prefix + "level_2_schwarz_coarse_correction") && A) {
                        MatrixCSR<K>* backup = Subdomain<K>::_a;
                        Subdomain<K>::_a = A;
                        coNeumann = new CoarseOperator<CoarseSolver, S, K>;
                        std::string filename = opt.prefix(prefix + "level_2_dump_matrix", true);
                        std::string filenameNeumann = std::string("-hpddm_") + prefix + std::string("level_2_dump_matrix ") + filename + std::string("_Neumann");
                        if(filename.size() > 0)
                            opt.parse(filenameNeumann);
                        if(!A->_ia && !A->_ja && A->_nnz == backup->_nnz) {
                            A->_ia = backup->_ia;
                            A->_ja = backup->_ja;
                        }
                        super::template buildTwo<excluded, MatrixAccumulation<Schwarz<Solver, CoarseSolver, S, K>, K>>(this, comm, coNeumann, overlap, reduction, sizes, extra);
                        if(A->_ia == backup->_ia && A->_ja == backup->_ja && A->_nnz == backup->_nnz) {
                            A->_ia = nullptr;
                            A->_ja = nullptr;
                        }
                        if(overlap.size())
                            MPI_Isend(overlap.data(), overlap.size(), Wrapper<K>::mpi_type(), 0, 300, coNeumann->getCommunicator(), &rs);
                        if(filename.size() > 0) {
                            filenameNeumann = std::string("-hpddm_") + prefix + std::string("level_2_dump_matrix ") + filename;
                            opt.parse(filenameNeumann);
                        }
                        Subdomain<K>::_a = backup;
                    }
                    opt.setPrefix(prefix + "level_2_");
                    super::_co->buildThree(coNeumann, reduction, sizes, extra);
                    delete coNeumann;
                    MPI_Wait(&rs, MPI_STATUS_IGNORE);
                }
                opt.setPrefix(std::string());
            }
#endif
            return ret;
        }
        template<bool excluded = false>
        bool start(const K* const b, K* const x, const unsigned short& mu = 1) const {
            bool allocate = Subdomain<K>::setBuffer();
            if(!excluded && Subdomain<K>::_a->_ia) {
                const std::unordered_map<unsigned int, K> map = Subdomain<K>::boundaryConditions();
                for(const std::pair<unsigned int, K>& p : map)
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        x[nu * Subdomain<K>::_dof + p.first] = b[nu * Subdomain<K>::_dof + p.first] / p.second;
            }
            exchange(x, mu);
            if(super::_co) {
                unsigned short k = 1;
                const std::string prefix = super::prefix();
                Option& opt = *Option::get();
                if(opt.any_of(prefix + "krylov_method", { HPDDM_KRYLOV_METHOD_GCRODR, HPDDM_KRYLOV_METHOD_BGCRODR }) && !opt.val<unsigned short>(prefix + "recycle_same_system"))
                    k = std::max(opt.val<int>(prefix + "recycle", 1), 1);
                super::start(mu * k);
            }
            return allocate;
        }
        /* Function: apply
         *
         *  Applies the global Schwarz preconditioner.
         *
         * Template Parameter:
         *    excluded       - Greater than 0 if the master processes are excluded from the domain decomposition, equal to 0 otherwise.
         *
         * Parameters:
         *    in             - Input vectors, modified internally if no workspace array is specified !
         *    out            - Output vectors.
         *    mu             - Number of vectors.
         *    work           - Workspace array. */
        template<bool excluded = false>
        void apply(const K* const in, K* const out, const unsigned short& mu = 1, K* work = nullptr) const {
            const char correction = Option::get()->val<char>(super::prefix("schwarz_coarse_correction"), -1);
            if(!super::_co || correction == -1) {
                if(_type == Prcndtnr::NO)
                    std::copy_n(in, mu * Subdomain<K>::_dof, out);
                else if(_type == Prcndtnr::GE || _type == Prcndtnr::OG) {
                    if(!excluded) {
                        super::_s.solve(in, out, mu);
                        exchange(out, mu);               // out = D A \ in
                    }
                }
                else {
                    if(!excluded) {
                        if(_type == Prcndtnr::OS) {
                            Wrapper<K>::diag(Subdomain<K>::_dof, _d, in, out, mu);
                            super::_s.solve(out, mu);
                            Wrapper<K>::diag(Subdomain<K>::_dof, _d, out, mu);
                        }
                        else
                            super::_s.solve(in, out, mu);
                        Subdomain<K>::exchange(out, mu); // out = A \ in
                    }
                }
            }
            else {
                int n = mu * Subdomain<K>::_dof;
                if(!work)
                    work = const_cast<K*>(in);
                else if(!excluded)
                    std::copy_n(in, n, work);
                if(correction == HPDDM_SCHWARZ_COARSE_CORRECTION_ADDITIVE) {
#if HPDDM_ICOLLECTIVE
                    MPI_Request rq[2];
                    Ideflation<excluded>(in, out, mu, rq);
                    if(!excluded) {
                        super::_s.solve(work, mu); // out = A \ in
                        MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
                        const int k = mu;
                        Blas<K>::gemm("N", "N", &(Subdomain<K>::_dof), &k, super::getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), super::_uc, super::getAddrLocal(), &(Wrapper<K>::d__0), out, &(Subdomain<K>::_dof)); // out = _ev E \ _ev^T D in
                        Blas<K>::axpy(&n, &(Wrapper<K>::d__1), work, &i__1, out, &i__1);
                        exchange(out, mu); // out = Z E \ Z^T in + A \ in
                    }
                    else
                        MPI_Wait(rq + 1, MPI_STATUS_IGNORE);
#else
                    deflation<excluded>(in, out, mu);
                    if(!excluded) {
                        super::_s.solve(work, mu);
                        Blas<K>::axpy(&n, &(Wrapper<K>::d__1), work, &i__1, out, &i__1);
                        exchange(out, mu);
                    }
#endif // HPDDM_ICOLLECTIVE
                }
                else {
                    deflation<excluded>(in, out, mu);                                                                  // out = Z E \ Z^T in
                    if(!excluded) {
                        if(!Subdomain<K>::_a->_ia && !Subdomain<K>::_a->_ja) {
                            K* tmp = new K[mu * Subdomain<K>::_dof];
                            GMV(out, tmp, mu);
                            Blas<K>::axpby(n, -1.0, tmp, 1, 1.0, work, 1);
                            delete [] tmp;
                        }
                        else {
                            if(HPDDM_NUMBERING == Wrapper<K>::I)
                                Wrapper<K>::csrmm("N", &(Subdomain<K>::_dof), &(n = mu), &(Subdomain<K>::_dof), &(Wrapper<K>::d__2), Subdomain<K>::_a->_sym, Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, out, &(Wrapper<K>::d__1), work);
                            else if(Subdomain<K>::_a->_ia[Subdomain<K>::_dof] == Subdomain<K>::_a->_nnz)
                                Wrapper<K>::template csrmm<'C'>("N", &(Subdomain<K>::_dof), &(n = mu), &(Subdomain<K>::_dof), &(Wrapper<K>::d__2), Subdomain<K>::_a->_sym, Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, out, &(Wrapper<K>::d__1), work);
                            else
                                Wrapper<K>::template csrmm<'F'>("N", &(Subdomain<K>::_dof), &(n = mu), &(Subdomain<K>::_dof), &(Wrapper<K>::d__2), Subdomain<K>::_a->_sym, Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, out, &(Wrapper<K>::d__1), work);
                        }
                        exchange(work, mu);                                                                            //  in = (I - A Z E \ Z^T) in
                        if(_type == Prcndtnr::OS)
                            Wrapper<K>::diag(Subdomain<K>::_dof, _d, work, mu);
                        super::_s.solve(work, mu);
                        exchange(work, mu);                                                                            //  in = D A \ (I - A Z E \ Z^T) in
                        n = mu * Subdomain<K>::_dof;
                        if(correction == HPDDM_SCHWARZ_COARSE_CORRECTION_BALANCED) {
                            if(!excluded) {
                                K* tmp = new K[n];
                                GMV(work, tmp, mu);
                                deflation<excluded>(nullptr, tmp, mu);
                                Blas<K>::axpy(&n, &(Wrapper<K>::d__2), tmp, &i__1, out, &i__1);
                            }
                            else
                                deflation<excluded>(nullptr, nullptr, mu);
                        }
                        Blas<K>::axpy(&n, &(Wrapper<K>::d__1), work, &i__1, out, &i__1);   // out = D A \ (I - A Z E \ Z^T) in + Z E \ Z^T in
                    }
                }
            }
        }
        /* Function: scaleIntoOverlap
         *
         *  Scales the input matrix using <Schwarz::d> on the overlap and sets the output matrix to zero elsewhere.
         *
         * Parameters:
         *    A              - Input matrix.
         *    B              - Output matrix used in GenEO.
         *
         * See also: <Schwarz::solveGEVP>. */
        template<char N = HPDDM_NUMBERING>
        void scaleIntoOverlap(const MatrixCSR<K>* const& A, MatrixCSR<K>*& B) const {
            std::set<unsigned int> intoOverlap;
            for(const pairNeighbor& neighbor : Subdomain<K>::_map)
                for(unsigned int i : neighbor.second)
                    if(_d[i] > HPDDM_EPS)
                        intoOverlap.insert(i);
            std::vector<std::vector<std::pair<unsigned int, K>>> tmp(intoOverlap.size());
            unsigned int k, iPrev = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, HPDDM_GRANULARITY) reduction(+ : iPrev)
#endif
            for(k = 0; k < intoOverlap.size(); ++k) {
                auto it = std::next(intoOverlap.cbegin(), k);
                tmp[k].reserve(A->_ia[*it + 1] - A->_ia[*it]);
                for(unsigned int j = A->_ia[*it] - (N == 'F'); j < A->_ia[*it + 1] - (N == 'F'); ++j) {
                    K value = _d[*it] * _d[A->_ja[j] - (N == 'F')] * A->_a[j];
                    if(std::abs(value) > HPDDM_EPS && intoOverlap.find(A->_ja[j] - (N == 'F')) != intoOverlap.cend())
                        tmp[k].emplace_back(A->_ja[j], value);
                }
                iPrev += tmp[k].size();
            }
            int nnz = iPrev;
            if(B)
                delete B;
            B = new MatrixCSR<K>(Subdomain<K>::_dof, Subdomain<K>::_dof, nnz, A->_sym);
            nnz = iPrev = k = 0;
            for(unsigned int i : intoOverlap) {
                std::fill(B->_ia + iPrev, B->_ia + i + 1, nnz + (N == 'F'));
                for(const std::pair<unsigned int, K>& p : tmp[k]) {
                    B->_ja[nnz] = p.first;
                    B->_a[nnz++] = p.second;
                }
                ++k;
                iPrev = i + 1;
            }
            std::fill(B->_ia + iPrev, B->_ia + Subdomain<K>::_dof + 1, nnz + (N == 'F'));
        }
        /* Function: solveGEVP
         *
         *  Solves the generalized eigenvalue problem Ax = l Bx.
         *
         * Parameters:
         *    A              - Left-hand side matrix.
         *    B              - Right-hand side matrix (optional). */
        template<template<class> class Eps>
        void solveGEVP(MatrixCSR<K>* const& A, MatrixCSR<K>* const& B = nullptr, const MatrixCSR<K>* const& pattern = nullptr) {
            Option& opt = *Option::get();
            const bool resetPrefix = (opt.getPrefix().size() == 0 && super::prefix().size() != 0);
            if(resetPrefix)
                opt.setPrefix(super::prefix());
            const underlying_type<K>& threshold = opt.val("geneo_threshold", 0.0);
            Eps<K> evp(threshold, Subdomain<K>::_dof, opt.val<unsigned short>("geneo_nu", 20));
#ifndef PY_MAJOR_VERSION
            bool free = pattern ? pattern->sameSparsity(A) : Subdomain<K>::_a->sameSparsity(A);
#else
            constexpr bool free = false;
#endif
            MatrixCSR<K>* rhs = nullptr;
            if(B)
                rhs = B;
            else
                scaleIntoOverlap(A, rhs);
            if(super::_ev) {
                if(*super::_ev)
                    delete [] *super::_ev;
                delete [] super::_ev;
            }
#if defined(MUMPSSUB) || defined(MKL_PARDISOSUB)
            if(threshold > 0.0 && opt.val<char>("geneo_estimate_nu", 0) && (!B || B->hashIndices() == A->hashIndices())) {
                K* difference = new K[A->_nnz];
                std::copy_n(A->_a, A->_nnz, difference);
                for(unsigned int i = 0; i < A->_n; ++i) {
                    int* it = A->_ja + A->_ia[i];
                    for(unsigned int j = rhs->_ia[i]; j < rhs->_ia[i + 1]; ++j) {
                        it = std::lower_bound(it, A->_ja + A->_ia[i + 1], rhs->_ja[j]);
                        difference[std::distance(A->_ja, it++)] -= threshold * rhs->_a[j];
                    }
                }
                std::swap(A->_a, difference);
                Solver<K> s;
                evp._nu = std::max(1, s.inertia(A));
                std::swap(A->_a, difference);
                delete [] difference;
            }
#endif
            evp.template solve<Solver>(A, rhs, super::_ev, Subdomain<K>::_communicator, free ? &(super::_s) : nullptr);
            if(rhs != B)
                delete rhs;
            if(free && A->getFree()) {
                A->_ia = nullptr;
                A->_ja = nullptr;
            }
            opt["geneo_nu"] = evp._nu;
            if(super::_co)
                super::_co->setLocal(evp._nu);
            const int n = Subdomain<K>::_dof;
            std::for_each(super::_ev, super::_ev + evp._nu, [&](K* const v) { std::replace_if(v, v + n, [](K x) { return std::abs(x) < 1.0 / (HPDDM_EPS * HPDDM_PEN); }, K()); });
            if(resetPrefix)
                opt.setPrefix("");
        }
        template<bool sorted = true, bool scale = false>
        void interaction(std::vector<const MatrixCSR<K>*>& blocks) const {
            Subdomain<K>::template interaction<HPDDM_NUMBERING, sorted, scale>(blocks, _d);
        }
        /* Function: GMV
         *
         *  Computes a global sparse matrix-vector product.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector. */
#if HPDDM_DENSE
        virtual void GMV(const K* const in, K* const out, const int& mu = 1) const = 0;
#else
        void GMV(const K* const in, K* const out, const int& mu = 1, MatrixCSR<K>* const& A = nullptr) const {
#if 0
            K* tmp = new K[mu * Subdomain<K>::_dof];
            Wrapper<K>::diag(Subdomain<K>::_dof, _d, in, tmp, mu);
            if(HPDDM_NUMBERING == Wrapper<K>::I)
                Wrapper<K>::csrmm(Subdomain<K>::_a->_sym, &(Subdomain<K>::_dof), &mu, Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, tmp, out);
            else if(Subdomain<K>::_a->_ia[Subdomain<K>::_dof] == Subdomain<K>::_a->_nnz)
                Wrapper<K>::template csrmm<'C'>(Subdomain<K>::_a->_sym, &(Subdomain<K>::_dof), &mu, Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, tmp, out);
            else
                Wrapper<K>::template csrmm<'F'>(Subdomain<K>::_a->_sym, &(Subdomain<K>::_dof), &mu, Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, tmp, out);
            delete [] tmp;
            Subdomain<K>::exchange(out, mu);
#else
            if(A)
                Wrapper<K>::csrmm(A->_sym, &A->_n, &mu, A->_a, A->_ia, A->_ja, in, out);
            else if(HPDDM_NUMBERING == Wrapper<K>::I)
                Wrapper<K>::csrmm(Subdomain<K>::_a->_sym, &(Subdomain<K>::_dof), &mu, Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, in, out);
            else if(Subdomain<K>::_a->_ia[Subdomain<K>::_dof] == Subdomain<K>::_a->_nnz)
                Wrapper<K>::template csrmm<'C'>(Subdomain<K>::_a->_sym, &(Subdomain<K>::_dof), &mu, Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, in, out);
            else
                Wrapper<K>::template csrmm<'F'>(Subdomain<K>::_a->_sym, &(Subdomain<K>::_dof), &mu, Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, in, out);
            exchange(out, mu);
#endif
        }
#endif
        /* Function: computeResidual
         *
         *  Computes the norms of right-hand sides and residual vectors.
         *
         * Parameters:
         *    x              - Solution vector.
         *    f              - Right-hand side.
         *    storage        - Array to store both values.
         *    mu             - Number of vectors.
         *    norm           - l^2, l^1, or l^\infty norm.
         *
         * See also: <Schur::computeResidual>. */
        void computeResidual(const K* const x, const K* const f, underlying_type<K>* const storage, const unsigned short mu = 1, const unsigned short norm = HPDDM_COMPUTE_RESIDUAL_L2) const {
            int dim = mu * Subdomain<K>::_dof;
            K* tmp = new K[dim];
            bool allocate = Subdomain<K>::setBuffer();
            GMV(x, tmp, mu);
            Subdomain<K>::clearBuffer(allocate);
            Blas<K>::axpy(&dim, &(Wrapper<K>::d__2), f, &i__1, tmp, &i__1);
            std::fill_n(storage, 2 * mu, 0.0);
            if(norm == HPDDM_COMPUTE_RESIDUAL_L1) {
                for(unsigned int i = 0; i < Subdomain<K>::_dof; ++i) {
                    bool boundary = (std::abs(Subdomain<K>::boundaryCond(i)) > HPDDM_EPS);
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        if(!boundary)
                            storage[2 * nu + 1] += _d[i] * std::abs(tmp[nu * Subdomain<K>::_dof + i]);
                        if(std::abs(f[nu * Subdomain<K>::_dof + i]) > HPDDM_EPS * HPDDM_PEN)
                            storage[2 * nu] += _d[i] * std::abs(f[nu * Subdomain<K>::_dof + i] / underlying_type<K>(HPDDM_PEN));
                        else
                            storage[2 * nu] += _d[i] * std::abs(f[nu * Subdomain<K>::_dof + i]);
                    }
                }
            }
            else if(norm == HPDDM_COMPUTE_RESIDUAL_LINFTY) {
                for(unsigned int i = 0; i < Subdomain<K>::_dof; ++i) {
                    bool boundary = (std::abs(Subdomain<K>::boundaryCond(i)) > HPDDM_EPS);
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        if(!boundary)
                            storage[2 * nu + 1] = std::max(std::abs(tmp[nu * Subdomain<K>::_dof + i]), storage[2 * nu + 1]);
                        if(std::abs(f[nu * Subdomain<K>::_dof + i]) > HPDDM_EPS * HPDDM_PEN)
                            storage[2 * nu] = std::max(std::abs(f[nu * Subdomain<K>::_dof + i] / underlying_type<K>(HPDDM_PEN)), storage[2 * nu]);
                        else
                            storage[2 * nu] = std::max(std::abs(f[nu * Subdomain<K>::_dof + i]), storage[2 * nu]);
                    }
                }
            }
            else {
                for(unsigned int i = 0; i < Subdomain<K>::_dof; ++i) {
                    bool boundary = (std::abs(Subdomain<K>::boundaryCond(i)) > HPDDM_EPS);
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        if(!boundary)
                            storage[2 * nu + 1] += _d[i] * std::norm(tmp[nu * Subdomain<K>::_dof + i]);
                        if(std::abs(f[nu * Subdomain<K>::_dof + i]) > HPDDM_EPS * HPDDM_PEN)
                            storage[2 * nu] += _d[i] * std::norm(f[nu * Subdomain<K>::_dof + i] / underlying_type<K>(HPDDM_PEN));
                        else
                            storage[2 * nu] += _d[i] * std::norm(f[nu * Subdomain<K>::_dof + i]);
                    }
                }
            }
            delete [] tmp;
            if(norm == HPDDM_COMPUTE_RESIDUAL_L2 || norm == HPDDM_COMPUTE_RESIDUAL_L1) {
                MPI_Allreduce(MPI_IN_PLACE, storage, 2 * mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, Subdomain<K>::_communicator);
                if(norm == HPDDM_COMPUTE_RESIDUAL_L2)
                    std::for_each(storage, storage + 2 * mu, [](underlying_type<K>& b) { b = std::sqrt(b); });
            }
            else
                MPI_Allreduce(MPI_IN_PLACE, storage, 2 * mu, Wrapper<K>::mpi_underlying_type(), MPI_MAX, Subdomain<K>::_communicator);
        }
#endif
        /* Function: getScaling
         *  Returns a constant pointer to <Schwarz::d>. */
        const underlying_type<K>* getScaling() const { return _d; }
        template<char N = HPDDM_NUMBERING>
        void distributedNumbering(unsigned int* const in, unsigned int& first, unsigned int& last, unsigned int& global) const {
            Subdomain<K>::template globalMapping<N>(in, in + Subdomain<K>::_dof, first, last, global, _d);
        }
        template<class T = K>
        bool distributedCSR(unsigned int* const num, unsigned int first, unsigned int last, int*& ia, int*& ja, T*& c) const {
            return Subdomain<K>::distributedCSR(num, first, last, ia, ja, c, Subdomain<K>::_a);
        }
};

template<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
    template<class> class Solver, template<class> class CoarseSolver,
#endif
    char S, class K>
struct hpddm_method_id<Schwarz<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
    Solver, CoarseSolver,
#endif
    S, K>> { static constexpr char value = 1; };
} // HPDDM
#endif // _HPDDM_SCHWARZ_
