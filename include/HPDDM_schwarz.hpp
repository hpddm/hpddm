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

#if HPDDM_PETSC
#include "HPDDM_dmatrix.hpp"
#include "HPDDM_coarse_operator_impl.hpp"
# if HPDDM_SLEPC
#  include "HPDDM_operator.hpp"
# endif
#endif

#include "HPDDM_preconditioner.hpp"

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
    template<class> class Solver, template<class> class CoarseSolver, char S,
#endif
    class K>
class Schwarz : public Preconditioner<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
                Solver, CoarseOperator<CoarseSolver, S, K>,
#elif HPDDM_PETSC
                CoarseOperator<DMatrix, K>,
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
        void operator=(const Schwarz& B) {
            dtor();
            Subdomain<K>::_a = B._a ? new MatrixCSR<K>(*B._a) : nullptr;
            Subdomain<K>::_buff = new K*[2 * B._map.size()];
            Subdomain<K>::_map = B._map;
            Subdomain<K>::_rq = new MPI_Request[2 * B._map.size()];
            Subdomain<K>::_communicator = B._communicator;
            Subdomain<K>::_dof = B._dof;
            _d = B._d;
        }
        void dtor() {
            super::super::dtor();
            super::dtor();
            _d = nullptr;
        }
        /* Typedef: super
         *  Type of the immediate parent class <Preconditioner>. */
        typedef Preconditioner<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
            Solver, CoarseOperator<CoarseSolver, S, K>,
#elif HPDDM_PETSC
            CoarseOperator<DMatrix, K>,
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
            const std::string prefix = super::prefix();
            const bool fact = super::setMatrix(a) && !Option::get()->any_of(prefix + "schwarz_method", { HPDDM_SCHWARZ_METHOD_ORAS, HPDDM_SCHWARZ_METHOD_SORAS, HPDDM_SCHWARZ_METHOD_OSM, HPDDM_SCHWARZ_METHOD_NONE });
            if(fact)
                callNumfact(a);
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
        typename super::co_type::return_type buildTwo(const MPI_Comm& comm, MatrixCSR<K>* const& A = nullptr) {
#if HPDDM_INEXACT_COARSE_OPERATOR
            Option& opt = *Option::get();
            const std::string prefix = super::prefix();
            if(opt.val<unsigned short>(prefix + "level_2_aggregate_size") != 1)
                opt.remove(prefix + "level_2_schwarz_method");
#endif
            auto ret = super::template buildTwo<excluded, MatrixMultiplication<Schwarz<Solver, CoarseSolver, S, K>, K>>(this, comm);
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
         *    in             - Input vectors, modified internally if no workspace array is specified!
         *    out            - Output vectors.
         *    mu             - Number of vectors.
         *    work           - Workspace array. */
        template<bool excluded = false>
        void apply(const K* const in, K* const out, const unsigned short& mu = 1, K* work = nullptr) const {
            const char correction = Option::get()->val<char>(super::prefix("schwarz_coarse_correction"), -1);
            if((!super::_co && !super::_cc) || correction == -1) {
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
                        exchange(out, mu);                                               // out = Z E \ Z^T in + A \ in
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
                    deflation<excluded>(in, out, mu);                                    // out = Z E \ Z^T in
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
                        exchange(work, mu);                                              //  in = (I - A Z E \ Z^T) in
                        if(_type == Prcndtnr::OS)
                            Wrapper<K>::diag(Subdomain<K>::_dof, _d, work, mu);
                        super::_s.solve(work, mu);
                        exchange(work, mu);                                              //  in = D A \ (I - A Z E \ Z^T) in
                        n = mu * Subdomain<K>::_dof;
                        if(correction == HPDDM_SCHWARZ_COARSE_CORRECTION_BALANCED) {
                            if(!excluded) {
                                K* tmp = new K[super::_cc ? 2 * n : n];
                                GMV(work, tmp, mu);
                                if(super::_cc) {
                                    deflation<excluded>(tmp, tmp + n, mu);
                                    Blas<K>::axpy(&n, &(Wrapper<K>::d__2), tmp + n, &i__1, work, &i__1);
                                }
                                else {
                                    deflation<excluded>(nullptr, tmp, mu);
                                    Blas<K>::axpy(&n, &(Wrapper<K>::d__2), tmp, &i__1, work, &i__1);
                                }
                                delete [] tmp;
                            }
                            else
                                deflation<excluded>(nullptr, nullptr, mu);
                        }
                        Blas<K>::axpy(&n, &(Wrapper<K>::d__1), work, &i__1, out, &i__1); // out = D A \ (I - A Z E \ Z^T) in + Z E \ Z^T in
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
#ifdef PETSCHPDDM_H
        static PetscErrorCode destroy(PC_HPDDM_Level* const ctx, PetscBool all) {
            PetscFunctionBeginUser;
            if(!ctx)
                SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "PCSHELL from PCHPDDM called with no context");
            static_assert(std::is_same<K, PetscScalar>::value, "Wrong type");
            if(ctx->P && ctx->P->_dof != -1) {
                if(all == PETSC_TRUE) {
                    if(!std::is_same<K, PetscReal>::value)
                        delete [] ctx->P->_d;
                    if(ctx->parent->levels[0] == ctx)
                        MPI_Comm_free(&ctx->P->_communicator);
                    delete ctx->P;
                    ctx->P = nullptr;
                }
                else {
                    ctx->P->super::dtor();
                    ctx->P->clearBuffer();
                }
            }
            PetscFunctionReturn(0);
        }
#endif
#if HPDDM_SLEPC
    private:
        HPDDM_HAS_MEMBER(Neumann)
        template<class Q>
        static constexpr typename std::enable_if<!has_Neumann<Q>::value, bool>::type Neumann(const Q parent) {
            return false;
        }
        template<class Q>
        static constexpr typename std::enable_if<has_Neumann<Q>::value, bool>::type Neumann(const Q parent) {
            return parent->Neumann == PETSC_TRUE;
        }
    public:
        typename super::co_type::return_type buildTwo(const MPI_Comm& comm, Mat D, PetscInt n, PetscInt M, PC_HPDDM_Level** const levels) {
            PetscErrorCode ierr;

            PetscFunctionBeginUser;
            ierr = super::template buildTwo<false, MatrixMultiplication<Schwarz<K>, K>>(this, comm, D, n, M, levels);
            PetscFunctionReturn(ierr);
        }
        PetscErrorCode initialize(const IS interior, IS is, const Mat D, Mat N, std::vector<Vec> initial, PetscInt* const n, PC_HPDDM_Level** const levels) {
            Mat                    P, local;
            ISLocalToGlobalMapping l2g;
            PetscReal              *d;
            const PetscInt         *ptr;
            PetscInt               m, bs;
            PetscBool              ismatis, solve = PETSC_FALSE, sym;
            PetscErrorCode         ierr;

            PetscFunctionBeginUser;
            ierr = PetscObjectTypeCompare((PetscObject)N, MATIS, &ismatis);CHKERRQ(ierr);
            if(!Subdomain<K>::_rq) {
                MPI_Comm_dup(PetscObjectComm((PetscObject)levels[0]->ksp), &(Subdomain<K>::_communicator));
                ierr = ISGetLocalSize(is, &m);CHKERRQ(ierr);
                Subdomain<K>::_dof = m;
                if(!std::is_same<PetscScalar, PetscReal>::value)
                    d = new PetscReal[Subdomain<K>::_dof];
                else
                    VecGetArray(levels[0]->D, reinterpret_cast<PetscScalar**>(&d));
                _d = d;
                ierr = ISGetIndices(is, &ptr);CHKERRQ(ierr);
                std::vector<std::pair<PetscInt, PetscInt>> v;
                if(ismatis) {
                    ierr = MatGetLocalToGlobalMapping(N, &l2g, nullptr);CHKERRQ(ierr);
                    ierr = ISLocalToGlobalMappingGetBlockSize(l2g, &bs);CHKERRQ(ierr);
                }
                else {
                    ierr = KSPGetOperators(levels[0]->ksp, nullptr, &P);CHKERRQ(ierr);
                    ierr = MatGetBlockSize(P, &bs);CHKERRQ(ierr);
                }
                if(Subdomain<K>::_dof % bs) {
                    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible local size %d and Pmat block size %D", Subdomain<K>::_dof, bs);
                }
                v.reserve(Subdomain<K>::_dof / bs);
                for(PetscInt i = 0; i < Subdomain<K>::_dof; i += bs)
                    v.emplace_back(std::make_pair(ptr[i], i));
                std::sort(v.begin(), v.end());
                if(ismatis) {
                    PetscInt nproc;
                    PetscInt* procs;
                    PetscInt* numprocs;
                    PetscInt** indices;
                    std::fill_n(d, Subdomain<K>::_dof, 0.0);
                    ierr = ISLocalToGlobalMappingGetBlockInfo(l2g, &nproc, &procs, &numprocs, &indices);CHKERRQ(ierr);
                    std::unordered_map<PetscInt, std::set<PetscInt>> boundary;
                    const PetscInt* idx;
                    ierr = ISLocalToGlobalMappingGetBlockIndices(l2g, &idx);CHKERRQ(ierr);
                    ierr = ISLocalToGlobalMappingGetSize(l2g, &m);CHKERRQ(ierr);
                    for(PetscInt i = 0; i < m / bs; ++i) {
                        std::vector<std::pair<PetscInt, PetscInt>>::const_iterator it = std::lower_bound(v.cbegin(), v.cend(), std::make_pair(bs * idx[i], static_cast<PetscInt>(0)));
                        if(it != v.cend() && it->first == bs * idx[i]) {
                            solve = PETSC_TRUE;
                            std::fill_n(d + it->second, bs, 1.0);
                        }
                    }
                    for(PetscInt i = 1; i < nproc; ++i) {
                        for(PetscInt j = 0; j < numprocs[i]; ++j) {
                            std::vector<std::pair<PetscInt, PetscInt>>::const_iterator it = std::lower_bound(v.cbegin(), v.cend(), std::make_pair(bs * idx[indices[i][j]], static_cast<PetscInt>(0)));
                            boundary[it->second].insert(procs[i]);
                        }
                    }
                    ierr = ISLocalToGlobalMappingRestoreBlockIndices(l2g, &idx);CHKERRQ(ierr);
                    PetscInt* sorted = new PetscInt[nproc - 1];
                    std::copy_n(procs + 1, nproc - 1, sorted);
                    Subdomain<K>::_map.resize(nproc - 1);
                    ierr = ISLocalToGlobalMappingRestoreBlockInfo(l2g, &nproc, &procs, &numprocs, &indices);CHKERRQ(ierr);
                    std::sort(sorted, sorted + Subdomain<K>::_map.size());
                    Subdomain<K>::_buff = new K*[2 * Subdomain<K>::_map.size()]();
                    Subdomain<K>::_rq = new MPI_Request[2 * Subdomain<K>::_map.size()];
                    std::unordered_set<PetscInt>* map = new std::unordered_set<PetscInt>[Subdomain<K>::_map.size()];
                    ierr = PetscObjectTypeCompare((PetscObject)D, MATSEQSBAIJ, &sym);CHKERRQ(ierr);
                    if(!sym) {
                        for(const std::pair<PetscInt, std::set<PetscInt>>& p : boundary) {
                            PetscInt ncols;
                            const PetscInt *cols;
                            ierr = MatGetRow(D, p.first, &ncols, &cols, nullptr);CHKERRQ(ierr);
                            for(const PetscInt& i : p.second) {
                                PetscInt* pos = std::lower_bound(sorted, sorted + Subdomain<K>::_map.size(), i);
                                map[std::distance(sorted, pos)].insert(cols, cols + ncols);
                            }
                            ierr = MatRestoreRow(D, p.first, &ncols, &cols, nullptr);CHKERRQ(ierr);
                            std::fill_n(d + p.first, bs, 1.0 / underlying_type<K>(1 + p.second.size()));
                        }
                    }
                    else {
                        ierr = MatSetOption(D, MAT_GETROW_UPPERTRIANGULAR, PETSC_TRUE);CHKERRQ(ierr);
                        for(PetscInt i = 0; i < Subdomain<K>::_dof; i += bs) {
                            PetscInt ncols;
                            const PetscInt *cols;
                            ierr = MatGetRow(D, i, &ncols, &cols, nullptr);CHKERRQ(ierr);
                            std::unordered_map<PetscInt, std::set<PetscInt>>::const_iterator it = boundary.find(i);
                            if(it != boundary.cend()) {
                                for(const PetscInt& i : it->second) {
                                    PetscInt* pos = std::lower_bound(sorted, sorted + Subdomain<K>::_map.size(), i);
                                    map[std::distance(sorted, pos)].insert(cols, cols + ncols);
                                }
                                std::fill_n(d + it->first, bs, 1.0 / underlying_type<K>(1 + it->second.size()));
                            }
                            for(PetscInt j = 0; j < ncols; j += bs) {
                                it = boundary.find(cols[j]);
                                if(it != boundary.cend()) {
                                    for(const PetscInt& k : it->second) {
                                        PetscInt* pos = std::lower_bound(sorted, sorted + Subdomain<K>::_map.size(), k);
                                        for(PetscInt j = 0; j < bs; ++j)
                                            map[std::distance(sorted, pos)].insert(i + j);
                                    }
                                }
                            }
                            ierr = MatRestoreRow(D, i, &ncols, &cols, nullptr);CHKERRQ(ierr);
                        }
                        ierr = MatSetOption(D, MAT_GETROW_UPPERTRIANGULAR, PETSC_FALSE);CHKERRQ(ierr);
                    }
                    for(unsigned short i = 0; i < Subdomain<K>::_map.size(); ++i) {
                        Subdomain<K>::_map[i].first = sorted[i];
                        Subdomain<K>::_map[i].second.reserve(map[i].size());
                        std::copy(map[i].cbegin(), map[i].cend(), std::back_inserter(Subdomain<K>::_map[i].second));
                        std::sort(Subdomain<K>::_map[i].second.begin(), Subdomain<K>::_map[i].second.end(), [&ptr] (const PetscInt& lhs, const PetscInt& rhs) { return ptr[lhs] < ptr[rhs]; });
                    }
                    ierr = ISRestoreIndices(is, &ptr);CHKERRQ(ierr);
                    delete [] sorted;
                    delete [] map;
                }
                else {
                    ierr = ISRestoreIndices(is, &ptr);CHKERRQ(ierr);
                    ierr = ISGetLocalSize(interior, &m);CHKERRQ(ierr);
                    ierr = ISGetIndices(interior, &ptr);CHKERRQ(ierr);
                    PetscInt rstart = m ? ptr[0] : 0;
                    PetscInt rend = rstart + m;
                    ierr = ISRestoreIndices(interior, &ptr);CHKERRQ(ierr);
                    ierr = ISGetIndices(is, &ptr);CHKERRQ(ierr);
                    PetscLayout rmap;
                    ierr = MatGetLayouts(P, &rmap, nullptr);CHKERRQ(ierr);
                    std::map<PetscMPIInt, std::set<PetscInt>> exchange;
                    for(PetscInt i = 0; i < Subdomain<K>::_dof; i += bs) {
                        if(ptr[i] >= rstart && ptr[i] < rend) {
                            std::fill_n(d + i, bs, 1.0);
                            solve = PETSC_TRUE;
                        }
                        else {
                            std::fill_n(d + i, bs, 0.0);
                            PetscMPIInt owner;
                            ierr = PetscLayoutFindOwner(rmap, ptr[i], &owner);CHKERRQ(ierr);
                            exchange[owner].insert(ptr[i]);
                        }
                    }
                    ierr = ISRestoreIndices(is, &ptr);CHKERRQ(ierr);
                    PetscMPIInt nfrom, *fromranks;
                    PetscInt *fromdata;
                    {
                        std::pair<std::vector<PetscMPIInt>, std::vector<PetscInt>> to;
                        to.first.reserve(exchange.size());
                        to.second.reserve(exchange.size());
                        for(const auto& p : exchange) {
                            to.first.emplace_back(p.first);
                            to.second.emplace_back(p.second.size());
                        }
                        ierr = PetscCommBuildTwoSided(PetscObjectComm((PetscObject)levels[0]->ksp), 1, MPIU_INT, exchange.size(), to.first.data(), to.second.data(), &nfrom, &fromranks, &fromdata);CHKERRQ(ierr);
                    }
                    for(PetscMPIInt i = 0; i < nfrom; ++i) {
                        if(exchange.find(fromranks[i]) == exchange.cend())
                            exchange[fromranks[i]] = std::set<PetscInt>();
                    }
                    Subdomain<K>::_map.resize(exchange.size());
                    Subdomain<K>::_buff = new K*[2 * Subdomain<K>::_map.size()]();
                    Subdomain<K>::_rq = new MPI_Request[2 * Subdomain<K>::_map.size()];
                    PetscInt i = 0;
                    for(const auto& p : exchange) {
                        i += p.second.size();
                        PetscMPIInt* const it = std::find(fromranks, fromranks + nfrom, p.first);
                        if(it != fromranks + nfrom)
                            i += fromdata[std::distance(fromranks, it)];
                    }
                    PetscInt** const buff = reinterpret_cast<PetscInt**>(Subdomain<K>::_buff);
                    *buff = new PetscInt[i];
                    i = 0;
                    PetscMPIInt j = 0;
                    for(const auto& p : exchange) {
                        Subdomain<K>::_map[j].first = p.first;
                        buff[j] = *buff + i;
                        std::copy(p.second.cbegin(), p.second.cend(), buff[j]);
                        MPI_Isend(buff[j], p.second.size(), MPIU_INT, p.first, 123, Subdomain<K>::_communicator, Subdomain<K>::_rq + j);
                        i += p.second.size();
                        buff[exchange.size() + j] = *buff + i;
                        PetscMPIInt* const it = std::find(fromranks, fromranks + nfrom, p.first);
                        if(it != fromranks + nfrom) {
                            i += fromdata[std::distance(fromranks, it)];
                            MPI_Irecv(buff[exchange.size() + j], fromdata[std::distance(fromranks, it)], MPIU_INT, p.first, 123, Subdomain<K>::_communicator, Subdomain<K>::_rq + exchange.size() + j);
                        }
                        else
                            MPI_Irecv(buff[exchange.size() + j], 0, MPIU_INT, p.first, 123, Subdomain<K>::_communicator, Subdomain<K>::_rq + exchange.size() + j);
                        ++j;
                    }
                    for(i = 0; i < exchange.size(); ++i) {
                        MPI_Waitany(exchange.size(), Subdomain<K>::_rq + exchange.size(), &j, MPI_STATUS_IGNORE);
                        std::map<PetscMPIInt, std::set<PetscInt>>::iterator it = exchange.find(Subdomain<K>::_map[j].first);
                        PetscMPIInt* const pt = std::find(fromranks, fromranks + nfrom, it->first);
                        if(pt != fromranks + nfrom) {
                            for(PetscInt k = 0; k < fromdata[std::distance(fromranks, pt)]; ++k)
                                it->second.insert(buff[exchange.size() + j][k]);
                        }
                        Subdomain<K>::_map[j].second.reserve(bs * it->second.size());
                        for(const PetscInt n : it->second) {
                            std::vector<std::pair<PetscInt, PetscInt>>::const_iterator it = std::lower_bound(v.begin(), v.end(), std::make_pair(n, static_cast<PetscInt>(0)));
                            for(PetscInt k = 0; k < bs; ++k)
                                Subdomain<K>::_map[j].second.emplace_back(it->second + k);
                        }
                    }
                    ierr = PetscFree(fromranks);CHKERRQ(ierr);
                    MPI_Waitall(exchange.size(), Subdomain<K>::_rq, MPI_STATUSES_IGNORE);
                    delete [] *buff;
                    *buff = nullptr;
                    ierr = PetscFree(fromdata);CHKERRQ(ierr);
                }
                if(!std::is_same<PetscScalar, PetscReal>::value) {
                    PetscScalar* c;
                    ierr = VecGetArray(levels[0]->D, &c);CHKERRQ(ierr);
                    std::copy_n(d, Subdomain<K>::_dof, c);
                }
                ierr = VecRestoreArray(levels[0]->D, nullptr);CHKERRQ(ierr);
            }
            else {
                if(Subdomain<K>::_dof == -1) {
                    PetscInt n;
                    VecGetLocalSize(levels[0]->D, &n);
                    Subdomain<K>::_dof = n;
                }
                for(int i = 0; i < Subdomain<K>::_dof && !solve; ++i)
                    if(std::abs(1.0 - _d[i]) < HPDDM_EPS)
                        solve = PETSC_TRUE;
                if(!std::is_same<PetscScalar, PetscReal>::value) {
                    PetscScalar* c;
                    ierr = VecGetArray(levels[0]->D, &c);CHKERRQ(ierr);
                    std::copy_n(_d, Subdomain<K>::_dof, c);
                    ierr = VecRestoreArray(levels[0]->D, nullptr);CHKERRQ(ierr);
                }
            }
            EPS eps;
            ST st;
            Mat weighted = nullptr, *resized;
            Vec vr, vreduced;
            IS sub[1] = { };
            PetscInt nev, nconv = 0;
            const char* prefix;
            PetscInt mbs;
            const PetscInt *ia, *ja;
            PetscScalar *v = nullptr;
            unsigned short type = 0;
            if(Neumann(levels[0]->parent) || ismatis) {
                if(!std::string(reinterpret_cast<PetscObject>(D)->type_name).compare("seqaij")) {
                    bs = 1;
                    ierr = MatSeqAIJGetArray(D, &v);CHKERRQ(ierr);
                }
#if PETSC_VERSION_GE(3, 12, 2)
                else if(!std::string(reinterpret_cast<PetscObject>(D)->type_name).compare("seqbaij")) {
                    ierr = MatGetBlockSize(D, &bs);CHKERRQ(ierr);
                    ierr = MatSeqBAIJGetArray(D, &v);CHKERRQ(ierr);
                    type = 1;
                }
#endif
                else if(!std::string(reinterpret_cast<PetscObject>(D)->type_name).compare("seqsbaij")) {
                    ierr = MatGetBlockSize(D, &bs);CHKERRQ(ierr);
                    ierr = MatSeqSBAIJGetArray(D, &v);CHKERRQ(ierr);
                    type = 2;
                }
                if(v) {
                    PetscBool done;
                    ierr = MatGetRowIJ(D, 0, PETSC_FALSE, !type ? PETSC_FALSE : PETSC_TRUE, &mbs, &ia, &ja, &done);CHKERRQ(ierr);
                    if(!done) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatGetRowIJ did not return appropriate arrays");
                }
            }
            if(!v) {
                ierr = MatConvert(D, MATSAME, MAT_INITIAL_MATRIX, &weighted);CHKERRQ(ierr);
                ierr = MatDiagonalScale(weighted, levels[0]->D, levels[0]->D);CHKERRQ(ierr);
            }
            else {
                PetscScalar* b = new PetscScalar[ia[mbs] * bs * bs];
                for(PetscInt i = 0; i < mbs; ++i) {
                    for(PetscInt j = ia[i]; j < ia[i + 1]; ++j) {
                        const underlying_type<K> scal = _d[i * bs] * _d[ja[j] * bs];
                        std::transform(v + j * bs * bs, v + (j + 1) * bs * bs, b + j * bs * bs, [&scal](PetscScalar x) { return x * scal; });
                    }
                }
                ierr = MatCreate(PETSC_COMM_SELF, &weighted);CHKERRQ(ierr);
                ierr = MatSetBlockSize(weighted, bs);CHKERRQ(ierr);
                ierr = MatSetSizes(weighted, mbs * bs, mbs * bs, mbs * bs, mbs * bs);CHKERRQ(ierr);
                if(!type) {
                    ierr = MatSetType(weighted, MATSEQAIJ);CHKERRQ(ierr);
                    ierr = MatSeqAIJSetPreallocationCSR(weighted, ia, ja, b);CHKERRQ(ierr);
                }
                else if(type == 1) {
                    ierr = MatSetType(weighted, MATSEQBAIJ);CHKERRQ(ierr);
                    ierr = MatSeqBAIJSetPreallocationCSR(weighted, bs, ia, ja, b);CHKERRQ(ierr);
                }
                else {
                    ierr = MatSetType(weighted, MATSEQSBAIJ);CHKERRQ(ierr);
                    ierr = MatSeqSBAIJSetPreallocationCSR(weighted, bs, ia, ja, b);CHKERRQ(ierr);
                }
                PetscBool done;
                ierr = MatRestoreRowIJ(D, 0, PETSC_FALSE, !type ? PETSC_FALSE : PETSC_TRUE, nullptr, &ia, &ja, &done);CHKERRQ(ierr);
                v = nullptr;
                delete [] b;
            }
            ierr = KSPGetOptionsPrefix(levels[0]->ksp, &prefix);CHKERRQ(ierr);
            ierr = EPSCreate(PETSC_COMM_SELF, &eps);CHKERRQ(ierr);
            ierr = EPSSetOptionsPrefix(eps, prefix);CHKERRQ(ierr);
            if(!ismatis) {
                ierr = EPSSetOperators(eps, N, weighted);CHKERRQ(ierr);
            }
            else {
                ierr = MatGetLocalToGlobalMapping(N, &l2g, nullptr);CHKERRQ(ierr);
                ierr = ISGlobalToLocalMappingApplyIS(l2g, IS_GTOLM_DROP, is, &sub[0]);CHKERRQ(ierr);
                ierr = ISDestroy(&is);CHKERRQ(ierr);
                ierr = MatCreateSubMatrices(weighted, 1, sub, sub, MAT_INITIAL_MATRIX, &resized);CHKERRQ(ierr);
                ierr = MatDestroy(&weighted);CHKERRQ(ierr);
                ierr = MatISGetLocalMat(N, &local);CHKERRQ(ierr);
                ierr = PetscObjectTypeCompare((PetscObject)local, MATSEQSBAIJ, &sym);CHKERRQ(ierr);
                if(sym) {
                    /* going back from SEQBAIJ to SEQSBAIJ */
                    ierr = MatSetOption(resized[0], MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
                    ierr = MatConvert(resized[0], MATSEQSBAIJ, MAT_INPLACE_MATRIX, &resized[0]);CHKERRQ(ierr);
                }
                ierr = EPSSetOperators(eps, local, resized[0]);CHKERRQ(ierr);
                if(!initial.empty()) {
                    std::vector<Vec> full = initial;
                    for(PetscInt i = 0; i < full.size(); ++i) {
                        ierr = MatCreateVecs(resized[0], &initial[i], nullptr);CHKERRQ(ierr);
                        ierr = VecISCopy(full[i], sub[0], SCATTER_REVERSE, initial[i]);CHKERRQ(ierr);
                        ierr = VecDestroy(&full[i]);CHKERRQ(ierr);
                    }
                }
            }
            ierr = EPSSetProblemType(eps, EPS_GHEP);CHKERRQ(ierr);
            ierr = EPSSetTarget(eps, 0.0);CHKERRQ(ierr);
            ierr = EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE);CHKERRQ(ierr);
            ierr = EPSGetST(eps, &st);CHKERRQ(ierr);
            ierr = STSetType(st, STSINVERT);CHKERRQ(ierr);
            if(Neumann(levels[0]->parent) || ismatis) {
                ierr = STSetMatStructure(st, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
            }
            ierr = EPSSetInitialSpace(eps, initial.size(), initial.data());CHKERRQ(ierr);
            std::for_each(initial.begin(), initial.end(), [&](Vec v) { VecDestroy(&v); });
            std::vector<Vec>().swap(initial);
            ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);
            ierr = EPSGetDimensions(eps, &nev, nullptr, nullptr);CHKERRQ(ierr);
            if(solve) {
                MatStructure str;
                ierr = STGetMatStructure(st, &str);CHKERRQ(ierr);
                if(str != SAME_NONZERO_PATTERN) {
                    ierr = PetscInfo2(st, "HPDDM: The MatStructure of the GenEO eigenproblem stencil is set to %d, -%sst_matstructure same is preferred depending on what is passed to PCHPDDMSetAuxiliaryMat\n", int(str), prefix);CHKERRQ(ierr);
                }
                ierr = EPSSolve(eps);CHKERRQ(ierr);
                ierr = EPSGetConverged(eps, &nconv);CHKERRQ(ierr);
            }
            levels[0]->nu = std::min(nconv, nev);
            if(levels[0]->threshold >= 0.0) {
                PetscInt i = 0;
                while(i < levels[0]->nu) {
                    PetscScalar eigr;
                    ierr = EPSGetEigenvalue(eps, i, &eigr, nullptr);CHKERRQ(ierr);
                    if(std::abs(eigr) > levels[0]->threshold) {
                      ierr = PetscInfo1(eps, "HPDDM: Discarding eigenvalue %g\n", double(std::abs(eigr)));CHKERRQ(ierr);
                        break;
                    }
                    else {
                      ierr = PetscInfo1(eps, "HPDDM: Using eigenvalue %g\n", double(std::abs(eigr)));CHKERRQ(ierr);
                    }
                    ++i;
                }
                levels[0]->nu = i;
            }
            ierr = PetscInfo2(eps, "HPDDM: Using %D out of %D computed eigenvectors\n", levels[0]->nu, nconv);CHKERRQ(ierr);
            if(levels[0]->nu) {
                super::_ev = new K*[levels[0]->nu];
                *super::_ev = new K[Subdomain<K>::_dof * levels[0]->nu]();
            }
            for(unsigned short i = 1; i < levels[0]->nu; ++i)
                super::_ev[i] = *super::_ev + i * Subdomain<K>::_dof;
            ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, Subdomain<K>::_dof, levels[0]->nu ? super::_ev[0] : nullptr, &vr);CHKERRQ(ierr);
            if(ismatis)
                ierr = MatCreateVecs(resized[0], &vreduced, nullptr);CHKERRQ(ierr);
            for(PetscInt i = 0; i < levels[0]->nu; ++i) {
                ierr = VecPlaceArray(vr, super::_ev[i]);CHKERRQ(ierr);
                if(!ismatis) {
                    ierr = EPSGetEigenvector(eps, i, vr, nullptr);CHKERRQ(ierr);
                }
                else {
                    ierr = EPSGetEigenvector(eps, i, vreduced, nullptr);CHKERRQ(ierr);
                    ierr = VecISCopy(vr, sub[0], SCATTER_FORWARD, vreduced);CHKERRQ(ierr);
                }
                ierr = VecResetArray(vr);CHKERRQ(ierr);
            }
            ierr = VecDestroy(&vr);CHKERRQ(ierr);
            ierr = EPSDestroy(&eps);CHKERRQ(ierr);
            if(!ismatis) {
                ierr = MatDestroy(&weighted);CHKERRQ(ierr);
            }
            else {
                ierr = VecDestroy(&vreduced);CHKERRQ(ierr);
                ierr = ISDestroy(&sub[0]);CHKERRQ(ierr);
                ierr = MatDestroySubMatrices(1, &resized);CHKERRQ(ierr);
                ierr = MatISRestoreLocalMat(N, &local);CHKERRQ(ierr);
            }
            Schwarz<PetscScalar>* decomposition = this;
            Mat A = D;
            if(ismatis) {
                P = N;
                ierr = MatISGetLocalMat(P, &N);CHKERRQ(ierr);
            }
            for(unsigned short i = 1; i < *n; ++i) {
                char fail[2] { };
                if(decomposition) {
                    ierr = decomposition->buildTwo(decomposition->getCommunicator(), A, i - 1, *n, levels);
                    if(ierr != PETSC_ERR_ARG_WRONG && levels[i]->ksp) {
                        Mat A;
                        PetscInt rstart, rend;
                        ierr = KSPGetOperators(levels[i]->ksp, &A, nullptr);CHKERRQ(ierr);
                        ierr = MatGetOwnershipRange(A, &rstart, &rend);CHKERRQ(ierr);
                        if(rstart == rend)
                            fail[1] = 1;
                    }
                }
                else
                    ierr = PetscErrorCode(0);
                fail[0] = (ierr == PETSC_ERR_ARG_WRONG ? 1 : 0);
                MPI_Allreduce(MPI_IN_PLACE, &fail, 2, MPI_CHAR, MPI_MAX, PetscObjectComm((PetscObject)(levels[0]->ksp)));
                if(fail[0]) { /* building level i + 1 failed because there was no deflation vector */
                    *n = i;
                    if(i > 1) {
                        ierr = MatDestroy(&N);CHKERRQ(ierr);
                    }
                }
                else {
                    CHKERRQ(ierr);
                }
                if(i > 1) {
                    ierr = MatDestroy(&A);CHKERRQ(ierr);
                }
                if(fail[1]) { /* cannot build level i + 1 because at least one subdomain is empty */
                    *n = i + 1;
                    break;
                }
                if(i + 1 < *n && decomposition) {
                    CoarseOperator<DMatrix, K>* coNeumann  = nullptr;
                    std::vector<K> overlap;
                    std::vector<std::vector<std::pair<unsigned short, unsigned short>>> reduction;
                    std::map<std::pair<unsigned short, unsigned short>, unsigned short> sizes;
                    std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>> extra;
                    MPI_Request rs = MPI_REQUEST_NULL;
                    coNeumann = new CoarseOperator<DMatrix, K>;
                    super::template buildTwo<false, MatrixAccumulation<Schwarz<K>, K>>(decomposition, decomposition->getCommunicator(), N, i - 1, *n, levels, coNeumann, overlap, reduction, sizes, extra);
                    if(i > 1) {
                        ierr = MatDestroy(&N);CHKERRQ(ierr);
                    }
                    if(overlap.size())
                        MPI_Isend(overlap.data(), overlap.size(), Wrapper<K>::mpi_type(), 0, 300, coNeumann->getCommunicator(), &rs);
                    ierr = decomposition->_co->buildThree(coNeumann, reduction, sizes, extra, &A, &N, levels[i]);CHKERRQ(ierr);
                    delete coNeumann;
                    if(i + 2 == *n) {
                        ierr = MatDestroy(&N);CHKERRQ(ierr);
                    }
                    if(A)
                        decomposition = decomposition->_co->getSubdomain()->P;
                    else
                        decomposition = nullptr;
                    MPI_Wait(&rs, MPI_STATUS_IGNORE);
                }
            }
            if(ismatis) {
                ierr = MatDestroy(&P);CHKERRQ(ierr);
            }
            for(unsigned short i = 0; i < *n - 1; ++i)
                if(levels[i]->P) {
                    levels[i]->P->setBuffer();
                    levels[i]->P->super::start();
                }
            PetscFunctionReturn(0);
        }
#endif
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
            if(super::_cc) {
                (*super::_cc)(in, out, Subdomain<K>::_dof, mu);
                return;
            }
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
    template<class> class Solver, template<class> class CoarseSolver, char S,
#endif
    class K>
struct hpddm_method_id<Schwarz<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
    Solver, CoarseSolver, S,
#endif
    K>> { static constexpr char value = 1; };
} // HPDDM
#if HPDDM_SLEPC
PETSC_EXTERN PetscErrorCode PCHPDDM_Internal(HPDDM::Schwarz<PetscScalar>* const P, IS const interior, IS is, Mat const D, Mat const N, std::vector<Vec> initial, PetscInt* const n, PC_HPDDM_Level** const levels) {
    if(!P) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "PCHPDDM_Internal called with no HPDDM object");
    if(!n) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "PCHPDDM_Internal called without a number of levels");
    else return P->initialize(interior, is, D, N, initial, n, levels);
}
#endif
#endif // _HPDDM_SCHWARZ_
