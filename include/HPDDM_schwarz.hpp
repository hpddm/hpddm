/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
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

#ifndef HPDDM_SCHWARZ_HPP_
#define HPDDM_SCHWARZ_HPP_

#if HPDDM_PETSC
#include "HPDDM_dmatrix.hpp"
#include "HPDDM_coarse_operator_impl.hpp"
# if HPDDM_SLEPC
PETSC_EXTERN PetscLogEvent PC_HPDDM_PtAP;
PETSC_EXTERN PetscLogEvent PC_HPDDM_PtBP;
PETSC_EXTERN PetscLogEvent PC_HPDDM_Next;
#  include "HPDDM_operator.hpp"
typedef struct _n_Aux *Aux;
struct _n_Aux {
    Mat V;
    Vec sigma;
    IS  is;
};
static PetscErrorCode MatMult_Aux(Mat, Vec, Vec);
typedef struct _n_Sum *Sum;
struct _n_Sum {
    Mat                       A[3];
    void*                     decomposition;
    std::set<unsigned short>* work;
    char                      status;
};
static PetscErrorCode MatMult_Sum(Mat, Vec, Vec);
static PetscErrorCode MatMult_Harmonic(Mat, Vec, Vec);
static PetscErrorCode MatDestroy_Harmonic(Mat);
# endif
# if defined(PETSC_HAVE_HTOOL) && HPDDM_SLEPC
#  include <petscmathtool.h>
#  include <htool/solvers/coarse_space.hpp>
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
        const underlying_type<K>* d_;
#if HPDDM_SCHWARZ
        std::size_t            hash_;
        /* Variable: type
         *  Type of <Prcndtnr> used in <Schwarz::apply> and <Schwarz::deflation>. */
        Prcndtnr               type_;
#endif
    public:
        Schwarz() : d_()
#if HPDDM_SCHWARZ
                        , hash_(), type_(Prcndtnr::NO)
#endif
                                                       { }
        explicit Schwarz(const Subdomain<K>& s) : super(s), d_()
#if HPDDM_SCHWARZ
                                                                , hash_(), type_(Prcndtnr::NO)
#endif
                                                                                               { }
#if HPDDM_DENSE
        virtual
#endif
        ~Schwarz() { d_ = nullptr; }
        void operator=(const Schwarz& B) {
            dtor();
            Subdomain<K>::a_ = B.a_ ? new MatrixCSR<K>(*B.a_) : nullptr;
            Subdomain<K>::buff_ = new K*[2 * B.map_.size()]();
            Subdomain<K>::map_ = B.map_;
            Subdomain<K>::rq_ = new MPI_Request[2 * B.map_.size()];
            Subdomain<K>::communicator_ = B.communicator_;
            Subdomain<K>::dof_ = B.dof_;
            d_ = B.d_;
        }
        void dtor() {
            super::super::dtor();
            super::dtor();
            d_ = nullptr;
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
            d_ = d;
        }
        /* Function: exchange */
        template<bool allocate = false>
        void exchange(K* const x, const unsigned short& mu = 1) const {
            bool free = false;
            if(allocate)
                free = Subdomain<K>::setBuffer();
            Wrapper<K>::diag(Subdomain<K>::dof_, d_, x, mu);
            Subdomain<K>::exchange(x, mu);
            if(allocate)
                Subdomain<K>::clearBuffer(free);
        }
#if !HPDDM_PETSC
        void exchange() const {
            std::vector<K>* send = new std::vector<K>[Subdomain<K>::map_.size()];
            unsigned int* sizes = new unsigned int[Subdomain<K>::map_.size()]();
            std::vector<std::pair<int, int>>* transpose = nullptr;
            if(Subdomain<K>::a_->sym_) {
                transpose = new std::vector<std::pair<int, int>>[Subdomain<K>::dof_]();
                for(int i = 0; i < Subdomain<K>::dof_; ++i)
                    for(int j = Subdomain<K>::a_->ia_[i] - (HPDDM_NUMBERING == 'F'); j < Subdomain<K>::a_->ia_[i + 1] - (HPDDM_NUMBERING == 'F'); ++j)
                        transpose[Subdomain<K>::a_->ja_[j] - (HPDDM_NUMBERING == 'F')].emplace_back(i, j);
            }
            for(unsigned short i = 0, size = Subdomain<K>::map_.size(); i < size; ++i) {
                const pairNeighbor& pair = Subdomain<K>::map_[i];
                std::vector<int> idx(pair.second.size());
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&pair](int lhs, int rhs) { return pair.second[lhs] < pair.second[rhs]; });
                for(unsigned int j = 0; j < pair.second.size(); ++j) {
                    unsigned int nnz = 0, n = send[i].size() + 1;
                    if(d_[pair.second[j]] > HPDDM_EPS) {
                        send[i].emplace_back(j);
                        send[i].emplace_back(0);
                    }
                    sizes[i] += 2;
                    std::vector<int>::const_iterator it = idx.cbegin();
                    for(int k = Subdomain<K>::a_->ia_[pair.second[j]] - (HPDDM_NUMBERING == 'F'); k < Subdomain<K>::a_->ia_[pair.second[j] + 1] - (HPDDM_NUMBERING == 'F'); ++k) {
                        it = std::lower_bound(it, idx.cend(), Subdomain<K>::a_->ja_[k] - (HPDDM_NUMBERING == 'F'), [&pair](int lhs, int rhs) { return pair.second[lhs] < rhs; });
                        if(it != idx.cend() && pair.second[*it] == Subdomain<K>::a_->ja_[k] - (HPDDM_NUMBERING == 'F')) {
                            if(d_[pair.second[j]] > HPDDM_EPS) {
                                send[i].emplace_back(*it);
                                send[i].emplace_back(Subdomain<K>::a_->a_[k]);
                                ++nnz;
                            }
                            sizes[i] += 2;
                        }
                    }
                    if(Subdomain<K>::a_->sym_) {
                        for(int k = 0; k < transpose[pair.second[j]].size(); ++k) {
                            std::vector<int>::const_iterator first = std::lower_bound(it, idx.cend(), transpose[pair.second[j]][k].first, [&pair](int lhs, int rhs) { return pair.second[lhs] < rhs; });
                            if(first != idx.cend() && pair.second[*first] == transpose[pair.second[j]][k].first) {
                                if(d_[pair.second[j]] > HPDDM_EPS) {
                                    send[i].emplace_back(*first);
                                    send[i].emplace_back(Subdomain<K>::a_->a_[transpose[pair.second[j]][k].second]);
                                    ++nnz;
                                }
                                sizes[i] += 2;
                            }
                        }
                    }
                    if(d_[pair.second[j]] > HPDDM_EPS)
                        send[i][n] = nnz;
                }
                MPI_Isend(send[i].data(), send[i].size(), Wrapper<K>::mpi_type(), Subdomain<K>::map_[i].first, 13, Subdomain<K>::communicator_, Subdomain<K>::rq_ + size + i);
            }
            delete [] transpose;
            K** recv = new K*[Subdomain<K>::map_.size()]();
            for(unsigned short i = 0, size = Subdomain<K>::map_.size(); i < size; ++i) {
                if(sizes[i]) {
                    recv[i] = new K[sizes[i]];
                    MPI_Irecv(recv[i], sizes[i], Wrapper<K>::mpi_type(), Subdomain<K>::map_[i].first, 13, Subdomain<K>::communicator_, Subdomain<K>::rq_ + i);
                }
                else
                    Subdomain<K>::rq_[i] = MPI_REQUEST_NULL;
            }
            for(unsigned short i = 0, size = Subdomain<K>::map_.size(); i < size; ++i) {
                int index;
                MPI_Status st;
                MPI_Waitany(size, Subdomain<K>::rq_, &index, &st);
                if(st.MPI_SOURCE != MPI_ANY_SOURCE && st.MPI_TAG == 13) {
                    int size;
                    MPI_Get_count(&st, Wrapper<K>::mpi_type(), &size);
                    for(unsigned int j = 0; j < size; ) {
                        const unsigned int row = Subdomain<K>::map_[index].second[std::lround(std::abs(recv[index][j]))];
                        const unsigned int nnz = std::lround(std::abs(recv[index][j + 1]));
                        j += 2;
                        for(unsigned int k = 0; k < nnz; ++k, j += 2) {
                            int* const pt = std::lower_bound(Subdomain<K>::a_->ja_ + Subdomain<K>::a_->ia_[row] - (HPDDM_NUMBERING == 'F'), Subdomain<K>::a_->ja_ + Subdomain<K>::a_->ia_[row + 1] - (HPDDM_NUMBERING == 'F'), Subdomain<K>::map_[index].second[std::lround(std::abs(recv[index][j]))] + (HPDDM_NUMBERING == 'F'));
                            if(pt != Subdomain<K>::a_->ja_ + Subdomain<K>::a_->ia_[row + 1] - (HPDDM_NUMBERING == 'F') && *pt == Subdomain<K>::map_[index].second[std::lround(std::abs(recv[index][j]))] + (HPDDM_NUMBERING == 'F'))
                                Subdomain<K>::a_->a_[std::distance(Subdomain<K>::a_->ja_, pt)] = recv[index][j + 1];
                        }
                    }
                }
            }
            std::for_each(recv, recv + Subdomain<K>::map_.size(), std::default_delete<K[]>());
            delete [] recv;
            MPI_Waitall(Subdomain<K>::map_.size(), Subdomain<K>::rq_ + Subdomain<K>::map_.size(), MPI_STATUSES_IGNORE);
            delete [] send;
            delete [] sizes;
        }
#endif
        bool restriction(underlying_type<K>* const D) const {
            unsigned int n = 0;
            for(const auto& i : Subdomain<K>::map_)
                n += i.second.size();
            std::vector<int> p;
            if(n && !Subdomain<K>::map_.empty()) {
                unsigned int** idx = new unsigned int*[Subdomain<K>::map_.size()];
                underlying_type<K>** const buff = new underlying_type<K>*[2 * Subdomain<K>::map_.size()];
                *idx = new unsigned int[n];
                *buff = new underlying_type<K>[2 * n];
                buff[Subdomain<K>::map_.size()] = *buff + n;
                n = 0;
                for(unsigned short i = 0, size = Subdomain<K>::map_.size(); i < size; ++i) {
                    idx[i] = *idx + n;
                    buff[i] = *buff + n;
                    buff[size + i] = buff[size] + n;
                    MPI_Irecv(buff[i], Subdomain<K>::map_[i].second.size(), Wrapper<K>::mpi_underlying_type(), Subdomain<K>::map_[i].first, 0, Subdomain<K>::communicator_, Subdomain<K>::rq_ + i);
                    Wrapper<underlying_type<K>>::gthr(Subdomain<K>::map_[i].second.size(), D, buff[size + i], Subdomain<K>::map_[i].second.data());
                    MPI_Isend(buff[size + i], Subdomain<K>::map_[i].second.size(), Wrapper<K>::mpi_underlying_type(), Subdomain<K>::map_[i].first, 0, Subdomain<K>::communicator_, Subdomain<K>::rq_ + size + i);
                    n += Subdomain<K>::map_[i].second.size();
                }
                for(unsigned short i = 0, size = Subdomain<K>::map_.size(); i < size; ++i) {
                    const std::vector<int>& v = Subdomain<K>::map_[i].second;
                    std::iota(idx[i], idx[i] + v.size(), 0);
                    std::sort(idx[i], idx[i] + v.size(), [&v] (const unsigned int& lhs, const unsigned int& rhs) { return v[lhs] < v[rhs]; });
                }
                underlying_type<K>* const d = new underlying_type<K>[Subdomain<K>::dof_];
                std::copy_n(D, Subdomain<K>::dof_, d);
                for(unsigned short i = 0, size = Subdomain<K>::map_.size(); i < size; ++i) {
                    int index;
                    MPI_Waitany(size, Subdomain<K>::rq_, &index, MPI_STATUS_IGNORE);
                    for(int j = 0; j < Subdomain<K>::map_[index].second.size(); ++j)
                        d[Subdomain<K>::map_[index].second[j]] += buff[index][j];
                }
                p.reserve(Subdomain<K>::dof_);
                for(int i = 0; i < Subdomain<K>::dof_; ++i)
                    if((HPDDM::abs(D[i] - 1.0) > HPDDM_EPS && HPDDM::abs(D[i]) > HPDDM_EPS) || HPDDM::abs(d[i] - 1.0) > HPDDM_EPS)
                        p.emplace_back(i);
                delete [] d;
                int rank;
                MPI_Comm_rank(Subdomain<K>::communicator_, &rank);
                for(int k = 0; k < p.size(); ++k) {
                    bool largest = true;
                    for(unsigned short i = 0, size = Subdomain<K>::map_.size(); i < size && largest; ++i) {
                        const std::vector<int>& v = Subdomain<K>::map_[i].second;
                        unsigned int* const it = std::lower_bound(idx[i], idx[i] + v.size(), p[k], [&v](unsigned int lhs, int rhs) { return v[lhs] < rhs; });
                        if(std::distance(idx[i], it) != v.size() && v[*it] == p[k]) {
                            const underlying_type<K> v = D[p[k]] - buff[i][*it];
                            largest = (v > HPDDM_EPS || (HPDDM::abs(v) < HPDDM_EPS && rank > Subdomain<K>::map_[i].first));
                        }
                    }
                    D[p[k]] = (largest ? 1.0 : 0.0);
                }
                MPI_Waitall(Subdomain<K>::map_.size(), Subdomain<K>::rq_ + Subdomain<K>::map_.size(), MPI_STATUSES_IGNORE);
                delete [] *buff;
                delete [] buff;
                delete [] *idx;
                delete [] idx;
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
                if(hash_ != hash) {
                    hash_ = hash;
                    super::destroySolver();
                }
            }
            switch(m) {
                case HPDDM_SCHWARZ_METHOD_SORAS: type_ = (A ? Prcndtnr::OS : Prcndtnr::SY); break;
                case HPDDM_SCHWARZ_METHOD_ASM:   type_ = Prcndtnr::SY; break;
                case HPDDM_SCHWARZ_METHOD_NONE:  type_ = Prcndtnr::NO; return;
                default:                         type_ = (A && (m == HPDDM_SCHWARZ_METHOD_ORAS || m == HPDDM_SCHWARZ_METHOD_OSM) ? Prcndtnr::OG : Prcndtnr::GE);
            }
            m = opt.val<unsigned short>("reuse_preconditioner");
            if(m <= 1)
                super::s_.template numfact<N>(type_ == Prcndtnr::OS || type_ == Prcndtnr::OG ? A : Subdomain<K>::a_);
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
            for(unsigned short i = 0, size = Subdomain<K>::map_.size(); i < size; ++i) {
                underlying_type<K>* const recv = reinterpret_cast<underlying_type<K>*>(Subdomain<K>::buff_[i]);
                underlying_type<K>* const send = reinterpret_cast<underlying_type<K>*>(Subdomain<K>::buff_[size + i]);
                MPI_Irecv(recv, Subdomain<K>::map_[i].second.size(), Wrapper<K>::mpi_underlying_type(), Subdomain<K>::map_[i].first, 0, Subdomain<K>::communicator_, Subdomain<K>::rq_ + i);
                Wrapper<underlying_type<K>>::gthr(Subdomain<K>::map_[i].second.size(), d, send, Subdomain<K>::map_[i].second.data());
                MPI_Isend(send, Subdomain<K>::map_[i].second.size(), Wrapper<K>::mpi_underlying_type(), Subdomain<K>::map_[i].first, 0, Subdomain<K>::communicator_, Subdomain<K>::rq_ + size + i);
            }
            std::fill_n(d, Subdomain<K>::dof_, 1.0);
            for(unsigned short i = 0, size = Subdomain<K>::map_.size(); i < size; ++i) {
                int index;
                MPI_Waitany(size, Subdomain<K>::rq_, &index, MPI_STATUS_IGNORE);
                underlying_type<K>* const recv = reinterpret_cast<underlying_type<K>*>(Subdomain<K>::buff_[index]);
                underlying_type<K>* const send = reinterpret_cast<underlying_type<K>*>(Subdomain<K>::buff_[size + index]);
                for(unsigned int j = 0; j < Subdomain<K>::map_[index].second.size(); ++j) {
                    if(std::abs(send[j]) < HPDDM_EPS)
                        d[Subdomain<K>::map_[index].second[j]] = 0.0;
                    else
                        d[Subdomain<K>::map_[index].second[j]] /= 1.0 + d[Subdomain<K>::map_[index].second[j]] * recv[j] / send[j];
                }
            }
            MPI_Waitall(Subdomain<K>::map_.size(), Subdomain<K>::rq_ + Subdomain<K>::map_.size(), MPI_STATUSES_IGNORE);
            Subdomain<K>::clearBuffer(allocate);
        }
#if HPDDM_ICOLLECTIVE
        /* Function: Ideflation
         *
         *  Computes the first part of a coarse correction asynchronously.
         *
         * Template parameter:
         *    excluded       - True if the main processes are excluded from the domain decomposition, false otherwise.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector.
         *    rq             - MPI request to check completion of the MPI transfers. */
        template<bool excluded>
        void Ideflation(const K* const in, K* const out, const unsigned short& mu, MPI_Request* rq) const {
            if(excluded)
                super::co_->template IcallSolver<excluded>(super::uc_, mu, rq);
            else {
                Wrapper<K>::diag(Subdomain<K>::dof_, d_, in, out, mu);
                int tmp = mu;
                Blas<K>::gemm(&(Wrapper<K>::transc), "N", super::getAddrLocal(), &tmp, &(Subdomain<K>::dof_), &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::dof_), out, &(Subdomain<K>::dof_), &(Wrapper<K>::d__0), super::uc_, super::getAddrLocal());
                super::co_->template IcallSolver<excluded>(super::uc_, mu, rq);
            }
        }
#endif // HPDDM_ICOLLECTIVE
        /* Function: buildTwo
         *
         *  Assembles and factorizes the coarse operator by calling <Preconditioner::buildTwo>.
         *
         * Template Parameter:
         *    excluded       - Greater than 0 if the main processes are excluded from the domain decomposition, equal to 0 otherwise.
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
            if(super::co_) {
                super::co_->setParent(this);
                const unsigned short p = opt.val<unsigned short>(prefix + "level_2_p", 1);
                const unsigned short method = opt.val<unsigned short>(prefix + "level_2_schwarz_method", HPDDM_SCHWARZ_METHOD_NONE);
                if(super::co_ && p > 1 && opt.val<unsigned short>(prefix + "level_2_aggregate_size", p) == 1 && (method == HPDDM_SCHWARZ_METHOD_RAS || method == HPDDM_SCHWARZ_METHOD_ASM) && opt.val<char>(prefix + "level_2_krylov_method", HPDDM_KRYLOV_METHOD_GMRES) != HPDDM_KRYLOV_METHOD_NONE) {
                    CoarseOperator<CoarseSolver, S, K>* coNeumann  = nullptr;
                    std::vector<K> overlap;
                    std::vector<std::vector<std::pair<unsigned short, unsigned short>>> reduction;
                    std::map<std::pair<unsigned short, unsigned short>, unsigned short> sizes;
                    std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>> extra;
                    MPI_Request rs = MPI_REQUEST_NULL;
                    if(opt.set(prefix + "level_2_schwarz_coarse_correction") && A) {
                        MatrixCSR<K>* backup = Subdomain<K>::a_;
                        Subdomain<K>::a_ = A;
                        coNeumann = new CoarseOperator<CoarseSolver, S, K>;
                        std::string filename = opt.prefix(prefix + "level_2_dump_matrix", true);
                        std::string filenameNeumann = std::string("-hpddm_") + prefix + std::string("level_2_dump_matrix ") + filename + std::string("_Neumann");
                        if(filename.size() > 0)
                            opt.parse(filenameNeumann);
                        if(!A->ia_ && !A->ja_ && A->nnz_ == backup->nnz_) {
                            A->ia_ = backup->ia_;
                            A->ja_ = backup->ja_;
                        }
                        super::template buildTwo<excluded, MatrixAccumulation<Schwarz<Solver, CoarseSolver, S, K>, K>>(this, comm, coNeumann, overlap, reduction, sizes, extra);
                        if(A->ia_ == backup->ia_ && A->ja_ == backup->ja_ && A->nnz_ == backup->nnz_) {
                            A->ia_ = nullptr;
                            A->ja_ = nullptr;
                        }
                        if(overlap.size())
                            MPI_Isend(overlap.data(), overlap.size(), Wrapper<K>::mpi_type(), 0, 300, coNeumann->getCommunicator(), &rs);
                        if(filename.size() > 0) {
                            filenameNeumann = std::string("-hpddm_") + prefix + std::string("level_2_dump_matrix ") + filename;
                            opt.parse(filenameNeumann);
                        }
                        Subdomain<K>::a_ = backup;
                    }
                    opt.setPrefix(prefix + "level_2_");
                    super::co_->buildThree(coNeumann, reduction, sizes, extra);
                    delete coNeumann;
                    MPI_Wait(&rs, MPI_STATUS_IGNORE);
                }
                opt.setPrefix(std::string());
            }
#else
            ignore(A);
#endif
            return ret;
        }
        template<bool excluded = false>
        bool start(const K* const b, K* const x, const unsigned short& mu = 1) const {
            bool allocate = Subdomain<K>::setBuffer();
            if(!excluded && Subdomain<K>::a_->ia_) {
                const std::unordered_map<unsigned int, K> map = Subdomain<K>::boundaryConditions();
                for(const std::pair<const unsigned int, K>& p : map)
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        x[nu * Subdomain<K>::dof_ + p.first] = b[nu * Subdomain<K>::dof_ + p.first] / p.second;
            }
            exchange(x, mu);
            if(super::co_) {
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
         *    excluded       - Greater than 0 if the main processes are excluded from the domain decomposition, equal to 0 otherwise.
         *
         * Parameters:
         *    in             - Input vectors, modified internally if no workspace array is specified!
         *    out            - Output vectors.
         *    mu             - Number of vectors.
         *    work           - Workspace array. */
        template<bool excluded = false>
        int apply(const K* const in, K* const out, const unsigned short& mu = 1, K* work = nullptr) const {
            const char correction = Option::get()->val<char>(super::prefix("schwarz_coarse_correction"), -1);
            if((!super::co_ && !super::cc_) || correction == -1) {
                if(type_ == Prcndtnr::NO)
                    std::copy_n(in, mu * Subdomain<K>::dof_, out);
                else if(type_ == Prcndtnr::GE || type_ == Prcndtnr::OG) {
                    if(!excluded) {
                        super::s_.solve(in, out, mu);
                        exchange(out, mu);               // out = D A \ in
                    }
                }
                else {
                    if(!excluded) {
                        if(type_ == Prcndtnr::OS) {
                            Wrapper<K>::diag(Subdomain<K>::dof_, d_, in, out, mu);
                            super::s_.solve(out, mu);
                            Wrapper<K>::diag(Subdomain<K>::dof_, d_, out, mu);
                        }
                        else
                            super::s_.solve(in, out, mu);
                        Subdomain<K>::exchange(out, mu); // out = A \ in
                    }
                }
            }
            else {
                int n = mu * Subdomain<K>::dof_;
                if(!work)
                    work = const_cast<K*>(in);
                else if(!excluded)
                    std::copy_n(in, n, work);
                if(correction == HPDDM_SCHWARZ_COARSE_CORRECTION_ADDITIVE) {
#if HPDDM_ICOLLECTIVE
                    MPI_Request rq[2];
                    Ideflation<excluded>(in, out, mu, rq);
                    if(!excluded) {
                        super::s_.solve(work, mu); // out = A \ in
                        MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
                        const int k = mu;
                        Blas<K>::gemm("N", "N", &(Subdomain<K>::dof_), &k, super::getAddrLocal(), &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::dof_), super::uc_, super::getAddrLocal(), &(Wrapper<K>::d__0), out, &(Subdomain<K>::dof_)); // out = ev_ E \ ev_^T D in
                        Blas<K>::axpy(&n, &(Wrapper<K>::d__1), work, &i__1, out, &i__1);
                        exchange(out, mu);                                               // out = Z E \ Z^T in + A \ in
                    }
                    else
                        MPI_Wait(rq + 1, MPI_STATUS_IGNORE);
#else
                    deflation<excluded>(in, out, mu);
                    if(!excluded) {
                        super::s_.solve(work, mu);
                        Blas<K>::axpy(&n, &(Wrapper<K>::d__1), work, &i__1, out, &i__1);
                        exchange(out, mu);
                    }
#endif // HPDDM_ICOLLECTIVE
                }
                else {
                    deflation<excluded>(in, out, mu);                                    // out = Z E \ Z^T in
                    if(!excluded) {
                        if(!Subdomain<K>::a_->ia_ && !Subdomain<K>::a_->ja_) {
                            K* tmp = new K[mu * Subdomain<K>::dof_];
                            GMV(out, tmp, mu);
                            Blas<K>::axpby(n, -1.0, tmp, 1, 1.0, work, 1);
                            delete [] tmp;
                        }
                        else {
                            if(HPDDM_NUMBERING == Wrapper<K>::I)
                                Wrapper<K>::csrmm("N", &(Subdomain<K>::dof_), &(n = mu), &(Subdomain<K>::dof_), &(Wrapper<K>::d__2), Subdomain<K>::a_->sym_, Subdomain<K>::a_->a_, Subdomain<K>::a_->ia_, Subdomain<K>::a_->ja_, out, &(Wrapper<K>::d__1), work);
                            else if(Subdomain<K>::a_->ia_[Subdomain<K>::dof_] == Subdomain<K>::a_->nnz_)
                                Wrapper<K>::template csrmm<'C'>("N", &(Subdomain<K>::dof_), &(n = mu), &(Subdomain<K>::dof_), &(Wrapper<K>::d__2), Subdomain<K>::a_->sym_, Subdomain<K>::a_->a_, Subdomain<K>::a_->ia_, Subdomain<K>::a_->ja_, out, &(Wrapper<K>::d__1), work);
                            else
                                Wrapper<K>::template csrmm<'F'>("N", &(Subdomain<K>::dof_), &(n = mu), &(Subdomain<K>::dof_), &(Wrapper<K>::d__2), Subdomain<K>::a_->sym_, Subdomain<K>::a_->a_, Subdomain<K>::a_->ia_, Subdomain<K>::a_->ja_, out, &(Wrapper<K>::d__1), work);
                        }
                        exchange(work, mu);                                              //  in = (I - A Z E \ Z^T) in
                        if(type_ == Prcndtnr::OS)
                            Wrapper<K>::diag(Subdomain<K>::dof_, d_, work, mu);
                        super::s_.solve(work, mu);
                        exchange(work, mu);                                              //  in = D A \ (I - A Z E \ Z^T) in
                        n = mu * Subdomain<K>::dof_;
                        if(correction == HPDDM_SCHWARZ_COARSE_CORRECTION_BALANCED) {
                            if(!excluded) {
                                K* tmp = new K[super::cc_ ? 2 * n : n];
                                GMV(work, tmp, mu);
                                if(super::cc_) {
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
            return 0;
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
            for(const pairNeighbor& neighbor : Subdomain<K>::map_)
                for(unsigned int i : neighbor.second)
                    if(d_[i] > HPDDM_EPS)
                        intoOverlap.insert(i);
            std::vector<std::vector<std::pair<unsigned int, K>>> tmp(intoOverlap.size());
            unsigned int k, iPrev = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, HPDDM_GRANULARITY) reduction(+ : iPrev)
#endif
            for(k = 0; k < intoOverlap.size(); ++k) {
                auto it = std::next(intoOverlap.cbegin(), k);
                tmp[k].reserve(A->ia_[*it + 1] - A->ia_[*it]);
                for(unsigned int j = A->ia_[*it] - (N == 'F'); j < A->ia_[*it + 1] - (N == 'F'); ++j) {
                    K value = d_[*it] * d_[A->ja_[j] - (N == 'F')] * A->a_[j];
                    if(std::abs(value) > HPDDM_EPS && intoOverlap.find(A->ja_[j] - (N == 'F')) != intoOverlap.cend())
                        tmp[k].emplace_back(A->ja_[j], value);
                }
                iPrev += tmp[k].size();
            }
            int nnz = iPrev;
            delete B;
            B = new MatrixCSR<K>(Subdomain<K>::dof_, Subdomain<K>::dof_, nnz, A->sym_);
            nnz = iPrev = k = 0;
            for(unsigned int i : intoOverlap) {
                std::fill(B->ia_ + iPrev, B->ia_ + i + 1, nnz + (N == 'F'));
                for(const std::pair<unsigned int, K>& p : tmp[k]) {
                    B->ja_[nnz] = p.first;
                    B->a_[nnz++] = p.second;
                }
                ++k;
                iPrev = i + 1;
            }
            std::fill(B->ia_ + iPrev, B->ia_ + Subdomain<K>::dof_ + 1, nnz + (N == 'F'));
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
            Eps<K> evp(threshold, Subdomain<K>::dof_, opt.val<unsigned short>("geneo_nu", 20));
#ifndef PY_MAJOR_VERSION
            bool free = pattern ? pattern->sameSparsity(A) : Subdomain<K>::a_->sameSparsity(A);
#else
            ignore(pattern);
            constexpr bool free = false;
#endif
            MatrixCSR<K>* rhs = nullptr;
            if(B)
                rhs = B;
            else
                scaleIntoOverlap(A, rhs);
            if(super::ev_) {
                delete [] *super::ev_;
                delete [] super::ev_;
            }
#if defined(MUMPSSUB) || defined(MKL_PARDISOSUB)
            if(threshold > 0.0 && opt.val<char>("geneo_estimate_nu", 0) && (!B || B->hashIndices() == A->hashIndices())) {
                K* difference = new K[A->nnz_];
                std::copy_n(A->a_, A->nnz_, difference);
                for(unsigned int i = 0; i < A->n_; ++i) {
                    int* it = A->ja_ + A->ia_[i];
                    for(unsigned int j = rhs->ia_[i]; j < rhs->ia_[i + 1]; ++j) {
                        it = std::lower_bound(it, A->ja_ + A->ia_[i + 1], rhs->ja_[j]);
                        difference[std::distance(A->ja_, it++)] -= threshold * rhs->a_[j];
                    }
                }
                std::swap(A->a_, difference);
                Solver<K> s;
                evp.nu_ = std::max(1, s.inertia(A));
                std::swap(A->a_, difference);
                delete [] difference;
            }
#endif
            evp.template solve<Solver>(A, rhs, super::ev_, Subdomain<K>::communicator_, free ? &(super::s_) : nullptr);
            if(rhs != B)
                delete rhs;
            if(free && A->getFree()) {
                A->ia_ = nullptr;
                A->ja_ = nullptr;
            }
            opt["geneo_nu"] = evp.nu_;
            if(super::co_)
                super::co_->setLocal(evp.nu_);
            const int n = Subdomain<K>::dof_;
            std::for_each(super::ev_, super::ev_ + evp.nu_, [&](K* const v) { std::replace_if(v, v + n, [](K x) { return std::abs(x) < 1.0 / (HPDDM_EPS * HPDDM_PEN); }, K()); });
            if(resetPrefix)
                opt.setPrefix("");
        }
        /* Function: GMV
         *
         *  Computes a global sparse matrix-vector product.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector. */
#if HPDDM_DENSE
        virtual int GMV(const K* const in, K* const out, const int& mu = 1) const = 0;
#else
        int GMV(const K* const in, K* const out, const int& mu = 1, MatrixCSR<K>* const& A = nullptr) const {
#if 0
            K* tmp = new K[mu * Subdomain<K>::dof_];
            Wrapper<K>::diag(Subdomain<K>::dof_, d_, in, tmp, mu);
            if(HPDDM_NUMBERING == Wrapper<K>::I)
                Wrapper<K>::csrmm(Subdomain<K>::a_->sym_, &(Subdomain<K>::dof_), &mu, Subdomain<K>::a_->a_, Subdomain<K>::a_->ia_, Subdomain<K>::a_->ja_, tmp, out);
            else if(Subdomain<K>::a_->ia_[Subdomain<K>::dof_] == Subdomain<K>::a_->nnz_)
                Wrapper<K>::template csrmm<'C'>(Subdomain<K>::a_->sym_, &(Subdomain<K>::dof_), &mu, Subdomain<K>::a_->a_, Subdomain<K>::a_->ia_, Subdomain<K>::a_->ja_, tmp, out);
            else
                Wrapper<K>::template csrmm<'F'>(Subdomain<K>::a_->sym_, &(Subdomain<K>::dof_), &mu, Subdomain<K>::a_->a_, Subdomain<K>::a_->ia_, Subdomain<K>::a_->ja_, tmp, out);
            delete [] tmp;
            Subdomain<K>::exchange(out, mu);
#else
            if(A)
                Wrapper<K>::csrmm(A->sym_, &A->n_, &mu, A->a_, A->ia_, A->ja_, in, out);
            else if(HPDDM_NUMBERING == Wrapper<K>::I)
                Wrapper<K>::csrmm(Subdomain<K>::a_->sym_, &(Subdomain<K>::dof_), &mu, Subdomain<K>::a_->a_, Subdomain<K>::a_->ia_, Subdomain<K>::a_->ja_, in, out);
            else if(Subdomain<K>::a_->ia_[Subdomain<K>::dof_] == Subdomain<K>::a_->nnz_)
                Wrapper<K>::template csrmm<'C'>(Subdomain<K>::a_->sym_, &(Subdomain<K>::dof_), &mu, Subdomain<K>::a_->a_, Subdomain<K>::a_->ia_, Subdomain<K>::a_->ja_, in, out);
            else
                Wrapper<K>::template csrmm<'F'>(Subdomain<K>::a_->sym_, &(Subdomain<K>::dof_), &mu, Subdomain<K>::a_->a_, Subdomain<K>::a_->ia_, Subdomain<K>::a_->ja_, in, out);
            exchange(out, mu);
#endif
            return 0;
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
            int dim = mu * Subdomain<K>::dof_;
            K* tmp = new K[dim];
            bool allocate = Subdomain<K>::setBuffer();
            GMV(x, tmp, mu);
            Subdomain<K>::clearBuffer(allocate);
            Blas<K>::axpy(&dim, &(Wrapper<K>::d__2), f, &i__1, tmp, &i__1);
            std::fill_n(storage, 2 * mu, 0.0);
            if(norm == HPDDM_COMPUTE_RESIDUAL_L1) {
                for(unsigned int i = 0; i < Subdomain<K>::dof_; ++i) {
                    bool boundary = (std::abs(Subdomain<K>::boundaryCond(i)) > HPDDM_EPS);
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        if(!boundary)
                            storage[2 * nu + 1] += d_[i] * std::abs(tmp[nu * Subdomain<K>::dof_ + i]);
                        if(std::abs(f[nu * Subdomain<K>::dof_ + i]) > HPDDM_EPS * HPDDM_PEN)
                            storage[2 * nu] += d_[i] * std::abs(f[nu * Subdomain<K>::dof_ + i] / underlying_type<K>(HPDDM_PEN));
                        else
                            storage[2 * nu] += d_[i] * std::abs(f[nu * Subdomain<K>::dof_ + i]);
                    }
                }
            }
            else if(norm == HPDDM_COMPUTE_RESIDUAL_LINFTY) {
                for(unsigned int i = 0; i < Subdomain<K>::dof_; ++i) {
                    bool boundary = (std::abs(Subdomain<K>::boundaryCond(i)) > HPDDM_EPS);
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        if(!boundary)
                            storage[2 * nu + 1] = std::max(std::abs(tmp[nu * Subdomain<K>::dof_ + i]), storage[2 * nu + 1]);
                        if(std::abs(f[nu * Subdomain<K>::dof_ + i]) > HPDDM_EPS * HPDDM_PEN)
                            storage[2 * nu] = std::max(std::abs(f[nu * Subdomain<K>::dof_ + i] / underlying_type<K>(HPDDM_PEN)), storage[2 * nu]);
                        else
                            storage[2 * nu] = std::max(std::abs(f[nu * Subdomain<K>::dof_ + i]), storage[2 * nu]);
                    }
                }
            }
            else {
                for(unsigned int i = 0; i < Subdomain<K>::dof_; ++i) {
                    bool boundary = (std::abs(Subdomain<K>::boundaryCond(i)) > HPDDM_EPS);
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        if(!boundary)
                            storage[2 * nu + 1] += d_[i] * std::norm(tmp[nu * Subdomain<K>::dof_ + i]);
                        if(std::abs(f[nu * Subdomain<K>::dof_ + i]) > HPDDM_EPS * HPDDM_PEN)
                            storage[2 * nu] += d_[i] * std::norm(f[nu * Subdomain<K>::dof_ + i] / underlying_type<K>(HPDDM_PEN));
                        else
                            storage[2 * nu] += d_[i] * std::norm(f[nu * Subdomain<K>::dof_ + i]);
                    }
                }
            }
            delete [] tmp;
            if(norm == HPDDM_COMPUTE_RESIDUAL_L2 || norm == HPDDM_COMPUTE_RESIDUAL_L1) {
                MPI_Allreduce(MPI_IN_PLACE, storage, 2 * mu, Wrapper<K>::mpi_underlying_type(), Wrapper<underlying_type<K>>::mpi_op(MPI_SUM), Subdomain<K>::communicator_);
                if(norm == HPDDM_COMPUTE_RESIDUAL_L2)
                    std::for_each(storage, storage + 2 * mu, [](underlying_type<K>& b) { b = std::sqrt(b); });
            }
            else
                MPI_Allreduce(MPI_IN_PLACE, storage, 2 * mu, Wrapper<K>::mpi_underlying_type(), MPI_MAX, Subdomain<K>::communicator_);
        }
#endif
#ifdef PETSC_PCHPDDM_MAXLEVELS
        static PetscErrorCode destroy(PC_HPDDM_Level* const ctx, PetscBool all) {
            PetscFunctionBeginUser;
            PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "PCSHELL from PCHPDDM called with no context");
            static_assert(std::is_same<K, PetscScalar>::value, "Wrong type");
            if(ctx->P && ctx->P->dof_ != -1) {
                if(all) {
                    if(!std::is_same<K, PetscReal>::value)
                        delete [] ctx->P->d_;
                    ctx->P->dtor();
                    delete ctx->P;
                    ctx->P = nullptr;
                }
                else {
                    ctx->P->super::dtor();
                    ctx->P->clearBuffer();
                }
            }
            PetscFunctionReturn(PETSC_SUCCESS);
        }
        PetscErrorCode structure(const IS interior, IS is, const Mat D, Mat N, PC_HPDDM_Level** const levels) {
            Mat                    P;
            ISLocalToGlobalMapping l2g;
            PetscReal              *d;
            const PetscInt         *ptr;
            PetscInt               m, bs;
            PetscBool              sym, ismatis;

            PetscFunctionBeginUser;
            PetscCall(PetscObjectTypeCompare((PetscObject)N, MATIS, &ismatis));
            if(!Subdomain<K>::rq_) {
                PetscCall(PetscObjectGetComm((PetscObject)levels[0]->ksp, &(Subdomain<K>::communicator_)));
                PetscCall(ISGetLocalSize(is, &m));
                Subdomain<K>::dof_ = m;
                if(!std::is_same<PetscScalar, PetscReal>::value)
                    d = new PetscReal[Subdomain<K>::dof_];
                else
                    PetscCall(VecGetArray(levels[0]->D, reinterpret_cast<PetscScalar**>(&d)));
                d_ = d;
                PetscCall(ISGetIndices(is, &ptr));
                std::vector<std::pair<PetscInt, PetscInt>> v;
                if(ismatis) {
                    PetscCall(MatISGetLocalToGlobalMapping(N, &l2g, nullptr));
                    PetscCall(ISLocalToGlobalMappingGetBlockSize(l2g, &bs));
                }
                else {
                    PetscCall(KSPGetOperators(levels[0]->ksp, nullptr, &P));
                    PetscCall(MatGetBlockSize(P, &bs));
                }
                PetscCheck(Subdomain<K>::dof_ % bs == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible local size %d and Pmat block size %" PetscInt_FMT, Subdomain<K>::dof_, bs);
                if(!ismatis) {
                    PetscInt* idx;
                    PetscCall(PetscMalloc1(Subdomain<K>::dof_ / bs, &idx));
                    for(PetscInt i = 0; i < Subdomain<K>::dof_ / bs; ++i)
                        idx[i] = ptr[i * bs] / bs;
                    PetscCall(ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)levels[0]->ksp), bs, Subdomain<K>::dof_ / bs, idx, PETSC_OWN_POINTER, &l2g));
                }
                v.reserve(Subdomain<K>::dof_ / bs);
                for(PetscInt i = 0; i < Subdomain<K>::dof_; i += bs)
                    v.emplace_back(std::make_pair(ptr[i], i));
                std::sort(v.begin(), v.end());
                PetscInt nproc;
                PetscInt* procs;
                PetscInt* numprocs;
                PetscInt** indices;
                PetscCall(ISLocalToGlobalMappingGetBlockInfo(l2g, &nproc, &procs, &numprocs, &indices));
                nproc = std::max(nproc, static_cast<PetscInt>(1));
                unsigned short* sorted = new unsigned short[nproc - 1];
                std::iota(sorted, sorted + nproc - 1, 0);
                std::sort(sorted, sorted + nproc - 1, [&procs](unsigned short lhs, unsigned short rhs) { return procs[1 + lhs] < procs[1 + rhs]; });
                Subdomain<K>::map_.resize(nproc - 1);
                Subdomain<K>::buff_ = new K*[2 * Subdomain<K>::map_.size()]();
                Subdomain<K>::rq_ = new MPI_Request[2 * Subdomain<K>::map_.size()];
                if(ismatis) {
                    std::fill_n(d, Subdomain<K>::dof_, 0.0);
                    std::unordered_map<PetscInt, std::unordered_set<PetscInt>> boundary;
                    const PetscInt* idx;
                    PetscCall(ISLocalToGlobalMappingGetBlockIndices(l2g, &idx));
                    PetscCall(ISLocalToGlobalMappingGetSize(l2g, &m));
                    for(PetscInt i = 0; i < m / bs; ++i) {
                        std::vector<std::pair<PetscInt, PetscInt>>::const_iterator it = std::lower_bound(v.cbegin(), v.cend(), std::make_pair(bs * idx[i], static_cast<PetscInt>(0)));
                        if(it != v.cend() && it->first == bs * idx[i])
                            std::fill_n(d + it->second, bs, 1.0);
                    }
                    for(PetscInt i = 1; i < nproc; ++i) {
                        for(PetscInt j = 0; j < numprocs[i]; ++j) {
                            std::vector<std::pair<PetscInt, PetscInt>>::const_iterator it = std::lower_bound(v.cbegin(), v.cend(), std::make_pair(bs * idx[indices[i][j]], static_cast<PetscInt>(0)));
                            boundary[it->second].insert(i - 1);
                        }
                    }
                    PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(l2g, &idx));
                    std::unordered_set<PetscInt>* map = new std::unordered_set<PetscInt>[Subdomain<K>::map_.size()];
                    PetscCall(PetscObjectTypeCompare((PetscObject)D, MATSEQSBAIJ, &sym));
                    if(!sym) {
                        for(const std::pair<const PetscInt, std::unordered_set<PetscInt>>& p : boundary) {
                            PetscInt ncols;
                            const PetscInt *cols;
                            PetscCall(MatGetRow(D, p.first, &ncols, &cols, nullptr));
                            for(const PetscInt& i : p.second)
                                map[i].insert(cols, cols + ncols);
                            PetscCall(MatRestoreRow(D, p.first, &ncols, &cols, nullptr));
                            std::fill_n(d + p.first, bs, underlying_type<K>(1.0) / underlying_type<K>(1 + p.second.size()));
                        }
                    }
                    else {
                        PetscCall(MatSetOption(D, MAT_GETROW_UPPERTRIANGULAR, PETSC_TRUE));
                        for(PetscInt i = 0; i < Subdomain<K>::dof_; i += bs) {
                            PetscInt ncols;
                            const PetscInt *cols;
                            PetscCall(MatGetRow(D, i, &ncols, &cols, nullptr));
                            std::unordered_map<PetscInt, std::unordered_set<PetscInt>>::const_iterator it = boundary.find(i);
                            if(it != boundary.cend()) {
                                for(const PetscInt& i : it->second)
                                    map[i].insert(cols, cols + ncols);
                                std::fill_n(d + it->first, bs, underlying_type<K>(1.0) / underlying_type<K>(1 + it->second.size()));
                            }
                            for(PetscInt j = 0; j < ncols; j += bs) {
                                it = boundary.find(cols[j]);
                                if(it != boundary.cend()) {
                                    for(const PetscInt& k : it->second)
                                        for(PetscInt j = 0; j < bs; ++j)
                                            map[k].insert(i + j);
                                }
                            }
                            PetscCall(MatRestoreRow(D, i, &ncols, &cols, nullptr));
                        }
                        PetscCall(MatSetOption(D, MAT_GETROW_UPPERTRIANGULAR, PETSC_FALSE));
                    }
                    for(unsigned short i = 0; i < Subdomain<K>::map_.size(); ++i) {
                        Subdomain<K>::map_[i].first = procs[1 + sorted[i]];
                        Subdomain<K>::map_[i].second.reserve(map[sorted[i]].size());
                        std::copy(map[sorted[i]].cbegin(), map[sorted[i]].cend(), std::back_inserter(Subdomain<K>::map_[i].second));
                        std::sort(Subdomain<K>::map_[i].second.begin(), Subdomain<K>::map_[i].second.end(), [&ptr] (const PetscInt& lhs, const PetscInt& rhs) { return ptr[lhs] < ptr[rhs]; });
                    }
                    PetscCall(ISLocalToGlobalMappingRestoreBlockInfo(l2g, &nproc, &procs, &numprocs, &indices));
                    delete [] sorted;
                    delete [] map;
                }
                else {
                    for(unsigned short i = 0; i < Subdomain<K>::map_.size(); ++i) {
                        Subdomain<K>::map_[i].first = procs[1 + sorted[i]];
                        Subdomain<K>::map_[i].second.reserve(bs * numprocs[1 + sorted[i]]);
                        for(PetscInt j = 0; j < numprocs[1 + sorted[i]]; ++j) {
                            for(PetscInt k = 0; k < bs; ++k)
                                Subdomain<K>::map_[i].second.emplace_back(indices[1 + sorted[i]][j] * bs + k);
                        }
                    }
                    delete [] sorted;
                    PetscCall(ISLocalToGlobalMappingRestoreBlockInfo(l2g, &nproc, &procs, &numprocs, &indices));
                    PetscCall(ISLocalToGlobalMappingDestroy(&l2g));
                    PetscCall(ISRestoreIndices(is, &ptr));
                    PetscCall(ISGetLocalSize(interior, &m));
                    PetscCall(ISGetIndices(interior, &ptr));
                    PetscInt rstart = m ? ptr[0] : 0;
                    PetscInt rend = rstart + m;
                    PetscCall(ISRestoreIndices(interior, &ptr));
                    PetscCall(ISGetIndices(is, &ptr));
                    for(PetscInt i = 0; i < Subdomain<K>::dof_; i += bs) {
                        if(ptr[i] >= rstart && ptr[i] < rend)
                            std::fill_n(d + i, bs, 1.0);
                        else
                            std::fill_n(d + i, bs, 0.0);
                    }
                }
                PetscCall(ISRestoreIndices(is, &ptr));
                if(!std::is_same<PetscScalar, PetscReal>::value) {
                    PetscScalar* c;
                    PetscCall(VecGetArray(levels[0]->D, &c));
                    std::copy_n(d, Subdomain<K>::dof_, c);
                }
                PetscCall(VecRestoreArray(levels[0]->D, nullptr));
            }
            else {
                if(Subdomain<K>::dof_ == -1) {
                    PetscInt n;
                    PetscCall(VecGetLocalSize(levels[0]->D, &n));
                    Subdomain<K>::dof_ = n;
                }
                if(!std::is_same<PetscScalar, PetscReal>::value) {
                    PetscScalar* c;
                    PetscCall(VecGetArray(levels[0]->D, &c));
                    std::copy_n(d_, Subdomain<K>::dof_, c);
                    PetscCall(VecRestoreArray(levels[0]->D, nullptr));
                }
            }
            PetscFunctionReturn(PETSC_SUCCESS);
        }
#endif
#if HPDDM_SLEPC
    public:
        typename super::co_type::return_type buildTwo(const MPI_Comm& comm, Mat D, PetscInt n, PetscInt M, PC_HPDDM_Level** const levels) {
            Mat       A;
            PetscBool flg;

            PetscFunctionBeginUser;
            PetscCall(KSPGetOperators(levels[n]->ksp, nullptr, &A));
#if defined(PETSC_HAVE_HTOOL)
            PetscCall(PetscObjectTypeCompare((PetscObject)A, MATHTOOL, &flg));
            if(!flg) {
#endif
                PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSCHURCOMPLEMENT, &flg));
                if (!flg)
                    PetscCall(super::template buildTwo<false, MatrixMultiplication<Schwarz<K>, K>>(this, comm, D, n, M, levels));
                else
                    PetscCall(super::template buildTwo<false, BddProjection<Schwarz<K>, K>>(this, comm, D, n, M, levels));
#if defined(PETSC_HAVE_HTOOL)
            }
            else {
                struct ClassWithPtr {
                    typedef Schwarz<PetscScalar> super;
                    const super*              const A_;
                    const PetscScalar* const        E_;
                    ClassWithPtr(const super* const A, const PetscScalar* const E) : A_(A), E_(E) { }
                    const MPI_Comm& getCommunicator() const { return A_->getCommunicator(); }
                    const vectorNeighbor& getMap() const { return A_->getMap(); }
                    constexpr int getDof() const { return A_->getDof(); }
                    constexpr unsigned short getLocal() const { return A_->getLocal(); }
                    const K* const* getVectors() const { return A_->getVectors(); }
                    const K* getOperator() const { return E_; }
                };
                const htool::VirtualHMatrix<PetscScalar>* hmatrix;
                PetscCall(MatHtoolGetHierarchicalMat(A, &hmatrix));
                std::vector<PetscScalar> E;
                htool::build_coarse_space_outside(hmatrix, levels[n]->nu, super::getDof(), super::getVectors(), E);
                ClassWithPtr Op(levels[n]->P, E.data());
                PetscCall(super::template buildTwo<false, UserCoarseOperator<ClassWithPtr, PetscScalar>>(&Op, comm, D, n, M, levels));
            }
#endif
            PetscFunctionReturn(PETSC_SUCCESS);
        }
        PetscErrorCode solveGEVP(IS is, Mat N, std::vector<Vec> initial, PC_HPDDM_Level** const levels, Mat weighted, Mat rhs) {
            EPS                    eps = nullptr;
            ST                     st;
            KSP                    empty = nullptr, *ksp;
            Mat                    local, *resized;
            Vec                    vr, vreduced;
            ISLocalToGlobalMapping l2g;
            IS                     sub[1] = { };
            PetscInt               nev, nconv = 0;
            PetscBool              flg, ismatis, solve = PETSC_FALSE;
            const char             *prefix;
            Aux                    aux = nullptr;
            Sum                    ctx = nullptr;
            Harmonic               h = nullptr;

            PetscFunctionBeginUser;
            if(!levels[0]->parent->deflation) {
                for(int i = 0; i < Subdomain<K>::dof_ && !solve; ++i)
                    if(HPDDM::abs(d_[i]) > underlying_type<K>(HPDDM_EPS))
                        solve = PETSC_TRUE;
                PetscCall(KSPGetOptionsPrefix(levels[0]->ksp, &prefix));
                PetscCall(PetscObjectTypeCompare((PetscObject)N, MATSHELL, &flg));
                if(flg && N == weighted && !rhs) {
                    std::for_each(initial.begin(), initial.end(), [&](Vec v) { PetscCallVoid(VecDestroy(&v)); });
                    std::vector<Vec>().swap(initial);
                    if(solve) {
                        PetscCall(MatShellGetContext(N, &h));
                        if(h->A[1]) {
                            Mat  A;
                            char type[256] = { MATAIJ };
                            PetscCall(PetscOptionsGetString(nullptr, prefix, "-eps_mat_type", type, sizeof(type), nullptr));
                            PetscCall(PetscStrcmp(type, MATSHELL, &flg));
                            if(!flg) {
                                Mat               B, C;
                                const PetscScalar *vals;
                                const PetscInt    *rows;
                                PetscInt          m, n;
                                PetscCall(MatGetLocalSize(N, nullptr, &n));
                                PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n, n, nullptr, &A));
                                PetscCall(MatShift(A, 1.0));
                                PetscCall(MatMatMult(N, A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &B));
                                PetscCall(MatDestroy(&A));
                                PetscCall(MatCreateSubMatrix(h->A[1], h->is[3], h->is[3], MAT_INITIAL_MATRIX, &local));
                                PetscCall(MatPtAP(local, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));
                                PetscCall(MatGetLocalSize(C, nullptr, &n));
                                PetscCall(MatDestroy(&local));
                                PetscCall(MatDestroy(&B));
                                PetscCall(MatCreate(PETSC_COMM_SELF, &A));
                                PetscCall(MatSetType(A, type));
                                PetscCall(MatGetLocalSize(h->A[1], &m, nullptr));
                                PetscCall(MatSetSizes(A, m, m, m, m));
                                PetscCall(ISBlockGetIndices(h->is[4], &rows));
                                PetscCall(ISGetBlockSize(h->is[4], &m));
                                if(m > 1) {
                                    PetscCall(MatSetBlockSize(A, m));
                                    PetscCall(MatSetOption(A, MAT_ROW_ORIENTED, PETSC_FALSE));
                                    n /= m;
                                    m *= m;
                                }
                                PetscCall(MatDenseGetArrayRead(C, &vals));
                                for(PetscInt i = 0; i < n; ++i)
                                    PetscCall(MatSetValuesBlocked(A, n, rows, 1, rows + i, vals + m * n * i, INSERT_VALUES));
                                PetscCall(ISBlockRestoreIndices(h->is[4], &rows));
                                PetscCall(MatDenseRestoreArrayRead(C, &vals));
                                PetscCall(MatDestroy(&C));
                                PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
                                PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
                            }
                            else {
                                std::tuple<Harmonic, Mat, Vec[4]> *p;
                                PetscInt                          m;
                                PetscCall(MatGetLocalSize(h->A[1], &m, nullptr));
                                PetscCall(PetscNew(&p));
                                std::get<0>(*p) = h;
                                std::get<1>(*p) = N;
                                PetscCall(MatCreateVecs(h->A[1], std::get<2>(*p) + 2, std::get<2>(*p) + 3));
                                PetscCall(MatCreateVecs(N, std::get<2>(*p), std::get<2>(*p) + 1));
                                PetscCall(MatCreateShell(PETSC_COMM_SELF, m, m, m, m, p, &A));
                                PetscCall(MatShellSetOperation(A, MATOP_MULT, (void (*)(void))MatMult_Harmonic));
                                PetscCall(MatShellSetOperation(A, MATOP_DESTROY, (void (*)(void))MatDestroy_Harmonic));
                            }
                            PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
                            PetscCall(EPSSetOptionsPrefix(eps, prefix));
                            PetscCall(EPSSetOperators(eps, h->A[1], A));
                            PetscCall(MatDestroy(&A));
                            PetscCall(MatCreateVecs(h->A[1], nullptr, &vreduced));
                        }
                        else {
                            SVD  svd;
                            void (*destroy)(void);
                            PetscCall(SVDCreate(PETSC_COMM_SELF, &svd));
                            PetscCall(SVDSetOptionsPrefix(svd, prefix));
                            PetscCall(SVDSetType(svd, SVDLANCZOS));
                            PetscCall(SVDSetOperators(svd, N, nullptr));
                            PetscCall(SVDSetFromOptions(svd));
                            PetscCall(SVDGetDimensions(svd, &nev, nullptr, nullptr));
                            PetscCall(MatShellGetOperation(N, MATOP_DESTROY, &destroy));
                            PetscCall(MatShellSetOperation(N, MATOP_DESTROY, nullptr));
                            PetscCall(SVDSetUp(svd));
                            PetscCall(SVDSolve(svd));
                            PetscCall(MatShellSetOperation(N, MATOP_DESTROY, destroy));
                            PetscCall(SVDGetConverged(svd, &nconv));
                            if(levels[0]->threshold >= PetscReal()) {
                                PetscReal theta;
                                PetscCall(SVDGetSingularTriplet(svd, 0, &theta, nullptr, nullptr));
                                for(levels[0]->nu = 0; levels[0]->nu < nconv; ++levels[0]->nu) {
                                    PetscReal sigma;
                                    PetscCall(SVDGetSingularTriplet(svd, levels[0]->nu, &sigma, nullptr, nullptr));
                                    if(sigma / theta < levels[0]->threshold) {
                                        PetscCall(PetscInfo(svd, "HPDDM: Discarding singular value %g, which is lower than the largest singular value %g times %g\n", double(sigma), double(theta), double(levels[0]->threshold)));
                                        break;
                                    }
                                    else
                                        PetscCall(PetscInfo(svd, "HPDDM: Using singular value %g\n", double(sigma)));
                                }
                                if(levels[0]->nu == nconv)
                                    PetscCall(PetscInfo(svd, "HPDDM: Not enough converged singular triplets to satisfy the threshold criterion\n"));
                            }
                            else
                                levels[0]->nu = std::min(nconv, nev);
                            PetscCall(PetscInfo(svd, "HPDDM: Using %" PetscInt_FMT " out of %" PetscInt_FMT " converged singular vectors\n", levels[0]->nu, nconv));
                            if(levels[0]->nu) {
                                Harmonic h;
                                Vec      u, full;
                                super::ev_ = new K*[levels[0]->nu];
                                *super::ev_ = new K[Subdomain<K>::dof_ * levels[0]->nu]();
                                PetscCall(MatShellGetContext(N, &h));
                                PetscCall(MatCreateVecs(N, nullptr, &u));
                                PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, Subdomain<K>::dof_, nullptr, &full));
                                for(PetscInt i = 0; i < levels[0]->nu; ++i) {
                                    super::ev_[i] = *super::ev_ + i * Subdomain<K>::dof_;
                                    PetscCall(VecPlaceArray(full, super::ev_[i]));
                                    PetscCall(SVDGetSingularTriplet(svd, i, nullptr, u, nullptr));
                                    PetscCall(VecISCopy(full, h->is[2], SCATTER_FORWARD, u));
                                    PetscCall(VecResetArray(full));
                                }
                                PetscCall(VecDestroy(&full));
                                PetscCall(VecDestroy(&u));
                            }
                            PetscCall(SVDDestroy(&svd));
                        }
                    }
                    if(!eps)
                        PetscFunctionReturn(PETSC_SUCCESS);
                }
                if(!eps)
                    PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
                PetscCall(EPSSetOptionsPrefix(eps, prefix));
                PetscCall(PetscObjectTypeCompare((PetscObject)N, MATIS, &ismatis));
                if(!ismatis) {
                    Mat compact;
                    IS  is;
                    PetscCall(PetscObjectQuery((PetscObject)N, "_PCHPDDM_Embed", (PetscObject*)&is));
                    PetscCall(PetscObjectQuery((PetscObject)N, "_PCHPDDM_Compact", (PetscObject*)&compact));
                    if(compact && is && solve) {
                        SVD         svd;
                        PetscScalar *values;
                        PetscInt    m, p;
                        PetscCall(PetscNew(&aux));
                        aux->is = is;
                        PetscCall(SVDCreate(PETSC_COMM_SELF, &svd));
                        PetscCall(SVDSetOptionsPrefix(svd, prefix));
                        PetscCall(SVDSetOperators(svd, compact, nullptr));
                        PetscCall(SVDSetType(svd, SVDLAPACK));
                        PetscCall(SVDSetFromOptions(svd));
                        PetscCall(SVDSetUp(svd));
                        PetscCall(SVDSolve(svd));
                        PetscCall(MatGetLocalSize(compact, &m, &p));
                        PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &aux->sigma));
                        PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, p, m, nullptr, &aux->V));
                        PetscCall(VecGetArrayWrite(aux->sigma, &values));
                        for(PetscInt n = 0; n < m; ++n) {
                            PetscReal s;
                            Vec       v;
                            PetscCall(MatDenseGetColumnVecWrite(aux->V, n, &v));
                            PetscCall(SVDGetSingularTriplet(svd, n, &s, nullptr, v));
                            values[n] = static_cast<PetscReal>(1.0) / s;
                            PetscCall(MatDenseRestoreColumnVecWrite(aux->V, n, &v));
                        }
                        PetscCall(MatCreateShell(PETSC_COMM_SELF, m, m, m, m, aux, &N));
                        PetscCall(MatShellSetOperation(N, MATOP_MULT, (void (*)(void))MatMult_Aux));
                        PetscCall(VecRestoreArrayWrite(aux->sigma, &values));
                        PetscCall(SVDDestroy(&svd));
                    }
                    if(weighted && !h)
                        PetscCall(EPSSetOperators(eps, N, weighted));
                }
                else {
                    PetscCall(MatISGetLocalToGlobalMapping(N, &l2g, nullptr));
                    PetscCall(ISGlobalToLocalMappingApplyIS(l2g, IS_GTOLM_DROP, is, &sub[0]));
                    PetscCall(ISDestroy(&is));
                    PetscCall(MatCreateSubMatrices(weighted, 1, sub, sub, MAT_INITIAL_MATRIX, &resized));
                    if(!rhs)
                        PetscCall(MatDestroy(&weighted));
                    PetscCall(MatISGetLocalMat(N, &local));
                    PetscCall(PetscObjectTypeCompare((PetscObject)local, MATSEQSBAIJ, &flg));
                    if(flg) {
                        /* going back from SEQBAIJ to SEQSBAIJ */
                        PetscCall(MatSetOption(resized[0], MAT_SYMMETRIC, PETSC_TRUE));
                        PetscCall(MatConvert(resized[0], MATSEQSBAIJ, MAT_INPLACE_MATRIX, &resized[0]));
                    }
                    PetscCall(EPSSetOperators(eps, local, resized[0]));
                    if(!initial.empty()) {
                        std::vector<Vec> full = initial;
                        for(PetscInt i = 0; i < full.size(); ++i) {
                            PetscCall(MatCreateVecs(resized[0], &initial[i], nullptr));
                            PetscCall(VecISCopy(full[i], sub[0], SCATTER_REVERSE, initial[i]));
                            PetscCall(VecDestroy(&full[i]));
                        }
                    }
                }
                PetscCall(EPSSetProblemType(eps, EPS_GHEP));
                PetscCall(EPSSetTarget(eps, 0.0));
                PetscCall(EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE));
                PetscCall(EPSGetST(eps, &st));
                PetscCall(STSetType(st, STSINVERT));
                PetscCall(EPSSetInitialSpace(eps, initial.size(), initial.data()));
                std::for_each(initial.begin(), initial.end(), [&](Vec v) { PetscCallVoid(VecDestroy(&v)); });
                std::vector<Vec>().swap(initial);
                PetscCall(EPSSetFromOptions(eps));
                if(weighted || ismatis) {
                    if(levels == levels[0]->parent->levels && levels[0]->parent->share) {
                        PetscCheck(levels[0]->pc, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "No fine-level PC attached?");
                        PetscUseMethod(levels[0]->pc, "PCASMGetSubKSP_C", (PC, PetscInt*, PetscInt*, KSP**), (levels[0]->pc, nullptr, nullptr, &ksp));
                        PetscCall(STGetKSP(st, &empty));
                        PetscCall(PetscObjectReference((PetscObject)empty));
                        PetscCall(STSetKSP(st, ksp[0]));
                        if(h)
                            PetscCall(KSPSetReusePreconditioner(ksp[0], PETSC_TRUE));
                    }
                    if(levels[0]->threshold >= PetscReal() && !h) {
                        flg = PETSC_FALSE;
                        PetscCall(PetscOptionsGetBool(nullptr, prefix, "-eps_use_inertia", &flg, nullptr));
                        if(flg) {
                            KSP           ksp;
                            PC            pc;
                            Mat           F;
                            MatSolverType type;
                            PetscScalar   target;
                            PetscInt      val;
                            PetscCall(EPSGetTarget(eps, &target));
                            PetscCall(EPSSetTarget(eps, levels[0]->threshold));
                            PetscCall(STGetOperator(st, nullptr));
                            PetscCall(STGetKSP(st, &ksp));
                            PetscCall(KSPGetPC(ksp, &pc));
                            PetscCall(PCSetFromOptions(pc));
                            PetscCall(PetscObjectTypeCompareAny((PetscObject)pc, &flg, PCLU, PCCHOLESKY, nullptr));
                            PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot use -%seps_use_inertia without PCLU or PCCHOLESKY", prefix);
                            if(PetscDefined(HAVE_MUMPS)) {
                                PetscCall(PCFactorGetMatSolverType(pc, &type));
                                if(!type) {
                                    PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
                                    PetscCall(PCSetFromOptions(pc));
                                }
                                else
                                    PetscCall(PetscStrcmp(type, MATSOLVERMUMPS, &flg));
                            }
#if PetscDefined(HAVE_MUMPS)
                            if(flg) {
                                PetscCall(PCFactorGetMatrix(pc, &F));
                                PetscCall(MatMumpsGetIcntl(F, 13, &val));
                                PetscCall(MatMumpsSetIcntl(F, 13, 1));
                                PetscCall(PCSetUp(pc));
                                PetscCall(PCFactorGetMatrix(pc, &F));
                                PetscCall(MatMumpsSetIcntl(F, 13, val));
                                PetscCall(MatMumpsGetInfog(F, 12, &val));
                            }
                            else
#endif
                            {
                                PetscCall(PCSetUp(pc));
                                PetscCall(PCFactorGetMatrix(pc, &F));
                                PetscCall(MatGetInertia(F, &val, nullptr, nullptr));
                            }
                            if(val > 0) {
                                PetscCall(EPSSetDimensions(eps, val, PETSC_DEFAULT, PETSC_DEFAULT));
                                PetscCall(EPSSetTarget(eps, target));
                            }
                            else {
                                solve = PETSC_FALSE;
                                nconv = 0;
                            }
                        }
                    }
                }
                else {
                    PetscScalar target;
                    PetscCheck(levels[0]->pc, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "No fine-level PC attached?");
                    PetscUseMethod(levels[0]->pc, "PCASMGetSubKSP_C", (PC, PetscInt*, PetscInt*, KSP**), (levels[0]->pc, nullptr, nullptr, &ksp));
                    PetscCall(PetscNew(&ctx));
                    PetscCall(KSPGetOperators(ksp[0], ctx->A, nullptr));
                    ctx->A[1] = ctx->A[2] = nullptr;
                    ctx->decomposition = this;
                    ctx->work = new std::set<unsigned short>();
                    for(unsigned short i = 0, size = Subdomain<K>::map_.size(); i < size; ++i)
                        ctx->work->insert(i);
                    ctx->status = 'a';
                    PetscCall(MatCreateShell(PETSC_COMM_SELF, Subdomain<K>::dof_, Subdomain<K>::dof_, Subdomain<K>::dof_, Subdomain<K>::dof_, ctx, &weighted));
                    PetscCall(MatShellSetOperation(weighted, MATOP_MULT, (void (*)(void))MatMult_Sum));
                    Subdomain<K>::setBuffer();
                    PetscCall(EPSSetOperators(eps, ctx->A[0], weighted));
                    PetscCall(EPSGetTarget(eps, &target));
                    PetscCheck(PetscAbsScalar(target) < PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot use -%seps_target with a magnitude %g (!= 0.0)", prefix, double(PetscAbsScalar(target)));
                    PetscCall(STGetKSP(st, &empty));
                    PetscCall(PetscObjectReference((PetscObject)empty));
                    PetscCall(STSetKSP(st, ksp[0]));
                }
                PetscCall(EPSGetDimensions(eps, &nev, nullptr, nullptr));
                if(solve) {
                    if(!ctx) {
                        MatStructure str;
                        PetscCall(STGetMatStructure(st, &str));
                        if(str != SAME_NONZERO_PATTERN)
                            PetscCall(PetscInfo(st, "HPDDM: The MatStructure of the GenEO eigenproblem stencil is set to %d, -%sst_matstructure same is preferred depending on what is passed to PCHPDDMSetAuxiliaryMat()\n", int(str), prefix));
                    }
                    PetscCall(EPSSolve(eps));
                    PetscCall(EPSGetConverged(eps, &nconv));
                }
                if(ctx) {
                    ctx->status = 'b';
                    while(!ctx->work->empty() || ctx->status != 'c')
                        PetscCall(MatMult_Sum(weighted, nullptr, nullptr));
                }
                if(levels[0]->threshold >= PetscReal()) {
                    PetscReal relative = 1.0;
                    for(levels[0]->nu = 0; levels[0]->nu < nconv; ++levels[0]->nu) {
                        PetscReal   re, im;
#if defined(PETSC_USE_COMPLEX)
                        PetscScalar eig;
                        PetscCall(EPSGetEigenvalue(eps, levels[0]->nu, &eig, nullptr));
                        re = PetscRealPart(eig);
                        im = PetscImaginaryPart(eig);
#else
                        PetscCall(EPSGetEigenvalue(eps, levels[0]->nu, &re, &im));
#endif
                        if(levels[0]->nu == 0 && h) relative = HPDDM::hypot(re, im);
                        if(HPDDM::hypot(re, im) / relative > levels[0]->threshold) {
                            if(HPDDM::abs(im) > std::numeric_limits<PetscReal>::epsilon()) {
                                if(!h)
                                    PetscCall(PetscInfo(eps, "HPDDM: Discarding eigenvalue (%g,%g), which magnitude is higher than %g\n", double(re), double(im), double(levels[0]->threshold)));
                                else
                                    PetscCall(PetscInfo(eps, "HPDDM: Discarding eigenvalue (%g,%g), which magnitude is higher than the one of the smallest eigenvalue %g times %g\n", double(re), double(im), double(relative), double(levels[0]->threshold)));
                            }
                            else {
                                if(!h)
                                    PetscCall(PetscInfo(eps, "HPDDM: Discarding eigenvalue %g, which magnitude is higher than %g\n", double(re), double(levels[0]->threshold)));
                                else
                                    PetscCall(PetscInfo(eps, "HPDDM: Discarding eigenvalue %g, which magnitude is higher than the one of the smallest eigenvalue %g times %g\n", double(re), double(relative), double(levels[0]->threshold)));
                            }
                            break;
                        }
                        else {
                            if(HPDDM::abs(im) > std::numeric_limits<PetscReal>::epsilon())
                                PetscCall(PetscInfo(eps, "HPDDM: Using eigenvalue (%g,%g)\n", double(re), double(im)));
                            else
                                PetscCall(PetscInfo(eps, "HPDDM: Using eigenvalue %g\n", double(re)));
                        }
                    }
                    if(levels[0]->nu == nconv)
                        PetscCall(PetscInfo(eps, "HPDDM: Not enough converged eigenpairs to satisfy the threshold criterion\n"));
                }
                else
                    levels[0]->nu = std::min(nconv, nev);
                PetscCall(PetscInfo(eps, "HPDDM: Using %" PetscInt_FMT " out of %" PetscInt_FMT " converged eigenvectors\n", levels[0]->nu, nconv));
            }
            else {
                PetscInt n;
                PetscCall(MatGetSize(weighted, nullptr, &n));
                levels[0]->nu = n;
            }
            if(levels[0]->nu) {
                super::ev_ = new K*[levels[0]->nu];
                *super::ev_ = new K[Subdomain<K>::dof_ * levels[0]->nu]();
            }
            for(unsigned short i = 1; i < levels[0]->nu; ++i)
                super::ev_[i] = *super::ev_ + i * Subdomain<K>::dof_;
            if(!levels[0]->parent->deflation) {
                PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, Subdomain<K>::dof_, levels[0]->nu ? super::ev_[0] : nullptr, &vr));
                if(ismatis)
                    PetscCall(MatCreateVecs(resized[0], &vreduced, nullptr));
                else if(ctx) {
                    for(unsigned short i = 0, size = Subdomain<K>::map_.size(); i < size; ++i)
                        ctx->work->insert(i);
                    ctx->status = 'a';
                }
                for(PetscInt i = 0, flg = PETSC_FALSE; i < levels[0]->nu; ++i) {
                    PetscCall(VecPlaceArray(vr, super::ev_[i]));
                    if(!h) {
                        if(!ismatis)
                            PetscCall(EPSGetEigenvector(eps, i, !flg ? vr : nullptr, !flg ? nullptr : vr));
                        else {
                            PetscCall(EPSGetEigenvector(eps, i, !flg ? vreduced : nullptr, !flg ? nullptr : vr));
                            PetscCall(VecISCopy(vr, sub[0], SCATTER_FORWARD, vreduced));
                        }
                    }
                    else {
                        PetscCall(EPSGetEigenvector(eps, i, !flg ? vreduced : nullptr, !flg ? nullptr : vreduced));
                        PetscCall(VecISCopy(vreduced, h->is[2], SCATTER_REVERSE, vr));
                    }
                    PetscCall(VecResetArray(vr));
#if !defined(PETSC_USE_COMPLEX)
                    if(!flg) {
                        PetscScalar eigi;
                        PetscCall(EPSGetEigenvalue(eps, i, nullptr, &eigi));
                        if(HPDDM::abs(eigi) > std::numeric_limits<PetscReal>::epsilon())
                            flg = PETSC_TRUE;
                    }
                    else
                        flg = PETSC_FALSE;
#endif
                }
                if(ctx) {
                    ctx->status = 'b';
                    while(!ctx->work->empty() || ctx->status != 'c')
                        PetscCall(MatMult_Sum(weighted, nullptr, nullptr));
                    PetscCall(MatDestroy(&weighted));
                    PetscCall(MatDestroy(ctx->A + 1));
                    PetscCall(MatDestroy(ctx->A + 2));
                    delete ctx->work;
                    PetscCall(PetscFree(ctx));
                    Subdomain<K>::clearBuffer(true);
                }
                PetscCall(VecDestroy(&vr));
                if(empty) {
                    PetscCall(STSetKSP(st, empty));
                    PetscCall(PetscObjectDereference((PetscObject)empty));
                }
                PetscCall(EPSDestroy(&eps));
                if(!h) {
                    if(!ismatis) {
                        if(!rhs)
                            PetscCall(MatDestroy(&weighted));
                        if(aux) {
                            PetscCall(MatDestroy(&aux->V));
                            PetscCall(VecDestroy(&aux->sigma));
                            PetscCall(PetscFree(aux));
                            PetscCall(MatDestroy(&N));
                        }
                    }
                    else {
                        PetscCall(VecDestroy(&vreduced));
                        PetscCall(ISDestroy(&sub[0]));
                        PetscCall(MatDestroySubMatrices(1, &resized));
                        PetscCall(MatISRestoreLocalMat(N, &local));
                    }
                }
                else
                    PetscCall(VecDestroy(&vreduced));
            }
            else {
                PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, Subdomain<K>::dof_, levels[0]->nu, *super::ev_, &local));
                PetscCall(MatCopy(weighted, local, SAME_NONZERO_PATTERN));
                PetscCall(MatDestroy(&local));
            }
            PetscFunctionReturn(PETSC_SUCCESS);
        }
        static PetscErrorCode next(Mat* A, Mat* N, PetscInt i, PetscInt* const n, PC_HPDDM_Level** const levels) {
            Mat            P;
            PetscInt       rstart, rend;
            PetscErrorCode ierr;

            PetscFunctionBeginUser;
            if (A && *A) PetscValidHeaderSpecific(*A, MAT_CLASSID, 1);
            if (N && *N) PetscValidHeaderSpecific(*N, MAT_CLASSID, 2);
            PetscAssertPointer(n, 4);
            PetscAssertPointer(levels, 5);
#if defined(PETSC_USE_LOG)
            if(!levels[0]->parent->log_separate)
                PetscCall(PetscLogEventBegin(PC_HPDDM_PtAP, levels[i]->ksp, nullptr, nullptr, nullptr));
#endif
            char fail[2] { };
            if(levels[i - 1]->P) {
                ierr = levels[i - 1]->P->buildTwo(levels[i - 1]->P->getCommunicator(), *A, i - 1, *n, levels);
                if(ierr != PETSC_ERR_ARG_WRONG && levels[i]->ksp) {
                    PetscCall(KSPGetOperators(levels[i]->ksp, &P, nullptr));
                    PetscCall(MatGetOwnershipRange(P, &rstart, &rend));
                    if(rstart == rend)
                        fail[1] = 1;
                }
            }
            else
                ierr = PetscErrorCode(PETSC_SUCCESS);
            fail[0] = (ierr == PETSC_ERR_ARG_WRONG ? 1 : 0);
            PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, fail, 2, MPI_CHAR, MPI_MAX, PetscObjectComm((PetscObject)(levels[0]->ksp))));
            if(fail[0]) { /* building level i + 1 failed because there was no deflation vector */
                *n = i;
                if(i > 1 && N)
                    PetscCall(MatDestroy(N));
            }
            else
                PetscCall(ierr);
            if(i > 1 && A)
                PetscCall(MatDestroy(A));
#if defined(PETSC_USE_LOG)
            if(!levels[0]->parent->log_separate)
                PetscCall(PetscLogEventEnd(PC_HPDDM_PtAP, levels[i]->ksp, nullptr, nullptr, nullptr));
#endif
            if(fail[1]) { /* cannot build level i + 1 because at least one subdomain is empty */
                *n = i + 1;
                PetscFunctionReturn(PETSC_SUCCESS);
            }
            if(i + 1 < *n && levels[i - 1]->P) {
                PetscBool algebraic;
                char type[256] = { }; /* same size as in src/ksp/pc/interface/pcset.c */
                std::string prefix(((PetscObject)levels[i - 1]->ksp)->prefix);
                unsigned int pos = prefix.rfind("levels_", prefix.size() - 1);
                unsigned short level = std::stoi(prefix.substr(pos + 7, prefix.size() - 1));
                prefix = prefix.substr(0, pos + 7) + std::to_string(level + 1) + "_";
                PetscCall(PetscOptionsGetString(nullptr, prefix.c_str(), "-st_pc_type", type, sizeof(type), nullptr));
                PetscCall(PetscStrcmp(type, PCMAT, &algebraic));
                if(!levels[0]->parent->aux || algebraic) {
                    if(levels[i]->ksp) {
                        Mat              *sub, weighted;
                        IS               loc, uis;
                        Vec              xin;
                        PetscInt         m;
                        std::vector<Vec> initial;
                        delete levels[i]->P;
                        levels[i]->P = new Schwarz<PetscScalar>();
                        PetscCall(ISCreateStride(PETSC_COMM_SELF, rend - rstart, rstart, 1, &uis));
                        PetscCall(ISDuplicate(uis, &loc));
                        PetscCall(MatIncreaseOverlap(P, 1, &uis, 1));
                        PetscCall(ISSort(uis));
                        PetscCall(ISSetInfo(uis, IS_SORTED, IS_GLOBAL, PETSC_TRUE, PETSC_TRUE));
                        PetscCall(ISGetLocalSize(uis, &m));
                        PetscCall(VecDestroy(&levels[i]->D));
                        PetscCall(VecCreateMPI(PETSC_COMM_SELF, m, PETSC_DETERMINE, &levels[i]->D));
                        PetscCall(VecScatterDestroy(&levels[i]->scatter));
                        PetscCall(MatCreateVecs(P, &xin, nullptr));
                        PetscCall(VecScatterCreate(xin, uis, levels[i]->D, nullptr, &levels[i]->scatter));
                        PetscCall(VecDestroy(&xin));
                        PetscUseMethod(levels[0]->parent->levels[0]->ksp->pc->pmat, "PCHPDDMAlgebraicAuxiliaryMat_Private_C", (Mat, IS*, Mat*[], PetscBool), (P, &uis, &sub, PETSC_FALSE));
                        PetscCall(levels[i]->P->structure(loc, uis, sub[0], nullptr, levels + i));
                        PetscCall(ISDestroy(&loc));
                        PetscCall(MatDuplicate(sub[0], MAT_COPY_VALUES, &weighted));
                        PetscCall(MatDiagonalScale(weighted, levels[i]->D, levels[i]->D));
                        PetscCall(MatPropagateSymmetryOptions(sub[0], weighted));
                        PetscCall(levels[i]->P->solveGEVP(uis, sub[0], initial, levels + 1, weighted, nullptr));
                        PetscCall(PetscObjectReference((PetscObject)sub[0]));
                        Mat Q = sub[0];
                        PetscCall(next(&Q, nullptr, i + 1, n, levels));
                        PetscCall(ISDestroy(&uis));
                        PetscCall(PetscObjectQuery((PetscObject)sub[0], "_PCHPDDM_Embed", (PetscObject*)&uis));
                        PetscCall(ISDestroy(&uis));
                        PetscCall(PetscObjectCompose((PetscObject)sub[0], "_PCHPDDM_Embed", nullptr));
                        PetscCall(PetscObjectCompose((PetscObject)sub[0], "_PCHPDDM_Compact", nullptr));
                        PetscCall(MatDestroySubMatrices(2, &sub));
                    }
                    else {
                        delete levels[i]->P;
                        levels[i]->P = nullptr;
                        PetscCall(next(nullptr, nullptr, i + 1, n, levels));
                    }
                    PetscCall(PetscObjectComposeFunction((PetscObject)levels[0]->parent->levels[0]->ksp, "PCHPDDMSetUp_Private_C", nullptr));
                }
                else {
#if defined(PETSC_USE_LOG)
                    if(!levels[0]->parent->log_separate)
                        PetscCall(PetscLogEventBegin(PC_HPDDM_PtBP, levels[i]->ksp, nullptr, nullptr, nullptr));
#endif
                    CoarseOperator<DMatrix, K>* coNeumann  = nullptr;
                    std::vector<K> overlap;
                    std::vector<std::vector<std::pair<unsigned short, unsigned short>>> reduction;
                    std::map<std::pair<unsigned short, unsigned short>, unsigned short> sizes;
                    std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>> extra;
                    MPI_Request rs = MPI_REQUEST_NULL;
                    coNeumann = new CoarseOperator<DMatrix, K>;
                    PetscCall(levels[i - 1]->P->super::template buildTwo<false, MatrixAccumulation<Schwarz<K>, K>>(levels[i - 1]->P, levels[i - 1]->P->getCommunicator(), *N, i - 1, *n, levels, coNeumann, overlap, reduction, sizes, extra));
                    if(i > 1)
                        PetscCall(MatDestroy(N));
                    if(overlap.size())
                        PetscCallMPI(MPI_Isend(overlap.data(), overlap.size(), Wrapper<K>::mpi_type(), 0, 300, coNeumann->getCommunicator(), &rs));
#if defined(PETSC_USE_LOG)
                    if(!levels[0]->parent->log_separate) {
                        PetscCall(PetscLogEventEnd(PC_HPDDM_PtBP, levels[i]->ksp, nullptr, nullptr, nullptr));
                        PetscCall(PetscLogEventBegin(PC_HPDDM_Next, levels[i]->ksp, nullptr, nullptr, nullptr));
                    }
#endif
                    PetscCall(levels[i - 1]->P->co_->buildThree(coNeumann, reduction, sizes, extra, A, N, levels[i]));
                    delete coNeumann;
                    if(i + 2 == *n)
                        PetscCall(MatDestroy(N));
                    if(*A)
                        levels[i]->P = levels[i - 1]->P->co_->getSubdomain()->P;
                    else
                        levels[i]->P = nullptr;
                    PetscCallMPI(MPI_Wait(&rs, MPI_STATUS_IGNORE));
#if defined(PETSC_USE_LOG)
                    if(!levels[0]->parent->log_separate)
                        PetscCall(PetscLogEventEnd(PC_HPDDM_Next, levels[i]->ksp, nullptr, nullptr, nullptr));
#endif
                }
            }
            PetscFunctionReturn(PETSC_SUCCESS);
        }
        PetscErrorCode initialize(IS is, Mat N, Mat weighted, Mat rhs, std::vector<Vec> initial, PC_HPDDM_Level** const levels) {
            PetscFunctionBeginUser;
            PetscCall(solveGEVP(is, N, initial, levels, weighted, rhs));
            PetscCall(PetscObjectComposeFunction((PetscObject)levels[0]->ksp, "PCHPDDMSetUp_Private_C", next));
            PetscFunctionReturn(PETSC_SUCCESS);
        }
#endif
        /* Function: deflation
         *
         *  Computes a coarse correction.
         *
         * Template parameter:
         *    excluded       - True if the main processes are excluded from the domain decomposition, false otherwise.
         *
         * Parameters:
         *    in             - Input vectors.
         *    out            - Output vectors.
         *    mu             - Number of vectors. */
        template<bool excluded>
        void deflation(const K* const in, K* const out, const unsigned short& mu) const {
#if HPDDM_PETSC
            PetscFunctionBeginUser;
#endif
#if HPDDM_SCHWARZ
            if(super::cc_) {
                (*super::cc_)(in, out, Subdomain<K>::dof_, mu);
                return;
            }
#endif
            if(excluded)
                super::co_->template callSolver<excluded>(super::uc_, mu);
            else {
                Wrapper<K>::diag(Subdomain<K>::dof_, d_, in, out, mu);                                                                                                                                                            // out = D in
                int tmp = mu;
                int local = super::getLocal();
                if(local)
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &local, &tmp, &(Subdomain<K>::dof_), &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::dof_), out, &(Subdomain<K>::dof_), &(Wrapper<K>::d__0), super::uc_, &local); // uc_ = ev_^T D in
                super::co_->template callSolver<excluded>(super::uc_, mu);                                                                                                                                                        // uc_ = E \ ev_^T D in
                if(local)
                    Blas<K>::gemm("N", "N", &(Subdomain<K>::dof_), &tmp, &local, &(Wrapper<K>::d__1), *super::ev_, &(Subdomain<K>::dof_), super::uc_, &local, &(Wrapper<K>::d__0), out, &(Subdomain<K>::dof_));                   // out = ev_ E \ ev_^T D in
                else
                    std::fill_n(out, mu * Subdomain<K>::dof_, 0.0);
                exchange(out, mu);
            }
#if HPDDM_PETSC
            PetscCallVoid(PetscLogFlops(mu * (super::getLocal() * (2 * Subdomain<K>::dof_ - 1) + Subdomain<K>::dof_ * (2 * super::getLocal() - 1) + Subdomain<K>::dof_)));
            PetscFunctionReturnVoid();
#endif
        }
        /* Function: getScaling
         *  Returns a constant pointer to <Schwarz::d>. */
        const underlying_type<K>* getScaling() const { return d_; }
        template<class I, char N = HPDDM_NUMBERING>
        void distributedNumbering(I* const in, I& first, I& last, long long& global) const {
            Subdomain<K>::template globalMapping<N>(in, in + Subdomain<K>::dof_, first, last, global, d_);
        }
        template<class I, class T = K>
        bool distributedCSR(I* const num, I first, I last, I*& ia, I*& ja, T*& c) const {
            return Subdomain<K>::distributedCSR(num, first, last, ia, ja, c, Subdomain<K>::a_);
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
PETSC_EXTERN PetscErrorCode PCHPDDM_Internal(HPDDM::Schwarz<PetscScalar>* const P, IS is, Mat const N, Mat const weighted, Mat const rhs, std::vector<Vec> initial, PC_HPDDM_Level** const levels) {
    PetscFunctionBeginUser;
    PetscCheck(P, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "PCHPDDM_Internal() called with no HPDDM object");
    PetscCall(P->initialize(is, N, weighted, rhs, initial, levels));
    PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatMult_Aux(Mat A, Vec x, Vec y) {
    Aux aux;
    Vec left, right, leftEcon;

    PetscFunctionBeginUser;
    PetscCall(MatShellGetContext(A, &aux));
    PetscCall(MatCreateVecs(aux->V, &right, &left));
    PetscCall(MatCreateVecs(aux->V, nullptr, &leftEcon));
    PetscCall(VecSet(left, 0.0));
    PetscCall(VecISCopy(left, aux->is, SCATTER_FORWARD, x));
    PetscCall(MatMultTranspose(aux->V, left, right));
    PetscCall(MatMult(aux->V, right, leftEcon));
    PetscCall(VecAXPY(leftEcon, -1.0, left));
    PetscCall(VecPointwiseMult(y, aux->sigma, right));
    PetscCall(MatMult(aux->V, y, left));
    PetscCall(VecAXPY(left, static_cast<PetscReal>(-1.0) / PETSC_SMALL, leftEcon));
    PetscCall(VecISCopy(left, aux->is, SCATTER_REVERSE, y));
    PetscCall(VecDestroy(&left));
    PetscCall(VecDestroy(&right));
    PetscCall(VecDestroy(&leftEcon));
    PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatMult_Sum(Mat A, Vec x, Vec y) {
    PetscScalar                 *out, *work, **buffer;
    const PetscScalar           *in;
    const PetscReal             *d;
    PetscInt                    n;
    Sum                         p;
    HPDDM::Schwarz<PetscScalar> *decomposition;
    HPDDM::vectorNeighbor       map;
    MPI_Request*                rq;
    MPI_Comm                    communicator;

    PetscFunctionBeginUser;
    PetscCall(MatShellGetContext(A, &p));
    decomposition = reinterpret_cast<HPDDM::Schwarz<PetscScalar>*>(p->decomposition);
    buffer = decomposition->getBuffer();
    d = decomposition->getScaling();
    map = decomposition->getMap();
    rq = decomposition->getRq();
    communicator = decomposition->getCommunicator();
    PetscCall(MatGetLocalSize(A, &n, nullptr));
    if(p->status == 'a') {
        PetscCall(VecGetArrayWrite(y, &out));
        PetscCall(VecGetArrayRead(x, &in));
        HPDDM::Wrapper<PetscScalar>::diag(n, d, in, out);
        PetscCall(VecRestoreArrayRead(x, &in));
    }
    for(unsigned short i = 0, size = map.size(); i < size; ++i) {
        if(p->work->find(i) != p->work->cend())
            PetscCallMPI(MPI_Irecv(buffer[i], map[i].second.size(), HPDDM::Wrapper<PetscScalar>::mpi_type(), map[i].first, 660, communicator, rq + i));
        else
            rq[i] = MPI_REQUEST_NULL;
        if(p->status == 'a') {
            HPDDM::Wrapper<PetscScalar>::gthr(map[i].second.size(), out, buffer[size + i], map[i].second.data());
            PetscCallMPI(MPI_Isend(buffer[size + i], map[i].second.size(), HPDDM::Wrapper<PetscScalar>::mpi_type(), map[i].first, 660, communicator, rq + size + i));
        }
        else if(p->status == 'b')
            PetscCallMPI(MPI_Isend(buffer[size + i], 0, HPDDM::Wrapper<PetscScalar>::mpi_type(), map[i].first, 660, communicator, rq + size + i));
    }
    if(!p->work->empty() || p->status == 'a') {
        PetscInt m = 0;
        if(p->A[1])
            PetscCall(MatGetLocalSize(p->A[1], nullptr, &m));
        if(m != p->work->size() + (p->status == 'a')) {
            PetscCall(MatDestroy(p->A + 1));
            PetscCall(MatDestroy(p->A + 2));
            PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n, p->work->size() + (p->status == 'a'), nullptr, p->A + 1));
            PetscCall(MatProductCreate(p->A[0], p->A[1], nullptr, p->A + 2));
            PetscCall(MatProductSetType(p->A[2], MATPRODUCT_AB));
            PetscCall(MatProductSetFromOptions(p->A[2]));
            PetscCall(MatProductSymbolic(p->A[2]));
        }
        PetscCall(MatDenseGetArrayWrite(p->A[1], &work));
    }
    if(p->status == 'a') {
        std::copy_n(out, n, work);
        PetscCall(VecRestoreArrayWrite(y, &out));
    }
    for(unsigned short i = 0, size = map.size(); i < size; ++i) {
        int index;
        MPI_Status st;
        PetscCallMPI(MPI_Waitany(map.size(), rq, &index, &st));
        std::set<unsigned short>::const_iterator it = p->work->find(index);
        if(it != p->work->cend()) {
            int count;
            PetscCallMPI(MPI_Get_count(&st, HPDDM::Wrapper<PetscScalar>::mpi_type(), &count));
            if(count == 0)
                p->work->erase(it);
        }
    }
    if(!p->work->empty() || p->status == 'a') {
        for(std::set<unsigned short>::const_iterator it = p->work->cbegin(); it != p->work->cend(); ++it)
            HPDDM::Wrapper<PetscScalar>::sctr(map[*it].second.size(), buffer[*it], map[*it].second.data(), work + (std::distance(p->work->cbegin(), it) + (p->status == 'a')) * n);
        PetscCall(MatDenseRestoreArrayWrite(p->A[1], &work));
        PetscCall(MatProductNumeric(p->A[2]));
        PetscCall(MatDenseGetArrayRead(p->A[2], &in));
    }
    if(p->status != 'c') {
        PetscCallMPI(MPI_Waitall(map.size(), rq + map.size(), MPI_STATUSES_IGNORE));
        if(p->status == 'b')
            p->status = 'c';
    }
    for(unsigned short i = 0, size = map.size(); i < size; ++i) {
        if(p->status == 'a')
            PetscCallMPI(MPI_Irecv(buffer[i], map[i].second.size(), HPDDM::Wrapper<PetscScalar>::mpi_type(), map[i].first, 661, communicator, rq + i));
        std::set<unsigned short>::const_iterator it = p->work->find(i);
        if(it != p->work->cend()) {
            HPDDM::Wrapper<PetscScalar>::gthr(map[i].second.size(), in + (std::distance(p->work->cbegin(), it) + (p->status == 'a')) * n, buffer[size + i], map[i].second.data());
            PetscCallMPI(MPI_Isend(buffer[size + i], map[i].second.size(), HPDDM::Wrapper<PetscScalar>::mpi_type(), map[i].first, 661, communicator, rq + size + i));
        }
        else
            rq[size + i] = MPI_REQUEST_NULL;
    }
    if(p->status == 'a') {
        PetscCall(VecGetArrayWrite(y, &out));
        std::copy_n(in, n, out);
        PetscCall(MatDenseRestoreArrayRead(p->A[2], &in));
        for(unsigned short i = 0, size = map.size(); i < size; ++i) {
            int index;
            PetscCallMPI(MPI_Waitany(map.size(), rq, &index, MPI_STATUS_IGNORE));
            for(unsigned int j = 0; j < map[index].second.size(); ++j)
                out[map[index].second[j]] += buffer[index][j];
        }
        HPDDM::Wrapper<PetscScalar>::diag(n, d, out);
        PetscCall(VecRestoreArrayWrite(y, &out));
    }
    else if(!p->work->empty())
        PetscCall(MatDenseRestoreArrayRead(p->A[2], &in));
    PetscCallMPI(MPI_Waitall(map.size(), rq + map.size(), MPI_STATUSES_IGNORE));
    PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatMult_Harmonic(Mat A, Vec x, Vec y) {
    std::tuple<Harmonic, Mat, Vec[4]> *p;

    PetscFunctionBeginUser;
    PetscCall(MatShellGetContext(A, &p));
    PetscCall(VecISCopy(x, std::get<0>(*p)->is[4], SCATTER_REVERSE, std::get<2>(*p)[0]));
    PetscCall(MatMult(std::get<1>(*p), std::get<2>(*p)[0], std::get<2>(*p)[1]));
    PetscCall(VecSet(std::get<2>(*p)[2], 0.0));
    PetscCall(VecISCopy(std::get<2>(*p)[2], std::get<0>(*p)->is[3], SCATTER_FORWARD, std::get<2>(*p)[1]));
    PetscCall(MatMult(std::get<0>(*p)->A[1], std::get<2>(*p)[2], std::get<2>(*p)[3]));
    PetscCall(VecISCopy(std::get<2>(*p)[3], std::get<0>(*p)->is[3], SCATTER_REVERSE, std::get<2>(*p)[1]));
    PetscCall(MatMultTranspose(std::get<1>(*p), std::get<2>(*p)[1], std::get<2>(*p)[0]));
    PetscCall(VecSet(y, 0.0));
    PetscCall(VecISCopy(y, std::get<0>(*p)->is[4], SCATTER_FORWARD, std::get<2>(*p)[0]));
    PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatDestroy_Harmonic(Mat A) {
    std::tuple<Harmonic, Mat, Vec[4]> *p;

    PetscFunctionBeginUser;
    PetscCall(MatShellGetContext(A, &p));
    for(PetscInt i = 0; i < 4; ++i)
        PetscCall(VecDestroy(std::get<2>(*p) + i));
    PetscCall(PetscFree(p));
    PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
#endif // HPDDM_SCHWARZ_HPP_
