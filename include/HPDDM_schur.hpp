/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
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

#ifndef HPDDM_SCHUR_HPP_
#define HPDDM_SCHUR_HPP_

#include "HPDDM_preconditioner.hpp"
#include "HPDDM_eigensolver.hpp"

namespace HPDDM {
/* Class: Schur
 *
 *  A class from which derives <Bdd> and <Feti> that inherits from <Preconditioner>.
 *
 * Template Parameters:
 *    Solver         - Solver used for the factorization of local matrices.
 *    CoarseOperator - Class of the coarse operator.
 *    K              - Scalar type. */
template<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
    template<class> class Solver, class CoarseOperator,
#endif
    class K>
class Schur : public Preconditioner<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
              Solver, CoarseOperator,
#elif HPDDM_PETSC
              DMatrix,
#endif
              K> {
#if HPDDM_FETI || HPDDM_BDD
    private:
        /* Function: exchangeSchurComplement
         *
         *  Exchanges the local Schur complements <Schur::schur> to form an explicit restriction of the global Schur complement.
         *
         * Template Parameter:
         *    L              - 'S'ymmetric or 'G'eneral transfer of the local Schur complements.
         *
         * Parameters:
         *    send           - Buffer for sending the local Schur complement.
         *    recv           - Buffer for receiving the local Schur complement of each neighboring subdomains.
         *    res            - Restriction of the global Schur complement. */
        template<char L>
        void exchangeSchurComplement(K* const* const& send, K* const* const& recv, K* const& res) const {
            if(send && recv && res) {
                if(L == 'S')
                    for(unsigned short i = 0; i < Subdomain<K>::map_.size(); ++i) {
                        MPI_Irecv(recv[i], (Subdomain<K>::map_[i].second.size() * (Subdomain<K>::map_[i].second.size() + 1)) / 2, Wrapper<K>::mpi_type(), Subdomain<K>::map_[i].first, 1, Subdomain<K>::communicator_, Subdomain<K>::rq_ + i);
                        for(unsigned int j = 0; j < Subdomain<K>::map_[i].second.size(); ++j)
                            for(unsigned int k = j; k < Subdomain<K>::map_[i].second.size(); ++k) {
                                if(Subdomain<K>::map_[i].second[j] < Subdomain<K>::map_[i].second[k])
                                    send[i][Subdomain<K>::map_[i].second.size() * j - (j * (j + 1)) / 2 + k] = schur_[Subdomain<K>::map_[i].second[j] * Subdomain<K>::dof_ + Subdomain<K>::map_[i].second[k]];
                                else
                                    send[i][Subdomain<K>::map_[i].second.size() * j - (j * (j + 1)) / 2 + k] = schur_[Subdomain<K>::map_[i].second[k] * Subdomain<K>::dof_ + Subdomain<K>::map_[i].second[j]];
                            }
                        MPI_Isend(send[i], (Subdomain<K>::map_[i].second.size() * (Subdomain<K>::map_[i].second.size() + 1)) / 2, Wrapper<K>::mpi_type(), Subdomain<K>::map_[i].first, 1, Subdomain<K>::communicator_, Subdomain<K>::rq_ + Subdomain<K>::map_.size() + i);
                    }
                else
                    for(unsigned short i = 0; i < Subdomain<K>::map_.size(); ++i) {
                        MPI_Irecv(recv[i], Subdomain<K>::map_[i].second.size() * Subdomain<K>::map_[i].second.size(), Wrapper<K>::mpi_type(), Subdomain<K>::map_[i].first, 1, Subdomain<K>::communicator_, Subdomain<K>::rq_ + i);
                        for(unsigned int j = 0; j < Subdomain<K>::map_[i].second.size(); ++j)
                            for(unsigned int k = 0; k < Subdomain<K>::map_[i].second.size(); ++k) {
                                if(Subdomain<K>::map_[i].second[j] < Subdomain<K>::map_[i].second[k])
                                    send[i][j * Subdomain<K>::map_[i].second.size() + k] = schur_[Subdomain<K>::map_[i].second[j] * Subdomain<K>::dof_ + Subdomain<K>::map_[i].second[k]];
                                else
                                    send[i][j * Subdomain<K>::map_[i].second.size() + k] = schur_[Subdomain<K>::map_[i].second[k] * Subdomain<K>::dof_ + Subdomain<K>::map_[i].second[j]];
                            }
                        MPI_Isend(send[i], Subdomain<K>::map_[i].second.size() * Subdomain<K>::map_[i].second.size(), Wrapper<K>::mpi_type(), Subdomain<K>::map_[i].first, 1, Subdomain<K>::communicator_, Subdomain<K>::rq_ + Subdomain<K>::map_.size() + i);
                    }
                Blas<K>::lacpy("L", &(Subdomain<K>::dof_), &(Subdomain<K>::dof_), schur_, &(Subdomain<K>::dof_), res, &(Subdomain<K>::dof_));
                if(L == 'S')
                    for(unsigned short i = 0; i < Subdomain<K>::map_.size(); ++i) {
                        int index;
                        MPI_Waitany(Subdomain<K>::map_.size(), Subdomain<K>::rq_, &index, MPI_STATUS_IGNORE);
                        for(unsigned int j = 0; j < Subdomain<K>::map_[index].second.size(); ++j) {
                            for(unsigned int k = 0; k < j; ++k)
                                if(Subdomain<K>::map_[index].second[j] <= Subdomain<K>::map_[index].second[k])
                                    res[Subdomain<K>::map_[index].second[j] * Subdomain<K>::dof_ + Subdomain<K>::map_[index].second[k]] += recv[index][Subdomain<K>::map_[index].second.size() * k - (k * (k + 1)) / 2 + j];
                            for(unsigned int k = j; k < Subdomain<K>::map_[index].second.size(); ++k)
                                if(Subdomain<K>::map_[index].second[j] <= Subdomain<K>::map_[index].second[k])
                                    res[Subdomain<K>::map_[index].second[j] * Subdomain<K>::dof_ + Subdomain<K>::map_[index].second[k]] += recv[index][Subdomain<K>::map_[index].second.size() * j - (j * (j + 1)) / 2 + k];
                        }
                    }
                else
                    for(unsigned short i = 0; i < Subdomain<K>::map_.size(); ++i) {
                        int index;
                        MPI_Waitany(Subdomain<K>::map_.size(), Subdomain<K>::rq_, &index, MPI_STATUS_IGNORE);
                        for(unsigned int j = 0; j < Subdomain<K>::map_[index].second.size(); ++j)
                            for(unsigned int k = 0; k < Subdomain<K>::map_[index].second.size(); ++k)
                                if(Subdomain<K>::map_[index].second[j] <= Subdomain<K>::map_[index].second[k])
                                    res[Subdomain<K>::map_[index].second[j] * Subdomain<K>::dof_ + Subdomain<K>::map_[index].second[k]] += recv[index][j * Subdomain<K>::map_[index].second.size() + k];
                    }
            }
        }
#endif
    protected:
        /* Variable: bb
         *  Local matrix assembled on boundary degrees of freedom. */
        MatrixCSR<K>*          bb_;
        /* Variable: ii
         *  Local matrix assembled on interior degrees of freedom. */
        MatrixCSR<K>*          ii_;
        /* Variable: bi
         *  Local matrix assembled on boundary and interior degrees of freedom. */
        MatrixCSR<K>*          bi_;
        /* Variable: schur
         *  Explicit local Schur complement. */
        K*                  schur_;
        /* Variable: work
         *  Workspace array. */
        K*                   work_;
        /* Variable: structure
         *  Workspace array of size lower than or equal to <Subdomain::dof>. */
        K*              structure_;
        /* Variable: pinv
         *  Solver used in <Schur::callNumfact> and <Bdd::callNumfact> for factorizing <Subdomain::a> or <Schur::schur>. */
        void*                pinv_;
        /* Variable: rankWorld
         *  Rank of the current subdomain in <Subdomain::communicator>. */
        int             rankWorld_;
        /* Variable: mult
         *  Number of local Lagrange multipliers. */
        int                  mult_;
        /* Variable: signed
         *  Number of neighboring subdomains in <Subdomain::communicator> with ranks lower than <rankWorld>. */
        unsigned short     signed_;
        /* Variable: deficiency
         *  Dimension of the kernel of <Subdomain::a>. */
        unsigned short deficiency_;
#if HPDDM_FETI || HPDDM_BDD
        /* Function: solveGEVP
         *
         *  Solves the GenEO problem.
         *
         * Template Parameter:
         *    L              - 'S'ymmetric or 'G'eneral transfer of the local Schur complements.
         *
         * Parameters:
         *    d              - Constant pointer to a partition of unity of the primal unknowns, cf. <Bdd::m>. */
        template<char L>
        void solveGEVP(const underlying_type<K>* const d) {
            Option& opt = *Option::get();
            const bool resetPrefix = (opt.getPrefix().size() == 0 && super::prefix().size() != 0);
            if(resetPrefix)
                opt.setPrefix(super::prefix());
            if(schur_) {
                K** send = Subdomain<K>::buff_;
                unsigned int size = std::accumulate(Subdomain<K>::map_.cbegin(), Subdomain<K>::map_.cend(), 0, [](unsigned int sum, const pairNeighbor& n) { return sum + (L == 'S' ? (n.second.size() * (n.second.size() + 1)) / 2 : n.second.size() * n.second.size()); });
                *send = new K[2 * size];
                K** recv = send + Subdomain<K>::map_.size();
                *recv = *send + size;
                if(L == 'S')
                    for(unsigned short i = 1; i < Subdomain<K>::map_.size(); ++i) {
                        send[i] = send[i - 1] + (Subdomain<K>::map_[i - 1].second.size() * (Subdomain<K>::map_[i - 1].second.size() + 1)) / 2;
                        recv[i] = recv[i - 1] + (Subdomain<K>::map_[i - 1].second.size() * (Subdomain<K>::map_[i - 1].second.size() + 1)) / 2;
                    }
                else
                    for(unsigned short i = 1; i < Subdomain<K>::map_.size(); ++i) {
                        send[i] = send[i - 1] + Subdomain<K>::map_[i - 1].second.size() * Subdomain<K>::map_[i - 1].second.size();
                        recv[i] = recv[i - 1] + Subdomain<K>::map_[i - 1].second.size() * Subdomain<K>::map_[i - 1].second.size();
                    }
                K* res = new K[Subdomain<K>::dof_ * Subdomain<K>::dof_];
                exchangeSchurComplement<L>(send, recv, res);
                unsigned short nu = opt.val<unsigned short>("geneo_nu", 20);
                const underlying_type<K> threshold = opt.val("geneo_threshold", 0.0);
                Eigensolver<K> evp(nu >= 10 ? (nu >= 40 ? 1.0e-14 : 1.0e-12) : 1.0e-8, threshold, Subdomain<K>::dof_, nu);
                K* A;
                if(size < Subdomain<K>::dof_ * Subdomain<K>::dof_)
                    A = new K[Subdomain<K>::dof_ * Subdomain<K>::dof_];
                else
                    A = *recv;
                Blas<K>::lacpy("L", &(Subdomain<K>::dof_), &(Subdomain<K>::dof_), schur_, &(Subdomain<K>::dof_), A, &(Subdomain<K>::dof_));
                if(d)
                    for(unsigned int i = 0; i < Subdomain<K>::dof_; ++i)
                        for(unsigned int j = i; j < Subdomain<K>::dof_; ++j)
                            res[j + i * Subdomain<K>::dof_] *= d[i] * d[j];
                int flag, info;
                Lapack<K>::potrf("L", &(Subdomain<K>::dof_), res, &(Subdomain<K>::dof_), &flag);
                Lapack<K>::gst(&i__1, "L", &(Subdomain<K>::dof_), A, &(Subdomain<K>::dof_), res, &(Subdomain<K>::dof_), &flag);
                int lwork = -1;
                {
                    K wkopt;
                    Lapack<K>::trd("L", &(Subdomain<K>::dof_), nullptr, &(Subdomain<K>::dof_), nullptr, nullptr, nullptr, &wkopt, &lwork, &info);
                    lwork = std::real(wkopt);
                }
                MPI_Testall(Subdomain<K>::map_.size(), Subdomain<K>::rq_ + Subdomain<K>::map_.size(), &flag, MPI_STATUSES_IGNORE);
                K* work;
                const int storage = !Wrapper<K>::is_complex ? 4 * Subdomain<K>::dof_ - 1 : 2 * Subdomain<K>::dof_;
                if(flag) {
                    if((lwork + storage) <= size || (A != *recv && (lwork + storage) <= 2 * size))
                        work = *send;
                    else
                        work = new K[lwork + storage];
                }
                else {
                    if(A != *recv && (lwork + storage) <= size)
                        work = *recv;
                    else
                        work = new K[lwork + storage];
                }
                {
                    K* tau = work + lwork;
                    underlying_type<K>* d = reinterpret_cast<underlying_type<K>*>(tau + Subdomain<K>::dof_);
                    underlying_type<K>* e = d + Subdomain<K>::dof_;
                    Lapack<K>::trd("L", &(Subdomain<K>::dof_), A, &(Subdomain<K>::dof_), d, e, tau, work, &lwork, &info);
                    underlying_type<K> vl = -1.0 / HPDDM_EPS;
                    underlying_type<K> vu = threshold;
                    int iu = evp.nu_;
                    int nsplit;
                    underlying_type<K>* evr = e + Subdomain<K>::dof_ - 1;
                    int* iblock = new int[5 * Subdomain<K>::dof_];
                    int* isplit = iblock + Subdomain<K>::dof_;
                    int* iwork = isplit + Subdomain<K>::dof_;
                    char range = threshold > 0.0 ? 'V' : 'I';
                    underlying_type<K> tol = evp.getTol();
                    Lapack<K>::stebz(&range, "B", &(Subdomain<K>::dof_), &vl, &vu, &i__1, &iu, &tol, d, e, &evp.nu_, &nsplit, evr, iblock, isplit, reinterpret_cast<underlying_type<K>*>(work), iwork, &info);
                    if(evp.nu_) {
                        if(super::ev_) {
                            delete [] *super::ev_;
                            delete [] super::ev_;
                        }
                        super::ev_ = new K*[evp.nu_];
                        *super::ev_ = new K[Subdomain<K>::dof_ * evp.nu_];
                        for(unsigned short i = 1; i < evp.nu_; ++i)
                            super::ev_[i] = *super::ev_ + i * Subdomain<K>::dof_;
                        int* ifailv = new int[evp.nu_];
                        Lapack<K>::stein(&(Subdomain<K>::dof_), d, e, &(evp.nu_), evr, iblock, isplit, *super::ev_, &(Subdomain<K>::dof_), reinterpret_cast<underlying_type<K>*>(work), iwork, ifailv, &info);
                        delete [] ifailv;
                        Lapack<K>::mtr("L", "L", "N", &(Subdomain<K>::dof_), &(evp.nu_), A, &(Subdomain<K>::dof_), tau, *super::ev_, &(Subdomain<K>::dof_), work, &lwork, &info);
                        if(!Wrapper<K>::is_complex)
                            lwork += 3 * Subdomain<K>::dof_ - 1;
                        else
                            lwork += 4 * Subdomain<K>::dof_ - 1;
                    }
                    delete [] iblock;
                }
                if(evp.nu_) {
                    if(*(reinterpret_cast<underlying_type<K>*>(work) + lwork) < 2 * evp.getTol()) {
                        deficiency_ = 1;
                        underlying_type<K> relative = *(reinterpret_cast<underlying_type<K>*>(work) + lwork);
                        while(deficiency_ < evp.nu_ && std::abs(*(reinterpret_cast<underlying_type<K>*>(work) + lwork + deficiency_) / relative) * std::cbrt(evp.getTol()) < 1)
                            ++deficiency_;
                    }
                    Lapack<K>::trtrs("L", "T", "N", &(Subdomain<K>::dof_), &(evp.nu_), res, &(Subdomain<K>::dof_), *super::ev_, &(Subdomain<K>::dof_), &info);
                }
                else if(super::ev_) {
                    delete [] *super::ev_;
                    delete []  super::ev_;
                    super::ev_ = nullptr;
                }
                evp.dump(work + lwork, super::ev_, Subdomain<K>::communicator_);
                if(threshold > 0.0)
                    evp.template selectNu<K, true>(work + lwork, super::ev_, Subdomain<K>::communicator_, deficiency_);
                opt["geneo_nu"] = evp.nu_;
                if(super::co_)
                    super::co_->setLocal(evp.nu_);
                if(A != *recv)
                    delete [] A;
                if(work != *recv && work != *send)
                    delete [] work;
                if(!flag)
                    MPI_Waitall(Subdomain<K>::map_.size(), Subdomain<K>::rq_ + Subdomain<K>::map_.size(), MPI_STATUSES_IGNORE);
                delete [] res;
                delete [] *send;
            }
            else
                opt["geneo_nu"] = 0;
            if(resetPrefix)
                opt.setPrefix("");
        }
        template<unsigned short excluded, class Operator, class Prcndtnr>
        std::pair<MPI_Request, const K*>* buildTwo(Prcndtnr* B, const MPI_Comm& comm) {
            if(!Option::get()->set(super::prefix("geneo_nu"))) {
                if(!super::co_)
                    super::co_ = new typename std::remove_reference<decltype(*super::co_)>::type;
                super::co_->setLocal(deficiency_);
            }
            return super::template buildTwo<excluded, Operator>(B, comm);
        }
#endif
    public:
        Schur() : bb_(), ii_(), bi_(), schur_(), work_(), structure_(), pinv_(), mult_(), signed_(), deficiency_() { }
        Schur(const Schur&) = delete;
        ~Schur() { dtor(); }
        void dtor() {
            super::super::dtor();
            super::dtor();
            delete bb_;
            bb_ = nullptr;
            delete bi_;
            bi_ = nullptr;
            delete ii_;
            ii_ = nullptr;
#if HPDDM_FETI || HPDDM_BDD
            if(!HPDDM_QR || !schur_)
                delete static_cast<Solver<K>*>(pinv_);
            else if(deficiency_)
                delete static_cast<QR<K>*>(pinv_);
            else
                delete [] static_cast<K*>(pinv_);
            pinv_ = nullptr;
#endif
            delete [] schur_;
            schur_ = nullptr;
            delete [] work_;
            work_ = nullptr;
        }
        /* Typedef: super
         *  Type of the immediate parent class <Preconditioner>. */
        typedef Preconditioner<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
            Solver, CoarseOperator,
#elif HPDDM_PETSC
            DMatrix,
#endif
            K> super;
        /* Function: initialize
         *  Sets <Schur::rankWorld> and <Schur::signed>, and allocates <Schur::mult>, <Schur::work>, and <Schur::structure>. */
        template<bool m>
        void initialize() {
            MPI_Comm_rank(Subdomain<K>::communicator_, &rankWorld_);
            for(const pairNeighbor& neighbor : Subdomain<K>::map_) {
                mult_ += neighbor.second.size();
                if(neighbor.first < rankWorld_)
                    ++signed_;
            }
            if(m) {
                work_ = new K[mult_ + Subdomain<K>::a_->n_];
                structure_ = work_ + mult_;
            }
            else {
                work_ = new K[Subdomain<K>::dof_ + Subdomain<K>::a_->n_];
                structure_ = work_ + Subdomain<K>::dof_;
            }
        }
#if HPDDM_FETI || HPDDM_BDD
        /* Function: callNumfact
         *  Factorizes <Subdomain::a>. */
        void callNumfact() {
            if(Subdomain<K>::a_) {
                pinv_ = new Solver<K>();
                Solver<K>* p = static_cast<Solver<K>*>(pinv_);
                if(deficiency_) {
#if defined(MUMPSSUB) || defined(PASTIXSUB)
                    p->numfact(Subdomain<K>::a_, true);
#else
                    for(unsigned short i = 0; i < deficiency_; ++i)
                        ii_->a_[ii_->ia_[((i + 1) * ii_->n_) / (deficiency_ + 1)] - 1] += HPDDM_PEN;
                    p->numfact(Subdomain<K>::a_);
                    for(unsigned short i = 0; i < deficiency_; ++i)
                        ii_->a_[ii_->ia_[((i + 1) * ii_->n_) / (deficiency_ + 1)] - 1] -= HPDDM_PEN;
#endif
                }
                else
                    p->numfact(Subdomain<K>::a_);
            }
            else
                std::cerr << "The matrix 'a_' has not been allocated => impossible to build the Neumann preconditioner" << std::endl;
        }
        /* Function: computeSchurComplement
         *  Computes the explicit Schur complement <Schur::schur>. */
        void computeSchurComplement() {
#if defined(MUMPSSUB) || defined(PASTIXSUB) || defined(MKL_PARDISOSUB)
            if(Subdomain<K>::a_) {
                delete ii_;
                ii_ = nullptr;
                if(!schur_) {
                    schur_ = new K[Subdomain<K>::dof_ * Subdomain<K>::dof_];
                    schur_[0] = Subdomain<K>::dof_;
#if defined(MKL_PARDISOSUB)
#pragma message("Consider changing your linear solver if you need to compute solutions of singular systems")
                    schur_[1] = bi_->m_;
#else
                    schur_[1] = bi_->m_ + 1;
#endif
                    super::s_.numfact(Subdomain<K>::a_, true, schur_);
                }
            }
            else
                std::cerr << "The matrix 'a_' has not been allocated => impossible to build the Schur complement" << std::endl;
#else
#pragma message("Consider changing your linear solver if you need to compute solutions of singular systems or Schur complements")
#endif
        }
        /* Function: callNumfactPreconditioner
         *  Factorizes <Schur::ii> if <Schur::schur> is not available. */
        void callNumfactPreconditioner() {
            if(!schur_) {
                if(ii_) {
                    if(ii_->n_)
                        super::s_.numfact(ii_);
                }
                else
                    std::cerr << "The matrix 'ii_' has not been allocated => impossible to build the Dirichlet preconditioner" << std::endl;
            }
        }
#endif
        /* Function: originalNumbering
         *
         *  Renumbers a vector according to the numbering of the user.
         *
         * Parameters:
         *    interface      - Numbering of the interface.
         *    in             - Input vector. */
        template<class Container>
        void originalNumbering(const Container& interface, K* const in) const {
            if(interface[0] != bi_->m_) {
                unsigned int end = Subdomain<K>::a_->n_;
                std::vector<K> backup(in + bi_->m_, in + end);
                unsigned int j = Subdomain<K>::dof_;
                while(j-- > 0 && j != interface[j]) {
                    std::copy_backward(in + interface[j] - j - 1, in + end - j - 1, in + end);
                    in[interface[j]] = backup[j];
                    end = interface[j];
                }
                if(j < Subdomain<K>::dof_) {
                    std::copy_backward(in, in + end - j - 1, in + end);
                    std::copy_n(backup.begin(), j + 1, in);
                }
            }
        }
        /* Function: renumber
         *
         *  Renumbers <Subdomain::a> and <Preconditioner::ev> to easily assemble <Schur::bb>, <Schur::ii>, and <Schur::bi>.
         *
         * Parameters:
         *    interface      - Numbering of the interface.
         *    f              - Right-hand side to renumber (optional). */
        template<bool trim = true, class Container = std::vector<int>>
        void renumber(const Container& interface, K* const& f = nullptr) {
            if(!interface.empty()) {
                if(!ii_) {
                    Subdomain<K>::dof_ = Subdomain<K>::a_->n_;
                    std::vector<signed int> vec;
                    vec.reserve(Subdomain<K>::dof_);
                    std::vector<std::vector<K>> deflationBoundary(
#if HPDDM_FETI || HPDDM_BDD
                            super::getLocal()
#else
                            0
#endif
                            );
                    for(std::vector<K>& deflation : deflationBoundary)
                        deflation.reserve(interface.size());
                    unsigned int j = 0;
                    for(unsigned int k = 0, i = 0; i < interface.size(); ++k) {
                        if(k == interface[i]) {
                            vec.emplace_back(++i);
#if HPDDM_FETI || HPDDM_BDD
                            for(unsigned short l = 0; l < deflationBoundary.size(); ++l)
                                deflationBoundary[l].emplace_back(super::ev_[l][k]);
#endif
                        }
                        else {
#if HPDDM_FETI || HPDDM_BDD
                            for(unsigned short l = 0; l < deflationBoundary.size(); ++l)
                                super::ev_[l][j] = super::ev_[l][k];
#endif
                            vec.emplace_back(-(++j));
                        }
                    }
                    for(unsigned int k = interface.back() + 1; k < Subdomain<K>::dof_; ++k) {
#if HPDDM_FETI || HPDDM_BDD
                        for(unsigned short l = 0; l < deflationBoundary.size(); ++l)
                            super::ev_[l][j] = super::ev_[l][k];
#endif
                        vec.emplace_back(-(++j));
                    }
#if HPDDM_FETI || HPDDM_BDD
                    for(unsigned short l = 0; l < deflationBoundary.size(); ++l)
                        std::copy(deflationBoundary[l].cbegin(), deflationBoundary[l].cend(), super::ev_[l] + Subdomain<K>::dof_ - interface.size());
#endif
                    std::vector<std::pair<unsigned int, K>> tmpInterior;
                    std::vector<std::pair<unsigned int, K>> tmpBoundary;
                    std::vector<std::vector<std::pair<unsigned int, K>>> tmpInteraction(interface.size());
                    tmpInterior.reserve(Subdomain<K>::a_->nnz_ * (Subdomain<K>::dof_ - interface.size()) / Subdomain<K>::dof_);
                    tmpBoundary.reserve(Subdomain<K>::a_->nnz_ * interface.size() / Subdomain<K>::dof_);
                    for(j = 0; j < interface.size(); ++j)
                        tmpInteraction[j].reserve(std::max(Subdomain<K>::a_->ia_[interface[j] + 1] - Subdomain<K>::a_->ia_[interface[j]] - 1, 0));
                    bb_ = new MatrixCSR<K>(interface.size(), interface.size(), true);
                    int* ii = new int[Subdomain<K>::dof_ + 1];
                    ii[0] = 0;
                    bb_->ia_[0] = (Wrapper<K>::I == 'F');
                    std::pair<std::vector<int>, std::vector<int>> boundaryCond;
                    if(!Subdomain<K>::a_->sym_) {
                        boundaryCond.first.reserve(Subdomain<K>::dof_ - interface.size());
                        boundaryCond.second.reserve(interface.size());
                    }
                    for(unsigned int i = 0; i < Subdomain<K>::dof_; ++i) {
                        signed int row = vec[i];
                        unsigned int stop;
                        if(!Subdomain<K>::a_->sym_)
                            stop = std::distance(Subdomain<K>::a_->ja_, std::upper_bound(Subdomain<K>::a_->ja_ + Subdomain<K>::a_->ia_[i], Subdomain<K>::a_->ja_ + Subdomain<K>::a_->ia_[i + 1], i));
                        else
                            stop = Subdomain<K>::a_->ia_[i + 1];
                        if(!Subdomain<K>::a_->sym_) {
                            bool isBoundaryCond = true;
                            for(j = Subdomain<K>::a_->ia_[i]; j < Subdomain<K>::a_->ia_[i + 1] && isBoundaryCond; ++j) {
                                if(i != Subdomain<K>::a_->ja_[j] && (!trim || std::abs(Subdomain<K>::a_->a_[j]) > HPDDM_EPS))
                                    isBoundaryCond = false;
                                else if(i == Subdomain<K>::a_->ja_[j] && (!trim || std::abs(Subdomain<K>::a_->a_[j] - K(1.0)) > HPDDM_EPS))
                                    isBoundaryCond = false;
                            }
                            if(isBoundaryCond) {
                                if(row > 0)
                                    boundaryCond.second.push_back(row);
                                else
                                    boundaryCond.first.push_back(-row);
                            }
                        }
                        for(j = Subdomain<K>::a_->ia_[i]; j < stop; ++j) {
                            const K val = Subdomain<K>::a_->a_[j];
                            if(!trim || std::abs(val) > HPDDM_EPS) {
                                const int col = vec[Subdomain<K>::a_->ja_[j]];
                                if(col > 0) {
                                    const bool cond = !std::binary_search(boundaryCond.second.cbegin(), boundaryCond.second.cend(), col);
                                    if(row < 0 && cond)
                                        tmpInteraction[col - 1].emplace_back(-row - (Wrapper<K>::I != 'F'), val);
                                    else if(col == row || cond)
                                        tmpBoundary.emplace_back(col - (Wrapper<K>::I != 'F'), val);
                                }
                                else if(col == row || !std::binary_search(boundaryCond.first.cbegin(), boundaryCond.first.cend(), -col)) {
                                    if(row < 0)
                                        tmpInterior.emplace_back(-col - 1, val);
                                    else
                                        tmpInteraction[row - 1].emplace_back(-col - (Wrapper<K>::I != 'F'), val);
                                }
                            }
                        }
                        if(row < 0)
                            ii[-row] = tmpInterior.size();
                        else
                            bb_->ia_[row] = tmpBoundary.size() + (Wrapper<K>::I == 'F');
                    }
                    for(j = 0; j < tmpInterior.size(); ++j) {
                        Subdomain<K>::a_->ja_[j] = tmpInterior[j].first;
                        Subdomain<K>::a_->a_[j] = tmpInterior[j].second;
                    }
                    bi_ = new MatrixCSR<K>(interface.size(), Subdomain<K>::dof_ - interface.size(), std::accumulate(tmpInteraction.cbegin(), tmpInteraction.cend(), 0, [](unsigned int sum, const std::vector<std::pair<unsigned int, K>>& v) { return sum + v.size(); }), false);
                    bi_->ia_[0] = (Wrapper<K>::I == 'F');
                    for(unsigned int i = 0, j = 0; i < tmpInteraction.size(); ++i) {
                        std::sort(tmpInteraction[i].begin(), tmpInteraction[i].end(), [](const std::pair<unsigned int, K>& lhs, const std::pair<unsigned int, K>& rhs) { return lhs.first < rhs.first; });
                        for(const std::pair<unsigned int, K>& p : tmpInteraction[i]) {
                            bi_->ja_[j] = p.first;
                            bi_->a_[j++] = p.second;
                        }
                        bi_->ia_[i + 1] = j + (Wrapper<K>::I == 'F');
                    }
                    bb_->nnz_ = tmpBoundary.size();
                    bb_->a_ = new K[bb_->nnz_];
                    bb_->ja_ = new int[bb_->nnz_];
                    for(j = 0; j < tmpBoundary.size(); ++j) {
                        bb_->ja_[j] = tmpBoundary[j].first;
                        bb_->a_[j] = tmpBoundary[j].second;
                    }
                    for(unsigned int i = 0; i < bb_->n_; ++i) {
                        if(Wrapper<K>::I == 'F')
                            for(j = 0; j < bi_->ia_[i + 1] - bi_->ia_[i]; ++j)
                                Subdomain<K>::a_->ja_[ii[bi_->m_ + i] + j] = bi_->ja_[bi_->ia_[i] - 1 + j] - 1;
                        else
                            std::copy(bi_->ja_ + bi_->ia_[i], bi_->ja_ + bi_->ia_[i + 1], Subdomain<K>::a_->ja_ + ii[bi_->m_ + i]);
                        std::copy(bi_->a_ + bi_->ia_[i] - (Wrapper<K>::I == 'F'), bi_->a_ + bi_->ia_[i + 1] - (Wrapper<K>::I == 'F'), Subdomain<K>::a_->a_ + ii[bi_->m_ + i]);
                        ii[bi_->m_ + i + 1] = ii[bi_->m_ + i] + bi_->ia_[i + 1] - bi_->ia_[i] - (bb_->ia_[i] - (Wrapper<K>::I == 'F'));
                        for(j = bb_->ia_[i] - (Wrapper<K>::I == 'F'); j < bb_->ia_[i + 1] - (Wrapper<K>::I == 'F'); ++j)
                            Subdomain<K>::a_->ja_[ii[bi_->m_ + i + 1] + j] = bb_->ja_[j] - (Wrapper<K>::I == 'F') + bi_->m_;
                        std::copy(bb_->a_ + bb_->ia_[i] - (Wrapper<K>::I == 'F'), bb_->a_ + bb_->ia_[i + 1] - (Wrapper<K>::I == 'F'), Subdomain<K>::a_->a_ + ii[bi_->m_ + i + 1] + bb_->ia_[i] - (Wrapper<K>::I == 'F'));
                        ii[bi_->m_ + i + 1] += bb_->ia_[i + 1] - (Wrapper<K>::I == 'F');
                    }
                    delete [] Subdomain<K>::a_->ia_;
                    Subdomain<K>::a_->ia_ = ii;
                    Subdomain<K>::a_->nnz_ = ii[Subdomain<K>::dof_];
                    Subdomain<K>::a_->sym_ = true;
                    ii_ = new MatrixCSR<K>(bi_->m_, bi_->m_, Subdomain<K>::a_->ia_[bi_->m_], Subdomain<K>::a_->a_, Subdomain<K>::a_->ia_, Subdomain<K>::a_->ja_, true);
                    Subdomain<K>::dof_ = bb_->n_;
                }
                if(f && interface[0] != bi_->m_) {
                    std::vector<K> backup;
                    backup.reserve(interface.size());
                    backup.emplace_back(f[interface[0]]);
                    unsigned int j = 0;
                    unsigned int start = 0;
                    while(++j < interface.size()) {
                        std::copy(f + interface[j - 1] + 1, f + interface[j], f + start);
                        start = interface[j] - j;
                        backup.emplace_back(f[interface[j]]);
                    }
                    std::copy(f + interface.back() + 1, f + Subdomain<K>::a_->n_, f + start);
                    std::copy(backup.cbegin(), backup.cend(), f + bi_->m_);
                }
            }
            else {
                std::cerr << "The container of the interface is empty => no static condensation" << std::endl;
                Subdomain<K>::dof_ = 0;
            }
        }
        /* Function: stiffnessScaling
         *
         *  Builds the stiffness scaling, cf. <Bdd::buildScaling> and <Feti::buildScaling>.
         *
         * Parameter:
         *    pt             - Reference to the array in which to store the values. */
        void stiffnessScaling(K* const& pt) {
            if(bb_) {
                for(unsigned int i = 0; i < Subdomain<K>::dof_; ++i) {
                    unsigned int idx = bb_->ia_[i + 1] - (Wrapper<K>::I == 'F' ? 2 : 1);
                    if(bb_->ja_[idx] != i + (Wrapper<K>::I == 'F')) {
                        std::cerr << "The matrix 'bb_' seems to be ill-formed" << std::endl;
                        pt[i] = 0;
                    }
                    else
                        pt[i] = bb_->a_[idx];
                }
            }
            else
                std::cerr << "The matrix 'bb_' has not been allocated => impossible to build the stiffness scaling" << std::endl;
        }
        /* Function: getMult
         *  Returns the value of <Schur::mult>. */
        int getMult() const { return mult_; }
        /* Function: getSigned
         *  Returns the value of <Schur::signed>. */
        unsigned short getSigned() const { return signed_; }
#if HPDDM_FETI || HPDDM_BDD
        /* Function: applyLocalSchurComplement(n)
         *
         *  Applies the local Schur complement to multiple right-hand sides.
         *
         * Parameters:
         *    u              - Input vectors.
         *    n              - Number of input vectors.
         *
         * See also: <Feti::applyLocalPreconditioner(n)>. */
        void applyLocalSchurComplement(K*& in, const int& n) const {
            K* out = new K[n * Subdomain<K>::dof_]();
            if(!schur_) {
                if(bi_->m_) {
                    K* tmp = new K[n * bi_->m_];
                    Wrapper<K>::template csrmm<Wrapper<K>::I>(&(Wrapper<K>::transc), &(Subdomain<K>::dof_), &n, &bi_->m_, &(Wrapper<K>::d__1), false, bi_->a_, bi_->ia_, bi_->ja_, in, &(Wrapper<K>::d__0), tmp);
                    super::s_.solve(tmp, n);
                    Wrapper<K>::template csrmm<Wrapper<K>::I>("N", &(Subdomain<K>::dof_), &n, &bi_->m_, &(Wrapper<K>::d__1), false, bi_->a_, bi_->ia_, bi_->ja_, tmp, &(Wrapper<K>::d__0), out);
                    delete [] tmp;
                }
                Wrapper<K>::template csrmm<Wrapper<K>::I>("N", &(Subdomain<K>::dof_), &n, &(Subdomain<K>::dof_), &(Wrapper<K>::d__1), true, bb_->a_, bb_->ia_, bb_->ja_, in, &(Wrapper<K>::d__2), out);
            }
            else
                Blas<K>::symm("L", "L", &(Subdomain<K>::dof_), &n, &(Wrapper<K>::d__1), schur_, &(Subdomain<K>::dof_), in, &(Subdomain<K>::dof_), &(Wrapper<K>::d__0), out, &(Subdomain<K>::dof_));
            delete [] in;
            in = out;
        }
        /* Function: applyLocalSchurComplement
         *
         *  Applies the local Schur complement to a single right-hand side.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector (optional).
         *
         * See also: <Feti::applyLocalPreconditioner> and <Bdd::apply>. */
        void applyLocalSchurComplement(K* const in, K* const& out = nullptr) const {
            if(!schur_) {
                Wrapper<K>::template csrmv<Wrapper<K>::I>(&(Wrapper<K>::transc), &(Subdomain<K>::dof_), &bi_->m_, &(Wrapper<K>::d__1), false, bi_->a_, bi_->ia_, bi_->ja_, in, &(Wrapper<K>::d__0), work_);
                if(bi_->m_)
                    super::s_.solve(work_);
                if(out) {
                    Wrapper<K>::template csrmv<Wrapper<K>::I>("N", &(Subdomain<K>::dof_), &bi_->m_, &(Wrapper<K>::d__1), false, bi_->a_, bi_->ia_, bi_->ja_, work_, &(Wrapper<K>::d__0), out);
                    Wrapper<K>::template csrmv<Wrapper<K>::I>("N", &(Subdomain<K>::dof_), &(Subdomain<K>::dof_), &(Wrapper<K>::d__1), true, bb_->a_, bb_->ia_, bb_->ja_, in, &(Wrapper<K>::d__2), out);
                }
                else {
                    Wrapper<K>::template csrmv<Wrapper<K>::I>("N", &(Subdomain<K>::dof_), &bi_->m_, &(Wrapper<K>::d__1), false, bi_->a_, bi_->ia_, bi_->ja_, work_, &(Wrapper<K>::d__0), work_ + bi_->m_);
                    Wrapper<K>::template csrmv<Wrapper<K>::I>("N", &(Subdomain<K>::dof_), &(Subdomain<K>::dof_), &(Wrapper<K>::d__1), true, bb_->a_, bb_->ia_, bb_->ja_, in, &(Wrapper<K>::d__2), work_ + bi_->m_);
                    std::copy_n(work_ + bi_->m_, Subdomain<K>::dof_, in);
                }
            }
            else if(out)
                Blas<K>::symv("L", &(Subdomain<K>::dof_), &(Wrapper<K>::d__1), schur_, &(Subdomain<K>::dof_), in, &i__1, &(Wrapper<K>::d__0), out, &i__1);
            else {
                Blas<K>::symv("L", &(Subdomain<K>::dof_), &(Wrapper<K>::d__1), schur_, &(Subdomain<K>::dof_), in, &i__1, &(Wrapper<K>::d__0), work_ + bi_->m_, &i__1);
                std::copy_n(work_ + bi_->m_, Subdomain<K>::dof_, in);
            }
        }
        /* Function: applyLocalLumpedMatrix(n)
         *
         *  Applies the local lumped matrix <Schur::bb> to multiple right-hand sides.
         *
         * Parameters:
         *    u              - Input vectors.
         *    n              - Number of input vectors.
         *
         * See also: <Feti::applyLocalPreconditioner(n)>. */
        void applyLocalLumpedMatrix(K*& in, const int& n) const {
            K* out = new K[n * Subdomain<K>::dof_];
            Wrapper<K>::template csrmm<Wrapper<K>::I>("N", &(Subdomain<K>::dof_), &n, &(Subdomain<K>::dof_), &(Wrapper<K>::d__1), true, bb_->a_, bb_->ia_, bb_->ja_, in, &(Wrapper<K>::d__0), out);
            delete [] in;
            in = out;
        }
        /* Function: applyLocalLumpedMatrix
         *
         *  Applies the local lumped matrix <Schur::bb> to a single right-hand side.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector (optional).
         *
         * See also: <Feti::applyLocalPreconditioner>. */
        void applyLocalLumpedMatrix(K* const in) const {
            Wrapper<K>::template csrmv<Wrapper<K>::I>("N", &(Subdomain<K>::dof_), &(Subdomain<K>::dof_), &(Wrapper<K>::d__1), true, bb_->a_, bb_->ia_, bb_->ja_, in, &(Wrapper<K>::d__0), work_ + bi_->m_);
            std::copy_n(work_ + bi_->m_, Subdomain<K>::dof_, in);
        }
        /* Function: applyLocalSuperlumpedMatrix(n)
         *
         *  Applies the local superlumped matrix diag(<Schur::bb>) to multiple right-hand sides.
         *
         * Parameters:
         *    u              - Input vectors.
         *    n              - Number of input vectors.
         *
         * See also: <Feti::applyLocalPreconditioner(n)>. */
        void applyLocalSuperlumpedMatrix(K*& in, const int& n) const {
            for(unsigned int i = 0; i < Subdomain<K>::dof_; ++i) {
                K d = bb_->a_[bb_->ia_[i + 1] - (Wrapper<K>::I == 'F' ? 2 : 1)];
                for(int j = 0; j < n; ++j)
                    in[i + j * Subdomain<K>::dof_] *= d;
            }
        }
        /* Function: applyLocalSuperlumpedMatrix
         *
         *  Applies the local superlumped matrix diag(<Schur::bb>) to a single right-hand side.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector (optional).
         *
         * See also: <Feti::applyLocalPreconditioner>. */
        void applyLocalSuperlumpedMatrix(K* const in) const {
            for(unsigned int i = 0; i < Subdomain<K>::dof_; ++i)
                in[i] *= bb_->a_[bb_->ia_[i + 1] - (Wrapper<K>::I == 'F' ? 2 : 1)];
        }
#endif
        /* Function: getRank
         *  Returns the value of <Schur::rankWorld>. */
        int getRank() const { return rankWorld_; }
        /* Function: getLDR
         *  Returns the address of the leading dimension of <Preconditioner::ev>. */
        const int* getLDR() const { return schur_ ? &bi_->n_ : &(super::a_->n_); }
        /* Function: getEliminated
         *  Returns the number of eliminated unknowns of <Subdomain<K>::a>, i.e. the number of columns of <Schur::bi>. */
        unsigned int getEliminated() const { return bi_ ? bi_->m_ : 0; }
        /* Function: setDeficiency
         *  Sets <Schur::deficiency>. */
        void setDeficiency(unsigned short deficiency) { deficiency_ = deficiency; }
#if HPDDM_FETI || HPDDM_BDD
        /* Function: condensateEffort
         *
         *  Performs static condensation.
         *
         * Parameters:
         *    f              - Input right-hand side.
         *    b              - Condensed right-hand side. */
        void condensateEffort(const K* const f, K* const b) const {
            if(bi_->m_)
                super::s_.solve(f, structure_);
            std::copy_n(f + bi_->m_, Subdomain<K>::dof_, b ? b : structure_ + bi_->m_);
            Wrapper<K>::template csrmv<Wrapper<K>::I>("N", &(Subdomain<K>::dof_), &bi_->m_, &(Wrapper<K>::d__2), false, bi_->a_, bi_->ia_, bi_->ja_, structure_, &(Wrapper<K>::d__1), b ? b : structure_ + bi_->m_);
        }
        /* Function: computeResidual
         *
         *  Computes the norms of right-hand sides and residual vectors.
         *
         * Parameters:
         *    x              - Solution vector.
         *    f              - Right-hand side.
         *    storage        - Array to store both values.
         *
         * See also: <Schwarz::computeResidual>. */
        void computeResidual(const K* const x, const K* const f, underlying_type<K>* const storage, const unsigned short, const unsigned short) const {
            K* tmp = new K[Subdomain<K>::a_->n_];
            std::copy_n(f, Subdomain<K>::a_->n_, tmp);
            bool allocate = Subdomain<K>::setBuffer();
            Subdomain<K>::exchange(tmp + bi_->m_);
            storage[0] = 0.0;
            for(unsigned short i = 0; i < Subdomain<K>::map_.size(); ++i)
                for(unsigned int j = 0; j < Subdomain<K>::map_[i].second.size(); ++j) {
                    bool boundary = (std::abs(Subdomain<K>::boundaryCond(bi_->m_ + i)) > HPDDM_EPS);
                    if(boundary && std::abs(f[bi_->m_ + Subdomain<K>::map_[i].second[j]]) > HPDDM_EPS * HPDDM_PEN)
                        storage[0] += std::real(std::conj(f[bi_->m_ + Subdomain<K>::map_[i].second[j]]) * Subdomain<K>::buff_[i][j]) / (underlying_type<K>(HPDDM_PEN * HPDDM_PEN));
                    else
                        storage[0] += std::real(std::conj(f[bi_->m_ + Subdomain<K>::map_[i].second[j]]) * Subdomain<K>::buff_[i][j]);
                }
            Wrapper<K>::csrmv(Subdomain<K>::a_->sym_, &(Subdomain<K>::a_->n_), Subdomain<K>::a_->a_, Subdomain<K>::a_->ia_, Subdomain<K>::a_->ja_, x, work_);
            Subdomain<K>::exchange(work_ + bi_->m_);
            Subdomain<K>::clearBuffer(allocate);
            Blas<K>::axpy(&(Subdomain<K>::a_->n_), &(Wrapper<K>::d__2), tmp, &i__1, work_, &i__1);
            storage[1] = std::real(Blas<K>::dot(&bi_->m_, work_, &i__1, work_, &i__1));
            std::fill_n(tmp, Subdomain<K>::dof_, K(1.0));
            for(const pairNeighbor& neighbor : Subdomain<K>::map_)
                for(const pairNeighbor::second_type::value_type& val : neighbor.second)
                        tmp[val] /= K(1.0) + tmp[val];
            for(unsigned short i = 0; i < Subdomain<K>::dof_; ++i) {
                bool boundary = (std::abs(Subdomain<K>::boundaryCond(i)) > HPDDM_EPS);
                if(!boundary) {
                    storage[0] += std::norm(f[i]);
                    storage[1] += std::real(tmp[i]) * std::norm(work_[bi_->m_ + i]);
                }
                else
                    storage[0] += std::norm(f[i] / underlying_type<K>(HPDDM_PEN));
            }
            delete [] tmp;
            MPI_Allreduce(MPI_IN_PLACE, storage, 2, Wrapper<K>::mpi_underlying_type(), MPI_SUM, Subdomain<K>::communicator_);
            storage[0] = std::sqrt(storage[0]);
            storage[1] = std::sqrt(storage[1]);
        }
#endif
        /* Function: getAllDof
         *  Returns the number of local interior and boundary degrees of freedom (with the right multiplicity). */
        unsigned int getAllDof() const {
            unsigned int dof = Subdomain<K>::a_->n_;
            for(unsigned int k = 0; k < Subdomain<K>::dof_; ++k) {
                bool exit = false;
                for(unsigned short i = 0; i < Subdomain<K>::map_.size() && Subdomain<K>::map_[i].first < rankWorld_ && !exit; ++i)
                    for(unsigned int j = 0; j < Subdomain<K>::map_[i].second.size() && !exit; ++j)
                        if(Subdomain<K>::map_[i].second[j] == k) {
                            --dof;
                            exit = true;
                        }
            }
            return dof;
        }
        template<class T, char N = HPDDM_NUMBERING>
        void distributedNumbering(T* const in, T& first, T& last, long long& global) const {
            Subdomain<K>::template globalMapping<N>(in + bi_->m_, in + Subdomain<K>::a_->n_, first, last, global);
            long long independent = bi_->m_;
            MPI_Allreduce(MPI_IN_PLACE, &independent, 1, MPI_LONG_LONG, MPI_SUM, Subdomain<K>::communicator_);
            global += independent;
            std::for_each(in + bi_->m_, in + Subdomain<K>::a_->n_, [&](T& i) { i += independent; });
            independent = bi_->m_;
            MPI_Exscan(MPI_IN_PLACE, &independent, 1, MPI_LONG_LONG, MPI_SUM, Subdomain<K>::communicator_);
            int rank;
            MPI_Comm_rank(Subdomain<K>::communicator_, &rank);
            if(!rank)
                independent = 0;
            std::iota(in, in + bi_->m_, independent + (N == 'F'));
        }
        template<class T>
        bool distributedCSR(T* const num, T first, T last, T*& ia, T*& ja, K*& c) const {
            return Subdomain<K>::distributedCSR(num, first, last, ia, ja, c, bb_);
        }
};
} // HPDDM
#endif // HPDDM_SCHUR_HPP_
