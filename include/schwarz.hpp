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
    private:
        /* Variable: d
         *  Local partition of unity. */
        const underlying_type<K>* _d;
        std::size_t            _hash;
        /* Variable: type
         *  Type of <Prcndtnr> used in <Schwarz::apply> and <Schwarz::deflation>. */
        Prcndtnr               _type;
    public:
        Schwarz() : _d(), _hash() { }
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
#if HPDDM_SCHWARZ
        /* Function: callNumfact
         *  Factorizes <Subdomain::a> or another user-supplied matrix, useful for <Prcndtnr::OS> and <Prcndtnr::OG>. */
        template<char N = HPDDM_NUMBERING>
        void callNumfact(MatrixCSR<K>* const& A = nullptr) {
            const std::string prefix = super::prefix();
            Option& opt = *Option::get();
            unsigned short m = opt.val<unsigned short>(prefix + "schwarz_method");
            if(A) {
                std::size_t hash = A->hashIndices();
                if(_hash != hash) {
                    _hash = hash;
                    super::destroySolver();
                }
            }
            switch(m) {
                case 2:  _type = (A ? Prcndtnr::OS : Prcndtnr::SY); break;
                case 3:  _type = Prcndtnr::SY; break;
                case 5:  _type = Prcndtnr::NO; return;
                default: _type = (A && (m == 1 || m == 4) ? Prcndtnr::OG : Prcndtnr::GE);
            }
            m = opt.val<unsigned short>(prefix + "reuse_preconditioner");
            if(m <= 1)
                super::_s.template numfact<N>(_type == Prcndtnr::OS || _type == Prcndtnr::OG ? A : Subdomain<K>::_a);
            if(m >= 1)
                opt[prefix + "reuse_preconditioner"] += 1;
        }
        void setMatrix(MatrixCSR<K>* const& a) {
            bool fact = super::setMatrix(a) && _type != Prcndtnr::OS && _type != Prcndtnr::OG;
            if(fact) {
                super::destroySolver();
                super::_s.numfact(a);
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
        /* Function: getScaling
         *  Returns a constant pointer to <Schwarz::d>. */
        const underlying_type<K>* getScaling() const { return _d; }
        /* Function: scaledExchange */
        template<bool allocate = false>
        void scaledExchange(K* const x, const unsigned short& mu = 1) const {
            bool free = false;
            if(allocate)
                free = Subdomain<K>::setBuffer();
            Wrapper<K>::diag(Subdomain<K>::_dof, _d, x, mu);
            Subdomain<K>::exchange(x, mu);
            if(allocate)
                Subdomain<K>::clearBuffer(free);
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
                Wrapper<K>::diag(Subdomain<K>::_dof, _d, in, out, mu);                                                                                                                                                                                      // out = D in
                int tmp = mu;
                Blas<K>::gemm(&(Wrapper<K>::transc), "N", super::getAddrLocal(), &tmp, &(Subdomain<K>::_dof), &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), out, &(Subdomain<K>::_dof), &(Wrapper<K>::d__0), super::_uc, super::getAddrLocal()); // _uc = _ev^T D in
                super::_co->template callSolver<excluded>(super::_uc, mu);                                                                                                                                                                                  // _uc = E \ _ev^T D in
                Blas<K>::gemm("N", "N", &(Subdomain<K>::_dof), &tmp, super::getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), super::_uc, super::getAddrLocal(), &(Wrapper<K>::d__0), out, &(Subdomain<K>::_dof));                   // out = _ev E \ _ev^T D in
                scaledExchange(out, mu);
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
        std::pair<MPI_Request, const K*>* buildTwo(const MPI_Comm& comm) {
            return super::template buildTwo<excluded, MatrixMultiplication<Schwarz<Solver, CoarseSolver, S, K>, K>>(this, comm);
        }
        template<bool excluded = false>
        bool start(const K* const b, K* const x, const unsigned short& mu = 1) const {
            bool allocate = Subdomain<K>::setBuffer();
            if(!excluded) {
                const std::unordered_map<unsigned int, K> map = Subdomain<K>::boundaryConditions();
                for(const std::pair<unsigned int, K>& p : map)
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        x[nu * Subdomain<K>::_dof + p.first] = b[nu * Subdomain<K>::_dof + p.first] / p.second;
            }
            scaledExchange(x, mu);
            if(super::_co) {
                unsigned short k = 1;
                const std::string prefix = super::prefix();
                Option& opt = *Option::get();
                if(opt.any_of(prefix + "krylov_method", { 4, 5 }) && !opt.val<unsigned short>(prefix + "recycle_same_system"))
                    k = std::max(opt.val<int>(prefix + "recycle", 1), 1);
                super::start(mu * k);
                if(opt.val<char>(prefix + "schwarz_coarse_correction") == 2) {
                    if(opt.val<char>("geneo_force_uniformity") == 1)
                        opt[prefix + "schwarz_coarse_correction"] = 0;
                    else if(!excluded) {
                        K* tmp = new K[mu * Subdomain<K>::_dof];
                        GMV(x, tmp, mu);                                                  // tmp = A x
                        Blas<K>::axpby(mu * Subdomain<K>::_dof, 1.0, b, 1, -1.0, tmp, 1); // tmp = b - A x
                        deflation<excluded>(nullptr, tmp, mu);                            // tmp = Z E \ Z^T (b - A x)
                        int dim = mu * Subdomain<K>::_dof;
                        Blas<K>::axpy(&dim, &(Wrapper<K>::d__1), tmp, &i__1, x, &i__1);   //   x = x + Z E \ Z^T (b - A x)
                        delete [] tmp;
                    }
                    else
                        deflation<excluded>(nullptr, nullptr, mu);
                }
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
                        scaledExchange(out, mu);         // out = D A \ in
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
                int tmp = mu * Subdomain<K>::_dof;
                if(!work)
                    work = const_cast<K*>(in);
                else if(!excluded)
                    std::copy_n(in, tmp, work);
                if(correction == 1) {
#if HPDDM_ICOLLECTIVE
                    MPI_Request rq[2];
                    Ideflation<excluded>(in, out, mu, rq);
                    if(!excluded) {
                        super::_s.solve(work, mu);                                                                                                                                                         // out = A \ in
                        MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
                        int k = mu;
                        Blas<K>::gemm("N", "N", &(Subdomain<K>::_dof), &k, super::getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), super::_uc, super::getAddrLocal(), &(Wrapper<K>::d__0), out, &(Subdomain<K>::_dof));                   // out = _ev E \ _ev^T D in
                        Blas<K>::axpy(&tmp, &(Wrapper<K>::d__1), work, &i__1, out, &i__1);
                        scaledExchange(out, mu);                                                                                                                                                                                                              // out = Z E \ Z^T in + A \ in
                    }
                    else
                        MPI_Wait(rq + 1, MPI_STATUS_IGNORE);
#else
                    deflation<excluded>(in, out, mu);
                    if(!excluded) {
                        super::_s.solve(work, mu);
                        Blas<K>::axpy(&tmp, &(Wrapper<K>::d__1), work, &i__1, out, &i__1);
                        scaledExchange(out, mu);
                    }
#endif // HPDDM_ICOLLECTIVE
                }
                else if(correction == 2) {
                    if(!excluded) {
                        if(_type == Prcndtnr::OS)
                            Wrapper<K>::diag(Subdomain<K>::_dof, _d, work, mu);
                        super::_s.solve(work, out, mu);
                        scaledExchange(out, mu);
                        GMV(out, work, mu);
                        deflation<excluded>(nullptr, work, mu);
                        Blas<K>::axpy(&tmp, &(Wrapper<K>::d__2), work, &i__1, out, &i__1);
                    }
                    else
                        deflation<excluded>(nullptr, nullptr, mu);
                }
                else {
                    deflation<excluded>(in, out, mu);                                                                  // out = Z E \ Z^T in
                    if(!excluded) {
                        if(HPDDM_NUMBERING == Wrapper<K>::I)
                            Wrapper<K>::csrmm("N", &(Subdomain<K>::_dof), &(tmp = mu), &(Subdomain<K>::_dof), &(Wrapper<K>::d__2), Subdomain<K>::_a->_sym, Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, out, &(Wrapper<K>::d__1), work);
                        else if(Subdomain<K>::_a->_ia[Subdomain<K>::_dof] == Subdomain<K>::_a->_nnz)
                            Wrapper<K>::template csrmm<'C'>("N", &(Subdomain<K>::_dof), &(tmp = mu), &(Subdomain<K>::_dof), &(Wrapper<K>::d__2), Subdomain<K>::_a->_sym, Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, out, &(Wrapper<K>::d__1), work);
                        else
                            Wrapper<K>::template csrmm<'F'>("N", &(Subdomain<K>::_dof), &(tmp = mu), &(Subdomain<K>::_dof), &(Wrapper<K>::d__2), Subdomain<K>::_a->_sym, Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, out, &(Wrapper<K>::d__1), work);
                        scaledExchange(work, mu);                                                                      //  in = (I - A Z E \ Z^T) in
                        if(_type == Prcndtnr::OS)
                            Wrapper<K>::diag(Subdomain<K>::_dof, _d, work, mu);
                        super::_s.solve(work, mu);
                        scaledExchange(work, mu);                                                                      //  in = D A \ (I - A Z E \ Z^T) in
                        Blas<K>::axpy(&(tmp = mu * Subdomain<K>::_dof), &(Wrapper<K>::d__1), work, &i__1, out, &i__1); // out = D A \ (I - A Z E \ Z^T) in + Z E \ Z^T in
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
            const std::string prefix = super::prefix();
            Option& opt = *Option::get();
            const underlying_type<K>& threshold = opt.val(prefix + "geneo_threshold", 0.0);
            Eps<K> evp(threshold, Subdomain<K>::_dof, opt.template val<unsigned short>(prefix + "geneo_nu", 20));
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
            if(threshold > 0.0 && opt.template val<char>(prefix + "geneo_estimate_nu", 0) && (!B || B->hashIndices() == A->hashIndices())) {
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
            if(free) {
                A->_ia = nullptr;
                A->_ja = nullptr;
            }
            opt[prefix + "geneo_nu"] = evp._nu;
            if(super::_co)
                super::_co->setLocal(evp._nu);
            const int n = Subdomain<K>::_dof;
            std::for_each(super::_ev, super::_ev + evp._nu, [&](K* const v) { std::replace_if(v, v + n, [](K x) { return std::abs(x) < 1.0 / (HPDDM_EPS * HPDDM_PEN); }, K()); });
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
            scaledExchange(out, mu);
#endif
        }
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
        void computeResidual(const K* const x, const K* const f, underlying_type<K>* const storage, const unsigned short& mu = 1, const unsigned short norm = 0) const {
            int dim = mu * Subdomain<K>::_dof;
            K* tmp = new K[dim];
            bool allocate = Subdomain<K>::setBuffer();
            GMV(x, tmp, mu);
            Subdomain<K>::clearBuffer(allocate);
            Blas<K>::axpy(&dim, &(Wrapper<K>::d__2), f, &i__1, tmp, &i__1);
            std::fill_n(storage, 2 * mu, 0.0);
            if(norm == 1) {
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
            else if(norm == 2) {
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
            if(norm == 0 || norm == 1) {
                MPI_Allreduce(MPI_IN_PLACE, storage, 2 * mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, Subdomain<K>::_communicator);
                if(norm == 0)
                    std::for_each(storage, storage + 2 * mu, [](underlying_type<K>& b) { b = std::sqrt(b); });
            }
            else
                MPI_Allreduce(MPI_IN_PLACE, storage, 2 * mu, Wrapper<K>::mpi_underlying_type(), MPI_MAX, Subdomain<K>::_communicator);
        }
#endif
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
