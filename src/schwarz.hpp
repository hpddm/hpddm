/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <jolivet@ann.jussieu.fr>
              Frédéric Nataf <nataf@ann.jussieu.fr>
        Date: 2013-03-10

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

#ifndef _SCHWARZ_
#define _SCHWARZ_

#include <set>

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
template<template<class> class Solver, template<class> class CoarseSolver, char S, class K>
class Schwarz : public Preconditioner<Solver, CoarseOperator<CoarseSolver, S, K>, K> {
    public:
        /* Enum: Prcndtnr
         *
         *  Defines the Schwarz method used as a preconditioner.
         *
         * NO           - No preconditioner.
         * SY           - Symmetric preconditioner, e.g. Additive Schwarz method.
         * GE           - Nonsymmetric preconditioner, e.g. Restricted Additive Schwarz method.
         * OS           - Optimized symmetric preconditioner, e.g. Optimized Schwarz method.
         * OG           - Optimized nonsymmetric preconditioner, e.g. Optimized Restricted Additive Schwarz method.
         * AD           - Additive two-level Schwarz method. */
        enum class Prcndtnr : char {
            NO, SY, GE, OS, OG, AD
        };
    private:
        /* Variable: d
         *  Local partition of unity. */
        const typename Wrapper<K>::ul_type* _d;
        /* Variable: type
         *  Type of <Prcndtnr> used in <Schwarz::apply> and <Schwarz::deflation>. */
        Prcndtnr                         _type;
#if HPDDM_GMV
        std::vector<std::pair<std::vector<int>,
                    std::vector<int>>>    _map;
#endif
    public:
        Schwarz() : _d() { }
        ~Schwarz() { }
        /* Typedef: super
         *  Type of the immediate parent class <Preconditioner>. */
        typedef Preconditioner<Solver, CoarseOperator<CoarseSolver, S, K>, K> super;
        /* Function: initialize
         *  Sets <Schwarz::d>. */
        template<class Container = std::vector<int>>
        inline void initialize(typename Wrapper<K>::ul_type* const& d) {
            _d = d;
#if HPDDM_GMV
            _map.resize(Subdomain<K>::_map.size());
            for(unsigned short i = 0; i < Subdomain<K>::_map.size(); ++i) {
                _map[i].first.reserve(Subdomain<K>::_map[i].second.size());
                _map[i].second.reserve(Subdomain<K>::_map[i].second.size());
                for(unsigned int j = 0; j < Subdomain<K>::_map[i].second.size(); ++j) {
                    const unsigned int k = Subdomain<K>::_map[i].second[j];
                    if(std::abs(_d[k] - 1.0) < HPDDM_EPS)
                        _map[i].first.emplace_back(k);
                    else if(std::abs(_d[k]) < HPDDM_EPS)
                        _map[i].second.emplace_back(k);
                }
            }
#endif
        }
        /* Function: setType
         *  Sets <Schwarz::type>. */
        inline void setType(bool sym) {
            _type = sym ? Prcndtnr::SY : Prcndtnr::GE;
        }
        inline void setType(Prcndtnr t) {
            _type = t;
        }
        /* Function: callNumfact
         *  Factorizes <Subdomain::a> or another user-supplied matrix, useful for <Prcndtnr::OS> and <Prcndtnr::OG>. */
        inline void callNumfact(MatrixCSR<K>* const& A = nullptr) {
            if(A != nullptr) {
                if(_type == Prcndtnr::SY)
                    _type = Prcndtnr::OS;
                else
                    _type = Prcndtnr::OG;
            }
            super::_s.numfact(A ? A : Subdomain<K>::_a, _type == Prcndtnr::OS ? true : false);
        }
        /* Function: multiplicityScaling
         *
         *  Builds the multiplicity scaling.
         *
         * Parameter:
         *    d              - Array of values. */
        inline void multiplicityScaling(typename Wrapper<K>::ul_type* const d) const {
            for(unsigned short i = 0; i < Subdomain<K>::_map.size(); ++i) {
                typename Wrapper<K>::ul_type* const recv = reinterpret_cast<typename Wrapper<K>::ul_type*>(Subdomain<K>::_rbuff[i]);
                typename Wrapper<K>::ul_type* const send = reinterpret_cast<typename Wrapper<K>::ul_type*>(Subdomain<K>::_sbuff[i]);
                MPI_Irecv(recv, Subdomain<K>::_map[i].second.size(), Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), Subdomain<K>::_map[i].first, 0, Subdomain<K>::_communicator, Subdomain<K>::_rq + i);
                Wrapper<typename Wrapper<K>::ul_type>::gthr(Subdomain<K>::_map[i].second.size(), d, send, Subdomain<K>::_map[i].second.data());
                MPI_Isend(send, Subdomain<K>::_map[i].second.size(), Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), Subdomain<K>::_map[i].first, 0, Subdomain<K>::_communicator, Subdomain<K>::_rq + Subdomain<K>::_map.size() + i);
            }
            std::fill(d, d + Subdomain<K>::_dof, 1.0);
            for(unsigned short i = 0; i < Subdomain<K>::_map.size(); ++i) {
                int index;
                MPI_Waitany(Subdomain<K>::_map.size(), Subdomain<K>::_rq, &index, MPI_STATUS_IGNORE);
                typename Wrapper<K>::ul_type* const recv = reinterpret_cast<typename Wrapper<K>::ul_type*>(Subdomain<K>::_rbuff[index]);
                typename Wrapper<K>::ul_type* const send = reinterpret_cast<typename Wrapper<K>::ul_type*>(Subdomain<K>::_sbuff[index]);
                for(unsigned int j = 0; j < Subdomain<K>::_map[index].second.size(); ++j) {
                    if(std::abs(send[j]) < HPDDM_EPS)
                        d[Subdomain<K>::_map[index].second[j]] = 0.0;
                    else
                        d[Subdomain<K>::_map[index].second[j]] /= 1.0 + d[Subdomain<K>::_map[index].second[j]] * recv[j] / send[j];
                }
            }
            MPI_Waitall(Subdomain<K>::_map.size(), Subdomain<K>::_rq + Subdomain<K>::_map.size(), MPI_STATUSES_IGNORE);
        }
        /* Function: getScaling
         *  Returns a constant pointer to <Schwarz::d>. */
        inline const typename Wrapper<K>::ul_type* getScaling() const { return _d; }
        /* Function: deflation
         *
         *  Computes a coarse correction.
         *
         * Template parameter:
         *    excluded       - True if the master processes are excluded from the domain decomposition, false otherwise. 
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector.
         *    fuse           - Number of fused reductions (optional). */
        template<bool excluded>
        inline void deflation(const K* const in, K* const out, const unsigned short& fuse = 0) const {
            if(fuse > 0) {
                super::_co->reallocateRHS(const_cast<K*&>(super::_uc), fuse);
                std::copy(out + Subdomain<K>::_dof, out + Subdomain<K>::_dof + fuse, super::_uc + super::getLocal());
            }
            if(excluded)
                super::_co->template callSolver<excluded>(super::_uc, fuse);
            else {
                Wrapper<K>::diagv(Subdomain<K>::_dof, _d, in, out);                                                                                                                                                 // out = D in
                Wrapper<K>::gemv(&(Wrapper<K>::transc), &(Subdomain<K>::_dof), super::getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), out, &i__1, &(Wrapper<K>::d__0), super::_uc, &i__1); // _uc = _ev^T D in
                super::_co->template callSolver<excluded>(super::_uc, fuse);                                                                                                                                        // _uc = E \ _ev^T D in
                Wrapper<K>::gemv(&transa, &(Subdomain<K>::_dof), super::getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), super::_uc, &i__1, &(Wrapper<K>::d__0), out, &i__1);               // out = _ev E \ _ev^T D in
                if(_type != Prcndtnr::AD) {
                    Wrapper<K>::diagv(Subdomain<K>::_dof, _d, out);
                    Subdomain<K>::exchange(out);
                }
            }
            if(fuse > 0)
                std::copy(super::_uc + super::getLocal(), super::_uc + super::getLocal() + fuse, out + Subdomain<K>::_dof);
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
         *    rq             - MPI request to check completion of the MPI transfers.
         *    fuse           - Number of fused reductions (optional). */
        template<bool excluded>
        inline void Ideflation(const K* const in, K* const out, MPI_Request* rq, const unsigned short& fuse = 0) const {
            if(fuse > 0) {
                super::_co->reallocateRHS(const_cast<K*&>(super::_uc), fuse);
                std::copy(out + Subdomain<K>::_dof, out + Subdomain<K>::_dof + fuse, super::_uc + super::getLocal());
            }
            if(excluded)
                super::_co->template IcallSolver<excluded>(super::_uc, rq, fuse);
            else {
                Wrapper<K>::diagv(Subdomain<K>::_dof, _d, in, out);
                Wrapper<K>::gemv(&(Wrapper<K>::transc), &(Subdomain<K>::_dof), super::getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), out, &i__1, &(Wrapper<K>::d__0), super::_uc, &i__1);
                super::_co->template IcallSolver<excluded>(super::_uc, rq, fuse);
            }
            if(fuse > 0)
                std::copy(super::_uc + super::getLocal(), super::_uc + super::getLocal() + fuse, out + Subdomain<K>::_dof);
        }
#endif // HPDDM_ICOLLECTIVE
        template<bool excluded>
        inline void deflation(K* const out, const unsigned short& fuse = 0) const {
            deflation<excluded>(nullptr, out, fuse);
        }
        /* Function: buildTwo
         *
         *  Assembles and factorizes the coarse operator by calling <Preconditioner::buildTwo>.
         *
         * Template Parameter:
         *    excluded       - Greater than 0 if the master processes are excluded from the domain decomposition, equal to 0 otherwise.
         *
         * Parameters:
         *    comm           - Global MPI communicator.
         *    parm           - Vector of parameters.
         *
         * See also: <Bdd::buildTwo>, <Feti::buildTwo>. */
        template<unsigned short excluded = 0, class Container>
        inline std::pair<MPI_Request, const K*>* buildTwo(const MPI_Comm& comm, Container& parm) {
            MatrixMultiplication<Schwarz<Solver, CoarseSolver, S, K>, K> A(*this, parm[NU]);
            auto ret = super::template buildTwo<excluded, 2>(A, comm, parm);
            return ret;
        }
        /* Function: apply
         *
         *  Applies the global Schwarz preconditioner.
         *
         * Template Parameter:
         *    excluded       - Greater than 0 if the master processes are excluded from the domain decomposition, equal to 0 otherwise.
         *
         * Parameters:
         *    in             - Input vector, modified internally !
         *    out            - Output vector.
         *    fuse           - Number of fused reductions (optional). */
        template<bool excluded = false>
        inline void apply(K* const in, K* const out, const unsigned short& fuse = 0) const {
            if(!super::_co) {
                if(_type == Prcndtnr::NO)
                    std::copy(in, in + Subdomain<K>::_dof, out);
                else if(_type == Prcndtnr::SY || _type == Prcndtnr::OS) {
                    if(!excluded) {
                        super::_s.solve(in, out);
                        Wrapper<K>::diagv(Subdomain<K>::_dof, _d, out);
                        Subdomain<K>::exchange(out);                                                         // out = D A \ in
                    }
                }
                else {
                    if(!excluded) {
                        super::_s.solve(in, out);
                        Subdomain<K>::exchange(out);                                                         // out = A \ in
                    }
                }
            }
            else {
                if(_type == Prcndtnr::AD) {
#if HPDDM_ICOLLECTIVE
                    MPI_Request rq[2];
                    Ideflation<excluded>(in, out, rq, fuse);
                    if(!excluded) {
                        super::_s.solve(in);                                                                                                                                                                  // out = A \ in
                        MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
                        Wrapper<K>::gemv(&transa, &(Subdomain<K>::_dof), super::getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), super::_uc, &i__1, &(Wrapper<K>::d__0), out, &i__1); // out = Z E \ Z^T in
                        Wrapper<K>::axpy(&(Subdomain<K>::_dof), &(Wrapper<K>::d__1), in, &i__1, out, &i__1);
                        Wrapper<K>::diagv(Subdomain<K>::_dof, _d, out);
                        Subdomain<K>::exchange(out);                                                                                                                                                          // out = Z E \ Z^T in + A \ in
                    }
                    else
                        MPI_Wait(rq + 1, MPI_STATUS_IGNORE);
#else
                    deflation<excluded>(in, out, fuse);
                    if(!excluded) {
                        super::_s.solve(in);
                        Wrapper<K>::axpy(&(Subdomain<K>::_dof), &(Wrapper<K>::d__1), in, &i__1, out, &i__1);
                        Wrapper<K>::diagv(Subdomain<K>::_dof, _d, out);
                        Subdomain<K>::exchange(out);
                    }
#endif // HPDDM_ICOLLECTIVE
                }
                else {
                    deflation<excluded>(in, out, fuse);                                                      // out = Z E \ Z^T in
                    if(!excluded) {
                        Wrapper<K>::template csrmv<'C'>(&transa, &(Subdomain<K>::_dof), &(Subdomain<K>::_dof), &(Wrapper<K>::d__2), Subdomain<K>::_a->_sym, Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, out, &(Wrapper<K>::d__1), in);
                        Wrapper<K>::diagv(Subdomain<K>::_dof, _d, in);
                        Subdomain<K>::exchange(in);                                                          //  in = (I - A Z E \ Z^T) in
                        if(_type == Prcndtnr::OS)
                            Wrapper<K>::diagv(Subdomain<K>::_dof, _d, in);
                        super::_s.solve(in);
                        Wrapper<K>::diagv(Subdomain<K>::_dof, _d, in);
                        Subdomain<K>::exchange(in);                                                          //  in = D A \ (I - A Z E \ Z^T) in
                        Wrapper<K>::axpy(&(Subdomain<K>::_dof), &(Wrapper<K>::d__1), in, &i__1, out, &i__1); // out = D A \ (I - A Z E \ Z^T) in + Z E \ Z^T in
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
        inline void scaleIntoOverlap(const MatrixCSR<K>* const& A, MatrixCSR<K>*& B) const {
            std::set<unsigned int> intoOverlap;
            for(const pairNeighbor& neighbor : Subdomain<K>::_map)
                for(unsigned int i : neighbor.second)
                    if(_d[i] > HPDDM_EPS)
                        intoOverlap.insert(i);
            std::vector<std::vector<std::pair<unsigned int, K>>> tmp(intoOverlap.size());
            unsigned int k, iPrev = 0;
#pragma omp parallel for schedule(static, HPDDM_GRANULARITY) reduction(+ : iPrev)
            for(k = 0; k < intoOverlap.size(); ++k) {
                auto it = intoOverlap.begin();
                std::advance(it, k);
                tmp[k].reserve(A->_ia[*it + 1] - A->_ia[*it]);
                for(unsigned int j = A->_ia[*it]; j < A->_ia[*it + 1]; ++j) {
                    K value = _d[*it] * _d[A->_ja[j]] * A->_a[j];
                    if(std::abs(value) > HPDDM_EPS && intoOverlap.find(A->_ja[j]) != intoOverlap.cend())
                            tmp[k].emplace_back(A->_ja[j], value);
                }
                iPrev += tmp[k].size();
            }
            int nnz = iPrev;
            if(B != nullptr)
                delete B;
            B = new MatrixCSR<K>(Subdomain<K>::_dof, Subdomain<K>::_dof, nnz, A->_sym);
            nnz = iPrev = k = 0;
            for(unsigned int i : intoOverlap) {
                std::fill(B->_ia + iPrev, B->_ia + i + 1, nnz);
                for(const std::pair<unsigned int, K>& p : tmp[k]) {
                    B->_ja[nnz] = p.first;
                    B->_a[nnz++] = p.second;
                }
                ++k;
                iPrev = i + 1;
            }
            std::fill(B->_ia + iPrev, B->_ia + Subdomain<K>::_dof + 1, nnz);
        }
        /* Function: solveGEVP
         *
         *  Solves the generalized eigenvalue problem Ax = l Bx.
         *
         * Parameters:
         *    A              - Left-hand side matrix.
         *    B              - Right-hand side matrix (optional).
         *    nu             - Number of eigenvectors requested.
         *    threshold      - Precision of the eigensolver. */
        template<template<class> class Eps>
        inline void solveGEVP(MatrixCSR<K>* const& A, unsigned short& nu, const typename Wrapper<K>::ul_type& threshold, MatrixCSR<K>* const& B = nullptr, const MatrixCSR<K>* const& pattern = nullptr) {
            Eps<K> evp(threshold, Subdomain<K>::_dof, nu);
            bool free = pattern ? pattern->sameSparsity(A) : Subdomain<K>::_a->sameSparsity(A);
            MatrixCSR<K>* rhs = nullptr;
            if(B)
                rhs = B;
            else
                scaleIntoOverlap(A, rhs);
            evp.template solve<Solver>(A, rhs, super::_ev, Subdomain<K>::_communicator, free ? &(super::_s) : nullptr);
            if(rhs != B)
                delete rhs;
            if(free) {
                A->_ia = nullptr;
                A->_ja = nullptr;
            }
            nu = evp.getNu();
        }
        template<bool sorted = true, bool scale = false>
        inline void interaction(std::vector<const MatrixCSR<K>*>& blocks) const {
            Subdomain<K>::template interaction<'C', sorted, scale>(blocks, _d);
        }
#if HPDDM_GMV
        inline void optimized_exchange(K* const out) const {
            for(unsigned short i = 0; i < Subdomain<K>::_map.size(); ++i) {
                MPI_Irecv(Subdomain<K>::_rbuff[i], _map[i].second.size(), Wrapper<K>::mpi_type(), Subdomain<K>::_map[i].first, 1, Subdomain<K>::_communicator, Subdomain<K>::_rq + i);
                Wrapper<K>::gthr(_map[i].first.size(), out, Subdomain<K>::_sbuff[i], _map[i].first.data());
                MPI_Isend(Subdomain<K>::_sbuff[i], _map[i].first.size(), Wrapper<K>::mpi_type(), Subdomain<K>::_map[i].first, 1, Subdomain<K>::_communicator, Subdomain<K>::_rq + Subdomain<K>::_map.size() + i);
            }
            for(unsigned short i = 0; i < Subdomain<K>::_map.size(); ++i) {
                int index;
                MPI_Waitany(Subdomain<K>::_map.size(), Subdomain<K>::_rq, &index, MPI_STATUS_IGNORE);
                Wrapper<K>::sctr(_map[index].second.size(), Subdomain<K>::_rbuff[index], _map[index].second.data(), out);
            }
            MPI_Waitall(Subdomain<K>::_map.size(), Subdomain<K>::_rq + Subdomain<K>::_map.size(), MPI_STATUSES_IGNORE);
        }
#endif
        /* Function: GMV
         *
         *  Computes a global sparse matrix-vector product.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector. */
        inline void GMV(const K* const in, K* const out) const {
#if 0
            typename Wrapper<K>::ul_type* tmp = new typename Wrapper<K>::ul_type[Subdomain<K>::_dof];
            Wrapper<K>::diagv(Subdomain<K>::_dof, _d, in, tmp);
            Wrapper<K>::template csrmv<'C'>(Subdomain<K>::_a->_sym, &(Subdomain<K>::_dof), Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, tmp, out);
            delete [] tmp;
            Subdomain<K>::exchange(out);
#else
            Wrapper<K>::template csrmv<'C'>(Subdomain<K>::_a->_sym, &(Subdomain<K>::_dof), Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, in, out);
#if HPDDM_GMV
            optimized_exchange(out);
#else
            Wrapper<K>::diagv(Subdomain<K>::_dof, _d, out);
            Subdomain<K>::exchange(out);
#endif
#endif
        }
        /* Function: computeError
         *
         *  Computes the Euclidean norm of a right-hand side and of the difference between a solution vector and a right-hand side.
         *
         * Parameters:
         *    x              - Solution vector.
         *    f              - Right-hand side.
         *    storage        - Array to store both values.
         *
         * See also: <Schur::computeError>. */
        inline void computeError(const K* const x, const K* const f, typename Wrapper<K>::ul_type* const storage) const {
            K* tmp = new K[Subdomain<K>::_dof];
            GMV(x, tmp);
            Wrapper<K>::axpy(&(Subdomain<K>::_dof), &(Wrapper<K>::d__2), f, &i__1, tmp, &i__1);
            storage[0] = storage[1] = 0.0;
            for(unsigned int i = 0; i < Subdomain<K>::_dof; ++i) {
                if(std::abs(f[i]) > HPDDM_PEN * HPDDM_EPS) {
                    storage[0] += _d[i] * std::norm(f[i]) / std::norm(HPDDM_PEN);
                    storage[1] += _d[i] * std::norm(tmp[i]) / std::norm(HPDDM_PEN);
                }
                else {
                    storage[0] += _d[i] * std::norm(f[i]);
                    storage[1] += _d[i] * std::norm(tmp[i]);
                }
            }
            delete [] tmp;
            MPI_Allreduce(MPI_IN_PLACE, storage, 2, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, Subdomain<K>::_communicator);
            storage[0] = std::sqrt(storage[0]);
            storage[1] = std::sqrt(storage[1]);
        }
        template<char N = 'C'>
        inline void distributedNumbering(unsigned int* const in, unsigned int& first, unsigned int& last, unsigned int& global) const {
            Subdomain<K>::template globalMapping<N>(in, in + Subdomain<K>::_dof, first, last, global, _d);
        }
        inline bool distributedCSR(unsigned int* const num, unsigned int first, unsigned int last, int*& ia, int*& ja, K*& c) const {
            return Subdomain<K>::distributedCSR(num, first, last, ia, ja, c, Subdomain<K>::_a);
        }
};
} // HPDDM
#endif // _SCHWARZ_
