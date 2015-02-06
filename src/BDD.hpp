/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <jolivet@ann.jussieu.fr>
        Date: 2013-06-03

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

#ifndef _BDD_
#define _BDD_

#include "schur.hpp"

namespace HPDDM {
/* Class: Bdd
 *
 *  A class for solving problems using the BDD method.
 *
 * Template Parameters:
 *    Solver         - Solver used for the factorization of local matrices.
 *    CoarseOperator - Class of the coarse operator.
 *    S              - 'S'ymmetric or 'G'eneral coarse operator.
 *    K              - Scalar type. */
template<template<class> class Solver, template<class> class CoarseSolver, char S, class K>
class Bdd : public Schur<Solver, CoarseOperator<CoarseSolver, S, K>, K> {
    private:
        /* Variable: m
         *  Local partition of unity. */
        typename Wrapper<K>::ul_type* _m;
    public:
        Bdd() : _m() { }
        ~Bdd() {
            delete []  _m;
        }
        /* Typedef: super
         *  Type of the immediate parent class <Schur>. */
        typedef Schur<Solver, CoarseOperator<CoarseSolver, S, K>, K> super;
        /* Function: initialize
         *  Allocates <Bdd::m> and calls <Schur::initialize>. */
        inline void initialize() {
            super::template initialize<false>();
            _m = new typename Wrapper<K>::ul_type[Subdomain<K>::_dof];
        }
        inline void allocateSingle(K*& primal) const {
            primal = new K[Subdomain<K>::_dof];
        }
        template<unsigned short N>
        inline void allocateArray(K* (&array)[N]) const {
            *array = new K[N * Subdomain<K>::_dof];
            for(unsigned short i = 1; i < N; ++i)
                array[i] = *array + i * Subdomain<K>::_dof;
        }
        /* Function: buildScaling
         *
         *  Builds the local partition of unity <Bdd::m>.
         *
         * Parameters:
         *    scaling        - Type of scaling (multiplicity, stiffness or coefficient scaling).
         *    rho            - Physical local coefficients (optional). */
        inline void buildScaling(const char& scaling, const K* const& rho = nullptr) {
            initialize();
            std::fill(_m, _m + Subdomain<K>::_dof, 1.0);
            if((scaling == 'r' && rho) || scaling == 'k') {
                if(scaling == 'k')
                    super::stiffnessScaling(super::_work);
                else
                    std::copy_n(rho + super::_bi->_m, Subdomain<K>::_dof, super::_work);
                Subdomain<K>::recvBuffer(super::_work);
                for(unsigned short i = 0; i < Subdomain<K>::_map.size(); ++i)
                    for(unsigned int j = 0; j < Subdomain<K>::_map[i].second.size(); ++j)
                        _m[Subdomain<K>::_map[i].second[j]] *= std::real(Subdomain<K>::_sbuff[i][j]) / std::real(Subdomain<K>::_sbuff[i][j] + _m[Subdomain<K>::_map[i].second[j]] * Subdomain<K>::_rbuff[i][j]);
            }
            else
                for(const pairNeighbor& neighbor : Subdomain<K>::_map)
                    for(pairNeighbor::second_type::const_reference p : neighbor.second)
                        _m[p] /= (1.0 + _m[p]);
        }
        /* Function: start
         *
         *  Projected Conjugate Gradient initialization.
         *
         * Template Parameter:
         *    excluded       - True if the master processes are excluded from the domain decomposition, false otherwise.
         *
         * Parameters:
         *    x              - Solution vector.
         *    f              - Right-hand side.
         *    b              - Condensed right-hand side.
         *    r              - First residual. */
        template<bool excluded>
        inline void start(K* const x, const K* const f, K* const b, K* r) const {
            if(super::_co) {
                if(!excluded) {
                    super::condensateEffort(f, b);
                    Subdomain<K>::exchange(b ? b : super::_structure + super::_bi->_m);
                    if(super::_ev) {
                        std::copy_n(b ? b : super::_structure + super::_bi->_m, Subdomain<K>::_dof, x);
                        Wrapper<K>::diagv(Subdomain<K>::_dof, _m, x);
                        if(super::_schur) {
                            Wrapper<K>::gemv(&transb, &(Subdomain<K>::_dof), super::_co->getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), x, &i__1, &(Wrapper<K>::d__0), super::_uc, &i__1);
                            super::_co->template callSolver<excluded>(super::_uc);
                            Wrapper<K>::gemv(&transa, &(Subdomain<K>::_dof), super::_co->getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), super::_uc, &i__1, &(Wrapper<K>::d__0), x, &i__1);
                        }
                        else {
                            Wrapper<K>::gemv(&transb, &(Subdomain<K>::_dof), super::_co->getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev + super::_bi->_m, &(Subdomain<K>::_a->_n), x, &i__1, &(Wrapper<K>::d__0), super::_uc, &i__1);
                            super::_co->template callSolver<excluded>(super::_uc);
                            Wrapper<K>::gemv(&transa, &(Subdomain<K>::_dof), super::_co->getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev + super::_bi->_m, &(Subdomain<K>::_a->_n), super::_uc, &i__1, &(Wrapper<K>::d__0), x, &i__1);
                        }
                        Wrapper<K>::diagv(Subdomain<K>::_dof, _m, x);
                    }
                    else {
                        std::fill(x, x + Subdomain<K>::_dof, 0.0);
                        super::_co->template callSolver<excluded>(super::_uc);
                    }
                    Subdomain<K>::exchange(x);
                    super::applyLocalSchurComplement(x, r);
                    Subdomain<K>::exchange(r);
                    Wrapper<K>::axpby(Subdomain<K>::_dof, 1.0, b ? b : super::_structure + super::_bi->_m, 1, -1.0, r, 1);
                }
                else
                    super::_co->template callSolver<excluded>(super::_uc);
            }
            else if(!excluded) {
                super::condensateEffort(f, r);
                Subdomain<K>::exchange(r);
                std::fill(x, x + Subdomain<K>::_dof, 0.0);
            }
        }
        /* Function: apply
         *
         *  Applies the global Schur complement to a single right-hand side.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector (optional). */
        inline void apply(K* const in, K* const out = nullptr) const {
            if(out) {
                super::applyLocalSchurComplement(in, out);
                Subdomain<K>::exchange(out);
            }
            else {
                super::applyLocalSchurComplement(in);
                Subdomain<K>::exchange(in);
            }
        }
        /* Function: precond
         *
         *  Applies the global preconditioner to a single right-hand side.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector (optional). */
        inline void precond(K* const in, K* const out = nullptr) const {
            Wrapper<K>::diagv(Subdomain<K>::_dof, _m, in, super::_work + super::_bi->_m);
            std::fill_n(super::_work, super::_bi->_m, 0.0);
            super::_p.solve(super::_work);
            if(out) {
                Wrapper<K>::diagv(Subdomain<K>::_dof, _m, super::_work + super::_bi->_m, out);
                Subdomain<K>::exchange(out);
            }
            else {
                Wrapper<K>::diagv(Subdomain<K>::_dof, _m, super::_work + super::_bi->_m, in);
                Subdomain<K>::exchange(in);
            }
        }
        /* Function: project
         *
         *  Projects into the coarse space.
         *
         * Template Parameters:
         *    excluded       - True if the master processes are excluded from the domain decomposition, false otherwise.
         *    trans          - 'T' if the transposed projection should be applied, 'N' otherwise.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector (optional). */
        template<bool excluded, char trans>
        inline void project(K* const in, K* const out = nullptr) const {
            static_assert(trans == 'T' || trans == 'N', "Unsupported value for argument 'trans'");
            if(super::_co) {
                if(!excluded) {
                    if(trans == 'N')
                        apply(in, super::_structure + super::_bi->_m);
                    if(super::_ev) {
                        if(trans == 'N')
                            Wrapper<K>::diagv(Subdomain<K>::_dof, _m, super::_structure + super::_bi->_m);
                        else
                            Wrapper<K>::diagv(Subdomain<K>::_dof, _m, in, super::_structure + super::_bi->_m);
                        if(super::_schur) {
                            Wrapper<K>::gemv(&transb, &(Subdomain<K>::_dof), super::_co->getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), super::_structure + super::_bi->_m, &i__1, &(Wrapper<K>::d__0), super::_uc, &i__1);
                            super::_co->template callSolver<excluded>(super::_uc);
                            Wrapper<K>::gemv(&transa, &(Subdomain<K>::_dof), super::_co->getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev, &(Subdomain<K>::_dof), super::_uc, &i__1, &(Wrapper<K>::d__0), super::_structure + super::_bi->_m, &i__1);
                        }
                        else {
                            Wrapper<K>::gemv(&transb, &(Subdomain<K>::_dof), super::_co->getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev + super::_bi->_m, &(Subdomain<K>::_a->_n), super::_structure + super::_bi->_m, &i__1, &(Wrapper<K>::d__0), super::_uc, &i__1);
                            super::_co->callSolver(super::_uc);
                            Wrapper<K>::gemv(&transa, &(Subdomain<K>::_dof), super::_co->getAddrLocal(), &(Wrapper<K>::d__1), *super::_ev + super::_bi->_m, &(Subdomain<K>::_a->_n), super::_uc, &i__1, &(Wrapper<K>::d__0), super::_structure + super::_bi->_m, &i__1);
                        }
                    }
                    else {
                        super::_co->callSolver(super::_uc);
                        std::fill_n(super::_structure + super::_bi->_m, Subdomain<K>::_dof, 0.0);
                    }
                    Wrapper<K>::diagv(Subdomain<K>::_dof, _m, super::_structure + super::_bi->_m);
                    Subdomain<K>::exchange(super::_structure + super::_bi->_m);
                    if(trans == 'T')
                        apply(super::_structure + super::_bi->_m);
                    if(out)
                        for(unsigned int i = 0; i < Subdomain<K>::_dof; ++i)
                            out[i] = in[i] - *(super::_structure + super::_bi->_m + i);
                    else
                        Wrapper<K>::axpy(&(Subdomain<K>::_dof), &(Wrapper<K>::d__2), super::_structure + super::_bi->_m, &i__1, in, &i__1);
                }
                else
                    super::_co->template callSolver<excluded>(super::_uc);
            }
            else if(!excluded && out)
                std::copy(in, in + Subdomain<K>::_dof, out);
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
         * See also: <Feti::buildTwo>, <Schwarz::buildTwo>.*/
        template<unsigned short excluded = 0, class Container>
        inline std::pair<MPI_Request, const K*>* buildTwo(const MPI_Comm& comm, Container& parm) {
            if(!super::_schur && parm[NU])
                super::_deficiency = parm[NU];
            return super::template buildTwo<excluded, 3>(std::move(BddProjection<Bdd<Solver, CoarseSolver, S, K>, K>(*this, parm[NU])), comm, parm);
        }
        /* Function: computeSolution
         *
         *  Computes the solution after convergence of the Projected Conjugate Gradient.
         *
         * Template Parameter:
         *    excluded       - True if the master processes are excluded from the domain decomposition, false otherwise.
         *
         * Parameters:
         *    x              - Solution vector.
         *    f              - Right-hand side. */
        template<bool excluded>
        inline void computeSolution(K* const x, const K* const f) const {
            if(!excluded) {
                std::copy(f, f + super::_bi->_m, x);
                Wrapper<K>::template csrmv<Wrapper<K>::I>(&transb, &(Subdomain<K>::_dof), &(super::_bi->_m), &(Wrapper<K>::d__2), false, super::_bi->_a, super::_bi->_ia, super::_bi->_ja, x + super::_bi->_m, &(Wrapper<K>::d__1), x);
                if(!super::_schur)
                    super::_s.solve(x);
                else {
                    std::copy(x, x + super::_bi->_m, super::_structure);
                    super::_s.solve(super::_structure);
                    std::copy(super::_structure, super::_structure + super::_bi->_m, x);
                }
            }
        }
        template<bool>
        inline void computeSolution(K* const, K* const* const) const { }
        /* Function: computeDot
         *
         *  Computes the dot product of two vectors.
         *
         * Template Parameter:
         *    excluded       - True if the master processes are excluded from the domain decomposition, false otherwise.
         *
         * Parameters:
         *    a              - Left-hand side.
         *    b              - Right-hand side. */
        template<bool excluded>
        inline void computeDot(typename Wrapper<K>::ul_type* const val, const K* const a, const K* const b, const MPI_Comm& comm) const {
            if(!excluded) {
                Wrapper<K>::diagv(Subdomain<K>::_dof, _m, a, super::_work);
                *val = Wrapper<K>::dot(&(Subdomain<K>::_dof), super::_work, &i__1, b, &i__1);
            }
            else
                *val = 0.0;
            MPI_Allreduce(MPI_IN_PLACE, val, 1, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
        }
        /* Function: getScaling
         *  Returns a constant pointer to <Bdd::m>. */
        inline const typename Wrapper<K>::ul_type* getScaling() const { return _m; }
        /* Function: solveGEVP
         *
         *  Solves the GenEO problem.
         *
         * Template Parameter:
         *    L              - 'S'ymmetric or 'G'eneral transfer of the local Schur complements.
         *
         * Parameters:
         *    nu             - Number of eigenvectors requested.
         *    threshold      - Criterion for selecting the eigenpairs (optional). */
        template<char L = 'S'>
        inline void solveGEVP(unsigned short& nu, const typename Wrapper<K>::ul_type& threshold = 0.0) {
            super::template solveGEVP<L>(_m, nu, threshold);
        }
};
} // HPDDM
#endif // _BDD_
