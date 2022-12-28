/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2015-03-28

   Copyright (C) 2015      Eidgenössische Technische Hochschule Zürich
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

#ifndef HPDDM_HYPRE_HPP_
#define HPDDM_HYPRE_HPP_

#include <HYPRE.h>
#include <_hypre_parcsr_ls.h>
#include <_hypre_IJ_mv.h>

namespace HPDDM {
#ifdef DHYPRE
#undef HPDDM_CHECK_SUBDOMAIN
#define HPDDM_CHECK_COARSEOPERATOR
#include "HPDDM_preprocessor_check.hpp"
#define COARSEOPERATOR HPDDM::Hypre
/* Class: Hypre
 *
 *  A class inheriting from <DMatrix> to use <Hypre>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Hypre : public DMatrix {
    private:
        /* Variable: A
         *  hypre IJ matrix. */
        HYPRE_IJMatrix           _A;
        /* Variable: b
         *  hypre IJ right-hand side. */
        HYPRE_IJVector           _b;
        /* Variable: x
         *  hypre IJ solution vector. */
        HYPRE_IJVector           _x;
        /* Variable: solver
         *  hypre solver. */
        HYPRE_Solver        _solver;
        /* Variable: precond
         *  hypre preconditioner (not used when <Hypre::strategy> is set to one). */
        HYPRE_Solver       _precond;
        int                  _local;
        char                  _type;
    protected:
        /* Variable: numbering
         *  0-based indexing. */
        static constexpr char _numbering = 'C';
    public:
        Hypre() : _A(), _b(), _x(), _solver(), _precond() { }
        ~Hypre() {
            if(DMatrix::_communicator != MPI_COMM_NULL) {
                if(_type == HPDDM_HYPRE_SOLVER_AMG)
                    HYPRE_BoomerAMGDestroy(_solver);
                else {
                    if(_type == HPDDM_HYPRE_SOLVER_PCG)
                        HYPRE_ParCSRPCGDestroy(_solver);
                    else
                        HYPRE_ParCSRFlexGMRESDestroy(_solver);
                    HYPRE_BoomerAMGDestroy(_precond);
                }
                HYPRE_IJVectorDestroy(_x);
                HYPRE_IJVectorDestroy(_b);
                HYPRE_IJMatrixDestroy(_A);
            }
        }
        /* Function: numfact
         *
         *  Initializes <Hypre::solver> and <Hypre::precond> if necessary and factorizes the supplied matrix.
         *
         * Template Parameter:
         *    S              - 'S'ymmetric or 'G'eneral factorization.
         *
         * Parameters:
         *    ncol           - Number of local rows.
         *    I              - Number of nonzero entries per line.
         *    loc2glob       - Local to global numbering.
         *    J              - Array of column indices.
         *    C              - Array of data. */
        template<char S>
        void numfact(unsigned int ncol, int* I, int* loc2glob, int* J, K* C) {
            static_assert(std::is_same<double, K>::value, "hypre only supports double-precision floating-point real numbers");
            static_assert(S == 'G', "hypre only supports nonsymmetric matrices");
            HYPRE_IJMatrixCreate(DMatrix::_communicator, loc2glob[0], loc2glob[1], loc2glob[0], loc2glob[1], &_A);
            HYPRE_IJMatrixSetObjectType(_A, HYPRE_PARCSR);
            HYPRE_IJMatrixSetRowSizes(_A, I + 1);
            _local = ncol;
            int* rows = new int[3 * _local]();
            int* diag_sizes = rows + _local;
            int* offdiag_sizes = diag_sizes + _local;
            rows[0] = I[0];
            for(unsigned int i = 0; i < _local; ++i) {
                std::for_each(J + rows[0], J + rows[0] + I[i + 1], [&](int& j) { (j < loc2glob[0] || loc2glob[1] < j) ? ++offdiag_sizes[i] : ++diag_sizes[i]; });
                rows[0] += I[i + 1];
            }
            HYPRE_IJMatrixSetDiagOffdSizes(_A, diag_sizes, offdiag_sizes);
            HYPRE_IJMatrixSetMaxOffProcElmts(_A, 0);
            HYPRE_IJMatrixInitialize(_A);
            std::iota(rows, rows + _local, loc2glob[0]);
            HYPRE_IJMatrixSetValues(_A, _local, I + 1, rows, J, C);
            HYPRE_IJMatrixAssemble(_A);
            HYPRE_IJVectorCreate(DMatrix::_communicator, loc2glob[0], loc2glob[1], &_b);
            HYPRE_IJVectorSetObjectType(_b, HYPRE_PARCSR);
            HYPRE_IJVectorInitialize(_b);
            HYPRE_IJVectorCreate(DMatrix::_communicator, loc2glob[0], loc2glob[1], &_x);
            HYPRE_IJVectorSetObjectType(_x, HYPRE_PARCSR);
            HYPRE_IJVectorInitialize(_x);
            delete [] rows;
            delete [] I;
            delete [] loc2glob;
            const Option& opt = *Option::get();
            _type = opt.val<char>("hypre_solver", HPDDM_HYPRE_SOLVER_FGMRES);
            HYPRE_Solver& s = (_type == HPDDM_HYPRE_SOLVER_AMG ? _solver : _precond);
            HYPRE_BoomerAMGCreate(&s);
            HYPRE_BoomerAMGSetCoarsenType(s, opt.val<char>("boomeramg_coarsen_type", 6));
            HYPRE_BoomerAMGSetRelaxType(s, opt.val<char>("boomeramg_relax_type", 3));
            HYPRE_BoomerAMGSetNumSweeps(s, opt.val<char>("boomeramg_num_sweeps", 1));
            HYPRE_BoomerAMGSetMaxLevels(s, opt.val<char>("boomeramg_max_levels", 10));
            HYPRE_BoomerAMGSetInterpType(s, opt.val<char>("boomeramg_interp_type", 0));
            HYPRE_ParCSRMatrix parcsr_A;
            HYPRE_IJMatrixGetObject(_A, reinterpret_cast<void**>(&parcsr_A));
            HYPRE_ParVector par_b;
            HYPRE_IJVectorGetObject(_b, reinterpret_cast<void**>(&par_b));
            HYPRE_ParVector par_x;
            HYPRE_IJVectorGetObject(_x, reinterpret_cast<void**>(&par_x));
            HYPRE_BoomerAMGSetPrintLevel(s, opt.val<char>("verbosity", 0) < 3 ? 0 : 1);
            if(_type == HPDDM_HYPRE_SOLVER_AMG) {
                HYPRE_BoomerAMGSetMaxIter(_solver, opt.val<unsigned int>("hypre_max_it", 500));
                HYPRE_BoomerAMGSetTol(_solver, opt.val("hypre_tol", 1.0e-12));
                HYPRE_BoomerAMGSetup(_solver, parcsr_A, nullptr, nullptr);
            }
            else {
                HYPRE_BoomerAMGSetTol(_precond, 0.0);
                HYPRE_BoomerAMGSetMaxIter(_precond, 1);
                if(_type == HPDDM_HYPRE_SOLVER_PCG) {
                    HYPRE_ParCSRPCGCreate(DMatrix::_communicator, &_solver);
                    HYPRE_PCGSetMaxIter(_solver, opt.val<unsigned int>("hypre_max_it", 500));
                    HYPRE_PCGSetTol(_solver, opt.val("hypre_tol", 1.0e-12));
                    HYPRE_PCGSetPrintLevel(_solver, 0);
                    HYPRE_PCGSetLogging(_solver, 0);
                    HYPRE_PCGSetPrecond(_solver, reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSolve), reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSetup), _precond);
                    HYPRE_ParCSRPCGSetup(_solver, parcsr_A, par_b, par_x);
                }
                else {
                    HYPRE_ParCSRFlexGMRESCreate(DMatrix::_communicator, &_solver);
                    HYPRE_FlexGMRESSetKDim(_solver, opt.val<unsigned short>("hypre_gmres_restart", 100));
                    HYPRE_FlexGMRESSetMaxIter(_solver, opt.val<unsigned int>("hypre_max_it", 500));
                    HYPRE_FlexGMRESSetTol(_solver, opt.val("hypre_tol", 1.0e-12));
                    HYPRE_FlexGMRESSetPrintLevel(_solver, 0);
                    HYPRE_FlexGMRESSetLogging(_solver, 0);
                    HYPRE_FlexGMRESSetPrecond(_solver, reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSolve), reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSetup), _precond);
                    HYPRE_ParCSRFlexGMRESSetup(_solver, parcsr_A, par_b, par_x);
                }
            }
            HYPRE_BoomerAMGSetPrintLevel(s, 0);
        }
        /* Function: solve
         *
         *  Solves the system in-place.
         *
         * Parameters:
         *    rhs            - Input right-hand sides, solution vectors are stored in-place.
         *    n              - Number of right-hand sides. */
        void solve(K* rhs, const unsigned short& n) {
            HYPRE_ParVector par_b;
            HYPRE_IJVectorGetObject(_b, reinterpret_cast<void**>(&par_b));
            HYPRE_ParVector par_x;
            HYPRE_IJVectorGetObject(_x, reinterpret_cast<void**>(&par_x));
            HYPRE_ParCSRMatrix parcsr_A;
            HYPRE_IJMatrixGetObject(_A, reinterpret_cast<void**>(&parcsr_A));
            int num_iterations;
            const Option& opt = *Option::get();
            hypre_Vector* loc = hypre_ParVectorLocalVector(reinterpret_cast<hypre_ParVector*>(hypre_IJVectorObject(reinterpret_cast<hypre_IJVector*>(_b))));
            const K* b = loc->data;
            for(unsigned short nu = 0; nu < n; ++nu) {
                loc->data = rhs + nu * _local;
                if(_type == HPDDM_HYPRE_SOLVER_AMG) {
                    HYPRE_BoomerAMGSolve(_solver, parcsr_A, par_b, par_x);
                    HYPRE_BoomerAMGGetNumIterations(_solver, &num_iterations);
                }
                else if(_type == HPDDM_HYPRE_SOLVER_PCG) {
                    HYPRE_ParCSRPCGSolve(_solver, parcsr_A, par_b, par_x);
                    HYPRE_PCGGetNumIterations(_solver, &num_iterations);
                }
                else {
                    HYPRE_ParCSRFlexGMRESSolve(_solver, parcsr_A, par_b, par_x);
                    HYPRE_GMRESGetNumIterations(_solver, &num_iterations);
                }
                std::copy_n(hypre_ParVectorLocalVector(reinterpret_cast<hypre_ParVector*>(hypre_IJVectorObject(reinterpret_cast<hypre_IJVector*>(_x))))->data, _local, rhs + nu * _local);
                if(DMatrix::_rank == 0 && opt.val<char>("verbosity", 0) > 3)
                    std::cout << " --- BoomerAMG performed " << num_iterations << " iteration" << (num_iterations > 1 ? "s" : "") << std::endl;
            }
            loc->data = const_cast<K*>(b);
        }
};
#endif // DHYPRE
} // HPDDM
#endif // HPDDM_HYPRE_HPP_
