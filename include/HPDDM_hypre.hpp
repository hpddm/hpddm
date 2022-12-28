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
        HYPRE_IJMatrix           A_;
        /* Variable: b
         *  hypre IJ right-hand side. */
        HYPRE_IJVector           b_;
        /* Variable: x
         *  hypre IJ solution vector. */
        HYPRE_IJVector           x_;
        /* Variable: solver
         *  hypre solver. */
        HYPRE_Solver        solver_;
        /* Variable: precond
         *  hypre preconditioner (not used when <Hypre::strategy> is set to one). */
        HYPRE_Solver       precond_;
        int                  local_;
        char                  type_;
    protected:
        /* Variable: numbering
         *  0-based indexing. */
        static constexpr char numbering_ = 'C';
    public:
        Hypre() : A_(), b_(), x_(), solver_(), precond_() { }
        ~Hypre() {
            if(DMatrix::communicator_ != MPI_COMM_NULL) {
                if(type_ == HPDDM_HYPRE_SOLVER_AMG)
                    HYPRE_BoomerAMGDestroy(solver_);
                else {
                    if(type_ == HPDDM_HYPRE_SOLVER_PCG)
                        HYPRE_ParCSRPCGDestroy(solver_);
                    else
                        HYPRE_ParCSRFlexGMRESDestroy(solver_);
                    HYPRE_BoomerAMGDestroy(precond_);
                }
                HYPRE_IJVectorDestroy(x_);
                HYPRE_IJVectorDestroy(b_);
                HYPRE_IJMatrixDestroy(A_);
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
            HYPRE_IJMatrixCreate(DMatrix::communicator_, loc2glob[0], loc2glob[1], loc2glob[0], loc2glob[1], &A_);
            HYPRE_IJMatrixSetObjectType(A_, HYPRE_PARCSR);
            HYPRE_IJMatrixSetRowSizes(A_, I + 1);
            local_ = ncol;
            int* rows = new int[3 * local_]();
            int* diag_sizes = rows + local_;
            int* offdiag_sizes = diag_sizes + local_;
            rows[0] = I[0];
            for(unsigned int i = 0; i < local_; ++i) {
                std::for_each(J + rows[0], J + rows[0] + I[i + 1], [&](int& j) { (j < loc2glob[0] || loc2glob[1] < j) ? ++offdiag_sizes[i] : ++diag_sizes[i]; });
                rows[0] += I[i + 1];
            }
            HYPRE_IJMatrixSetDiagOffdSizes(A_, diag_sizes, offdiag_sizes);
            HYPRE_IJMatrixSetMaxOffProcElmts(A_, 0);
            HYPRE_IJMatrixInitialize(A_);
            std::iota(rows, rows + local_, loc2glob[0]);
            HYPRE_IJMatrixSetValues(A_, local_, I + 1, rows, J, C);
            HYPRE_IJMatrixAssemble(A_);
            HYPRE_IJVectorCreate(DMatrix::communicator_, loc2glob[0], loc2glob[1], &b_);
            HYPRE_IJVectorSetObjectType(b_, HYPRE_PARCSR);
            HYPRE_IJVectorInitialize(b_);
            HYPRE_IJVectorCreate(DMatrix::communicator_, loc2glob[0], loc2glob[1], &x_);
            HYPRE_IJVectorSetObjectType(x_, HYPRE_PARCSR);
            HYPRE_IJVectorInitialize(x_);
            delete [] rows;
            delete [] I;
            delete [] loc2glob;
            const Option& opt = *Option::get();
            type_ = opt.val<char>("hypre_solver", HPDDM_HYPRE_SOLVER_FGMRES);
            HYPRE_Solver& s = (type_ == HPDDM_HYPRE_SOLVER_AMG ? solver_ : precond_);
            HYPRE_BoomerAMGCreate(&s);
            HYPRE_BoomerAMGSetCoarsenType(s, opt.val<char>("boomeramg_coarsen_type", 6));
            HYPRE_BoomerAMGSetRelaxType(s, opt.val<char>("boomeramg_relax_type", 3));
            HYPRE_BoomerAMGSetNumSweeps(s, opt.val<char>("boomeramg_num_sweeps", 1));
            HYPRE_BoomerAMGSetMaxLevels(s, opt.val<char>("boomeramg_max_levels", 10));
            HYPRE_BoomerAMGSetInterpType(s, opt.val<char>("boomeramg_interp_type", 0));
            HYPRE_ParCSRMatrix parcsr_A;
            HYPRE_IJMatrixGetObject(A_, reinterpret_cast<void**>(&parcsr_A));
            HYPRE_ParVector par_b;
            HYPRE_IJVectorGetObject(b_, reinterpret_cast<void**>(&par_b));
            HYPRE_ParVector par_x;
            HYPRE_IJVectorGetObject(x_, reinterpret_cast<void**>(&par_x));
            HYPRE_BoomerAMGSetPrintLevel(s, opt.val<char>("verbosity", 0) < 3 ? 0 : 1);
            if(type_ == HPDDM_HYPRE_SOLVER_AMG) {
                HYPRE_BoomerAMGSetMaxIter(solver_, opt.val<unsigned int>("hypre_max_it", 500));
                HYPRE_BoomerAMGSetTol(solver_, opt.val("hypre_tol", 1.0e-12));
                HYPRE_BoomerAMGSetup(solver_, parcsr_A, nullptr, nullptr);
            }
            else {
                HYPRE_BoomerAMGSetTol(precond_, 0.0);
                HYPRE_BoomerAMGSetMaxIter(precond_, 1);
                if(type_ == HPDDM_HYPRE_SOLVER_PCG) {
                    HYPRE_ParCSRPCGCreate(DMatrix::communicator_, &solver_);
                    HYPRE_PCGSetMaxIter(solver_, opt.val<unsigned int>("hypre_max_it", 500));
                    HYPRE_PCGSetTol(solver_, opt.val("hypre_tol", 1.0e-12));
                    HYPRE_PCGSetPrintLevel(solver_, 0);
                    HYPRE_PCGSetLogging(solver_, 0);
                    HYPRE_PCGSetPrecond(solver_, reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSolve), reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSetup), precond_);
                    HYPRE_ParCSRPCGSetup(solver_, parcsr_A, par_b, par_x);
                }
                else {
                    HYPRE_ParCSRFlexGMRESCreate(DMatrix::communicator_, &solver_);
                    HYPRE_FlexGMRESSetKDim(solver_, opt.val<unsigned short>("hypre_gmres_restart", 100));
                    HYPRE_FlexGMRESSetMaxIter(solver_, opt.val<unsigned int>("hypre_max_it", 500));
                    HYPRE_FlexGMRESSetTol(solver_, opt.val("hypre_tol", 1.0e-12));
                    HYPRE_FlexGMRESSetPrintLevel(solver_, 0);
                    HYPRE_FlexGMRESSetLogging(solver_, 0);
                    HYPRE_FlexGMRESSetPrecond(solver_, reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSolve), reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSetup), precond_);
                    HYPRE_ParCSRFlexGMRESSetup(solver_, parcsr_A, par_b, par_x);
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
            HYPRE_IJVectorGetObject(b_, reinterpret_cast<void**>(&par_b));
            HYPRE_ParVector par_x;
            HYPRE_IJVectorGetObject(x_, reinterpret_cast<void**>(&par_x));
            HYPRE_ParCSRMatrix parcsr_A;
            HYPRE_IJMatrixGetObject(A_, reinterpret_cast<void**>(&parcsr_A));
            int num_iterations;
            const Option& opt = *Option::get();
            hypre_Vector* loc = hypre_ParVectorLocalVector(reinterpret_cast<hypre_ParVector*>(hypre_IJVectorObject(reinterpret_cast<hypre_IJVector*>(b_))));
            const K* b = loc->data;
            for(unsigned short nu = 0; nu < n; ++nu) {
                loc->data = rhs + nu * local_;
                if(type_ == HPDDM_HYPRE_SOLVER_AMG) {
                    HYPRE_BoomerAMGSolve(solver_, parcsr_A, par_b, par_x);
                    HYPRE_BoomerAMGGetNumIterations(solver_, &num_iterations);
                }
                else if(type_ == HPDDM_HYPRE_SOLVER_PCG) {
                    HYPRE_ParCSRPCGSolve(solver_, parcsr_A, par_b, par_x);
                    HYPRE_PCGGetNumIterations(solver_, &num_iterations);
                }
                else {
                    HYPRE_ParCSRFlexGMRESSolve(solver_, parcsr_A, par_b, par_x);
                    HYPRE_GMRESGetNumIterations(solver_, &num_iterations);
                }
                std::copy_n(hypre_ParVectorLocalVector(reinterpret_cast<hypre_ParVector*>(hypre_IJVectorObject(reinterpret_cast<hypre_IJVector*>(x_))))->data, local_, rhs + nu * local_);
                if(DMatrix::rank_ == 0 && opt.val<char>("verbosity", 0) > 3)
                    std::cout << " --- BoomerAMG performed " << num_iterations << " iteration" << (num_iterations > 1 ? "s" : "") << std::endl;
            }
            loc->data = const_cast<K*>(b);
        }
};
#endif // DHYPRE
} // HPDDM
#endif // HPDDM_HYPRE_HPP_
