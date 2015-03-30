/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@inf.ethz.ch>
        Date: 2015-03-28

   Copyright (C) 2015      Eidgenössische Technische Hochschule Zürich

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

#ifndef _HYPRE_
#define _HYPRE_

#include <HYPRE.h>
#include <_hypre_parcsr_ls.h>
#include <_hypre_IJ_mv.h>

namespace HPDDM {
#ifdef DHYPRE
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
        HYPRE_IJMatrix           _A;
        HYPRE_IJVector           _b;
        HYPRE_IJVector           _x;
        HYPRE_Solver        _solver;
        HYPRE_Solver       _precond;
        int                  _local;
        int               _strategy;
    protected:
        /* Variable: numbering
         *  0-based indexing. */
        static constexpr char _numbering = 'C';
    public:
        Hypre() : _A(), _b(), _x(), _solver(), _precond() { }
        ~Hypre() {
            if(DMatrix::_communicator != MPI_COMM_NULL) {
                if(_strategy == 1)
                    HYPRE_BoomerAMGDestroy(_solver);
                else {
                    if(_strategy == 2)
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
        inline void numfact(unsigned int ncol, int* I, int* loc2glob, int* J, K* C) {
            static_assert(std::is_same<double, K>::value, "Hypre only supports double-precision floating-point real numbers");
            static_assert(S == 'G', "Hypre only supports nonsymmetric matrices");
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
            HYPRE_BoomerAMGCreate(_strategy == 1 ? &_solver : &_precond);
            HYPRE_BoomerAMGSetCoarsenType(_strategy == 1 ? _solver : _precond, 6); /* Falgout coarsening */
            HYPRE_BoomerAMGSetRelaxType(_strategy == 1 ? _solver : _precond, 6);   /* G-S/Jacobi hybrid relaxation */
            HYPRE_BoomerAMGSetNumSweeps(_strategy == 1 ? _solver : _precond, 1);   /* sweeps on each level */
            HYPRE_BoomerAMGSetMaxLevels(_strategy == 1 ? _solver : _precond, 10);  /* maximum number of levels */
            HYPRE_ParCSRMatrix parcsr_A;
            HYPRE_IJMatrixGetObject(_A, reinterpret_cast<void**>(&parcsr_A));
            HYPRE_ParVector par_b;
            HYPRE_IJVectorGetObject(_b, reinterpret_cast<void**>(&par_b));
            HYPRE_ParVector par_x;
            HYPRE_IJVectorGetObject(_x, reinterpret_cast<void**>(&par_x));
            if(_strategy == 1) {
                HYPRE_BoomerAMGSetTol(_solver, 1.0e-8);
                HYPRE_BoomerAMGSetMaxIter(_solver, 1000);
                HYPRE_BoomerAMGSetPrintLevel(_solver, 1);
                HYPRE_BoomerAMGSetup(_solver, parcsr_A, nullptr, nullptr);
            }
            else {
                HYPRE_BoomerAMGSetTol(_precond, 0.0);
                HYPRE_BoomerAMGSetMaxIter(_precond, 1);
                HYPRE_BoomerAMGSetPrintLevel(_precond, 1);
                if(_strategy == 2) {
                    HYPRE_ParCSRPCGCreate(DMatrix::_communicator, &_solver);
                    HYPRE_PCGSetMaxIter(_solver, 500);
                    HYPRE_PCGSetTol(_solver, 1.0e-8);
                    HYPRE_PCGSetTwoNorm(_solver, 1);
                    HYPRE_PCGSetPrintLevel(_solver, 1);
                    HYPRE_PCGSetLogging(_solver, 1);
                    HYPRE_PCGSetPrecond(_solver, reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSolve), reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSetup), _precond);
                    HYPRE_ParCSRPCGSetup(_solver, parcsr_A, par_b, par_x);
                }
                else {
                    HYPRE_ParCSRFlexGMRESCreate(DMatrix::_communicator, &_solver);
                    HYPRE_FlexGMRESSetKDim(_solver, 50);
                    HYPRE_FlexGMRESSetMaxIter(_solver, 500);
                    HYPRE_FlexGMRESSetTol(_solver, 1.0e-8);
                    HYPRE_FlexGMRESSetPrintLevel(_solver, 1);
                    HYPRE_FlexGMRESSetLogging(_solver, 1);
                    HYPRE_FlexGMRESSetPrecond(_solver, reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSolve), reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSetup), _precond);
                    HYPRE_ParCSRFlexGMRESSetup(_solver, parcsr_A, par_b, par_x);
                }
            }
        }
        /* Function: solve
         *
         *  Solves the system in-place.
         *
         * Template Parameter:
         *    D              - Distribution of right-hand sides and solution vectors.
         *
         * Parameter:
         *    rhs            - Input right-hand side, solution vector is stored in-place. */
        template<DMatrix::Distribution D>
        inline void solve(K* rhs) {
            hypre_Vector* loc = hypre_ParVectorLocalVector(reinterpret_cast<hypre_ParVector*>(hypre_IJVectorObject(reinterpret_cast<hypre_IJVector*>(_b))));
            K* b = loc->data;
            loc->data = rhs;
            HYPRE_ParVector par_b;
            HYPRE_IJVectorGetObject(_b, reinterpret_cast<void**>(&par_b));
            HYPRE_ParVector par_x;
            HYPRE_IJVectorGetObject(_x, reinterpret_cast<void**>(&par_x));
            HYPRE_ParCSRMatrix parcsr_A;
            HYPRE_IJMatrixGetObject(_A, reinterpret_cast<void**>(&parcsr_A));
            if(_strategy == 1)      /* AMG */
                HYPRE_BoomerAMGSolve(_solver, parcsr_A, par_b, par_x);
            else if(_strategy == 2) /* PCG with AMG */
                HYPRE_ParCSRPCGSolve(_solver, parcsr_A, par_b, par_x);
            else                    /* GMRES with AMG */
                HYPRE_ParCSRFlexGMRESSolve(_solver, parcsr_A, par_b, par_x);
            loc->data = b;
            loc = hypre_ParVectorLocalVector(reinterpret_cast<hypre_ParVector*>(hypre_IJVectorObject(reinterpret_cast<hypre_IJVector*>(_x))));
            std::copy(loc->data, loc->data + _local, rhs);
        }
        /* Function: initialize
         *
         *  Initializes <Hypre::strategy>, <DMatrix::rank>, and <DMatrix::distribution>.
         *
         * Parameter:
         *    parm           - Vector of parameters. */
        template<class Container>
        inline void initialize(Container& parm) {
            if(DMatrix::_communicator != MPI_COMM_NULL)
                MPI_Comm_rank(DMatrix::_communicator, &(DMatrix::_rank));
            if(parm[DISTRIBUTION] != DMatrix::DISTRIBUTED_SOL_AND_RHS) {
                if(DMatrix::_communicator != MPI_COMM_NULL && DMatrix::_rank == 0)
                    std::cout << "WARNING -- only distributed solution and RHS supported by the Hypre interface, forcing the distribution to DISTRIBUTED_SOL_AND_RHS" << std::endl;
                parm[DISTRIBUTION] = DMatrix::DISTRIBUTED_SOL_AND_RHS;
            }
            DMatrix::_distribution = DMatrix::DISTRIBUTED_SOL_AND_RHS;
            _strategy = parm[STRATEGY];
        }
};
#endif // DHYPRE
} // HPDDM
#endif // _HYPRE_
